# Dashboard Conditioning Completion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the paused dashboard conditioning work so the Simulator and Patient Atlas generate samples conditioned on covariate values + gating group, using a faithful logistic-normal sampler over the exported block-wise Σ.

**Architecture:** Two small Python export additions (correlation `reference_topic`; gating `group_var_label`/`group_labels`/`group_proportions`), a hand-rolled front-end numerical kernel (Cholesky + multivariate-normal draw, no npm deps), a shared `sampleConditionedTheta` forward sampler that draws θ = softmax(η), η ~ Normal(Γᵀx, Σ_allowed), a per-patient `marginalSampler`, per-panel conditioning state in the store, and wiring into the Simulator + Patient Atlas. The four-quadrant display reader (`prevalenceReader`) and correlation heatmap are already shipped and unchanged.

**Tech Stack:** Python 3.12 (charmpheno export, pytest), Svelte 5 + TypeScript + vitest (dashboard), Spark (cloud builder — `py_compile` parity only).

## Global Constraints

- spark-vi stays domain-agnostic: **no spark-vi change** in this plan (charmpheno + dashboard only).
- No LaTeX in any prose/docstring/comment: use Unicode Greek (Σ η θ μ Γ) + plain text; write `E(β)` not `E[β]`.
- Cite any literature-derived method/default in docstrings (logistic-normal: Blei & Lafferty 2007; Cholesky/MVN draw is textbook — no citation needed).
- Markdown-linkable code refs in any prose (`[name](path#Lstart-Lend)`).
- TDD throughout: write the failing test, watch it fail, minimal implementation, watch it pass, commit.
- Hand-rolled numerics only — **no new npm dependency** (matches existing `sampling.ts` samplers).
- Export changes touch BOTH builders: `analysis/local/build_dashboard.py` (real test) and `analysis/cloud/build_dashboard_cloud.py` (mirrored edit + `py_compile`), per the cloud-parity convention.
- STM detection for the sampler branch: a bundle uses the logistic-normal path iff it carries BOTH `covariateEffects` and `correlation`; otherwise the existing Dirichlet path.
- The STM reference topic is pinned η=0, excluded from Γ's effect rows (zero row) and from `correlation.R`/`topic_order`. Free topics are the K-1 in `topic_order`.

---

## File Structure

**Python export (charmpheno):**
- `charmpheno/charmpheno/export/correlation.py` — MODIFY `build_correlation_json` to emit `reference_topic`.
- `charmpheno/charmpheno/export/gating.py` — MODIFY `build_gating_json` to emit `group_var_label`, `group_labels`, `group_proportions`; ADD a `_humanize` helper.
- `charmpheno/tests/test_correlation_export.py`, `charmpheno/tests/test_gating_export.py` — extend.
- `analysis/local/build_dashboard.py`, `analysis/cloud/build_dashboard_cloud.py` — pass the new `build_gating_json` args (group_counts already in scope as `gc`/`stm_gc`).

**Front-end (dashboard/src/lib):**
- `sampling.ts` — ADD `sampleStandardNormal(rng)`.
- `conditioning/logisticNormal.ts` — CREATE `cholesky`, `mvnDraw`, `sampleConditionedTheta`.
- `conditioning/marginalSampler.ts` — CREATE `sampleMarginalCovariates`, `sampleMarginalGroup`.
- `types.ts` — ADD `Correlation.reference_topic?`; `GatingSpec.group_var_label?`/`group_labels?`/`group_proportions?`.
- `store.ts` — per-panel conditioning stores + cohort-change reset.
- `simulator/runSamples.ts`, `tabs/Simulator.svelte` — conditioned draw for STM.
- `cohort.ts`, `tabs/Patient.svelte`, `patient/PatientMap.svelte` — conditioned `generateCohort`, per-patient group, sample-vs-set, color-by-group.
- `atlas/CovariatePanel.svelte` — DELETE (orphaned).
- `App.svelte` — remove global `ConditioningBar` mount.
- `conditioning/*.test.ts`, `cohort.test.ts` — vitest.

**Docs:**
- `docs/decisions/0035-dashboard-logistic-normal-forward-sampler.md` — CREATE (supersedes 0028).

---

## Task 1: Export — `correlation.json` `reference_topic`

**Files:**
- Modify: `charmpheno/charmpheno/export/correlation.py:18-43`
- Test: `charmpheno/tests/test_correlation_export.py`

**Interfaces:**
- Consumes: `build_correlation_json(R, identified, support, partition, kept_topic_ids, reference_id=None)` (existing signature; already drops `reference_id` from `topic_order`).
- Produces: the returned dict gains `"reference_topic": reference_id` (int or None).

- [ ] **Step 1: Write the failing test**

Add to `charmpheno/tests/test_correlation_export.py`:

```python
def test_build_correlation_json_emits_reference_topic():
    """The reference topic id (pinned eta=0, excluded from R/topic_order) must
    be reported explicitly so the dashboard sampler can place the K-1 free
    topics into the K-topic softmax without inferring it from a zero Gamma row."""
    from charmpheno.export.correlation import build_correlation_json

    class _P:
        group_var = "source_cohort"
        groups = ["cancer"]
        def topic_labels(self):
            return ["background", "background", "cancer"]
    R = [[1.0, 0.2, 0.1], [0.2, 1.0, 0.0], [0.1, 0.0, 1.0]]
    ident = [[True] * 3 for _ in range(3)]
    sup = [[9] * 3 for _ in range(3)]
    out = build_correlation_json(R, ident, sup, _P(), [0, 1, 2], reference_id=0)
    assert out["reference_topic"] == 0
    assert 0 not in out["topic_order"]        # reference excluded from R order

def test_build_correlation_json_reference_topic_none_when_absent():
    from charmpheno.export.correlation import build_correlation_json

    class _P:
        group_var = "g"
        groups = []
        def topic_labels(self):
            return ["background", "background"]
    R = [[1.0, 0.3], [0.3, 1.0]]
    ident = [[True, True], [True, True]]
    sup = [[9, 9], [9, 9]]
    out = build_correlation_json(R, ident, sup, _P(), [0, 1], reference_id=None)
    assert out["reference_topic"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && python3 -m pytest charmpheno/tests/test_correlation_export.py -k reference_topic -q`
Expected: FAIL with `KeyError: 'reference_topic'`.

- [ ] **Step 3: Write minimal implementation**

In `charmpheno/charmpheno/export/correlation.py`, add one key to the returned dict:

```python
    return {
        "topic_order": [int(i) for i in order],
        "block_labels": block_labels,
        "R": R_out,
        "identified": id_out,
        "support": sup_out,
        "reference_topic": (int(reference_id) if reference_id is not None else None),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest charmpheno/tests/test_correlation_export.py -q`
Expected: PASS (all correlation-export tests).

- [ ] **Step 5: Commit**

```bash
git add charmpheno/charmpheno/export/correlation.py charmpheno/tests/test_correlation_export.py
git commit -m "feat(export): correlation.json reports explicit reference_topic id"
```

---

## Task 2: Export — `gating.json` labels + `group_proportions`

**Files:**
- Modify: `charmpheno/charmpheno/export/gating.py:23-40`
- Modify: `analysis/local/build_dashboard.py:201`, `analysis/cloud/build_dashboard_cloud.py:604-605`
- Test: `charmpheno/tests/test_gating_export.py`

**Interfaces:**
- Consumes: `build_gating_json(partition, group_counts, k, kept_topic_ids)` (existing). `partition.group_var` (str), `partition.groups` (list[str]), `group_counts` (dict[str,int]).
- Produces: returned dict gains `group_var_label` (str), `group_labels` (dict[str,str]), `group_proportions` (dict[str,float], k-anon-safe over kept groups, summing to 1). Optional `group_label_overrides: dict | None = None` param for authored labels.

- [ ] **Step 1: Write the failing test**

Add to `charmpheno/tests/test_gating_export.py`:

```python
def test_build_gating_json_emits_labels_and_proportions():
    """Gated bundle carries humanized group_var_label + per-group labels, and a
    k-anon-safe group_proportions over kept groups (summing to 1) for the
    dashboard's per-patient group draw."""
    from charmpheno.export.gating import build_gating_json

    class _P:
        group_var = "source_cohort"
        groups = ["cancer", "dementia"]
        def topic_labels(self):
            return ["background"] * 30 + ["cancer"] * 10 + ["dementia"] * 10
        def block_indices(self, g):
            return range(30, 40) if g == "cancer" else range(40, 50)
    counts = {"cancer": 9000, "dementia": 2000}
    kept = list(range(50))
    out = build_gating_json(_P(), counts, k=20, kept_topic_ids=kept)
    assert out["group_var_label"] == "Source cohort"     # humanized
    assert out["group_labels"] == {"cancer": "Cancer", "dementia": "Dementia"}
    props = out["group_proportions"]
    assert abs(sum(props.values()) - 1.0) < 1e-9
    assert abs(props["cancer"] - 9000 / 11000) < 1e-9

def test_build_gating_json_label_override():
    from charmpheno.export.gating import build_gating_json

    class _P:
        group_var = "rare_dx"
        groups = ["rare_dx"]
        def topic_labels(self):
            return ["background"] * 2 + ["rare_dx"]
        def block_indices(self, g):
            return [2]
    out = build_gating_json(
        _P(), {"rare_dx": 100}, k=20, kept_topic_ids=[0, 1, 2],
        group_label_overrides={"rare_dx": "Rare diabetes cohort"},
    )
    assert out["group_labels"]["rare_dx"] == "Rare diabetes cohort"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest charmpheno/tests/test_gating_export.py -k "labels_and_proportions or label_override" -q`
Expected: FAIL with `KeyError: 'group_var_label'`.

- [ ] **Step 3: Write minimal implementation**

Replace the body of `build_gating_json` in `charmpheno/charmpheno/export/gating.py`:

```python
def _humanize(raw: str) -> str:
    """Humanize a raw id for display: underscores to spaces, sentence-case.
    A reasonable default label so no authoring is required (source_cohort ->
    'Source cohort'); overridable at the call site for a real name."""
    s = str(raw).replace("_", " ").strip()
    return s[:1].upper() + s[1:] if s else s


def build_gating_json(partition, group_counts, k, kept_topic_ids,
                      group_label_overrides=None):
    """gating.json: kept groups (count >= k) + per-kept-topic block label +
    humanized group labels + a k-anon-safe group_proportions map over kept
    groups (fractions summing to 1; sub-k groups already excluded)."""
    overrides = group_label_overrides or {}
    kept_groups = [g for g in partition.groups
                   if int(group_counts.get(g, 0)) >= int(k)]
    labels = partition.topic_labels()                 # length K, by original id
    topic_blocks = [labels[i] for i in kept_topic_ids]
    kept_counts = {g: int(group_counts.get(g, 0)) for g in kept_groups}
    total = sum(kept_counts.values()) or 1
    return {
        "group_var": partition.group_var,
        "group_var_label": _humanize(partition.group_var),
        "groups": kept_groups,
        "group_labels": {g: overrides.get(g, _humanize(g)) for g in kept_groups},
        "group_proportions": {g: kept_counts[g] / total for g in kept_groups},
        "topic_blocks": topic_blocks,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest charmpheno/tests/test_gating_export.py -q`
Expected: PASS.

- [ ] **Step 5: Verify both builders still call correctly (no signature break)**

The new param is optional, so existing calls at `analysis/local/build_dashboard.py:201` and `analysis/cloud/build_dashboard_cloud.py:604-605` remain valid. Confirm cloud driver still imports:

Run: `cd analysis/cloud && python3 -c "import ast; ast.parse(open('build_dashboard_cloud.py').read()); print('ok')"`
Expected: `ok`

- [ ] **Step 6: Commit**

```bash
git add charmpheno/charmpheno/export/gating.py charmpheno/tests/test_gating_export.py
git commit -m "feat(export): gating.json emits group_var_label, group_labels, group_proportions"
```

---

## Task 3: FE kernel — `sampleStandardNormal` + `cholesky` + `mvnDraw`

**Files:**
- Modify: `dashboard/src/lib/sampling.ts`
- Create: `dashboard/src/lib/conditioning/logisticNormal.ts`
- Test: `dashboard/src/lib/conditioning/logisticNormal.test.ts`

**Interfaces:**
- Consumes: `createRng(seed): () => number` from `../sampling`.
- Produces:
  - `sampleStandardNormal(rng: () => number): number` (in `sampling.ts`).
  - `cholesky(A: number[][]): number[][]` — lower-triangular L with L·Lᵀ = A; throws on non-PD.
  - `mvnDraw(mean: number[], L: number[][], rng: () => number): number[]` — one draw mean + L·z.

- [ ] **Step 1: Write the failing test**

Create `dashboard/src/lib/conditioning/logisticNormal.test.ts`:

```typescript
import { describe, it, expect } from 'vitest'
import { createRng } from '../sampling'
import { cholesky, mvnDraw } from './logisticNormal'

describe('cholesky', () => {
  it('reconstructs the matrix: L Lᵀ = A', () => {
    const A = [[4, 2, 0], [2, 5, 1], [0, 1, 3]]
    const L = cholesky(A)
    const K = A.length
    for (let i = 0; i < K; i++)
      for (let j = 0; j < K; j++) {
        let s = 0
        for (let k = 0; k < K; k++) s += L[i][k] * L[j][k]
        expect(s).toBeCloseTo(A[i][j], 10)
      }
    // lower-triangular
    expect(L[0][1]).toBe(0)
    expect(L[0][2]).toBe(0)
    expect(L[1][2]).toBe(0)
  })

  it('throws on a non-positive-definite matrix', () => {
    expect(() => cholesky([[1, 2], [2, 1]])).toThrow()
  })
})

describe('mvnDraw', () => {
  it('sample mean and covariance converge to (mean, Sigma)', () => {
    const mean = [1, -2]
    const Sigma = [[2, 0.8], [0.8, 1]]
    const L = cholesky(Sigma)
    const rng = createRng(123)
    const N = 40000
    const draws: number[][] = []
    for (let i = 0; i < N; i++) draws.push(mvnDraw(mean, L, rng))
    const m = [0, 0]
    for (const d of draws) { m[0] += d[0]; m[1] += d[1] }
    m[0] /= N; m[1] /= N
    expect(m[0]).toBeCloseTo(mean[0], 1)
    expect(m[1]).toBeCloseTo(mean[1], 1)
    let c00 = 0, c01 = 0, c11 = 0
    for (const d of draws) {
      c00 += (d[0] - m[0]) ** 2
      c01 += (d[0] - m[0]) * (d[1] - m[1])
      c11 += (d[1] - m[1]) ** 2
    }
    expect(c00 / N).toBeCloseTo(Sigma[0][0], 1)
    expect(c01 / N).toBeCloseTo(Sigma[0][1], 1)
    expect(c11 / N).toBeCloseTo(Sigma[1][1], 1)
  })

  it('is deterministic under a seeded RNG', () => {
    const L = cholesky([[1, 0], [0, 1]])
    const a = mvnDraw([0, 0], L, createRng(7))
    const b = mvnDraw([0, 0], L, createRng(7))
    expect(a).toEqual(b)
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd dashboard && npx vitest run src/lib/conditioning/logisticNormal.test.ts`
Expected: FAIL — cannot import `cholesky`/`mvnDraw` (module missing).

- [ ] **Step 3: Write minimal implementation**

Append to `dashboard/src/lib/sampling.ts`:

```typescript
// Standard-normal draw via the Marsaglia polar method (rejection form of
// Box-Muller). Reuses the shared uniform RNG so seeded runs stay reproducible.
export function sampleStandardNormal(rng: () => number): number {
  let u: number, v: number, s: number
  do {
    u = 2 * rng() - 1
    v = 2 * rng() - 1
    s = u * u + v * v
  } while (s >= 1 || s === 0)
  return u * Math.sqrt((-2 * Math.log(s)) / s)
}
```

Create `dashboard/src/lib/conditioning/logisticNormal.ts`:

```typescript
import { sampleStandardNormal } from '../sampling'

// Lower-triangular Cholesky factor L with L Lᵀ = A. Throws if A is not
// positive-definite (a non-positive pivot). Textbook Cholesky-Banachiewicz;
// the covariance sub-blocks it factors here are small (~40x40).
export function cholesky(A: number[][]): number[][] {
  const n = A.length
  const L: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0))
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = A[i][j]
      for (let k = 0; k < j; k++) sum -= L[i][k] * L[j][k]
      if (i === j) {
        if (sum <= 0) throw new Error('cholesky: matrix is not positive-definite')
        L[i][j] = Math.sqrt(sum)
      } else {
        L[i][j] = sum / L[j][j]
      }
    }
  }
  return L
}

// One draw from Normal(mean, L Lᵀ): mean + L z, z standard-normal.
export function mvnDraw(mean: number[], L: number[][], rng: () => number): number[] {
  const n = mean.length
  const z = new Array<number>(n)
  for (let i = 0; i < n; i++) z[i] = sampleStandardNormal(rng)
  const out = new Array<number>(n)
  for (let i = 0; i < n; i++) {
    let s = mean[i]
    for (let k = 0; k <= i; k++) s += L[i][k] * z[k]
    out[i] = s
  }
  return out
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd dashboard && npx vitest run src/lib/conditioning/logisticNormal.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/lib/sampling.ts dashboard/src/lib/conditioning/logisticNormal.ts dashboard/src/lib/conditioning/logisticNormal.test.ts
git commit -m "feat(dashboard): hand-rolled cholesky + mvnDraw kernel (no deps)"
```

---

## Task 4: FE — `sampleConditionedTheta` + type additions

**Files:**
- Modify: `dashboard/src/lib/conditioning/logisticNormal.ts`
- Modify: `dashboard/src/lib/types.ts:65-77`
- Test: `dashboard/src/lib/conditioning/logisticNormal.test.ts`

**Interfaces:**
- Consumes: `cholesky`, `mvnDraw` (Task 3); `buildDesignVector` / `covariatePrevalence` exist in `covariate.ts`; types `CovariateEffects`, `Correlation`, `GatingSpec` from `../types`.
- Produces:
  - `types.ts`: `Correlation` gains `reference_topic?: number | null`; `GatingSpec` gains `group_var_label?: string`, `group_labels?: Record<string, string>`, `group_proportions?: Record<string, number>`.
  - `sampleConditionedTheta(args): number[]` returning a length-K θ over all display topics (masked topics exactly 0). Signature:
    ```typescript
    sampleConditionedTheta(args: {
      effects: CovariateEffects        // Gamma rows; per_topic length K
      x: number[]                      // design vector (buildDesignVector)
      correlation: Correlation         // R over K-1 free topics + reference_topic + topic_order
      topicBlocks: string[] | null     // gating.topic_blocks (length K) or null
      group: string | null             // selected group, or null
      rng: () => number
    }): number[]
    ```

- [ ] **Step 1: Write the failing test**

Add to `dashboard/src/lib/conditioning/logisticNormal.test.ts`:

```typescript
import { sampleConditionedTheta } from './logisticNormal'
import type { Correlation, CovariateEffects } from '../types'

function identityCorr(K1: number, order: number[]): Correlation {
  const R = Array.from({ length: K1 }, (_, i) =>
    Array.from({ length: K1 }, (_, j) => (i === j ? 1 : 0)))
  return {
    topic_order: order,
    block_labels: order.map(() => 'background'),
    R,
    identified: R.map((row) => row.map(() => true)),
    support: R.map((row) => row.map(() => 9)),
    reference_topic: 0,
  }
}

describe('sampleConditionedTheta', () => {
  it('returns a length-K distribution with reference topic drawn around eta=0', () => {
    // K=3: reference topic 0, free topics 1..2. Effects zero -> mean eta = 0.
    const effects: CovariateEffects = [
      { covariate: 'Intercept', per_topic: [0, 0, 0] },
    ]
    const corr = identityCorr(2, [1, 2])
    const theta = sampleConditionedTheta({
      effects, x: [1], correlation: corr,
      topicBlocks: null, group: null, rng: createRng(3),
    })
    expect(theta.length).toBe(3)
    const sum = theta.reduce((a, b) => a + b, 0)
    expect(sum).toBeCloseTo(1, 10)
    for (const p of theta) expect(p).toBeGreaterThan(0)
  })

  it('gives out-of-group foreground topics exactly zero mass', () => {
    // K=4: topic 0 reference(bg), 1 bg, 2 cancer, 3 dementia. Select cancer.
    const effects: CovariateEffects = [
      { covariate: 'Intercept', per_topic: [0, 0, 0, 0] },
    ]
    const corr = identityCorr(3, [1, 2, 3])
    const theta = sampleConditionedTheta({
      effects, x: [1], correlation: corr,
      topicBlocks: ['background', 'background', 'cancer', 'dementia'],
      group: 'cancer', rng: createRng(5),
    })
    expect(theta[3]).toBe(0)          // dementia foreground masked out
    expect(theta[2]).toBeGreaterThan(0) // cancer foreground allowed
    expect(theta.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10)
  })

  it('shifts the mean when a covariate effect is applied', () => {
    // Effect pushes free topic 2 up; its mean share should exceed topic 1's.
    const effects: CovariateEffects = [
      { covariate: 'Intercept', per_topic: [0, 0, 0] },
      { covariate: 'age', per_topic: [0, 0, 3] },
    ]
    const corr = identityCorr(2, [1, 2])
    const rng = createRng(11)
    let s1 = 0, s2 = 0
    for (let i = 0; i < 2000; i++) {
      const t = sampleConditionedTheta({
        effects, x: [1, 1], correlation: corr,
        topicBlocks: null, group: null, rng,
      })
      s1 += t[1]; s2 += t[2]
    }
    expect(s2).toBeGreaterThan(s1)
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd dashboard && npx vitest run src/lib/conditioning/logisticNormal.test.ts`
Expected: FAIL — `sampleConditionedTheta` not exported; `Correlation.reference_topic` type error.

- [ ] **Step 3: Add the type fields**

In `dashboard/src/lib/types.ts` replace the `GatingSpec` and `Correlation` interfaces:

```typescript
export interface GatingSpec {
  group_var: string
  groups: string[]
  topic_blocks: string[]
  group_var_label?: string
  group_labels?: Record<string, string>
  group_proportions?: Record<string, number>
}

export interface Correlation {
  topic_order: number[]
  block_labels: string[]
  R: (number | null)[][]
  identified: boolean[][]
  support: number[][]
  reference_topic?: number | null
}
```

- [ ] **Step 4: Implement `sampleConditionedTheta`**

Append to `dashboard/src/lib/conditioning/logisticNormal.ts`:

```typescript
import type { CovariateEffects, Correlation } from '../types'

// Faithful STM forward draw: theta = softmax(eta), eta ~ Normal(Gamma^T x, Sigma)
// (logistic-normal prior; Blei & Lafferty 2007). The reference topic is pinned
// eta = 0 and excluded from Gamma's non-zero rows and from Sigma (correlation.R
// over the K-1 free topics). For a gated draw we restrict to the allowed set
// (background union the selected group) so the Sigma sub-block never includes a
// cross-group (unidentified/null) cell and is positive-definite by construction.
export function sampleConditionedTheta(args: {
  effects: CovariateEffects
  x: number[]
  correlation: Correlation
  topicBlocks: string[] | null
  group: string | null
  rng: () => number
}): number[] {
  const { effects, x, correlation, topicBlocks, group, rng } = args
  const K = effects[0]?.per_topic.length ?? 0
  const ref = correlation.reference_topic ?? -1
  const order = correlation.topic_order        // display id per R row (free topics)

  // Allowed display-topic ids: all topics if not gated, else background plus the
  // selected group's foreground (null group = background only).
  const allowed = (k: number): boolean => {
    if (!topicBlocks) return true
    const b = topicBlocks[k]
    return b === 'background' || b === group
  }

  // Free R rows to sample: in topic_order, allowed, and not the reference.
  const freeIdx: number[] = []          // indices into correlation.R / order
  for (let r = 0; r < order.length; r++) {
    const k = order[r]
    if (k !== ref && allowed(k)) freeIdx.push(r)
  }

  // Mean eta over the free rows: mu_k = Gamma^T x (sum over covariate effects).
  const mean = freeIdx.map((r) => {
    const k = order[r]
    let m = 0
    for (const e of effects) m += e.per_topic[k] * x[effects.indexOf(e)]
    return m
  })

  // Sigma sub-block over the free rows (guaranteed non-null / PD).
  const Sigma = freeIdx.map((ri) =>
    freeIdx.map((rj) => correlation.R[ri][rj] as number))

  const etaFree = freeIdx.length
    ? mvnDraw(mean, cholesky(Sigma), rng)
    : []

  // Assemble eta over all K display topics: reference -> 0, free -> drawn,
  // masked -> -Infinity (exactly zero after softmax).
  const eta = new Array<number>(K).fill(-Infinity)
  if (ref >= 0 && allowed(ref)) eta[ref] = 0
  freeIdx.forEach((r, i) => { eta[order[r]] = etaFree[i] })

  const finite = eta.filter((e) => e !== -Infinity)
  const mx = finite.length ? Math.max(...finite) : 0
  const exp = eta.map((e) => (e === -Infinity ? 0 : Math.exp(e - mx)))
  const s = exp.reduce((a, b) => a + b, 0) || 1
  return exp.map((e) => e / s)
}
```

Note for the implementer: `effects.indexOf(e)` inside the map is O(P) per row — acceptable (P is tiny, ~3). If a reviewer objects, replace with `effects.forEach((e, p) => m += e.per_topic[k] * x[p])`.

- [ ] **Step 5: Run test to verify it passes**

Run: `cd dashboard && npx vitest run src/lib/conditioning/logisticNormal.test.ts && npx svelte-check --tsconfig ./tsconfig.json`
Expected: PASS; no type errors.

- [ ] **Step 6: Commit**

```bash
git add dashboard/src/lib/conditioning/logisticNormal.ts dashboard/src/lib/types.ts dashboard/src/lib/conditioning/logisticNormal.test.ts
git commit -m "feat(dashboard): sampleConditionedTheta logistic-normal forward sampler"
```

---

## Task 5: FE — `marginalSampler.ts`

**Files:**
- Create: `dashboard/src/lib/conditioning/marginalSampler.ts`
- Test: `dashboard/src/lib/conditioning/marginalSampler.test.ts`

**Interfaces:**
- Consumes: `createRng` from `../sampling`; types `CovariateSchema`, `GatingSpec`.
- Produces:
  - `sampleMarginalCovariates(schema: CovariateSchema, rng): Record<string, number | string>` — per-control draw: categorical from `proportions`, continuous from a triangular distribution on `[range[0], range[1]]` peaked at `default`.
  - `sampleMarginalGroup(gating: GatingSpec, rng): string` — draw a group from `group_proportions`; uniform over `groups` + `console.warn` when `group_proportions` absent.

- [ ] **Step 1: Write the failing test**

Create `dashboard/src/lib/conditioning/marginalSampler.test.ts`:

```typescript
import { describe, it, expect } from 'vitest'
import { createRng } from '../sampling'
import { sampleMarginalCovariates, sampleMarginalGroup } from './marginalSampler'
import type { CovariateSchema, GatingSpec } from '../types'

const schema: CovariateSchema = {
  k: 2,
  controls: [
    { name: 'age', type: 'continuous', range: [40, 80], default: 70 },
    { name: 'sex', type: 'categorical', reference: 'F',
      levels: ['F', 'M'], proportions: { F: 0.9, M: 0.1 } },
  ],
  design_columns: [],
  unsupported: [],
}

describe('sampleMarginalCovariates', () => {
  it('draws continuous within range and categorical from proportions', () => {
    const rng = createRng(1)
    let mCount = 0
    for (let i = 0; i < 4000; i++) {
      const v = sampleMarginalCovariates(schema, rng)
      expect(typeof v.age).toBe('number')
      expect(v.age as number).toBeGreaterThanOrEqual(40)
      expect(v.age as number).toBeLessThanOrEqual(80)
      if (v.sex === 'M') mCount++
    }
    // ~10% M, allow slack
    expect(mCount / 4000).toBeGreaterThan(0.05)
    expect(mCount / 4000).toBeLessThan(0.15)
  })
})

describe('sampleMarginalGroup', () => {
  it('respects group_proportions', () => {
    const gating: GatingSpec = {
      group_var: 'source_cohort', groups: ['cancer', 'dementia'],
      topic_blocks: [], group_proportions: { cancer: 0.8, dementia: 0.2 },
    }
    const rng = createRng(2)
    let cancer = 0
    for (let i = 0; i < 4000; i++)
      if (sampleMarginalGroup(gating, rng) === 'cancer') cancer++
    expect(cancer / 4000).toBeGreaterThan(0.7)
    expect(cancer / 4000).toBeLessThan(0.9)
  })

  it('falls back to uniform when group_proportions absent', () => {
    const gating: GatingSpec = {
      group_var: 'g', groups: ['a', 'b'], topic_blocks: [],
    }
    const rng = createRng(3)
    const seen = new Set<string>()
    for (let i = 0; i < 50; i++) seen.add(sampleMarginalGroup(gating, rng))
    expect(seen.has('a') && seen.has('b')).toBe(true)
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd dashboard && npx vitest run src/lib/conditioning/marginalSampler.test.ts`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

Create `dashboard/src/lib/conditioning/marginalSampler.ts`:

```typescript
import { sampleCategorical } from '../sampling'
import type { CovariateSchema, GatingSpec } from '../types'

// Triangular draw on [a, b] with mode c (Stein & Keblis 2009 triangular
// inverse-CDF). A marginal-only approximation of a continuous covariate's
// spread; independent across covariates (no interactions) by design.
function sampleTriangular(a: number, b: number, c: number, rng: () => number): number {
  if (b <= a) return a
  const u = rng()
  const fc = (c - a) / (b - a)
  return u < fc
    ? a + Math.sqrt(u * (b - a) * (c - a))
    : b - Math.sqrt((1 - u) * (b - a) * (b - c))
}

// Draw a per-patient covariate value set from the model's reported marginals.
export function sampleMarginalCovariates(
  schema: CovariateSchema, rng: () => number,
): Record<string, number | string> {
  const values: Record<string, number | string> = {}
  for (const c of schema.controls) {
    if (c.type === 'continuous') {
      const [a, b] = c.range ?? [0, 1]
      const mode = c.default ?? (a + b) / 2
      values[c.name] = sampleTriangular(a, b, mode, rng)
    } else {
      const levels = c.levels ?? []
      const props = c.proportions
      if (props && levels.length) {
        const p = levels.map((l) => props[l] ?? 0)
        const s = p.reduce((x, y) => x + y, 0) || 1
        values[c.name] = levels[sampleCategorical(p.map((x) => x / s), rng)]
      } else if (levels.length) {
        values[c.name] = levels[Math.floor(rng() * levels.length)]
      }
    }
  }
  return values
}

// Draw a per-patient group from group_proportions; uniform fallback + warn.
export function sampleMarginalGroup(gating: GatingSpec, rng: () => number): string {
  const groups = gating.groups
  const props = gating.group_proportions
  if (props && groups.length) {
    const p = groups.map((g) => props[g] ?? 0)
    const s = p.reduce((x, y) => x + y, 0) || 1
    return groups[sampleCategorical(p.map((x) => x / s), rng)]
  }
  console.warn('[marginalSampler] gating.group_proportions absent; sampling groups uniformly')
  return groups[Math.floor(rng() * groups.length)]
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd dashboard && npx vitest run src/lib/conditioning/marginalSampler.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/lib/conditioning/marginalSampler.ts dashboard/src/lib/conditioning/marginalSampler.test.ts
git commit -m "feat(dashboard): per-patient marginal covariate/group sampler"
```

---

## Task 6: Store — per-panel conditioning state + cohort-change reset

**Files:**
- Modify: `dashboard/src/lib/store.ts:70-77`
- Test: `dashboard/src/lib/store.covariate.test.ts` (extend)

**Interfaces:**
- Consumes: existing `bundle` store; `Conditioning` interface.
- Produces:
  - A factory `createConditioning(): Writable<Conditioning>` and three named panel stores `atlasConditioning`, `simulatorConditioning`, `patientConditioning`, each an independent `Conditioning` store.
  - `resetConditioningForCohort()` — resets all three to `{covariateActive:false, values:{}, group:null}`; wired to fire on cohort id change (not tab switch).
  - The existing `prevalenceReader` reads `atlasConditioning` (renamed from `conditioning`).

- [ ] **Step 1: Write the failing test**

Add to `dashboard/src/lib/store.covariate.test.ts`:

```typescript
import { get } from 'svelte/store'
import {
  atlasConditioning, simulatorConditioning, patientConditioning,
  resetConditioningForCohort,
} from './store'

it('panel conditioning stores are independent', () => {
  atlasConditioning.set({ covariateActive: true, values: { age: 70 }, group: 'cancer' })
  simulatorConditioning.set({ covariateActive: false, values: {}, group: null })
  expect(get(atlasConditioning).group).toBe('cancer')
  expect(get(simulatorConditioning).group).toBe(null)   // not shared
})

it('resetConditioningForCohort clears all panels', () => {
  atlasConditioning.set({ covariateActive: true, values: { age: 70 }, group: 'cancer' })
  patientConditioning.set({ covariateActive: true, values: { age: 40 }, group: 'dementia' })
  resetConditioningForCohort()
  expect(get(atlasConditioning)).toEqual({ covariateActive: false, values: {}, group: null })
  expect(get(patientConditioning)).toEqual({ covariateActive: false, values: {}, group: null })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd dashboard && npx vitest run src/lib/store.covariate.test.ts`
Expected: FAIL — `atlasConditioning` not exported.

- [ ] **Step 3: Write minimal implementation**

In `dashboard/src/lib/store.ts` replace the `conditioning` block:

```typescript
export interface Conditioning {
  covariateActive: boolean
  values: Record<string, number | string>
  group: string | null
}

function createConditioning() {
  return writable<Conditioning>({ covariateActive: false, values: {}, group: null })
}

// Per-panel, independent conditioning state. Each survives its own panel's
// unmount/remount (fixing the Phase-1 tab-switch-resets bug); state is shared
// by NO other panel. Reset only on cohort/bundle change (see below).
export const atlasConditioning = createConditioning()
export const simulatorConditioning = createConditioning()
export const patientConditioning = createConditioning()

export function resetConditioningForCohort(): void {
  for (const c of [atlasConditioning, simulatorConditioning, patientConditioning])
    c.set({ covariateActive: false, values: {}, group: null })
}

// Back-compat alias: the shipped four-quadrant prevalenceReader reads the
// Phenotype Atlas's conditioning.
export const conditioning = atlasConditioning
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd dashboard && npx vitest run src/lib/store.covariate.test.ts`
Expected: PASS. Also run the full FE suite so `prevalenceReader` (which imported `conditioning`) still resolves via the alias:
Run: `npx vitest run && npx svelte-check --tsconfig ./tsconfig.json`
Expected: PASS; no type errors.

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/lib/store.ts dashboard/src/lib/store.covariate.test.ts
git commit -m "feat(dashboard): per-panel conditioning stores + cohort-change reset"
```

---

## Task 7: Simulator wiring — conditioned draw for STM

**Files:**
- Modify: `dashboard/src/lib/simulator/runSamples.ts`
- Modify: `dashboard/src/lib/tabs/Simulator.svelte:34-49`
- Test: `dashboard/src/lib/simulator/runSamples.test.ts` (create if absent)

**Interfaces:**
- Consumes: `sampleConditionedTheta` (Task 4), `atlasConditioning`/`simulatorConditioning` (Task 6), `buildDesignVector` (`covariate.ts`), bundle fields `covariateEffects`, `correlation`, `gating`.
- Produces: `runSimulator` accepts an optional `conditionedTheta?: () => number[]` factory; when present it replaces the Dirichlet-prior θ for each sample (the prefix E-step, when a prefix exists, is unchanged and seeded from that θ's implied prior mean). When absent, behavior is exactly today's.

- [ ] **Step 1: Write the failing test**

Create `dashboard/src/lib/simulator/runSamples.test.ts`:

```typescript
import { describe, it, expect } from 'vitest'
import { runSimulator } from './runSamples'

describe('runSimulator conditioned θ', () => {
  it('uses the injected conditionedTheta for the no-prefix draw', () => {
    // A conditionedTheta that always puts all mass on topic 1 -> generated
    // codes come only from beta[1]; the reported theta concentrates on 1.
    const beta = [[0.5, 0.5], [0.0, 1.0]]   // topic 1 emits code 1 only
    const res = runSimulator({
      alpha: [1, 1], beta, meanCodesPerDoc: 20, prefix: [],
      nSamples: 5, seed: 1,
      conditionedTheta: () => [0, 1],
    })
    // All sampled codes should be code index 1.
    for (const bag of res.codeCountsSamples) {
      for (const [w] of bag) expect(w).toBe(1)
    }
  })

  it('without conditionedTheta behaves as before (Dirichlet path)', () => {
    const beta = [[0.5, 0.5], [0.5, 0.5]]
    const res = runSimulator({
      alpha: [1, 1], beta, meanCodesPerDoc: 10, prefix: [], nSamples: 3, seed: 1,
    })
    expect(res.thetaSamples.length).toBe(3)
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd dashboard && npx vitest run src/lib/simulator/runSamples.test.ts`
Expected: FAIL — `conditionedTheta` option ignored (codes not all index 1).

- [ ] **Step 3: Write minimal implementation**

In `dashboard/src/lib/simulator/runSamples.ts`, add `conditionedTheta` to the input interface and use it. Replace the sample loop body:

```typescript
export interface SimulatorRunInput {
  alpha: number[]
  beta: number[][]
  meanCodesPerDoc: number
  prefix: number[]
  nSamples: number
  seed: number
  autoregressive?: boolean
  // STM only: a factory returning one conditioned theta draw. When present,
  // each sample's generative theta is this logistic-normal draw instead of a
  // Dirichlet prior draw. The prefix E-step (if a prefix exists) still refines
  // theta against the observed codes; it is seeded from `alpha` as before.
  conditionedTheta?: () => number[]
}
```

Inside `runSimulator`, destructure `conditionedTheta`, and where each sample's initial θ is established:

```typescript
  for (let s = 0; s < nSamples; s++) {
    const nNew = Math.max(1, samplePoisson(meanCodesPerDoc, rng))
    const sampleCounts = new Map(prefixCounts)
    // Generative theta for THIS sample: conditioned logistic-normal draw when
    // provided (STM), else the prefix E-step's Dirichlet-based estimate.
    let genTheta: number[]
    let est = variationalEStep({ alpha, beta, codeCounts: sampleCounts })
    if (conditionedTheta) {
      genTheta = conditionedTheta()
    } else {
      genTheta = est.theta
    }
    for (let n = 0; n < nNew; n++) {
      const z = sampleCategorical(genTheta, rng)
      const w = sampleCategorical(beta[z], rng)
      sampleCounts.set(w, (sampleCounts.get(w) ?? 0) + 1)
      if (autoregressive && !conditionedTheta) {
        est = variationalEStep({ alpha, beta, codeCounts: sampleCounts })
        genTheta = est.theta
      }
    }
    // Reported theta: refine against all counts (prefix + generated) unless a
    // conditioned draw was used with no prefix, in which case report it directly.
    if (!conditionedTheta || prefix.length > 0) {
      est = variationalEStep({ alpha, beta, codeCounts: sampleCounts })
      thetas.push(est.theta)
    } else {
      thetas.push(genTheta)
    }
    const completion = new Map<number, number>()
    for (const [w, c] of sampleCounts) {
      const pre = prefixCounts.get(w) ?? 0
      if (c - pre > 0) completion.set(w, c - pre)
    }
    bags.push(completion)
  }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd dashboard && npx vitest run src/lib/simulator/runSamples.test.ts`
Expected: PASS.

- [ ] **Step 5: Wire the Simulator panel**

In `dashboard/src/lib/tabs/Simulator.svelte`, build the conditioned factory when the bundle is STM and pass it. Add near the `simulate()` body, replacing the `runSimulator({...})` call:

```typescript
  import { simulatorConditioning } from '../store'
  import { buildDesignVector } from '../covariate'
  import { sampleConditionedTheta } from '../conditioning/logisticNormal'
  import { createRng } from '../sampling'
```

```typescript
    const b = $bundle
    const cond = $simulatorConditioning
    const isStm = !!b.covariateEffects && !!b.correlation
    let conditionedTheta: (() => number[]) | undefined
    if (isStm) {
      const schema = b.covariateSchema!
      const x = buildDesignVector(schema.design_columns, cond.values)
      const tRng = createRng(seed ^ 0x9e3779b9)
      conditionedTheta = () => sampleConditionedTheta({
        effects: b.covariateEffects!, x, correlation: b.correlation!,
        topicBlocks: b.gating?.topic_blocks ?? null, group: cond.group, rng: tRng,
      })
    }
    result = runSimulator({
      alpha: b.model.alpha, beta: b.model.beta,
      meanCodesPerDoc: b.corpusStats.mean_codes_per_doc,
      prefix: $simulatorPrefix, nSamples, seed, autoregressive,
      conditionedTheta,
    })
```

- [ ] **Step 6: Verify build + type-check**

Run: `cd dashboard && npx vitest run && npx svelte-check --tsconfig ./tsconfig.json && npm run build`
Expected: PASS; build succeeds.

- [ ] **Step 7: Commit**

```bash
git add dashboard/src/lib/simulator/runSamples.ts dashboard/src/lib/simulator/runSamples.test.ts dashboard/src/lib/tabs/Simulator.svelte
git commit -m "feat(dashboard): Simulator conditions its draw via logistic-normal sampler (STM)"
```

---

## Task 8: Patient Atlas wiring — conditioned cohort, per-patient group, sample/set, color-by-group

**Files:**
- Modify: `dashboard/src/lib/cohort.ts` (`CohortInput`, `generateCohort`), `dashboard/src/lib/types.ts` (`SyntheticPatient` gains `group?`)
- Modify: `dashboard/src/lib/tabs/Patient.svelte`, `dashboard/src/lib/patient/PatientMap.svelte`
- Test: `dashboard/src/lib/cohort.test.ts` (extend)

**Interfaces:**
- Consumes: `sampleConditionedTheta`, `sampleMarginalCovariates`, `sampleMarginalGroup`, `buildDesignVector`, `patientConditioning`, bundle STM fields.
- Produces: `CohortInput` gains optional `conditioning?: { mode: 'sample' | 'set'; values: Record<string,number|string>; group: string|null; bundle: DashboardBundle }`. When present and the bundle is STM, each patient's θ comes from `sampleConditionedTheta` (set mode: shared x/group; sample mode: per-patient marginal draw). Each `SyntheticPatient` gains `group?: string | null`.

- [ ] **Step 1: Write the failing test**

Add to `dashboard/src/lib/cohort.test.ts`:

```typescript
it('set mode: all patients conditioned at the same group; per-patient group recorded', () => {
  const bundle: any = {
    model: { K: 4, V: 2, alpha: [1, 1, 1, 1], beta: [[.5, .5], [.5, .5], [.9, .1], [.1, .9]] },
    covariateSchema: { k: 1, controls: [], design_columns: [{ name: 'Intercept', recipe: { kind: 'intercept' } }], unsupported: [] },
    covariateEffects: [{ covariate: 'Intercept', per_topic: [0, 0, 0, 0] }],
    correlation: {
      topic_order: [1, 2, 3], block_labels: ['background', 'cancer', 'dementia'],
      R: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
      identified: [[true, true, true], [true, true, true], [true, true, true]],
      support: [[9, 9, 9], [9, 9, 9], [9, 9, 9]], reference_topic: 0,
    },
    gating: { group_var: 'g', groups: ['cancer', 'dementia'], topic_blocks: ['background', 'background', 'cancer', 'dementia'], group_proportions: { cancer: 0.8, dementia: 0.2 } },
    corpusStats: { mean_codes_per_doc: 10 },
  }
  const c = generateCohort({
    model: bundle.model, meanCodesPerDoc: 10, n: 20, seed: 1, nNeighbors: 3,
    conditioning: { mode: 'set', values: {}, group: 'cancer', bundle },
  })
  // set mode -> every patient is cancer; dementia foreground (topic 3) is masked.
  for (const p of c.patients) {
    expect(p.group).toBe('cancer')
    expect(p.theta[3]).toBe(0)
  }
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd dashboard && npx vitest run src/lib/cohort.test.ts`
Expected: FAIL — `conditioning` option ignored / `p.group` undefined.

- [ ] **Step 3: Implement**

In `dashboard/src/lib/types.ts`, add `group?: string | null` to `SyntheticPatient`.

In `dashboard/src/lib/cohort.ts`: extend `CohortInput`, import the samplers, and in `drawOne` produce θ (and group) from the conditioning when present. Replace the top imports + `CohortInput` + the θ line inside `drawOne`:

```typescript
import {
  createRng, sampleDirichlet, sampleCategorical, samplePoisson,
} from './sampling'
import { sampleConditionedTheta } from './conditioning/logisticNormal'
import { sampleMarginalCovariates, sampleMarginalGroup } from './conditioning/marginalSampler'
import { buildDesignVector } from './covariate'
import type { DashboardBundle } from './types'
```

```typescript
export interface CohortConditioning {
  mode: 'sample' | 'set'
  values: Record<string, number | string>
  group: string | null
  bundle: DashboardBundle
}
export interface CohortInput {
  model: Model
  meanCodesPerDoc: number
  n: number
  seed: number
  nNeighbors: number
  qualityByPhenotype?: (PhenotypeQuality | null)[]
  conditioning?: CohortConditioning
}
```

Inside `generateCohort`, before `drawOne`, precompute the STM flag and a fixed design vector for set mode:

```typescript
  const cc = input.conditioning
  const stm = !!cc && !!cc.bundle.covariateEffects && !!cc.bundle.correlation
  const setX = stm && cc!.mode === 'set'
    ? buildDesignVector(cc!.bundle.covariateSchema!.design_columns, cc!.values)
    : null
```

Replace the θ + group assignment inside `drawOne`:

```typescript
  const drawOne = () => {
    let theta: number[]
    let group: string | null = null
    if (stm) {
      const b = cc!.bundle
      if (cc!.mode === 'set') {
        group = cc!.group
        theta = sampleConditionedTheta({
          effects: b.covariateEffects!, x: setX!, correlation: b.correlation!,
          topicBlocks: b.gating?.topic_blocks ?? null, group, rng,
        })
      } else {
        const vals = sampleMarginalCovariates(b.covariateSchema!, rng)
        group = b.gating ? sampleMarginalGroup(b.gating, rng) : null
        const x = buildDesignVector(b.covariateSchema!.design_columns, vals)
        theta = sampleConditionedTheta({
          effects: b.covariateEffects!, x, correlation: b.correlation!,
          topicBlocks: b.gating?.topic_blocks ?? null, group, rng,
        })
      }
    } else {
      theta = sampleDirichlet(model.alpha, rng)
    }
    const nCodes = Math.max(1, samplePoisson(meanCodesPerDoc, rng))
    const bag: number[] = []
    for (let c = 0; c < nCodes; c++) {
      const z = sampleCategorical(theta, rng)
      const w = sampleCategorical(model.beta[z], rng)
      bag.push(w)
    }
    let isClean = true
    if (qualityByPhenotype) {
      const q = qualityByPhenotype[dominantIdx(theta)]
      isClean = !(q === 'dead' || q === 'mixed')
    }
    return { theta, bag, isClean, group }
  }
```

Thread `group` through the accumulation (both the adaptive `drawAndPush` and the plain loop push a parallel `groups` array), and add it to the emitted patient:

```typescript
  // ...alongside thetas/bags/isCleanFlags, maintain `let groups: (string|null)[] = []`
  // push d.group in every place d.theta is pushed, truncate in lockstep, then:
  const patients: SyntheticPatient[] = thetas.map((theta, i) => ({
    id: pad(i),
    theta,
    code_bag: bags[i],
    neighbors: nbrIdx[i].map(pad),
    isClean: isCleanFlags[i],
    group: groups[i],
  }))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd dashboard && npx vitest run src/lib/cohort.test.ts`
Expected: PASS.

- [ ] **Step 5: Wire the panel controls (sample/set toggle + color-by-group)**

In `dashboard/src/lib/tabs/Patient.svelte`: read `patientConditioning`; add a `mode: 'sample' | 'set'` local (default `'sample'`) and a `colorByGroup` boolean (gated bundles only); pass `conditioning` into the `generateCohort` call that runs on **Regenerate**. In `dashboard/src/lib/patient/PatientMap.svelte`: when `colorByGroup` is on and patients carry `group`, color each point by `group` using the existing palette (`palette.ts`); otherwise keep today's coloring. Keep these controls next to the existing Regenerate button.

Concrete `generateCohort` call in Patient.svelte's regenerate handler:

```typescript
  const cond = $patientConditioning
  const isStm = !!$bundle.covariateEffects && !!$bundle.correlation
  const cohort = generateCohort({
    model: $bundle.model,
    meanCodesPerDoc: $bundle.corpusStats.mean_codes_per_doc,
    n: BATCH, seed, nNeighbors: N_NEIGHBORS,
    qualityByPhenotype,
    conditioning: isStm
      ? { mode, values: cond.values, group: cond.group, bundle: $bundle }
      : undefined,
  })
```

- [ ] **Step 6: Verify build + type-check**

Run: `cd dashboard && npx vitest run && npx svelte-check --tsconfig ./tsconfig.json && npm run build`
Expected: PASS; build succeeds.

- [ ] **Step 7: Commit**

```bash
git add dashboard/src/lib/cohort.ts dashboard/src/lib/cohort.test.ts dashboard/src/lib/types.ts dashboard/src/lib/tabs/Patient.svelte dashboard/src/lib/patient/PatientMap.svelte
git commit -m "feat(dashboard): Patient Atlas conditioned cohort (set/sample) + per-patient group + color-by-group"
```

---

## Task 9: Per-panel controls placement + cleanup

**Files:**
- Modify: `dashboard/src/App.svelte:214` (remove global `ConditioningBar` mount), `dashboard/src/App.svelte` (fire `resetConditioningForCohort` on cohort id change)
- Modify: `dashboard/src/lib/tabs/Atlas.svelte`, `Simulator.svelte`, `Patient.svelte` — mount the `ConditioningBar` cluster inside each panel, bound to that panel's store
- Modify: `dashboard/src/lib/conditioning/ConditioningBar.svelte` — accept a `store` prop (which panel store to read/write) instead of importing the global `conditioning`
- Delete: `dashboard/src/lib/atlas/CovariatePanel.svelte`
- Test: existing FE suite + `npm run build`

**Interfaces:**
- Consumes: `atlasConditioning`/`simulatorConditioning`/`patientConditioning`, `resetConditioningForCohort` (Task 6).
- Produces: `ConditioningBar` takes `export let store: Writable<Conditioning>`; each panel passes its own; no app-global mount.

- [ ] **Step 1: Make `ConditioningBar` take a `store` prop**

In `dashboard/src/lib/conditioning/ConditioningBar.svelte`, replace `import { conditioning, bundle } from '../store'` with `import { bundle } from '../store'` + `export let store: import('svelte/store').Writable<import('../store').Conditioning>` and use `$store` everywhere the component currently uses `$conditioning`.

- [ ] **Step 2: Mount per-panel + remove global**

- Remove `<ConditioningBar />` and its import from `dashboard/src/App.svelte`.
- In `Atlas.svelte`, `Simulator.svelte`, `Patient.svelte`, import `ConditioningBar` and the panel's store, and render `<ConditioningBar store={atlasConditioning} />` (resp. simulator/patient) in the panel's control area.

- [ ] **Step 3: Reset on cohort change (not tab switch)**

In `dashboard/src/App.svelte`, where the selected cohort id is observed (the bundle-load effect), call `resetConditioningForCohort()` when the cohort id changes. Do NOT reset on tab change.

- [ ] **Step 4: Delete the orphan**

```bash
git rm dashboard/src/lib/atlas/CovariatePanel.svelte
```

- [ ] **Step 5: Verify build + type-check + full suite**

Run: `cd dashboard && npx vitest run && npx svelte-check --tsconfig ./tsconfig.json && npm run build`
Expected: PASS; build succeeds; no dangling import of `CovariatePanel` or the app-global `ConditioningBar`.

Run: `grep -rn "CovariatePanel" dashboard/src || echo "clean"`
Expected: `clean`

- [ ] **Step 6: Commit**

```bash
git add -A dashboard/src
git commit -m "refactor(dashboard): per-panel conditioning controls; drop global bar + orphaned CovariatePanel"
```

---

## Task 10: ADR — logistic-normal forward sampler (supersedes 0028)

**Files:**
- Create: `docs/decisions/0035-dashboard-logistic-normal-forward-sampler.md`
- Modify: `docs/decisions/0028-dashboard-conditioned-dirichlet-prior.md` (supersession note at top)

- [ ] **Step 1: Write the ADR**

Create `docs/decisions/0035-dashboard-logistic-normal-forward-sampler.md` recording: Status Accepted; Context (Σ now exported via correlation.json's R, ADR 0034; ADR 0028's Dirichlet mean-match was chosen only because Σ was unavailable); Decision (forward sampling draws θ = softmax(η), η ~ Normal(Γᵀx, Σ_allowed), per-group PD sub-block, reference topic pinned η=0; hand-rolled cholesky + mvnDraw, no npm dependency; forward-only scope, prefix-posterior Laplace E-step deferred; non-STM bundles keep the Dirichlet path); Consequences (sampled patients show real topic co-occurrence; supersedes ADR 0028's mean-match). Use Unicode Greek, no LaTeX. Cite Blei & Lafferty 2007 for logistic-normal.

- [ ] **Step 2: Add supersession note to ADR 0028**

Add a top note to `docs/decisions/0028-dashboard-conditioned-dirichlet-prior.md`: "Superseded 2026-07-01 by ADR 0035 for STM bundles — Σ is now exported, so the generative panels sample from the faithful logistic-normal prior. The Dirichlet mean-match remains only for non-STM (LDA/HDP) bundles, which have no Σ."

- [ ] **Step 3: Commit**

```bash
git add docs/decisions/0035-dashboard-logistic-normal-forward-sampler.md docs/decisions/0028-dashboard-conditioned-dirichlet-prior.md
git commit -m "docs(adr): 0035 dashboard logistic-normal forward sampler (supersedes 0028)"
```

---

## Self-Review

**Spec coverage:**
- Per-panel rework (independent state, reset-on-cohort, delete CovariatePanel, remove global bar) → Tasks 6, 9. ✓
- Logistic-normal forward sampler (cholesky/mvnDraw/sampleConditionedTheta, reference topic, per-group PD sub-block, no deps) → Tasks 3, 4. ✓
- marginalSampler → Task 5. ✓
- Simulator wiring (STM→conditioned, non-STM→Dirichlet, prefix E-step seeded) → Task 7. ✓
- Patient Atlas (set/sample, per-patient group, color-by-group) → Task 8. ✓
- Export: gating labels + group_proportions (both builders), correlation reference_topic, types → Tasks 1, 2, 4. ✓
- ADR superseding 0028 → Task 10. ✓
- Display reader + correlation heatmap unchanged → not touched (confirmed). ✓

**Type consistency:** `Conditioning` shape identical across store + panels; `sampleConditionedTheta` arg object identical in Tasks 4/7/8; `Correlation.reference_topic?` added in Task 4 and consumed in 4/7/8; `GatingSpec.group_proportions?` added in Task 4, produced in Task 2, consumed in Task 5/8; `SyntheticPatient.group?` added in Task 8.

**Placeholder scan:** No TBD/TODO; every code step shows complete code; commands have expected output.

**Notes for the executor:** Run FE tests from `dashboard/`. Run Python tests from repo root with the workspace venv (`.venv/bin/python -m pytest ...`) since system Python lacks pydantic/pyspark for some suites; the export tests here are pure Python and run under either. Cloud builder changes are parity-only (`ast.parse`/`py_compile`), never executed locally.
