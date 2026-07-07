# Conditioned-Coverage Bubble Plot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Phenotype Atlas bubble size a single consistent quantity — the fraction of generatively-sampled patients who express a phenotype above a threshold τ — that modulates as demographic covariates change, replacing the Task-6b predictive-gain encoding and the inconsistent point-estimate covariate branch.

**Architecture:** Draw N patient θ from the already-built, already-tested conditional logistic-normal sampler (η ~ Normal(Γᵀx, c*·R), gated, softmax), aggregate to per-topic coverage (fraction θ_k > τ), and drive bubble size from it. Covariate-off samples the corpus's marginal covariate/group mix; covariate-on fixes the demographic profile but keeps group marginal (so foreground bubbles persist). Bubble scale is absolute, anchored to the marginal (baseline) coverage. Non-STM bundles (HDP/LDA) keep the empirical histogram fallback unchanged.

**Tech Stack:** TypeScript, Svelte stores, vitest, d3 (existing). Frontend only — no Python, no bundle-format change, no cluster re-fit or re-export. Design spec: `docs/superpowers/specs/2026-07-06-conditioned-coverage-bubble-plot-design.md`.

## Global Constraints

- Bubble size = **patient coverage** = fraction of sampled patients with θ_k > τ (a fraction in [0,1]), on an **absolute scale** anchored to the marginal (baseline) coverage — no per-state renormalization.
- **STM bundles** (have `covariateEffects` AND `correlation`) use generative coverage; **non-STM bundles** fall back to the existing `fractionAboveTau` (unchanged) — back-compat is mandatory.
- τ default is **0.01** (was 0.02). The τ store `tauThreshold` stays the single source; there is no user-facing slider.
- Cohort size **N = 1500**, fixed seed **20260706** (stable bubbles — no flicker between renders). Sampling is **synchronous** (consumer tests stay synchronous).
- Reuse the tested sampler primitives (`sampleConditionedTheta`, `sampleMarginalCovariates`, `sampleMarginalGroup`, `buildDesignVector`). **Do NOT call `generateCohort`** for the atlas (its O(N²) neighbors/bags are too heavy) and **do NOT modify `generateCohort` or the Simulator/Patient paths.**
- Group is **always** drawn per-patient from the marginal for the atlas cohort (no group fixing) so foreground/gated phenotypes stay represented.
- The predictive-gain metric (`mean_gain`/`presence`/`depth`) is **demoted, not deleted**: `mean_gain` survives as an advanced-view "distinctiveness" stat with the null band shown; `presence`/`depth` drop from the UI. The store readers `presenceReader`/`depthReader`/`meanGainReader` and the `predictiveGain` accessor **stay** (keep `store.predictive_gain.test.ts` green). The estimator/bundle/Fable research thread are untouched.
- User-facing copy says **"coverage" / "% of patients"**, never "prevalence". The internal bundle key `corpus_prevalence` is unchanged.
- No LaTeX in any copy/comment — Unicode Greek (θ, τ, Γ, Σ, η) and plain text only.

---

### Task 1: `coverage.ts` — theta-cohort sampler + coverage aggregation

**Files:**
- Create: `dashboard/src/lib/conditioning/coverage.ts`
- Test: `dashboard/src/lib/conditioning/coverage.test.ts`

**Interfaces:**
- Consumes (existing, do not modify): `sampleConditionedTheta({effects, x, correlation, topicBlocks, group, rng}): number[]` from `./logisticNormal`; `sampleMarginalCovariates(schema, rng)` and `sampleMarginalGroup(gating, rng)` from `./marginalSampler`; `buildDesignVector(design_columns, values): number[]` from `../covariate`; `createRng(seed): () => number` from `../sampling`; types `DashboardBundle` from `../types`.
- Produces (later tasks rely on these exact signatures):
  - `sampleThetaCohort(input: ThetaCohortInput): number[][]` where `ThetaCohortInput = { bundle: DashboardBundle; active: boolean; values: Record<string, number | string>; n: number; seed: number }`
  - `cohortCoverage(thetas: number[][], tau: number, K: number): number[]`

- [ ] **Step 1: Write the failing tests**

Create `dashboard/src/lib/conditioning/coverage.test.ts`:

```ts
import { describe, it, expect } from 'vitest'
import { cohortCoverage, sampleThetaCohort } from './coverage'
import { makeStmBundleFixture } from '../test-fixtures'

describe('cohortCoverage', () => {
  it('counts the fraction of patients with theta_k strictly greater than tau', () => {
    const thetas = [
      [0.5, 0.01, 0.0],
      [0.3, 0.03, 0.0],
    ]
    // tau=0.02: topic0 -> both>0.02 (1.0); topic1 -> only 0.03>0.02 (0.5);
    // topic2 -> none (0.0). Strict >: 0.02 itself would NOT count.
    expect(cohortCoverage(thetas, 0.02, 3)).toEqual([1.0, 0.5, 0.0])
  })

  it('treats theta_k == tau as NOT covered (strict inequality)', () => {
    expect(cohortCoverage([[0.02, 0.0, 0.0]], 0.02, 3)).toEqual([0.0, 0.0, 0.0])
  })

  it('returns all-zero coverage for an empty cohort', () => {
    expect(cohortCoverage([], 0.01, 4)).toEqual([0, 0, 0, 0])
  })
})

describe('sampleThetaCohort', () => {
  const base = { values: {}, n: 200, seed: 20260706 }

  it('returns n theta vectors of length K that sum to ~1 (softmax over free rows)', () => {
    const bundle = makeStmBundleFixture()
    const thetas = sampleThetaCohort({ bundle, active: false, ...base })
    expect(thetas).toHaveLength(200)
    for (const t of thetas) {
      expect(t).toHaveLength(3)
      expect(t.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6)
    }
  })

  it('is deterministic for a fixed seed', () => {
    const bundle = makeStmBundleFixture()
    const a = sampleThetaCohort({ bundle, active: false, ...base })
    const b = sampleThetaCohort({ bundle, active: false, ...base })
    expect(a).toEqual(b)
  })

  it('with active covariates fixed at high age, coverage shifts to the age-loaded topic', () => {
    // Fixture Gamma: topic1 eta = 1.0 - 0.02*age, topic2 eta = 0.5 + 0.03*age.
    // At age=100 topic2 dominates topic1, so its coverage is higher.
    const bundle = makeStmBundleFixture()
    const thetas = sampleThetaCohort({ bundle, active: true, values: { age: 100 }, n: 1500, seed: 20260706 })
    const cov = cohortCoverage(thetas, 0.01, 3)
    expect(cov[2]).toBeGreaterThan(cov[1])
  })
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd dashboard && npx vitest run src/lib/conditioning/coverage.test.ts`
Expected: FAIL — `Failed to resolve import "./coverage"` (module does not exist yet).

- [ ] **Step 3: Write the implementation**

Create `dashboard/src/lib/conditioning/coverage.ts`:

```ts
import type { DashboardBundle } from '../types'
import { createRng } from '../sampling'
import { buildDesignVector } from '../covariate'
import { sampleConditionedTheta } from './logisticNormal'
import { sampleMarginalCovariates, sampleMarginalGroup } from './marginalSampler'

export interface ThetaCohortInput {
  bundle: DashboardBundle
  // true  -> covariates fixed to `values`, group still per-patient marginal.
  // false -> covariates AND group both per-patient marginal (corpus mix).
  active: boolean
  values: Record<string, number | string>
  n: number
  seed: number
}

// Draw `n` patient theta vectors from the faithful conditional logistic-normal
// (sampleConditionedTheta). STM bundles only — callers must check
// covariateEffects + correlation are present before invoking. Group is ALWAYS a
// per-patient marginal draw so gated foreground phenotypes remain represented.
export function sampleThetaCohort(input: ThetaCohortInput): number[][] {
  const { bundle: b, active, values, n, seed } = input
  const effects = b.covariateEffects!
  const correlation = b.correlation!
  const schema = b.covariateSchema!
  const topicBlocks = b.gating?.topic_blocks ?? null
  const rng = createRng(seed)
  const fixedX = active ? buildDesignVector(schema.design_columns, values) : null
  const thetas: number[][] = []
  for (let i = 0; i < n; i++) {
    const group = b.gating ? sampleMarginalGroup(b.gating, rng) : null
    const x = fixedX ?? buildDesignVector(schema.design_columns, sampleMarginalCovariates(schema, rng))
    thetas.push(sampleConditionedTheta({ effects, x, correlation, topicBlocks, group, rng }))
  }
  return thetas
}

// Per-topic fraction of the cohort with theta_k > tau (strict). K is the topic
// count; an empty cohort yields all-zero coverage.
export function cohortCoverage(thetas: number[][], tau: number, K: number): number[] {
  const cov = new Array<number>(K).fill(0)
  if (thetas.length === 0) return cov
  for (const theta of thetas)
    for (let k = 0; k < K; k++)
      if (theta[k] > tau) cov[k] += 1
  for (let k = 0; k < K; k++) cov[k] /= thetas.length
  return cov
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd dashboard && npx vitest run src/lib/conditioning/coverage.test.ts`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/lib/conditioning/coverage.ts dashboard/src/lib/conditioning/coverage.test.ts
git commit -m "feat(dashboard): theta-cohort sampler + per-topic coverage (conditioned bubble plot)"
```

---

### Task 2: Store wiring — coverage readers, τ=0.01, rename `prevalenceReader`

**Files:**
- Modify: `dashboard/src/lib/store.ts` (τ default line 80; remove old `prevalenceReader` and its covariate import; add cohorts + coverage readers)
- Modify: `dashboard/src/App.svelte:6,49` · `dashboard/src/lib/atlas/CodePanel.svelte:5,36` · `dashboard/src/lib/atlas/PhenotypeBrowser.svelte:5,14` · `dashboard/src/lib/atlas/TopicMap.svelte:7,105` (mechanical rename `prevalenceReader` → `coverageReader`)
- Modify (test migration): `dashboard/src/lib/store.covariate.test.ts` · `dashboard/src/lib/atlas/PhenotypeBrowser.test.ts`

**Interfaces:**
- Consumes: `sampleThetaCohort`, `cohortCoverage` from Task 1; existing `fractionAboveTau(p, edges, tau)` (keep as-is), `atlasConditioning`, `bundle`, `tauThreshold`, type `Phenotype`, `DashboardBundle`.
- Produces:
  - `atlasBaselineThetaCohort` (derived `number[][] | null`) — marginal cohort, recomputed only on bundle change.
  - `atlasThetaCohort` (derived `number[][] | null`) — display cohort (marginal off / covariate-fixed on).
  - `coverageReader` (derived `(p: Phenotype) => number`) — display coverage; STM → cohort, non-STM → `fractionAboveTau`.
  - `baselineCoverageReader` (derived `(p: Phenotype) => number`) — marginal coverage, used by TopicMap (Task 3) as the absolute-scale anchor.

- [ ] **Step 1: Write the failing tests (migrate the two affected tests)**

Replace the softmax test in `dashboard/src/lib/store.covariate.test.ts`. Change the import on line 4 from `prevalenceReader` to `coverageReader`, and replace the test at lines 28–34 with:

```ts
it('covariateActive drives coverageReader from the generative cohort (age-loaded topic wins)', () => {
  // Fixture Gamma: topic2 carries a strong positive age effect, topic1 a small
  // negative one. At age=100 topic2's expected mass — and thus its patient
  // coverage — exceeds topic1's. (Old behavior asserted an exact softmax(Gamma^T x)
  // point estimate; coverage is now a sampled fraction, so we assert the order.)
  bundle.set(makeStmBundleFixture())
  conditioning.set({ covariateActive: true, values: { age: 100 }, group: null })
  const reader = get(coverageReader)
  expect(reader({ id: 2 } as any)).toBeGreaterThan(reader({ id: 1 } as any))
})
```

Add the fixture import at the top of the file: `import { makeStmBundleFixture } from './test-fixtures'`. Leave the other four tests unchanged (they use non-STM/gated fixtures without `correlation`, so they hit the `fractionAboveTau` fallback and still assert `corpus_prevalence`; rename their `prevalenceReader` references to `coverageReader`).

In `dashboard/src/lib/atlas/PhenotypeBrowser.test.ts`, change the flip in the re-sort test (lines 22–23). The marginal (covariate-off) cohort already favors the age-loaded topic 2, so flip to a LOW age where topic 1 leads (fixture crossover is age≈10):

```ts
  // Marginal coverage favors topic 2 (age-loaded); a young profile flips it to topic 1.
  atlasConditioning.set({ covariateActive: true, values: { age: 0 }, group: null })
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd dashboard && npx vitest run src/lib/store.covariate.test.ts src/lib/atlas/PhenotypeBrowser.test.ts`
Expected: FAIL — `coverageReader` is not exported yet (import error).

- [ ] **Step 3: Implement the store changes**

In `dashboard/src/lib/store.ts`:

(a) Change τ default (line 80) and its comment:

```ts
// Patient-coverage threshold τ. A patient is counted as "having" the phenotype
// when at least 1% of their coded activity is attributed to the topic. Exposed
// as a store so the components that read $tauThreshold work unchanged; there is
// no user-facing slider.
export const tauThreshold = writable<number>(0.01)
```

(b) Remove the now-unused covariate import on line 6:

```ts
// DELETE: import { buildDesignVector, covariatePrevalence } from './covariate'
```

(c) Add the coverage import near the top (after the existing imports):

```ts
import { cohortCoverage, sampleThetaCohort } from './conditioning/coverage'
```

(d) Replace the entire `prevalenceReader` derived block (lines 175–201, the comment through the closing `)`), keeping the `fractionAboveTau` function (lines 154–166) and the `tauThreshold`-comment above it, with:

```ts
const ATLAS_COHORT_N = 1500
const ATLAS_COHORT_SEED = 20260706

function isStmBundle(b: DashboardBundle | null): b is DashboardBundle {
  return !!b && !!b.covariateEffects && !!b.correlation
}

// Marginal (baseline) atlas cohort: the corpus's natural covariate/group mix.
// Recomputed ONLY when the bundle changes (not on covariate edits), so it is a
// stable per-bundle reference for absolute bubble scaling. Null for non-STM.
export const atlasBaselineThetaCohort = derived(bundle, ($b) =>
  isStmBundle($b)
    ? sampleThetaCohort({ bundle: $b, active: false, values: {}, n: ATLAS_COHORT_N, seed: ATLAS_COHORT_SEED })
    : null
)

// Display atlas cohort: the marginal baseline when covariates are off (reused —
// no resample), or a covariate-fixed cohort (group still marginal) when on.
export const atlasThetaCohort = derived(
  [bundle, atlasConditioning, atlasBaselineThetaCohort],
  ([$b, $cond, $baseline]) => {
    if (!isStmBundle($b)) return null
    if (!$cond.covariateActive) return $baseline
    return sampleThetaCohort({ bundle: $b, active: true, values: $cond.values, n: ATLAS_COHORT_N, seed: ATLAS_COHORT_SEED })
  }
)

// (Phenotype) -> coverage. STM bundles read the sampled cohort (fraction of
// patients with θ > τ); non-STM bundles fall back to the empirical θ-histogram
// fractionAboveTau, unchanged. The atlas encodes cohort as COLOR (not a filter),
// so coverage is never masked by group.
function coverageFrom(
  cohort: number[][] | null, b: DashboardBundle | null, tau: number,
): (p: Phenotype) => number {
  const edges = b?.phenotypes.theta_histogram_bin_edges
  if (cohort && b) {
    const cov = cohortCoverage(cohort, tau, b.model.K)
    return (p: Phenotype) => cov[p.id] ?? 0
  }
  return (p: Phenotype) => fractionAboveTau(p, edges, tau)
}

// Display coverage reader (the atlas's current state).
export const coverageReader = derived(
  [bundle, atlasThetaCohort, tauThreshold],
  ([$b, $cohort, $tau]) => coverageFrom($cohort, $b, $tau)
)

// Marginal (baseline) coverage reader — the stable absolute-scale anchor for
// bubble size (see TopicMap). Equals coverageReader when covariates are off.
export const baselineCoverageReader = derived(
  [bundle, atlasBaselineThetaCohort, tauThreshold],
  ([$b, $cohort, $tau]) => coverageFrom($cohort, $b, $tau)
)
```

(e) Mechanically rename `prevalenceReader` → `coverageReader` in the four consumers:
- `dashboard/src/App.svelte` line 6 (import) and line 49 (`get(prevalenceReader)` → `get(coverageReader)`).
- `dashboard/src/lib/atlas/CodePanel.svelte` line 5 (import) and line 36 (`$: reader = $prevalenceReader` → `$coverageReader`).
- `dashboard/src/lib/atlas/PhenotypeBrowser.svelte` line 5 (import), line 14 (`$prevalenceReader` → `$coverageReader`), and the two comment references on lines 51/82.
- `dashboard/src/lib/atlas/TopicMap.svelte` line 7 (import) and line 105 (`$: reader = $prevalenceReader` → `$coverageReader`). (TopicMap's size/tooltip/legend logic is reworked in Task 3; here it is only the rename so the file compiles.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd dashboard && npx vitest run src/lib/store.covariate.test.ts src/lib/atlas/PhenotypeBrowser.test.ts src/lib/store.predictive_gain.test.ts`
Expected: PASS. (`store.predictive_gain.test.ts` must stay green — the `presence`/`depth`/`meanGain` readers and `predictiveGain` accessor are untouched.)

- [ ] **Step 5: Typecheck and commit**

Run: `cd dashboard && npm run check` — Expected: no NEW errors versus baseline (the rename must not introduce any).

```bash
git add dashboard/src/lib/store.ts dashboard/src/App.svelte dashboard/src/lib/atlas/CodePanel.svelte dashboard/src/lib/atlas/PhenotypeBrowser.svelte dashboard/src/lib/atlas/TopicMap.svelte dashboard/src/lib/store.covariate.test.ts dashboard/src/lib/atlas/PhenotypeBrowser.test.ts
git commit -m "feat(dashboard): coverage readers from generative cohort; τ default 0.01; rename prevalenceReader→coverageReader"
```

---

### Task 3: TopicMap — bubble size = coverage, absolute baseline anchor, honest tooltip/legend

**Files:**
- Modify: `dashboard/src/lib/atlas/TopicMap.svelte` (size source, domain anchor, tooltip, legend; remove the Task-6b predictive-gain size branch)
- Modify: `dashboard/src/lib/copy.ts` (add `atlas.legend.coverage`)
- Replace: delete `dashboard/src/lib/atlas/TopicMap.predictive_gain.test.ts`; create `dashboard/src/lib/atlas/TopicMap.coverage.test.ts`

**Interfaces:**
- Consumes: `coverageReader`, `baselineCoverageReader` (Task 2); existing `$conditioning`, `$tauThreshold`, `$bundle`.
- Produces: no new exports (component behavior only).

- [ ] **Step 1: Write the failing test (replace the obsolete size-swap test)**

Delete `dashboard/src/lib/atlas/TopicMap.predictive_gain.test.ts` and create `dashboard/src/lib/atlas/TopicMap.coverage.test.ts`:

```ts
// Bubble size follows generative patient coverage for STM bundles, and the
// empirical fractionAboveTau fallback for non-STM bundles. Reads the main
// bubble's `r` off the rendered DOM so it catches real wiring regressions.
import { it, expect, afterEach } from 'vitest'
import { render, cleanup } from '@testing-library/svelte'
import TopicMap from './TopicMap.svelte'
import { bundle, atlasConditioning } from '../store'
import { makeStmBundleFixture } from '../test-fixtures'
import type { PredictiveGain } from '../types'

afterEach(() => { cleanup(); atlasConditioning.set({ covariateActive: false, values: {}, group: null }) })

function mainBubbleRadii(container: HTMLElement): number[] {
  return Array.from(container.querySelectorAll('svg g.node')).map((n) => {
    const c = n.querySelector('g.inner circle')
    return c ? Number(c.getAttribute('r')) : NaN
  })
}

it('STM bundle: bubble size follows generative coverage (age-loaded topic 2 > topic 1 at marginal)', () => {
  bundle.set(makeStmBundleFixture())
  const { container } = render(TopicMap)
  const [, r1, r2] = mainBubbleRadii(container)
  // Marginal cohort favors topic 2 (strong positive age effect) over topic 1.
  expect(r2).toBeGreaterThan(r1)
})

it('non-STM bundle: bubble size falls back to fractionAboveTau (corpus_prevalence order)', () => {
  const b = makeStmBundleFixture()
  // Strip the STM signals so the reader uses the histogram/corpus_prevalence fallback.
  delete (b as any).covariateEffects
  delete (b as any).correlation
  bundle.set(b)
  const { container } = render(TopicMap)
  const [, r1, r2] = mainBubbleRadii(container)
  // corpus_prevalence: id1=0.3 > id2=0.2 -> id1 bigger.
  expect(r1).toBeGreaterThan(r2)
})

it('predictive_gain present does NOT change the size source (still coverage)', () => {
  const b = makeStmBundleFixture()
  const PG: PredictiveGain = {
    presence: [0.4, 0.4, 0.4], mean_gain: [1, 1, 20], depth: [0.1, 0.1, 0.1],
    prominence_hist: [[1], [1], [1]], length_corr: [0, 0, 0], dedup_gain: [0, 0, 0],
    prominence_bin_edges: [0, 1], null_band: { mean: 0, std: 1, n: 100, p95: 1, hist: [1] },
    observed_delta_range: [-1, 1], downdate_audit: { max_abs_overall: 0.01, n_docs_audited: 100 },
    scale: 1.0, n_docs: 100,
  }
  b.phenotypes.predictive_gain = PG
  b.phenotypes.phenotypes.forEach((p, i) => { (p as any).mean_gain = PG.mean_gain[i] })
  bundle.set(b)
  const { container } = render(TopicMap)
  const [, r1, r2] = mainBubbleRadii(container)
  // mean_gain would make id2 huge either way; assert coverage order holds (r2 > r1)
  // AND that id1 is not collapsed by a mean_gain=1 (i.e. size ignores mean_gain).
  expect(r2).toBeGreaterThan(r1)
  expect(r1).toBeGreaterThan(0)
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd dashboard && npx vitest run src/lib/atlas/TopicMap.coverage.test.ts`
Expected: FAIL — the current TopicMap still sizes by `mean_gain` when `predictive_gain` is present (test 1/3 fail) and anchors the domain to `corpus_prevalence` under gating.

- [ ] **Step 3: Rework TopicMap size/anchor/tooltip/legend**

In `dashboard/src/lib/atlas/TopicMap.svelte`:

(a) Imports (line 7): drop `meanGainReader`, add `baselineCoverageReader`:

```ts
    coverageReader, baselineCoverageReader, tauThreshold, isVisibleInCurrentMode, conditioning,
```

(b) Replace the size-source block (lines 104–115) with:

```ts
  $: coords = $phenotypeCoords
  $: reader = $coverageReader
  // Bubble SIZE = patient coverage (fraction of sampled patients with θ > τ).
  // The baseline (marginal) coverage anchors an ABSOLUTE scale so covariates
  // make a phenotype genuinely grow/shrink rather than re-pinning the top bubble.
  $: sizeReader = reader
  $: baselineReader = $baselineCoverageReader
```

(c) Replace the domain-max block (lines 142–159) with a stable baseline anchor:

```ts
    // Bubble size source is coverage; anchor the size scale's domain to the
    // stable per-bundle MARGINAL coverage max so covariate changes read as
    // absolute growth/shrink. scaleSqrt = area-proportional; clamp so a topic
    // whose conditioned coverage exceeds its marginal max caps at max radius.
    const r_of = sizeReader
    const domainMax = Math.max(...allPhenotypes.map(baselineReader), 1e-9)
```

Keep the `d3.scaleSqrt().domain([0, domainMax]).range([2, 26])` that follows, and add `.clamp(true)` to it:

```ts
    const r = d3.scaleSqrt()
      .domain([0, domainMax])
      .range([2, 26])
      .clamp(true)
```

Delete the now-unused `conditioningActive` line and the `hasPredictiveGain` line.

(d) Tooltip (lines 335–351): remove `gainSuffix` and use coverage wording:

```ts
    nodes.attr('data-tip', (p) => {
      const pat = (reader(p) * 100).toFixed(1)
      const npmi = p.npmi == null ? '—' : p.npmi.toFixed(3)
      const tauStr = $tauThreshold.toFixed(2)
      const label = p.label || `Phenotype ${p.id}`
      if ($advancedView) {
        const mass = (p.corpus_prevalence * 100).toFixed(1)
        return `${label}\nCoherence ${npmi} · coverage ${pat}% of patients (θ > ${tauStr}) · topic mass ${mass}%`
      }
      return `${label}\nCoherence ${npmi} · coverage ${pat}% of patients (θ > ${tauStr})`
    })
```

(e) Reactive re-render line (358): drop `sizeReader`'s now-redundant twin but keep `reader`/`baselineReader` so the atlas re-renders on any coverage change:

```ts
  $: reader, baselineReader, $conditioning, $tauThreshold, $selectedPhenotypeId, $hoveredCodeIdx, $advancedView, $searchedConditionIdx, $bundle && svgEl && coords.length && render()
```

(f) Legend (lines 398–410): remove the `{#if hasPredictiveGain}` branch entirely and always render the coverage legend group:

```svelte
        <div class="legend-group">
          <span class="eyebrow" title={copy.atlas.legend.coverage($tauThreshold)}>Patient coverage<span class="help-mark" aria-hidden="true">?</span></span>
          <span class="size-marks" aria-hidden="true">
            <span class="dot s1"></span><span class="dot s2"></span><span class="dot s3"></span>
          </span>
        </div>
```

(Search the rest of the file for any remaining `hasPredictiveGain` / `meanGain` references and remove them; there should be none after the above.)

(g) In `dashboard/src/lib/copy.ts`, add a `coverage` entry to `atlas.legend` (next to the existing `prevalence` entry near line 61):

```ts
      coverage: (tau: number): string =>
        `Patient coverage: the fraction of patients for whom this phenotype accounts for more than ${pct(tau)}% of their coded activity (θ > ${tau.toFixed(2)}). Bubble size scales with this value; covariate sliders resample it, so bubbles grow or shrink with the population you condition on.`,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd dashboard && npx vitest run src/lib/atlas/TopicMap.coverage.test.ts`
Expected: PASS (3 tests).

- [ ] **Step 5: Typecheck and commit**

Run: `cd dashboard && npm run check` — Expected: no new errors.

```bash
git add dashboard/src/lib/atlas/TopicMap.svelte dashboard/src/lib/copy.ts dashboard/src/lib/atlas/TopicMap.coverage.test.ts
git rm dashboard/src/lib/atlas/TopicMap.predictive_gain.test.ts
git commit -m "feat(dashboard): atlas bubble size = generative patient coverage (absolute baseline anchor); honest coverage copy"
```

---

### Task 4: CodePanel + copy — coverage labels, demote predictive-gain to advanced "distinctiveness"

**Files:**
- Modify: `dashboard/src/lib/atlas/CodePanel.svelte` (stat labels; replace the mean_gain/presence/depth row with an advanced "distinctiveness" stat + null band)
- Modify: `dashboard/src/lib/copy.ts` (detail labels/tips; distinctiveness tip; atlas kicker)
- Modify: `dashboard/src/App.svelte` (kicker call site, if it passes the predictive-gain flag)
- Test: `dashboard/src/lib/atlas/CodePanel.predictive_gain.test.ts` (extend — its two histogram tests stay; add distinctiveness assertions)

**Interfaces:**
- Consumes: `coverageReader` (already wired in Task 2), the hydrated `pheno.mean_gain`, and `$predictiveGain.null_band` (bundle-level accessor already exported).
- Produces: no new exports.

- [ ] **Step 1: Write the failing tests (append to the existing predictive-gain panel test)**

Add to `dashboard/src/lib/atlas/CodePanel.predictive_gain.test.ts` (keep the two existing histogram tests unchanged). Add `predictiveGain`'s bundle wiring is already covered; append:

```ts
it('advanced view shows a Distinctiveness stat (mean_gain) and NO Presence/Depth stats', () => {
  const b = makeStmBundleFixture()
  b.phenotypes.predictive_gain = PG
  b.phenotypes.phenotypes.forEach((p, i) => {
    (p as any).mean_gain = PG.mean_gain[i]
    ;(p as any).presence = PG.presence[i]
    ;(p as any).depth = PG.depth[i]
  })
  bundle.set(b)
  const { queryByText } = render(CodePanel)   // advancedView is true (beforeEach)
  expect(queryByText('Distinctiveness')).toBeTruthy()
  expect(queryByText('Presence')).toBeNull()
  expect(queryByText('Depth')).toBeNull()
})

it('basic view hides the Distinctiveness stat', () => {
  advancedView.set(false)
  const b = makeStmBundleFixture()
  b.phenotypes.predictive_gain = PG
  b.phenotypes.phenotypes.forEach((p, i) => { (p as any).mean_gain = PG.mean_gain[i] })
  bundle.set(b)
  const { queryByText } = render(CodePanel)
  expect(queryByText('Distinctiveness')).toBeNull()
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd dashboard && npx vitest run src/lib/atlas/CodePanel.predictive_gain.test.ts`
Expected: FAIL — the panel currently renders "Mean gain"/"Presence"/"Depth" (not "Distinctiveness") and shows them outside `$advancedView`.

- [ ] **Step 3: Implement the CodePanel + copy changes**

(a) In `dashboard/src/lib/atlas/CodePanel.svelte`, import `predictiveGain` from the store (add to the line 2–6 import): `predictiveGain,`.

(b) Replace the predictive-gain stats block (lines 138–156) with an advanced-only single distinctiveness stat plus the null-band context:

```svelte
      {#if hasPredictiveGainFields && $advancedView}
        <!-- Distinctiveness = mean unique held-out predictive gain (nats): how
             specific this phenotype's vocabulary is vs. the corpus background.
             Shown with the permutation null band so a value inside the noise
             floor is visibly untrustworthy. NOT bubble size (that is coverage). -->
        <div class="stats" data-numeric>
          <span class="stat" title={copy.phenotypeDetail.distinctivenessTip}>
            <span class="stat-k">Distinctiveness<span class="help-mark" aria-hidden="true">?</span></span>
            <span class="stat-v">{pheno.mean_gain == null ? '—' : `${pheno.mean_gain.toFixed(2)} nats`}</span>
          </span>
          {#if $predictiveGain?.null_band}
            <span class="stat" title={copy.phenotypeDetail.nullBandTip}>
              <span class="stat-k">Noise floor</span>
              <span class="stat-v">{$predictiveGain.null_band.p95.toFixed(2)} nats (p95)</span>
            </span>
          {/if}
        </div>
      {/if}
```

(c) Update the main coverage stat label/tips (lines 100–108) so the panel says "coverage" not "prevalence". In `copy.ts`, the `phenotypeDetail.prevalence` object (labels/tips) is renamed to `coverage`:

In `dashboard/src/lib/copy.ts`, rename the `phenotypeDetail.prevalence` block (around line 75) to `coverage` and update wording:

```ts
    coverage: {
      labelBasic: `Coverage`,
      labelAdvanced: `Coverage`,
      tipBasic: (tau: number): string =>
        `Coverage: the share of patients for whom this phenotype accounts for more than ${pct(tau)}% of their coded activity (θ > ${tau.toFixed(2)}).`,
      tipAdvanced: (tau: number): string =>
        `Coverage: the fraction of patients with θ > τ = ${tau.toFixed(2)} — at least ${pct(tau)}% of their coded activity is attributed to this phenotype. For STM bundles this is estimated from patients sampled at the current covariate profile; otherwise from the θ-histogram.`,
      tipNoHistogram: `Coverage: the corpus-average share of activity attributed to this phenotype (no per-patient histogram available for this bundle).`,
    },
```

Then in CodePanel.svelte update the three references (lines 100–106) `copy.phenotypeDetail.prevalence.*` → `copy.phenotypeDetail.coverage.*` and the label expression to use `labelBasic`/`labelAdvanced` (already named the same, just under `coverage`).

(d) Add the distinctiveness/null-band tips to `phenotypeDetail` in `copy.ts` (repurpose the old `meanGainTip`; the `presenceTip`/`depthTip` may remain defined but are no longer referenced):

```ts
    distinctivenessTip: `Distinctiveness: this phenotype's mean unique held-out predictive gain (nats) — how specific its vocabulary is versus the corpus background (a niche phenotype scores high; a common one whose words are everywhere scores low). This is NOT how common the phenotype is (that is coverage), and it is not a per-patient count.`,
    nullBandTip: `Noise floor: the 95th-percentile predictive gain of a randomized (null) phenotype, in nats. A distinctiveness value near or below this is inside the noise and should not be over-read.`,
```

(e) Atlas kicker: in `copy.ts`, the `atlas.kicker(hasPredictiveGain)` branch (lines 48–50) no longer applies — bubble size is always coverage. Replace it with a single coverage-oriented sentence and drop the parameter:

```ts
    kicker: (): string =>
      `Each marker is a learned phenotype. Bubbles that sit closer together share more of their leading conditions; bubble size shows patient coverage — how many patients express the phenotype — and resizes as you condition on covariates.`,
```

Update the one caller — `dashboard/src/lib/tabs/Atlas.svelte:32`:

```svelte
        <p class="kicker">{copy.atlas.kicker()}</p>
```

(was `copy.atlas.kicker(!!$bundle?.phenotypes.predictive_gain)`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd dashboard && npx vitest run src/lib/atlas/CodePanel.predictive_gain.test.ts`
Expected: PASS (2 existing histogram tests + 2 new distinctiveness tests).

- [ ] **Step 5: Full suite, typecheck, build, commit**

Run: `cd dashboard && npx vitest run` — Expected: all green.
Run: `cd dashboard && npm run check` — Expected: no new errors.
Run: `cd dashboard && npm run build` — Expected: clean build.

```bash
git add dashboard/src/lib/atlas/CodePanel.svelte dashboard/src/lib/copy.ts dashboard/src/App.svelte dashboard/src/lib/atlas/CodePanel.predictive_gain.test.ts
git commit -m "feat(dashboard): detail panel coverage labels; demote predictive-gain to advanced 'distinctiveness' with null band"
```

---

## Verification (whole feature)

- `cd dashboard && npx vitest run` — full suite green (new: `coverage.test.ts`, `TopicMap.coverage.test.ts`; migrated: `store.covariate.test.ts`, `PhenotypeBrowser.test.ts`, `CodePanel.predictive_gain.test.ts`; unchanged-green: `store.predictive_gain.test.ts`, `bundle.test.ts`, `logisticNormal.test.ts`).
- `cd dashboard && npx svelte-check --threshold error` — no new errors vs. baseline.
- `cd dashboard && npm run build` — clean.
- Manual (`npm run dev`, population_cancer): bubbles sized by coverage; opening the covariate drawer and moving age/sex visibly resizes bubbles (prostate grows for male, pregnancy shrinks for older); cancer bubbles are a distinct color and smaller; tooltips say "coverage … % of patients"; advanced detail shows "Distinctiveness … nats" + "Noise floor", no "Presence"/"Depth"; no "prevalence" wording remains in the atlas.

## Notes / possible follow-ups (out of scope)

- If dragging a covariate slider feels janky (each settle resamples N=1500), debounce the atlas covariate-value writes at the input layer. Correctness does not depend on it.
- The `presenceReader`/`depthReader`/`meanGainReader` store exports remain (kept green by `store.predictive_gain.test.ts`); a later cleanup may remove the two now-unused readers.
- Reviving covariate response in the Simulator/Patient tabs (the rest of ADR 0028) and an optional directional-differentiation hue channel remain future work.
