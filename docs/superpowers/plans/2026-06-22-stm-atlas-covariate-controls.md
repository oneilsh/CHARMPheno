# STM Atlas covariate controls — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Interactive covariate controls in the Phenotype Atlas that resize topic bubbles live via `softmax(Γᵀ x)`, driven by a scrubbed covariate schema the dashboard build derives in-enclave.

**Architecture:** A pure Python function builds a scrubbed `covariate_schema.json` (controls + per-design-column recipes) from the saved ModelSpec + covariate sidecar; the dashboard build wires it in. The Svelte client loads the schema + Γ, evaluates recipes → x → `softmax(Γᵀ x)` purely client-side, and feeds the existing `prevalenceReader` so bubbles resize in place (positions fixed).

**Tech Stack:** Python/PySpark + pytest (backend); Svelte + TypeScript + vitest (frontend).

## Global Constraints

- Spec: [docs/superpowers/specs/2026-06-22-stm-atlas-covariate-controls-design.md](../specs/2026-06-22-stm-atlas-covariate-controls-design.md). Every task implicitly includes these.
- Schema derived **at dashboard-build time**, in-enclave, from `(cov_df, model_spec, covariate_names)` returned by the existing `try_load`. No re-fit, no raw person-table reload.
- **Safety (reuse existing patterns, same `k = min_patient_count`, default 20):** categorical levels with < `k` patients are suppressed/omitted; continuous ranges are coarsened percentiles (p5/p95) with default p50 — **never min/max**.
- `design_columns` are index-aligned with the Γ rows in `covariate_effects.json` (order = `covariate_names`). `name` is the join key.
- Recipe kinds: `intercept` → 1; `main` → value of `controls[var]`; `dummy` → 1 if `controls[var]==level` else 0; `interaction` → product of `factors`.
- If the sidecar is unavailable, no `covariate_schema.json` is written and the Atlas panel is hidden (graceful, like the faithful corpus_prevalence fallback).
- In covariate mode the τ slider does not apply (no per-profile histogram); bubbles size by `softmax(Γᵀ x)` mean proportion.
- Python tests: from a package dir, `../.venv/bin/python -m pytest ...`. Frontend tests: `cd dashboard && npm test` (vitest). No emojis/LaTeX. Commit trailers: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

## File Structure

- `charmpheno/charmpheno/export/covariate_schema.py` (new) — pure `build_covariate_schema(...)`.
- `analysis/cloud/build_dashboard_cloud.py` (modify) — STM branch: derive stats from sidecar, call builder, write `covariate_schema.json`.
- `dashboard/src/lib/types.ts` (modify) — `CovariateSchema`, `CovariateEffects` types + optional bundle fields.
- `dashboard/src/lib/bundle.ts` (modify) — fetch the two STM JSONs (tolerate 404).
- `dashboard/src/lib/covariate.ts` (new) — `evalRecipe`, `buildDesignVector`, `covariatePrevalence` (pure).
- `dashboard/src/lib/store.ts` (modify) — `covariateMode`, `covariateValues` stores; make `prevalenceReader` covariate-aware.
- `dashboard/src/lib/atlas/CovariatePanel.svelte` (new) + Atlas wiring.

---

### Task 1: Pure schema builder (backend)

**Files:**
- Create: `charmpheno/charmpheno/export/covariate_schema.py`
- Test: `charmpheno/tests/test_covariate_schema.py`

**Interfaces:**
- Produces: `build_covariate_schema(*, covariate_names: list[str], continuous_cols: list[str], categorical_levels: dict[str, dict], level_counts: dict[str, int], continuous_stats: dict[str, tuple[float, float, float]], k: int) -> dict`.
  - `categorical_levels[var] = {"levels": [...all levels...], "reference": "<lvl>"}`.
  - `level_counts[design_col_name] = patient_count` for each `C(var)[T.level]` dummy; reference count = `N` is NOT needed (suppression of the reference is not applied — reference is always selectable).
  - `continuous_stats[var] = (p5, p50, p95)` already coarsened/rounded by the caller.
  - Returns the schema dict per the spec (`k`, `controls`, `design_columns`, `unsupported`).

- [ ] **Step 1: Write the failing test**

Create `charmpheno/tests/test_covariate_schema.py`:

```python
from charmpheno.export.covariate_schema import build_covariate_schema


def _base_inputs():
    return dict(
        covariate_names=[
            "Intercept",
            "C(source_cohort)[T.dementia]",
            "C(sex)[T.M]",
            "age",
        ],
        continuous_cols=["age"],
        categorical_levels={
            "source_cohort": {"levels": ["cancer", "dementia"], "reference": "cancer"},
            "sex": {"levels": ["F", "M"], "reference": "F"},
        },
        level_counts={
            "C(source_cohort)[T.dementia]": 5000,
            "C(sex)[T.M]": 4000,
        },
        continuous_stats={"age": (40.0, 65.0, 90.0)},
        k=20,
    )


def test_builds_controls_and_recipes_aligned_to_gamma():
    s = build_covariate_schema(**_base_inputs())
    # design_columns aligned with covariate_names order
    assert [d["name"] for d in s["design_columns"]] == _base_inputs()["covariate_names"]
    kinds = [d["recipe"]["kind"] for d in s["design_columns"]]
    assert kinds == ["intercept", "dummy", "dummy", "main"]
    # controls: one per variable; continuous range/default from percentiles
    age = next(c for c in s["controls"] if c["name"] == "age")
    assert age["type"] == "continuous" and age["range"] == [40.0, 90.0] and age["default"] == 65.0
    sc = next(c for c in s["controls"] if c["name"] == "source_cohort")
    assert sc["type"] == "categorical" and sc["reference"] == "cancer"
    assert sc["levels"] == ["cancer", "dementia"]
    assert s["unsupported"] == []
    assert s["k"] == 20


def test_suppresses_categorical_level_below_k():
    inp = _base_inputs()
    inp["level_counts"]["C(sex)[T.M]"] = 3   # below k=20
    s = build_covariate_schema(**inp)
    sex = next(c for c in s["controls"] if c["name"] == "sex")
    # the under-k level M is omitted; reference F always stays
    assert "M" not in sex["levels"] and "F" in sex["levels"]


def test_unparseable_design_column_goes_to_unsupported():
    inp = _base_inputs()
    inp["covariate_names"] = inp["covariate_names"] + ["weird_basis_col_3"]
    inp["continuous_cols"] = ["age"]   # weird col is neither continuous nor C(...)
    s = build_covariate_schema(**inp)
    assert "weird_basis_col_3" in s["unsupported"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd charmpheno && ../.venv/bin/python -m pytest tests/test_covariate_schema.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'charmpheno.export.covariate_schema'`.

- [ ] **Step 3: Write minimal implementation**

Create `charmpheno/charmpheno/export/covariate_schema.py`:

```python
"""Build the scrubbed covariate schema the Atlas uses to render STM controls.

Pure: takes pre-computed level counts + continuous percentiles (the caller does
the in-enclave Spark aggregation) and the design-column names, and emits the
`covariate_schema.json` payload — controls (one per variable) + per-design-column
recipes (index-aligned with Gamma) + unsupported. Categorical levels under `k`
patients are suppressed here; continuous ranges come from percentiles (never
min/max), so nothing single-patient can be reconstructed.
"""
from __future__ import annotations

import re

_DUMMY_RE = re.compile(r"^C\((?P<var>[^)]+)\)\[T\.(?P<level>.+)\]$")


def _recipe_for(name: str, continuous_cols: list[str]):
    """Return a recipe dict for one design-column name, or None if unparseable."""
    if name == "Intercept":
        return {"kind": "intercept"}
    if name in continuous_cols:
        return {"kind": "main", "var": name}
    m = _DUMMY_RE.match(name)
    if m:
        return {"kind": "dummy", "var": m.group("var"), "level": m.group("level")}
    if ":" in name:
        parts = name.split(":")
        factors = [_recipe_for(p, continuous_cols) for p in parts]
        if all(f is not None for f in factors):
            return {"kind": "interaction", "factors": factors}
    return None


def build_covariate_schema(
    *,
    covariate_names: list[str],
    continuous_cols: list[str],
    categorical_levels: dict[str, dict],
    level_counts: dict[str, int],
    continuous_stats: dict[str, tuple[float, float, float]],
    k: int,
) -> dict:
    design_columns = []
    unsupported = []
    for name in covariate_names:
        recipe = _recipe_for(name, continuous_cols)
        if recipe is None:
            unsupported.append(name)
        else:
            design_columns.append({"name": name, "recipe": recipe})

    # Which categorical levels survive the k-anon guard. A level is kept if it
    # is the reference (always selectable) or its dummy count >= k.
    kept_levels: dict[str, set] = {}
    for name, cnt in level_counts.items():
        m = _DUMMY_RE.match(name)
        if m and cnt >= k:
            kept_levels.setdefault(m.group("var"), set()).add(m.group("level"))

    controls = []
    for var in continuous_cols:
        p5, p50, p95 = continuous_stats[var]
        controls.append({
            "name": var, "type": "continuous",
            "range": [p5, p95], "default": p50,
        })
    for var, info in categorical_levels.items():
        ref = info["reference"]
        surviving = [
            lvl for lvl in info["levels"]
            if lvl == ref or lvl in kept_levels.get(var, set())
        ]
        controls.append({
            "name": var, "type": "categorical",
            "reference": ref, "levels": surviving,
        })

    return {
        "k": k,
        "controls": controls,
        "design_columns": design_columns,
        "unsupported": unsupported,
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd charmpheno && ../.venv/bin/python -m pytest tests/test_covariate_schema.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add charmpheno/charmpheno/export/covariate_schema.py charmpheno/tests/test_covariate_schema.py
git commit -m "feat(charmpheno/export): pure covariate-schema builder for STM Atlas controls

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Wire schema export into the dashboard build

**Files:**
- Modify: `analysis/cloud/build_dashboard_cloud.py` (the STM branch, near where `_stm_corpus_prevalence` / `dashboard_adapt_stm` run)
- Test: cluster-validated; local gate is `py_compile`. The pure logic is covered by Task 1.

**Interfaces:**
- Consumes: Task 1 `build_covariate_schema`; the existing sidecar load (`try_load` → `cov_df, model_spec, covariate_names`).
- Produces: `covariate_schema.json` in the bundle out_dir when the sidecar is available.

- [ ] **Step 1: Add a helper to compute the stats + write the file**

In `analysis/cloud/build_dashboard_cloud.py`, add a module-level function that mirrors `_stm_corpus_prevalence`'s sidecar load and then aggregates the safety stats. Use `pyspark.ml.functions.vector_to_array` to project the design columns, sum the dummy columns, and `approxQuantile` the continuous columns:

```python
def _write_covariate_schema(spark, *, result, corpus, source_table, cohort_name,
                            cache_uri, out_dir, log):
    """Derive + write covariate_schema.json from the covariate sidecar.

    No-op (logs a warning) when the sidecar is unavailable, so the Atlas panel
    simply hides. All stats are in-enclave aggregates (dummy-column sums,
    coarse percentiles) — nothing single-patient leaves.
    """
    if not cache_uri:
        log.warning("STM: no --cache-uri; covariate_schema.json not written.")
        return
    try:
        import json
        import math
        from pyspark.sql import functions as F
        from pyspark.ml.functions import vector_to_array
        from _covariates_cache import compute_cache_key, try_load
        from charmpheno.export.covariate_schema import build_covariate_schema

        cov_manifest = result.metadata["covariate_manifest"]
        key = compute_cache_key(
            covariate_formula=cov_manifest["covariate_formula"],
            person_mod=corpus["person_mod"], cdr=corpus["cdr"],
            source_table=source_table, cohort=cohort_name,
        )
        cached = try_load(spark, cache_uri, key)
        if cached is None:
            log.warning("STM: covariate-cache MISS; covariate_schema.json not written.")
            return
        cov_df, model_spec, covariate_names = cached
        continuous_cols = list(cov_manifest.get("continuous_cols", []))
        k = int(corpus.get("min_patient_count", 20))

        # Project the design vector to an array column once.
        arr = cov_df.select(vector_to_array("covariates").alias("x"))
        name_idx = {n: i for i, n in enumerate(covariate_names)}

        # Dummy-column sums (= per-level patient counts) for every C(var)[T.level].
        import re as _re
        dummy_names = [n for n in covariate_names if _re.match(r"^C\(.+\)\[T\..+\]$", n)]
        if dummy_names:
            sums = arr.agg(*[
                F.sum(F.col("x")[name_idx[n]]).alias(n) for n in dummy_names
            ]).collect()[0].asDict()
            level_counts = {n: int(sums[n]) for n in dummy_names}
        else:
            level_counts = {}

        # Coarse percentiles for continuous columns (p5, p50, p95), rounded.
        continuous_stats = {}
        for var in continuous_cols:
            q = arr.approxQuantile(  # operates on array element via a temp col
                _quant_col(arr, name_idx[var]), [0.05, 0.5, 0.95], 0.01)
            continuous_stats[var] = tuple(round(v) for v in q)

        # Levels + reference from the fitted ModelSpec.
        categorical_levels = _categorical_levels_from_spec(model_spec)

        schema = build_covariate_schema(
            covariate_names=covariate_names, continuous_cols=continuous_cols,
            categorical_levels=categorical_levels, level_counts=level_counts,
            continuous_stats=continuous_stats, k=k,
        )
        (out_dir / "covariate_schema.json").write_text(json.dumps(schema, indent=2))
        log.info("STM: wrote covariate_schema.json (controls=%d, unsupported=%d)",
                 len(schema["controls"]), len(schema["unsupported"]))
        print("[driver]   covariate_schema:", json.dumps(schema, indent=2), flush=True)
    except Exception as exc:  # cosmetic-only; never fail the bundle build
        log.warning("STM: covariate_schema derivation failed (%s); skipping.", exc)
```

Add two small helpers in the same module: `_quant_col` (materializes the array element into a named column so `approxQuantile` can target it) and `_categorical_levels_from_spec` (reads each categorical's levels + reference from the formulaic `model_spec`; the reference is the level absent from the `[T.*]` columns). Implement `_categorical_levels_from_spec` by reading `model_spec.structure` factor records; if the exact attribute differs in the installed formulaic, derive levels as the union of `{reference}` and the `[T.level]` levels parsed from `covariate_names`, taking `reference` from the spec's stored categories. (This runs only on the cluster; confirm against the real `model_spec` object there.)

- [ ] **Step 2: Call it in the STM branch**

Where the STM branch already calls `dashboard_adapt_stm(...)`, add right after:

```python
                _write_covariate_schema(
                    spark, result=result, corpus=corpus,
                    source_table=source_table, cohort_name=cohort_name,
                    cache_uri=args.cache_uri, out_dir=out_dir, log=log,
                )
```

- [ ] **Step 3: Syntax check**

Run: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && .venv/bin/python -m py_compile analysis/cloud/build_dashboard_cloud.py`
Expected: compiles cleanly.

- [ ] **Step 4: Commit**

```bash
git add analysis/cloud/build_dashboard_cloud.py
git commit -m "feat(cloud/stm): write covariate_schema.json at dashboard build

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Frontend types + bundle loading

**Files:**
- Modify: `dashboard/src/lib/types.ts`, `dashboard/src/lib/bundle.ts`
- Test: `dashboard/src/lib/bundle.test.ts`

**Interfaces:**
- Produces: `CovariateSchema` / `CovariateEffects` types; `DashboardBundle` gains optional `covariateSchema?` and `covariateEffects?`; `loadBundle` fetches both, tolerating 404 (→ undefined).

- [ ] **Step 1: Write the failing test**

Add to `dashboard/src/lib/bundle.test.ts` a case where the two STM files are present and one where they 404:

```typescript
it('loads covariate schema + effects when present', async () => {
  const files: Record<string, unknown> = {
    'data/cd/model.json': { K: 2, V: 2, alpha: [0.5, 0.5], beta: [[0.6,0.4],[0.3,0.7]] },
    'data/cd/phenotypes.json': { phenotypes: [] },
    'data/cd/vocab.json': { codes: [] },
    'data/cd/corpus_stats.json': { corpus_size_docs: 10, mean_codes_per_doc: 5, k: 2, v: 2, v_full: 2 },
    'data/cd/covariate_effects.json': [{ covariate: 'Intercept', per_topic: [0.1, 0.2] }],
    'data/cd/covariate_schema.json': { k: 20, controls: [], design_columns: [], unsupported: [] },
    'data/manifest.json': { default: 'cd', cohorts: [{ id: 'cd', label: 'CD', description: '' }] },
  }
  // (reuse this file's existing fetch-mock helper to back `files`)
  const b = await loadBundle('/', 'cd')
  expect(b.covariateSchema?.k).toBe(20)
  expect(b.covariateEffects?.length).toBe(1)
})

it('leaves covariate fields undefined for non-STM bundles (404)', async () => {
  // same files MINUS the two covariate_*.json; fetch returns 404 for them
  const b = await loadBundle('/', 'cancer')
  expect(b.covariateSchema).toBeUndefined()
  expect(b.covariateEffects).toBeUndefined()
})
```

(Match the existing test's fetch-mock mechanism in `bundle.test.ts` — back the `files` map and make missing keys resolve to `{ ok: false, status: 404 }`.)

- [ ] **Step 2: Run to verify it fails**

Run: `cd dashboard && npm test -- bundle`
Expected: FAIL — `covariateSchema`/`covariateEffects` undefined (not loaded) / type errors.

- [ ] **Step 3: Implement**

In `types.ts` add:

```typescript
export type CovariateRecipe =
  | { kind: 'intercept' }
  | { kind: 'main'; var: string }
  | { kind: 'dummy'; var: string; level: string }
  | { kind: 'interaction'; factors: CovariateRecipe[] }

export interface CovariateControl {
  name: string
  type: 'continuous' | 'categorical'
  range?: [number, number]
  default?: number
  reference?: string
  levels?: string[]
}
export interface CovariateSchema {
  k: number
  controls: CovariateControl[]
  design_columns: { name: string; recipe: CovariateRecipe }[]
  unsupported: string[]
}
export type CovariateEffects = { covariate: string; per_topic: number[] }[]
```

Add to the `DashboardBundle` interface: `covariateSchema?: CovariateSchema; covariateEffects?: CovariateEffects`.

In `bundle.ts`, add a 404-tolerant fetch and include the two files:

```typescript
async function fetchJsonOptional<T>(url: string): Promise<T | undefined> {
  const r = await fetch(url)
  if (!r.ok) return undefined
  return r.json() as Promise<T>
}
```

In `loadBundle`, after the existing `Promise.all`, fetch the two optional files and attach them:

```typescript
  const [covariateSchema, covariateEffects] = await Promise.all([
    fetchJsonOptional<CovariateSchema>(`${base}data/${cohortId}/covariate_schema.json`),
    fetchJsonOptional<CovariateEffects>(`${base}data/${cohortId}/covariate_effects.json`),
  ])
  return { /* ...existing fields..., */ covariateSchema, covariateEffects }
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd dashboard && npm test -- bundle`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/lib/types.ts dashboard/src/lib/bundle.ts dashboard/src/lib/bundle.test.ts
git commit -m "feat(dashboard): load covariate schema + effects for STM bundles

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Pure recipe eval + softmax reader (frontend)

**Files:**
- Create: `dashboard/src/lib/covariate.ts`, `dashboard/src/lib/covariate.test.ts`

**Interfaces:**
- Produces: `evalRecipe(recipe, values) -> number`; `buildDesignVector(design_columns, values) -> number[]`; `covariatePrevalence(effects, x) -> number[]` (softmax over topics of `Σ_p effects[p].per_topic[k] * x[p]`).
- `values: Record<string, number | string>` — control values (continuous → number, categorical → selected level string).

- [ ] **Step 1: Write the failing test**

Create `dashboard/src/lib/covariate.test.ts`:

```typescript
import { describe, it, expect } from 'vitest'
import { evalRecipe, buildDesignVector, covariatePrevalence } from './covariate'

const values = { age: 70, sex: 'M', source_cohort: 'dementia' }

describe('evalRecipe', () => {
  it('intercept -> 1', () => expect(evalRecipe({ kind: 'intercept' }, values)).toBe(1))
  it('main -> control value', () => expect(evalRecipe({ kind: 'main', var: 'age' }, values)).toBe(70))
  it('dummy match -> 1, else 0', () => {
    expect(evalRecipe({ kind: 'dummy', var: 'sex', level: 'M' }, values)).toBe(1)
    expect(evalRecipe({ kind: 'dummy', var: 'sex', level: 'F' }, values)).toBe(0)
  })
  it('interaction -> product', () => {
    const r = { kind: 'interaction', factors: [
      { kind: 'main', var: 'age' }, { kind: 'dummy', var: 'sex', level: 'M' }] } as const
    expect(evalRecipe(r, values)).toBe(70)
  })
})

it('buildDesignVector aligns to design_columns order', () => {
  const dc = [
    { name: 'Intercept', recipe: { kind: 'intercept' } },
    { name: 'C(sex)[T.M]', recipe: { kind: 'dummy', var: 'sex', level: 'M' } },
    { name: 'age', recipe: { kind: 'main', var: 'age' } },
  ] as const
  expect(buildDesignVector(dc as any, values)).toEqual([1, 1, 70])
})

it('covariatePrevalence softmaxes Gamma^T x', () => {
  // effects rows index-aligned with x; 2 topics.
  const effects = [
    { covariate: 'Intercept', per_topic: [0, 0] },
    { covariate: 'age', per_topic: [1, 0] },   // topic 0 gets +x_age
  ]
  const x = [1, Math.log(2)]   // eta = [log2, 0] -> softmax = [2/3, 1/3]
  const p = covariatePrevalence(effects, x)
  expect(p[0]).toBeCloseTo(2 / 3, 6)
  expect(p[1]).toBeCloseTo(1 / 3, 6)
})
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd dashboard && npm test -- covariate`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

Create `dashboard/src/lib/covariate.ts`:

```typescript
import type { CovariateRecipe, CovariateEffects } from './types'

type Values = Record<string, number | string>

export function evalRecipe(recipe: CovariateRecipe, values: Values): number {
  switch (recipe.kind) {
    case 'intercept': return 1
    case 'main': return Number(values[recipe.var] ?? 0)
    case 'dummy': return values[recipe.var] === recipe.level ? 1 : 0
    case 'interaction':
      return recipe.factors.reduce((acc, f) => acc * evalRecipe(f, values), 1)
  }
}

export function buildDesignVector(
  designColumns: { name: string; recipe: CovariateRecipe }[],
  values: Values,
): number[] {
  return designColumns.map((d) => evalRecipe(d.recipe, values))
}

export function covariatePrevalence(effects: CovariateEffects, x: number[]): number[] {
  const K = effects[0]?.per_topic.length ?? 0
  const eta = new Array(K).fill(0)
  for (let p = 0; p < effects.length; p++) {
    const row = effects[p].per_topic
    for (let k = 0; k < K; k++) eta[k] += row[k] * x[p]
  }
  const m = Math.max(...eta)
  const exp = eta.map((e) => Math.exp(e - m))
  const s = exp.reduce((a, b) => a + b, 0) || 1
  return exp.map((e) => e / s)
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd dashboard && npm test -- covariate`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/lib/covariate.ts dashboard/src/lib/covariate.test.ts
git commit -m "feat(dashboard): pure recipe eval + softmax covariate prevalence

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Covariate-aware prevalence reader (stores)

**Files:**
- Modify: `dashboard/src/lib/store.ts`
- Test: `dashboard/src/lib/store.covariate.test.ts` (new)

**Interfaces:**
- Consumes: Task 4 `buildDesignVector`, `covariatePrevalence`; Task 3 `bundle.covariateSchema` / `.covariateEffects`.
- Produces: `covariateMode` (writable boolean), `covariateValues` (writable `Record<string, number|string>`); `prevalenceReader` extended so that when `covariateMode` is true and the bundle has schema+effects, it returns `(p) => prevalence[p.id]` from `softmax(Γᵀ x)`; otherwise unchanged (`fractionAboveTau`).

- [ ] **Step 1: Write the failing test**

Create `dashboard/src/lib/store.covariate.test.ts`:

```typescript
import { describe, it, expect } from 'vitest'
import { get } from 'svelte/store'
import { bundle, covariateMode, covariateValues, prevalenceReader } from './store'

it('covariate mode makes prevalenceReader use softmax(Gamma^T x)', () => {
  bundle.set({
    // minimal bundle with two phenotypes id 0,1 + covariate schema/effects
    phenotypes: { phenotypes: [{ id: 0 }, { id: 1 }] },
    covariateSchema: { k: 20, unsupported: [],
      controls: [{ name: 'age', type: 'continuous', range: [0, 100], default: 50 }],
      design_columns: [
        { name: 'Intercept', recipe: { kind: 'intercept' } },
        { name: 'age', recipe: { kind: 'main', var: 'age' } },
      ] },
    covariateEffects: [
      { covariate: 'Intercept', per_topic: [0, 0] },
      { covariate: 'age', per_topic: [1, 0] },
    ],
  } as any)
  covariateValues.set({ age: Math.log(2) })
  covariateMode.set(true)
  const reader = get(prevalenceReader)
  expect(reader({ id: 0 } as any)).toBeCloseTo(2 / 3, 6)
  expect(reader({ id: 1 } as any)).toBeCloseTo(1 / 3, 6)
})
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd dashboard && npm test -- store.covariate`
Expected: FAIL — `covariateMode`/`covariateValues` not exported, or reader ignores them.

- [ ] **Step 3: Implement**

In `store.ts` add the stores and fold them into `prevalenceReader`:

```typescript
export const covariateMode = writable(false)
export const covariateValues = writable<Record<string, number | string>>({})
```

Replace the `prevalenceReader` derivation to include the covariate path (keep the existing `fractionAboveTau` branch as the default):

```typescript
import { buildDesignVector, covariatePrevalence } from './covariate'

export const prevalenceReader = derived(
  [bundle, tauThreshold, covariateMode, covariateValues],
  ([$b, $tau, $mode, $vals]) => {
    const schema = $b?.covariateSchema
    const effects = $b?.covariateEffects
    if ($mode && schema && effects && schema.unsupported.length === 0) {
      const x = buildDesignVector(schema.design_columns, $vals)
      const prev = covariatePrevalence(effects, x)
      return (p: Phenotype) => prev[p.id] ?? 0
    }
    const edges = $b?.phenotypes.theta_histogram_bin_edges
    return (p: Phenotype) => fractionAboveTau(p, edges, $tau)
  }
)
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd dashboard && npm test -- store.covariate` then the full frontend suite `cd dashboard && npm test`
Expected: PASS, no regressions.

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/lib/store.ts dashboard/src/lib/store.covariate.test.ts
git commit -m "feat(dashboard): covariate-aware prevalenceReader (softmax on control state)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: CovariatePanel + Atlas wiring

**Files:**
- Create: `dashboard/src/lib/atlas/CovariatePanel.svelte`
- Modify: `dashboard/src/lib/tabs/Atlas.svelte`
- Test: `dashboard/src/lib/atlas/CovariatePanel.test.ts` (new; tests the no-Svelte helper) + manual/visual

**Interfaces:**
- Consumes: `covariateMode`, `covariateValues` (Task 5); `bundle.covariateSchema`.
- Produces: a panel rendered in the Atlas only when `covariateSchema` is present; controls per `schema.controls`; suppressed levels already absent; if `schema.unsupported.length > 0` the panel renders read-only with an "unavailable for this formula" note and does not enable covariate mode.

- [ ] **Step 1: Write the failing test (pure helper)**

To keep logic testable outside Svelte, put the "initial control values + can-interact" logic in a helper. Create `dashboard/src/lib/atlas/covariate-panel.ts`:

Test `dashboard/src/lib/atlas/CovariatePanel.test.ts`:

```typescript
import { describe, it, expect } from 'vitest'
import { initialValues, canInteract } from './covariate-panel'

const schema = { k: 20, unsupported: [] as string[],
  controls: [
    { name: 'age', type: 'continuous', range: [40, 90], default: 65 },
    { name: 'sex', type: 'categorical', reference: 'F', levels: ['F', 'M'] },
  ],
  design_columns: [] }

it('initialValues uses continuous default + categorical reference', () => {
  expect(initialValues(schema as any)).toEqual({ age: 65, sex: 'F' })
})
it('canInteract is false when unsupported non-empty', () => {
  expect(canInteract(schema as any)).toBe(true)
  expect(canInteract({ ...schema, unsupported: ['x'] } as any)).toBe(false)
})
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd dashboard && npm test -- CovariatePanel`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement the helper, then the component**

Create `dashboard/src/lib/atlas/covariate-panel.ts`:

```typescript
import type { CovariateSchema } from '../types'

export function initialValues(schema: CovariateSchema): Record<string, number | string> {
  const v: Record<string, number | string> = {}
  for (const c of schema.controls) {
    if (c.type === 'continuous') v[c.name] = c.default ?? 0
    else v[c.name] = c.reference ?? (c.levels?.[0] ?? '')
  }
  return v
}

export function canInteract(schema: CovariateSchema): boolean {
  return schema.unsupported.length === 0
}
```

Create `dashboard/src/lib/atlas/CovariatePanel.svelte` following the project's existing component idioms (see a sibling like `CodePanel.svelte`): subscribe to `bundle`; on mount set `covariateValues = initialValues(schema)`; render one control per `schema.controls` entry — `continuous` → `<input type="range" min={range[0]} max={range[1]} bind:value>`, categorical 2-level → toggle, n-level → `<select>`; bind each to the matching key in `covariateValues` (writing back via `covariateValues.update`). A header toggle binds `covariateMode`. When `!canInteract(schema)`, render the controls disabled with a note `copy`-style string "Covariate controls unavailable for this formula" and force `covariateMode = false`. A "Reset to corpus average" button sets `covariateMode = false`.

In `tabs/Atlas.svelte`, import `CovariatePanel` and render it (near the existing prevalence legend / `TopicMap`) only when `$bundle?.covariateSchema` is defined.

- [ ] **Step 4: Run tests + build**

Run: `cd dashboard && npm test` (helper tests pass, no regressions) and `cd dashboard && npm run build` (component compiles).
Expected: tests PASS; build succeeds.

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/lib/atlas/CovariatePanel.svelte dashboard/src/lib/atlas/covariate-panel.ts dashboard/src/lib/atlas/CovariatePanel.test.ts dashboard/src/lib/tabs/Atlas.svelte
git commit -m "feat(dashboard/atlas): covariate control panel wired to live bubble resize

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Final Verification (after all tasks)

```bash
cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno
( cd charmpheno && ../.venv/bin/python -m pytest -q -m "" 2>&1 | tail -3 )
( cd dashboard && npm test 2>&1 | tail -5 )
( cd dashboard && npm run build 2>&1 | tail -3 )
.venv/bin/python -m py_compile analysis/cloud/build_dashboard_cloud.py && echo "driver compiles"
```
Expected: backend + frontend suites green; dashboard builds. Then on the cluster, after the STM fit finishes: re-push, `make build-dashboard-exp ID=2` writes `covariate_schema.json`, and the Atlas shows live covariate controls.

## Self-Review

- **Spec coverage:** schema contract → Task 1 (shape) + Task 3 (types); in-enclave derivation + safety (k-suppression, percentile ranges, dummy-sum counts) → Tasks 1–2; client recipe eval + softmax → Task 4; covariate-aware reader (τ ignored in covariate mode) → Task 5; panel + suppressed-levels-absent + unsupported-disables + non-STM-hidden → Task 6; "no re-fit / regenerate is enough" → Task 2 (build-time derivation). Covered.
- **Placeholder scan:** none; all code shown. The two backend helpers `_quant_col` / `_categorical_levels_from_spec` are described with a concrete fallback because they introspect the installed formulaic `model_spec` (cluster-only object); the pure, testable logic is fully specified in Task 1.
- **Type consistency:** `CovariateRecipe` / `CovariateControl` / `CovariateSchema` / `CovariateEffects` defined in Task 3 are used consistently in Tasks 4–6; `design_columns` order = Γ order is asserted in Tasks 1 and 4; recipe kinds match between Task 1 (Python) and Task 4 (TS).
