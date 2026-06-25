# Conditioning Context — Phase 1 (shared bar + Atlas + axis decoupling) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Atlas-local covariate panel with a shared, schema-driven conditioning bar below the tabs, and decouple group gating from prevalence covariates so all four model quadrants (plain LDA / STM / gated LDA / gated STM) render correctly on the Phenotype Atlas.

**Architecture:** A single front-end `conditioning` store (`covariateActive`, `values`, `group`) is the shared context. The Phenotype Atlas `prevalenceReader` composes two independent axes — covariate base (corpus `fractionAboveTau` vs covariate-predicted softmax) then a group mask — across the four quadrants. The controls move from the stacked `CovariatePanel` into a new global `ConditioningBar` mounted between the tab nav and tab content. One small server-side addition emits k-anon-safe categorical `proportions` for a population readout.

**Tech Stack:** TypeScript, Svelte, Vite, vitest (front-end); Python, pytest (export).

## Global Constraints

- Two orthogonal axes: prevalence covariates (`covariate_schema` + `covariate_effects`, STM only) and group gating (`gating.json`). They are handled independently; group masking must work with zero covariates and must NOT require `covariate_effects`.
- Non-covariate base prevalence is the EXISTING `fractionAboveTau(p, edges, tau)` reader (falls back to `corpus_prevalence` with no histogram). This design does not change today's default display.
- For a gated bundle the group mask is ALWAYS applied for the current `group` (`null` = "Background only" = background topics only; `g` = background ∪ `g`'s foreground). Not conditional on covariate mode.
- Mask-before-softmax on the covariate path is unchanged (`covariatePrevalenceGated`). The non-covariate masked path zeros hidden foreground with NO renormalization (matches the k-anon non-renormalization rule).
- Schema-driven: a model with different covariates, none, or no gating shows only the controls it supports. Bar hidden when a bundle has neither axis.
- `proportions` are k-anon-safe (only levels surviving the existing small-cell filter) and normalized to sum to 1 over surviving levels.
- No LaTeX in prose/comments (Unicode Greek OK); no emojis in committed files. Commit messages end with: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Front-end tests/build: `cd dashboard && npm run test` and `npm run build`. Python export test: `poetry run python -m pytest charmpheno/tests/ -k covariate_schema -v` from the repo root.

---

## File Structure

- `charmpheno/charmpheno/export/covariate_schema.py` — `build_covariate_schema` gains `n_total` and emits `proportions` per categorical control.
- `analysis/local/build_dashboard.py`, `analysis/cloud/build_dashboard_cloud.py` — pass `n_total` (design-matrix row count) into `build_covariate_schema`.
- `dashboard/src/lib/types.ts` — `proportions?` on `CovariateControl`.
- `dashboard/src/lib/covariate.ts` — add `maskGroupPrevalence`.
- `dashboard/src/lib/conditioning/population.ts` (new) — readout-string helpers.
- `dashboard/src/lib/store.ts` — `conditioning` store; rewrite `prevalenceReader` (4 quadrants).
- `dashboard/src/lib/atlas/TopicMap.svelte` — domain anchoring + render trigger read `conditioning`.
- `dashboard/src/lib/tabs/Atlas.svelte` — reset `conditioning` on bundle change; unmount `CovariatePanel`.
- `dashboard/src/lib/atlas/CovariatePanel.svelte` — repoint writes to `conditioning` (Task 4), then its widgets are reused by the bar (Task 5).
- `dashboard/src/lib/conditioning/ConditioningBar.svelte` (new) — the bar.
- `dashboard/src/App.svelte` — mount `ConditioningBar` between `Tabs` and tab content.

---

## Task 1: Categorical `proportions` in the covariate-schema export

**Files:**
- Modify: `charmpheno/charmpheno/export/covariate_schema.py` (`build_covariate_schema`)
- Modify: `analysis/local/build_dashboard.py` (its `build_covariate_schema(...)` call), `analysis/cloud/build_dashboard_cloud.py` (its call)
- Test: `charmpheno/tests/test_covariate_schema.py` (add; create if absent)

**Interfaces:**
- Consumes: existing `build_covariate_schema(*, covariate_names, continuous_cols, categorical_levels, level_counts, continuous_stats, k)`.
- Produces: `build_covariate_schema(..., k, n_total: int)` — each categorical control gains `"proportions": {level: float}` over surviving levels, summing to 1. The reference level's count is `n_total - sum(all non-reference dummy counts for that var)`.

- [ ] **Step 1: Write the failing test**

```python
# charmpheno/tests/test_covariate_schema.py
from charmpheno.export.covariate_schema import build_covariate_schema


def test_categorical_control_has_kanon_safe_proportions():
    schema = build_covariate_schema(
        covariate_names=["Intercept", "C(sex)[T.M]", "age"],
        continuous_cols=["age"],
        categorical_levels={"sex": {"levels": ["F", "M"], "reference": "F"}},
        level_counts={"C(sex)[T.M]": 48},   # 48 of 100 are M; reference F = 52
        continuous_stats={"age": (41.0, 55.0, 68.0)},
        k=20,
        n_total=100,
    )
    sex = next(c for c in schema["controls"] if c["name"] == "sex")
    assert sex["proportions"] == {"F": 0.52, "M": 0.48}
    assert abs(sum(sex["proportions"].values()) - 1.0) < 1e-9


def test_subk_level_is_dropped_from_proportions():
    # M has 8 (< k=20) so it is suppressed from levels AND proportions;
    # the surviving distribution renormalizes over kept levels (just F here).
    schema = build_covariate_schema(
        covariate_names=["Intercept", "C(sex)[T.M]", "age"],
        continuous_cols=["age"],
        categorical_levels={"sex": {"levels": ["F", "M"], "reference": "F"}},
        level_counts={"C(sex)[T.M]": 8},
        continuous_stats={"age": (41.0, 55.0, 68.0)},
        k=20,
        n_total=100,
    )
    sex = next(c for c in schema["controls"] if c["name"] == "sex")
    assert sex["levels"] == ["F"]
    assert sex["proportions"] == {"F": 1.0}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && poetry run python -m pytest charmpheno/tests/test_covariate_schema.py -v`
Expected: FAIL — `build_covariate_schema()` got an unexpected keyword argument `n_total` (and no `proportions` key).

- [ ] **Step 3: Implement**

In `charmpheno/charmpheno/export/covariate_schema.py`, add `n_total: int` to the keyword-only signature (after `k`). Replace the categorical-control loop so it emits `proportions`:

```python
    for var, info in categorical_levels.items():
        ref = info["reference"]
        surviving = [
            lvl for lvl in info["levels"]
            if lvl == ref or lvl in kept_levels.get(var, set())
        ]
        # Per-level counts. Non-reference levels come from the dummy sums;
        # the reference level has no dummy, so its count is n_total minus all
        # non-reference counts for this variable (every patient has exactly
        # one level).
        nonref_counts = {}
        for name, cnt in level_counts.items():
            m = _DUMMY_RE.match(name)
            if m and m.group("var") == var:
                nonref_counts[m.group("level")] = int(cnt)
        ref_count = int(n_total) - sum(nonref_counts.values())
        kept_counts = {
            lvl: (ref_count if lvl == ref else nonref_counts.get(lvl, 0))
            for lvl in surviving
        }
        total = sum(kept_counts.values()) or 1
        proportions = {lvl: kept_counts[lvl] / total for lvl in surviving}
        controls.append({
            "name": var, "type": "categorical",
            "reference": ref, "levels": surviving,
            "proportions": proportions,
        })
```

Then update the two callers to pass `n_total` (the design-matrix row count, i.e. the number of covariate rows):
- `analysis/local/build_dashboard.py` — in `_write_local_covariate_schema`, the local covariate matrix `X` is in scope; pass `n_total=int(X.shape[0])`.
- `analysis/cloud/build_dashboard_cloud.py` — in `_write_covariate_schema`, pass `n_total=int(cov_df.count())` (the same `cov_df` already loaded there; `.count()` once, before any per-column work).

- [ ] **Step 4: Run to verify it passes**

Run: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && poetry run python -m pytest charmpheno/tests/test_covariate_schema.py -v`
Expected: PASS (2). Then compile the cloud caller: `cd analysis/cloud && python -m py_compile build_dashboard_cloud.py` — Expected: clean.

- [ ] **Step 5: Run the local gated integration test (proportions flow end-to-end)**

Run: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && poetry run python -m pytest tests/scripts/test_build_dashboard_gated.py -v`
Expected: PASS (the chain still builds; ~15s).

- [ ] **Step 6: Commit**

```bash
git add charmpheno/charmpheno/export/covariate_schema.py analysis/local/build_dashboard.py analysis/cloud/build_dashboard_cloud.py charmpheno/tests/test_covariate_schema.py
git commit -m "$(printf 'feat(export): k-anon-safe categorical proportions in covariate_schema\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 2: `maskGroupPrevalence` (non-covariate gating mask)

**Files:**
- Modify: `dashboard/src/lib/covariate.ts`
- Test: `dashboard/src/lib/covariate.test.ts`

**Interfaces:**
- Consumes: existing `allowedMaskForGroup(topicBlocks: string[], selectedGroup: string | null): boolean[]`.
- Produces: `maskGroupPrevalence(values: number[], topicBlocks: string[], group: string | null): number[]` — returns a copy of `values` with hidden-foreground topics set to 0 (background and the selected group preserved); no renormalization; needs no covariate effects.

- [ ] **Step 1: Write the failing test**

```typescript
// dashboard/src/lib/covariate.test.ts (add)
import { maskGroupPrevalence } from './covariate'

describe('maskGroupPrevalence', () => {
  it('zeros out-of-group foreground, preserves background + selected group, no renormalization', () => {
    const values = [0.3, 0.3, 0.2, 0.2]              // per-topic base values
    const blocks = ['background', 'background', 'rare_dx', 'other']
    expect(maskGroupPrevalence(values, blocks, 'rare_dx')).toEqual([0.3, 0.3, 0.2, 0])
    // background-only: all foreground hidden
    expect(maskGroupPrevalence(values, blocks, null)).toEqual([0.3, 0.3, 0, 0])
  })
})
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd dashboard && npm run test -- covariate`
Expected: FAIL — `maskGroupPrevalence` is not exported.

- [ ] **Step 3: Implement**

In `dashboard/src/lib/covariate.ts`, add (reusing `allowedMaskForGroup`):

```typescript
export function maskGroupPrevalence(
  values: number[], topicBlocks: string[], group: string | null,
): number[] {
  const mask = allowedMaskForGroup(topicBlocks, group)
  return values.map((v, k) => (mask[k] ? v : 0))
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd dashboard && npm run test -- covariate`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/lib/covariate.ts dashboard/src/lib/covariate.test.ts
git commit -m "$(printf 'feat(dashboard-fe): maskGroupPrevalence for the gating-only quadrant\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 3: `proportions?` type + population-readout helpers

**Files:**
- Modify: `dashboard/src/lib/types.ts` (`CovariateControl`)
- Create: `dashboard/src/lib/conditioning/population.ts`
- Test: `dashboard/src/lib/conditioning/population.test.ts`

**Interfaces:**
- Consumes: `CovariateControl` (gains `proportions?: Record<string, number>`).
- Produces: `populationLines(schema: CovariateSchema | undefined): { name: string; summary: string }[]` — one line per control: continuous → `"41-68 (med 55)"`; categorical → `"F 52% / M 48%"` (omits the percentages when `proportions` is absent, falling back to the level list `"F / M"`).

- [ ] **Step 1: Write the failing test**

```typescript
// dashboard/src/lib/conditioning/population.test.ts
import { describe, it, expect } from 'vitest'
import { populationLines } from './population'

describe('populationLines', () => {
  it('summarizes continuous range+median and categorical proportions', () => {
    const schema = {
      k: 20, design_columns: [], unsupported: [],
      controls: [
        { name: 'age', type: 'continuous', range: [41, 68], default: 55 },
        { name: 'sex', type: 'categorical', reference: 'F', levels: ['F', 'M'],
          proportions: { F: 0.52, M: 0.48 } },
      ],
    } as any
    expect(populationLines(schema)).toEqual([
      { name: 'age', summary: '41-68 (med 55)' },
      { name: 'sex', summary: 'F 52% / M 48%' },
    ])
  })

  it('falls back to the level list when proportions are absent', () => {
    const schema = {
      k: 20, design_columns: [], unsupported: [],
      controls: [{ name: 'sex', type: 'categorical', reference: 'F', levels: ['F', 'M'] }],
    } as any
    expect(populationLines(schema)).toEqual([{ name: 'sex', summary: 'F / M' }])
  })

  it('returns [] for an undefined schema', () => {
    expect(populationLines(undefined)).toEqual([])
  })
})
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd dashboard && npm run test -- population`
Expected: FAIL — module `./population` not found.

- [ ] **Step 3: Implement**

In `dashboard/src/lib/types.ts`, add `proportions?: Record<string, number>` to the `CovariateControl` interface (after `levels?`).

Create `dashboard/src/lib/conditioning/population.ts`:

```typescript
import type { CovariateSchema } from '../types'

function pct(x: number): string {
  return `${Math.round(x * 100)}%`
}

export function populationLines(
  schema: CovariateSchema | undefined,
): { name: string; summary: string }[] {
  if (!schema) return []
  return schema.controls.map((c) => {
    if (c.type === 'continuous') {
      const [lo, hi] = c.range ?? [0, 0]
      return { name: c.name, summary: `${lo}-${hi} (med ${c.default ?? ''})` }
    }
    const levels = c.levels ?? []
    const summary = c.proportions
      ? levels.map((l) => `${l} ${pct(c.proportions![l] ?? 0)}`).join(' / ')
      : levels.join(' / ')
    return { name: c.name, summary }
  })
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd dashboard && npm run test -- population`
Expected: PASS (3).

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/lib/types.ts dashboard/src/lib/conditioning/population.ts dashboard/src/lib/conditioning/population.test.ts
git commit -m "$(printf 'feat(dashboard-fe): proportions type + population-readout helpers\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 4: `conditioning` store + four-quadrant `prevalenceReader`

**Files:**
- Modify: `dashboard/src/lib/store.ts`
- Modify: `dashboard/src/lib/atlas/TopicMap.svelte`, `dashboard/src/lib/tabs/Atlas.svelte`, `dashboard/src/lib/atlas/CovariatePanel.svelte`
- Test: `dashboard/src/lib/store.covariate.test.ts` (rewrite to the new store + add quadrant cases)

**Interfaces:**
- Consumes: `maskGroupPrevalence` (Task 2), existing `covariatePrevalence` / `covariatePrevalenceGated` / `allowedMaskForGroup` / `buildDesignVector` / `fractionAboveTau`.
- Produces:
  - `conditioning` writable: `{ covariateActive: boolean; values: Record<string, number|string>; group: string | null }` (initial `{ covariateActive: false, values: {}, group: null }`).
  - `prevalenceReader` derived on `[bundle, tauThreshold, conditioning]` returning `(p: Phenotype) => number` per the quadrant logic below.
  - The old `covariateMode` / `covariateValues` / `selectedGroup` exports are removed.

- [ ] **Step 1: Rewrite the store test (failing)**

Replace `dashboard/src/lib/store.covariate.test.ts` so it drives the new store. Keep the existing covariate-mode assertion and add the gating-only and plain quadrants:

```typescript
import { it, expect, beforeEach } from 'vitest'
import { get } from 'svelte/store'
import { bundle, conditioning, prevalenceReader } from './store'

beforeEach(() => {
  bundle.set(null)
  conditioning.set({ covariateActive: false, values: {}, group: null })
})

const COV_BUNDLE = {
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
}

it('covariateActive makes prevalenceReader use softmax(Gamma^T x)', () => {
  bundle.set(COV_BUNDLE as any)
  conditioning.set({ covariateActive: true, values: { age: Math.log(2) }, group: null })
  const reader = get(prevalenceReader)
  expect(reader({ id: 0 } as any)).toBeCloseTo(2 / 3, 6)
  expect(reader({ id: 1 } as any)).toBeCloseTo(1 / 3, 6)
})

it('gating-only quadrant masks the non-covariate base without covariate_effects', () => {
  // No covariateSchema / covariateEffects; gating present. corpus_prevalence is
  // the base (no theta histogram), masked by group.
  bundle.set({
    phenotypes: { phenotypes: [
      { id: 0, corpus_prevalence: 0.5 },
      { id: 1, corpus_prevalence: 0.3 },
    ] },
    gating: { group_var: 'g', groups: ['rare_dx'],
      topic_blocks: ['background', 'rare_dx'] },
  } as any)
  // Background only (group null): foreground topic 1 hidden.
  let reader = get(prevalenceReader)
  expect(reader({ id: 0, corpus_prevalence: 0.5 } as any)).toBeCloseTo(0.5, 6)
  expect(reader({ id: 1, corpus_prevalence: 0.3 } as any)).toBe(0)
  // Select rare_dx: foreground topic 1 revealed at its base value.
  conditioning.set({ covariateActive: false, values: {}, group: 'rare_dx' })
  reader = get(prevalenceReader)
  expect(reader({ id: 1, corpus_prevalence: 0.3 } as any)).toBeCloseTo(0.3, 6)
})

it('plain bundle uses the unchanged fractionAboveTau base', () => {
  bundle.set({
    phenotypes: { phenotypes: [{ id: 0, corpus_prevalence: 0.42 }] },
  } as any)
  const reader = get(prevalenceReader)
  expect(reader({ id: 0, corpus_prevalence: 0.42 } as any)).toBeCloseTo(0.42, 6)
})
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd dashboard && npm run test -- store.covariate`
Expected: FAIL — `conditioning` is not exported; reader signature mismatch.

- [ ] **Step 3: Implement the store**

In `dashboard/src/lib/store.ts`:

1. Remove the three lines exporting `covariateMode`, `covariateValues`, `selectedGroup`.
2. Add the shared store:

```typescript
export interface Conditioning {
  covariateActive: boolean
  values: Record<string, number | string>
  group: string | null
}
export const conditioning = writable<Conditioning>({
  covariateActive: false, values: {}, group: null,
})
```

3. Update the import at the top of the file to include `maskGroupPrevalence`:

```typescript
import { buildDesignVector, covariatePrevalence, allowedMaskForGroup, covariatePrevalenceGated, maskGroupPrevalence } from './covariate'
```

4. Replace `prevalenceReader` with the four-quadrant version:

```typescript
export const prevalenceReader = derived(
  [bundle, tauThreshold, conditioning],
  ([$b, $tau, $cond]) => {
    const schema = $b?.covariateSchema
    const effects = $b?.covariateEffects
    const gating = $b?.gating
    const edges = $b?.phenotypes.theta_histogram_bin_edges

    // Covariate axis: when active and renderable, base = softmax(Gamma^T x).
    const covariateOn =
      $cond.covariateActive && !!schema && !!effects && schema.unsupported.length === 0
    if (covariateOn) {
      const x = buildDesignVector(schema!.design_columns, $cond.values)
      if (gating) {
        const mask = allowedMaskForGroup(gating.topic_blocks, $cond.group)
        const prev = covariatePrevalenceGated(effects!, x, mask)
        return (p: Phenotype) => prev[p.id] ?? 0
      }
      const prev = covariatePrevalence(effects!, x)
      return (p: Phenotype) => prev[p.id] ?? 0
    }

    // Non-covariate base = fractionAboveTau; mask by group when gated.
    if (gating) {
      // Build the per-topic base indexed by topic id (= displayed index, the
      // same key topic_blocks uses), then mask hidden foreground to 0.
      const base: number[] = []
      for (const p of $b!.phenotypes.phenotypes) base[p.id] = fractionAboveTau(p, edges, $tau)
      const masked = maskGroupPrevalence(base, gating.topic_blocks, $cond.group)
      return (p: Phenotype) => masked[p.id] ?? 0
    }
    return (p: Phenotype) => fractionAboveTau(p, edges, $tau)
  }
)
```

(`maskGroupPrevalence` indexes by topic id, aligned to `gating.topic_blocks` exactly as the covariate path's `prev[p.id]` does.)

5. Repoint `dashboard/src/lib/atlas/CovariatePanel.svelte` to the new store so it keeps working until it is replaced in Task 5. Change its store import and its reads/writes:
   - import `conditioning` instead of `covariateMode, covariateValues, selectedGroup`.
   - covariate-mode toggle: `bind:checked` cannot bind a nested field; replace the toggle `<input>` binding with `checked={$conditioning.covariateActive}` plus `on:change={(e) => conditioning.update((c) => ({ ...c, covariateActive: e.currentTarget.checked }))}`.
   - the local `covariateValues.set(local)` reactive write becomes `conditioning.update((c) => ({ ...c, values: local }))`.
   - the group `<select>` `bind:value={$selectedGroup}` becomes `value={$conditioning.group}` + `on:change={(e) => conditioning.update((c) => ({ ...c, group: e.currentTarget.value === '' ? null : e.currentTarget.value }))}` (map the "Background only" option to `value=""`).
   - `reset()` sets `conditioning.update((c) => ({ ...c, covariateActive: false }))` and reseeds `local`.

6. Update `dashboard/src/lib/atlas/TopicMap.svelte`:
   - import `conditioning` instead of `covariateMode`.
   - domain anchoring keyed on any conditioning active:
     ```typescript
     const conditioningActive = $conditioning.covariateActive || !!$bundle?.gating
     const domainMax = conditioningActive
       ? Math.max(...allPhenotypes.map((p) => p.corpus_prevalence), 1e-9)
       : Math.max(...allPhenotypes.map(r_of), 1e-9)
     ```
   - render trigger: replace `$covariateMode` with `$conditioning` in the reactive `$:` statement dependency list.

7. Update `dashboard/src/lib/tabs/Atlas.svelte`:
   - import `conditioning` instead of `selectedGroup`.
   - reset on bundle change: `$: { $bundle; conditioning.set({ covariateActive: false, values: {}, group: null }) }`.

- [ ] **Step 4: Run to verify it passes**

Run: `cd dashboard && npm run test -- store.covariate`
Expected: PASS (4). Then the full suite + build: `cd dashboard && npm run test && npm run build` — Expected: PASS, build clean.

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/lib/store.ts dashboard/src/lib/atlas/TopicMap.svelte dashboard/src/lib/tabs/Atlas.svelte dashboard/src/lib/atlas/CovariatePanel.svelte dashboard/src/lib/store.covariate.test.ts
git commit -m "$(printf 'feat(dashboard-fe): shared conditioning store + four-quadrant prevalence reader\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 5: ConditioningBar + relocation below the tabs

**Files:**
- Create: `dashboard/src/lib/conditioning/ConditioningBar.svelte`
- Modify: `dashboard/src/App.svelte` (mount the bar), `dashboard/src/lib/tabs/Atlas.svelte` (remove the stacked `CovariatePanel`)
- Test: `cd dashboard && npm run build` + `npm run test` (no new unit test; behavior is exercised by Task 4's reader tests). Manual visual check.

**Interfaces:**
- Consumes: `conditioning` store (Task 4), `populationLines` (Task 3), `bundle` store, the control-widget markup from `CovariatePanel.svelte` (sliders, 2-level toggle, n-level select, group select).
- Produces: a full-width bar with an independent group section (iff `bundle.gating`) and covariate section (iff `bundle.covariateSchema` with `unsupported.length === 0`), plus the population readout. Hidden entirely when the bundle has neither axis.

- [ ] **Step 1: Build the bar**

Create `dashboard/src/lib/conditioning/ConditioningBar.svelte`. It reuses the exact control widgets from `dashboard/src/lib/atlas/CovariatePanel.svelte` (read that file first and lift the `<script>` seeding logic — `initialValues`, `canInteract` from `../atlas/covariate-panel` — and the controls markup). Structure:

```svelte
<script lang="ts">
  import { conditioning, bundle } from '../store'
  import { populationLines } from './population'
  import { initialValues, canInteract } from '../atlas/covariate-panel'

  $: schema = $bundle?.covariateSchema
  $: gating = $bundle?.gating
  $: hasCovariates = !!schema && canInteract(schema)
  $: hasGroup = !!gating
  $: visible = hasCovariates || hasGroup

  // Seed covariate values whenever the schema changes.
  let local: Record<string, number | string> = {}
  $: if (schema) { local = initialValues(schema); conditioning.update((c) => ({ ...c, values: local })) }
  $: conditioning.update((c) => ({ ...c, values: local }))

  $: lines = populationLines(schema)
</script>

{#if visible}
  <div class="conditioning-bar">
    {#if hasGroup}
      <div class="group-section">
        <span class="label">{gating.group_var}</span>
        <select
          value={$conditioning.group ?? ''}
          on:change={(e) => conditioning.update((c) => ({ ...c, group: e.currentTarget.value === '' ? null : e.currentTarget.value }))}
        >
          <option value="">Background only</option>
          {#each gating.groups as g}<option value={g}>{g}</option>{/each}
        </select>
      </div>
    {/if}
    {#if hasCovariates}
      <div class="covariate-section">
        <label class="toggle">
          <input type="checkbox"
            checked={$conditioning.covariateActive}
            on:change={(e) => conditioning.update((c) => ({ ...c, covariateActive: e.currentTarget.checked }))} />
          {$conditioning.covariateActive ? 'covariate prevalence' : 'corpus average'}
        </label>
        <!-- Lift the per-control widgets (slider / cat-toggle / cat-select)
             from CovariatePanel.svelte, binding to local[control.name]. -->
        {#each schema.controls as control (control.name)}
          <!-- ...same markup as CovariatePanel's controls loop... -->
        {/each}
        <div class="population-readout">
          {#each lines as l}<span class="pop-line">{l.name}: {l.summary}</span>{/each}
        </div>
      </div>
    {/if}
  </div>
{/if}

<style>
  .conditioning-bar {
    display: flex; align-items: center; gap: 1.5rem; flex-wrap: wrap;
    padding: 0.5rem 0; border-bottom: 1px solid var(--rule);
  }
  /* reuse the control styles from CovariatePanel as needed */
</style>
```

(Reuse the existing widget markup and class names from `CovariatePanel.svelte` rather than inventing new ones; keep the slim/non-intrusive look. The "slim chip when off" requirement is satisfied by the bar collapsing to just the group selector and/or the covariate toggle when their controls are not expanded — keep the covariate sliders visible only when `$conditioning.covariateActive`.)

- [ ] **Step 2: Mount the bar in App, remove the stacked panel**

In `dashboard/src/App.svelte`, mount the bar between `<Tabs />` and the routed component:

```svelte
    <Tabs />
    <ConditioningBar />
    <svelte:component this={TAB_COMPONENTS[$route]} />
```

Add `import ConditioningBar from './lib/conditioning/ConditioningBar.svelte'` with the other imports.

In `dashboard/src/lib/tabs/Atlas.svelte`, remove the `CovariatePanel` import and its `{#if $bundle?.covariateSchema}<CovariatePanel ... />{/if}` block from the left column (the controls now live in the global bar). Leave the `conditioning` reset reactive statement (added in Task 4).

- [ ] **Step 3: Build + run + manual check**

Run: `cd dashboard && npm run build && npm run test`
Expected: build clean, full suite passes.

Then `cd dashboard && npm run dev` and verify against the local `gated_demo` cohort (rebuild it first if needed: `bash scripts/gated_dashboard_demo.sh`):
- The bar appears below the tabs with a group selector and a covariate toggle.
- Toggling covariate prevalence + dragging age resizes bubbles (absolute scale); switching the group reveals/vanishes the rare_dx foreground — both now driven from the bar.
- A non-gated, no-covariate bundle (e.g. `cancer`) shows no bar.

- [ ] **Step 4: Commit**

```bash
git add dashboard/src/lib/conditioning/ConditioningBar.svelte dashboard/src/App.svelte dashboard/src/lib/tabs/Atlas.svelte
git commit -m "$(printf 'feat(dashboard-fe): relocate conditioning controls to a shared bar below the tabs\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Final verification

- [ ] `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && poetry run python -m pytest charmpheno/tests/test_covariate_schema.py tests/scripts/test_build_dashboard_gated.py -v` — PASS.
- [ ] `cd dashboard && npm run test && npm run build` — PASS, build clean.
- [ ] Manual: gated_demo shows the bar; covariates + group both drive the Atlas from the bar; the four quadrants render (a plain LDA cohort shows no bar; a gating-only or covariates-only bundle shows only its section).
- [ ] Whole-branch review (opus) before finishing; then `superpowers:finishing-a-development-branch` (likely "keep as-is on stm").

## Notes for the implementer

- The non-covariate base MUST stay `fractionAboveTau` (today's reader). Do not switch it to raw `corpus_prevalence` — `fractionAboveTau` already falls back to `corpus_prevalence` when there is no histogram, and STM gated bundles have no histogram, so the masked-corpus behavior is correct either way; the point is not to change the plain/STM-no-gating display.
- `gating.json` carries only `group_var` / `groups` / `topic_blocks` — NO group proportions. The marginal sampler that needs them is Phase 2, not here.
- Cloud builder changes (Task 1) cannot be run locally (BigQuery-bound); `py_compile` + mirroring the local change is the bar, consistent with the existing cloud-parity convention.
- The `conditioning` reset on bundle change lives in `Atlas.svelte` today; if Phase 2/3 need the reset to fire app-wide, move it to `App.svelte`'s bundle subscription then.
