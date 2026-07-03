# Dashboard Reorganization + Faithful STM Record-Completion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the dashboard into two top-level tabs (Phenotype Atlas, Simulator) each with two subtabs, add a Compare/Difference pane, and make STM record-completion a faithful logistic-normal posterior over the observed prefix (unifying the Simulator with the Patient Atlas).

**Architecture:** Front-end only (Svelte 5 + TypeScript). A two-level hash router drives a reusable `SubTabs` control; the existing tab bodies become subtab bodies. A pure `difference.ts` ranks conditions by relevance-delta between a clicked heatmap pair. A pure `recordPosterior.ts` finds the mode of p(η | prefix) under the covariate/group-conditioned logistic-normal prior and draws from a Laplace Gaussian around it; empty prefix reduces exactly to the existing prior draw, so cohort generation and record completion are one path. A positive-definite guard fixes the currently-throwing group Σ sub-block.

**Tech Stack:** Svelte 5, TypeScript, Vitest + @testing-library/svelte, hand-rolled numerics (no new npm deps).

## Global Constraints

- spark-vi stays domain-agnostic — **dashboard only**; no spark-vi change.
- No new export fields — bundles already carry `reference_topic`, `group_proportions`, `group_labels`, `covariateEffects`, `correlation`.
- No LaTeX in prose/comments/docstrings: Unicode Greek (Σ η θ μ Γ λ φ) + plain text; write `E(β)` not `E[β]`.
- Cite any literature-derived method/default in docstrings (relevance: Sievert & Shirley 2014; logistic-normal posterior / variational: Blei & Lafferty 2007; diagonal loading is standard ridge regularization — no citation needed).
- Markdown-linkable code refs in prose (`[name](path#Lstart-Lend)`).
- Hand-rolled numerics — **no new npm dependency**.
- TDD throughout: write the failing test, watch it fail, minimal implementation, watch it pass, commit.
- Run FE commands from `dashboard/`. Test: `npm run test` (= `vitest run`). Type-check: `npx svelte-check --threshold error`. Build: `npx vite build`.
- Every git commit message ends with: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## File Structure

**New:**
- `dashboard/src/lib/SubTabs.svelte` — second-level segmented nav for the active top tab.
- `dashboard/src/lib/tabs/PhenotypeAtlas.svelte` — top container: renders SubTabs + Explore|Compare body.
- `dashboard/src/lib/tabs/SimulatorGroup.svelte` — top container: renders SubTabs + Simulate|Explore body.
- `dashboard/src/lib/atlas/Compare.svelte` — Compare subtab body (heatmap + Difference sidebar).
- `dashboard/src/lib/atlas/DifferencePane.svelte` — the Phenotype Difference sidebar.
- `dashboard/src/lib/atlas/difference.ts` (+ `.test.ts`) — relevance-delta ranking (pure).
- `dashboard/src/lib/conditioning/linalg.ts` (+ `.test.ts`) — `choleskyPD`, `invSPD`, `solveSPD`.
- `dashboard/src/lib/conditioning/recordPosterior.ts` (+ `.test.ts`) — `sampleRecordPosterior`.
- `docs/decisions/0036-dashboard-stm-record-completion-posterior.md` — ADR (extends 0035).

**Modified:**
- `dashboard/src/lib/router.ts` — two-level routing.
- `dashboard/src/lib/Tabs.svelte` — top-level tabs from the new list.
- `dashboard/src/App.svelte` — top-component map + render.
- `dashboard/src/lib/tabs/Atlas.svelte` — becomes the **Explore** body (drop the viz-switch + correlation branch; those move to Compare).
- `dashboard/src/lib/atlas/PhenotypeBrowser.svelte` — drop `Topic mass` column; prevalence re-sort.
- `dashboard/src/lib/store.ts` — drop `topic_mass` sort key; add `comparePair`.
- `dashboard/src/lib/atlas/CorrelationHeatmap.svelte` — cell click sets a pair + highlight.
- `dashboard/src/lib/conditioning/logisticNormal.ts` — use `choleskyPD` (PD guard).
- `dashboard/src/lib/cohort.ts` — `generateCohort` gains a prefix (record-completion) path.
- `dashboard/src/lib/tabs/Simulator.svelte` — Simulate Cohort writes the shared cohort; uses the posterior draw.
- `dashboard/src/lib/tabs/Patient.svelte` — Explore Cohort reads the shared cohort.
- `dashboard/src/lib/tour.ts` — update `data-tour`/tab-id references to the new ids.

---

# PART 1 — Information architecture

## Task 1: Two-level hash router

**Files:**
- Modify: `dashboard/src/lib/router.ts` (entire file)
- Test: `dashboard/src/lib/router.test.ts` (create)

**Interfaces:**
- Produces:
  - `TOP_TABS: readonly {id: TopId; label: string}[]` with ids `'atlas' | 'sim'`.
  - `SUBTABS: Record<TopId, readonly {id: string; label: string}[]>`.
  - `topRoute: Writable<TopId>`, `subRoute: Writable<string>`.
  - `go(top: TopId, sub?: string): void`.
  - `parseRoute(hash: string): {top: TopId; sub: string}` (exported, pure, testable).

- [ ] **Step 1: Write the failing test**

Create `dashboard/src/lib/router.test.ts`:

```typescript
import { describe, it, expect } from 'vitest'
import { parseRoute } from './router'

describe('parseRoute', () => {
  it('parses a two-level hash', () => {
    expect(parseRoute('#/atlas/compare')).toEqual({ top: 'atlas', sub: 'compare' })
    expect(parseRoute('#/sim/simulate')).toEqual({ top: 'sim', sub: 'simulate' })
  })
  it('defaults the subtab to the first for the top tab', () => {
    expect(parseRoute('#/atlas')).toEqual({ top: 'atlas', sub: 'explore' })
    expect(parseRoute('#/sim')).toEqual({ top: 'sim', sub: 'simulate' })
  })
  it('falls back to atlas/explore on empty or unknown top', () => {
    expect(parseRoute('')).toEqual({ top: 'atlas', sub: 'explore' })
    expect(parseRoute('#/nope/x')).toEqual({ top: 'atlas', sub: 'explore' })
  })
  it('redirects legacy single-segment hashes', () => {
    expect(parseRoute('#/patient')).toEqual({ top: 'sim', sub: 'explore' })
    expect(parseRoute('#/simulator')).toEqual({ top: 'sim', sub: 'simulate' })
  })
  it('falls back an unknown sub to the top tab default', () => {
    expect(parseRoute('#/atlas/bogus')).toEqual({ top: 'atlas', sub: 'explore' })
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm run test -- src/lib/router.test.ts`
Expected: FAIL — `parseRoute` is not exported.

- [ ] **Step 3: Rewrite `router.ts`**

Replace the entire file `dashboard/src/lib/router.ts`:

```typescript
import { writable } from 'svelte/store'

// Two-level hash routing: `#/<top>/<sub>`. TOP_TABS drives the top nav
// (Tabs.svelte); SUBTABS drives the second-level nav (SubTabs.svelte). App.svelte
// renders the active top component; each top component renders its active subtab.
export const TOP_TABS = [
  { id: 'atlas', label: 'Phenotype Atlas' },
  { id: 'sim', label: 'Simulator' },
] as const

export type TopId = (typeof TOP_TABS)[number]['id']

export const SUBTABS: Record<TopId, readonly { id: string; label: string }[]> = {
  atlas: [
    { id: 'explore', label: 'Explore' },
    { id: 'compare', label: 'Compare' },
  ],
  sim: [
    { id: 'simulate', label: 'Simulate Cohort' },
    { id: 'explore', label: 'Explore Cohort' },
  ],
}

const TOP_IDS = TOP_TABS.map((t) => t.id) as readonly string[]

// Legacy single-segment hashes from the old three-tab layout.
const LEGACY: Record<string, { top: TopId; sub: string }> = {
  atlas: { top: 'atlas', sub: 'explore' },
  patient: { top: 'sim', sub: 'explore' },
  simulator: { top: 'sim', sub: 'simulate' },
}

export function parseRoute(hash: string): { top: TopId; sub: string } {
  const path = hash.replace(/^#\/?/, '')
  const [rawTop, rawSub] = path.split('/')
  if (rawTop && !rawSub && LEGACY[rawTop]) return LEGACY[rawTop]
  const top = (TOP_IDS.includes(rawTop) ? rawTop : 'atlas') as TopId
  const subs = SUBTABS[top].map((s) => s.id)
  const sub = subs.includes(rawSub) ? rawSub : subs[0]
  return { top, sub }
}

function current() {
  return parseRoute(typeof window === 'undefined' ? '' : window.location.hash)
}

const first = current()
export const topRoute = writable<TopId>(first.top)
export const subRoute = writable<string>(first.sub)

if (typeof window !== 'undefined') {
  window.addEventListener('hashchange', () => {
    const r = current()
    topRoute.set(r.top)
    subRoute.set(r.sub)
  })
}

export function go(top: TopId, sub?: string): void {
  const subs = SUBTABS[top].map((s) => s.id)
  const s = sub && subs.includes(sub) ? sub : subs[0]
  window.location.hash = `#/${top}/${s}`
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm run test -- src/lib/router.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/lib/router.ts dashboard/src/lib/router.test.ts
git commit -m "feat(dashboard): two-level hash router (top tab + subtab)"
```

---

## Task 2: SubTabs control + top containers + App wiring

**Files:**
- Create: `dashboard/src/lib/SubTabs.svelte`, `dashboard/src/lib/tabs/PhenotypeAtlas.svelte`, `dashboard/src/lib/tabs/SimulatorGroup.svelte`
- Modify: `dashboard/src/lib/Tabs.svelte`, `dashboard/src/App.svelte`, `dashboard/src/lib/tour.ts`

**Interfaces:**
- Consumes: `TOP_TABS`, `SUBTABS`, `topRoute`, `subRoute`, `go` (Task 1).
- Produces: `PhenotypeAtlas.svelte` renders `Atlas` (Explore) when `$subRoute==='explore'`, `Compare` when `'compare'`; `SimulatorGroup.svelte` renders `Simulator` when `$subRoute==='simulate'`, `Patient` when `'explore'`. (`Compare.svelte` from Task 6 — until then, PhenotypeAtlas may render `Atlas` for both; wire `Compare` in Task 6.)

- [ ] **Step 1: Update `Tabs.svelte` to the top-level list**

In `dashboard/src/lib/Tabs.svelte`, change the import and iteration from `route, go, TABS` to the new API:

```svelte
<script lang="ts">
  import { topRoute, go, TOP_TABS } from './router'
</script>

<nav class="tabs" aria-label="Sections">
  {#each TOP_TABS as t}
    <button
      class="tab"
      class:active={$topRoute === t.id}
      on:click={() => go(t.id)}
      aria-current={$topRoute === t.id ? 'page' : undefined}
      data-tour="tab-{t.id}"
    >
      <span class="label">{t.label}</span>
    </button>
  {/each}
</nav>
```

Leave the `<style>` block unchanged.

- [ ] **Step 2: Create `SubTabs.svelte`**

Create `dashboard/src/lib/SubTabs.svelte` (segmented control lifted from Atlas's `.viz-switch`):

```svelte
<script lang="ts">
  import { subRoute, go, SUBTABS, type TopId } from './router'
  export let top: TopId
  $: subs = SUBTABS[top]
</script>

<div class="subtabs" role="tablist" aria-label="View">
  {#each subs as s}
    <button
      type="button"
      role="tab"
      aria-selected={$subRoute === s.id}
      class:active={$subRoute === s.id}
      on:click={() => go(top, s.id)}
    >{s.label}</button>
  {/each}
</div>

<style>
  .subtabs {
    display: inline-flex;
    gap: 0.25rem;
    padding: 0.25rem;
    margin-bottom: 1.25rem;
    background: var(--surface-sunk, rgba(0, 0, 0, 0.03));
    border: 1px solid var(--rule);
    border-radius: 8px;
  }
  .subtabs button {
    padding: 0.35rem 0.9rem;
    border: 0;
    border-radius: 6px;
    background: transparent;
    color: var(--ink-muted);
    font-family: var(--font-body);
    font-size: var(--fs-small);
    cursor: pointer;
    transition: background 0.15s ease, color 0.15s ease;
  }
  .subtabs button:hover { color: var(--ink); }
  .subtabs button.active { background: var(--surface); color: var(--ink); box-shadow: 0 1px 2px rgba(0, 0, 0, 0.08); }
</style>
```

- [ ] **Step 3: Create the two top containers**

Create `dashboard/src/lib/tabs/PhenotypeAtlas.svelte`:

```svelte
<script lang="ts">
  import SubTabs from '../SubTabs.svelte'
  import { subRoute } from '../router'
  import Atlas from './Atlas.svelte'
  import Compare from '../atlas/Compare.svelte'
</script>

<SubTabs top="atlas" />
{#if $subRoute === 'compare'}
  <Compare />
{:else}
  <Atlas />
{/if}
```

Create `dashboard/src/lib/tabs/SimulatorGroup.svelte`:

```svelte
<script lang="ts">
  import SubTabs from '../SubTabs.svelte'
  import { subRoute } from '../router'
  import Simulator from './Simulator.svelte'
  import Patient from './Patient.svelte'
</script>

<SubTabs top="sim" />
{#if $subRoute === 'explore'}
  <Patient />
{:else}
  <Simulator />
{/if}
```

Note: `Compare.svelte` is created in Task 6. Create a minimal placeholder now so this compiles: `dashboard/src/lib/atlas/Compare.svelte` with `<p>Compare</p>` (Task 6 replaces it).

- [ ] **Step 4: Wire `App.svelte`**

In `dashboard/src/App.svelte`:
- Replace the tab-body imports (`Atlas`, `Patient`, `Simulator`) with the two containers:
  ```typescript
  import { topRoute, type TopId } from './lib/router'
  import PhenotypeAtlas from './lib/tabs/PhenotypeAtlas.svelte'
  import SimulatorGroup from './lib/tabs/SimulatorGroup.svelte'
  ```
  (remove the old `import Atlas/Patient/Simulator` and the `route, type Route` import.)
- Replace the `TAB_COMPONENTS` map:
  ```typescript
  const TOP_COMPONENTS: Record<TopId, ConstructorOfATypedSvelteComponent> = {
    atlas: PhenotypeAtlas,
    sim: SimulatorGroup,
  }
  ```
- Replace the render line:
  ```svelte
    <Tabs />
    <svelte:component this={TOP_COMPONENTS[$topRoute]} />
  ```

- [ ] **Step 5: Update the tour ids**

In `dashboard/src/lib/tour.ts`, update any step that targets `tab-patient` or `tab-simulator` (now under `tab-sim`) or references the old route ids. Keep the tour functional (it may simply target `tab-atlas` and `tab-sim`). If a tour step drove the old `patient`/`simulator` tabs specifically, retarget it to `sim` (and, if it depended on a subtab, call `go('sim','explore')` / `go('sim','simulate')`).

- [ ] **Step 6: Verify build + type-check**

Run: `cd dashboard && npx svelte-check --threshold error && npx vite build`
Expected: builds; no new type errors (pre-existing baseline: umap-js cosine, App.svelte import.meta.env ×2, covariate.test.ts).

Manual smoke: `npx vite preview` — top tabs switch Phenotype Atlas/Simulator; subtabs switch Explore/Compare and Simulate Cohort/Explore Cohort; a legacy `#/patient` hash lands on Simulator → Explore Cohort.

- [ ] **Step 7: Commit**

```bash
git add dashboard/src/lib/SubTabs.svelte dashboard/src/lib/tabs/PhenotypeAtlas.svelte dashboard/src/lib/tabs/SimulatorGroup.svelte dashboard/src/lib/atlas/Compare.svelte dashboard/src/lib/Tabs.svelte dashboard/src/App.svelte dashboard/src/lib/tour.ts
git commit -m "feat(dashboard): two top tabs with subtabs (Explore/Compare, Simulate/Explore Cohort)"
```

---

## Task 3: Explore subtab — drop Topic mass column; prevalence re-sorts

**Files:**
- Modify: `dashboard/src/lib/store.ts` (drop `topic_mass` from `PhenotypeSortKey`), `dashboard/src/lib/atlas/PhenotypeBrowser.svelte`, `dashboard/src/lib/tabs/Atlas.svelte` (remove the viz-switch + correlation branch — Explore is bubbles-only now)
- Test: `dashboard/src/lib/atlas/PhenotypeBrowser.test.ts` (create)

**Interfaces:**
- Consumes: `prevalenceReader`, `phenotypeSortBy`, `phenotypeSortDir` (existing).
- Produces: `PhenotypeSortKey = 'id' | 'label' | 'cohort' | 'prevalence' | 'coherence'` (no `topic_mass`).

- [ ] **Step 1: Write the failing test**

Create `dashboard/src/lib/atlas/PhenotypeBrowser.test.ts`. It asserts (a) there is no "Topic mass" column, and (b) when sorted by prevalence, changing the conditioning re-orders rows. Because prevalence depends on the bundle + conditioning stores, build a minimal STM bundle and drive `atlasConditioning`:

```typescript
import { it, expect, afterEach } from 'vitest'
import { render, cleanup } from '@testing-library/svelte'
import { get } from 'svelte/store'
import PhenotypeBrowser from './PhenotypeBrowser.svelte'
import { bundle, phenotypeSortBy, atlasConditioning } from '../store'
import { makeStmBundleFixture } from '../test-fixtures'

afterEach(() => cleanup())

it('renders no Topic mass column', () => {
  bundle.set(makeStmBundleFixture())
  const { queryByText } = render(PhenotypeBrowser)
  expect(queryByText('Topic mass')).toBeNull()
})

it('re-sorts when conditioning changes the prevalence order', async () => {
  bundle.set(makeStmBundleFixture())
  phenotypeSortBy.set('prevalence')
  atlasConditioning.set({ covariateActive: false, values: {}, group: null })
  const { container } = render(PhenotypeBrowser)
  const firstRowId = () => container.querySelector('tbody tr')?.getAttribute('data-pid')
  const before = firstRowId()
  // Flip to a covariate setting that reorders prevalence for the fixture.
  atlasConditioning.set({ covariateActive: true, values: { age: 80 }, group: null })
  await Promise.resolve()
  const after = firstRowId()
  expect(after).not.toBe(before)
})
```

You must add a shared fixture `dashboard/src/lib/test-fixtures.ts` exporting `makeStmBundleFixture()` — a small STM `DashboardBundle` (K=3, reference 0, two free topics whose covariate effects invert their prevalence order between age=60 and age=80) with `covariateEffects`, `covariateSchema` (age control), `correlation` (reference_topic 0), `phenotypes`, `model`, `corpusStats`. Add `data-pid={p.id}` to the `PhenotypeBrowser` row `<tr>` if it is not already present (needed by the test's `firstRowId`).

- [ ] **Step 2: Run test to verify it fails**

Run: `cd dashboard && npm run test -- src/lib/atlas/PhenotypeBrowser.test.ts`
Expected: FAIL — "Topic mass" still present / rows do not re-order / fixture missing.

- [ ] **Step 3: Drop the `topic_mass` sort key**

In `dashboard/src/lib/store.ts`, change the `PhenotypeSortKey` union (currently lines 137-138) to drop `'topic_mass'`:

```typescript
export type PhenotypeSortKey =
  | 'id' | 'label' | 'cohort' | 'prevalence' | 'coherence'
```

- [ ] **Step 4: Remove the Topic mass column + confirm reactive rows**

In `dashboard/src/lib/atlas/PhenotypeBrowser.svelte`:
- Delete the `topic_mass` header `<th>` (lines 171-173) and its `<td>` cell (lines 208-215).
- Delete the `case 'topic_mass':` branch in the comparator (lines 58-59) and any `maxMass` computation and `.mass-fill` CSS that becomes unused.
- Ensure the rendered rows are a reactive derivation of `$prevalenceReader` so a conditioning change recomputes prevalence and — when the active sort is `prevalence` — re-sorts. Confirm the sorted list is built inside a `$:` reactive that reads `$prevalenceReader` (not computed once `onMount`). If it currently sorts in a non-reactive block, move the sort into a `$:` that depends on `$prevalenceReader`, `$phenotypeSortBy`, `$phenotypeSortDir`.
- Add `data-pid={p.id}` to the row `<tr>` if not present.

- [ ] **Step 5: Make Explore bubbles-only**

In `dashboard/src/lib/tabs/Atlas.svelte`, remove the `.viz-switch` segmented control (lines 54-71) and the `{#if view === 'correlation'}` branch (lines 73-82); keep only the `<TopicMap>` block. Remove the now-unused `let view` (line 17), the `CorrelationHeatmap` import, and the `.viz-switch`/`.corr-wrap`/`.corr-kicker` CSS. The correlation heatmap now lives in `Compare.svelte` (Task 6). Keep `<PhenotypeBrowser />` and `<CodePanel />`.

- [ ] **Step 6: Run test + type-check**

Run: `cd dashboard && npm run test -- src/lib/atlas/PhenotypeBrowser.test.ts && npx svelte-check --threshold error`
Expected: PASS; no unused-CSS or dangling-import warnings for the removed pieces.

- [ ] **Step 7: Commit**

```bash
git add dashboard/src/lib/store.ts dashboard/src/lib/atlas/PhenotypeBrowser.svelte dashboard/src/lib/atlas/PhenotypeBrowser.test.ts dashboard/src/lib/tabs/Atlas.svelte dashboard/src/lib/test-fixtures.ts
git commit -m "feat(dashboard): Explore drops Topic mass column; prevalence re-sorts on conditioning"
```

---

# PART 2 — Compare subtab + Phenotype Difference pane

## Task 4: `difference.ts` — relevance-delta ranking

**Files:**
- Create: `dashboard/src/lib/atlas/difference.ts`, `dashboard/src/lib/atlas/difference.test.ts`

**Interfaces:**
- Consumes: `relevance` from `../inference` (`relevance(pwk, pw, lambda): number`).
- Produces:
  ```typescript
  export interface RankedDelta { index: number; delta: number; relA: number; relB: number }
  export function topDifferentialCodes(input: {
    betaA: number[]; betaB: number[]; pw: number[]; lambda: number; n: number
  }): { aSide: RankedDelta[]; bSide: RankedDelta[] }
  ```
  `delta(w) = relevance(betaA[w], pw[w], λ) − relevance(betaB[w], pw[w], λ)`. `aSide` = top-n by descending delta (finite only); `bSide` = top-n by ascending delta (finite only). Terms where a side's β is 0 (relevance −∞) are excluded from the side where they would be −∞ but may appear on the other side.

- [ ] **Step 1: Write the failing test**

Create `dashboard/src/lib/atlas/difference.test.ts`:

```typescript
import { describe, it, expect } from 'vitest'
import { topDifferentialCodes } from './difference'

describe('topDifferentialCodes', () => {
  // 4 terms. A loads term 0 heavily, B loads term 3 heavily.
  const betaA = [0.7, 0.2, 0.09, 0.01]
  const betaB = [0.01, 0.2, 0.09, 0.7]
  const pw = [0.25, 0.25, 0.25, 0.25]

  it('ranks A-distinctive terms on aSide and B-distinctive on bSide', () => {
    const { aSide, bSide } = topDifferentialCodes({ betaA, betaB, pw, lambda: 0.6, n: 2 })
    expect(aSide[0].index).toBe(0)   // term 0 most elevated in A
    expect(bSide[0].index).toBe(3)   // term 3 most elevated in B
  })

  it('is antisymmetric: swapping A/B negates deltas and swaps the sides', () => {
    const ab = topDifferentialCodes({ betaA, betaB, pw, lambda: 0.6, n: 4 })
    const ba = topDifferentialCodes({ betaA: betaB, betaB: betaA, pw, lambda: 0.6, n: 4 })
    expect(ba.aSide[0].index).toBe(ab.bSide[0].index)
    expect(ba.aSide[0].delta).toBeCloseTo(-ab.bSide[0].delta, 10)
  })

  it('excludes a term from the side where its beta is zero (relevance -Infinity)', () => {
    const bA = [0.5, 0.5, 0.0]   // term 2 absent from A
    const bB = [0.4, 0.3, 0.3]
    const { aSide } = topDifferentialCodes({ betaA: bA, betaB: bB, pw: [0.33, 0.33, 0.34], lambda: 0.6, n: 3 })
    expect(aSide.some((r) => r.index === 2)).toBe(false)   // -Inf delta not on A side
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd dashboard && npm run test -- src/lib/atlas/difference.test.ts`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

Create `dashboard/src/lib/atlas/difference.ts`:

```typescript
import { relevance } from '../inference'

// Rank conditions by the difference in term relevance between two phenotypes A
// and B. Relevance is the LDAvis measure (Sievert & Shirley 2014):
// lambda*log(beta) + (1-lambda)*log(beta/p). delta(w) = rel_A(w) - rel_B(w);
// large positive delta marks a condition distinctive of A, large negative of B.
// The metric is a pure beta (term-weight) contrast, so it is defined even for
// phenotype pairs whose topic correlation is unidentified (a cross-group cell).
export interface RankedDelta {
  index: number
  delta: number
  relA: number
  relB: number
}

export function topDifferentialCodes(input: {
  betaA: number[]
  betaB: number[]
  pw: number[]
  lambda: number
  n: number
}): { aSide: RankedDelta[]; bSide: RankedDelta[] } {
  const { betaA, betaB, pw, lambda, n } = input
  const rows: RankedDelta[] = betaA.map((_, i) => {
    const relA = relevance(betaA[i], pw[i] ?? 0, lambda)
    const relB = relevance(betaB[i], pw[i] ?? 0, lambda)
    return { index: i, delta: relA - relB, relA, relB }
  })
  // Ascending-delta candidates need finite delta; a -Infinity delta means A has
  // beta 0 there (belongs to B side, not A) and vice versa. NaN (both 0) drops.
  const finite = rows.filter((r) => Number.isFinite(r.delta))
  const desc = finite.slice().sort((a, b) => b.delta - a.delta)
  const asc = finite.slice().sort((a, b) => a.delta - b.delta)
  return { aSide: desc.slice(0, n), bSide: asc.slice(0, n) }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd dashboard && npm run test -- src/lib/atlas/difference.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/lib/atlas/difference.ts dashboard/src/lib/atlas/difference.test.ts
git commit -m "feat(dashboard): relevance-delta ranking for the phenotype Difference pane"
```

---

## Task 5: Correlation heatmap — cell click sets a pair + highlight

**Files:**
- Modify: `dashboard/src/lib/store.ts` (add `comparePair`), `dashboard/src/lib/atlas/CorrelationHeatmap.svelte`
- Test: `dashboard/src/lib/atlas/CorrelationHeatmap.test.ts` (extend)

**Interfaces:**
- Produces: `store.ts` `comparePair: Writable<{ a: number; b: number } | null>`.
- `CorrelationHeatmap` gains an optional prop `pairSelect = false`; when true, clicking cell (mr, mc) sets `comparePair = { a: order[mr], b: order[mc] }` (and still leaves `selectedPhenotypeId` behavior for the Explore usage untouched — but Explore no longer mounts the heatmap, so default `pairSelect=false` keeps the existing single-select tests valid). The clicked cell and its row/column get a highlight class.

- [ ] **Step 1: Write the failing test**

Add to `dashboard/src/lib/atlas/CorrelationHeatmap.test.ts`:

```typescript
import { comparePair } from '../store'

it('in pair-select mode, clicking a cell sets the comparePair (row=A, col=B)', async () => {
  comparePair.set(null)
  const { container } = render(CorrelationHeatmap, { props: { correlation, pairSelect: true } })
  // default All × All -> background cells (matrix rows/cols {0,1})
  const cell = container.querySelector('rect.cell[data-mr="0"][data-mc="1"]') as SVGRectElement
  await fireEvent.click(cell)
  expect(get(comparePair)).toEqual({ a: 0, b: 1 })   // order[0]=0, order[1]=1
})

it('diagonal click (A===B) clears the pair', async () => {
  comparePair.set({ a: 9, b: 9 })
  const { container } = render(CorrelationHeatmap, { props: { correlation, pairSelect: true } })
  const cell = container.querySelector('rect.cell[data-mr="0"][data-mc="0"]') as SVGRectElement
  await fireEvent.click(cell)
  expect(get(comparePair)).toBeNull()
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd dashboard && npm run test -- src/lib/atlas/CorrelationHeatmap.test.ts`
Expected: FAIL — `comparePair` not exported / `pairSelect` ignored.

- [ ] **Step 3: Add the store**

In `dashboard/src/lib/store.ts` add near `selectedPhenotypeId` (line 57):

```typescript
// Compare subtab: the two phenotypes being contrasted in the Difference pane.
// Set by clicking a correlation cell (a = row phenotype, b = column phenotype);
// a === b (a diagonal click) clears it.
export const comparePair = writable<{ a: number; b: number } | null>(null)
```

- [ ] **Step 4: Wire the heatmap**

In `dashboard/src/lib/atlas/CorrelationHeatmap.svelte`:
- Add `export let pairSelect = false` and import `comparePair`.
- Change the cell `on:click` (line 156) to branch:
  ```svelte
        on:click={() => onCellClick(c.mr, c.mc)}
  ```
  and add:
  ```typescript
  function onCellClick(mr: number, mc: number) {
    if (pairSelect) {
      const a = order[mr], b = order[mc]
      comparePair.set(a === b ? null : { a, b })
    } else {
      selectCol(mc)
    }
  }
  ```
- Add a highlight: give the `<rect class="cell">` a `class:selected={pairSelect && $comparePair && ((order[c.mr] === $comparePair.a && order[c.mc] === $comparePair.b))}` and a `.cell.selected { stroke: var(--accent); stroke-width: 2; }` CSS rule.

- [ ] **Step 5: Run test + full heatmap suite**

Run: `cd dashboard && npm run test -- src/lib/atlas/CorrelationHeatmap.test.ts`
Expected: PASS (new pair tests + the existing single-select/NA tests unchanged).

- [ ] **Step 6: Commit**

```bash
git add dashboard/src/lib/store.ts dashboard/src/lib/atlas/CorrelationHeatmap.svelte dashboard/src/lib/atlas/CorrelationHeatmap.test.ts
git commit -m "feat(dashboard): heatmap pair-select mode drives comparePair + cell highlight"
```

---

## Task 6: DifferencePane + Compare subtab body

**Files:**
- Create: `dashboard/src/lib/atlas/DifferencePane.svelte`
- Replace: `dashboard/src/lib/atlas/Compare.svelte` (the Task 2 placeholder)

**Interfaces:**
- Consumes: `topDifferentialCodes` (Task 4), `comparePair` (Task 5), `bundle`, `phenotypesById`; the bundle's `model.beta` (K×V), corpus term frequency (the same `pw`/`corpusFreq` array the CodePanel uses — confirm its store/derivation name in `store.ts`/`CodePanel.svelte` and reuse it), and `vocab` for condition labels.
- Produces: the Compare subtab — `CorrelationHeatmap` with `pairSelect` + the `DifferencePane` sidebar, mirroring the Explore grid.

- [ ] **Step 1: Build `DifferencePane.svelte`**

Create `dashboard/src/lib/atlas/DifferencePane.svelte`:
- Local `let lambda = 0.6` with a range `<input type="range" min="0" max="1" step="0.05">` labeled "Lift ↔ Frequency" (λ). No conditioning controls.
- Reactive: when `$comparePair` is null or `a===b`, show an empty-state hint ("Click a cell in the heatmap to compare two phenotypes."). Otherwise compute
  ```typescript
  $: result = $comparePair && $bundle
    ? topDifferentialCodes({
        betaA: $bundle.model.beta[$comparePair.a],
        betaB: $bundle.model.beta[$comparePair.b],
        pw: corpusFreq,          // reuse the CodePanel's corpus term frequency
        lambda, n: 15,
      })
    : null
  ```
- Render two labeled lists: `More in {nameOf(a)}` (result.aSide) and `More in {nameOf(b)}` (result.bSide), each row showing the condition description (from `vocab`) and its delta. Use `phenotypesById`/the label map for names (mirror `CorrelationHeatmap`'s `labelById`).

- [ ] **Step 2: Build `Compare.svelte`**

Replace `dashboard/src/lib/atlas/Compare.svelte` with the heatmap + Difference sidebar, mirroring Explore's grid (`.grid` → `.left-col` + right sidebar):

```svelte
<script lang="ts">
  import { bundle } from '../store'
  import { copy } from '../copy'
  import CorrelationHeatmap from './CorrelationHeatmap.svelte'
  import DifferencePane from './DifferencePane.svelte'
</script>

<div class="grid">
  <div class="left-col">
    {#if $bundle?.correlation}
      <p class="corr-kicker">{copy.correlation.kicker}</p>
      <CorrelationHeatmap correlation={$bundle.correlation} pairSelect />
    {:else}
      <p>Correlations are not available for this bundle.</p>
    {/if}
  </div>
  <DifferencePane />
</div>
```

(Reuse the `.grid`/`.left-col`/`.corr-kicker` CSS pattern from the old Atlas layout; copy those rules into `Compare.svelte`'s `<style>`.)

- [ ] **Step 3: Verify build + type-check + manual**

Run: `cd dashboard && npm run test && npx svelte-check --threshold error && npx vite build`
Expected: all green. Manual: on Phenotype Atlas → Compare, click a heatmap cell → Difference pane fills with two lists; the λ slider re-ranks; a diagonal click clears to the empty state; a cross-group NA cell still yields a ranking.

- [ ] **Step 4: Commit**

```bash
git add dashboard/src/lib/atlas/DifferencePane.svelte dashboard/src/lib/atlas/Compare.svelte
git commit -m "feat(dashboard): Compare subtab — heatmap pair-select + Phenotype Difference pane"
```

---

# PART 3 — Faithful STM record-completion + unification

## Task 7: SPD linear algebra + PD guard; fix the group-Σ throw

**Files:**
- Create: `dashboard/src/lib/conditioning/linalg.ts`, `dashboard/src/lib/conditioning/linalg.test.ts`
- Modify: `dashboard/src/lib/conditioning/logisticNormal.ts` (use `choleskyPD`)

**Interfaces:**
- Consumes: `cholesky` from `./logisticNormal` (existing, throws on non-PD).
- Produces:
  - `choleskyPD(A: number[][]): number[][]` — cholesky with adaptive diagonal loading (ridge) so an indefinite/singular symmetric matrix still factors.
  - `solveSPD(A: number[][], b: number[]): number[]` — solves A x = b for symmetric-PD-ish A.
  - `invSPD(A: number[][]): number[][]` — inverse of symmetric-PD-ish A.

- [ ] **Step 1: Write the failing test**

Create `dashboard/src/lib/conditioning/linalg.test.ts`:

```typescript
import { describe, it, expect } from 'vitest'
import { choleskyPD, solveSPD, invSPD } from './linalg'

describe('choleskyPD', () => {
  it('factors a PD matrix (L Lᵀ = A)', () => {
    const A = [[4, 2], [2, 3]]
    const L = choleskyPD(A)
    expect(L[0][0] * L[0][0]).toBeCloseTo(4, 9)
    expect(L[1][0] * L[0][0]).toBeCloseTo(2, 9)
    expect(L[1][0] ** 2 + L[1][1] ** 2).toBeCloseTo(3, 9)
  })
  it('regularizes a non-PD matrix instead of throwing', () => {
    const indefinite = [[1, 2], [2, 1]]   // eigenvalues 3, -1
    expect(() => choleskyPD(indefinite)).not.toThrow()
    const L = choleskyPD(indefinite)
    expect(Number.isFinite(L[0][0])).toBe(true)
  })
})

describe('solveSPD / invSPD', () => {
  it('solves A x = b', () => {
    const A = [[4, 1], [1, 3]]
    const x = solveSPD(A, [1, 2])
    // A x should be ~ [1,2]
    expect(A[0][0] * x[0] + A[0][1] * x[1]).toBeCloseTo(1, 8)
    expect(A[1][0] * x[0] + A[1][1] * x[1]).toBeCloseTo(2, 8)
  })
  it('inverts A (A · A⁻¹ = I)', () => {
    const A = [[4, 1], [1, 3]]
    const Ai = invSPD(A)
    const p00 = A[0][0] * Ai[0][0] + A[0][1] * Ai[1][0]
    const p01 = A[0][0] * Ai[0][1] + A[0][1] * Ai[1][1]
    expect(p00).toBeCloseTo(1, 8)
    expect(p01).toBeCloseTo(0, 8)
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd dashboard && npm run test -- src/lib/conditioning/linalg.test.ts`
Expected: FAIL — module missing.

- [ ] **Step 3: Write `linalg.ts`**

Create `dashboard/src/lib/conditioning/linalg.ts`:

```typescript
import { cholesky } from './logisticNormal'

// Cholesky with adaptive diagonal loading (ridge / Tikhonov regularization).
// The exported topic-correlation sub-block is only guaranteed positive-definite
// within a single block; a background-union-group sub-block can be indefinite,
// on which the bare `cholesky` throws. Adding a small multiple of the identity
// (grown geometrically until the factor exists) yields a usable, minimally
// perturbed factor. The load starts negligibly small so a genuinely-PD input is
// essentially unperturbed.
export function choleskyPD(A: number[][]): number[][] {
  const n = A.length
  if (n === 0) return []
  let meanDiag = 0
  for (let i = 0; i < n; i++) meanDiag += A[i][i]
  meanDiag = Math.abs(meanDiag) / n || 1
  const base = meanDiag * 1e-12
  for (let t = 0; t < 60; t++) {
    const load = t === 0 ? 0 : base * Math.pow(2, t)
    try {
      const M = load === 0 ? A : A.map((row, i) => row.map((v, j) => (i === j ? v + load : v)))
      return cholesky(M)
    } catch {
      // not PD at this load; increase and retry
    }
  }
  throw new Error('choleskyPD: not factorable even with regularization')
}

function forwardSub(L: number[][], b: number[]): number[] {
  const n = L.length
  const y = new Array<number>(n)
  for (let i = 0; i < n; i++) {
    let s = b[i]
    for (let k = 0; k < i; k++) s -= L[i][k] * y[k]
    y[i] = s / L[i][i]
  }
  return y
}

function backSub(L: number[][], y: number[]): number[] {
  const n = L.length
  const x = new Array<number>(n)
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i]
    for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k]
    x[i] = s / L[i][i]
  }
  return x
}

// Solve A x = b for symmetric (regularized-)PD A via its Cholesky factor.
export function solveSPD(A: number[][], b: number[]): number[] {
  const L = choleskyPD(A)
  return backSub(L, forwardSub(L, b))
}

// Inverse of symmetric (regularized-)PD A, column-by-column against I.
export function invSPD(A: number[][]): number[][] {
  const n = A.length
  const L = choleskyPD(A)
  const inv: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0))
  for (let j = 0; j < n; j++) {
    const e = new Array<number>(n).fill(0)
    e[j] = 1
    const col = backSub(L, forwardSub(L, e))
    for (let i = 0; i < n; i++) inv[i][j] = col[i]
  }
  return inv
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd dashboard && npm run test -- src/lib/conditioning/linalg.test.ts`
Expected: PASS.

- [ ] **Step 5: Use `choleskyPD` in `sampleConditionedTheta` (fix the throw)**

In `dashboard/src/lib/conditioning/logisticNormal.ts`, import `choleskyPD` from `./linalg` and replace the `cholesky(Sigma)` call (line 86) with `choleskyPD(Sigma)`. Add a one-line failing-then-passing test in `logisticNormal.test.ts` that a full (all-topics) correlation sub-block which is not PD no longer throws:

```typescript
it('does not throw when the Sigma sub-block is not positive-definite', () => {
  // 2 free topics with correlation 1.1 in magnitude -> indefinite 2x2 block.
  const corr: Correlation = {
    topic_order: [1, 2], block_labels: ['background', 'background'],
    R: [[1, 1.1], [1.1, 1]], identified: [[true, true], [true, true]],
    support: [[9, 9], [9, 9]], reference_topic: 0,
  }
  const effects: CovariateEffects = [{ covariate: 'Intercept', per_topic: [0, 0, 0] }]
  expect(() => sampleConditionedTheta({
    effects, x: [1], correlation: corr, topicBlocks: null, group: null, rng: createRng(1),
  })).not.toThrow()
})
```

(Note: `cholesky` remains exported from `logisticNormal.ts` for `linalg.ts` to import.)

- [ ] **Step 6: Run tests + type-check**

Run: `cd dashboard && npm run test -- src/lib/conditioning && npx svelte-check --threshold error`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add dashboard/src/lib/conditioning/linalg.ts dashboard/src/lib/conditioning/linalg.test.ts dashboard/src/lib/conditioning/logisticNormal.ts dashboard/src/lib/conditioning/logisticNormal.test.ts
git commit -m "feat(dashboard): PD-guarded cholesky + SPD solve/inverse; fix group-Sigma throw"
```

---

## Task 8: `recordPosterior.ts` — logistic-normal posterior given the prefix

**Files:**
- Create: `dashboard/src/lib/conditioning/recordPosterior.ts`, `dashboard/src/lib/conditioning/recordPosterior.test.ts`

**Interfaces:**
- Consumes: `sampleConditionedTheta`, `mvnDraw` from `./logisticNormal`; `choleskyPD`, `invSPD`, `solveSPD` from `./linalg`; types `CovariateEffects`, `Correlation`.
- Produces:
  ```typescript
  export function sampleRecordPosterior(args: {
    effects: CovariateEffects
    x: number[]
    correlation: Correlation
    topicBlocks: string[] | null
    group: string | null
    prefixCounts: Map<number, number>   // observed prefix code index -> count
    beta: number[][]                    // K x V
    rng: () => number
  }): number[]
  ```
  Returns a length-K θ over display topics (masked/out-of-group topics exactly 0).
  **Empty `prefixCounts` ⇒ identical to `sampleConditionedTheta`** (same rng consumption): the record-completion path reduces to the prior cohort draw.

**The math (document in the file's header comment, no LaTeX):**
Prior η ~ Normal(μ, Σ) over the allowed free topics (μ = Γᵀx, Σ the correlation
sub-block; reference pinned η=0, masked topics excluded), θ = softmax(η). Given
prefix codes D, the posterior is p(η|D) ∝ Normal(η;μ,Σ)·∏₍w∈D₎ (Σ_k θ_k β_{k,w}).
Find the mode by Fisher scoring: exact gradient
`g = −Σ⁻¹(η−μ) + Σ_w n_w (φ_w − θ_free)` with responsibilities φ_{w,k}=θ_k β_{k,w}/s_w,
s_w=Σ_k θ_k β_{k,w}; step with the PD expected-information curvature
`H = Σ⁻¹ + Σ_w n_w (diag(θ_free) − θ_free θ_freeᵀ)` (a Gauss-Newton approximation —
PD by construction). Draw η ~ Normal(η*, H⁻¹) (Laplace approximation; Blei &
Lafferty 2007). The reference contributes to each s_w but its η stays 0.

- [ ] **Step 1: Write the failing test**

Create `dashboard/src/lib/conditioning/recordPosterior.test.ts`:

```typescript
import { describe, it, expect } from 'vitest'
import { createRng } from '../sampling'
import { sampleConditionedTheta } from './logisticNormal'
import { sampleRecordPosterior } from './recordPosterior'
import type { Correlation, CovariateEffects } from '../types'

function identityCorr(order: number[]): Correlation {
  const K1 = order.length
  const R = Array.from({ length: K1 }, (_, i) => Array.from({ length: K1 }, (_, j) => (i === j ? 1 : 0)))
  return { topic_order: order, block_labels: order.map(() => 'background'),
    R, identified: R.map((r) => r.map(() => true)), support: R.map((r) => r.map(() => 9)),
    reference_topic: 0 }
}
const beta3 = [[0.34, 0.33, 0.33], [0.9, 0.05, 0.05], [0.05, 0.9, 0.05]]  // K=3, V=3

describe('sampleRecordPosterior', () => {
  it('empty prefix reduces exactly to the prior draw', () => {
    const effects: CovariateEffects = [{ covariate: 'Intercept', per_topic: [0, 0.3, -0.2] }]
    const corr = identityCorr([1, 2])
    const args = { effects, x: [1], correlation: corr, topicBlocks: null, group: null }
    const prior = sampleConditionedTheta({ ...args, rng: createRng(42) })
    const post = sampleRecordPosterior({ ...args, prefixCounts: new Map(), beta: beta3, rng: createRng(42) })
    expect(post).toEqual(prior)
  })

  it('a prefix loading topic 1 concentrates the posterior on topic 1', () => {
    const effects: CovariateEffects = [{ covariate: 'Intercept', per_topic: [0, 0, 0] }]
    const corr = identityCorr([1, 2])
    // term 0 is emitted almost only by topic 1 (beta3[1][0]=0.9)
    const rng = createRng(7)
    let s1 = 0, s2 = 0
    for (let i = 0; i < 400; i++) {
      const t = sampleRecordPosterior({ effects, x: [1], correlation: corr,
        topicBlocks: null, group: null, prefixCounts: new Map([[0, 15]]), beta: beta3, rng })
      s1 += t[1]; s2 += t[2]
    }
    expect(s1).toBeGreaterThan(s2 * 3)   // strongly concentrated on topic 1
  })

  it('keeps out-of-group foreground topics at exactly 0', () => {
    const effects: CovariateEffects = [{ covariate: 'Intercept', per_topic: [0, 0, 0, 0] }]
    const corr = identityCorr([1, 2, 3])
    const beta4 = [[0.5, 0.5], [0.5, 0.5], [0.9, 0.1], [0.1, 0.9]]
    const t = sampleRecordPosterior({
      effects, x: [1], correlation: corr,
      topicBlocks: ['background', 'background', 'cancer', 'dementia'],
      group: 'cancer', prefixCounts: new Map([[0, 5]]), beta: beta4, rng: createRng(3),
    })
    expect(t[3]).toBe(0)                 // dementia masked
    expect(t.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10)
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd dashboard && npm run test -- src/lib/conditioning/recordPosterior.test.ts`
Expected: FAIL — module missing.

- [ ] **Step 3: Write `recordPosterior.ts`**

Create `dashboard/src/lib/conditioning/recordPosterior.ts`:

```typescript
import { sampleConditionedTheta, mvnDraw } from './logisticNormal'
import { choleskyPD, invSPD, solveSPD } from './linalg'
import type { CovariateEffects, Correlation } from '../types'

// Faithful STM record completion. The prefix (observed starting conditions)
// conditions theta via the posterior over eta under the covariate/group prior
// eta ~ Normal(mu, Sigma) (Blei & Lafferty 2007 logistic-normal). We find the
// posterior mode by Fisher scoring and draw from a Laplace Gaussian around it.
// An empty prefix has no likelihood term, so the mode is mu and the Laplace
// covariance is Sigma -> this reduces exactly to sampleConditionedTheta (the
// covariate/group prior draw), which is why cohort generation and record
// completion are a single code path.
export function sampleRecordPosterior(args: {
  effects: CovariateEffects
  x: number[]
  correlation: Correlation
  topicBlocks: string[] | null
  group: string | null
  prefixCounts: Map<number, number>
  beta: number[][]
  rng: () => number
}): number[] {
  const { effects, x, correlation, topicBlocks, group, prefixCounts, beta, rng } = args

  // Empty prefix: the posterior is the prior. Delegate so the draw is identical.
  if (prefixCounts.size === 0) {
    return sampleConditionedTheta({ effects, x, correlation, topicBlocks, group, rng })
  }

  const K = effects[0]?.per_topic.length ?? 0
  const ref = correlation.reference_topic ?? -1
  const order = correlation.topic_order

  const allowed = (k: number): boolean => {
    if (!topicBlocks) return true
    const b = topicBlocks[k]
    return b === 'background' || b === group
  }

  // Free R rows: allowed, non-reference. `ids[i]` is the display topic id of free i.
  const freeIdx: number[] = []
  for (let r = 0; r < order.length; r++) {
    const k = order[r]
    if (k !== ref && allowed(k)) freeIdx.push(r)
  }
  const ids = freeIdx.map((r) => order[r])
  const F = ids.length

  const refAllowed = ref >= 0 && allowed(ref)

  // Prior mean mu and covariance Sigma over the free rows.
  const mu = freeIdx.map((r) => {
    const k = order[r]
    let m = 0
    effects.forEach((e, p) => { m += e.per_topic[k] * x[p] })
    return m
  })
  const Sigma = freeIdx.map((ri) => freeIdx.map((rj) => correlation.R[ri][rj] as number))

  if (F === 0) {
    // Only the reference is allowed: all mass on it (or empty -> uniform-safe).
    const theta = new Array<number>(K).fill(0)
    if (refAllowed) theta[ref] = 1
    return theta
  }

  const Sinv = invSPD(Sigma)

  // Prefix codes as arrays for the likelihood loop.
  const codes = [...prefixCounts.keys()]
  const counts = codes.map((w) => prefixCounts.get(w)!)

  // theta over the allowed set from free eta (reference eta = 0).
  const thetaFromEta = (eta: number[]): { thetaFree: number[]; thetaRef: number } => {
    const logits = refAllowed ? [0, ...eta] : [...eta]
    const mx = Math.max(...logits)
    const ex = logits.map((v) => Math.exp(v - mx))
    const s = ex.reduce((a, b) => a + b, 0) || 1
    if (refAllowed) {
      const thetaRef = ex[0] / s
      return { thetaFree: ex.slice(1).map((e) => e / s), thetaRef }
    }
    return { thetaFree: ex.map((e) => e / s), thetaRef: 0 }
  }

  // Fisher scoring to the posterior mode.
  let eta = mu.slice()
  for (let iter = 0; iter < 50; iter++) {
    const { thetaFree, thetaRef } = thetaFromEta(eta)

    // Gradient of the log-likelihood: sum_w n_w (phi_free - thetaFree).
    const gLik = new Array<number>(F).fill(0)
    // Expected-information curvature accumulator starts from the prior precision.
    const H = Sinv.map((row) => row.slice())
    for (let c = 0; c < codes.length; c++) {
      const w = codes[c]
      const nw = counts[c]
      // s_w over allowed topics (reference included), phi over free topics.
      let sw = refAllowed ? thetaRef * (beta[ref]?.[w] ?? 0) : 0
      const phi = new Array<number>(F)
      for (let i = 0; i < F; i++) {
        const contrib = thetaFree[i] * (beta[ids[i]]?.[w] ?? 0)
        phi[i] = contrib
        sw += contrib
      }
      if (sw <= 0) continue
      for (let i = 0; i < F; i++) {
        phi[i] /= sw
        gLik[i] += nw * (phi[i] - thetaFree[i])
      }
      // Add n_w (diag(thetaFree) - thetaFree thetaFreeᵀ) to H (PSD data term).
      for (let i = 0; i < F; i++) {
        H[i][i] += nw * thetaFree[i]
        for (let j = 0; j < F; j++) H[i][j] -= nw * thetaFree[i] * thetaFree[j]
      }
    }
    // Prior gradient: -Sinv (eta - mu).
    const dm = eta.map((v, i) => v - mu[i])
    const grad = gLik.map((g, i) => {
      let pg = 0
      for (let j = 0; j < F; j++) pg += Sinv[i][j] * dm[j]
      return g - pg
    })
    const step = solveSPD(H, grad)
    let maxAbs = 0
    for (let i = 0; i < F; i++) { eta[i] += step[i]; maxAbs = Math.max(maxAbs, Math.abs(step[i])) }
    if (maxAbs < 1e-6) break
  }

  // Laplace covariance = H⁻¹ at the mode; draw eta ~ Normal(eta*, H⁻¹).
  const { thetaFree } = thetaFromEta(eta)
  const H = Sinv.map((row) => row.slice())
  for (let c = 0; c < codes.length; c++) {
    const nw = counts[c]
    for (let i = 0; i < F; i++) {
      H[i][i] += nw * thetaFree[i]
      for (let j = 0; j < F; j++) H[i][j] -= nw * thetaFree[i] * thetaFree[j]
    }
  }
  const cov = invSPD(H)
  const etaDraw = mvnDraw(eta, choleskyPD(cov), rng)

  // Assemble theta over all K display topics.
  const logits = new Array<number>(K).fill(-Infinity)
  if (refAllowed) logits[ref] = 0
  ids.forEach((k, i) => { logits[k] = etaDraw[i] })
  const finite = logits.filter((e) => e !== -Infinity)
  const mx = finite.length ? Math.max(...finite) : 0
  const ex = logits.map((e) => (e === -Infinity ? 0 : Math.exp(e - mx)))
  const s = ex.reduce((a, b) => a + b, 0) || 1
  return ex.map((e) => e / s)
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd dashboard && npm run test -- src/lib/conditioning/recordPosterior.test.ts`
Expected: PASS. (If the "concentrates on topic 1" test is flaky at the chosen factor, raise the prefix count or the assertion multiple — do NOT weaken to a trivial assertion.)

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/lib/conditioning/recordPosterior.ts dashboard/src/lib/conditioning/recordPosterior.test.ts
git commit -m "feat(dashboard): sampleRecordPosterior — logistic-normal posterior given the prefix"
```

---

## Task 9: Unify Simulate Cohort ↔ Explore Cohort on the posterior draw

**Files:**
- Modify: `dashboard/src/lib/cohort.ts` (thread a prefix into `generateCohort`), `dashboard/src/lib/tabs/Simulator.svelte` (generate the shared cohort via the posterior; write `cohort`), `dashboard/src/lib/tabs/Patient.svelte` (read the shared cohort; regenerate delegates to Simulate)
- Test: `dashboard/src/lib/cohort.test.ts` (extend)

**Interfaces:**
- Consumes: `sampleRecordPosterior` (Task 8), `simulatorConditioning`/`patientConditioning`, `simulatorPrefix`, `cohort` store.
- Produces: `CohortConditioning` gains `prefixCounts?: Map<number, number>` and `beta?: number[][]`; when the prefix is non-empty and the bundle is STM, `drawOne` uses `sampleRecordPosterior` instead of `sampleConditionedTheta`. Empty prefix keeps today's path (which now equals the posterior with no data).

- [ ] **Step 1: Write the failing test**

Add to `dashboard/src/lib/cohort.test.ts` (reuse the fixture shape already in that file / `test-fixtures.ts`):

```typescript
it('set mode with a prefix concentrates the cohort toward the prefix topic', () => {
  const bundle: any = makeStmGatedFixture()   // K=4, ref 0, topic 2 emits code 0 strongly
  const noPrefix = generateCohort({
    model: bundle.model, meanCodesPerDoc: 10, n: 30, seed: 1, nNeighbors: 3,
    conditioning: { mode: 'set', values: {}, group: 'cancer', bundle },
  })
  const withPrefix = generateCohort({
    model: bundle.model, meanCodesPerDoc: 10, n: 30, seed: 1, nNeighbors: 3,
    conditioning: { mode: 'set', values: {}, group: 'cancer', bundle,
      prefixCounts: new Map([[0, 12]]), beta: bundle.model.beta },
  })
  const meanTopic2 = (c: any) => c.patients.reduce((s: number, p: any) => s + p.theta[2], 0) / c.patients.length
  expect(meanTopic2(withPrefix)).toBeGreaterThan(meanTopic2(noPrefix))
})
```

Add `makeStmGatedFixture()` to `test-fixtures.ts` if not present (K=4, reference 0, `topic_blocks` background/background/cancer/dementia, `model.beta` where topic 2 emits code 0 strongly).

- [ ] **Step 2: Run test to verify it fails**

Run: `cd dashboard && npm run test -- src/lib/cohort.test.ts`
Expected: FAIL — `prefixCounts` ignored.

- [ ] **Step 3: Thread the prefix through `cohort.ts`**

In `dashboard/src/lib/cohort.ts`:
- Add to `CohortConditioning`: `prefixCounts?: Map<number, number>` and `beta?: number[][]`.
- Import `sampleRecordPosterior` from `./conditioning/recordPosterior`.
- In `drawOne`, when `stm` and `cc!.prefixCounts && cc!.prefixCounts.size > 0 && cc!.beta`, compute θ via `sampleRecordPosterior({ effects, x, correlation, topicBlocks, group, prefixCounts: cc!.prefixCounts, beta: cc!.beta, rng })` in BOTH the `set` and `sample` branches (using that branch's `x`/`group`), replacing the `sampleConditionedTheta` call. Otherwise keep `sampleConditionedTheta`. (Because empty-prefix `sampleRecordPosterior` delegates to `sampleConditionedTheta`, you may also route unconditionally through `sampleRecordPosterior` when `beta` is provided — either is acceptable; keep the branch explicit for clarity.)

- [ ] **Step 4: Simulate Cohort writes the shared cohort**

In `dashboard/src/lib/tabs/Simulator.svelte`, change `simulate()` so that, in addition to (or instead of) the local `runSimulator` preview, it generates the full cohort via `generateCohort(...)` with `conditioning: { mode: 'set', values: $simulatorConditioning.values, group: $simulatorConditioning.group, bundle: b, prefixCounts: countsFromPrefix($simulatorPrefix), beta: b.model.beta }` and writes it to the shared `cohort` store, so Explore Cohort shows it. Keep the preview visuals (StructurePlot/PredictedRecord/SimMiniMap) driven by a small `runSimulator` sample of that same configuration. Add a helper `countsFromPrefix(prefix: number[]): Map<number, number>`.

Also fix the preview path itself: the `runSimulator` STM branch must condition on the prefix. Replace the `conditionedTheta` factory in `Simulator.svelte` (lines 50-64) with one that calls `sampleRecordPosterior` (prefix present) or `sampleConditionedTheta` (empty), so the preview matches the cohort. This removes the "prefix ignored" bug.

- [ ] **Step 5: Explore Cohort reads the shared cohort**

In `dashboard/src/lib/tabs/Patient.svelte`, ensure the view renders whatever is in the shared `cohort` store (it already reads `cohort`). Replace its standalone "Regenerate" so it either (a) routes to Simulate Cohort (`go('sim','simulate')`) or (b) calls the same shared-cohort generation with `patientConditioning`. Prefer routing to Simulate to keep one generation entry point; keep the sample/set toggle + color-by-group controls where the cohort is displayed.

- [ ] **Step 6: Run tests + type-check + build**

Run: `cd dashboard && npm run test && npx svelte-check --threshold error && npx vite build`
Expected: all green. Manual: Simulate Cohort with starting conditions → the per-sample strip is coherent (not a rainbow) and reflects the prefix; switching to Explore Cohort shows that same cohort; empty prefix reproduces the prior cohort.

- [ ] **Step 7: Commit**

```bash
git add dashboard/src/lib/cohort.ts dashboard/src/lib/cohort.test.ts dashboard/src/lib/tabs/Simulator.svelte dashboard/src/lib/tabs/Patient.svelte dashboard/src/lib/test-fixtures.ts
git commit -m "feat(dashboard): unify Simulate/Explore cohort on the posterior draw; prefix now conditions generation"
```

---

## Task 10: ADR — STM record-completion posterior (extends 0035)

**Files:**
- Create: `docs/decisions/0036-dashboard-stm-record-completion-posterior.md`
- Modify: `docs/decisions/0035-dashboard-logistic-normal-forward-sampler.md` (add a "See also 0036" note)

- [ ] **Step 1: Write the ADR**

Create `docs/decisions/0036-dashboard-stm-record-completion-posterior.md`: Status Accepted; Context (0035 shipped forward-only conditioning; the Simulator wired that prior draw into a record-completion UI and thereby ignored the prefix — each sample an independent prior draw, a rainbow with a flat mean; see the 2026-07-03 spec Appendix A); Decision (record completion is the logistic-normal posterior over η given the observed prefix, found by Fisher scoring with a PD expected-information curvature and drawn from a Laplace Gaussian; empty prefix reduces exactly to the 0035 prior draw, so cohort generation and completion share one path; a diagonal-loading PD guard makes any group Σ sub-block factorable; non-STM bundles keep the Dirichlet E-step); Consequences (Simulate Cohort and the Patient Atlas unify; the prefix genuinely steers generation; Laplace is an approximation — the mode is exact, the width is a Gauss-Newton approximation). Unicode Greek, no LaTeX; cite Blei & Lafferty 2007.

- [ ] **Step 2: Add the cross-reference to 0035**

Add a top note to `docs/decisions/0035-dashboard-logistic-normal-forward-sampler.md`: "Extended 2026-07-03 by ADR 0036 — record completion conditions the same logistic-normal prior on the observed prefix via a Laplace posterior; the forward draw here is the empty-prefix special case."

- [ ] **Step 3: Commit**

```bash
git add docs/decisions/0036-dashboard-stm-record-completion-posterior.md docs/decisions/0035-dashboard-logistic-normal-forward-sampler.md
git commit -m "docs(adr): 0036 STM record-completion posterior (extends 0035)"
```

---

## Self-Review

**Spec coverage:**
- Two-level IA / subtabs → Tasks 1, 2. ✓
- Explore: drop Topic mass, prevalence re-sorts → Task 3. ✓
- Compare + Difference pane (cell→pair, relevance-delta, λ only, NA pairs) → Tasks 4, 5, 6. ✓
- Faithful prefix-conditioning (posterior, Laplace, empty=prior) → Task 8; wiring/unification → Task 9. ✓
- PD guard (group-Σ throw) → Task 7. ✓
- ADR → Task 10. ✓
- Correlation heatmap rendering + prevalenceReader unchanged (only cell-click pair mode added). ✓
- No new export fields; spark-vi untouched. ✓

**Type consistency:** `sampleRecordPosterior` arg object identical in Tasks 8/9; `comparePair: {a,b}|null` set in Task 5, read in Task 6; `RankedDelta`/`topDifferentialCodes` signature identical in Tasks 4/6; `choleskyPD`/`invSPD`/`solveSPD` defined in Task 7, consumed in Task 8; `PhenotypeSortKey` loses `topic_mass` in Task 3 (all `topic_mass` uses removed in the same task); `CohortConditioning` gains `prefixCounts?`/`beta?` in Task 9 matching `sampleRecordPosterior`.

**Placeholder scan:** numeric/logic tasks carry complete code + tests; Svelte-restructure tasks carry exact files, the specific markup/line edits, component test code where applicable, and explicit build/type-check/manual verify gates. `Compare.svelte` is created as a placeholder in Task 2 and filled in Task 6 (noted in both).

**Executor notes:** run all FE commands from `dashboard/`. The shared `test-fixtures.ts` (introduced in Task 3) is extended in Tasks 8/9 — keep its exports additive. Pre-existing svelte-check baseline is 4 errors / 2 warnings unrelated to this work; "green" means no *new* errors.
