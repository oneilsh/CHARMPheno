# Contributing-Codes Phenotype-Composition View — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Execution is BLOCKED** until the concurrent `stm`-branch thread lands — see the design doc's status. `stm` auto-pushes, so do not start until that thread is clear. The commit steps below apply at execution time.

**Goal:** Turn each code in the patient panel's "top contributing codes" list into a normalized, always-on phenotype-composition bar (per-code φ attribution), with band-selection focus and a prior-vs-evidence residual overlay on the profile bar.

**Architecture:** All logic lives in a new pure, Svelte-free module (`codeComposition.ts`) that is fully unit-tested; the `ContributingCodes.svelte` and `ProfileBar.svelte` components become thin renderers over it. β and θ are already client-side in `$bundle.model` — no model, export, or data-format changes.

**Tech Stack:** Svelte + TypeScript, Vitest, @testing-library/svelte.

**Design:** [docs/superpowers/specs/2026-07-06-contributing-codes-phenotype-composition-design.md](../specs/2026-07-06-contributing-codes-phenotype-composition-design.md)

## Global Constraints

- No model / export / data-format changes; read β, θ, K from `$bundle.model` only.
- Phenotype→hue is the shared `phenotypeHue` derived store; a phenotype is the same color here, in the profile bar, and in the atlas.
- Tail bucketing uses `OTHER_THRESHOLD = 0.05` and the `-1` "Other" sentinel, matching `ProfileBar` and today's `ContributingCodes`.
- Per-code bars are **normalized** (each sums to 1); they do not sum to θ.
- Interaction is **always-on + focus**: render every code without a selection; a selection sorts + emphasizes, never hides codes.
- The residual "evidence vs. prior" split is a heuristic under STM (θ = softmax(η̂) is non-additive); label it as approximate, not an exact decomposition.
- Test runner: `npm test` (`vitest run`) from `dashboard/`.

---

### Task 1: `codeComposition()` — per-code φ split with Other bucketing

**Files:**
- Create: `dashboard/src/lib/patient/codeComposition.ts`
- Test: `dashboard/src/lib/patient/codeComposition.test.ts`

**Interfaces:**
- Consumes: `Model.beta` (`number[][]`), `theta` (`number[]`), `codeBag` (`number[]`), `K` (`number`).
- Produces: `OTHER_ID = -1`; `interface PhenotypeSegment { k: number; weight: number }`; `interface CodeRow { w: number; count: number; segments: PhenotypeSegment[] }`; `codeComposition(theta, codeBag, beta, K, otherThreshold?=0.05): CodeRow[]`.

- [ ] **Step 1: Write the failing test**

```ts
import { describe, it, expect } from 'vitest'
import { codeComposition, OTHER_ID } from './codeComposition'

// K=3, V=3. Topic 0 emits code 0, topic 1 emits code 1, topic 2 emits code 2.
const beta = [
  [0.8, 0.1, 0.1],
  [0.1, 0.8, 0.1],
  [0.1, 0.1, 0.8],
]

describe('codeComposition', () => {
  it('each code row segments sum to 1', () => {
    const theta = [0.5, 0.4, 0.1]
    const rows = codeComposition(theta, [0, 0, 1], beta, 3, 0.05)
    for (const row of rows) {
      const s = row.segments.reduce((a, b) => a + b.weight, 0)
      expect(s).toBeCloseTo(1, 10)
    }
  })

  it('counts repeated codes', () => {
    const rows = codeComposition([0.5, 0.4, 0.1], [0, 0, 1], beta, 3, 0.05)
    expect(rows.find((r) => r.w === 0)!.count).toBe(2)
    expect(rows.find((r) => r.w === 1)!.count).toBe(1)
  })

  it('buckets tail phenotypes (theta < threshold) into OTHER_ID', () => {
    // theta[2] = 0.02 < 0.05, so any weight on topic 2 goes to Other.
    const rows = codeComposition([0.5, 0.48, 0.02], [2], beta, 3, 0.05)
    const seg2 = rows[0].segments.find((s) => s.k === 2)
    expect(seg2).toBeUndefined()
    expect(rows[0].segments.some((s) => s.k === OTHER_ID)).toBe(true)
  })

  it('emits empty segments for a code no expressed topic can generate (z=0)', () => {
    const zeroBeta = [[0, 1], [0, 1]] // neither topic emits code 0
    const rows = codeComposition([0.5, 0.5], [0], zeroBeta, 2, 0.05)
    expect(rows[0].segments).toEqual([])
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd dashboard && npx vitest run src/lib/patient/codeComposition.test.ts`
Expected: FAIL — "Failed to resolve import './codeComposition'".

- [ ] **Step 3: Write minimal implementation**

```ts
import type { Model } from '../types'

// Sentinel phenotype id for the aggregated long-tail band, matching
// ProfileBar / ContributingCodes ($selectedPhenotypeId === -1).
export const OTHER_ID = -1

export interface PhenotypeSegment {
  k: number      // phenotype id, or OTHER_ID for the aggregated tail
  weight: number // share of this code's attribution; segments sum to 1 (or 0 if unexplained)
}

export interface CodeRow {
  w: number
  count: number
  segments: PhenotypeSegment[]
}

// Per-code posterior split phi(w,k) = theta[k]*beta[k][w] / sum_j theta[j]*beta[j][w],
// reduced to the phenotypes the profile bar shows (theta >= threshold) plus an
// aggregated Other bucket for the tail. Patient-conditioned: depends on theta.
export function codeComposition(
  theta: number[],
  codeBag: number[],
  beta: Model['beta'],
  K: number,
  otherThreshold = 0.05,
): CodeRow[] {
  const counts = new Map<number, number>()
  for (const w of codeBag) counts.set(w, (counts.get(w) ?? 0) + 1)

  const rows: CodeRow[] = []
  for (const [w, count] of counts) {
    let z = 0
    for (let j = 0; j < K; j++) z += beta[j][w] * theta[j]
    const segments: PhenotypeSegment[] = []
    let other = 0
    if (z > 0) {
      for (let j = 0; j < K; j++) {
        const weight = (beta[j][w] * theta[j]) / z
        if (weight === 0) continue
        if (theta[j] >= otherThreshold) segments.push({ k: j, weight })
        else other += weight
      }
    }
    if (other > 0) segments.push({ k: OTHER_ID, weight: other })
    rows.push({ w, count, segments })
  }
  return rows
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd dashboard && npx vitest run src/lib/patient/codeComposition.test.ts`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/lib/patient/codeComposition.ts dashboard/src/lib/patient/codeComposition.test.ts
git commit -m "feat(dashboard): codeComposition — per-code phenotype split for contributing codes"
```

---

### Task 2: `explainedVsPrior()` — aggregate evidence-vs-prior reconciliation

**Files:**
- Modify: `dashboard/src/lib/patient/codeComposition.ts`
- Test: `dashboard/src/lib/patient/codeComposition.test.ts`

**Interfaces:**
- Consumes: same inputs as Task 1.
- Produces: `explainedVsPrior(theta, codeBag, beta, K): { explained: number[]; prior: number[] }` — length-K arrays in θ units where `explained[k] + prior[k] === theta[k]`.

- [ ] **Step 1: Write the failing test**

```ts
import { explainedVsPrior } from './codeComposition'

describe('explainedVsPrior', () => {
  const beta = [
    [0.8, 0.1, 0.1],
    [0.1, 0.8, 0.1],
    [0.1, 0.1, 0.8],
  ]

  it('explained + prior equals theta per phenotype', () => {
    const theta = [0.5, 0.4, 0.1]
    const { explained, prior } = explainedVsPrior(theta, [0, 1, 2], beta, 3)
    for (let k = 0; k < 3; k++) {
      expect(explained[k] + prior[k]).toBeCloseTo(theta[k], 10)
    }
  })

  it('clamps when codes over-explain a phenotype: prior=0, explained=theta', () => {
    // Only code 0 present → code evidence concentrates on topic 0, over-explaining it
    // relative to theta[0], while theta[1] gets no code support.
    const theta = [0.34, 0.33, 0.33]
    const { explained, prior } = explainedVsPrior(theta, [0], beta, 3)
    expect(prior[0]).toBe(0)
    expect(explained[0]).toBeCloseTo(theta[0], 10)
    expect(prior[1]).toBeGreaterThan(0) // no code speaks to topic 1 → prior-supported
  })

  it('returns zero explained when codeBag is empty', () => {
    const { explained, prior } = explainedVsPrior([0.5, 0.5, 0], [], beta, 3)
    expect(explained).toEqual([0, 0, 0])
    expect(prior).toEqual([0.5, 0.5, 0])
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd dashboard && npx vitest run src/lib/patient/codeComposition.test.ts -t explainedVsPrior`
Expected: FAIL — "explainedVsPrior is not a function".

- [ ] **Step 3: Write minimal implementation** (append to `codeComposition.ts`)

```ts
// Aggregate reconciliation for the profile-bar residual overlay. theta_data(k) is the
// occurrence-weighted, normalized code evidence; the prior-supported remainder is the
// part of theta(k) the codes do not account for. explained + prior === theta per k.
//
// EXACTNESS: additive "prior + evidence" is exact only for the Dirichlet/LDA engine
// (gamma = alpha + sum_w c*phi). Under STM (softmax(eta_hat)) this is a principled
// heuristic; present it as an approximate cue, not an identity. See insight 0028.
export function explainedVsPrior(
  theta: number[],
  codeBag: number[],
  beta: Model['beta'],
  K: number,
): { explained: number[]; prior: number[] } {
  const counts = new Map<number, number>()
  for (const w of codeBag) counts.set(w, (counts.get(w) ?? 0) + 1)

  const raw = new Array(K).fill(0)
  for (const [w, c] of counts) {
    let z = 0
    for (let j = 0; j < K; j++) z += beta[j][w] * theta[j]
    if (z <= 0) continue
    for (let j = 0; j < K; j++) raw[j] += (c * beta[j][w] * theta[j]) / z
  }
  const total = raw.reduce((a, b) => a + b, 0)
  const thetaData = total > 0 ? raw.map((x) => x / total) : new Array(K).fill(0)

  const explained = new Array(K).fill(0)
  const prior = new Array(K).fill(0)
  for (let k = 0; k < K; k++) {
    explained[k] = Math.min(theta[k], thetaData[k])
    prior[k] = Math.max(0, theta[k] - thetaData[k])
  }
  return { explained, prior }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd dashboard && npx vitest run src/lib/patient/codeComposition.test.ts`
Expected: PASS (7 tests total).

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/lib/patient/codeComposition.ts dashboard/src/lib/patient/codeComposition.test.ts
git commit -m "feat(dashboard): explainedVsPrior — evidence-vs-prior reconciliation for profile residual"
```

---

### Task 3: `sortRowsForSelection()` — selection-driven focus ordering

**Files:**
- Modify: `dashboard/src/lib/patient/codeComposition.ts`
- Test: `dashboard/src/lib/patient/codeComposition.test.ts`

**Interfaces:**
- Consumes: `CodeRow[]` (Task 1), `selectedId: number | null` (a phenotype id, `OTHER_ID`, or `null`).
- Produces: `sortRowsForSelection(rows, selectedId): CodeRow[]` — new array; `null` → by count desc; an id → by that phenotype's segment weight desc.

- [ ] **Step 1: Write the failing test**

```ts
import { sortRowsForSelection } from './codeComposition'

describe('sortRowsForSelection', () => {
  const beta = [
    [0.8, 0.1, 0.1],
    [0.1, 0.8, 0.1],
    [0.1, 0.1, 0.8],
  ]

  it('null selection sorts by occurrence count desc', () => {
    const rows = codeComposition([0.4, 0.4, 0.2], [0, 1, 1], beta, 3, 0.05)
    const sorted = sortRowsForSelection(rows, null)
    expect(sorted[0].count).toBeGreaterThanOrEqual(sorted[1].count)
    expect(sorted[0].w).toBe(1) // appears twice
  })

  it('a phenotype selection sorts by that phenotype weight desc', () => {
    const rows = codeComposition([0.4, 0.4, 0.2], [0, 1], beta, 3, 0.05)
    const sorted = sortRowsForSelection(rows, 1)
    // code 1 loads mostly on topic 1, so it ranks first when topic 1 is selected
    expect(sorted[0].w).toBe(1)
  })

  it('does not mutate the input array', () => {
    const rows = codeComposition([0.5, 0.5, 0], [0, 1], beta, 3, 0.05)
    const before = rows.map((r) => r.w)
    sortRowsForSelection(rows, 0)
    expect(rows.map((r) => r.w)).toEqual(before)
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd dashboard && npx vitest run src/lib/patient/codeComposition.test.ts -t sortRowsForSelection`
Expected: FAIL — "sortRowsForSelection is not a function".

- [ ] **Step 3: Write minimal implementation** (append to `codeComposition.ts`)

```ts
// Focus ordering. No selection → by occurrence count. A selected phenotype (or OTHER_ID)
// → by that band's share of each code, so the codes that most drove the clicked band rise
// to the top. Never filters; returns a new array (input untouched).
export function sortRowsForSelection(rows: CodeRow[], selectedId: number | null): CodeRow[] {
  const weightFor = (row: CodeRow, id: number) =>
    row.segments.find((s) => s.k === id)?.weight ?? 0
  const copy = rows.slice()
  if (selectedId === null) copy.sort((a, b) => b.count - a.count)
  else copy.sort((a, b) => weightFor(b, selectedId) - weightFor(a, selectedId))
  return copy
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd dashboard && npx vitest run src/lib/patient/codeComposition.test.ts`
Expected: PASS (10 tests total).

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/lib/patient/codeComposition.ts dashboard/src/lib/patient/codeComposition.test.ts
git commit -m "feat(dashboard): sortRowsForSelection — focus ordering for contributing codes"
```

---

### Task 4: `ContributingCodes.svelte` — always-on stacked bars + focus

**Files:**
- Modify: `dashboard/src/lib/patient/ContributingCodes.svelte` (full script + template rewrite)
- Test: `dashboard/src/lib/patient/ContributingCodes.test.ts`

**Interfaces:**
- Consumes: props `theta: number[]`, `codeBag: number[]`; stores `bundle`, `selectedPhenotypeId`, `phenotypeHue`, `searchedConditionIdx`; `codeComposition`, `sortRowsForSelection`, `OTHER_ID`.
- Produces: rendered `<li>` per code (cap 12), each a normalized stacked bar of phenotype segments.

- [ ] **Step 1: Write the failing test**

```ts
import { it, expect, afterEach, beforeEach } from 'vitest'
import { render, cleanup } from '@testing-library/svelte'
import ContributingCodes from './ContributingCodes.svelte'
import { bundle, selectedPhenotypeId } from '../store'
import { makeStmBundleFixture } from '../test-fixtures'

afterEach(() => cleanup())
beforeEach(() => {
  bundle.set(makeStmBundleFixture())
  selectedPhenotypeId.set(null)
})

it('renders one row per unique code with no selection', () => {
  const b = makeStmBundleFixture()
  const theta = b.phenotypes.phenotypes.map((_, k) => (k === 0 ? 0.6 : 0.4 / (b.model.K - 1)))
  const { container } = render(ContributingCodes, { props: { theta, codeBag: [0, 0, 1] } })
  expect(container.querySelectorAll('li.code').length).toBe(2) // codes 0 and 1
  expect(container.querySelectorAll('li.code .seg').length).toBeGreaterThan(0)
})

it('still renders all codes when a phenotype is selected (focus, not filter)', () => {
  const b = makeStmBundleFixture()
  const theta = b.phenotypes.phenotypes.map((_, k) => (k === 0 ? 0.6 : 0.4 / (b.model.K - 1)))
  selectedPhenotypeId.set(0)
  const { container } = render(ContributingCodes, { props: { theta, codeBag: [0, 0, 1] } })
  expect(container.querySelectorAll('li.code').length).toBe(2)
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd dashboard && npx vitest run src/lib/patient/ContributingCodes.test.ts`
Expected: FAIL — no `li.code` / `.seg` elements (current template uses `.codes li` + single `.spark-bar`).

- [ ] **Step 3: Rewrite the component script + list template**

Replace the `<script>` reactive block (the `top`/`maxScore`/`counts` computeds) with the module-backed version, and the `<ol class="codes">` list with stacked bars. Key script:

```svelte
<script lang="ts">
  import { bundle, selectedPhenotypeId, searchedConditionIdx } from '../store'
  import { phenotypeHue } from '../palette'
  import { copy } from '../copy'
  import { codeComposition, sortRowsForSelection, OTHER_ID, type CodeRow } from './codeComposition'

  export let theta: number[]
  export let codeBag: number[]

  const MAX_ROWS = 12

  $: rows = $bundle
    ? codeComposition(theta, codeBag, $bundle.model.beta, $bundle.model.K)
    : []
  $: sorted = sortRowsForSelection(rows, $selectedPhenotypeId).slice(0, MAX_ROWS)
  $: hasSelection = $selectedPhenotypeId !== null

  // Focus: when a band is selected, dim every segment that is not it.
  function segActive(k: number): boolean {
    return !hasSelection || k === $selectedPhenotypeId
  }
  function segColor(k: number): string {
    return k === OTHER_ID ? 'var(--surface-deep)' : $phenotypeHue(k)
  }
</script>
```

Template (replace the `{#if} … {/if}` body under `<header>`):

```svelte
{#if !$bundle || rows.length === 0}
  <p class="hint">{copy.contributingCodes.hintNoCodes(null)}</p>
{:else}
  <ol class="codes">
    {#each sorted as row (row.w)}
      {@const c = $bundle.vocab.codes[row.w]}
      {@const matched = $searchedConditionIdx === row.w}
      <li class="code" class:matched>
        <span class="desc">
          {#if matched}<span class="match-dot" aria-hidden="true"></span>{/if}{c.description || c.code}
        </span>
        <span class="bar" aria-hidden="true">
          {#each row.segments as s}
            <span
              class="seg"
              class:dim={!segActive(s.k)}
              style="width: {(s.weight * 100).toFixed(2)}%; background: {segColor(s.k)}"
            ></span>
          {/each}
        </span>
        <span class="count" data-numeric>×{row.count}</span>
      </li>
    {/each}
  </ol>
{/if}
```

Add styles (replace `.spark`/`.spark-bar` rules):

```svelte
  .codes li.code {
    display: grid;
    grid-template-columns: 1fr 8rem 2.5rem;
    align-items: center;
    gap: 0.85rem;
    padding: 0.45rem 0;
    border-bottom: 1px solid var(--rule-faint);
    font-size: var(--fs-small);
  }
  .bar {
    display: flex;
    height: 8px;
    border-radius: 2px;
    overflow: hidden;
    background: var(--surface-recessed);
  }
  .seg {
    height: 100%;
    /* 2px surface gap between fills, per dataviz mark spec */
    box-shadow: inset -2px 0 0 var(--surface);
    transition: opacity 0.15s ease;
  }
  .seg.dim { opacity: 0.2; }
  .seg:last-child { box-shadow: none; }
```

Also delete the now-unused `phenotypesById` import, the `OTHER_THRESHOLD`/`isOther`/`counts`/`top`/`maxScore` computeds, and the `$selectedPhenotypeId === null` "click a band" empty state (the panel now always renders). Keep the header, `open in atlas` button, and the `link-dot`/Other-hatch styles.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd dashboard && npx vitest run src/lib/patient/ContributingCodes.test.ts`
Expected: PASS (2 tests).

- [ ] **Step 5: Type-check and commit**

Run: `cd dashboard && npm run check`
Expected: no new errors in `ContributingCodes.svelte`.

```bash
git add dashboard/src/lib/patient/ContributingCodes.svelte dashboard/src/lib/patient/ContributingCodes.test.ts
git commit -m "feat(dashboard): always-on phenotype-composition bars in contributing codes"
```

---

### Task 5: `ProfileBar.svelte` — evidence-vs-prior residual overlay

**Files:**
- Modify: `dashboard/src/lib/patient/ProfileBar.svelte`
- Test: `dashboard/src/lib/patient/ProfileBar.test.ts`

**Interfaces:**
- Consumes: prop `theta: number[]`; needs `codeBag: number[]` (new prop, threaded from `Patient.svelte`); `explainedVsPrior`.
- Produces: each main band split into a solid explained portion + a hatched prior portion.

- [ ] **Step 1: Thread the new prop from the parent**

In `dashboard/src/lib/tabs/Patient.svelte:199`, pass the code bag to the profile bar (it already passes `theta`):

```svelte
<ProfileBar theta={current.theta} codeBag={current.code_bag} />
```

- [ ] **Step 2: Write the failing test**

```ts
import { it, expect, afterEach, beforeEach } from 'vitest'
import { render, cleanup } from '@testing-library/svelte'
import ProfileBar from './ProfileBar.svelte'
import { bundle } from '../store'
import { makeStmBundleFixture } from '../test-fixtures'

afterEach(() => cleanup())
beforeEach(() => bundle.set(makeStmBundleFixture()))

it('renders a prior sub-segment on bands the codes do not fully explain', () => {
  const b = makeStmBundleFixture()
  const theta = b.phenotypes.phenotypes.map((_, k) => (k === 0 ? 0.6 : 0.4 / (b.model.K - 1)))
  // codeBag speaks only to code 0 → other bands lean on the prior
  const { container } = render(ProfileBar, { props: { theta, codeBag: [0] } })
  expect(container.querySelectorAll('.band .prior-fill').length).toBeGreaterThan(0)
})
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd dashboard && npx vitest run src/lib/patient/ProfileBar.test.ts`
Expected: FAIL — no `.prior-fill` elements.

- [ ] **Step 4: Implement the overlay**

Add to the `<script>`:

```svelte
  export let codeBag: number[] = []
  import { explainedVsPrior } from './codeComposition'

  $: split = $bundle
    ? explainedVsPrior(theta, codeBag, $bundle.model.beta, $bundle.model.K)
    : { explained: theta.map(() => 0), prior: theta.slice() }
  // Fraction of each band that is prior-supported (0..1 within the band).
  function priorFrac(k: number): number {
    return theta[k] > 0 ? split.prior[k] / theta[k] : 0
  }
```

In the main-band button (`ProfileBar.svelte:70-72`), overlay a hatched fill sized to the prior fraction on top of the solid hue:

```svelte
<button class="band"
  style="width: {(b.v * 100).toFixed(2)}%; background: {$phenotypeHue(b.k)};"
  ...>
  <span class="prior-fill" style="width: {(priorFrac(b.k) * 100).toFixed(2)}%"></span>
</button>
```

Add the hatched style (reuse the Other gradient idiom already in the file):

```svelte
  .band { position: relative; }
  .prior-fill {
    position: absolute;
    top: 0; right: 0; height: 100%;
    background-image: repeating-linear-gradient(
      45deg, transparent, transparent 2px,
      rgba(255, 255, 255, 0.45) 2px, rgba(255, 255, 255, 0.45) 3px
    );
    pointer-events: none;
  }
```

Add a caption/tooltip near the bar noting the split is an approximate "evidence vs. prior" cue (add a string to `copy.ts` `contributingCodes` and render it), per the Global Constraints heuristic-labeling rule.

- [ ] **Step 5: Run test to verify it passes**

Run: `cd dashboard && npx vitest run src/lib/patient/ProfileBar.test.ts`
Expected: PASS (1 test).

- [ ] **Step 6: Type-check, full test, commit**

Run: `cd dashboard && npm run check && npm test`
Expected: no new type errors; all tests pass.

```bash
git add dashboard/src/lib/patient/ProfileBar.svelte dashboard/src/lib/patient/ProfileBar.test.ts dashboard/src/lib/tabs/Patient.svelte dashboard/src/lib/copy.ts
git commit -m "feat(dashboard): evidence-vs-prior residual overlay on patient profile bar"
```

---

## Self-Review

**Spec coverage:**
- φ full split → Task 1 (`codeComposition`). ✓
- Always-on render + selection focus/sort/desaturate → Task 4 (`sortRowsForSelection` + `.seg.dim`). ✓
- Normalized bars → Task 4 (segment widths = φ weight, sum to 1). ✓
- Prior-residual overlay + STM heuristic labeling → Tasks 2 & 5. ✓
- Reuse `phenotypeHue`, `OTHER_THRESHOLD`/`-1`, Other hatch → Tasks 1, 4, 5. ✓
- No export/model change → all logic reads `$bundle.model`. ✓
- Deferred (specificity sort, ground-truth-z, weighted toggle) → intentionally absent. ✓

**Placeholder scan:** none — every code/test step has complete content.

**Type consistency:** `CodeRow` / `PhenotypeSegment` / `OTHER_ID` defined in Task 1 and consumed verbatim in Tasks 3–4; `explainedVsPrior` return shape `{ explained, prior }` defined in Task 2 and consumed in Task 5; `codeBag` prop added to `ProfileBar` in Task 5 Step 1 and used in Step 4.

**Note for executor:** the component tests assume `makeStmBundleFixture()` exposes `model.beta`, `model.K`, `vocab.codes`, and `phenotypes.phenotypes`. If the fixture's phenotype count or vocab differs, adjust the `theta`/`codeBag` literals to match its K and V — the assertions (row counts, presence of `.prior-fill`) hold regardless of exact values.
