# Dashboard Reorganization + Faithful STM Record-Completion — Design

**Date:** 2026-07-03
**Status:** Approved (brainstorm), pending implementation plan
**Scope:** `dashboard/` (Svelte front-end) only. No spark-vi change. No new export fields.

## Motivation

Two threads converge here:

1. **Higher-level app organization.** The dashboard's three flat tabs (Phenotype
   Atlas / Patient Atlas / Simulator) don't reflect how the pieces relate. The
   bubble browser and the correlation view are two ways of looking at the
   phenotype space; the Simulator and the Patient Atlas are two halves of one
   generative story (make a cohort, then explore it).

2. **The Simulator "mess."** On a gated STM bundle at "Background only / Corpus
   average" with a few starting conditions, the Simulator's per-sample view is a
   structureless rainbow and the mean phenotype mix is nearly flat. Diagnosis (see
   Appendix A): the STM path in [runSamples.ts:44-53](../../../dashboard/src/lib/simulator/runSamples.ts#L44-L53)
   **discards the prefix** — it generates each sample from an independent
   logistic-normal *prior* draw and never conditions on the observed starting
   conditions. The prefix-conditioning the user expects only runs on the non-STM
   (Dirichlet) path. The fix is to do STM prefix-conditioning *properly* — as a
   posterior over the observed codes — which also unifies the Simulator with the
   Patient Atlas.

The logistic-normal *forward* sampler (`sampleConditionedTheta`, hand-rolled
cholesky/mvnDraw, per-panel conditioning stores, the export fields
`reference_topic`/`group_proportions`/`group_labels`, ADR 0035) is **already
built and deployed** — the `2026-07-01-dashboard-conditioning-completion.md` plan
is complete. This design builds on that; it does not rebuild it.

## Goals

- Reorganize into two top-level tabs, each with two subtabs, deep-linkable.
- A new **Compare** subtab: correlation heatmap + a **Phenotype Difference** pane
  driven by clicking a heatmap cell.
- **Explore** subtab refinements: drop the browse-table `Topic mass` column;
  prevalence reacts to the Patient Features conditioning and re-sorts.
- **Faithful STM prefix-conditioning**: record completion becomes a real
  posterior over η given the observed prefix codes, seeded from the
  covariate/group-conditioned prior. Empty prefix reduces to the prior cohort draw.
- **Unify** Simulate Cohort and the Patient Atlas onto one generation path; the
  cohort Simulate produces flows into Explore Cohort.
- Fix the **non-positive-definite Σ** sub-block so any group draw is well-posed.

## Non-goals

- No spark-vi change; no new export fields (bundles already carry everything).
- The four-quadrant display reader (`prevalenceReader`) and the correlation
  heatmap rendering are shipped and unchanged (Compare adds the sidebar and the
  cell→pair selection, not a heatmap rewrite).
- The unit-diagonal-correlation-as-Σ scale question (Appendix A, mechanism 2) is
  **out of scope**: the conditioned draw is adequately structured (~29% top,
  ~15 effective topics); it was not the cause of the mess.

## Global constraints

- spark-vi stays domain-agnostic — dashboard only.
- No LaTeX in prose/comments/docstrings: Unicode Greek (Σ η θ μ Γ λ) + plain text;
  write `E(β)` not `E[β]`.
- Cite any literature-derived method/default in docstrings (relevance: Sievert &
  Shirley 2014; logistic-normal: Blei & Lafferty 2007; nearest-PD: Higham 2002).
- Markdown-linkable code refs in prose.
- TDD throughout: vitest for pure helpers; watch each test fail first.
- Hand-rolled numerics — no new npm dependency.

---

## Part 1 — Information architecture

### Routing

Today `dashboard/src/lib/router.ts` holds a flat `TABS` list and a `route` store;
`App.svelte` maps `route`→component; `Tabs.svelte` renders the nav. Move to a
**two-level hash** `#/<top>/<sub>`:

- `#/atlas/explore`, `#/atlas/compare`
- `#/sim/simulate`, `#/sim/explore`

`router.ts` parses `top` and `sub` from the hash (with sensible defaults:
`atlas/explore`), exposes `topRoute` and `subRoute` stores, and a `go(top, sub)`.
Legacy single-segment hashes redirect to their new home (`#atlas`→`#/atlas/explore`,
`#patient`→`#/sim/explore`, `#simulator`→`#/sim/simulate`).

### Components

- `Tabs.svelte` renders the two top-level tabs (Phenotype Atlas, Simulator).
- A new **`SubTabs.svelte`** renders the second level for the active top tab —
  generalized from the existing Bubbles/Correlations segmented control in
  `tabs/Atlas.svelte` (its `.viz-switch` markup/CSS is the template).
- Two thin top-level container components decide which subtab component to show:
  - `tabs/PhenotypeAtlas.svelte` → `Explore` | `Compare`
  - `tabs/Simulator*` grouping → `SimulateCohort` | `ExploreCohort`

The existing `tabs/Atlas.svelte`, `tabs/Patient.svelte`, `tabs/Simulator.svelte`
become the subtab bodies (renamed/retargeted as needed): Explore ← Atlas's Bubbles
view; Compare ← Atlas's Correlations view (plus the new sidebar); Simulate Cohort ←
Simulator; Explore Cohort ← Patient.

### Explore subtab refinements

- **Drop the `Topic mass` column** from `atlas/PhenotypeBrowser.svelte`
  (the `topic_mass` sort key and its advanced-mode column) — it does not react to
  conditioning and clutters the table. Keep `Coherence`/`Quality` under advanced.
- **Prevalence reacts + re-sorts.** `prevalenceReader` already recomputes under
  `atlasConditioning` (covariate-softmax vs corpus-average). Ensure the table's
  rows are a reactive derivation of `prevalenceReader` so that when the Patient
  Features drawer changes, prevalence bars update and — when the active sort is
  `prevalence` — the row order re-sorts. This is a reactivity check, likely small.

---

## Part 2 — Compare subtab + Phenotype Difference pane

### Layout

Mirror the Explore skeleton: heatmap in the main column, a right sidebar. The
sidebar hosts the new **`atlas/DifferencePane.svelte`**. No browse table.

### Pair selection

`CorrelationHeatmap.svelte` currently selects a single topic on cell click
(`selectedPhenotypeId`). In Compare, clicking cell (i,j) sets a **pair**:
A = row phenotype (`order[mr]`), B = column phenotype (`order[mc]`). Introduce a
small store `comparePair: { a: number; b: number } | null` (or two ids). The
heatmap highlights the clicked cell and its row/column headers. Diagonal
(A=B) clears/ignores. Clicking is enabled for every cell, including NA-correlation
cross-group cells (the difference is a β contrast, defined regardless).

### The difference metric

For each vocabulary term w, using the model's β and corpus marginal p (same inputs
the CodePanel already uses), define the per-phenotype **relevance** (Sievert &
Shirley 2014, LDAvis): `relevance(w|k) = λ·log β_{k,w} + (1−λ)·log(β_{k,w}/p_w)`.
The Difference pane ranks by

```
delta(w) = relevance(w | A) − relevance(w | B)
```

Top-N by descending delta = conditions that make **A** distinctive relative to B;
top-N by ascending delta = conditions distinctive of **B**. The pane shows both
directions (A-side and B-side lists), labeled with A and B phenotype names.

Reuse `relevance` from [inference.ts:71-75](../../../dashboard/src/lib/inference.ts#L71-L75).
Handle β=0 (log −∞) by excluding those terms from the side where they vanish.
The only control is the **λ lift/frequency slider** (default 0.6), local to the
pane. No conditioning controls (β does not depend on covariates).

New pure helper `atlas/difference.ts` `topDifferentialCodes({ betaA, betaB, pw, lambda, n })`
returning `{ aSide: RankedDelta[]; bSide: RankedDelta[] }`, unit-tested.

---

## Part 3 — Faithful STM record-completion + unification

### The two conditioning operations

- **Covariates/group** condition the *prior*: η ~ Normal(μ, Σ), μ = Γᵀx over the
  allowed free topics, Σ the correlation sub-block, reference pinned η=0,
  out-of-group foreground topics masked. θ = softmax(η). (This is the existing
  `sampleConditionedTheta`.)
- **Observed codes (the prefix)** condition via the *posterior*. Under STM this is
  a logistic-normal posterior over η given the codes — not the Dirichlet E-step.

### Posterior given the prefix

For prefix codes D = {w₁…w_M}:

```
p(η | D, x) ∝ Normal(η; μ, Σ) · ∏_{w∈D} ( Σ_k softmax(η)_k · β_{k,w} )
```

Inference (new module `conditioning/recordPosterior.ts`):

1. **Mode.** Maximize `log p(η | D, x)` over the allowed free η. Gradient:
   `∇ = −Σ⁻¹(η − μ) + Σ_w (φ_w − θ)`, where φ_{w,k} ∝ θ_k β_{k,w} is the per-code
   topic responsibility and θ = softmax(η). The Gaussian prior term is strongly
   concave and regularizes the (non-concave) likelihood; use Newton steps with the
   prior Hessian `−Σ⁻¹` dominating, with backtracking / a gradient-ascent fallback.
   Cite Blei & Lafferty 2007 (variational logistic-normal) for the objective.
2. **Laplace covariance.** At the mode, the negative Hessian
   `H = Σ⁻¹ + Σ_w (diag(φ_w) − φ_w φ_wᵀ)` (per-code multinomial curvature) gives a
   Gaussian approximation Normal(η*, H⁻¹) to the posterior.
3. **Draw.** Each simulated patient draws η ~ Normal(η*, H⁻¹) via the existing
   `mvnDraw`(η*, cholesky(H⁻¹)); θ = softmax(η); complete ~Poisson(mean codes).

`sampleRecordPosterior({ effects, x, correlation, topicBlocks, group, prefixCounts, beta, rng })`
returns one θ draw; **empty prefix ⇒ H = Σ⁻¹ ⇒ posterior = prior**, i.e. it
reduces exactly to `sampleConditionedTheta`. This is the invariant that keeps the
cohort path and the completion path a single code path.

### PD safeguard (bug fix)

The full-topic Σ sub-block (any real group selected → background ∪ group) is not
positive-definite; `cholesky` throws (only the background-only block is PD today).
Add a nearest-PD guard applied to any Σ sub-block before factoring: symmetrize,
then either clip eigenvalues to a small floor or diagonally load
(cite Higham 2002, nearest correlation/PD matrix). Applied in both
`sampleConditionedTheta` and `sampleRecordPosterior`.

### Unification

- Simulate Cohort and Explore Cohort share one generation entry point. Simulate
  Cohort's controls (starting conditions / prefix, covariates/group via the
  conditioning bar, N patients) generate a cohort via `sampleRecordPosterior`
  (prefix present) or the equivalent prior draw (empty prefix), write it to the
  shared `cohort` store together with the generation params, and render the preview
  visuals (StructurePlot, PredictedRecord, SimMiniMap).
- Explore Cohort (the Patient Atlas) reads that `cohort`. Its previous standalone
  "Regenerate" is subsumed by Simulate Cohort (or becomes a shortcut back to it).
  `cohort.ts` `generateCohort` gains the prefix path so a single call produces the
  shared cohort.
- Non-STM (LDA/HDP) bundles keep the Dirichlet prefix-posterior E-step unchanged.

### Data flow

```
SimulateCohort controls ─┐
  (prefix, covariates,    │  sampleRecordPosterior / generateCohort
   group, N)              ├─────────────────────────────────────────▶ cohort store
                          │                                              │
  preview (structure,  ◀──┘                                              ▼
   predicted record)                                          ExploreCohort (Patient Atlas)
```

---

## Testing (TDD)

Pure helpers (vitest):

- `difference.ts`: delta ranking is antisymmetric (swap A/B negates deltas and
  swaps the side lists); λ=1 ranks by log β, λ=0 by lift; a term with β=0 in B
  still ranks on the A side; NA-correlation pair (valid β) still produces a ranking.
- `recordPosterior.ts`:
  - **Empty prefix reproduces the prior draw** — sample statistics of
    `sampleRecordPosterior(prefix={})` match `sampleConditionedTheta` (same seed
    family) within tolerance.
  - A prefix whose codes load only topic k's β drives the posterior θ to
    concentrate on k (top-topic mass rises vs the prior).
  - A covariate effect shifts the mode (mean θ moves toward the up-weighted topic).
  - Masked (out-of-group) topics stay exactly 0 in every draw.
  - Mode-finder converges (gradient norm → 0) and the Laplace H is PD.
- PD guard: a constructed non-PD symmetric input yields a usable lower factor;
  `sampleConditionedTheta` on the cancer group no longer throws.
- Router/subtabs (component tests): a two-level hash resolves to the right subtab;
  legacy hashes redirect; Simulate→Explore cohort handoff (generate writes the
  store, Explore reads it).

Build/type gates: `npx vitest run`, `npx svelte-check --threshold error`,
`npx vite build` all green.

---

## File map

New:
- `dashboard/src/lib/SubTabs.svelte`
- `dashboard/src/lib/tabs/PhenotypeAtlas.svelte` (top container: Explore|Compare)
- `dashboard/src/lib/tabs/SimulatorGroup.svelte` (top container: Simulate|Explore)
- `dashboard/src/lib/atlas/DifferencePane.svelte`
- `dashboard/src/lib/atlas/difference.ts` (+ `.test.ts`)
- `dashboard/src/lib/conditioning/recordPosterior.ts` (+ `.test.ts`)

Modified:
- `dashboard/src/lib/router.ts`, `dashboard/src/App.svelte`, `dashboard/src/lib/Tabs.svelte`
- `dashboard/src/lib/tabs/Atlas.svelte` → Explore body; `tabs/Patient.svelte` →
  Explore Cohort; `tabs/Simulator.svelte` → Simulate Cohort
- `dashboard/src/lib/atlas/CorrelationHeatmap.svelte` (cell→pair selection + highlight)
- `dashboard/src/lib/atlas/PhenotypeBrowser.svelte` (drop Topic mass; prevalence re-sort)
- `dashboard/src/lib/conditioning/logisticNormal.ts` (PD guard shared)
- `dashboard/src/lib/simulator/runSamples.ts` (STM path → posterior draw) and/or
  `dashboard/src/lib/cohort.ts` (`generateCohort` prefix path)
- `dashboard/src/lib/store.ts` (`comparePair`; shared cohort handoff)

Docs:
- New ADR: STM record-completion via logistic-normal posterior (extends ADR 0035
  from forward-only to prefix-conditioned; records the Laplace choice and the PD
  guard).

## Sequencing

Parts 1–2 are pure UI and can land first (they don't depend on Part 3). Part 3 is
the numerics-heavy piece. The implementation plan should order them so each ships
independently: (1) routing + subtabs + Explore tweaks, (2) Compare + Difference
pane, (3) PD guard + record posterior + unification.

---

## Appendix A — Diagnosis of the Simulator "mess"

Investigated against the real `population_cancer` bundle (K=60, 40 background +
20 cancer topics, reference topic 0, Σ = unit-diagonal correlation).

- **The conditioned draw is fine.** Monte-Carlo of `sampleConditionedTheta` at
  "Background only / neutral covariates": mean top-topic ≈ 29%, ≈ 15 effective
  topics of 40. Structured, not a rainbow.
- **The E-step does not inflate entropy.** Re-inferring θ from codes generated by a
  draw *concentrates* slightly (≈ 15 → ≈ 10 effective topics). So neither the draw
  nor the re-inference is the rainbow.
- **The rainbow is across-draw variation with the prefix ignored.** With
  conditioning on, [runSamples.ts:44-53](../../../dashboard/src/lib/simulator/runSamples.ts#L44-L53)
  sets `genTheta = conditionedTheta()` — an independent prior draw — and discards
  the prefix E-step. Each of the 200 samples peaks on a *different* background
  topic (18 distinct dominant topics across 200 draws), so the mean phenotype mix
  is flat and the per-sample strip is rainbow noise. The 3 prefix codes are ~6% of
  the ~47-code E-step used only for reporting, so they barely register.
- **Contrast.** The Patient Atlas is *supposed* to be independent prior draws (a
  cohort); its UMAP cloud + group coloring shows that structure well. The Simulator
  promises "complete THIS patient" but delivers 200 unconditioned draws — hence the
  perceived mess.
- **Separate real bug.** The full-topic Σ sub-block is not positive-definite;
  `cholesky` throws for a cancer-group draw. Fixed by the Part 3 PD guard.

Mechanism (2), Σ being the unit-diagonal correlation rather than the full-scale
covariance, is real but not the cause here and is out of scope.
