# Cross-tab conditioning context (dashboard) — design

## Overview

The dashboard's covariate/group controls currently live in a panel stacked
below the Phenotype Atlas's bubble map, are Atlas-only, and only act when
"covariate mode" is on. This design promotes them to a single **shared
conditioning context** rendered as a full-width bar below the tab navigation,
and defines how all three tabs (Phenotype Atlas, Patient Atlas, Simulator)
read that context.

The unifying idea: covariates (age, sex, ...) plus the gating group form the
context the model is *viewed under*. Every tab is a view of the model under
that context — the Phenotype Atlas shows the expected prevalence (the
marginal), the Patient Atlas shows a sampled cohort, the Simulator shows a
single sampled record. What differs per tab is only *when* the context is
applied.

This is a dashboard-front-end design plus one small export addition. It does
not change the model, the fit drivers, or the gating math.

## Goals

- One shared, schema-driven conditioning context, set once and reflected
  across tabs.
- A non-intrusive home: a slim persistent toggle that expands to a full-width
  control strip below the tabs, above the tab content.
- Honest per-tab application that respects cost: live where recompute is cheap
  (Phenotype Atlas, Simulator), on explicit regenerate where it is not
  (Patient Atlas).
- Schema-driven rendering: a model with different covariates, no covariates, or
  no gating shows only the controls it actually supports.
- A read-only "population reference" readout of the model's covariate marginals
  (continuous percentiles + categorical proportions).

## Non-goals (deferred)

- Color a Patient-Atlas cohort by an arbitrary categorical covariate (only
  color-by-group is in scope). Cool, but not worth the complexity yet.
- Complex covariate interactions in the marginal sampler — independent marginal
  sampling is explicitly accepted as "good enough."
- Any change to the fit drivers, gating engine, or eval.

## Two orthogonal axes

Prevalence covariates and group gating are independent model features and must
be handled independently:

- **Prevalence covariates** — the bundle carries `covariate_schema` +
  `covariate_effects` (the per-covariate Gamma rows). STM fits produce them;
  LDA does not.
- **Group gating** — the bundle carries `gating.json` (per-topic block labels +
  groups). A model may be gated whether or not it has covariates: gated STM has
  both; a future gated LDA would have gating but no covariates.

Four quadrants, all first-class:

| | no gating | gating |
|---|---|---|
| **no covariates** (e.g. LDA) | corpus prevalence (today's default) | masked corpus prevalence |
| **covariates** (STM) | covariate-predicted prevalence | masked covariate-predicted prevalence |

The bottom-left and top-right quadrants must not assume the other axis exists.
In particular, group masking is meaningful with zero covariates (select a group,
size its foreground topics by their corpus prevalence, hide the other groups'),
so it must NOT be gated behind covariate mode, and the masked path must not
require `covariate_effects`.

## The shared conditioning context

A single front-end store, replacing the current Atlas-local `covariateMode` /
`covariateValues` / `selectedGroup` trio. The two axes are independent fields,
not one master switch:

```
conditioning = {
  covariateActive: boolean,               // covariate axis on/off (covariate bundles only)
  values: Record<string, number|string>,  // schema-driven covariate values
  group: string | null,                    // gating axis; null = "Background only"
}
```

- `covariateActive` and `values` apply only when the bundle has
  `covariate_schema`/`covariate_effects`. Off = corpus-average prevalence (the
  default); on = covariate-predicted at `values`.
- `group` applies only when the bundle has `gating`; its options are
  `gating.groups` plus "Background only" (null). It is a live selection in its
  own right — selecting a group masks foreground immediately, independent of
  `covariateActive`.
- `values` keys and control types come from `covariate_schema.controls`
  (continuous slider, 2-level toggle, n-level select) — no hardcoded age/sex.
- The context persists across tab switches (module-level store state).
- On cohort/bundle change the context resets (`covariateActive=false`, `values`
  reseeded to schema defaults, `group=null`) — a new model may differ on both
  axes.

### Prevalence reader (composes the two axes)

The Phenotype Atlas reader picks a base prevalence by the covariate axis, then
applies the group mask by the gating axis:

- base = `covariateActive` ? `covariatePrevalence(effects, x)` :
  per-topic corpus prevalence (`phenotype.corpus_prevalence`).
- if `gating` and a group is selected: apply the group's allowed-topic mask.
  On the covariate base this is the existing mask-before-softmax
  (`covariatePrevalenceGated`, renormalized over allowed). On the corpus base it
  is a new `maskedCorpusPrevalence` that zeros hidden foreground topics (no
  renormalization — corpus prevalence is a per-topic display quantity, not a
  distribution; matches the existing k-anon non-renormalization rule) and needs
  no `covariate_effects`.
- if no `gating`: no mask; the base is used directly.

This makes all four quadrants well-defined, including gating-without-covariates.

## Placement

A full-width **conditioning bar** sits between the tab navigation and the tab
content (so it persists across tabs and reads as a lens on the current view).

The bar holds up to two independent sections, each shown only when its axis
exists in the bundle:

- **Group section** (iff `gating`): the group selector. Live — no separate
  on/off; selecting a group is the action.
- **Covariate section** (iff `covariate_schema` with renderable controls): an
  on/off toggle (corpus average vs covariate prevalence), the schema-driven
  sliders/selectors, and the population-reference readout.

Resting state is non-intrusive: when neither section is active the bar collapses
to a slim chip. A gating-only bundle shows just the group selector; a
covariates-only bundle shows just the covariate toggle/controls; a both-axes
bundle shows both, side by side.

The bar is global chrome (one instance, app-level), not per-tab. Tab-specific
controls that are *not* part of the shared context (see Patient Atlas below)
live inside their own tab, not in this bar.

When a bundle has neither covariates (intercept-only schema, or
`unsupported.length > 0`) nor gating, the bar is hidden entirely — there is
nothing to condition.

## Population-reference readout

A compact, read-only panel inside the bar (when expanded) showing the model's
covariate marginals:

- continuous: p5-p95 range and median (p50) — already present in
  `covariate_schema.controls[*]` as `range` + `default`.
- categorical: per-level proportions (e.g. "F 52% / M 48%").

It serves two purposes: it orients the user while setting sliders (realistic
ranges), and it answers "what are this model's cohort properties" on its own.

**Export addition (small):** `covariate_schema` categorical controls currently
carry `levels` + `reference` but drop the per-level counts. Add a k-anon-safe
`proportions` map (level -> fraction, summing to 1 over surviving levels;
levels below the small-cell threshold are already filtered upstream) to each
categorical control. This is the only server-side change and is shared by the
readout and the Phase-2/3 marginal sampler. Continuous marginals need no new
data (percentiles already present). Applies to both the local
(`analysis/local/build_dashboard.py`) and cloud
(`analysis/cloud/build_dashboard_cloud.py`) builders via the shared
`charmpheno/charmpheno/export/covariate_schema.py`.

## Per-tab application

Each axis is optional per the bundle; a tab applies only the axes its bundle
has.

| Tab | When the context applies | Notes |
|-----|--------------------------|-------|
| Phenotype Atlas | Live (cheap recompute) | reader composes the two axes (quadrant table above) |
| Simulator | Live (it re-samples on change anyway) | covariate conditioning iff covariates; group = which group to simulate iff gating |
| Patient Atlas | On explicit "Regenerate cohort" (accepts the one-time UMAP cost; never live) | own local controls (below) |

### Phenotype Atlas (Phase 1)

Reads the shared context via the composed reader: base prevalence by the
covariate axis (corpus or covariate-predicted), foreground masked by the group
axis. A covariates-only bundle uses the covariate base with no mask; a
gating-only bundle uses the corpus base with the group mask
(`maskedCorpusPrevalence`); a both-axes bundle uses the masked covariate path; a
plain bundle uses the corpus base. The absolute domain anchoring already
shipped. The old `CovariatePanel` is unmounted from the Atlas tab; its control
widgets are factored out for reuse by the conditioning bar (see file structure).

### Simulator (Phase 3)

The Simulator already recomputes its samples on input change, so it reads the
shared context live. When the bundle has covariates, each sampled record's topic
prior is conditioned on `values` (via `covariate_effects`); when it has gating,
the prior is masked to background ∪ `group` (the group whose record is
simulated). Either axis may be absent. The conditions-prefix editor composes
with whatever axes apply.

### Patient Atlas (Phase 2)

The Patient Atlas is UMAP-bound, so it never applies the context live. Instead
it owns a small local control cluster next to the existing **Regenerate cohort**
button:

- **Sample mode toggle** ("sample from distribution" vs "use set
  covariates/group"):
  - The **first** generation and any regenerate in *sample* mode draw each
    synthetic patient's covariates (and group, for gated bundles) from the
    reported marginals — a realistic, representative cohort.
  - In *use-set* mode, the next regenerate builds a this-subpopulation cohort at
    the shared context's `values` + `group`.
- **Color-by-group** toggle (gated bundles only): colors the existing cloud by
  each synthetic patient's group. Most informative on a sampled (mixed) cohort.
  Independent of the shared `group` selection.

Each synthetic patient carries its sampled/assigned group so color-by-group and
group-aware projection are possible without recomputation beyond the regenerate.

### Marginal sampler (shared by Phase 2/3)

Given a model's `covariate_schema`, sample one covariate set:

- categorical: draw a level from `proportions`.
- continuous: draw from a percentile-based approximation (e.g. a triangular
  distribution peaked at p50 with support [p5, p95]) — a marginal-only
  approximation, independent across covariates, explicitly accepted as good
  enough (no interactions).

For gated bundles, the group is drawn from the model's group proportions
(carried in `gating` / derivable from the per-group counts already used for
k-anon; if not present, fall back to uniform over `gating.groups` and note it).
The sampled raw values are turned into a design vector with the existing
`buildDesignVector(design_columns, values)`; conditioned topic priors reuse
`covariate_effects` (the same Gamma rows the Atlas uses), masked to the group's
allowed topics for gated bundles.

## Phasing

One spec, three implementation plans, each shippable on its own:

- **Phase 1 — shared context + global bar + Atlas (both axes):** introduce the
  shared store with independent covariate and group axes, build the conditioning
  bar (independent group section and covariate section, each shown per bundle),
  add the `maskedCorpusPrevalence` reader so the gating-only quadrant works
  without `covariate_effects`, add the `proportions` export field + population
  readout, repoint the Phenotype Atlas at the composed reader, and unmount the
  old stacked `CovariatePanel`. This delivers the user-visible relocation, the
  axis decoupling (so gated LDA / STM-no-gating render correctly), and is the
  highest-value slice.
- **Phase 2 — Patient Atlas:** marginal sampler, conditioned regenerate with
  the sample-vs-set local toggle, per-patient group assignment, color-by-group.
- **Phase 3 — Simulator:** live conditioning of sampled records on the shared
  context (covariates + group mask), composed with the conditions prefix.

## Components and file structure

Phase 1 (front-end unless noted):

- `dashboard/src/lib/store.ts` — replace `covariateMode`/`covariateValues`/
  `selectedGroup` with the shared `conditioning` store (independent
  `covariateActive` + `values` + `group`); rewrite `prevalenceReader` to compose
  the two axes across the four quadrants (keep thin backward-compatible derived
  exports if it reduces churn in unchanged readers).
- `dashboard/src/lib/covariate.ts` — add `maskedCorpusPrevalence(corpusPrev,
  topicBlocks, group)` (zero hidden foreground, no renormalization, no
  `covariate_effects`) for the gating-only quadrant; existing
  `covariatePrevalence` / `covariatePrevalenceGated` / `allowedMaskForGroup`
  unchanged.
- `dashboard/src/lib/conditioning/ConditioningBar.svelte` (new) — the bar: the
  independent group section and covariate section (each rendered per bundle),
  reusing the control widgets currently in `atlas/CovariatePanel.svelte`, plus
  the population readout.
- `dashboard/src/lib/conditioning/population.ts` (new) — pure helpers to derive
  the readout strings from `covariate_schema`.
- `dashboard/src/App.svelte` / layout — mount `ConditioningBar` between
  `Tabs` and the routed tab content.
- `dashboard/src/lib/tabs/Atlas.svelte` — remove the stacked `CovariatePanel`;
  the Atlas now reads the shared context only.
- `dashboard/src/lib/atlas/CovariatePanel.svelte` — retire or reduce to the
  reusable control widgets imported by `ConditioningBar`.
- `charmpheno/charmpheno/export/covariate_schema.py` — add `proportions` to
  categorical controls (k-anon-safe).
- `analysis/local/build_dashboard.py`, `analysis/cloud/build_dashboard_cloud.py`
  — pass the level counts through so `proportions` is emitted (both already
  compute level counts).
- `dashboard/src/lib/types.ts` — `proportions?` on the categorical control type.

Phase 2/3 add files under `dashboard/src/lib/conditioning/` (the marginal
sampler) and touch `patient/` and `simulator/` components plus their generation
helpers (`cohort.ts`, `simulator/runSamples.ts`).

## Data flow (Phase 1)

1. `loadBundle` provides `covariate_schema` (now with categorical
   `proportions`) and `gating` — either or both may be absent.
2. `ConditioningBar` renders the covariate section from `covariate_schema`
   (when present) and the group section from `gating.groups` (when present);
   writes user input into the `conditioning` store's independent axes.
3. The store's `prevalenceReader` composes the two axes (quadrant table) and
   feeds the Phenotype Atlas's `TopicMap`. For gating-only bundles it uses
   `maskedCorpusPrevalence`; for plain bundles it falls through to today's
   corpus/`fractionAboveTau` path unchanged.
4. The population readout derives its strings from `covariate_schema` via
   `population.ts`.

## Error handling and edge cases

The four quadrants and their degenerate cases:

- No covariates and no gating (plain LDA) → the bar is hidden; the Atlas uses
  today's corpus/`fractionAboveTau` reader unchanged.
- Covariates, no gating (STM) → covariate section only (toggle + sliders +
  readout); no group section.
- Gating, no covariates at all (gated LDA — no `covariate_schema`/
  `covariate_effects`) → group section only; the Atlas uses
  `maskedCorpusPrevalence`. This is the quadrant the earlier design missed.
- Gating and covariates (gated STM) → both sections; the reader masks the
  covariate-predicted prevalence.
- Gated with intercept-only covariates → group section + (degenerate) covariate
  section with no sliders; treat as gating-only for prevalence.
- `unsupported.length > 0` (a covariate the dashboard cannot render) → the
  covariate section is disabled with the existing "unavailable" note; the group
  section still works if gated.
- Bundle/cohort switch → reset the context (`covariateActive` off, `values`
  reseeded to defaults, `group` null), so a stale selection never carries into a
  different schema/gating shape.
- Missing categorical `proportions` (older bundle) → the readout omits
  proportions for that covariate and the sampler falls back to uniform, logging
  the fallback; nothing throws.

## Testing strategy

- Pure helpers (`population.ts`, the marginal sampler) are unit-tested with
  vitest: percentile/proportion formatting; sampler draws respect proportions
  and percentile support; deterministic under a seeded RNG.
- Reader quadrants: `prevalenceReader` produces the right value in all four
  quadrants — corpus (plain), covariate-predicted (covariates-only), masked
  corpus (gating-only, asserting it never touches `covariate_effects`), and
  masked covariate-predicted (both). `maskedCorpusPrevalence` is unit-tested
  directly (hidden foreground -> 0, background + selected group preserved, no
  renormalization). The existing `store.covariate.test.ts` continues to pass,
  repointed at the shared store.
- Schema-driven rendering: ConditioningBar shows the right sections for each
  bundle shape (covariates-only, gating-only, both, neither) — component or
  build-level checks plus the existing FE suite + `npm run build`.
- Export: `covariate_schema` emits k-anon-safe `proportions`; covered by the
  charmpheno export tests and the local gated integration test.
- Backward compatibility: a non-gated, no-covariate bundle (e.g. an LDA cohort)
  renders with no bar and is otherwise unchanged.

## Out of scope

Color-by arbitrary categorical covariate on the Patient Atlas; covariate
interactions in the sampler; any model/fit/eval change.
