# Cross-tab conditioning context (dashboard) — design

> **Revised 2026-06-25.** The original design placed the controls in a single
> global full-width bar below the tab navigation (shipped as Phase 1, commits
> `1dc025c`..`54610ca`). In use that proved unintuitive: the conditioning
> *state* is global but its *effect* differs per panel (live prevalence on the
> Phenotype Atlas, generative sampling on the Patient Atlas and Simulator), and a
> single global control surfaced that relationship only in hovertext. This
> revision moves the controls to **per-panel** clusters with **per-panel state**,
> clarifies the Patient Atlas (it is a synthetic, client-regenerable cohort that
> re-fits its UMAP projection on each regenerate — conditioning applies on
> regenerate, where that refit already happens), and unifies the generative panels
> on a shared conditioned prior (see ADR 0028). The internal axis/quadrant work
> from Phase 1 (the store, the four-quadrant reader, `maskGroupPrevalence`, the
> `proportions` export, the population readout) is kept as-is.

## Overview

The dashboard's covariate/group controls began as a panel stacked below the
Phenotype Atlas's bubble map, Atlas-only, acting only in "covariate mode." This
design generalizes them into a **per-panel conditioning context**: each tab
(Phenotype Atlas, Patient Atlas, Simulator) renders the conditioning controls its
bundle supports, framed by what that panel does with them, and reads the context
through shared pure helpers.

The unifying idea: prevalence covariates (age, sex, ...) and the gating group are
two independent lenses the model is *viewed under*. Every tab is a view of the
model under whichever lenses its bundle has — the Phenotype Atlas shows the
expected prevalence (the marginal), the Patient Atlas shows a sampled cohort, the
Simulator shows a single sampled record. What differs per tab is *when* the
context is applied and *what* it drives (a display reader vs a generative prior);
the two axes are otherwise handled independently (see "Two orthogonal axes").

This is a dashboard-front-end design plus two small export additions (categorical
`proportions`; group labels and `group_proportions`). It does not change the
model, the fit drivers, or the gating math.

## Goals

- A per-panel conditioning context: each panel owns its controls and its own
  state, set in the context of the panel you are looking at.
- Honest per-panel application: live where recompute is cheap (Phenotype Atlas,
  Simulator), on explicit regenerate where re-rolling a cohort is the natural
  action (Patient Atlas).
- One shared definition of the axis logic (the four quadrants) driving both the
  display reader and the generative prior, so display and generation never
  diverge.
- Schema-driven rendering: a model with different covariates, no covariates, or
  no gating shows only the controls it actually supports.
- A read-only "population reference" readout of the model's covariate marginals
  (continuous percentiles + categorical proportions).
- Human-readable group labels sourced from the export, consistent with the
  existing `{id, label}` metadata convention.

## Non-goals (deferred)

- Color a Patient-Atlas cohort by an arbitrary categorical covariate (only
  color-by-group is in scope).
- Complex covariate interactions in the marginal sampler — independent marginal
  sampling is explicitly accepted as "good enough."
- A logistic-normal sampler faithful to STM's true prior (ADR 0028 alternative B);
  the Dirichlet mean-match is the accepted approximation for forward
  visualization.
- Any change to the fit drivers, gating engine, or eval.

## Two orthogonal axes

Prevalence covariates and group gating are independent model features and must be
handled independently:

- **Prevalence covariates** — the bundle carries `covariate_schema` +
  `covariate_effects` (the per-covariate Gamma rows). STM fits produce them; LDA
  does not.
- **Group gating** — the bundle carries `gating.json` (per-topic block labels +
  groups). A model may be gated whether or not it has covariates: gated STM has
  both; gated LDA has gating but no covariates.

Four quadrants, all first-class:

| | no gating | gating |
|---|---|---|
| **no covariates** (e.g. LDA) | corpus prevalence (today's default) | masked corpus prevalence |
| **covariates** (STM) | covariate-predicted prevalence | masked covariate-predicted prevalence |

The bottom-left and top-right quadrants must not assume the other axis exists. In
particular, group masking is meaningful with zero covariates (select a group,
size its foreground topics by their corpus prevalence, hide the other groups'), so
it must NOT be gated behind covariate mode, and the masked path must not require
`covariate_effects`.

## The per-panel conditioning context

Each panel that conditions owns a state object of the same shape, replacing the
original Atlas-local `covariateMode` / `covariateValues` / `selectedGroup` trio.
The two axes are independent fields, not one master switch:

```
conditioning = {
  covariateActive: boolean,               // covariate axis on/off (covariate bundles only)
  values: Record<string, number|string>,  // schema-driven covariate values
  group: string | null,                    // gating axis; null = "Background only"
}
```

- `covariateActive` and `values` apply only when the bundle has
  `covariate_schema`/`covariate_effects`. Off = corpus-average (the default); on =
  covariate-conditioned at `values`.
- `group` applies only when the bundle has `gating`; its options are
  `gating.groups` plus "Background only" (null). On the Phenotype Atlas it is a
  live selection (masking foreground immediately); on the generative panels it is
  the group whose cohort/record is generated.
- `values` keys and control types come from `covariate_schema.controls`
  (continuous slider, 2-level toggle, n-level select) — no hardcoded age/sex.

**State is per-panel and independent.** Each conditioning panel keeps its own
module-level state, so it survives that panel's own tab unmount/remount (fixing
the bug where revisiting a tab reset the controls), and panels do not share —
setting age on the Simulator does not move the Phenotype Atlas. State resets only
on **cohort/bundle change**, not on tab switch: a new model may differ on both
axes. (The Phase 1 reset fired on the Atlas component's remount, which a tab
switch triggers; the reset must be keyed on the cohort id changing, observed
where it does not remount per tab visit.)

### Prevalence reader (display; composes the two axes)

The Phenotype Atlas reader picks a base prevalence by the covariate axis, then
applies the group mask by the gating axis:

- **base by covariate axis:** `covariateActive` → `covariatePrevalence(effects,
  x)` (softmax of Gamma^T x); otherwise the existing non-covariate reader (the
  per-topic `fractionAboveTau(p, edges, tau)` value, which itself falls back to
  `corpus_prevalence` when a bundle has no theta histogram). The non-covariate
  base is exactly today's behavior — this design does not change it.
- **mask by gating axis:** for a gated bundle the mask is *always* applied for the
  current `group` (`null` = "Background only" → background topics only; `g` =
  background union `g`'s foreground). Not conditional on covariate mode.
  - On the covariate base: mask-before-softmax (`covariatePrevalenceGated`,
    renormalized over allowed topics).
  - On the non-covariate base: `maskGroupPrevalence(perTopicValues, topicBlocks,
    group)` zeros hidden foreground topics (no renormalization — a per-topic
    display quantity, not a distribution; matches the k-anon non-renormalization
    rule) and needs no `covariate_effects`.
- **no gating:** no mask; the base is used directly.

The absolute-bubble-scale anchoring is keyed on "any conditioning active"
(`covariateActive` OR a gated bundle) so switching groups does not re-pin the
largest bubble.

### Conditioned prior (generation; the sampler twin) — ADR 0028

The Patient Atlas and Simulator generate samples client-side from a **Dirichlet**
document-topic prior (corpus-average `model.alpha`, then a variational E-step
against `beta`). To condition generation, replace that prior with a
`conditionedAlpha` whose **mean is the conditioned topic proportions** and whose
**total concentration is the corpus prior's**:

    alpha_cond[k] = alpha0 * p_cond[k],   alpha0 = sum_k(alpha_corpus[k])

`p_cond` is chosen by the **same four quadrants** as the display reader — the
sampler is the generative twin of the reader:

| Quadrant | p_cond | mask |
|---|---|---|
| plain | alpha_corpus normalized (today's prior) | none |
| STM (covariate) | softmax(Gamma^T x) = `covariatePrevalence` | none |
| gated LDA | alpha_corpus normalized | hidden foreground floored ~1e-12 |
| gated STM | `covariatePrevalenceGated` (mask-before-softmax) | built in |

Masked foreground topics are floored at a tiny epsilon (mirroring the engine's
gated-block pin) so the Dirichlet and the E-step stay numerically defined while
those topics are effectively never drawn. The decision, the alternatives (a true
logistic-normal sampler; hard-dropping masked topics), and the honest consequence
(an STM bundle sampled through a Dirichlet looks more peaked than a true STM draw;
insight 0028) are recorded in ADR 0028. `conditionedAlpha` is one shared helper,
consumed by both generative panels and by the marginal sampler.

## Placement

Conditioning controls live **inside each panel**, not in global chrome. Each
conditioning panel renders the same controls cluster (the group selector and the
covariate toggle + schema-driven sliders/selectors + population readout), reusing
the widget component built in Phase 1 (`ConditioningBar.svelte`, mounted inside
the panel rather than app-global), with a short heading that frames that panel's
effect ("prevalence at these covariate values" on the Phenotype Atlas; "simulate
a record at..." on the Simulator; the regenerate cluster on the Patient Atlas).

Each section appears only when its axis exists in the bundle: the group section
iff `gating`, the covariate section iff `covariate_schema` with renderable
controls (`unsupported.length === 0`). A panel with neither axis renders no
conditioning controls. A gating-only bundle shows just the group selector; a
covariates-only bundle shows just the covariate toggle/controls; a both-axes
bundle shows both.

The Patient Atlas adds its own local controls beyond the shared cluster (sample-
vs-set toggle, color-by-group) next to its **Regenerate cohort** button (see
below).

## Population-reference readout

A compact, read-only readout (rendered with the covariate section) showing the
model's covariate marginals:

- continuous: p5-p95 range and median (p50) — already in
  `covariate_schema.controls[*]` as `range` + `default`.
- categorical: per-level proportions (e.g. "F 52% / M 48%").

It orients the user while setting sliders (realistic ranges) and answers "what are
this model's cohort properties" on its own. It reflects the **corpus** marginals
(a fixed reference), not the live slider selection; the heading says so to avoid
confusion.

**Export addition (shipped in Phase 1):** `covariate_schema` categorical controls
carry a k-anon-safe `proportions` map (level -> fraction, summing to 1 over
surviving levels; sub-threshold levels already filtered upstream). Shared by the
readout and the marginal sampler. Continuous marginals need no new data.

## Group labels (export addition)

`gating.json` carries raw `group_var` and raw `groups` ids (e.g.
`SOURCE_COHORT`, `rare_dx`). To show human labels, follow the existing
`{id, label}` metadata convention (cohorts and phenotypes already carry authored
labels; the front-end renders `label` and never humanizes raw ids itself):

- `gating.py` emits `group_var_label` and a `group_labels` map (raw id ->
  display label) alongside the existing fields, in both builders.
- The label is **humanized by default at export time** (`source_cohort` ->
  "Source cohort", `rare_dx` -> "Rare dx": replace underscores, sentence-case) so
  no authoring is required for a reasonable label, and **overridable** via an
  optional config map when a real name is wanted ("Rare diabetes cohort").
- The `GatingSpec` type gains `group_var_label?` and `group_labels?`; the group
  selector renders the label and keeps the raw id as the stable value. Older
  bundles without the fields fall back to the raw id.

## Per-panel application

Each axis is optional per the bundle; a panel applies only the axes its bundle
has.

| Tab | When the context applies | What it drives |
|-----|--------------------------|----------------|
| Phenotype Atlas | Live (cheap recompute) | the display reader (quadrant table above) |
| Simulator | Live (it re-samples on each run anyway) | the conditioned prior (`conditionedAlpha`) for each sampled record |
| Patient Atlas | On explicit "Regenerate cohort" (re-rolls the synthetic cohort; not live, to avoid the cloud reshuffling on every tick) | the conditioned prior for cohort generation |

### Phenotype Atlas

Reads the context via the composed display reader (base prevalence by the
covariate axis, foreground masked by the group axis) across the four quadrants, as
in Phase 1. The controls move from a global bar into this panel; the absolute
domain anchoring is keyed to "any conditioning active."

### Simulator

The Simulator recomputes its samples on each run, so it conditions live. It builds
`conditionedAlpha` from the panel's `values` (covariate axis) and `group` (gating
axis) and passes it to `runSimulator` in place of `model.alpha`; the rest of the
sampler (prefix, E-step, autoregressive option) is unchanged. Either axis may be
absent. The conditions-prefix editor composes with whatever axes apply. Because
the Simulator explicitly recalculates, it carries the full covariate controls.

### Patient Atlas

The Patient Atlas builds a **synthetic** cohort client-side (`generateCohort`
samples from the prior; `PatientMap` fits a UMAP projection of the cohort,
cached in `patientProjection` and reused by the Simulator's mini-atlas) and
already regenerates on demand. Conditioning applies on regenerate — re-rolling
the cohort and re-fitting its UMAP already happens there, so no new cost is
added; live re-rolling on every slider tick would refit UMAP repeatedly and
reshuffle the cloud. It owns a small local control cluster next to **Regenerate
cohort**:

- **Sample-vs-set toggle** ("sample from distribution" vs "use set
  covariates/group"):
  - *Sample* mode (the default, including first generation): each synthetic
    patient draws its own covariates (and group, for gated bundles) from the
    reported marginals via the marginal sampler, then generates from that
    patient's `conditionedAlpha` — a realistic, representative, mixed cohort.
  - *Use-set* mode: the next regenerate builds a this-subpopulation cohort at the
    panel's `values` + `group` (one `conditionedAlpha` for all patients). Use-set
    reads those fields directly regardless of `covariateActive`, applying only the
    axes the bundle has.
- **Color-by-group toggle** (gated bundles only): colors the existing cloud by
  each synthetic patient's group. Most informative on a sampled (mixed) cohort.
  Independent of the `group` selection.

Each synthetic patient carries its sampled/assigned group so color-by-group works
without recomputation beyond the regenerate.

### Marginal sampler (shared by the generative panels)

The sampler degrades per axis — it samples only the axes the bundle has.

Covariate axis (only when `covariate_schema` is present):

- categorical: draw a level from `proportions`.
- continuous: draw from a percentile-based approximation (a triangular
  distribution peaked at p50 with support [p5, p95]) — a marginal-only
  approximation, independent across covariates, explicitly accepted as good enough
  (no interactions).
- The sampled raw values become a design vector via the existing
  `buildDesignVector(design_columns, values)`; the conditioned prior reuses
  `covariate_effects` through `conditionedAlpha` (the same Gamma rows the Atlas
  uses).

Gating axis (only when `gating` is present):

- draw a group per patient from the model's `group_proportions`, then mask that
  patient's prior to the group's allowed topics. **Group proportions are not in
  `gating.json` today** (it carries only `group_var`, `groups`, `topic_blocks`);
  this increment adds a k-anon-safe `group_proportions` map to `gating.json`
  (computed from the per-group patient counts already used for k-anon, in the same
  builders), mirroring the categorical `proportions` addition. Until present, the
  sampler falls back to uniform over `gating.groups` and logs it.

A gated LDA model (no covariates) samples only the group axis; a no-gating
covariate model samples only covariates; a plain model samples neither (today's
generation).

## Components and file structure

This increment combines the Phase 1 per-panel rework with the former Phase 2
(Patient Atlas) and Phase 3 (Simulator). Front-end unless noted.

Per-panel rework (from the shipped global bar):

- `dashboard/src/lib/conditioning/` — keep `population.ts` and the
  `ConditioningBar.svelte` widget cluster; it is mounted **inside** the
  conditioning panels rather than app-global. Add `conditionedAlpha.ts` (the
  shared conditioned-prior helper, four quadrants) and `marginalSampler.ts` (the
  per-patient covariate/group draw).
- `dashboard/src/App.svelte` — remove the global `ConditioningBar` mount.
- `dashboard/src/lib/store.ts` — per-panel conditioning state (independent
  module-level stores per conditioning panel) instead of one shared store; the
  cohort-change reset moves to where the cohort id is observed without per-tab
  remount; the four-quadrant `prevalenceReader` is unchanged in logic.
- `dashboard/src/lib/tabs/Atlas.svelte` — render its own conditioning cluster;
  reset keyed on cohort change.
- `dashboard/src/lib/atlas/CovariatePanel.svelte` — delete (orphaned; its widgets
  were lifted into `ConditioningBar.svelte` in Phase 1).

Generative panels:

- `dashboard/src/lib/simulator/runSamples.ts`, `dashboard/src/lib/tabs/Simulator.svelte`
  — accept/pass `conditionedAlpha`; render the Simulator's conditioning cluster.
- `dashboard/src/lib/cohort.ts`, `dashboard/src/lib/tabs/Patient.svelte`,
  `dashboard/src/lib/patient/PatientMap.svelte` — conditioned `generateCohort`
  (set + marginal-sample paths), per-patient group, sample-vs-set toggle,
  color-by-group.

Export:

- `charmpheno/charmpheno/export/gating.py` + both builders
  (`analysis/local/build_dashboard.py`, `analysis/cloud/build_dashboard_cloud.py`)
  — `group_var_label`, `group_labels`, and k-anon-safe `group_proportions` in
  `gating.json` (per-group counts already computed for k-anon).
- `dashboard/src/lib/types.ts` — `GatingSpec` gains `group_var_label?`,
  `group_labels?`, `group_proportions?`.

## Error handling and edge cases

- Plain (no covariates, no gating) → no conditioning controls; the Atlas uses
  today's reader; the generative panels use `model.alpha` unchanged.
- Covariates, no gating (STM) → covariate section only; conditioned prior =
  softmax(Gamma^T x).
- Gating, no covariates (gated LDA) → group section only; the Atlas masks the
  `fractionAboveTau` base via `maskGroupPrevalence`; the prior masks the corpus
  alpha.
- Gating and covariates (gated STM) → both sections; reader and prior both mask
  the covariate-conditioned values.
- Gated with intercept-only covariates, or `unsupported.length > 0` → covariate
  section disabled/absent; treat as gating-only.
- Bundle/cohort switch → reset each panel's context (covariate off, values
  reseeded, group null). Tab switch does NOT reset.
- Missing categorical `proportions` or `group_proportions` (older bundle) → the
  readout omits proportions / the sampler falls back to uniform and logs it;
  nothing throws. Missing `group_labels` → raw id shown.
- Masked-topic floor: `conditionedAlpha` never produces a zero-length or
  all-zero prior; out-of-group foreground carries negligible mass.

## Testing strategy

- Pure helpers — `population.ts`, `conditionedAlpha.ts`, `marginalSampler.ts` —
  unit-tested with vitest: readout strings; `conditionedAlpha` mean matches
  `p_cond` and total concentration matches alpha0 in each quadrant, masked topics
  floored; sampler draws respect proportions/percentile support, deterministic
  under a seeded RNG, degrades per missing axis.
- Reader quadrants: `prevalenceReader` produces the right value in all four
  quadrants, including the covariate+gating store-level case (the Phase 1 gap);
  `maskGroupPrevalence` direct (hidden foreground -> 0, no renormalization).
- Generative panels: `runSimulator` uses the conditioned alpha (a covariate shift
  moves the sampled mean in the expected direction); `generateCohort` set vs
  sample paths; per-patient group assignment present for color-by-group.
- Schema-driven rendering: each panel shows the right sections for each bundle
  shape — component/build-level checks plus the FE suite + `npm run build`.
- Export: `covariate_schema` `proportions` (already covered) and `gating.json`
  `group_proportions`/`group_labels` — charmpheno export tests + the local gated
  integration test; cloud via `py_compile` + mirrored change (cloud-parity
  convention).
- Backward compatibility: a non-gated, no-covariate bundle renders with no
  conditioning controls and is otherwise unchanged.

## Out of scope

Color-by arbitrary categorical covariate on the Patient Atlas; covariate
interactions in the sampler; a logistic-normal (true STM) sampler; any
model/fit/eval change.
