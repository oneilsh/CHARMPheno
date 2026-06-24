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

## The shared conditioning context

A single front-end store, replacing the current Atlas-local `covariateMode` /
`covariateValues` / `selectedGroup` trio with a shared object:

```
conditioning = {
  active: boolean,                       // master on/off (off = today's defaults)
  values: Record<string, number|string>, // schema-driven covariate values
  group: string | null,                  // gating group; null = "Background only"
}
```

- `values` keys and control types come from the bundle's
  `covariate_schema.controls` (continuous slider, 2-level toggle, n-level
  select) — there is no hardcoded age/sex anywhere.
- `group` is present only for gated bundles (those with `gating`); its options
  are `gating.groups` plus "Background only" (null).
- The context persists across tab switches (it is module-level store state).
- On cohort/bundle change the context resets (`active=false`, `values` reseeded
  to schema defaults, `group=null`) — a new model may have a different schema.

The existing `prevalenceReader` (Phenotype Atlas) keeps working unchanged in
spirit: it already reads covariate mode + values + group; those reads are
repointed at the shared context.

## Placement

A full-width **conditioning bar** sits between the tab navigation and the tab
content (so it persists across tabs and reads as a lens on the current view).

- **Off (resting):** a single slim toggle chip ("Conditioning") — nothing else.
  Non-intrusive.
- **On:** the chip stays, and the strip expands in place to show, left to
  right: the schema-driven covariate controls, the group selector (gated
  bundles only), and the population-reference readout.

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

| Tab | When the context applies | Group handling |
|-----|--------------------------|----------------|
| Phenotype Atlas | Live (cheap softmax recompute) | single-select; masks foreground (existing `covariatePrevalenceGated`) |
| Simulator | Live (it re-samples on change anyway) | single-select; the group whose record is simulated |
| Patient Atlas | On explicit "Regenerate cohort" (accepts the one-time UMAP cost; never live) | own local controls (below) |

### Phenotype Atlas (Phase 1)

No behavior change beyond reading the shared context: bubble size = predicted
prevalence at `values`, foreground masked to background ∪ `group`. The absolute
domain anchoring already shipped. The old `CovariatePanel` is unmounted from the
Atlas tab; its control widgets are factored out for reuse by the conditioning
bar (see file structure).

### Simulator (Phase 3)

The Simulator already recomputes its samples on input change, so it reads the
shared context live: each sampled record's topic prior is conditioned on
`values` (via `covariate_effects`) and, for gated bundles, masked to background
∪ `group`. Covariate sliders are fully usable here. The conditions-prefix
editor composes with the context.

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

- **Phase 1 — shared context + global bar + Atlas:** introduce the shared
  store, build the conditioning bar (toggle, schema-driven controls, group
  selector, population readout), add the `proportions` export field, repoint the
  Phenotype Atlas at the shared context, and remove the old stacked
  `CovariatePanel`. This delivers the user-visible relocation and is the
  highest-value, lowest-risk slice.
- **Phase 2 — Patient Atlas:** marginal sampler, conditioned regenerate with
  the sample-vs-set local toggle, per-patient group assignment, color-by-group.
- **Phase 3 — Simulator:** live conditioning of sampled records on the shared
  context (covariates + group mask), composed with the conditions prefix.

## Components and file structure

Phase 1 (front-end unless noted):

- `dashboard/src/lib/store.ts` — replace `covariateMode`/`covariateValues`/
  `selectedGroup` with the shared `conditioning` store (keep thin
  backward-compatible derived exports if it reduces churn in unchanged readers).
- `dashboard/src/lib/conditioning/ConditioningBar.svelte` (new) — the bar:
  toggle, schema-driven controls (reuse the control widgets currently in
  `atlas/CovariatePanel.svelte`), group selector, population readout.
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
   `proportions`) and `gating`.
2. `ConditioningBar` renders controls from `covariate_schema.controls` and the
   group selector from `gating.groups`; writes user input into the
   `conditioning` store.
3. The store's `prevalenceReader` (unchanged logic, repointed inputs) feeds the
   Phenotype Atlas's `TopicMap` exactly as today.
4. The population readout derives its strings from `covariate_schema` via
   `population.ts`.

## Error handling and edge cases

- No covariates and no gating → the bar is hidden (nothing to condition).
- Covariates present, no gating → controls + readout, no group selector.
- Gated, intercept-only covariates → group selector + readout, no sliders.
- `unsupported.length > 0` (a covariate the dashboard cannot render) → the
  covariate controls are disabled with the existing "unavailable" note; the
  group selector still works if gated.
- Bundle/cohort switch → reset the context (active off, defaults, group null),
  so a stale selection never carries into a different schema.
- Missing categorical `proportions` (older bundle) → the readout omits
  proportions for that covariate and the sampler falls back to uniform, logging
  the fallback; nothing throws.

## Testing strategy

- Pure helpers (`population.ts`, the marginal sampler) are unit-tested with
  vitest: percentile/proportion formatting; sampler draws respect proportions
  and percentile support; deterministic under a seeded RNG.
- Store behavior: covariate-mode/gated `prevalenceReader` parity after the
  store refactor (existing `store.covariate.test.ts` continues to pass,
  repointed at the shared store).
- Schema-driven rendering: ConditioningBar shows the right controls for each
  schema shape (covariates-only, gated-only, both, neither) — component or
  build-level checks plus the existing FE suite + `npm run build`.
- Export: `covariate_schema` emits k-anon-safe `proportions`; covered by the
  charmpheno export tests and the local gated integration test.
- Backward compatibility: a non-gated, no-covariate bundle (e.g. an LDA cohort)
  renders with no bar and is otherwise unchanged.

## Out of scope

Color-by arbitrary categorical covariate on the Patient Atlas; covariate
interactions in the sampler; any model/fit/eval change.
