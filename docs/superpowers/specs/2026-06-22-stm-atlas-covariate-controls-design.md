# STM Atlas covariate controls — Design

**Date:** 2026-06-22
**Status:** Brainstorm-grade design, awaiting user review.
**Scope:** Add interactive covariate controls to the dashboard's **Phenotype Atlas** tab for STM bundles. Moving a control (slider / dropdown / toggle) recomputes per-topic prevalence from the fitted Γ and resizes the topic bubbles live; bubble positions never move. The dashboard build derives a scrubbed covariate schema in-enclave (from the saved ModelSpec + covariate sidecar, no re-fit); the client is a pure renderer. Atlas only — Simulator/Patient tabs are deferred.

---

## Context

Prevalence-only STM puts a logistic-normal prior on logit-θ: η_d ~ N(Γᵀ x_d, Σ). The expected topic proportions for a patient with covariates x are softmax(Γᵀ x). The fit already exports Γ as `covariate_effects.json` (one row per design-matrix column, K topic effects each), and the dashboard already exports the corpus-mean topic prevalence `corpus_prevalence`. Nothing yet lets a viewer *ask* "what does the phenotype landscape look like for a 70-year-old male dementia patient?"

The Atlas sizes each topic bubble from a single `$prevalenceReader` store and positions it from a fixed 2D embedding `coords[p.id]` ([TopicMap.svelte](../../../dashboard/src/lib/atlas/TopicMap.svelte)). Position is covariate-independent because prevalence-only STM leaves β (topic-word content) unchanged by covariates — only the θ prior shifts. The Simulator tab already demonstrates swapping in a live, client-computed prevalence reader. So an STM covariate panel is a new reader plus a controls UI; the resize plumbing exists.

## Goals

1. An **STM covariate panel** in the Atlas, shown only when the bundle is STM, that builds controls automatically from the run's formula (fully formula-driven).
2. Moving a control recomputes `pₖ = softmaxₖ(Γᵀ x)` purely client-side and feeds `$prevalenceReader`, resizing bubbles in place. Positions never change.
3. A scrubbed, machine-readable **`covariate_schema.json`** contract derived in-enclave, reusing the bundle's existing privacy guards. The client never re-parses formulae and never sees single-patient values.
4. Graceful degradation: non-STM bundles never show the panel. Because `softmax(Γᵀ x)` needs a value for **every** design column, an incomplete x cannot be evaluated — so if **any** design column is unsupported, the interactive reader is disabled and the panel shows an "unavailable for this formula" note. `corpus_prevalence` is always the fallback.

## Key modeling fact (why bubbles resize but never move)

Covariates affect only the θ prior, never β. Topic semantics — and thus the fixed embedding that positions each bubble — are covariate-independent. Only `r = scaleSqrt(prevalence)` changes. (Bubbles would move only under content covariates, which are out of scope.)

## Design-matrix naming gloss (for future readers)

Γ rows are labeled with formulaic's design-column names. A categorical factor `var` with treatment coding produces names like `C(var)[T.level]`: the `T.` means **Treatment (dummy) coding** — the column is 1 when `var == level` else 0, and **one level (the reference) gets no column**, its baseline absorbed into `Intercept`. A 2-level categorical yields one dummy. The coefficient reads as "effect of `level` relative to the reference." The client does NOT parse these strings; the backend translates each into a structured recipe (below). The strings are only Γ-row join keys.

## The covariate schema contract (`covariate_schema.json`)

Derived in-enclave **at dashboard-build time** from the saved formulaic ModelSpec (`model_spec.pkl` in the covariate sidecar cache) plus the covariate sidecar itself — no re-fit needed. Scrubbed, then written to `covariate_schema.json` in the bundle alongside `covariate_effects.json`. Deriving here (rather than at fit time) means re-generating the dashboard from an existing checkpoint is enough to get the feature.

```jsonc
{
  "k": 20,                          // privacy threshold (same k as corpus_stats), for transparency
  "controls": [                     // one per underlying variable -> one widget each
    {"name": "age", "type": "continuous",
     "range": [40, 90], "default": 65},          // coarsened safe percentiles (p5 / p95 / p50) — NEVER min/max
    {"name": "source_cohort", "type": "categorical",
     "reference": "cancer", "levels": ["cancer", "dementia"]},  // levels with < k patients suppressed/omitted
    {"name": "sex", "type": "categorical",
     "reference": "F", "levels": ["F", "M"]}
  ],
  "design_columns": [               // SAME order as covariate_effects.json / Γ rows; `name` is the join key
    {"name": "Intercept",                       "recipe": {"kind": "intercept"}},
    {"name": "C(source_cohort)[T.dementia]",    "recipe": {"kind": "dummy", "var": "source_cohort", "level": "dementia"}},
    {"name": "C(sex)[T.M]",                     "recipe": {"kind": "dummy", "var": "sex", "level": "M"}},
    {"name": "age",                             "recipe": {"kind": "main", "var": "age"}}
    // interaction example: {"name": "age:C(sex)[T.M]", "recipe": {"kind": "interaction", "factors": [ {"kind":"main","var":"age"}, {"kind":"dummy","var":"sex","level":"M"} ]}}
  ],
  "unsupported": []                 // design-column names the backend could not express; if NON-EMPTY the client disables the interactive reader (x cannot be completed)
}
```

Recipe kinds and their client evaluation against the current control values:
- `intercept` -> 1.0
- `main` (continuous main effect) -> value of `controls[var]`.
- `dummy` -> 1.0 if `controls[var] == level` else 0.0. (Reference level selected => all that variable's dummies are 0.)
- `interaction` -> product of its `factors` evaluated recursively.

## Backend derivation + safety

At **dashboard-build time**, load the covariate sidecar with the same `try_load` the faithful `corpus_prevalence` path already uses — it returns `(cov_df, model_spec, covariate_names)`. Because the formula bans standardization, a continuous design column equals its raw value and a categorical dummy is a 0/1 indicator, so both the safe ranges and the level counts are recoverable from the **encoded** sidecar — no raw person-table reload.

1. **Classify variables:** from `categorical_cols` / `continuous_cols` (in the covariate manifest) and the ModelSpec factor structure.
2. **Categorical levels + reference:** read levels from the ModelSpec's factor maps; the reference is the level with no `[T.…]` column. **Level counts:** sum each `C(var)[T.level]` dummy column over `cov_df` (its sum = patients at that level; reference count = N − Σ dummies for that variable). **Safety:** suppress (omit) any level with fewer than `k` patients — reuse the small-cell-suppression pattern from `code_doc_counts` ([corpus_stats.py](../../../charmpheno/charmpheno/export/corpus_stats.py)), same `k` (`min_patient_count`, default 20).
3. **Continuous range + default:** take the design column equal to the variable (raw value, since standardization is banned) and compute coarsened percentiles (p5, p95, p50) over `cov_df`, rounded to a safe granularity — reuse the `theta_percentiles` precedent. **Never** min/max.
4. **Recipes:** walk the ModelSpec term structure to emit one recipe per design column, index-aligned with the Γ rows (`covariate_names`). A design column that cannot be classified into a known recipe kind is added to `unsupported`.
5. **Emit** the scrubbed schema directly to `covariate_schema.json` in the bundle, mirroring how `adapt_stm` writes `covariate_effects.json`. Echo a human-readable rendering to the build log for review.

If the sidecar is unavailable (no `cache_uri`, or a cache miss), no schema is written — the Atlas simply hides the panel (the same graceful path the faithful `corpus_prevalence` already falls back from).

## Client components

- **`covariateReader` store** (`dashboard/src/lib/`): holds the current control values; on change, evaluates `design_columns` recipes -> design vector x; returns a `(Phenotype) -> number` reader computing `pₖ = softmaxₖ(Γᵀ x)`. Mirrors the existing Simulator reader pattern so it plugs into the `prevalenceReader` selection.
- **`CovariatePanel.svelte`** (`dashboard/src/lib/atlas/`): renders one widget per `schema.controls` entry — continuous -> slider over `range` (default at `default`); 2-level categorical -> toggle; n-level -> dropdown; suppressed levels never appear. Includes a "reset to corpus average" affordance that returns the Atlas to the default `corpus_prevalence` reader.
- **Atlas wiring:** non-STM bundles (no `covariate_schema.json`) never show the panel — behavior unchanged. For an STM bundle the panel renders; if `unsupported` is non-empty the controls render read-only with an "unavailable for this formula" note and the covariate reader is not selectable (corpus_prevalence stays active).

## Data flow

```
DASHBOARD BUILD (in-enclave): try_load sidecar -> (cov_df, model_spec, covariate_names)
   -> classify vars; levels+counts from dummy-column sums; safe percentiles from raw continuous columns; recipes from ModelSpec
   -> SCRUB (suppress levels < k; percentile ranges, never min/max)
   -> covariate_schema.json   (alongside covariate_effects.json / Gamma)
   [no sidecar -> no schema -> panel hidden]
CLIENT: load covariate_schema.json + covariate_effects.json
   -> render controls -> on change: recipes -> x -> softmax(Gamma^T x) -> prevalenceReader
   -> TopicMap resizes bubbles (coords fixed)
```

## Testing

- **Backend (pytest):** schema derivation from a synthetic ModelSpec + synthetic sidecar `cov_df` — variable classification, categorical levels + reference, recipe kinds (intercept / main / dummy / interaction), and recipe order index-aligned with Γ. Level counts come from dummy-column sums; a rare level (< k) is suppressed/omitted. Continuous range/default come from coarsened percentiles over the raw continuous column, asserted to differ from min/max. A non-expressible column lands in `unsupported`.
- **Client (vitest if configured, else a small JS harness):** recipe evaluation for main / dummy / interaction; selecting the reference level zeros that variable's dummies; `softmax(Γᵀ x)` matches a hand-computed Γ on a tiny fixture; suppressed levels are absent from the rendered controls; a schema with non-empty `unsupported` disables the interactive reader (corpus_prevalence stays active).
- **Integration:** a fixture `covariate_schema.json` + `covariate_effects.json` -> controls render -> changing a control changes the reader's per-topic values (bubbles would resize) while `coords` are untouched; a bundle with no schema hides the panel.

## Out of scope

- Simulator and Patient tabs (the same `covariateReader` can plug in later).
- Content covariates (covariate-dependent β, which would move bubbles).
- Full support for interaction-heavy or otherwise exotic formulas beyond emitting them as recipes where expressible and `unsupported` where not.
- A Γ heatmap / forest-plot visualization (its own future design).
