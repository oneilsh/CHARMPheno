# Design: rare-disease + sparse demo models on a generalized population+disease cohort

Date: 2026-07-03
Status: approved (design), pending implementation plan

## Goal

Fit two new gated-STM models to broaden the demo dropdown beyond the current
default (exp 0028, `population_cancer`):

1. **Population + Ehlers-Danlos (rare disease)** — the headline
   rare-disease-on-a-background-population case.
2. **Population + Sparse (no disease anchor)** — the "better 0029": whole
   population split by in-window coding density into a dense background and a
   light-coder `sparse` foreground, with no cancer arm mixed in.

`population_cancer` (0028) stays as the exported default anchor and is **not
refit**. Comorbid (cancer+dementia) stays out of the manifest.

While adding these, generalize the `population+disease` cohort so future rare
diseases are config-only, not copy-paste. The density/sparse cohort deliberately
stays **outside** that disease framework (it has no concept set).

## Cohort layer (`charmpheno/charmpheno/omop/cohorts.py`)

### Generalized disease primitive

Add a generic first-diagnosis window primitive, parameterized so a disease is
fully described by concept ancestors + a window:

```
apply_first_diagnosis_year_cohort(
    cond_df, *,
    inclusion_ancestors,        # Sequence[int]
    exclusion_ancestors=(),     # Sequence[int]
    window_days=365,
    spark, cdr_dataset, billing_project, date_col, prior_obs_days=window_days,
)
```

- Concept set = (⋃ descendants of `inclusion_ancestors`) − (⋃ descendants of
  `exclusion_ancestors`). **Includes-then-excludes** priority: a concept in both
  sets is excluded.
- Multiple inclusion ancestors are unioned (supports diseases whose OMOP
  hierarchy has more than one top node).
- `window_days` replaces the hardcoded 365 post-dx window.

### Disease registry

A small module-level registry maps a disease key to its concept spec + display
metadata:

```
_DISEASE_REGISTRY = {
    "cancer": {label, description, inclusion_ancestors=(443392,),
               exclusion_ancestors=(<existing skin/in-situ exclusions>)},
    "eds":    {label, description, inclusion_ancestors=(79145,),
               exclusion_ancestors=()},
}
```

79145 is the OMOP top-level concept for Ehlers-Danlos syndrome (provided by
domain owner; verify descendant + patient counts on the cluster before the fit,
mirroring the dementia count-check comment already in the file).

### Generalized population+disease wrapper

```
apply_population_disease_cohort(
    cond_df, *, disease, window_days=365, spark, cdr_dataset,
    billing_project, date_col, prior_obs_days=window_days,
)
```

Same structure as today's `apply_population_cancer_cohort`: disease arm from the
registry (via `apply_first_diagnosis_year_cohort`), `general` arm from
`_random_observed_year_cohort`, tagged `source_cohort ∈ {<disease>, 'general'}`,
disjoint by person. `window_days` threads to **both** arms so foreground and
background windows match.

### Scope guardrails

- The validated `apply_first_cancer_year_cohort` and
  `apply_first_dementia_year_cohort` are **not touched** (still used by
  `apply_cancer_or_dementia_cohort`). Only the population+disease *wrapper* and a
  new generic primitive are added — the "full generalize" of the first-year
  functions is explicitly deferred.
- `population_cancer` output stays identical (same cancer concept set), so the
  exported 0028 model needs no refit. Its `apply_cohort` routing may delegate to
  `apply_population_disease_cohort(disease='cancer')` since that produces the
  same concept set; the existing `apply_population_cancer_cohort` is kept as a
  thin back-compat wrapper.

### Sparse cohort (outside the disease framework)

```
apply_population_sparse_cohort(
    cond_df, *, window_days=365, sparse_min=5, dense_min=20, spark, ...
)
```

Whole population (no disease arm) → `_random_observed_year_cohort` →
`_bucket_general_by_density` → `source_cohort='general'` (dense background) +
`source_cohort='sparse'` (light-coder foreground); `< sparse_min` dropped. Cancer
and EDS patients simply fold into the population here — the intended "clean
population" reference.

### Registration + tests

- Register `population_eds` and `population_sparse` in `SUPPORTED_COHORTS`,
  `COHORT_METADATA`, and `apply_cohort`.
- Add `test_cohorts.py` coverage: registry lookup, multi-inclusion /
  includes-then-excludes concept-set logic, `window_days` threading, and the two
  new cohorts' `source_cohort` tagging (mirroring existing cohort tests).

## Experiments

Two new experiment docs under `docs/experiments/`, both on the 0028 gentle +
hardened stack (subsample 0.1, tau0 128, max_iter 200, kappa 0.7, sigma_init 1,
reference topic + dense spectral init, min_pair_support 10,
`~ C(sex) + age`, `known_sex_only: true`, random_seed 42, block-wise
unit-diagonal Σ / ADR 0034).

### exp 0030 — `population_eds`

| Field | Value | Note |
|---|---|---|
| cohort / cohort_def | `population_eds` | new |
| K | 60 | 40 background + 20 EDS foreground |
| background_k | 40 | |
| foreground | `eds:20` | |
| group_var | `source_cohort` | |
| person_mod | **1** | full population — maximize the rare EDS arm + rich background |
| prior_obs_days | 0 | admit prevalent EDS cases |
| doc_min_length | 10 | |

Full-population fit is the heaviest run in this batch; expect a large `general`
background arm. Verify the EDS arm's doc count in corpus diagnostics.

### exp 0031 — `population_sparse`

| Field | Value | Note |
|---|---|---|
| cohort / cohort_def | `population_sparse` | new |
| K | 50 | 40 background + 10 sparse foreground |
| background_k | 40 | |
| foreground | `sparse:10` | |
| group_var | `source_cohort` | |
| person_mod | 4 | 25% sample; plenty for a whole-population light-coder read |
| prior_obs_days | 0 | no disease index |
| doc_min_length | 5 | sparse band is 5–19 codes |

New `experiments/defaults/population_eds.yaml` and
`experiments/defaults/population_sparse.yaml` (cohort + cohort_def only, like the
existing population_* defaults).

## Fit + export + dashboard

For each new experiment: `make build-covariates` then the STM fit driver, then
`make build-dashboard-exp` to export the bundle into
`dashboard/public/data/<cohort>/`. Add `population_eds` and `population_sparse`
to `dashboard/public/data/manifest.json` alongside `population_cancer` (which
stays the default). Confirm each bundle passes `test_bundle_completeness`.

## Out of scope

- Refitting `population_cancer` (0028).
- Collapsing `apply_first_{cancer,dementia}_year_cohort` into the generic
  primitive (deferred; the wrapper generalization is enough for this batch).
- Re-adding comorbid to the manifest.

## Open items to resolve during implementation

- Confirm OMOP concept 79145 resolves to the intended EDS descendant set and a
  workable patient count on the cluster before committing the fit.
- Confirm the EDS full-population fit fits the cluster's time/memory envelope;
  fall back to person_mod 2 if the background arm is unwieldy.
