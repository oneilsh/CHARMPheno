---
id: 29
slug: stm-population-cancer-sparse-foreground
status: draft
model_class: stm
cohort: population_cancer_sparse
cohort_def: population_cancer_sparse
prior_obs_days: 0
person_mod: 4
doc_unit: patient_cohort
doc_min_length: 5
covariate_formula: "~ C(sex) + age"
categorical_cols: [sex]
continuous_cols: [age]
random_seed: 42
cache_uri: hdfs:///user/dataproc/charm/covariates_cache
K: 70
background_k: 40
foreground: "cancer:20,sparse:10"
group_var: source_cohort
max_iter: 200
subsampling_rate: 0.1
tau0: 128
kappa: 0.7
sigma_init: 1.0
reference_topic: true
spectral_init: true
spectral_method: dense
min_pair_support: 10
---

# Experiment 0029 (DRAFT) — Sparse-general-year foreground

> **STATUS: draft — NOT runnable yet.** Needs the `population_cancer_sparse`
> cohort implemented (see "Implementation TODO"). Design captured here so it
> survives a context compaction.

## Question

Is the assumption that light-coder ("sparse") general years are low-signal
("generic checkup stuff, ~one topic") actually true? Rather than infer it from a
doc_min_length sweep, **give the sparse years their own foreground block and read
the topics**. If they come out as wellness/screening/routine codes, dropping
short docs is cheap and the floor choice (exp 0028's 10) is well-justified. If
they show structured conditions, short docs carry real signal we're discarding.

This is the sharp instrument for the exchange in the 0028 thread: it answers the
"content" AND "structure" questions in one fit, and the gating architecture
already supports arbitrary foreground groups
([partition.py `allowed_indices`](../../spark-vi/spark_vi/models/topic/partition.py)),
so no engine change is needed.

## Design

Three disjoint `source_cohort` tags, one document per person:

- **cancer** — first-cancer-year (SNOMED 443392−exclusions, 365d post-dx). 20
  cancer foreground topics. Unchanged from 0028.
- **general** — non-cancer persons whose event-anchored 365d window has **>= 20
  codes** (the "dense" general): `source_cohort='general'` → background-only.
- **sparse** — non-cancer persons whose window has **5–19 codes**:
  `source_cohort='sparse'` → its own 10-topic foreground block.

Persons with < 5 codes in the window are dropped (`doc_min_length: 5`). K=70 =
40 background + 20 cancer + 10 sparse. The sparse foreground topics ARE the
answer: their top codes + NPMI show what light-coder years are made of.

## Implementation TODO (the one wrinkle)

Doc length is only known after windowing, so the split is by event count in the
window, computed inside the cohort:

1. New cohort `population_cancer_sparse` in
   [cohorts.py](../../charmpheno/charmpheno/omop/cohorts.py): cancer arm as in
   `apply_population_cancer_cohort`; for the general (non-cancer) arm, after
   `_random_event_windows` + windowing the events, `groupBy(person_id).count()`
   the in-window events and tag `source_cohort` = `'sparse'` (5–19) vs
   `'general'` (>= 20); drop < 5.
2. Register in `SUPPORTED_COHORTS` + `COHORT_METADATA` + `apply_cohort` dispatch.
3. Add `experiments/defaults/population_cancer_sparse.yaml` (cohort + cohort_def).
4. Validation + windowing/bucketing tests in `test_cohorts.py`.
5. Flip status draft → pending; `make exp ID=29`.

Note: the corpus-build content-peek (`group_top_codes` top-codes[sparse]) already
gives a no-fit preview of the sparse band's codes; 0029 adds the topic structure.

## Interpretation

- Sparse foreground ≈ wellness/screening/routine → "generic checkup" confirmed;
  0028's floor of 10 is a fine, low-cost cut.
- Sparse foreground shows structured/varied conditions → short docs carry real
  signal; revisit the floor (or keep sparse as a permanent modeled group).

## Related

Follows exp 0028 (population + cancer gated, doc_min_length 10). Same slower
schedule + hardening stack. Uses the doc-length + top-codes diagnostics added in
the 0028 thread.
