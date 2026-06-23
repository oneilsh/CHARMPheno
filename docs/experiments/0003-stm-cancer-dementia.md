---
id: 3
slug: stm-cancer-dementia
status: pending
model_class: stm
cohort: cancer_or_dementia
cohort_def: cancer_or_dementia
prior_obs_days: 0
person_mod: 4
doc_unit: patient_cohort
covariate_formula: "~ C(source_cohort) + C(sex) + age"
categorical_cols: [source_cohort, sex]
continuous_cols: [age]
random_seed: 42
cache_uri: hdfs:///user/dataproc/charm/covariates_cache
K: 40
max_iter: 100
---

# STM validation: combined cancer/dementia cohort with source covariate

Validates STM prevalence covariates end to end. The combined corpus unions the
first-cancer-year and first-dementia-year cohorts; each document is labeled by
its `source_cohort`. A comorbid patient contributes two documents (one per
cohort). Success criterion: the fitted Gamma shows strong, opposite-sign
`source_cohort` loadings on cancer-leaning vs dementia-leaning topics, and the
dashboard's faithful corpus_prevalence reflects the cohort mix.

`prior_obs_days: 0` drops the default 365-day pre-index observation lookback,
widening both arms (especially dementia, where insidious onset and shorter
records make a year of prior coverage scarce) at the cost of admitting
prevalent cases whose first *recorded* dx may not be their true first. The
fully-observed 365-day follow-up window is unchanged.

`person_mod: 4` samples ~25% of patients (vs the _base 10%). At 10% the
combined corpus was ~5.3k docs (cancer 4,217 / dementia 1,034); extrapolation
suggested dementia is near AoU's ~8k ceiling, so 25% roughly scales both arms
(cancer ~10.5k / dementia ~2.6k) rather than uncovering hidden dementia
patients. Kept a subset deliberately: a model that separates cleanly here
should do at least as well on the full cohort.
