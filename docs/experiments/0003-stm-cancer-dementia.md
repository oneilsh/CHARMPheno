---
id: 3
slug: stm-cancer-dementia
status: pending
model_class: stm
cohort: cancer_or_dementia
cohort_def: cancer_or_dementia
prior_obs_days: 0
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
fully-observed 365-day follow-up window is unchanged. Sampling stays at
`person_mod: 10` (the _base default).
