---
id: 2
slug: stm-cancer-dementia
status: pending
model_class: stm
cohort: cancer_or_dementia
cohort_def: cancer_or_dementia
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
