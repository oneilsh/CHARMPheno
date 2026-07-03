---
id: 31
slug: stm-population-sparse-gated
status: pending
model_class: stm
cohort: population_sparse
cohort_def: population_sparse
prior_obs_days: 0
person_mod: 4
doc_unit: patient_cohort
doc_min_length: 5
covariate_formula: "~ C(sex) + age"
categorical_cols: [sex]
continuous_cols: [age]
known_sex_only: true
random_seed: 42
cache_uri: hdfs:///user/dataproc/charm/covariates_cache
K: 50
background_k: 40
foreground: "sparse:10"
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

# Experiment 0031 — Whole-population density-split gated STM (no disease anchor)

## Goal

The "better 0029": read what light-coder general years are made of, against a
clean whole-population background with NO cancer arm mixed in. The population is
windowed then split by in-window coding density — dense years form the
background, light-coder (5–19 code) years get their own `sparse` foreground
block. If the sparse foreground reads as wellness/screening/routine, the
short-doc floor is well-justified; if it shows structured conditions, short docs
carry real signal.

## Cohort

New `population_sparse` cohort (`apply_population_sparse_cohort`, outside the
disease framework — no concept set):

- **general** (`source_cohort='general'`): persons whose event-anchored 365-day
  window has >= 20 codes; background-only.
- **sparse** (`source_cohort='sparse'`): persons whose window has 5–19 codes;
  10-topic foreground block. Persons with < 5 codes dropped
  (`doc_min_length: 5`).

## Configuration

K=50 = 40 background + 10 sparse. 25% sample (`person_mod: 4`) — ample for a
whole-population light-coder read. Otherwise the exp 0028 gentle + hardened
stack, `~ C(sex) + age`, `known_sex_only`.

## Success criteria

- Sparse foreground topics interpretable (wellness/screening vs structured
  conditions) — the answer to the short-doc-floor question.
- Σ variance bounded; honest correlation report.

## Related

Reframes exp 0029 (population + cancer + sparse) without the cancer arm.
