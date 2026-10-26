---
id: 165
slug: stm-glp1-random-init
status: planned
model_class: stm
cohort: population_glp1
cohort_def: population_glp1
prior_obs_days: 365
person_mod: 1
doc_unit: patient_cohort
doc_min_length: 10
covariate_formula: "~ C(sex) + age"
categorical_cols: [sex]
continuous_cols: [age]
known_sex_only: true
random_seed: 42
cache_uri: hdfs:///user/dataproc/charm/covariates_cache
K: 110
background_k: 80
foreground: "glp1_ra:15,sglt2i:15"
group_var: source_cohort
max_iter: 300
subsampling_rate: 0.1
tau0: 256
kappa: 0.7
sigma_init: 1.0
reference_topic: true
spectral_init: false
min_pair_support: 10
---

# 0165 — STM on population_glp1 (exp 0044's configuration), random init

## Why

The STM's anchor-word spectral init has been on by default since insight 0030
(cancer cohort, exps 0012-0016). There it was the stabilizer that stopped the free
diagonal Σ running away (Σ max ~5e10 -> 7.6) and let all 40 topics resolve at
σ_init 1 instead of ~14. A few days later ADR 0034 made Σ a block-wise unit-diagonal
correlation matrix, which pins every variance at 1 and removes the runaway by
construction (insights 0033/0034). Whether spectral init still earns its place under
that Σ was never tested on real data; ADR 0050 asks whether the flat spectral init
(~940 lines) can go.

The pair reruns exp 0044's configuration (population_glp1, full population,
K = 110, the hardened + slowed stack, block-wise unit-diagonal Σ) on current code,
differing ONLY in `spectral_init`: 0164 on (= 0044 reproduced), 0165 off (random
init). Rerunning the spectral arm rather than reusing 0044's July record keeps the
pair on one code version and one corpus build.

## Read (0165 against 0164)

1. Topic resolution: dead / near-empty topics (Σλ at the floor, E[β] ~0) and
   clones; the 0012 -> 0015 failure was ~14 of 40 resolving.
2. Σ: `Σ_var` must stay [1, 1]; the eigenvalue range and the block masses.
3. Background NPMI (0044: mean +0.183, median +0.173) and ELBO (0044: -5.53e6).
4. The foreground blocks: SGLT2i cardiorenal and GLP-1 obesity/GI topics (0044's
   recovered structure, insight 0041).

If 0165 matches 0164 on all four, the flat spectral init is removed from the STM
(ADR 0050); if random init loses topics, it stays as the STM default.

## Run log
