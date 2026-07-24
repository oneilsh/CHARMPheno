---
id: 51
slug: pg-stm-distributed-mle-cancer-dementia
status: planned
model_class: pg_stm
cohort: cancer_or_dementia
cohort_def: cancer_or_dementia
prior_obs_days: 0
person_mod: 4
doc_unit: patient_cohort
covariate_formula: "~ C(sex) + age"
categorical_cols: [sex]
continuous_cols: [age]
random_seed: 42
cache_uri: hdfs:///user/dataproc/charm/covariates_cache
K: 50
background_k: 30
foreground: "cancer:10,dementia:10"
group_var: source_cohort
max_iter: 100
sigma_mode: mle
gibbs_sweeps: 0
---

# PG-STM distributed, MLE Σ — cancer + dementia (the runaway contrast arm)

The un-regularized **contrast arm** for exp
[0050](0050-pg-stm-distributed-iw-cancer-dementia.md). Identical corpus, gating, covariates,
and distributed PG-SVI kernel as exp 0050 — duplicated from exp
[0027](0027-stm-comorbid-blockwise-unit-diagonal.md), NOT overwriting it — differing ONLY in
the Σ M-step: `sigma_mode: mle` uses the un-regularized `scatter/n` point estimate (feed Σ
back into its own prior each iteration, no trust region), the estimator class that drove the
insight-0033 10^10 runaway.

Because the two arms share the SAME E-step and differ only in `sigma_mode`, any divergence in
Σ behavior is attributable to the ESTIMATOR — the clean at-scale isolation the toy scale
(exp 0049) could not provide. No Phase-2 Σ read-out here (`gibbs_sweeps: 0`); this arm exists
to show the runaway, not to read comorbidity.

**Expected outcome:** this arm reproduces the insight-0033 pathology at scale — Σ grows very
large (toward the ~10^10 regime) OR loses positive-definiteness across SVI iterations — while
exp 0050's IW arm stays bounded and PD. The driver guards against a NaN-crash so the
divergence is captured in `sigma_max_trace` rather than aborting the job.

## Run

`make exp ID=51`, then compare against exp 0050 per exp 0050's pre-registered success criteria.

## Results

_(to be filled after the cluster run)_
