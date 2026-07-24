---
id: 50
slug: pg-stm-distributed-iw-cancer-dementia
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
sigma_mode: iw
gibbs_sweeps: 40
sigma_readout_subsample: 20000
---

# PG-STM distributed, IW Σ — cancer + dementia (the runaway-cure test, IW arm)

Duplicates the **exp [0027](0027-stm-comorbid-blockwise-unit-diagonal.md)** corpus/gating
configuration (gated multi-group, K=50 = 30 background + 10 cancer + 10 dementia,
`~ C(sex) + age`, cancer_or_dementia cohort) — the corpus where the softmax point-EM Σ
runaway to ~10^10 was first observed (insight
[0033](../insights/0033-gated-fullcov-variance-runaway-is-an-init-identifiability-failure.md)).
It does NOT overwrite exp 0027; it runs the **distributed Pólya-Gamma / full-Bayes engine**
(sub-project 2) on the same data. This is the **IW arm**; exp
[0051](0051-pg-stm-distributed-mle-cancer-dementia.md) is the un-regularized **mle contrast
arm**. Together they are the decisive at-scale test the toy scale could not give (exp 0049,
milestone-1: the toy "MLE indefinite" contrast was a block-Σ zero-fill artifact, and the
10^10 severity never reproduced at K=6).

Design spec: `docs/superpowers/specs/2026-07-12-pg-stm-distributed-svi-gibbs-sigma-design.md`.
Engine: `spark_vi.mllib.topic.pg_stm.StreamingPGSTM` (Phase 1, `sigma_mode: iw`) +
`pg_stm_sigma_readout` (Phase 2, the comorbidity Σ read-out mean-field VI cannot produce —
insight [0044](../insights/0044-meanfield-vi-fails-sigma-correlation-even-when-identified.md)).

## The bet

A PROPER block inverse-Wishart posterior mean `E[Σ] = (Ψ0 + scatter)/(ν0 + n − dim − 1)`
(ν0 > dim+1 → finite PD mean even at n_docs → 0) gives Σ a genuine trust region, so on the
real cancer/dementia corpus — same gated nested stick-breaking model, same E-step, only the
Σ M-step differs from the mle arm — IW keeps Σ bounded and PD where the un-regularized
`scatter/n` point estimate reproduces the insight-0033 pathology.

## Pre-registered success criteria (fill Results on the cluster run)

1. **Runaway cure (decisive, vs exp 0051):** the mle arm reproduces the insight-0033
   pathology at scale (Σ → very large OR loses positive-definiteness across SVI iterations),
   while THIS iw arm over the identical E-step keeps `max|Σ| = O(1–10)` AND stays PD
   (Cholesky succeeds), with a bounded `sigma_max_trace`.
2. **Phenotype quality preserved:** the iw fit still surfaces the cancer/dementia
   sub-phenotypes (topic coherence / recognizable top terms comparable to the exp-0027 STM
   baseline); IW regularization is not paid for by washing out topics.
3. **Comorbidity read-out (Phase 2):** the exact-Gibbs Σ correlation matrix is a valid,
   interpretable comorbidity structure (symmetric, PD, in-range correlations) — the
   trustworthy-correlation read-out mean-field VI cannot produce (insight 0044).

## Run

`make exp ID=50`. Then compare the
`sigma_max_trace` / final Σ eigmin / max|Σ| / Cholesky status against exp 0051 (mle), and
record whether IW cures the runaway at scale. If confirmed, write the insight.

## Results

_(to be filled after the cluster run)_
