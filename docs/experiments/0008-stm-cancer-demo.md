---
id: 8
slug: stm-cancer-demo
status: pending
model_class: stm
cohort: cancer
cohort_def: first_cancer_year
prior_obs_days: 0
person_mod: 4
doc_unit: patient
covariate_formula: "~ C(sex) + age"
categorical_cols: [sex]
continuous_cols: [age]
random_seed: 42
cache_uri: hdfs:///user/dataproc/charm/covariates_cache
K: 40
max_iter: 100
---

# STM demo re-fit: cancer cohort with sex + age covariates

Demo-grade STM prevalence fit on the **cancer-only** cohort, on the current
post-review engine. Scoped to a single, homogeneous, majority cohort (rather
than the combined cancer/dementia corpus) so the result is cleaner and directly
comparable to the earlier cancer-only LDA run. The covariate-conditioning story
still holds — `sex` and `age` drive per-topic prevalence — without the
minority-washout dynamics that make the combined corpus noisy.

This is a fresh run because two correctness fixes landed after the earlier STM
checkpoints were taken and change fit results:

- **Lazy block updates** (ADR [0027](../decisions/0027-lazy-block-updates-for-gated-svi-mstep.md),
  `970d6d5`) — a minibatch with no documents for a gating block no longer
  ρ-blends that block's parameters toward zero. (No gating here, but the same
  M-step path is exercised; the fix restores the batch invariant.)
- **SPD Hessian guard** (ADR [0029](../decisions/0029-spd-guard-on-stm-laplace-hessian.md),
  `86fcb5a`) — the per-document Laplace covariance ν_d is now formed by an
  SPD-safe inverse (Cholesky fast-path, eigenvalue-floor repair otherwise),
  because the logistic-normal neg-log-joint is not globally convex and the raw
  Hessian can be indefinite.

Prior STM checkpoints (exp 0004/0005) are therefore **stale** and must not be
re-exported; this experiment produces the faithful bundle.

## Why cancer-only for the demo

STM's distinctive capability is **covariate-dependent prevalence**: the fitted
Γ lets the dashboard condition per-topic prevalence on the prevalence
covariates, so the conditioning sliders visibly shift the topic mix. The
cancer cohort is the demo's clean stage: a large, single-disease corpus where
topics separate sharply (the prior cancer LDA run recovered crisp
sub-phenotypes) and the only covariate-prevalence signal is the demographics.
The combined cancer/dementia corpus and the gated variants remain available
(exp 0003/0004) but, per insight
[0028](../insights/0028-dirichlet-vs-logistic-normal-rare-phenotype-recovery-gated-lda-is-plda.md),
the minority dementia structure collapses under the logistic-normal prior —
the wrong story for a "shiny" demo. Rare-subgroup *content* recovery is PLDA's
job, tracked separately.

## Covariates

`~ C(sex) + age` — two prevalence covariates, hence two dashboard sliders.
`source_cohort` is deliberately **absent**: with a single cohort it would be a
constant column (rank-deficient in the prevalence regression). The fitted Γ
should show interpretable demographic prevalence gradients (e.g. sex-skewed
cancers loading on `C(sex)`, age-associated cancers loading on `age`).

## Sampling

`prior_obs_days: 0` drops the 365-day pre-index lookback (consistent with the
STM experiment line). `person_mod: 4` samples ~25% of patients (~10k cancer
docs) — ample for a demo and faster than the full cohort. `random_seed: 42`
for reproducibility. `cohort_def: first_cancer_year` indexes each patient at
their first cancer-diagnosis year (matches cancer.yaml).

## Run

```
make exp ID=8
```

(or `scripts/run_experiment.py` with `WORKSPACE_CDR` + `GOOGLE_CLOUD_PROJECT`
set). Driver: `analysis/cloud/stm_bigquery_cloud.py`. After the fit completes,
re-export the dashboard bundle (`make build-dashboard-exp ID=8`) and verify the
new export fields land: `model.json` `sigma`, `covariate_effects.json`,
`covariate_schema.json`. Then run the labeler (`scripts/label_phenotypes.py`);
the labeler now handles STM bundles (no per-topic α — it uses corpus mass as
the background/dead disambiguator).

## Success criterion

Fitted Γ shows interpretable `sex`/`age` prevalence loadings, and the
dashboard's conditioning sliders visibly shift `corpus_prevalence` when
toggling sex / age. Topic content should reproduce the crisp cancer
sub-phenotypes seen in the prior cancer LDA run.
