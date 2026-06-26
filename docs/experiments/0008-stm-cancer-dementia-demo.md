---
id: 8
slug: stm-cancer-dementia-demo
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

# STM demo re-fit: covariate-rich non-gated cancer/dementia

Demo-grade re-fit of the non-gated STM prevalence model (the covariate
"shine" configuration) on the **current post-review engine**. Reproduces the
experiment 0003 design — same cohort, covariates, K, and sampling — but is a
fresh run because two correctness fixes landed after the earlier checkpoints
were taken and change fit results:

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

## Why non-gated for the demo

STM's distinctive capability is **covariate-dependent prevalence**: the fitted
Γ lets the dashboard condition per-topic prevalence on `source_cohort`, `sex`,
and `age`, so the conditioning sliders visibly shift the topic mix. That is the
demo angle. The gated variant (exp 0004) showcases the block partition but per
insight [0028](../insights/0028-dirichlet-vs-logistic-normal-rare-phenotype-recovery-gated-lda-is-plda.md) its rare
dementia foreground collapses under the logistic-normal prior — the wrong story
for a "shiny" demo. Rare-subgroup *content* recovery is PLDA's job, tracked
separately.

## Covariates

`~ C(source_cohort) + C(sex) + age` — three prevalence covariates, hence three
dashboard sliders. `source_cohort` (cancer vs dementia) is the strongest signal
and should produce large opposite-sign Γ loadings on cancer- vs dementia-leaning
topics; `sex` and `age` add finer prevalence modulation. Unlike the gated run,
`source_cohort` IS in the formula here (no foreground block makes it
rank-deficient), so it drives prevalence directly.

## Sampling

`prior_obs_days: 0` drops the 365-day pre-index lookback (widens both arms,
especially dementia). `person_mod: 4` samples ~25% of patients (combined corpus
~13k docs: cancer ~10.5k / dementia ~2.6k) — ample for a demo and faster than
the full cohort. `random_seed: 42` for reproducibility.

## Run

```
make exp ID=8
```

(or `scripts/run_experiment.py` with `WORKSPACE_CDR` + `GOOGLE_CLOUD_PROJECT`
set). Driver: `analysis/cloud/stm_bigquery_cloud.py`. After the fit completes,
re-export the dashboard bundle (`make build-dashboard-exp ID=8`) and verify the
new export fields land: `model.json` `sigma`, `covariate_effects.json`,
`covariate_schema.json`. Then run the labeler (`scripts/label_phenotypes.py`).

## Success criterion

Fitted Γ shows strong, opposite-sign `source_cohort` loadings on cancer-leaning
vs dementia-leaning topics, and the dashboard's conditioning sliders visibly
shift `corpus_prevalence` when toggling cohort / sex / age. Topic content need
not separate rare dementia sub-phenotypes — that is explicitly out of scope for
non-gated STM (insight 0028).
