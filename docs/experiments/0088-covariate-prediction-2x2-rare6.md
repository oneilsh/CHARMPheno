---
id: 88
slug: covariate-prediction-2x2-rare6
status: planned
model_class: dag_placement
cohort: population_rare6
cohort_def: population_rare6
disease: rare6
person_mod: 1
prior_obs_days: 0
doc_min_length: 10
min_n: 20
n_bg: 40
tpn: 5
print_topics_every: 10
holdout_frac: 0.2
init: spectral
node_alpha_scale: 1.0
optimize_doc_concentration: true
transform_alpha_mode: symmetric
spectral_max_vocab: 12000
spectral_method: scalable
anchor_scope: frontier
strip_mode: both
window_mode: lookback
lookback_days: 365
label_window_days: 365
spectral_topo_order: forward
max_iter: 200
seed: 42
cache_uri: gs://dataproc-staging-getting-started-with-registered-tier-data-copy/charm/case_finding_cache
# --- Patient covariates (cheap PREDICTION axis of the 2x2; plan
#     docs/superpowers/plans/2026-08-06-covariate-prevalence-prediction-2x2.md).
#     x_d is demographic/nuisance only -- the gating label (source_cohort) is
#     deliberately ABSENT from the formula (validate_label_not_covariate enforces).
covariate_formula: "~ C(sex) + age"
categorical_cols: [sex]
continuous_cols: [age]
known_sex_only: true
pred_cov: on
---

# exp 0088 — patient covariates for gated case-finding: the cheap PREDICTION axis

Identical corpus + fit to exp 0067 (forward spectral init, learned per-node alpha
fit, symmetric deploy, 1yr lookback, n_bg 40, frontier anchors, tpn 5, seed 42) —
**the ONLY change is patient covariates fed to the decision.** This is the cheap
axis of the prevalence×prediction 2×2 (plan
`docs/superpowers/plans/2026-08-06-covariate-prevalence-prediction-2x2.md`): a
post-fit, covariate-adjusted per-node/detection readout. Nothing in the fit is
reshaped — the DMR **prevalence** axis (which would reshape which topics a patient
expresses) is NOT built yet and is deferred pending this result.

## One run gives BOTH cells of the cheap comparison

`--pred-cov on` makes `dag_placement.evaluate` fit, per node and at the detection
level, an out-of-fold-CV L2 logistic on **[placement_score, x_d]** *and*,
separately, on **[placement_score] alone**, reporting AUC/AP for each. So the
manifest's `metrics.covariate_adjusted` block carries the whole comparison in a
single run:

| | prediction cov OFF | prediction cov ON |
|---|---|---|
| **prevalence OFF (this run)** | `*_score_cv` (score-only CV baseline) | `*_adj` (score + x_d) |
| **prevalence ON (DMR)** | not built (deferred) | not built (deferred) |

The delta `adj − score_cv` is the covariate's marginal lift **at the decision**.

## Pre-registered hypothesis (insight 0026)

Prevalence covariates reshape *which* topics fire, not their *content*, so they
tend to do little for case-finding; the cheap prediction axis instead captures
demographic confounding directly at the decision. Prediction:
`detection_auc_adj` clears `detection_auc_score_cv` by a modest but real margin
(age/sex carry genuine confounding of rare-disease membership), while the fit is
untouched.

**Decision rule.** If `detection_auc_adj ≈ detection_auc_score_cv` (adj − score_cv
within CV noise, say < ~0.005), covariates add nothing at the decision → the DMR
prevalence axis is **not worth building** for case-finding (a valid, useful
negative; stop). If the lift is real, it motivates building the prevalence axis
(plan Tasks 2–4) to test whether reshaping the representation does more than
adjusting the decision.

## Readout
```
cd ~/repos/CHARMPheno/analysis/cloud
make setup                 # first time on a cluster only (workspace env + formulaic overlay)
make clean-exp ID=88       # ensure a fresh run dir (exp auto-resumes an existing one)
make exp ID=88             # fit + inline covariate-adjusted placement eval
make summary-tail ID=88    # the fit's metrics block (incl. covariate_adjusted)
```
The covariate sidecar is loaded/built inside the fit driver (shared
`_covariates_load`, cached under `cache_uri`); no separate covariate-build step is
needed. `known_sex_only: true` avoids the single-sex-level collapse that drops the
sex column from the design matrix (see `_covariates_load` level diagnostics).

## Acceptance
1. **Runs clean, baseline untouched.** `manifest.covariates.pred_cov == "on"`,
   `manifest.covariates.names` lists the design columns (sex level(s) + age), and
   the non-covariate metrics (`node_auc`, `detection.auc`, `fdr`) match exp 0067
   to noise — covariates must not perturb the fit or the analytic metrics.
2. **The covariate_adjusted block is present** with `detection_auc_adj`,
   `detection_auc_score_cv`, `auc_adj_macro`, `auc_score_cv_macro`, and the
   per-node dicts, plus `n_covariates` == design width.
3. **Read the delta and apply the decision rule above.** Record
   `detection_auc_adj − detection_auc_score_cv` and the per-node macro delta.

## If the cheap axis earns its keep (only then)
Build the DMR prevalence axis (plan Tasks 2–4: per-doc α_d in the gated E-step,
Newton M-step on Λ, oracle) and add the two `prevalence ON` cells here as exp
0089, completing the 2×2. Otherwise close the covariate thread with this negative.
