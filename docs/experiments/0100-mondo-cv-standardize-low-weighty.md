---
id: 100
slug: mondo-cv-standardize-low-weighty
status: superseded
# SUPERSEDED (not run). Premised on "shaping over-drive is the harm, so lower weight_y."
# The sparse harness refuted that (shaping helped across corr up to 2.0); the lever is the
# co-fit HEAD quality, not shaping strength. Head-quality work continues in 0102.
model_class: gated_pc
cohort: population_mondo_cardiovascular
cohort_def: population_mondo_cardiovascular
disease: rare_priority
# THE FIX RUN (ADR 0044 / insight 0072). The supervised λ-correction is now a NATURAL
# gradient (∂L/∂E[logβ] = grad_eb·eb, scale-stable) instead of the old raw gradient
# (∝1/λ², which vanished at whole-population λ and made weight_y a silent no-op —
# gated topics ≡ unsup topics, corr_relΔλ ≈ 0 across 0090/0091). This is the first run
# where PC supervision can actually shape the topics at scale. weight_y is RE-CALIBRATED
# down (10, from the old 50 that only compensated for the ≈0 correction) — with the
# correction live, 50 can floor rare-topic λ cells in one step.
dag_source: mondo
mondo_branch: MONDO:0004995
min_positives: 100
mondo_version: 2026-06-02
mondo_cache_dir: data/mondo
extra_domains: measurement,drug
label_mask_mode: closure
localize_head: true
head_intercept: true
head_standardize: true
doc_concentration: 0.5
readout_sample_frac: 0.3
# --- the deltas vs 0091: natural-gradient correction (engine, automatic) + recalibrated wy ---
weight_y: 2.0
head_lr: 1.0
# --- everything else identical to 0091 ---
person_mod: 1
prior_obs_days: 0
doc_min_length: 10
min_n: 20
holdout_frac: 0.2
vocab_size: 5000
min_df: 20
min_patient_count: 20
window_mode: lookback
lookback_days: 1825
label_window_days: 365
strip_mode: both
n_bg: 8
tpn: 1
optimize_doc_concentration: true
head_optimizer: newton
head_newton_ridge: 0.05
head_l2: 0.01
grad_cavi_iters: 15
topic_trust: 0.05
weight_y_warmup_iters: 25
max_iter: 100
subsampling_rate: 0.1
tau0: 64.0
kappa: 0.51
cavi_max_iter: 100
cavi_tol: 0.001
skip_unsup_gated: false
with_dag_head: false
baseline_max_iter: 100
min_label_count: 20
eval_every: 0
num_partitions: 96
seed: 42
cache_uri: hdfs:///user/dataproc/charm/case_finding_cache
---

# 0100 — Mondo cardiovascular: standardized head + LOW weight_y (moderate the shaping)

exp 0099 (standardize, weight_y=16) HURT at K=444 because the huge standardized-head
gradient over-drove the shaping (corr_relΔλ=1.83, 183%/step) and thrashed the topics. The
sparse-harness (scratch) proved the standardization BENEFIT is real (co-fit 0.50->0.75,
readout Δ+0.34) and is the 1/σ amplification — a σ-floor kills it. So the fix is NOT to
floor σ but to MODERATE the shaping: keep standardization, drop weight_y 16->2 so
corr_relΔλ lands in the healthy ~2-5% band instead of 183%.

## What to read
1. FIT-HEALTH corr_relΔλ — must be ~0.02-0.05 (NOT >1). If still high, weight_y=1.
2. HEADLINE gated_pc vs unsup_gated — does the readout LIFT now (0099 was Δ-0.04)?
3. co-fit head AUC + ECE (unified head) — standardization should lift it toward ~0.6.
4. |w_CK|max WILL be large (standardization; cosmetic) — judge by corr + readout, not |w|.

## Run
```bash
cd ~/repos/CHARMPheno && git pull origin claude/spectral-anchor-topic-k-200nqp && \
  CHARM_SPARK_CONF='spark.locality.wait=0s' make -C analysis/cloud exp ID=100
make -C analysis/cloud report ID=100
```

## Run log
_(pending first run)_
