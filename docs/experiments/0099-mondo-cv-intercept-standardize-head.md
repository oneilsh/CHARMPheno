---
id: 99
slug: mondo-cv-intercept-standardize-head
status: pending
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
weight_y: 16.0
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

# 0099 — Mondo cardiovascular: intercept + standardized co-fit head (the head fix)

exp 0098 showed strong stable shaping (weight_y=16, EG-stable) but PC HURT the readout
(Δ−0.017) because the co-fit head was near-chance (0.506) — a blind supervisor shapes
topics the wrong way. Root cause (head-formulation ladder + insight): the co-fit head
omitted the two things sklearn does for free — an unpenalized INTERCEPT and per-topic
STANDARDIZATION. Raw θ features span Σλ 1e2..1e6, so the logistic is so ill-conditioned
that even an intercept can't help; z-scoring per topic is the unlock.

Local realistic validation (held-out, weight_y=16): baseline co-fit HEAD 0.55 / readout
Δ+0.00 → +intercept+standardize co-fit HEAD **0.84** / readout **Δ+0.29** over unsup. Both
the unified-head goal AND the supervision-helps goal, at once. This run tests it at scale.

Deltas vs 0098: head_intercept=true, head_standardize=true (engine ADR-0045-follow-up,
now shim+driver-wired). Everything else identical (weight_y=16, doc_concentration=0.5,
per-doc-mean head, EG mass-preserving correction, grad_cavi_iters=15).

## What to read (make -C analysis/cloud report ID=99)

1. **HEADLINE gated_pc vs unsup_gated (pc_topics_lr)** — does supervision now LIFT the
   readout (Δ>0), reversing 0098's −0.017? The whole point.
2. **RARITY SPLIT** — is the lift on rare AND common, or concentrated?
3. **co-fit head macro AUC + conditional ECE** — the UNIFIED head. 0098 was 0.506 /
   ECE 0.24; a good standardized head should jump toward the readout and calibrate.
4. **FIT-HEALTH** — |w_CK|max (standardization maps w=w_z/σ, so watch for a σ-floor-sized
   balloon; local hit 543 harmlessly, but flag if it destabilizes), corr, ELBO rising.

## Sequence after this

- Lift positive + head good: the thesis lands (supervision helps + unified VOI head).
  Sweep weight_y for the best lift; calibrate the co-fit head (isotonic) for VOI; then
  whole-Mondo K≈3800.
- |w| balloons / unstable: add a σ-floor (or keep the head in z-space) and re-run.
- Still flat: the local +0.29 didn\'t transfer to K=444 sparsity — investigate the
  standardization at scale (per-node vs global θ moments).

## Run

```bash
cd ~/repos/CHARMPheno && git pull origin claude/spectral-anchor-topic-k-200nqp && \
  CHARM_SPARK_CONF='spark.locality.wait=0s' make -C analysis/cloud exp ID=99
make -C analysis/cloud report ID=99
```

## Run log

_(pending first run)_
