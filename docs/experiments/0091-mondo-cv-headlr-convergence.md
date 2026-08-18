---
id: 91
slug: mondo-cv-headlr-convergence
status: pending
model_class: gated_pc
cohort: population_mondo_cardiovascular
cohort_def: population_mondo_cardiovascular
disease: rare_priority
# CONVERGENCE TEST for the co-fit head (exp 0090 run-4 ladder). The head-formulation
# ladder showed the co-fit head's deficit is dominated by UNDER-CONVERGENCE (+0.087
# cond_AUC from converging the localized head) then a missing INTERCEPT (+0.060); the
# ridge type barely matters. The shipped head_lr=0.3 damps each Newton step to 30%, so
# against a moving θ the head never settles (|w|=273 mid-trajectory; the ladder's
# full-Newton converged head settled at |w|=71 and scored 0.61). head_lr=1.0 is the
# PRINCIPLED full-Newton step, not a tuned knob. This run tests whether full Newton
# (a) recovers the co-fit head's conditional AUC toward its ~0.61 convergence ceiling
# AND (b) UN-STICKS the topic shaping: the new `corr_relΔλ` per-iter diagnostic shows
# how far the supervised correction moves λ off the unsupervised update — 0090 had
# gated≡unsup topics (bit-identical ELBO) because the saturated head's ∂loss/∂θ was
# starved. If corr_relΔλ rises and gated readout > unsup readout, PC shaping is back.
dag_source: mondo
mondo_branch: MONDO:0004995
min_positives: 100
mondo_version: 2026-06-02
mondo_cache_dir: data/mondo
extra_domains: measurement,drug
label_mask_mode: closure
localize_head: true
readout_sample_frac: 0.3
# --- the ONE delta vs 0090: full-Newton head step ---
head_lr: 1.0
# --- everything else identical to 0090 ---
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
weight_y: 50.0
head_optimizer: newton
head_newton_ridge: 0.05
head_l2: 0.01
grad_cavi_iters: 30
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

# 0091 — Mondo cardiovascular: full-Newton head step (convergence test)

Config-only variation of 0090: `head_lr: 1.0` (full Newton) instead of 0.3. The 0090
head-formulation ladder identified UNDER-CONVERGENCE as the co-fit head's biggest
single deficit (+0.087 cond_AUC), and `head_lr=0.3` — a 30% damping of each Newton
step — as the likely cause (the head sat at `|w|=273` instead of the converged ~71).

## What to read

1. **co-fit head conditional AUC** — does full Newton lift it from 0.52 toward the
   ladder's converged ceiling (~0.61)? And does `|w_CK|max` fall from 273 toward ~71?
2. **`corr_relΔλ`** (new per-iter log) — the supervised λ-correction magnitude. 0090's
   was effectively 0 (gated topics ≡ unsup topics). If it rises here, the un-saturated
   head is finally feeding a real shaping signal to the topics.
3. **shaping ablation** (`HEADLINE gated_pc vs unsup_gated`) — 0090 was Δ−0.0000
   (identical). If the readout delta goes positive, PC shaping is recovered — the big
   prize (it would mean the neutral-PC result was a broken-head artifact, not a data
   truth).
4. **the ladder** — the co-fit(as-trained) row should now sit near the converged row
   if full Newton did its job.

If full Newton recovers convergence AND un-sticks shaping, the remaining lever is the
unpenalized per-node INTERCEPT (+0.060 in the ladder) — an engine change staged next.

## Run

```bash
cd ~/repos/CHARMPheno && git pull origin claude/spectral-anchor-topic-k-200nqp && \
  CHARM_SPARK_CONF='spark.locality.wait=0s' make -C analysis/cloud exp ID=91
# then paste the COMPACT digest instead of the full log:
make -C analysis/cloud report ID=91
```

## Run log

_(pending first run)_
