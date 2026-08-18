---
id: 96
slug: mondo-cv-perdoc-newton-stability
status: pending
model_class: gated_pc
cohort: population_mondo_cardiovascular
cohort_def: population_mondo_cardiovascular
disease: rare_priority
# FAST head-starvation probe (minutes, not hours). Same Mondo-cardiovascular corpus as
# 0092, but: max_iter=8, skip the unsup twin, and --diag-only — fit the gated_pc arm
# ONLY, then print the per-node head-magnitude histogram (|w_c| on each node's support
# vs its positive count) and EXIT before the slow θ-collect readouts/baselines/ladder.
# Directly tests the exp 0092 hypothesis: is the whole-Mondo neutral-PC / corr≈0 caused
# by head STARVATION (the low-positive nodes' localized heads never train, |w_c|≈0)?
# The per-iter ||grad_y|| / |w_CK|max trajectory (b67c3c3) also prints during the fit.
dag_source: mondo
mondo_branch: MONDO:0004995
min_positives: 100
mondo_version: 2026-06-02
mondo_cache_dir: data/mondo
extra_domains: measurement,drug
label_mask_mode: closure
localize_head: true
diag_only: true
doc_concentration: 0.5
skip_unsup_gated: true
# --- the ONLY deltas vs 0092: fast probe (few iters, no readout, per-node head dump) ---
max_iter: 8
weight_y: 2.0
head_lr: 1.0
# --- everything else identical to 0092 ---
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
grad_cavi_iters: 30
topic_trust: 0.05
weight_y_warmup_iters: 25
subsampling_rate: 0.1
tau0: 64.0
kappa: 0.51
cavi_max_iter: 100
cavi_tol: 0.001
with_dag_head: false
baseline_max_iter: 100
min_label_count: 20
eval_every: 0
num_partitions: 96
seed: 42
cache_uri: hdfs:///user/dataproc/charm/case_finding_cache
---

# 0096 — Mondo cardiovascular: per-doc-mean newton stability (shaping alive, bounded)

exp 0095 CONFIRMED the alpha-collapse fix: with `doc_concentration=0.5` the shaping
gradient came alive for the first time in the project (iter 2: `corr_relΔλ=0.56`,
`||grad_y||=3.4e8`, vs bit-exact 0 forever before). But it then BLEW UP — a positive-
feedback runaway:

```
0095:  |w_CK|max: 113 -> 5133 -> 2.38e5 -> 4.4e5   (head runaway)
       corr_relΔλ: 0.56 -> 21.7 (λ moving 21x!)     ELBO -> -1e33
```

Root: `head_l2=0.01` is ABSOLUTE and negligible vs the ~110k-doc corpus-SUMMED newton
gradient, so |w| runs away; a runaway |w| feeds a giant `grad_topics ∝ w`, `weight_y=10 ×`
that detonates λ, topics thrash, head chases. Two fixes:
  1. **per-doc-mean newton** (engine, automatic): divide the newton g,H by n_docs so
     `head_l2` is a SCALE-INVARIANT ridge — |w| settles at ~|g_mean|/head_l2 (corpus-
     independent) instead of running to 1e5. The unregularized step H⁻¹g is unchanged.
  2. **weight_y 10 -> 2**: even at |w|=113 the correction was 56%/step; the reference
     weight_y is O(1). 2 targets a ~10%/step λ move.

## What to read (the in-fit `iter N/8:` trajectory)

- **SUCCESS**: `|w_CK|max` STAYS bounded (~tens–low-hundreds, no 1e5 runaway),
  `corr_relΔλ` is small-and-steady (~0.05–0.2, not 20+), `||grad_y||` finite (not 1e10+),
  and ELBO stays ~-1e7..-1e8 and RISES (no -1e33 explosion). That = a stable, shaping PC
  fit at whole-Mondo scale — the basic method working. Then: drop diag_only for a full
  fit + `pc_topics_lr` readout vs unsup (the actual deliverable / lift).
- **still climbing |w| / corr**: lower `weight_y` further (1) and/or raise `head_l2`
  (0.05) — both now corpus-invariant knobs, tune once.
- **corr ~0, shaping gone**: weight_y=2 too weak given per-doc-mean rescale — raise it.

## Run

```bash
cd ~/repos/CHARMPheno && git pull origin claude/spectral-anchor-topic-k-200nqp && \
  CHARM_SPARK_CONF='spark.locality.wait=0s' make -C analysis/cloud exp ID=96
```

(Paste the 8 `iter N/8:` lines + the `[head-starvation probe]` block.)

## Run log

_(pending first run)_
