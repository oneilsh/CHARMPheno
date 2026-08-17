---
id: 82
slug: rare-priority-closure
status: pending
model_class: gated_pc
cohort: population_rare_priority
cohort_def: population_rare_priority
disease: rare_priority
# SCALE-UP: the expanded 41-anchor rare-disease forest (6 rare6 + 35 Monarch
# dismech; cohorts._RARE_PRIORITY_ANCESTORS) x THREE domains (condition +
# value-aware measurement + drug). The wider, ontologically-related forest gives
# the conditional-diagnosis task (P(child|parent)) far more depth and confusable-
# sibling structure than rare6; the third domain (drug) adds the MG-style
# treatment signal alongside measurement. Full mask (default) so we get BOTH
# detection AND the mask-independent conditional readout, and no closure-mask head
# blow-up. Built fresh (no cache).
extra_domains: measurement,drug
label_mask_mode: closure
# --- corpus / DAG (same knobs as 0078; disease + extra_domains are the deltas) ---
person_mod: 1
prior_obs_days: 0
doc_min_length: 10
min_n: 20
holdout_frac: 0.2
vocab_size: 5000
min_df: 20
min_patient_count: 20
window_mode: lookback     # REQUIRED for multi-domain
lookback_days: 1825
label_window_days: 365
strip_mode: both
# --- gate topic-block layout. K is emergent = n_bg + nodes*tpn; with ~41 anchors'
#     surviving descendants the node count (and K, C) is much larger than rare6 —
#     expect a heavier fit (hybrid insight 0076 saw ~95 nodes at this anchor set). ---
n_bg: 8
tpn: 1
optimize_doc_concentration: true
# --- PC head: run 7's known-good Path A (aggregated one-step Newton + ridge) ---
weight_y: 50.0
head_optimizer: newton
head_penalty: none
head_inner_iters: 0
head_lr: 0.3
head_newton_ridge: 0.05
head_l2: 0.01             # the run-4 known-good ridge (bounds |w| on the larger head)
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

# CLOSURE mask (conditional objective) at scale

Exp 0081 variation (config-only). Same as 0081 but label_mask_mode=closure and eval off. Tests whether training the CONDITIONAL objective helps subtyping at scale, where full-mask 0081 HURT it (cond AP Δ−0.013). head_l2=0.01 should keep the head bounded (unlike rare6 0079 which had no ridge and blew up). Read: does the gated_pc-vs-unsup CONDITIONAL delta flip positive (as rare6 0079 hinted), and does detection collapse (expected under closure)?

```bash
cd ~/repos/CHARMPheno && git pull origin claude/spectral-anchor-topic-k-200nqp && \
  CHARM_SPARK_CONF='spark.locality.wait=0s' make -C analysis/cloud exp ID=82
```

## Run log

### Run 1 (41 anchors, 3 domains, CLOSURE mask, eval off) — mask dichotomy FLIPS at scale; closure viable with the ridge; VOI-ready calibration

Fit ~15 min (eval_every=0; vs 0081's 113 min — the per-iter eval was the cost).

- **Sign flip confirmed at scale (the clean result).** Supervised vs unsupervised,
  same closure mask: cond AUC Δ**+0.0038**, cond AP +0.0015, top1 +0.0003 — all
  non-negative, where full-mask 0081 was NEGATIVE (cond AUC −0.010, cond AP −0.013).
  Train the conditional objective (closure) → supervision helps conditional; train
  detection (full) → it hurts. "Mask = task selector" now confirmed at rare6 (0078/
  0079) AND 41-anchor scale.
- **Closure is viable at scale BECAUSE of the ridge.** |w_CK|max ~2100 — bigger than
  full-mask's ~20 but tamed from rare6-0079's 1.3e4 by head_l2=0.01. The co-fit head
  is usable (sane conditional numbers, ECE 0.0098), where 0079's was garbage. So
  "closure blows up the head" was really "closure + no ridge".
- **Calibration excellent:** conditional ECE 0.0119 (better than 0081's 0.04). VOI-ready.
- **Detection dies** (AUC 0.50, AP = prev 0.101) — total under closure (background
  unobserved). A DIAGNOSIS model, not screening → the two-stage split (full screen ×
  closure sharpen).
- **Honesty note:** the absolute cond AUC (0.70) > 0081's full-mask (0.65), but part
  of that is the readout LR being fit on sibling-contrasts under closure (BOTH arms
  benefit, incl. unsup whose θ is identical to 0081-unsup) — not purely the model.
  The within-run sup-vs-unsup sign flip is the clean effect; the cross-run absolute
  jump is partly a readout artifact.
- Domain λ-mass stayed clinically sensible (ALS drug 0.39 riluzole-defined; MS/SLE
  measurement-heavy; sarcoid/HCM condition-heavy). Noisier than 0081; many rare nodes
  at the prior (0.431/0.431/0.138).

**Read:** closure-mask is the right training objective for the conditional-diagnosis
model, and at scale (ridge-bounded head) it's a viable, well-calibrated diagnosis
model — at the cost of detection (as designed). Confirms the two-stage architecture.
