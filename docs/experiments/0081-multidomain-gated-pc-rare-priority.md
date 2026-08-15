---
id: 81
slug: multidomain-gated-pc-rare-priority
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
label_mask_mode: full
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
eval_every: 20
num_partitions: 96
seed: 42
cache_uri: hdfs:///user/dataproc/charm/case_finding_cache
---

# 0081 — Multi-domain Gated-PC on the expanded rare-disease set (3 domains)

The scale-up of the working conditional-diagnosis thread: the **41-anchor** expanded
forest (rare6 + 35 Monarch dismech priorities) × **three domains** (condition +
value-aware measurement + drug). Two changes vs the rare6 runs (0078/0079), both
motivated:

- **Wider disease set** (`disease: rare_priority`): more anchors, arranged in
  ontology-related families, so the conditional-diagnosis task has deeper
  parent→child→sibling structure to sharpen within. (The `anchor_selection`
  pipeline that generated these 41 — and scales toward the full ~1000-disease
  dismech list — is a separable later pull; the frozen tuple is enough to fit now.)
- **Three domains** (`measurement,drug`): drug adds the MG-style treatment signal
  alongside measurement; the per-domain λ + per-node scatter handle N domains.

Full mask so both readouts are available: detection (case vs bg) AND the
mask-independent conditional sharpening (P(child|parent), cond_AUC headline).

## What to read

- **Conditional sharpening**, per-parent, with majority baselines + balanced
  accuracy — the honest subtyping read, now over MANY more parents. Which families
  the model can subtype (bal_acc > majority) vs which it can't.
- **per-node domain λ-mass over 3 columns** (condition / measurement / drug) — does
  the specialization stay clinically sensible at scale (labs for antibody-diagnosed,
  drugs for treatment-defined)?
- **Detection AP** — does the aggregate still sit at the information-limited ceiling
  with more (and noisier) anchors? Expected, per hybrid insight 0076 — but that was
  the old post-hoc weighting, not per-node PC.
- Fit size: K/C will be much larger (~100+ nodes) → a heavier, slower fit. Watch
  `|w_CK|max` (head_l2=0.01 should bound it) and utilization.

## Run

```bash
cd ~/repos/CHARMPheno && \
  git fetch origin claude/spectral-anchor-topic-k-200nqp && \
  git checkout claude/spectral-anchor-topic-k-200nqp && \
  git pull origin claude/spectral-anchor-topic-k-200nqp && \
  CHARM_SPARK_CONF='spark.locality.wait=0s' make -C analysis/cloud exp ID=81
```

## Run log

_(pending first run)_
