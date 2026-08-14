---
id: 79
slug: multidomain-gated-pc-closure-mask
status: pending
model_class: gated_pc
cohort: population_rare6
cohort_def: population_rare6
disease: rare6
# Conditions + value-aware measurement (as 0078) but with the CONDITIONAL training
# objective: label_mask_mode=closure trains each node's head against its DAG
# SIBLINGS (the parent cohort's other children), not against all background.
# Motivated by 0078 run 2: full-mask PC supervision HELPS detection but NOT
# conditional sharpening (P(child|parent)) — because full-mask IS a detection
# objective. Hypothesis: closure-mask aligns the objective with the sharpening
# task, so it should improve the conditional metric where full-mask didn't.
extra_domains: measurement
label_mask_mode: closure   # <-- the one delta vs 0078: conditional (vs-siblings) objective
# --- everything else identical to 0078 ---
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
head_penalty: none
head_inner_iters: 0
head_lr: 0.3
head_newton_ridge: 0.05
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

# 0079 — Multi-domain Gated-PC with the CONDITIONAL (closure-mask) objective

0078 run 2 delivered a clean dichotomy: PC supervision (with `label_mask_mode=full`)
**helps de-novo detection** (det AP Δ+0.025 over the unsupervised twin) **but not
conditional sharpening** — P(child|parent) was marginally *worse* than the
unsupervised fit. The mechanism: full-mask trains every node against *background*,
so it optimizes marginal detection and can blur within-parent subtype distinctions.

`label_mask_mode=closure` observes only each active node's DAG **closure + its
siblings** (the near-boundary negatives = the parent cohort's other children),
leaving distant nodes unobserved. That is exactly the **conditional (vs-siblings)
objective** — training the head to discriminate *within a parent*, which is the
clinician's "which subtype?" task.

## The test

Same readouts as 0078; the number that matters is the **conditional sharpening**
headline (gated_pc vs unsup_gated): cond AP / cond AUC / multiclass top-1 for
P(child|parent). 

- **Hypothesis:** closure-mask flips the sign — conditional sharpening now beats the
  unsupervised twin (where full-mask lost, Δ−0.013).
- **Watch the trade:** closure-mask observes far fewer cells (background docs
  contribute nothing), so de-novo detection AP may *drop* vs 0078's full-mask. The
  interesting outcome is a clean trade — closure BUYS sharpening, full BUYS detection
  — which would say the mask mode is the knob that targets the clinical task.

## Run

```bash
cd ~/repos/CHARMPheno && \
  git fetch origin claude/spectral-anchor-topic-k-200nqp && \
  git checkout claude/spectral-anchor-topic-k-200nqp && \
  git pull origin claude/spectral-anchor-topic-k-200nqp && \
  CHARM_SPARK_CONF='spark.locality.wait=0s' make -C analysis/cloud exp ID=79
```

## Run log

_(pending first run)_
