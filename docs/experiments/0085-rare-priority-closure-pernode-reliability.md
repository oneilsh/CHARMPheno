---
id: 85
slug: rare-priority-closure-pernode-reliability
status: pending
model_class: gated_pc
cohort: population_rare_priority
cohort_def: population_rare_priority
disease: rare_priority
# CONFIRMATION RUN for the UNIFIED co-fit head. Exact 0082 config (41-anchor forest x
# 3 domains, closure mask, head_l2=0.01 ridge, NO Firth — ADR 0043) re-run with the
# new PER-NODE reliability readout (insight 0069). 0082 showed the ridge-bounded co-fit
# head is calibrated at scale by POOLED ECE (0.0098) and competitive with the two-stage
# readout LR — but pooled ECE can average an over- against an under-confident node. This
# run prints per-node ECE (mean/max/worst) for BOTH the co-fit head and the readout LR,
# to confirm the calibration holds node-by-node before we bless the single-stage model.
extra_domains: measurement,drug
label_mask_mode: closure
# --- corpus / DAG: identical to 0082 ---
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
n_bg: 8
tpn: 1
optimize_doc_concentration: true
# --- PC head: one-step ridge-Newton (ADR 0039/0041). head_l2 is the SOLE head
#     regularizer now (ADR 0043 removed Firth + the inner-loop Path B). ---
weight_y: 50.0
head_optimizer: newton
head_lr: 0.3
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

# 0085 — Per-node reliability of the unified co-fit head (0082 config, ridge-only)

Config-only re-run of 0082 (nothing in the fit changes — same seed, same knobs, Firth
was never on) whose ONLY delta is the driver readout: `conditional_readout` now emits a
**per-node ECE** for every scored parent→child edge and a **mean/max/worst** summary
beside the pooled ECE, for both the co-fit head and the head-independent readout LR.

## What to read

The two `[conditional sharpening: ...]` blocks each now print a line:

```
per-node reliability (ECE over N nodes): mean=… max=… (worst A->B)  vs pooled=…
```

- **The decisive number is `max` vs `pooled`.** If `max ≈ pooled`, calibration is
  uniform and the pooled 0.0098 is honest — the unified single-stage P(child|parent)
  head is blessed (no post-hoc fit, no Firth). If `max >> pooled` (e.g. a node at ECE
  0.05+ while pooled sits at 0.01), pooling was flattering and that node needs attention
  (likely a small-cohort node — cf. Amyloidosis n=66 in 0082).
- **Compare the co-fit head vs readout-LR per-node summaries.** 0082 had the co-fit head
  ahead on pooled ECE (0.0098 vs 0.0119); confirm the co-fit head is not worse on the
  WORST node (the failure mode pooling would hide).
- Everything else should reproduce 0082 exactly (cond_AUC by depth, per-parent top-1 vs
  majority, |w|max~2126) — a sanity check that the readout change didn't perturb the fit.

## Run

```bash
cd ~/repos/CHARMPheno && git pull origin claude/spectral-anchor-topic-k-200nqp && \
  CHARM_SPARK_CONF='spark.locality.wait=0s' make -C analysis/cloud exp ID=85
```

~15 min (eval off), same as 0082.

## Run log

_(pending — paste the two conditional-sharpening blocks incl. the per-node reliability
lines)_
