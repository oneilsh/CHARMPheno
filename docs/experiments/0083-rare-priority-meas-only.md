---
id: 83
slug: rare-priority-meas-only
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
extra_domains: measurement
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
eval_every: 0
num_partitions: 96
seed: 42
cache_uri: hdfs:///user/dataproc/charm/case_finding_cache
---

# measurement-only ablation

Exp 0081 variation (config-only). Same as 0081 but drop drug (condition + measurement only). Isolates measurement's solo contribution at scale — compare detection AP + per-node specialization to 0081 (all 3) and 0084 (drug-only).

```bash
cd ~/repos/CHARMPheno && git pull origin claude/spectral-anchor-topic-k-200nqp && \
  CHARM_SPARK_CONF='spark.locality.wait=0s' make -C analysis/cloud exp ID=83
```

## Run log

### Run 1 (condition + measurement, FULL mask, 41 anchors) — measurement-only recovers ~all of the 3-domain detection; drug's aggregate contribution is marginal

K=101, 55 scored nodes, prev 0.101. Full mask (detection alive). Fit ~62 min (gated_pc
arm 3725s; unsup arm 830s). Ran on the PRE-ADR-0043 driver, so no per-node reliability
line (that lands in 0085).

- **Drop drug → detection barely moves (the ablation headline).** gated_pc detection AP
  **0.2437** vs 0081's 3-domain **0.249** (Δ−0.005); pc_topics_lr macro AUC 0.7704 vs
  0.7724-ish. Measurement + condition alone recover essentially all of the 3-domain
  aggregate detection. **Drug is the sparse, specialized domain** — it carries per-node
  signal for treatment-defined diseases but does not move the aggregate.
- **Where drug's signal went (per-node λ-mass, now 2-column).** Without a drug column the
  treatment-defined diseases lean CONDITION-heavy instead: Temporal arteritis 0.808 cond
  (was drug 0.51 in 0081), Myasthenia gravis 0.690 cond (was drug 0.46). The treatment
  signal is absorbed by condition when drug is absent — so drug is the RIGHT home for
  those dx, not a necessary one for detection.
- **Specialization otherwise intact:** measurement-heavy = lab-diagnosed (Amyloidosis
  0.545, SLE 0.529, MS 0.519); condition-heavy = code-defined (ALS 0.913, senile
  dementia LBD 0.959, HCM 0.900). Same clinical logic as 0081, present in the unsup arm
  (representation property).
- **Mask dichotomy reconfirmed (full mask):** PC helps DETECTION (AP Δ+0.0117 over the
  unsup twin) but NOT conditional (cond AP Δ−0.0035, cond AUC Δ+0.0002 ≈ flat). Same sign
  as 0081. Full mask = detection objective.
- **Unified head holds at full mask too.** co-fit head vs readout LR: pooled ECE
  **0.0409 vs 0.0408** (identical), cond_AUC by depth ~0.02 below the readout (0.634/
  0.617/0.648 vs 0.658/0.629/0.668). |w_CK|max **16.7** — full mask keeps the head tiny
  (vs closure's ~2126), the well-conditioned regime. Same "calibrated + small
  discrimination tax" pattern as 0082's closure run.

**Read:** measurement is the workhorse second domain; drug is specialized (treatment-
defined dx) and marginal for aggregate detection. The 3-domain stack is justified by
per-node interpretability (drug as the treatment-signal home), not by a detection lift.
0084 (drug-only) would confirm the converse (drug alone underperforms) — low marginal
value, deprioritized in favor of 0085.
