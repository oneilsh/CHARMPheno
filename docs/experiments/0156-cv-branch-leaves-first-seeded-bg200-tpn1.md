---
id: 156
slug: cv-branch-leaves-first-seeded-bg200-tpn1
status: planned
model_class: gated_pc
cohort: population_mondo_all
cohort_def: population_mondo_all
disease: rare_priority
dag_source: mondo_native
mondo_branch: MONDO:0004995   # cardiovascular branch (0132-0134): ~300 nodes
tpn: 1                      # as 0152; K = 200 + subtree nodes
max_iter: 60   # 3 iters/level x height leaves >= 30 full-gate iters if height <= 10
diag_only: true
init: spectral
spectral_method: scalable
spectral_d: 768             # 0134's: only 8 bg anchors + 3 per node are anchored
anchor_scope: closure       # on a flat forest, closure(node) = {node, root}: own block
spectral_topo_order: forward
count_transform: none
preindex_closure: false
gate_attestation: ever      # 0146: neutral on AUC, the cleaner gate
readout_mode: distributed
readout_theta_mass: 0.99    # excess-over-prior-floor coverage (0135 launch 3 fix)
weight_y: 0.0
weight_y_warmup_iters: 0
skip_unsup_gated: true
min_positives: 100
mondo_version: 2026-06-02
mondo_cache_dir: data/mondo
extra_domains: measurement,drug
label_mask_mode: closure
localize_head: true
head_support: path_cousins_kids
head_intercept: true
head_standardize: true
doc_concentration: 0.5
head_lr: 1.0
person_mod: 1
prior_obs_days: 0
doc_min_length: 10
min_n: 0
holdout_frac: 0.2
vocab_size: 5000
min_df: 20
min_patient_count: 20
window_mode: lookback
lookback_days: 1825
label_window_days: 365
strip_mode: both
n_bg: 200   # 1200 would keep a background doc's CAVI at 0152's cost; same in both arms
spectral_bg_anchors: 8
optimize_doc_concentration: false   # insight 0091; see 0134's LANDMINE note
head_optimizer: newton
head_newton_ridge: 0.05
head_l2: 0.01
grad_cavi_iters: 15
topic_trust: 0.05
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
spark_conf:
  # 0134's geometry. K is ~4x 0134's; if executors OOM in the E-step, raise
  # spark.executor.memory before touching min_positives.
  spark.executor.cores: 2
  spark.executor.memory: 8g
  spark.executor.memoryOverhead: 3g
  spark.dynamicAllocation.enabled: "false"
  spark.executor.instances: 20
  spark.excludeOnFailure.timeout: 10m
  spark.cleaner.periodicGC.interval: "5min"
gate_warm: 3   # leaves-first: iters per height level, then the full gate
gate_open_seed: true   # seed each block when it opens (subtree word counts)
---

# 0156 — cardiovascular branch, leaves-first gate with seeded opening

## Why

See 0155. One change: `gate_warm: 3` (3 iterations per height level, then the full
gate), with the subtree-mean seed at opening (ADR 0049). Known weakness from 0154: the
seed weights children by size, so a parent with one dominant child opens close to that
child; the child-balanced, own-code-excluded seed is the candidate fix if the leak
recurs here.

The run, read and comparison commands are in 0155 (one background job runs both).

## Run log
