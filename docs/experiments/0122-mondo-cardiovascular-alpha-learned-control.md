---
id: 122
slug: mondo-cardiovascular-alpha-learned-control
status: planned
model_class: gated_pc
cohort: population_mondo_all
cohort_def: population_mondo_all
disease: rare_priority
# THE CONTROL FOR 0121: uniform alpha init (0113's 0.5 everywhere) + the per-node
# empirical-Bayes alpha, now actually wired on the Gated-PC path (insight 0090 point 4:
# `optimize_doc_concentration: true` was inert on 0113-0120). Isolates the OPTIMIZER from
# 0121's equalized INIT: 0122 vs 0113 prices learning alpha from the uniform basin;
# 0121 vs 0122 prices the children-first basin. Everything else identical to 0113.
dag_source: mondo_native
mondo_branch: MONDO:0004995
tpn: 5
max_iter: 50
diag_only: true
preindex_closure: false
readout_mode: distributed
readout_theta_topm: 256
readout_l2: 100
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
n_bg: 8
optimize_doc_concentration: true
alpha_init: uniform
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
  # 0113's geometry, K = 1,498 as there.
  spark.executor.cores: 2
  spark.executor.memory: 8g
  spark.executor.memoryOverhead: 3g
  spark.dynamicAllocation.enabled: "false"
  spark.executor.instances: 20
  spark.excludeOnFailure.timeout: 10m
  spark.cleaner.periodicGC.interval: "5min"
---

# 0122 — cardiovascular branch, uniform α init + learned α (0121's control)

0113 with only the per-node empirical-Bayes α switched on for real (the front-matter flag
was inert on the Gated-PC path until the 0121 build). Pairs with 0121 to separate the
init from the optimizer.

## Run

Same as 0121 with `ID=122` (see that doc); same bundle key.

## Read

The 0121 table's third column, plus the learned α by depth.

## Results

*(pending)*
