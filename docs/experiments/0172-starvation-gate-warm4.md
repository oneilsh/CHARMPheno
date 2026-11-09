---
id: 172
slug: starvation-gate-warm4
status: planned
model_class: dag_gated
# 0171 with twice as many iterations per level while the leaves-first gate opens
# (gate_warm 4: 60 iterations of rounds over heights 0..14, then 90 at the full gate),
# to test whether blocks starve because they open with too little training.
cohort: population_mondo_all
cohort_def: population_mondo_all
disease: rare_priority
dag_source: mondo_native
mondo_branch: ""
mondo_version: 2026-06-02
mondo_cache_dir: data/mondo
min_docs_per_node: 100   # spec 2026-10-30: replaces min_positives 100
extra_domains: measurement,drug
gate_attestation: ever   # the node set and the gate read gate_labels (all-ever)
label_mask_mode: closure
strip_mode: both
window_mode: lookback
lookback_days: 1825
label_window_days: 365
prior_obs_days: 0
person_mod: 1
doc_min_length: 10
holdout_frac: 0.2
vocab_size: 5000
min_df: 20
min_patient_count: 20
n_bg: 200
tpn: 1
max_iter: 150   # 0168: 60; ~2.5x the production iterations after the leaves-first rounds
doc_concentration: 0.5
optimize_doc_concentration: false
gate_warm: 4   # 0169/0171: 2
gate_seed_mode: balanced
gate_seed_leaves: true
subsampling_rate: 0.1
tau0: 64.0
kappa: 0.51
cavi_max_iter: 100
cavi_tol: 0.001
num_partitions: 96
seed: 42
trace_flat_every: 5   # print starved node topics every 5 iterations
cache_uri: hdfs:///user/dataproc/charm/case_finding_cache
spark_conf:
  # 0163's geometry, unchanged.
  spark.executor.cores: 2
  spark.executor.memory: 8g
  spark.executor.memoryOverhead: 3g
  spark.dynamicAllocation.enabled: "false"
  spark.executor.instances: 20
  spark.excludeOnFailure.timeout: 10m
  spark.cleaner.periodicGC.interval: "5min"
---

# 0172 — More iterations per level while the gate opens (gate_warm 4)

**Status:** planned. **Date:** 2026-11-09. **Compares with:** [0171](0171-starvation-trace-0169-config.md).

## Why

If blocks starve because they open with too little training (2 iterations per level,
then the full gate's competition), doubling the iterations per level should leave fewer
starved blocks at 150 iterations. Same trace as 0171; same 150-iteration budget, so the
full-gate phase is 90 iterations instead of 122.

## Run

Chained after 0171 (see 0171's Run).

## Results

(pending)
