---
id: 174
slug: model-of-record-v9-cdr
status: planned
model_class: dag_gated
# The model of record on the v9 workspace's CDR: 0173's configuration (gate_warm 4, 150
# iterations, unsupported nodes dropped at supportFraction > 0.9), re-fit in the new
# workspace (the CDR comes from the workspace env, `make setup`). The job also writes
# NPMI at the exportable pair minimum (20) and the browsable export.
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
drop_unsupported: true
drop_max_support_fraction: 0.9   # the stable core (0171/0172: starved blocks sit at ~0.999)
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

# 0174 — The model of record on the v9 workspace (0173's configuration)

**Status:** planned. **Date:** 2026-11-13. **Follows:** [0173](0173-cleaned-model-of-record-gate-warm4.md).

## Why

The v9 workspace has a new CDR and none of the earlier runs, so the model of record is
re-fit there: 0173's configuration unchanged (the cleaned long fit: `gate_warm` 4, 150
iterations, flat blocks dropped). Compare with 0173's pooled figures (node set kept /
pruned / spliced / unsupported, census, NPMI) to see what the new CDR moves.

## Run

On a fresh cluster in the new workspace (n2-standard-8 master, 8+ workers; the first
`make` target runs `make setup`, which reads the CDR and billing project with `wb`):

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
export RUNS_DIR=<the new workspace mount>/runs; mkdir -p $RUNS_DIR/0174-model-of-record-v9-cdr
L=$RUNS_DIR/0174-model-of-record-v9-cdr/job_log.md
nohup bash scripts/jobs/exp_0174.sh > "$L" 2>&1 &
sleep 5; ls -la "$L"; tail -f "$L"
```

## Results

(pending)
