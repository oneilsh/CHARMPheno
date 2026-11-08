---
id: 170
slug: local-engine-fit-0168-config
status: planned
model_class: dag_gated
# 0168's configuration with the fit's E-step on the driver: LocalRunner over forked
# processes (engine 'local'). The bundle, node set, seed passes and NPMI stay on Spark.
# Same bundle key as 0168/0169 (the engine is not a bundle input).
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
max_iter: 60
doc_concentration: 0.5
optimize_doc_concentration: false
gate_warm: 2
gate_seed_mode: balanced
gate_seed_leaves: true
subsampling_rate: 0.1
tau0: 64.0
kappa: 0.51
cavi_max_iter: 100
cavi_tol: 0.001
num_partitions: 96
seed: 42
fit_engine: local
fit_processes: 0   # one per master CPU (8 on n2-standard-8)
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

# 0170 — The fit on the local engine (0168's configuration)

**Status:** planned. **Date:** 2026-11-08. **Compares with:** [0168](0168-dag-gated-api-validation-on-0163-config.md).

## Why

0168's Spark fit spent ~50 s per mini-batch iteration (about 22k documents) over 40
executor cores, while one core of the gated E-step costs ~2-5 ms per document
(measured 2026-11-08 at K = 3,045, V = 11,601, synthetic documents of 40-200 unique
tokens): about 45-110 s per batch on ONE core. So most of a Spark iteration is overhead
(broadcasting the 270 MB topic matrix, task scheduling, merging the statistics), and the
in-memory engine on the master's 8 cores should be several times faster per iteration.
This run measures that, and checks the local fit lands where 0168's did.

## What changes

`engine: local` (`DagGatedLDAEstimator`, `--fit-engine local`): after the node set and
the seed passes (Spark), the training documents are collected to the driver once
(`toLocalIterator`) and `LocalRunner` runs the same model, gate schedule and loop with
the E-step over `fit_processes` forked workers (0 = one per CPU). Mini-batches are drawn
by NumPy instead of `RDD.sample`: the same distribution, not the same batches.

## What success means

- Wall time per iteration well under 0168's (~50-77 s); total fit time reported.
- The same node set (6,024 given / 2,845 kept / 3,027 pruned / 152 spliced).
- The fit agrees with 0168 in distribution: ELBO at iteration 60 near 0168's
  (−64.2M), census and NPMI within run-to-run noise of 0168's.
- Master memory: the collected corpus plus (processes + 2) copies of the statistics
  (each up to 0.3 GB) fit in 32 GB.

## Run

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUNS=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs
mkdir -p $RUNS/0170-local-engine-fit-0168-config
L=$RUNS/0170-local-engine-fit-0168-config/job_log.md
nohup bash scripts/jobs/exp_0170.sh > "$L" 2>&1 &
sleep 5; ls -la "$L"; tail -f "$L"
```

In the log: `[dag-gated-lda] local engine: N documents collected to the driver in Xs;
E-step over 8 process(es)`, then the iteration lines (their last field is the wall time).

## Results

(pending)
