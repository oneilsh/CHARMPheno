---
id: 170
slug: local-engine-fit-0168-config
status: done
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

**Status:** done. **Date:** 2026-11-08. **Compares with:** [0168](0168-dag-gated-api-validation-on-0163-config.md).

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

Ran 2026-11-08/09 on an n2-standard-8 master (8 vCPU, 31 GB; 9 GB used during the fit)
with **2 executors** (0168: 20), on a fresh cluster (bundle rebuilt).

**The local engine is correct but slower at 8 cores.**

| | 0168 (Spark, 40 executor cores) | 0170 (local, 8 processes) |
|---|---|---|
| wall time per iteration, median / mean | 54.0 / 58.8 s | 239.6 / 236.7 s |
| SVI loop total (60 iterations) | 3,530 s | 14,203 s |
| core-seconds per iteration | ~2,160 | ~1,900 |
| documents collected to the driver | n/a | 220,160 in 8.5 s |
| ELBO at iteration 60 | −64.16M | −64.18M |
| node set | 6,024 / 2,845 / 3,027 / 152 | identical |
| census: signature / stratum / starved | 96% / 4% / 0 | 97% / 3% / 0 |
| NPMI, median paired difference (both domains, gated and all) | | +0.0000 |

- **Both engines are compute-bound.** The two spend about the same core time per
  iteration, so 0168's Spark iteration was NOT mostly overhead, as this doc's Why
  assumed. That assumption came from a synthetic benchmark with one or two labels per
  document; real documents under the all-ever gate carry large frontiers, so their
  allowed sets (200 background topics plus their closure blocks) are wide, and the
  E-step costs about 70 ms of core time per document, not 2-5 ms.
- **The local fit reproduces the Spark fit** in distribution (different mini-batches,
  the same ELBO trajectory, node set, census and coherence).
- **When local pays:** with as many cores as the cluster. By the same arithmetic a
  single 64-vCPU machine would run about 25-30 s per iteration, with no executors and no
  topic-matrix broadcast. At 8 cores, Spark stays the engine for fits this size.
- The Spark phases outside the loop (bundle rebuild, support count, seed passes, NPMI
  2,531 s vs 0168's 622 s) ran on 2 executors and are not comparable with 0168's.
