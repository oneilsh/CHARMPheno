---
id: 176
slug: eta-inv-vocab-seed-replicate
status: planned
model_class: dag_gated
# 0175 (the model of record) with seed 43 instead of 42; everything else identical:
# a seed replicate of the flat-block count (0174: 176, 0175: 195).
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
topic_concentration: inv_vocab   # eta_m = 1 / V_m per domain (0174: unset = 1/K)
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
seed: 43   # 0175: 42
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

# 0176 — Seed replicate of 0175 (η = 1/V_m): is the +19 flat blocks noise?

**Status:** planned. **Date:** 2026-11-24. **Follows:** [0175](0175-eta-inv-vocab-per-domain.md), insight 0106.

## Why

0175 (η_m = 1/V_m) had 195 flat node blocks before the drop where 0174 (η = 1/K) had
176, and was adopted as the model of record anyway (its Decision). Insight 0106 reads the
difference as a weaker prior, but the flat count's run-to-run spread was never
measured: starvation is a cliff, and 19 of 3,229 blocks may be within it. This run
changes only the seed.

## Reading the result

The headline is the flat count under the rule both earlier runs used (dominant domain,
supportFraction > 0.9), recomputed from `node_statistics.tsv` by the job (the driver
now applies the every-domain rule of spec R14 amendment 3; both counts are printed).

- About 176 or lower: the +19 was mostly seed noise; insight 0106 is weakened or refuted.
- About 195 or higher: the increase holds across seeds; insight 0106 is confirmed and
  the seed spread is small next to it.
- In between: a spread of the same order as the difference; neither reading is settled
  without more replicates.

Also recorded: the every-domain count (the new rule; the run drops those) and the
model as fitted (`model_asfitted/`, kept since `c607182`), which lets the count be
recomputed under any rule later. No readouts. NPMI paired with 0175 as a check that the
seed does not move coherence.

## Run

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
export RUNS_DIR=/home/dataproc/workspace/dataproc-staging-bayesian-phenotyping-for-rare-disease-v9/runs
mkdir -p $RUNS_DIR/0176-eta-inv-vocab-seed-replicate
L=$RUNS_DIR/0176-eta-inv-vocab-seed-replicate/job_log.md
nohup bash scripts/jobs/exp_0176.sh > "$L" 2>&1 &
sleep 5; ls -la "$L"; tail -f "$L"
```

Ran on a cluster with 2 executors instead of the usual 8 (about 220 s per iteration
against 0175's 80 s). The fit is the same except for wall time; the partition count
follows the cluster's parallelism, so the minibatch draws differ from an 8-executor
run with seed 43, which a seed replicate does not care about.

## Results

(pending)
