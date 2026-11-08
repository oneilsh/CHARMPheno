---
id: 169
slug: descriptive-model-of-record-0163-config-long-fit
status: done
model_class: dag_gated
# The descriptive model of record: 0168's configuration (0163's, on the new API) with a
# longer fit, for the topic descriptions themselves (eventual export; no readout).
# Same bundle key as 0168 (max_iter is not a bundle input), so on the same cluster the
# bundle is a cache HIT. holdout_frac stays 0.2: a train-on-everyone fit (0.0) is an
# untested path and a second bundle build, and the held-out 20% keeps the model
# validatable later.
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

# 0169 — The descriptive model of record (0163's configuration, long fit)

**Status:** done. **Date:** 2026-11-07. **Follows:** [0168](0168-dag-gated-api-validation-on-0163-config.md).

## Why

A model whose topic descriptions are the record: what each Mondo node's block and each
background topic says, in names, for eventual export (the dashboard does not render
DAG-gated models yet). 0168 checks the new API reproduces 0163; this run uses the same
configuration and corpus but trains 150 iterations instead of 60, so the topics are past
the point where more iterations move them (watch the ELBO trace and the census).

## What it writes (all in the run dir, workspace-internal)

- `model/`: the fitted `DagGatedLDAModel` (MLlib layout: Params metadata + `result/`),
  the source of record for any export: `dag()`, `nodeTopics()`, `describeNodes()` with
  term ids, the node set (kept / pruned / spliced).
- `label_dag.json`, `bundle_meta.json`, `code_map.tsv`, the npz: as every fit.
- `census_own10.md` (legibility census), `topics_digest.md` (the compact digest, drug and
  measurement names after the condition names), `topics_report.md` (every node topic,
  ordered by depth, 20 terms each), `npmi_top10.json` (coherence).

Egress: these are model parameters over a vocabulary floored at 20 patients per term
(`min_patient_count: 20`); per-node tables stay in the workspace. Anything committed
from this run carries pooled figures and node counts only.

## Run

Chained after 0168 in one job (0168's doc); alone:

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUNS=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs
mkdir -p $RUNS/0169-descriptive-model-of-record-0163-config-long-fit
L=$RUNS/0169-descriptive-model-of-record-0163-config-long-fit/job_log.md
nohup bash scripts/jobs/exp_0169.sh > "$L" 2>&1 &
sleep 2; ls -la "$L"; tail -f "$L"
```

In the fit log: the `[dag-gated-lda] node set:` line should match 0168's (same data, same
`minDocsPerNode`), and the bundle should be a cache HIT when run on 0168's cluster.

## Results

Ran 2026-11-08 (bundle a cache HIT; fit 8,253 s, 150 iterations at ~50 s; the node set
identical to 0168's: 6,024 given / 2,845 kept / 3,027 pruned / 152 spliced). An earlier
attempt (2026-11-07) fitted all 150 iterations and then failed to save the model: MLlib's
metadata writer runs a Spark job, so an executor tried to write the master-only run dir.
Fixed in `3d45004` (metadata written and read on the driver; the fit now parks its result
in `fit_ckpt` before any post-fit write).

| | 0168 (60 iterations) | 0169 (150 iterations) |
|---|---|---|
| ELBO, last iteration (mini-batch estimate) | −64.2M | −31.7M (−37.1M at 100, −33.7M at 120, −32.1M at 140) |
| census: signature / stratum / starved blocks | 96% / 4% / 0 | 90% / 3% / **189 (7%)** |
| gated NPMI, condition domain | +0.184 | +0.190 (median B − A +0.001, 2,828 nodes) |
| all-document NPMI | +0.236 | +0.247 (median B − A +0.002, 2,845 nodes) |

- **Longer training is more coherent but starves some blocks.** NPMI rises a little at
  every depth, and the ELBO roughly halves (still falling ~1% per 10 iterations at 150,
  so not converged). But 189 node blocks are starved where 0168 has none, and the share
  grows with depth (5% at depth 4-6, 7-10% at depth 7-12). A starved block's description
  is close to its prior, so it describes nothing.
- **Not yet the model of record.** Before exporting either: find why blocks starve
  between iteration 60 and 150 (likely the 200 background topics absorbing words from
  deep blocks once the step size has decayed), and decide whether the record is 0168,
  0169 with its starved nodes flagged, or a refit with that cause addressed.
