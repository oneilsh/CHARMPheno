---
id: 175
slug: eta-inv-vocab-per-domain
status: planned
model_class: dag_gated
# 0174 with eta = 1/V_m per domain (topic_concentration inv_vocab) instead of the 1/K
# default; everything else identical. A model-of-record candidate (criteria below).
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

# 0175 — η = 1/V_m per domain: a model-of-record candidate

**Status:** planned. **Date:** 2026-11-23. **Follows:** [0174](0174-model-of-record-v9-cdr.md).

## Why

η (`topicConcentration`) is the Dirichlet pseudo-count per (topic, term). Every gated
run so far left it unset, so it took MLlib's default 1/K. That ties the word prior to
the kept node set (any change to `minDocsPerNode` or the data moves K, hence η) and
gives domains of different vocabulary size one value. Its natural scale is the
vocabulary: η_m = 1/V_m. Raised in the walkthrough of 2026-11-23. At 0174's size
(K ≈ 3,250, V_m ≈ 5,000) the two are within a factor of about 1.5, so the expected
effect is small. Under-dispersed topics, or starvation that moves with η, would be the
surprise. Code: `topicConcentration` takes one value per domain; the driver resolves
`inv_vocab` from the domain vocabulary sizes and logs them.

## Design

0174's front matter with one change, `topic_concentration: inv_vocab`. The same v9
CDR, cohort, split, seed, node threshold and preset (`gate_warm` 4, 150 iterations,
flat blocks dropped at supportFraction > 0.9). The job also runs the ribbon readout
for 0174 if it is missing (0174's job did not), so the two compare on the same nodes.

## Decision criteria (fixed before the run)

0175 **replaces 0174 as the model of record** if all of these hold:

1. **Starvation:** blocks with supportFraction > 0.9 (dropped) and > 0.5, before the
   drop, are no more than 0174's + 5%.
2. **Detection** (stacked root head, ribbon nodes): AUC and AP no more than 0.005
   below 0174's. **Ranking** macro AUC within the parent cohort: likewise.
3. **New diagnoses** (stacked, paired by Mondo id over the shared ribbon nodes): the
   median per-node AUC delta is ≥ −0.001.
4. **Coherence:** median NPMI per exported domain (pair minimum 20) no more than 0.01
   below 0174's.

The tie-breaker, if 1-4 hold with no gain anywhere, is the principle: 0175 still wins,
because its η does not move with K. If any criterion fails, 0174 stays, and the
failure becomes an insight on how the gated fit depends on η.

## Run

On the warm v9 cluster:

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
export RUNS_DIR=/home/dataproc/workspace/dataproc-staging-bayesian-phenotyping-for-rare-disease-v9/runs
mkdir -p $RUNS_DIR/0175-eta-inv-vocab-per-domain
L=$RUNS_DIR/0175-eta-inv-vocab-per-domain/job_log.md
nohup bash scripts/jobs/exp_0175.sh > "$L" 2>&1 &
sleep 5; ls -la "$L"; tail -f "$L"
```

## Results

(pending)
