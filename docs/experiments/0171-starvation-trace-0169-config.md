---
id: 171
slug: starvation-trace-0169-config
status: planned
model_class: dag_gated
# 0169's configuration (150 iterations) with the starved-topic trace on, to see WHEN
# node blocks starve; the job first runs the offline starvation report on the fitted
# models of 0168, 0170 and 0169. Same bundle key as 0168-0170.
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

# 0171 — When and why node blocks starve (0169's configuration, traced)

**Status:** planned. **Date:** 2026-11-09. **Follows:** [0169](0169-descriptive-model-of-record-0163-config-long-fit.md).

## Why

0169 (150 iterations) left 189 of 2,845 node blocks starved, where 0168 (60 iterations,
otherwise identical) and 0170 left none; the share rises with depth. A starved topic
sits at the flat η prior: under SVI a block whose documents stop sending it words
decays back to η, so these blocks were most likely fed earlier and lost their words.
Two questions:

1. **Which nodes starve?** The offline report (`analysis/cloud/starvation.py`, no refit:
   it reads `<run>/model/result`) bins the starved rate by the node's document support,
   its share of its parent's documents, its kept children, its largest child's share
   of its documents, its depth, and its λ mass relative to its parent's; it then
   follows 0169's starved nodes back into 0168 and 0170. Hypothesis: the starved nodes
   are ones whose documents are mostly also their parent's or one child's (a nearly
   redundant block), which the parent or child out-competes once the step size decays.
2. **When?** The refit prints `[trace] iter t: starved node topics n/2845 (x%) · node
   topics hold y% of the lambda mass` every 5 iterations. Starvation during the
   leaves-first rounds (iterations 1-28) points at the opening schedule (0172 tests
   more iterations per level); starvation that grows through the full-gate phase
   points at competition at the full gate (a later leaves-first re-pass, or a cap on
   the background's share, are the candidates).

## The offline report (2026-11-09, before the refit)

Which of 0169's 189 starved blocks (pooled counts; `starvation.py` over 0168, 0170, 0169):
- **Support decides it:** starved 11% of nodes with 100-200 documents, 20% at 200-500,
  8% at 500-1,000, 1 of 1,507 above 1,000 (starved median 275 documents, fed 1,430).
- Small niches: median 0.07 of the parent's documents (fed 0.25), at most 2 kept
  children (3+ children: 2 of 512); depth adds little once support is fixed (5-7% at
  every depth from 4 down).
- **Already marginal at 60 iterations, the same in 0168 and 0170** (different batches):
  support fraction median 0.24 (the other 2,656: 0.01), λ mass / parent's 0.0067 (0.22).

Reading: the data never sustained these blocks; the open-time and leaf seeds' β did, and
SVI keeps only the product of (1 - ρ) of an unrenewed λ (about e^-7 between iterations
60 and 150 here). Every DAG-gated fit before 0169 stopped at 60 iterations or fewer,
which is why none showed this (0163: "starved 15 -> 0" from the leaf seed). Consistent
with insight 0091 (the word competition is decided on β: only the β start fed deep
blocks; α, learned or held, does not and is not a lever). Prediction for this trace:
near zero through the leaves-first rounds, then a steady climb through the full gate;
0172 (slower opening) ends about the same. Candidate fix on the β side: keep a seeded
block's seed as its prior (anchored / seeded topics), so an unrenewed block decays to its
seed rather than to flat η; or minDocsPerNode about 1,000; or 0168 (60 iterations) as the
record with weak blocks flagged.

## Run

Chained with 0172 (one job):

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUNS=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs
mkdir -p $RUNS/0171-starvation-trace-0169-config $RUNS/0172-starvation-gate-warm4
L1=$RUNS/0171-starvation-trace-0169-config/job_log.md
L2=$RUNS/0172-starvation-gate-warm4/job_log.md
nohup bash -c 'bash scripts/jobs/exp_0171.sh > "'$L1'" 2>&1; bash scripts/jobs/exp_0172.sh > "'$L2'" 2>&1' > /dev/null 2>&1 &
sleep 5; ls -la "$L1"; tail -f "$L1"
```

## Results

(pending)
