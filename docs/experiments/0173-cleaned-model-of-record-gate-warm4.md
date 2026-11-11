---
id: 173
slug: cleaned-model-of-record-gate-warm4
status: done
model_class: dag_gated
# The first cleaned model of record (spec R14): 0172's configuration (gate_warm 4,
# 150 iterations: past the starvation cliff), then dropUnsupportedNodes at a strict
# threshold (supportFraction > 0.9: the stable core of flat blocks, not the marginal
# pool), then the ribbon readout. First end-to-end run of the drivers on a cleaned model.
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

# 0173 — The first cleaned model of record (0172's configuration, unsupported nodes dropped)

**Status:** done. **Date:** 2026-11-10. **Follows:** [0172](0172-starvation-gate-warm4.md), spec R14.

## Why

0169-0172 settled why node blocks starve: a block seeded at opening that its data never
renews decays to the flat η prior about 75 iterations after seeding; the schedule knobs
shift the count (`gate_warm` 4: 168 vs 205, the gain mostly among small leaves, which
train alone for longer) but not the mechanism. Spec R14 reports `nodeStatistics` by
default and drops flat nodes only on request. This run makes that request at a strict
threshold: supportFraction > 0.9 drops the core of flat blocks (which reach ~0.999) and
leaves the marginal ones (which vary run to run) in the model, visible in the
statistics. It is also the first end-to-end run of the readout, census and NPMI on a
cleaned model.

## What to look for

- The fit log: the node-set line, the node-statistics line (flat at 0.9, by support
  band), `dropped N unsupported node(s) (supportFraction > 0.9); M kept`.
- The census on the cleaned model: starved near 0 (blocks between 0.5 and 0.9 can still
  count as starved there).
- The ribbon readout on the cleaned node set; new-diagnosis AUC paired with 0168 over
  the shared ribbon nodes (dropped nodes have no head: missing from the pairing, not
  scored worse).

## Run

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUNS=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs
mkdir -p $RUNS/0173-cleaned-model-of-record-gate-warm4
L=$RUNS/0173-cleaned-model-of-record-gate-warm4/job_log.md
nohup bash scripts/jobs/exp_0173.sh > "$L" 2>&1 &
sleep 5; ls -la "$L"; tail -f "$L"
```

## Results

Ran 2026-11-10/11 (fresh cluster: bundle and conversion sidecar rebuilt; fit phase
13,784 s). The drivers ran end to end on the cleaned model (census, NPMI, sidecar, ribbon
readout, comparisons).

- **Dropped 150 of 2,845 node blocks; 2,695 kept** (`unsupportedNodes()`), flat by
  support band: 28/432 at 100-200 documents, 104/548 at 200-500, 18/358 at 500-1,000,
  none above. Trace: 0 starved from iteration 60, the cliff from ~75, plateau 150 by
  iteration 120 (0172, same configuration: 168).
- **No marginal band:** at iteration 150, 150 blocks have supportFraction > 0.5 and the
  same 150 > 0.9; none lie between. Flatness is all-or-nothing (a block stays fed or
  collapses to ~0.999), so the threshold choice does not matter in practice. What
  varies between runs (205 / 168 / 150) is which small nodes collapse, not where a
  borderline sits.

| | 0168 (60 iterations) | 0173 (150, cleaned) |
|---|---|---|
| census: signature / stratum / starved | 96% / 4% / 0 | 97% / 3% / 0 |
| NPMI condition domain, gated / all | +0.184 / +0.237 | +0.194 / +0.252 (median B − A +0.005) |
| NPMI drug domain, gated / all | +0.286 / +0.296 | +0.283 / +0.290 (median −0.001 / −0.003) |
| detection, root head = stacked (AUC / AP) | 0.8610 / 0.9917 | 0.8657 / 0.9920 |
| detection, flat max (AUC) | 0.610 | 0.618 |
| ranking within parent cohort, macro AUC / AP | 0.7322 / 0.4573 (932 nodes) | 0.7356 / 0.4699 (904 nodes) |
| new-diagnosis AUC, stacked, 841 shared ribbon nodes | 0.7170 | 0.7145 (median −0.0007; B wins 400) |
| new-diagnosis AUC, flat | 0.6104 | 0.6101 (median −0.0001) |

**Reading:** the longer fit sharpens the condition topics (and detection with them),
drug topics are level, per-node new-diagnosis accuracy is unchanged; dropped nodes have
no head rather than a flat one. 0173 is the candidate model of record: no starved
blocks by construction, and the 150 nodes the data did not sustain are named by the
model.
