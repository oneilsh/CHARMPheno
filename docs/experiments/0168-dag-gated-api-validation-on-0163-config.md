---
id: 168
slug: dag-gated-api-validation-on-0163-config
status: planned
model_class: dag_gated
# The cluster validation of the DAG-gated user API redesign (plan 2026-11-02, WP 14):
# 0163's configuration, re-fit on the new estimator. Every key below is read by
# scripts/run_experiment.py and reaches a dag_gated_fit.py flag (checked against both
# on 2026-11-03). 0163's knobs that the drivers no longer accept are dropped:
# diag_only, init, count_transform, preindex_closure, readout_mode, readout_theta_mass,
# weight_y, weight_y_warmup_iters, skip_unsup_gated, localize_head, head_*,
# grad_cavi_iters, topic_trust, with_dag_head, baseline_max_iter, min_label_count,
# eval_every, min_n. min_positives is replaced by min_docs_per_node (counted on the
# TRAIN documents by the estimator, not on the whole population by the Mondo build).
# Left to the estimator's recommended preset (run_experiment never emits them):
# gammaShape 100 and gateOpenSeed true, both 0163's values. The knobs set below that
# the preset also covers (n_bg, tpn, max_iter, SVI, doc_concentration,
# optimize_doc_concentration, gate_*) carry the preset's values, i.e. 0163's.
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

# 0168 — the DAG-gated API redesign, validated on 0163's configuration

## Why

The redesign (spec [2026-10-30](../superpowers/specs/2026-10-30-dag-gated-user-api-design.md),
plan [2026-11-02](../superpowers/plans/2026-11-02-dag-gated-user-api-plan.md), ADR 0052)
moved the node set out of the Mondo build and into the estimator: the build hands over
the coded Mondo subgraph (WP 11), the estimator learns the kept / pruned / spliced nodes
from the training documents with `minDocsPerNode` (WP 4-5), and the drivers fit, save
and read the model's node set (WP 12). Every bundle key and the conversion and workup
sidecar keys moved. This run is the plan's cluster validation: the central configuration
(0163) re-fit on the new code, read the same way, compared against 0163 by Mondo id.

Expected differences, none of them a regression:

- **Node set.** 0163 kept the terms with at least 100 persons in their closure over the
  whole population (Mondo build), then took the induced Hasse diagram and spliced it
  (`collapse`). 0168 counts closure support on the training documents only (80% of
  persons; `gate_labels`, the all-ever column) at the same 100, so nodes with
  whole-population support between about 100 and 125 should drop, and the splice is the
  estimator's. Expect fewer kept nodes than 0163's 2,694 (K 2,894 = 200 + 2,694), most of
  them shared by Mondo id.
- **Readout width.** 0163 read θ at full K (mass 0.99 resolved to the full width); the
  walkthrough fixed the readout at top-m 100 (`DEFAULT_READOUT_THETA_TOPM`). The AUC
  pairing below therefore folds the top-m change in with the redesign; a
  `--readout-theta-topm 0` re-readout separates them if the difference is not small.
- **Strip.** The old path stripped OMOP concept ids that happened to equal a Mondo number
  (`strip_mode: both`); on curies the strip is a no-op, so a few feature tokens can
  return. The vocabularies are built before the strip and must be identical.

## What success means

1. **Corpus.** Each domain's vocabulary identical to 0163's, term for term and index for
   index (`bundle_meta.json` `vocab_maps`; the corpus code and keys did not change, only
   the label side did). This is the hard gate; the job stops if it fails. Neither run
   records documents or tokens per domain (0163's ledger is the old pruning receipt:
   kept / dropped / by depth / coarsening / `test_fg_docs`; 0168's is `n_dag_nodes` /
   `train_fg_docs` / `test_fg_docs`), so the document check is the shared key
   `test_fg_docs`: equal or slightly larger (the new subgraph keeps every coded term; a
   drop needs explaining).
2. **Node set.** Kept nodes a few percent below 0163's 2,694, almost all shared by Mondo
   id; the 0168-only nodes few (splice differences). Given / kept / pruned / spliced
   reported beside 0163's powered / Hasse / final counts.
3. **Fit.** The gate line ends `leaf blocks seeded at iteration 0 from their own
   documents`; wall time near 0163's 1.1 h.
4. **New-diagnosis AUC, 827 ribbon nodes (paired by Mondo id).** Macro within about
   0.005 of 0163's (stacked 0.7104, flat 0.6059), paired median B − A within ±0.005
   (0161 → 0163 moved 0.002).
5. **Stacked detection.** Within about 0.005 of 0163's stacked and root-only AUCs.
6. **Legibility census.** Own-code identity near 0163's 1,379 of 1,442 decidable (96%),
   signature share near 95%.
7. **NPMI** (cheap, exercises the gate-column path): paired medians about 0.
8. **Framings and workup.** Framing 1 median kappa change about 0 against
   `survey_inc_top`; the workup survey shows 0167's run-4 picture (fact-level ranking
   positive in the powered groups, small gains).

## Run (one background job)

Verified against the code at `04fae73b` (2026-11-03): targets `exp`, `inspect-topics`,
`dag-gated-readout`, `build-conversion-sidecar`, `build-workup-sidecar`; make variables
`DGR_SPARK_CONF` / `DGR_ARGS` (the `gated-pc-readout` target and `GPR_*` are gone);
every readout flag in `dag_gated_readout.build_parser` (`--readout-mode`,
`--readout-theta-mass`, `--diff-next-question` and `--diff-nq-hide` no longer exist and
are dropped). No pre-fit `--bundle-only`: the readout's `--bundle-only` loads
`<run>/model` first, so on a new run the bundle is built by the fit's first phase and
the fit writes `bundle_meta.json`; stage 1 reads that.

Output names at the default top-m (no tag for the width, since the new manifest records
no mass target): the ribbon readout's tag is `stacked_nodes` (`stacked` +
`--readout-nodes`), so `results_readout_stacked_nodes.json`,
`readout_heads_gated_pc_stacked_nodes.npz`, `prevalence_check_stacked_nodes.json`; the
framings read those heads with `--acq-heads stacked_nodes` (the default `stacked` is the
unmasked readout's). 0163's are the same names with `_mass99`. The ribbon readout is
kept (not a full-node `stacked` readout) so that 0163's per-node results, detection and
survey heads are compared like for like.

Order: fit (bundle MISS rebuild inside, ~20 min, then ~1.1 h), corpus / node-set
comparison, census, NPMI, conversion sidecar (key moved: rebuilds), ribbon readout (the
long step, 0163: 5.5 h), detection and new-diagnosis comparisons, workup sidecar
(rebuilds), framing-1 survey, workup survey.

The job body is also committed as [`scripts/jobs/exp_0168.sh`](../../scripts/jobs/exp_0168.sh) (same steps); the launch that chains 0168 and then [0169](0169-descriptive-model-of-record-0163-config-long-fit.md) is:

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUNS=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs
mkdir -p $RUNS/0168-dag-gated-api-validation-on-0163-config $RUNS/0169-descriptive-model-of-record-0163-config-long-fit
L8=$RUNS/0168-dag-gated-api-validation-on-0163-config/job_log.md
L9=$RUNS/0169-descriptive-model-of-record-0163-config-long-fit/job_log.md
nohup bash -c 'bash scripts/jobs/exp_0168.sh > "'$L8'" 2>&1; bash scripts/jobs/exp_0169.sh > "'$L9'" 2>&1' > /dev/null 2>&1 &
sleep 5; ls -la "$L8"; tail -f "$L8"
```

The original inline form:

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUNS=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs
RIBBON=$HOME/repos/CHARMPheno/analysis/cloud/label_sets/dismech_ribbon.tsv
R152=$(ls -d $RUNS/0152-*); R163=$(ls -d $RUNS/0163-*)
ls -la $RIBBON $R152/groups_4b.json $R163/bundle_meta.json $R163/census_own10.md $R163/npmi_top10.json $R163/prevalence_check_stacked_nodes_mass99.json $R163/differential_survey_inc_top.json
mkdir -p $RUNS/0168-dag-gated-api-validation-on-0163-config
L=$RUNS/0168-dag-gated-api-validation-on-0163-config/job_log.md
nohup bash -c '
set -o pipefail
RIBBON='$RIBBON'; R152='$R152'; R163='$R163'
echo "### fit 168"
make -C analysis/cloud exp ID=168 DGR_SPARK_CONF=spark.eventLog.enabled=false || { echo "FAILED: fit 168"; exit 1; }
R168=$(ls -d '$RUNS'/0168-*)
echo "### corpus + node set: 0163 vs 0168"
python3 - "$R163" "$R168" <<"PY" || { echo "FAILED: the corpus differs from 0163 (stop and report)"; exit 1; }
import json, sys
from pathlib import Path
A, B = (json.loads((Path(r) / "bundle_meta.json").read_text()) for r in sys.argv[1:3])
va_all, vb_all = A["vocab_maps"], B["vocab_maps"]
ok = len(va_all) == len(vb_all)
print(f"domains: 0163 {len(va_all)}, 0168 {len(vb_all)}")
for d, (va, vb) in enumerate(zip(va_all, vb_all)):
    same = sum(1 for t, i in va.items() if vb.get(t) == i)
    print(f"domain {d}: V 0163 {len(va)}, 0168 {len(vb)}; same term at the same index {same}")
    ok = ok and same == len(va) == len(vb)
la, lb = A.get("ledger") or {}, B.get("ledger") or {}
print("ledger 0163:", json.dumps(la))
print("ledger 0168:", json.dumps(lb))
for k in sorted(set(la) & set(lb)):
    print(f"  shared key {k}: 0163 {la[k]}, 0168 {lb[k]}")
ca = {int(c) for c in A.get("int2cid", {}).values() if int(c) >= 0}
cb = {int(c) for c in B.get("int2cid", {}).values() if int(c) >= 0}
print(f"fitted nodes by Mondo id: 0163 {len(ca)}, 0168 {len(cb)}; shared {len(ca & cb)}, "
      f"only 0163 {len(ca - cb)}, only 0168 {len(cb - ca)}")
n_given = len({e[0] for e in B["label_dag"]})   # an edge list [[child, parent|null], ...]
print(f"label subgraph handed to the 0168 estimator: {n_given} Mondo terms")
print("VOCABULARIES IDENTICAL" if ok else "VOCABULARIES DIFFER")
sys.exit(0 if ok else 1)
PY
echo "-- 0163 build receipts"
grep -h -m3 -E "\[mondo-native\] (powering|label DAG)" "$R163"/driver_log.md | cut -c1-600
echo "-- 0168 fit receipts"
grep -h -E "\[dag-gated-lda\] (node set|leaves-first gate)|\[driver\] +(ledger|label DAG|corpus|node set):" "$R168"/driver_log.md | cut -c1-600
echo "### census 168 (then 0163 for reference)"
make -s -C analysis/cloud inspect-topics ID=168 RESOLVE_NAMES=1 INSPECT_ARGS="--profile-census --own-codes --own-top 10 --out $R168/census_own10.md" > /dev/null
for R in "$R168" "$R163"; do echo "-- $(basename $R)"; sed -n "/^block topics/p;/^own-code-decidable/p;/^depth · nodes/,/^\$/p" "$R"/census_own10.md; done
echo "### npmi 168"
make -C analysis/cloud dag-gated-readout ID=168 DGR_SPARK_CONF=spark.eventLog.enabled=false DGR_ARGS="--npmi --npmi-top 10 --npmi-domains 0,2 --acq-tag top10" || { echo "FAILED: npmi 168"; exit 1; }
echo "### npmi 0163 vs 0168"
python3 analysis/cloud/topic_npmi.py compare "$R163" "$R168" top10
echo "### conversion sidecar 168 (key moved: rebuilds)"
make -C analysis/cloud build-conversion-sidecar ID=168 || { echo "FAILED: conversion sidecar 168"; exit 1; }
echo "### ribbon readout 168 (stacked, top-m 100)"
make -C analysis/cloud dag-gated-readout ID=168 DGR_SPARK_CONF=spark.eventLog.enabled=false DGR_ARGS="--readout-l2 100 --readout-stacked --readout-hierarchy mondo --readout-nodes $RIBBON --prev-history" || { echo "FAILED: readout 168"; exit 1; }
echo "### stacked detection 0163 vs 0168"
grep -h -m1 -A6 -F "[stacked readout: gated_pc_stacked_nodes_mass99]" "$R163"/readout_log.md "$R163"/job_log.md | cut -c1-220
grep -h -m1 -A6 -F "[stacked readout: gated_pc_stacked_nodes]" "$R168"/readout_log.md | cut -c1-220
echo "### new diagnoses 0163 (A) vs 0168 (B), ribbon nodes by Mondo id"
python3 analysis/cloud/compare_incident_runs.py "$R163" prevalence_check_stacked_nodes_mass99.json "$R168" prevalence_check_stacked_nodes.json | cut -c1-220
echo "### workup sidecar 168 (keyed on the conversion key: rebuilds)"
make -C analysis/cloud build-workup-sidecar ID=168 || { echo "FAILED: workup sidecar 168"; exit 1; }
BASE="--differential . --diff-survey 25 --diff-members top --diff-incident --diff-groups-like $R152/groups_4b.json --acq-heads stacked_nodes --diff-max-patients 1500 --diff-crossfit"
echo "### survey (9 same-question groups) 168"
make -C analysis/cloud dag-gated-readout ID=168 DGR_SPARK_CONF=spark.eventLog.enabled=false DGR_ARGS="$BASE --acq-tag survey_0168" || { echo "FAILED: survey 168"; exit 1; }
echo "### framings 0168 vs 0163 (same-question groups)"
python3 analysis/cloud/survey_analysis.py "$R168" survey_0168 "$R163" survey_inc_top | cut -c1-220
echo "### workup survey 168 (strict window, new facts only)"
make -C analysis/cloud dag-gated-readout ID=168 DGR_SPARK_CONF=spark.eventLog.enabled=false DGR_ARGS="$BASE --diff-workup --acq-tag survey_0168_workup" || { echo "FAILED: workup 168"; exit 1; }
python3 analysis/cloud/survey_analysis.py "$R168" survey_0168_workup | sed -n "/## workup/,/^E5 reasons/p" | cut -c1-260
echo "-- 0163 workup surveys on record (newest first; compare with the newest)"
ls -t "$R163"/differential_survey_0167_*.json | head -3
echo "### DONE"
' > "$L" 2>&1 &
sleep 2; ls -la "$L"; tail -f "$L"
```

## What to look for in the log

- **Fit, early (bundle).** `[driver]   ledger: {"n_dag_nodes": …, "train_fg_docs": …,
  "test_fg_docs": …}`, `[driver]   label DAG: N node(s) given; gate labels
  'gate_labels'; nodeTerms for … node(s)`, `[driver]   corpus: V=(condition:… +
  measurement:… + drug:…) vocab` (0163's three widths should repeat).
- **Fit, node set.** The estimator's line `[dag-gated-lda] node set: D docs; nodes given G
  / kept K / pruned P / spliced S (minDocsPerNode=100)`, then the driver's receipt
  `[driver]   node set: G given / K kept / P pruned / S spliced (minDocsPerNode=100,
  counted on D train docs); K=… (200 bg + K nodes x 1), C=… heads`. Against 0163: its
  `[mondo-native] label DAG …: n powered term(s) …; induced Hasse … node(s) …` line
  (printed above as the 0163 receipt), and K_nodes 2,694.
- **Fit, gate.** `[dag-gated-lda] leaves-first gate: 2 iters/level over heights 0..H (full
  gate from iter 2H); open-time seed ON (balanced); leaf blocks seeded at iteration 0
  from their own documents`.
- **Corpus comparison.** `VOCABULARIES IDENTICAL` and `same term at the same index` equal
  to V in every domain; the shared ledger key `test_fg_docs`; the shared / only-0163 /
  only-0168 node counts.
- **Readout.** `[readout]   fitted model loaded: K kept node(s), …`, `theta top-m=100
  (from default)`, `node subset …: N terms -> … of … readout heads kept`, `STACKED arm
  requested: results -> results_readout_stacked_nodes.json`; the drift gate must not
  print `DRIFTED`.
- **Comparisons.** The detection blocks (flat / root head alone / stacked AUC);
  `compare_incident_runs.py`'s `## history_stacked — n nodes shared` and `history_flat`
  lines (A = 0163, B = 0168: macro, median B − A, wins, by incident band); the census
  rows; `survey_analysis.py`'s paired table and the workup section.

## Run log
