#!/usr/bin/env bash
# Exp 0168 job: the DAG-gated API redesign validated on 0163's configuration.
# Run from the repo root on the cluster (docs/experiments/0168-*.md has the launch line).
set -o pipefail
RUNS=${RUNS:-/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs}
RIBBON=${RIBBON:-$PWD/analysis/cloud/label_sets/dismech_ribbon.tsv}
R152=$(ls -d "$RUNS"/0152-*); R163=$(ls -d "$RUNS"/0163-*)
# Required inputs stop the job; 0163's comparison files only skip their comparison.
for f in "$RIBBON" "$R152"/groups_4b.json; do
  [ -e "$f" ] || { echo "FAILED: required input missing: $f"; exit 1; }
done
for f in bundle_meta.json census_own10.md npmi_top10.json \
         prevalence_check_stacked_nodes_mass99.json differential_survey_inc_top.json; do
  if [ -e "$R163/$f" ]; then echo "found 0163 $f"; else echo "WARN: 0163 $f missing: its comparison is skipped"; fi
done
echo "### fit 168"
make -C analysis/cloud exp ID=168 DGR_SPARK_CONF=spark.eventLog.enabled=false || { echo "FAILED: fit 168"; exit 1; }
R168=$(ls -d "$RUNS"/0168-*)
echo "### corpus + node set: 0163 vs 0168"
if [ -e "$R163"/bundle_meta.json ]; then
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
else echo "SKIP: 0163 bundle_meta.json missing (corpus check not run; fit receipts below)"; fi
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
[ -e "$R163"/npmi_top10.json ] && python3 analysis/cloud/topic_npmi.py compare "$R163" "$R168" top10
echo "### conversion sidecar 168 (key moved: rebuilds)"
make -C analysis/cloud build-conversion-sidecar ID=168 || { echo "FAILED: conversion sidecar 168"; exit 1; }
echo "### ribbon readout 168 (stacked, top-m 100)"
make -C analysis/cloud dag-gated-readout ID=168 DGR_SPARK_CONF=spark.eventLog.enabled=false DGR_ARGS="--readout-l2 100 --readout-stacked --readout-hierarchy mondo --readout-nodes $RIBBON --prev-history" || { echo "FAILED: readout 168"; exit 1; }
echo "### stacked detection 0163 vs 0168"
grep -h -m1 -A6 -F "[stacked readout: gated_pc_stacked_nodes_mass99]" "$R163"/readout_log.md "$R163"/job_log.md | cut -c1-220
grep -h -m1 -A6 -F "[stacked readout: gated_pc_stacked_nodes]" "$R168"/readout_log.md | cut -c1-220
echo "### new diagnoses 0163 (A) vs 0168 (B), ribbon nodes by Mondo id"
[ -e "$R163"/prevalence_check_stacked_nodes_mass99.json ] && python3 analysis/cloud/compare_incident_runs.py "$R163" prevalence_check_stacked_nodes_mass99.json "$R168" prevalence_check_stacked_nodes.json | cut -c1-220
echo "### workup sidecar 168 (keyed on the conversion key: rebuilds)"
make -C analysis/cloud build-workup-sidecar ID=168 || { echo "FAILED: workup sidecar 168"; exit 1; }
BASE="--differential . --diff-survey 25 --diff-members top --diff-incident --diff-groups-like $R152/groups_4b.json --acq-heads stacked_nodes --diff-max-patients 1500 --diff-crossfit"
echo "### survey (9 same-question groups) 168"
make -C analysis/cloud dag-gated-readout ID=168 DGR_SPARK_CONF=spark.eventLog.enabled=false DGR_ARGS="$BASE --acq-tag survey_0168" || { echo "FAILED: survey 168"; exit 1; }
echo "### framings 0168 vs 0163 (same-question groups)"
[ -e "$R163"/differential_survey_inc_top.json ] && python3 analysis/cloud/survey_analysis.py "$R168" survey_0168 "$R163" survey_inc_top | cut -c1-220
echo "### workup survey 168 (strict window, new facts only)"
make -C analysis/cloud dag-gated-readout ID=168 DGR_SPARK_CONF=spark.eventLog.enabled=false DGR_ARGS="$BASE --diff-workup --acq-tag survey_0168_workup" || { echo "FAILED: workup 168"; exit 1; }
python3 analysis/cloud/survey_analysis.py "$R168" survey_0168_workup | sed -n "/## workup/,/^E5 reasons/p" | cut -c1-260
echo "-- 0163 workup surveys on record (newest first; compare with the newest)"
ls -t "$R163"/differential_survey_0167_*.json | head -3
echo "### DONE"
