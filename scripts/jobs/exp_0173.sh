#!/usr/bin/env bash
# Exp 0173 job (docs/experiments/0173-*.md): the first cleaned model of record.
# Run from the repo root on the cluster.
set -o pipefail
RUNS=${RUNS:-/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs}
export PYTHONPATH="$PWD:$PWD/spark-vi:$PWD/charmpheno:$PWD/analysis/cloud"
RIBBON=${RIBBON:-$PWD/analysis/cloud/label_sets/dismech_ribbon.tsv}
[ -e "$RIBBON" ] || { echo "FAILED: ribbon missing: $RIBBON"; exit 1; }
R168=$(ls -d "$RUNS"/0168-* 2>/dev/null | head -1)
echo "### fit 173 (gate_warm 4, 150 iterations, drop supportFraction > 0.9)"
make -C analysis/cloud exp ID=173 DGR_SPARK_CONF=spark.eventLog.enabled=false || { echo "FAILED: fit 173"; exit 1; }
R=$(ls -d "$RUNS"/0173-*)
echo "### receipts"
grep -h -E "\[dag-gated-lda\] (node set|leaves-first gate)|node statistics|dropped [0-9]+ unsupported|\] +node set:|case-finding-cache (HIT|MISS)|<<< gated_pc fit|saved the fitted model" "$R"/driver_log.md "$R"/job_log.md 2>/dev/null | sort -u | cut -c1-260
echo "### flat blocks before the drop, at 0.5 and 0.9 (from node_statistics.tsv; counts only)"
python3 - "$R"/node_statistics.tsv <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
sf = [float(r["supportFraction"]) for r in rows if r["supportFraction"]]
for t in (0.5, 0.9):
    print(f"supportFraction > {t}: {sum(x > t for x in sf)} of {len(sf)} nodes")
print(f"in (0.5, 0.9]: {sum(0.5 < x <= 0.9 for x in sf)} nodes (kept: the marginal band)")
PY
echo "### trace (every 20)"; grep -h "^\[trace\]" "$R"/job_log.md | sort -u | sort -t' ' -k3,3n | awk 'NR%4==1' | cut -c1-120
echo "### starvation report on the cleaned model"
python3 analysis/cloud/starvation.py "$R" | grep -E "^## |^  (support|parent share):" | cut -c1-200
echo "### census 173 (cleaned)"
make -s -C analysis/cloud inspect-topics ID=173 RESOLVE_NAMES=1 INSPECT_ARGS="--profile-census --own-codes --own-top 10 --out $R/census_own10.md" > /dev/null || echo "WARN: census failed"
sed -n "/^block topics/p;/^own-code-decidable/p" "$R"/census_own10.md
echo "### npmi 173"
make -C analysis/cloud dag-gated-readout ID=173 DGR_SPARK_CONF=spark.eventLog.enabled=false DGR_ARGS="--npmi --npmi-top 10 --npmi-domains 0,2 --acq-tag top10" || echo "WARN: npmi failed"
[ -f "$R168"/npmi_top10.json ] && { echo "### npmi 0168 (A) vs 0173 (B)"; python3 analysis/cloud/topic_npmi.py compare "$R168" "$R" top10 | grep -E "^#|^domain"; }
echo "### conversion sidecar 173"
make -C analysis/cloud build-conversion-sidecar ID=173 || { echo "FAILED: conversion sidecar 173"; exit 1; }
echo "### ribbon readout 173 (stacked, top-m 100)"
make -C analysis/cloud dag-gated-readout ID=173 DGR_SPARK_CONF=spark.eventLog.enabled=false DGR_ARGS="--readout-l2 100 --readout-stacked --readout-hierarchy mondo --readout-nodes $RIBBON --prev-history" || { echo "FAILED: readout 173"; exit 1; }
echo "### stacked detection 0168 vs 0173"
for X in "$R168" "$R"; do grep -h -m1 -A6 -F "[stacked readout: gated_pc_stacked_nodes]" "$X"/readout_log.md | cut -c1-200; done
echo "### new diagnoses 0168 (A) vs 0173 (B), ribbon nodes by Mondo id"
[ -e "$R168"/prevalence_check_stacked_nodes.json ] && python3 analysis/cloud/compare_incident_runs.py "$R168" prevalence_check_stacked_nodes.json "$R" prevalence_check_stacked_nodes.json | cut -c1-220
echo "### DONE 173"
