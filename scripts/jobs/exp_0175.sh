#!/usr/bin/env bash
# Exp 0175 job (docs/experiments/0175-*.md): 0174 with eta = 1/V_m per domain, compared
# with 0174 on the pre-registered criteria. Run from the repo root on the cluster with
# RUNS_DIR exported (the v9 workspace's runs/).
set -o pipefail
: "${RUNS_DIR:?export RUNS_DIR=<the workspace mount>/runs first}"
export RUNS_DIR
export PYTHONPATH="$PWD:$PWD/spark-vi:$PWD/charmpheno:$PWD/analysis/cloud"
RIBBON=${RIBBON:-$PWD/analysis/cloud/label_sets/dismech_ribbon.tsv}
[ -e "$RIBBON" ] || { echo "FAILED: ribbon missing: $RIBBON"; exit 1; }
R174=$(ls -d "$RUNS_DIR"/0174-* 2>/dev/null | head -1)
[ -n "$R174" ] || { echo "FAILED: no 0174 run under $RUNS_DIR to compare with"; exit 1; }
SPARK="DGR_SPARK_CONF=spark.eventLog.enabled=false"
READOUT_ARGS="--readout-l2 100 --readout-stacked --readout-hierarchy mondo --readout-nodes $RIBBON --prev-history"

echo "### fit 175 (0174 + topic_concentration inv_vocab)"
make -C analysis/cloud exp ID=175 $SPARK || { echo "FAILED: fit 175"; exit 1; }
R=$(ls -d "$RUNS_DIR"/0175-*)
echo "### receipts 175"
grep -h -E "\[dag-gated-lda\] (node set|leaves-first gate)|topicConcentration|node statistics|dropped [0-9]+ unsupported|\] +node set:|case-finding-cache (HIT|MISS)|<<< gated_pc fit|saved the fitted model" "$R"/driver_log.md "$R"/job_log.md 2>/dev/null | sort -u | cut -c1-260
echo "### criterion 1: flat blocks before the drop (counts only), 0174 then 0175"
for X in "$R174" "$R"; do
  python3 - "$X"/node_statistics.tsv <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
sf = [float(r["supportFraction"]) for r in rows if r["supportFraction"]]
print(f"{sys.argv[1].split('/')[-2]}: supportFraction > 0.9: {sum(x > 0.9 for x in sf)}, "
      f"> 0.5: {sum(x > 0.5 for x in sf)} of {len(sf)} nodes")
PY
done
echo "### trace 175 (every 20)"; grep -h "^\[trace\]" "$R"/job_log.md | sort -u | sort -t' ' -k3,3n | awk 'NR%4==1' | cut -c1-120
echo "### census 175"
make -s -C analysis/cloud inspect-topics ID=175 RESOLVE_NAMES=1 INSPECT_ARGS="--profile-census --own-codes --own-top 10 --out $R/census_own10.md" > /dev/null || echo "WARN: census failed"
for X in "$R174" "$R"; do echo "# $(basename "$X")"; sed -n "/^block topics/p;/^own-code-decidable/p" "$X"/census_own10.md 2>/dev/null; done
echo "### criterion 4: npmi 175 (pair minimum 20), then 0174 (A) vs 0175 (B)"
make -C analysis/cloud dag-gated-readout ID=175 $SPARK DGR_ARGS="--npmi --npmi-top 10 --npmi-domains 0,2 --npmi-min-pair-count 20 --acq-tag top10" || echo "WARN: npmi failed"
[ -f "$R174"/npmi_top10.json ] && python3 analysis/cloud/topic_npmi.py compare "$R174" "$R" top10 | grep -E "^#|^domain"
echo "### conversion sidecar (shared by 0174 and 0175: the key does not depend on the fit)"
make -C analysis/cloud build-conversion-sidecar ID=175 || { echo "FAILED: conversion sidecar"; exit 1; }
if [ ! -e "$R174"/prevalence_check_stacked_nodes.json ]; then
  echo "### ribbon readout 0174 (missing: 0174's job did not run it)"
  make -C analysis/cloud dag-gated-readout ID=174 $SPARK DGR_ARGS="$READOUT_ARGS" || { echo "FAILED: readout 174"; exit 1; }
fi
echo "### ribbon readout 175 (stacked, top-m 100)"
make -C analysis/cloud dag-gated-readout ID=175 $SPARK DGR_ARGS="$READOUT_ARGS" || { echo "FAILED: readout 175"; exit 1; }
echo "### criterion 2: stacked detection and ranking, 0174 then 0175"
for X in "$R174" "$R"; do echo "# $(basename "$X")"; grep -h -m1 -A6 -F "[stacked readout: gated_pc_stacked_nodes]" "$X"/readout_log.md | cut -c1-200; grep -h -m1 "ranking (within parent cohort" "$X"/readout_log.md | cut -c1-200; done
echo "### criterion 3: new diagnoses 0174 (A) vs 0175 (B), ribbon nodes by Mondo id"
python3 analysis/cloud/compare_incident_runs.py "$R174" prevalence_check_stacked_nodes.json "$R" prevalence_check_stacked_nodes.json | cut -c1-220
echo "### DONE 175"
