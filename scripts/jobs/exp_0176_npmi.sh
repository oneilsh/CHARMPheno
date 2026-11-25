#!/usr/bin/env bash
# Exp 0176 follow-up (docs/experiments/0176-*.md): the identity census and the topic
# NPMI the fit job skipped, the NPMI comparison with 0175, then the dashboard export
# again so it carries the NPMI. Run from the repo root on the cluster with RUNS_DIR
# exported, as for the fit job. Writes its log to <run>/npmi_job_log.md. The
# readout rebuilds the bundle on a fresh cluster (HDFS cache MISS): use the
# n2-standard-8 master.
set -o pipefail
: "${RUNS_DIR:?export RUNS_DIR=<the workspace mount>/runs first}"
export RUNS_DIR
export PYTHONPATH="$PWD:$PWD/spark-vi:$PWD/charmpheno:$PWD/analysis/cloud"
R=$(ls -d "$RUNS_DIR"/0176-* 2>/dev/null | head -1)
[ -n "$R" ] || { echo "FAILED: no 0176 run under $RUNS_DIR"; exit 1; }
R175=$(ls -d "$RUNS_DIR"/0175-* 2>/dev/null | head -1)
SPARK="DGR_SPARK_CONF=spark.eventLog.enabled=false"
exec > >(tee -a "$R"/npmi_job_log.md) 2>&1
echo "### census 176"
make -s -C analysis/cloud inspect-topics ID=176 RESOLVE_NAMES=1 INSPECT_ARGS="--profile-census --own-codes --own-top 10 --out $R/census_own10.md" > /dev/null || echo "WARN: census failed"
for X in "$R175" "$R"; do echo "# $(basename "$X")"; sed -n "/^block topics/p;/^own-code-decidable/p" "$X"/census_own10.md 2>/dev/null; done
echo "### npmi 176 (pair minimum 20), then 0175 (A) vs 0176 (B)"
make -C analysis/cloud dag-gated-readout ID=176 $SPARK DGR_ARGS="--npmi --npmi-top 10 --npmi-domains 0,2 --npmi-min-pair-count 20 --acq-tag top10" || { echo "FAILED: npmi 176"; exit 1; }
[ -n "$R175" ] && [ -f "$R175"/npmi_top10.json ] && python3 analysis/cloud/topic_npmi.py compare "$R175" "$R" top10 | grep -E "^#|^domain"
echo "### export 176 (now with NPMI)"
make -C analysis/cloud export-dag-gated ID=176 RESOLVE_NAMES=1 || { echo "FAILED: export 176"; exit 1; }
echo "### DONE 176 npmi"
