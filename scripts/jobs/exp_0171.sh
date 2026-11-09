#!/usr/bin/env bash
# Exp 0171 job (docs/experiments/0171-*.md). Run from the repo root on the cluster.
set -o pipefail
RUNS=${RUNS:-/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs}
export PYTHONPATH="$PWD:$PWD/spark-vi:$PWD/charmpheno:$PWD/analysis/cloud"
echo "### offline starvation report: 0168, 0170, 0169 (the last is the one followed back)"
RUNS_IN=""
for id in 0168 0170 0169; do R=$(ls -d "$RUNS"/$id-* 2>/dev/null | head -1); [ -n "$R" ] && [ -d "$R"/model/result ] && RUNS_IN="$RUNS_IN $R"; done
[ -n "$RUNS_IN" ] && python3 analysis/cloud/starvation.py $RUNS_IN || echo "WARN: no fitted models found for the report"
echo "### fit 171 (0169's configuration, traced)"
make -C analysis/cloud exp ID=171 DGR_SPARK_CONF=spark.eventLog.enabled=false || { echo "FAILED: fit 171"; exit 1; }
R=$(ls -d "$RUNS"/0171-*)
echo "### trace 171"; grep -h "^\[trace\]" "$R"/job_log.md "$R"/driver_log.md 2>/dev/null | sort -u -t: -k1,1 | sort -t' ' -k3,3n | cut -c1-160
echo "### starvation report 0169 vs 0171 (the last is followed back)"
python3 analysis/cloud/starvation.py "$(ls -d "$RUNS"/0169-* | head -1)" "$R"
echo "### DONE 171"
