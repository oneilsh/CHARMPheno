#!/usr/bin/env bash
# Exp 0172 job (docs/experiments/0172-*.md). Run from the repo root on the cluster.
set -o pipefail
RUNS=${RUNS:-/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs}
export PYTHONPATH="$PWD:$PWD/spark-vi:$PWD/charmpheno:$PWD/analysis/cloud"
echo "### fit 172 (gate_warm 4, traced)"
make -C analysis/cloud exp ID=172 DGR_SPARK_CONF=spark.eventLog.enabled=false || { echo "FAILED: fit 172"; exit 1; }
R=$(ls -d "$RUNS"/0172-*)
echo "### trace 172"; grep -h "^\[trace\]" "$R"/job_log.md "$R"/driver_log.md 2>/dev/null | sort -u -t: -k1,1 | sort -t' ' -k3,3n | cut -c1-160
R171=$(ls -d "$RUNS"/0171-* 2>/dev/null | head -1)
echo "### starvation report 0171 vs 0172 (the last is followed back)"
python3 analysis/cloud/starvation.py $R171 "$R"
echo "### census 172"
make -s -C analysis/cloud inspect-topics ID=172 RESOLVE_NAMES=1 INSPECT_ARGS="--profile-census --own-codes --own-top 10 --out $R/census_own10.md" > /dev/null || echo "WARN: census failed"
sed -n "/^block topics/p;/^depth · nodes/,/^\$/p" "$R"/census_own10.md
echo "### DONE 172"
