#!/usr/bin/env bash
# Exp 0174 job (docs/experiments/0174-*.md): the model of record on the v9 workspace.
# Run from the repo root on the cluster with RUNS_DIR exported (the new workspace's runs/).
set -o pipefail
: "${RUNS_DIR:?export RUNS_DIR=<the workspace mount>/runs first}"
export RUNS_DIR
export PYTHONPATH="$PWD:$PWD/spark-vi:$PWD/charmpheno:$PWD/analysis/cloud"
echo "### fit 174 (0173's configuration: gate_warm 4, 150 iterations, drop supportFraction > 0.9)"
make -C analysis/cloud exp ID=174 DGR_SPARK_CONF=spark.eventLog.enabled=false || { echo "FAILED: fit 174"; exit 1; }
R=$(ls -d "$RUNS_DIR"/0174-*)
echo "### workspace (from make setup)"; grep -E "^(export )?WORKSPACE_CDR=" analysis/cloud/.workspace_env 2>/dev/null | head -1
echo "### receipts"
grep -h -E "\[dag-gated-lda\] (node set|leaves-first gate)|node statistics|dropped [0-9]+ unsupported|\] +node set:|case-finding-cache (HIT|MISS)|<<< gated_pc fit|saved the fitted model|\] +corpus:|\] +ledger:" "$R"/driver_log.md "$R"/job_log.md 2>/dev/null | sort -u | cut -c1-260
echo "### trace (every 20)"; grep -h "^\[trace\]" "$R"/job_log.md | sort -u | sort -t' ' -k3,3n | awk 'NR%4==1' | cut -c1-120
echo "### census 174"
make -s -C analysis/cloud inspect-topics ID=174 RESOLVE_NAMES=1 INSPECT_ARGS="--profile-census --own-codes --own-top 10 --out $R/census_own10.md" > /dev/null || echo "WARN: census failed"
sed -n "/^block topics/p;/^own-code-decidable/p" "$R"/census_own10.md
echo "### npmi 174 (pair minimum 20: exportable)"
make -C analysis/cloud dag-gated-readout ID=174 DGR_SPARK_CONF=spark.eventLog.enabled=false DGR_ARGS="--npmi --npmi-top 10 --npmi-domains 0,2 --npmi-min-pair-count 20 --acq-tag top10" || echo "WARN: npmi failed"
grep -h -E "median NPMI" "$R"/npmi_top10.md 2>/dev/null | head -6
echo "### export (dag_export/ + dag_view.html)"
make -C analysis/cloud export-dag-gated ID=174 RESOLVE_NAMES=1 || echo "WARN: export failed"
ls -la "$R"/dag_view.html 2>/dev/null
echo "### DONE 174"
