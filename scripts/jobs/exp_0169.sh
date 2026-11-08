#!/usr/bin/env bash
# Exp 0169 job: the descriptive model of record (0163's configuration, 150 iterations).
# Run from the repo root on the cluster (docs/experiments/0169-*.md has the launch line).
set -o pipefail
RUNS=${RUNS:-/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs}
echo "### fit 169"
make -C analysis/cloud exp ID=169 DGR_SPARK_CONF=spark.eventLog.enabled=false || { echo "FAILED: fit 169"; exit 1; }
R169=$(ls -d "$RUNS"/0169-*)
echo "-- 0169 fit receipts"
grep -h -E "\[dag-gated-lda\] (node set|leaves-first gate)|\[driver\] +(ledger|label DAG|corpus|node set):" "$R169"/driver_log.md | cut -c1-600
ls -la "$R169"/model "$R169"/label_dag.json "$R169"/bundle_meta.json || { echo "FAILED: fit outputs missing"; exit 1; }
echo "### census 169"
make -s -C analysis/cloud inspect-topics ID=169 RESOLVE_NAMES=1 INSPECT_ARGS="--profile-census --own-codes --own-top 10 --out $R169/census_own10.md" > /dev/null || echo "WARN: census failed"
sed -n "/^block topics/p;/^own-code-decidable/p;/^depth · nodes/,/^\$/p" "$R169"/census_own10.md
echo "### digest 169"
make -s -C analysis/cloud inspect-topics ID=169 RESOLVE_NAMES=1 INSPECT_ARGS="--digest --digest-domains drug,measurement --out $R169/topics_digest.md" > /dev/null || echo "WARN: digest failed"
head -40 "$R169"/topics_digest.md | cut -c1-260
echo "### full topic report 169 (every node topic, by depth)"
make -s -C analysis/cloud inspect-topics ID=169 RESOLVE_NAMES=1 INSPECT_ARGS="--top-topics 100000 --top-words 20 --sort depth --out $R169/topics_report.md" > /dev/null || echo "WARN: report failed"
ls -la "$R169"/topics_report.md
echo "### npmi 169"
make -C analysis/cloud dag-gated-readout ID=169 DGR_SPARK_CONF=spark.eventLog.enabled=false DGR_ARGS="--npmi --npmi-top 10 --npmi-domains 0,2 --acq-tag top10" || echo "WARN: npmi failed"
R168=$(ls -d "$RUNS"/0168-* 2>/dev/null | head -1)
if [ -n "$R168" ] && [ -f "$R168"/npmi_top10.json ]; then echo "### npmi 0168 vs 0169"; PYTHONPATH="$PWD:$PWD/spark-vi:$PWD/charmpheno" python3 analysis/cloud/topic_npmi.py compare "$R168" "$R169" top10; fi
echo "### DONE 169"
