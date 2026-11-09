#!/usr/bin/env bash
# Exp 0170 job: 0168's configuration with the fit on the local engine (LocalRunner over
# forked processes on the master). Run from the repo root on the cluster.
set -o pipefail
RUNS=${RUNS:-/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs}
export PYTHONPATH="$PWD:$PWD/spark-vi:$PWD/charmpheno"
echo "### machine"; nproc; free -g | head -2
echo "### fit 170 (local engine)"
make -C analysis/cloud exp ID=170 DGR_SPARK_CONF=spark.eventLog.enabled=false || { echo "FAILED: fit 170"; exit 1; }
R170=$(ls -d "$RUNS"/0170-*); R168=$(ls -d "$RUNS"/0168-* 2>/dev/null | head -1)
echo "### receipts"
grep -h -E "\[dag-gated-lda\] (node set|leaves-first gate|local engine)|case-finding-cache (HIT|MISS)|<<< gated_pc fit|saved the fitted model" "$R170"/driver_log.md | cut -c1-240
echo "### iteration wall time: 0168 (spark) vs 0170 (local)"
for R in "$R168" "$R170"; do
  [ -n "$R" ] || continue
  # The SVI loop's iteration lines land in the job log (the driver log tees phases).
  T=$(cat "$R"/job_log.md "$R"/driver_log.md 2>/dev/null | grep -E "^\[driver\]   iter [0-9]+/" | sort -u | sed -E 's/.*, ([0-9.]+)s$/\1/')
  n=$(echo "$T" | grep -c .); [ "$n" -gt 0 ] || { echo "$(basename "$R"): no iteration lines"; continue; }
  med=$(echo "$T" | sort -n | sed -n "$(( (n + 1) / 2 ))p")
  echo "$T" | awk -v r="$(basename "$R")" -v med="$med" '{s+=$1; n++} END {printf "%s: %d iters, mean %.1fs, median %ss, total %.0fs\n", r, n, s/n, med, s}'
  cat "$R"/job_log.md "$R"/driver_log.md 2>/dev/null | grep -E "^\[driver\]   iter (1|30|60)/" | sort -u | cut -c1-110
done
echo "### census 170 (then 0168)"
make -s -C analysis/cloud inspect-topics ID=170 RESOLVE_NAMES=1 INSPECT_ARGS="--profile-census --own-codes --own-top 10 --out $R170/census_own10.md" > /dev/null || echo "WARN: census failed"
for R in "$R170" "$R168"; do [ -f "$R"/census_own10.md ] && { echo "-- $(basename $R)"; sed -n "/^block topics/p;/^own-code-decidable/p" "$R"/census_own10.md; }; done
echo "### npmi 170"
make -C analysis/cloud dag-gated-readout ID=170 DGR_SPARK_CONF=spark.eventLog.enabled=false DGR_ARGS="--npmi --npmi-top 10 --npmi-domains 0,2 --acq-tag top10" || echo "WARN: npmi failed"
[ -f "$R168"/npmi_top10.json ] && { echo "### npmi 0168 (A) vs 0170 (B)"; python3 analysis/cloud/topic_npmi.py compare "$R168" "$R170" top10 | grep -E "^#|^domain"; }
echo "### DONE 170"
