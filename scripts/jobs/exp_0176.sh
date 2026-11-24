#!/usr/bin/env bash
# Exp 0176 job (docs/experiments/0176-*.md): 0175 with seed 43, a seed replicate of the
# flat-block count. Run from the repo root on the cluster with RUNS_DIR exported.
set -o pipefail
: "${RUNS_DIR:?export RUNS_DIR=<the workspace mount>/runs first}"
export RUNS_DIR
export PYTHONPATH="$PWD:$PWD/spark-vi:$PWD/charmpheno:$PWD/analysis/cloud"
R175=$(ls -d "$RUNS_DIR"/0175-* 2>/dev/null | head -1)
SPARK="DGR_SPARK_CONF=spark.eventLog.enabled=false"
echo "### fit 176 (0175 with seed 43)"
make -C analysis/cloud exp ID=176 $SPARK || { echo "FAILED: fit 176"; exit 1; }
R=$(ls -d "$RUNS_DIR"/0176-*)
echo "### receipts 176"
grep -h -E "\[dag-gated-lda\] (node set|leaves-first gate)|topicConcentration|node statistics|dropped [0-9]+ unsupported|\] +node set:|kept the model as fitted|<<< gated_pc fit|saved the fitted model" "$R"/driver_log.md "$R"/job_log.md 2>/dev/null | sort -u | cut -c1-260
echo "### flat blocks before the drop, both rules (0174: 176 / 177, 0175: 195 / 195 by the dominant rule)"
python3 - "$R" <<'PY'
import csv, json, sys
run = sys.argv[1]
man = json.load(open(f"{run}/manifest.json"))
eta = man["topic_concentration"]["eta"]
V = man["domain_vocab_sizes"]
eta = eta * len(V) if len(eta) == 1 else eta
rows = list(csv.DictReader(open(f"{run}/node_statistics.tsv"), delimiter="\t"))
dom_09 = dom_05 = all_09 = all_05 = 0
for r in rows:
    fr = [float(x) for x in r["supportFractionByDomain"].split(",")]
    dm = [float(x) for x in r["dataMassByDomain"].split(",")]
    mass = [d + e * v for d, e, v in zip(dm, eta, V)]       # row mass incl. the prior
    f_dom = fr[max(range(len(mass)), key=lambda m: mass[m])]
    dom_09 += f_dom > 0.9; dom_05 += f_dom > 0.5
    all_09 += min(fr) > 0.9; all_05 += min(fr) > 0.5
print(f"dominant-domain rule (as 0174/0175): > 0.9: {dom_09}, > 0.5: {dom_05} of {len(rows)} nodes")
print(f"every-domain rule (the driver's now): > 0.9: {all_09}, > 0.5: {all_05} of {len(rows)} nodes")
PY
echo "### census 176"
make -s -C analysis/cloud inspect-topics ID=176 RESOLVE_NAMES=1 INSPECT_ARGS="--profile-census --own-codes --own-top 10 --out $R/census_own10.md" > /dev/null || echo "WARN: census failed"
sed -n "/^block topics/p;/^own-code-decidable/p" "$R"/census_own10.md
echo "### npmi 176, then 0175 (A) vs 0176 (B)"
make -C analysis/cloud dag-gated-readout ID=176 $SPARK DGR_ARGS="--npmi --npmi-top 10 --npmi-domains 0,2 --npmi-min-pair-count 20 --acq-tag top10" || echo "WARN: npmi failed"
[ -n "$R175" ] && [ -f "$R175"/npmi_top10.json ] && python3 analysis/cloud/topic_npmi.py compare "$R175" "$R" top10 | grep -E "^#|^domain"
echo "### DONE 176"
