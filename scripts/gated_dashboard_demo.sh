#!/usr/bin/env bash
# scripts/gated_dashboard_demo.sh - end-to-end local gated STM dashboard demo.
#
# Chains simulate_gated_omop -> fit_stm_local -> build_dashboard, copies the
# resulting bundle into dashboard/public/data/gated_demo/, and ensures the
# dashboard cohort manifest lists it. Idempotent; everything runs under
# `poetry run` (the root .venv has a working distutils for the Spark workers).
#
# Defaults: N=5000 patients, rare_dx group at ~1% (~50 patients, well over the
# k=20 small-cell floor) so its foreground topics survive k-anon and appear in
# the dashboard. Override with N=... SEED=... env vars.
set -euo pipefail
cd "$(dirname "$0")/.."

SEED=${SEED:-0}; N=${N:-5000}
OMOP="data/simulated/gated_omop_N${N}_seed${SEED}.parquet"
PERSON="data/simulated/gated_person_N${N}_seed${SEED}.parquet"
RUN_DIR="data/runs/gated_demo"
BUNDLE_DIR="${RUN_DIR}/dashboard_bundle"
DEST="dashboard/public/data/gated_demo"

echo "[demo] 1/4 simulate gated OMOP (N=${N}, seed=${SEED})"
poetry run python scripts/simulate_gated_omop.py --n-patients "$N" --seed "$SEED" \
    --background-k 3 --foreground rare_dx:2 \
    --group-props common:0.99,rare_dx:0.01 --age-means common:55,rare_dx:72 \
    --n-background-concepts 40 --n-group-concepts 12

echo "[demo] 2/4 fit gated STM (K=5: 3 background + 2 rare_dx foreground)"
poetry run python analysis/local/fit_stm_local.py \
    --omop "$OMOP" --person "$PERSON" \
    --K 5 --background-k 3 --foreground rare_dx:2 \
    --covariate-formula "~ C(sex) + age" --max-iter 40 --out-dir "$RUN_DIR"

echo "[demo] 3/4 build dashboard bundle (masked prevalence + gating.json)"
poetry run python analysis/local/build_dashboard.py \
    --checkpoint "$RUN_DIR" --input "$OMOP" --out-dir "$BUNDLE_DIR"

echo "[demo] 4/4 publish bundle to ${DEST} and register the cohort"
mkdir -p "$DEST"
cp "$BUNDLE_DIR"/*.json "$DEST"/

# Idempotently ensure a gated_demo cohort entry in the dashboard manifest.
poetry run python - <<'PY'
import json
from pathlib import Path

manifest = Path("dashboard/public/data/manifest.json")
data = json.loads(manifest.read_text())
cohorts = data.setdefault("cohorts", [])
if not any(c.get("id") == "gated_demo" for c in cohorts):
    cohorts.append({
        "id": "gated_demo",
        "label": "Gated STM demo (background + rare_dx foreground)",
        "description": (
            "Local synthetic demo of the gated background/foreground STM. "
            "A shared background block (all patients) plus a rare_dx "
            "foreground block (~1% of the cohort, above the k=20 small-cell "
            "floor). In covariate mode, the Group selector switches between "
            "'Background only' and 'rare_dx'; the rare_dx foreground topics "
            "appear only for the rare_dx group and vanish for background-only."
        ),
    })
    manifest.write_text(json.dumps(data, indent=2) + "\n")
    print("[demo] registered gated_demo cohort in manifest.json")
else:
    print("[demo] gated_demo cohort already in manifest.json")
PY

echo
echo "[demo] Bundle ready in ${DEST} (gating.json present)."
echo "[demo] Run: cd dashboard && npm install && npm run dev  -> select cohort 'gated_demo'"
echo "[demo] In the Atlas: toggle covariate mode, then switch the Group selector"
echo "[demo] between 'rare_dx' and 'Background only' to see the foreground appear/vanish."
