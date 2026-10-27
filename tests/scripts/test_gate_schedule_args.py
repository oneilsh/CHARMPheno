"""Leaves-first gate schedule plumbing (exp 0153/0154): front matter -> fit argv,
and the driver flags -> estimator Params."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "scripts"), str(REPO_ROOT / "analysis" / "cloud"),
           str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_experiment as rex  # noqa: E402

BASE = {"source_table": "t", "person_mod": 1, "vocab_size": 5000, "min_df": 20,
        "min_patient_count": 20, "doc_min_length": 10, "max_iter": 100,
        "min_n": 0, "n_bg": 8, "tpn": 1, "seed": 0, "readout_mode": "distributed"}


def test_gate_warm_front_matter_is_emitted_only_when_set(monkeypatch):
    monkeypatch.setenv("WORKSPACE_CDR", "cdr")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    plain = rex.build_gated_pc_args(dict(BASE), "/tmp/out")
    assert "--gate-warm" not in plain and "--gate-no-open-seed" not in plain
    on = rex.build_gated_pc_args(dict(BASE, gate_warm=5), "/tmp/out")
    assert on[on.index("--gate-warm") + 1] == "5" and "--gate-no-open-seed" not in on


def test_driver_flags_reach_the_estimator():
    import argparse
    import gated_pc_cloud as gpc
    args = argparse.Namespace(gate_warm=4, gate_seed_mode="balanced",
                              gate_seed_leaves=True, _parent_int={1: [0], 2: [1]},
                              _gate_seed_own={2: [3]})
    for k in ("max_iter", "seed", "subsampling_rate", "tau0", "kappa",
              "gamma_shape", "cavi_max_iter", "cavi_tol",
              "optimize_doc_concentration", "n_bg", "tpn"):
        setattr(args, k, {"optimize_doc_concentration": False,
                          "subsampling_rate": 0.1, "tau0": 64.0, "kappa": 0.51,
                          "gamma_shape": 100.0, "cavi_tol": 1e-3}.get(k, 2))
    est = gpc._build_estimator(args)
    assert type(est).__name__ == "DagGatedLDAEstimator"
    assert est.getOrDefault("gateWarm") == 4 and est.getOrDefault("gateOpenSeed") is True
    assert est.getOrDefault("gateSeedMode") == "balanced"
    assert est.getOrDefault("gateSeedLeaves") is True
    assert est.getGateSeedOwnCodes() == {2: [3]}
    assert est._layout().K == 2 + 2 * 2
