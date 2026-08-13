"""run_experiment wiring for model_class=gated_pc (Step B: Gated-PC case-finding)."""
import importlib
import sys
from pathlib import Path


def _run_exp(monkeypatch):
    mod = importlib.import_module("run_experiment")
    monkeypatch.setattr(mod, "_require_workspace_env", lambda: ("proj.ds", "bill"))
    return mod


def _base_eff():
    return {"model_class": "gated_pc", "source_table": "condition_era",
            "person_mod": 10, "vocab_size": 5000, "min_df": 20,
            "min_patient_count": 20, "doc_min_length": 0, "prior_obs_days": 365,
            "window_days": 365, "disease": "rare6", "min_n": 50, "holdout_frac": 0.2,
            "n_bg": 2, "tpn": 1, "max_iter": 100}


def test_validate_frontmatter_accepts_gated_pc(monkeypatch):
    mod = _run_exp(monkeypatch)
    mod.validate_frontmatter({
        "id": 90, "slug": "x", "cohort": "population_rare6",
        "model_class": "gated_pc"})


def test_driver_path_for_gated_pc(monkeypatch):
    mod = _run_exp(monkeypatch)
    assert mod.build_fit_driver_path({"model_class": "gated_pc"}) \
        == "analysis/cloud/gated_pc_cloud.py"


def test_build_gated_pc_args_shape(monkeypatch):
    mod = _run_exp(monkeypatch)
    eff = {**_base_eff(), "weight_y": 80.0, "seed": 42, "cache_uri": "hdfs:///c",
           "with_dag_head": True}
    args = mod.build_gated_pc_args(eff, "/out")
    assert args[args.index("--disease") + 1] == "rare6"
    assert args[args.index("--weight-y") + 1] == "80.0"
    assert args[args.index("--out-dir") + 1] == "/out"
    assert args[args.index("--cache-uri") + 1] == "hdfs:///c"
    assert args[args.index("--head-optimizer") + 1] == "newton"
    assert args[args.index("--head-l2") + 1] == "0.001"
    assert "--with-dag-head" in args
    assert "--skip-unsup-gated" not in args        # absent by default
    assert "--resume-from" not in args             # resume unsupported


def test_build_fit_args_routes_gated_pc(monkeypatch):
    mod = _run_exp(monkeypatch)
    args = mod.build_fit_args(_base_eff(), "/out")
    assert "--disease" in args and "--weight-y" in args   # routed to gated_pc builder


def test_gated_pc_store_true_flags(monkeypatch):
    mod = _run_exp(monkeypatch)
    base = _base_eff()
    assert "--skip-unsup-gated" not in mod.build_gated_pc_args(base, "/out")
    assert "--with-dag-head" not in mod.build_gated_pc_args(base, "/out")
    on = {**base, "skip_unsup_gated": True, "with_dag_head": True,
          "optimize_doc_concentration": True}
    a = mod.build_gated_pc_args(on, "/out")
    assert "--skip-unsup-gated" in a and "--with-dag-head" in a
    assert "--optimize-doc-concentration" in a


def test_built_args_parse_through_driver(monkeypatch):
    """The argv run_experiment builds must parse cleanly through the driver's own
    parse_args — the contract between the config layer and the driver CLI."""
    mod = _run_exp(monkeypatch)
    args = mod.build_gated_pc_args({**_base_eff(), "seed": 7, "with_dag_head": True},
                                   "/out")
    cloud = str(Path(mod.__file__).resolve().parent.parent / "analysis" / "cloud")
    if cloud not in sys.path:
        sys.path.insert(0, cloud)
    import gated_pc_cloud
    parsed = gated_pc_cloud.parse_args(args)
    assert parsed.disease == "rare6" and parsed.with_dag_head is True
    assert parsed.head_l2 == 1e-3 and parsed.out_dir == "/out"
