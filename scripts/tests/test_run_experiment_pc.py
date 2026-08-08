"""run_experiment wiring for model_class=pc (Phase-C PC replication)."""
import importlib
from pathlib import Path


def _run_exp(monkeypatch):
    mod = importlib.import_module("run_experiment")
    monkeypatch.setattr(mod, "_require_workspace_env", lambda: ("proj.ds", "bill"))
    return mod


def _eff(**over):
    base = {
        "model_class": "pc", "cohort": "mdd_antidepressant",
        "K": 25, "weight_y": 100.0, "alpha": 1.1, "tau": 1.1, "pi_iters": 100,
        "max_iter": 500, "lookback_days": 365, "window_days": 365,
        "stability_days": 90, "grace_gap_days": 30, "vocab_size": 2000,
        "min_df": 20, "min_patient_count": 20, "person_mod": 1,
        "test_frac": 0.25, "seed": 0,
    }
    base.update(over)
    return base


def test_validate_frontmatter_accepts_pc(monkeypatch):
    mod = _run_exp(monkeypatch)
    # minimal valid frontmatter; must not sys.exit (no stm-style required block)
    mod.validate_frontmatter({
        "id": 70, "slug": "x", "cohort": "mdd_antidepressant",
        "model_class": "pc"})


def test_driver_path_for_pc(monkeypatch):
    mod = _run_exp(monkeypatch)
    assert mod.build_fit_driver_path({"model_class": "pc"}) \
        == "analysis/cloud/pc_antidepressant_cloud.py"


def test_build_pc_args_key_to_flag_mapping(monkeypatch):
    mod = _run_exp(monkeypatch)
    args = mod.build_pc_args(_eff(), "/runs/0070-x")
    d = dict(zip(args[::2], args[1::2]))
    assert d["--K"] == "25"
    assert d["--weight-y"] == "100.0"
    assert d["--vocab-size"] == "2000"
    assert d["--stability-days"] == "90"
    assert d["--grace-gap-days"] == "30"
    assert d["--lookback-days"] == "365"
    assert d["--test-frac"] == "0.25"
    assert d["--cdr"] == "proj.ds" and d["--billing"] == "bill"
    # PC writes ONE results JSON into the run dir (no --out-dir like the others)
    assert d["--out"] == str(Path("/runs/0070-x") / "pc_results.json")
    # cohort is hard-wired in the driver; no --cohort / --prior-obs-days flags
    assert "--cohort" not in args
    assert "--prior-obs-days" not in args
    # resume unsupported; K IS a real flag (unlike dag_placement)
    assert "--resume-from" not in args


def test_build_pc_args_backend_defaults_inmem_no_svi_knobs(monkeypatch):
    mod = _run_exp(monkeypatch)
    # No backend key -> default inmem: --backend inmem passed, SVI knobs omitted
    # so the inmem argv is byte-for-byte the prior command line (plus --backend).
    args = mod.build_pc_args(_eff(), "/runs/0070-x")
    d = dict(zip(args[::2], args[1::2]))
    assert d["--backend"] == "inmem"
    for knob in ("--subsampling-rate", "--tau0", "--kappa"):
        assert knob not in args


def test_build_pc_args_backend_vi_threads_svi_knobs(monkeypatch):
    mod = _run_exp(monkeypatch)
    args = mod.build_pc_args(
        _eff(backend="vi", subsampling_rate=0.1, tau0=64.0, kappa=0.6),
        "/runs/0071-x",
    )
    d = dict(zip(args[::2], args[1::2]))
    assert d["--backend"] == "vi"
    assert d["--subsampling-rate"] == "0.1"
    assert d["--tau0"] == "64.0"
    assert d["--kappa"] == "0.6"


def test_build_pc_args_backend_vi_svi_knob_defaults(monkeypatch):
    mod = _run_exp(monkeypatch)
    # backend vi with no explicit knobs -> the driver-matching defaults appear.
    args = mod.build_pc_args(_eff(backend="vi"), "/out")
    d = dict(zip(args[::2], args[1::2]))
    assert d["--subsampling-rate"] == "0.05"
    assert d["--tau0"] == "1024.0"
    assert d["--kappa"] == "0.51"


def test_build_pc_args_cache_uri_optional(monkeypatch):
    mod = _run_exp(monkeypatch)
    assert "--cache-uri" not in mod.build_pc_args(_eff(), "/out")
    on = mod.build_pc_args(_eff(cache_uri="gs://c"), "/out")
    assert on[on.index("--cache-uri") + 1] == "gs://c"


def test_build_fit_args_routes_pc(monkeypatch):
    mod = _run_exp(monkeypatch)
    args = mod.build_fit_args(_eff(), "/out")
    assert "--weight-y" in args  # routed to build_pc_args


def test_pc_gets_larger_driver_memory(monkeypatch):
    mod = _run_exp(monkeypatch)
    monkeypatch.delenv("CHARM_DRIVER_MEMORY", raising=False)
    assert mod._driver_memory_for("pc") == "8g"
    assert mod._driver_memory_for("lda") == "4g"
    # env override wins for either
    monkeypatch.setenv("CHARM_DRIVER_MEMORY", "16g")
    assert mod._driver_memory_for("pc") == "16g"
    assert mod._driver_memory_for("lda") == "16g"


def test_spark_cmd_carries_driver_memory(monkeypatch):
    mod = _run_exp(monkeypatch)
    cmd = mod.build_spark_submit_cmd("x.py", ["--K", "25"], mod.REPO_ROOT,
                                     driver_memory="8g")
    assert cmd[cmd.index("--driver-memory") + 1] == "8g"
    # default stays 4g for callers that don't pass it (lda/stm/dag/eval)
    cmd2 = mod.build_spark_submit_cmd("x.py", [], mod.REPO_ROOT)
    assert cmd2[cmd2.index("--driver-memory") + 1] == "4g"
