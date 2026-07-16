"""run_experiment wiring for model_class=dag_placement (piece 3)."""
import importlib


def _run_exp(monkeypatch):
    mod = importlib.import_module("run_experiment")
    monkeypatch.setattr(mod, "_require_workspace_env", lambda: ("proj.ds", "bill"))
    return mod


def test_validate_frontmatter_accepts_dag_placement(monkeypatch):
    mod = _run_exp(monkeypatch)
    # minimal valid frontmatter; must not sys.exit
    mod.validate_frontmatter({
        "id": 52, "slug": "x", "cohort": "population_diabetes",
        "model_class": "dag_placement"})


def test_driver_path_for_dag_placement(monkeypatch):
    mod = _run_exp(monkeypatch)
    assert mod.build_fit_driver_path({"model_class": "dag_placement"}) \
        == "analysis/cloud/dag_placement_cloud.py"


def test_build_dag_placement_args_shape(monkeypatch):
    mod = _run_exp(monkeypatch)
    eff = {"model_class": "dag_placement", "source_table": "condition_era",
           "person_mod": 10, "vocab_size": 5000, "min_df": 20,
           "min_patient_count": 20, "doc_min_length": 0, "prior_obs_days": 365,
           "window_days": 365, "anchor": 201820, "min_n": 50, "holdout_frac": 0.2,
           "n_bg": 2, "tpn": 1, "max_iter": 100, "seed": 42, "init": "spectral",
           "spectral_max_vocab": 8000, "cache_uri": "hdfs:///c"}
    args = mod.build_dag_placement_args(eff, "/out")
    assert "--init" in args and args[args.index("--init") + 1] == "spectral"
    assert "--anchor" in args and args[args.index("--anchor") + 1] == "201820"
    assert "--out-dir" in args and args[args.index("--out-dir") + 1] == "/out"
    assert "--cache-uri" in args and args[args.index("--cache-uri") + 1] == "hdfs:///c"
    assert "--K" not in args                       # K is emergent
    assert "--resume-from" not in args             # resume unsupported


def test_build_fit_args_routes_dag_placement(monkeypatch):
    mod = _run_exp(monkeypatch)
    eff = {"model_class": "dag_placement", "source_table": "condition_era",
           "person_mod": 10, "vocab_size": 5000, "min_df": 20,
           "min_patient_count": 20, "doc_min_length": 0, "prior_obs_days": 365,
           "window_days": 365, "anchor": 201820, "min_n": 50, "holdout_frac": 0.2,
           "n_bg": 2, "tpn": 1, "max_iter": 100, "seed": 42, "init": "random",
           "spectral_max_vocab": 8000}
    args = mod.build_fit_args(eff, "/out")
    assert "--anchor" in args     # routed to build_dag_placement_args


def test_build_dag_placement_args_includes_strip_mode(monkeypatch):
    import importlib
    mod = importlib.import_module("run_experiment")
    monkeypatch.setattr(mod, "_require_workspace_env", lambda: ("p.d", "bill"))
    eff = {"model_class": "dag_placement", "source_table": "condition_era",
           "person_mod": 10, "vocab_size": 5000, "min_df": 20,
           "min_patient_count": 20, "doc_min_length": 0, "min_n": 50,
           "n_bg": 2, "tpn": 1, "max_iter": 100, "strip_mode": "both"}
    args = mod.build_dag_placement_args(eff, "/out")
    assert args[args.index("--strip-mode") + 1] == "both"


def test_build_dag_placement_args_emits_topic_logging_flags(monkeypatch):
    import importlib
    mod = importlib.import_module("run_experiment")
    monkeypatch.setattr(mod, "_require_workspace_env", lambda: ("p.d", "bill"))
    eff = {"model_class": "dag_placement", "source_table": "condition_era",
           "person_mod": 10, "vocab_size": 5000, "min_df": 20,
           "min_patient_count": 20, "doc_min_length": 0, "min_n": 50,
           "n_bg": 2, "tpn": 1, "max_iter": 100,
           "print_topics_every": 10, "top_n_tokens": 6}
    args = mod.build_dag_placement_args(eff, "/out")
    assert args[args.index("--print-topics-every") + 1] == "10"
    assert args[args.index("--top-n-tokens") + 1] == "6"
