"""run_experiment wiring for model_class=mondo_usage (whole-Mondo EHR-usage export)."""
import importlib


def _run_exp(monkeypatch):
    mod = importlib.import_module("run_experiment")
    monkeypatch.setattr(mod, "_require_workspace_env", lambda: ("proj.ds", "bill"))
    return mod


def test_validate_frontmatter_accepts_mondo_usage(monkeypatch):
    mod = _run_exp(monkeypatch)
    # minimal valid frontmatter; must not sys.exit (no stm-style required block)
    mod.validate_frontmatter({
        "id": 107, "slug": "x", "cohort": "population_rare_priority",
        "model_class": "mondo_usage"})


def test_driver_path_for_mondo_usage(monkeypatch):
    mod = _run_exp(monkeypatch)
    assert mod.build_fit_driver_path({"model_class": "mondo_usage"}) \
        == "analysis/cloud/mondo_usage_cloud.py"


def test_build_mondo_usage_args_source_climb(monkeypatch):
    mod = _run_exp(monkeypatch)
    eff = {"model_class": "mondo_usage", "count_space": "source_climb",
           "source_table": "condition_occurrence", "min_cell": 20,
           "mondo_version": "2026-06-02", "mondo_cache_dir": "data/mondo"}
    args = mod.build_fit_args(eff, "/runs/0107-x")   # exercises the dispatch too
    d = dict(zip(args[::2], args[1::2]))
    assert d["--cdr"] == "proj.ds" and d["--billing"] == "bill"
    assert d["--out"] == "/runs/0107-x"          # writes the export INTO the run dir
    assert d["--count-space"] == "source_climb"
    assert d["--source-table"] == "condition_occurrence"
    assert d["--min-cell"] == "20"


def test_build_mondo_usage_args_defaults_when_minimal(monkeypatch):
    mod = _run_exp(monkeypatch)
    # a minimal frontmatter still runs: count-space falls back to standard, etc.
    args = mod.build_mondo_usage_args({"model_class": "mondo_usage"}, "/runs/x")
    d = dict(zip(args[::2], args[1::2]))
    assert d["--count-space"] == "standard"
    assert d["--source-table"] == "condition_occurrence"
    assert d["--min-cell"] == "20"
    assert d["--mondo-version"] == "2026-06-02"


def test_build_mondo_usage_args_pins_hpo_release(monkeypatch):
    # The live purl's 2026-09-01 HPO release carries no SNOMED xrefs, which silently
    # emptied the HPO axis (exp 0120 first run); experiments pin the release.
    mod = _run_exp(monkeypatch)
    url = "http://purl.obolibrary.org/obo/hp/releases/2026-06-23/hp.obo"
    args = mod.build_mondo_usage_args(
        {"model_class": "mondo_usage", "with_hpo": True, "hpo_obo_url": url}, "/runs/x")
    assert args[args.index("--hpo-obo-url") + 1] == url and "--with-hpo" in args
    assert "--hpo-obo-url" not in mod.build_mondo_usage_args({"model_class": "mondo_usage"}, "/r")
