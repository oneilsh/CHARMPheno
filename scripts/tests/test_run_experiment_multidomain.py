"""run_experiment wiring for model_class=multidomain (two-domain gated fit)."""
import importlib


def _run_exp(monkeypatch):
    mod = importlib.import_module("run_experiment")
    monkeypatch.setattr(mod, "_require_workspace_env", lambda: ("proj.ds", "bill"))
    return mod


def _min_eff():
    """Minimal effective config with every key build_multidomain_args indexes
    directly (no .get default): seed, person_mod, min_n, n_bg, tpn, max_iter,
    doc_min_length."""
    return {"model_class": "multidomain", "seed": 42, "person_mod": 10,
            "min_n": 50, "n_bg": 20, "tpn": 5, "max_iter": 100,
            "doc_min_length": 10}


def test_validate_frontmatter_accepts_multidomain(monkeypatch):
    mod = _run_exp(monkeypatch)
    # minimal valid frontmatter; must not sys.exit
    mod.validate_frontmatter({
        "id": 70, "slug": "x", "cohort": "population_diabetes",
        "model_class": "multidomain"})


def test_driver_path_for_multidomain(monkeypatch):
    mod = _run_exp(monkeypatch)
    assert mod.build_fit_driver_path({"model_class": "multidomain"}) \
        == "analysis/cloud/multidomain_cloud.py"


def test_build_multidomain_args_shape(monkeypatch):
    mod = _run_exp(monkeypatch)
    args = mod.build_multidomain_args(_min_eff(), "/out")
    assert "--out-dir" in args and args[args.index("--out-dir") + 1] == "/out"
    assert "--seed" in args and args[args.index("--seed") + 1] == "42"
    assert "--cdr" in args and args[args.index("--cdr") + 1] == "proj.ds"
    assert "--billing" in args and args[args.index("--billing") + 1] == "bill"
    # per-domain vocab controls present for both domains
    for flag in ("--cond-vocab-size", "--cond-min-df", "--cond-min-patient-count",
                 "--drug-vocab-size", "--drug-min-df", "--drug-min-patient-count"):
        assert flag in args, flag
    assert "--K" not in args                     # K is emergent
    assert "--resume-from" not in args           # resume unsupported


def test_build_multidomain_args_omits_omega_eta_when_unset(monkeypatch):
    # Unset omega/eta MUST NOT be emitted: the driver's None default routes to the
    # shim's pre-multi-domain scalar default; a degenerate list would instead
    # assert a per-domain vector (gated_lda.py). This is load-bearing.
    mod = _run_exp(monkeypatch)
    args = mod.build_multidomain_args(_min_eff(), "/out")
    assert "--omega" not in args
    assert "--eta-per-domain" not in args


def test_build_multidomain_args_emits_omega_eta_from_list_and_string(monkeypatch):
    mod = _run_exp(monkeypatch)
    # YAML list form
    eff = {**_min_eff(), "omega": [1.0, 0.5], "eta_per_domain": [0.1, 0.2]}
    args = mod.build_multidomain_args(eff, "/out")
    assert args[args.index("--omega") + 1] == "1.0,0.5"
    assert args[args.index("--eta-per-domain") + 1] == "0.1,0.2"
    # comma-string form (how a frontmatter scalar would arrive)
    eff2 = {**_min_eff(), "omega": "1.0,0.5"}
    args2 = mod.build_multidomain_args(eff2, "/out")
    assert args2[args2.index("--omega") + 1] == "1.0,0.5"


def test_build_multidomain_args_emits_top_n_tokens(monkeypatch):
    mod = _run_exp(monkeypatch)
    # from _base.yaml (top_n_tokens: 6) via merge; drives the final topic dump.
    args = mod.build_multidomain_args({**_min_eff(), "top_n_tokens": 6}, "/out")
    assert args[args.index("--top-n-tokens") + 1] == "6"
    # default when unset in effective config
    args2 = mod.build_multidomain_args(_min_eff(), "/out")
    assert args2[args2.index("--top-n-tokens") + 1] == "8"


def test_build_fit_args_routes_multidomain(monkeypatch):
    mod = _run_exp(monkeypatch)
    args = mod.build_fit_args(_min_eff(), "/out")
    # routed to build_multidomain_args: has the two-domain drug source flag,
    # which no other builder emits.
    assert "--source-table-drug" in args
