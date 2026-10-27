"""run_experiment wiring for model_class=gated_pc (Step B: Gated-PC case-finding)."""
import importlib
import sys

import pytest
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
    eff = {**_base_eff(), "seed": 42, "cache_uri": "hdfs:///c",
           "num_partitions": 96}
    args = mod.build_gated_pc_args(eff, "/out")
    assert args[args.index("--disease") + 1] == "rare6"
    assert args[args.index("--num-partitions") + 1] == "96"
    assert args[args.index("--out-dir") + 1] == "/out"
    assert args[args.index("--cache-uri") + 1] == "hdfs:///c"
    for gone in ("--weight-y", "--head-optimizer", "--head-l2", "--grad-cavi-iters",
                 "--topic-trust", "--localize-head", "--init"):
        assert gone not in args                    # removed by ADR 0050
    assert "--resume-from" not in args             # resume unsupported
    for gone in ("--with-dag-head", "--skip-unsup-gated", "--diag-only", "--k",
                 "--readout-mode", "--eval-every", "--min-label-count"):
        assert gone not in args                    # fit ends at save (2026-10)


def test_build_gated_pc_args_fit_save_interval_default_off(monkeypatch):
    mod = _run_exp(monkeypatch)
    args = mod.build_gated_pc_args(_base_eff(), "/out")
    assert args[args.index("--fit-save-interval") + 1] == "0"


def test_build_gated_pc_args_fit_save_interval_from_config(monkeypatch):
    mod = _run_exp(monkeypatch)
    args = mod.build_gated_pc_args({**_base_eff(), "fit_save_interval": 5}, "/out")
    assert args[args.index("--fit-save-interval") + 1] == "5"


def test_build_fit_args_routes_gated_pc(monkeypatch):
    mod = _run_exp(monkeypatch)
    args = mod.build_fit_args(_base_eff(), "/out")
    assert "--disease" in args and "--gate-attestation" not in args  # gated_pc builder
    assert "--n-bg" in args


def test_gated_pc_store_true_flags(monkeypatch):
    mod = _run_exp(monkeypatch)
    base = _base_eff()
    assert "--optimize-doc-concentration" not in mod.build_gated_pc_args(base, "/out")
    a = mod.build_gated_pc_args({**base, "optimize_doc_concentration": True}, "/out")
    assert "--optimize-doc-concentration" in a


def test_built_args_parse_through_driver(monkeypatch):
    """The argv run_experiment builds must parse cleanly through the driver's own
    parse_args — the contract between the config layer and the driver CLI."""
    mod = _run_exp(monkeypatch)
    args = mod.build_gated_pc_args({**_base_eff(), "seed": 7},
                                   "/out")
    cloud = str(Path(mod.__file__).resolve().parent.parent / "analysis" / "cloud")
    if cloud not in sys.path:
        sys.path.insert(0, cloud)
    import gated_pc_cloud
    parsed = gated_pc_cloud.parse_args(args)
    assert parsed.disease == "rare6" and parsed.seed == 7
    assert parsed.out_dir == "/out"


def test_config_round_trips_frontmatter_to_the_estimator(monkeypatch):
    """Full config chain: frontmatter-style dict -> build_gated_pc_args ->
    gated_pc_cloud.parse_args -> DagGatedLDAEstimator, the fit knobs carried
    through."""
    mod = _run_exp(monkeypatch)
    eff = {**_base_eff(), "doc_concentration": 0.5, "gate_warm": 2,
           "gate_seed_mode": "balanced", "gate_seed_leaves": True}
    args = mod.build_gated_pc_args(eff, "/out")
    cloud = str(Path(mod.__file__).resolve().parent.parent / "analysis" / "cloud")
    if cloud not in sys.path:
        sys.path.insert(0, cloud)
    import gated_pc_cloud
    parsed = gated_pc_cloud.parse_args(args)
    parsed._parent_int = {1: [0], 2: [0]}
    est = gated_pc_cloud._build_estimator(parsed)
    assert est.getOrDefault("docConcentration") == [0.5]
    assert est.getOrDefault("gateWarm") == 2
    assert est.getOrDefault("gateSeedMode") == "balanced"
    assert est.getOrDefault("gateSeedLeaves") is True
    assert est.getOrDefault("maxIter") == 100


def test_anchor_select_route(monkeypatch):
    """model_class=anchor_select validates, resolves to the anchor-selection driver,
    and builds argv (cdr/billing from workspace env + seed/min-positives)."""
    mod = _run_exp(monkeypatch)
    mod.validate_frontmatter({
        "id": 86, "slug": "mondo-coverage", "cohort": "population_rare_priority",
        "model_class": "anchor_select"})
    assert mod.build_fit_driver_path({"model_class": "anchor_select"}) \
        == "analysis/cloud/anchor_selection_cloud.py"
    eff = {"model_class": "anchor_select", "min_positives": 100,
           "mondo_version": "2026-06-02"}
    args = mod.build_fit_args(eff, "/out")
    assert args[args.index("--cdr") + 1] == "proj.ds"
    assert args[args.index("--billing") + 1] == "bill"
    assert args[args.index("--min-positives") + 1] == "100"
    assert args[args.index("--mondo-version") + 1] == "2026-06-02"
    assert args[args.index("--out") + 1] == "/out/candidates_with_counts.tsv"
    assert args[args.index("--seed-tsv") + 1].endswith("priority_seed.tsv")





def test_mondo_native_dag_source_threads_end_to_end(monkeypatch):
    """exp 0110: `dag_source: mondo_native` front matter -> --dag-source ->
    the driver parses it -> the corpus spec routes it through the Mondo branch of
    the multi-domain assembler (population index, min_n=0, the Mondo build inputs
    kept) with `dag_collapse` pinned OFF, because the native build applies the
    splice itself and asking again would double-apply it."""
    mod = _run_exp(monkeypatch)
    eff = {**_base_eff(), "dag_source": "mondo_native", "min_positives": 100,
           "mondo_version": "2026-06-02", "window_mode": "lookback",
           "label_mask_mode": "closure", "localize_head": True}
    args = mod.build_gated_pc_args(eff, "/out")
    assert args[args.index("--dag-source") + 1] == "mondo_native"

    cloud = str(Path(mod.__file__).resolve().parent.parent / "analysis" / "cloud")
    if cloud not in sys.path:
        sys.path.insert(0, cloud)
    import gated_pc_cloud
    parsed = gated_pc_cloud.parse_args(args)
    assert parsed.dag_source == "mondo_native"
    spec = gated_pc_cloud.multidomain_corpus_spec(parsed, ("drug",))
    assert spec["dag_source"] == "mondo_native"
    assert spec["index_mode"] == "population" and spec["min_n"] == 0
    assert spec["min_positives"] == 100 and spec["mondo_version"] == "2026-06-02"
    assert spec["dag_collapse"] is False

    with pytest.raises(SystemExit):            # the flag is gone (2026-10 cleanup)
        gated_pc_cloud.parse_args(args + ["--dag-collapse"])


def test_mondo_native_selects_the_native_assembler(monkeypatch):
    """The dispatch itself: `mondo_assemble_fn` must hand back the NATIVE closure
    for a native spec and the anchor-hierarchy one otherwise. Checked through the
    seam (a fake builder), since the real one is minutes of BigQuery."""
    mod = _run_exp(monkeypatch)
    cloud = str(Path(mod.__file__).resolve().parent.parent / "analysis" / "cloud")
    if cloud not in sys.path:
        sys.path.insert(0, cloud)
    import gated_pc_cloud

    seen = {}

    def _fake_assemble(spark, *, before_dag, attested_provider, **kw):
        seen["dag"] = before_dag
        return "bundle"

    class _Dag:
        names = {-1: "root"}

        def nodes(self):
            return {-1, 4995}

    def _fake_native_build(spark, **kw):
        seen["native_kw"] = kw
        return (_Dag(), None, {4995}, {4995: 400},
                {"version": "native-mondo-v1", "n_kept": 1, "n_coded_kept": 1,
                 "n_hasse_nodes": 1, "n_hasse_multi_parent": 0,
                 "n_final_nodes": 1, "n_final_multi_parent": 0,
                 "min_positives": 100, "n_codes_resolved": 2, "n_coded_terms": 1,
                 "n_terms_with_any_support": 3, "n_powered": 1,
                 "min_support_kept": 400, "n_codes_attesting": 2, "branch": "",
                 "collapse": {"spliced": 0, "dropped_childless": 0, "passes": 0,
                              "predicted_degenerate": 1}})

    spec = {"dag_source": "mondo_native", "cdr": "p.d", "billing": "bp",
            "mondo_version": "2026-06-02", "mondo_cache_dir": "data/mondo",
            "min_positives": 100, "mondo_branch": ""}
    fn = gated_pc_cloud.mondo_assemble_fn(
        spec, _build_inputs=_fake_native_build, _assemble=_fake_assemble)
    assert fn(None) == "bundle"
    assert seen["native_kw"]["min_positives"] == 100
    assert seen["native_kw"]["branch_root"] is None
    assert 4995 in seen["dag"].nodes()          # Mondo term ids, not OMOP cids


def test_the_0110_front_matter_asks_for_the_preindex_column():
    """The census (E-census) is a property of 0110's CORPUS and has to be measured
    on the corpus the record run reports — so the flag lives in 0110's own front
    matter, not in an ad-hoc second bundle. This is the wiring's end: front matter
    -> CLI -> spec -> key -> the column the census reads."""
    import re
    doc = (Path(__file__).resolve().parents[2] / "docs" / "experiments"
           / "0110-native-mondo-label-space.md").read_text()
    front = doc.split("---")[1]
    assert re.search(r"^preindex_closure:\s*true\s*$", front, re.M)


def test_a_removed_option_is_refused_and_its_inert_value_accepted(monkeypatch):
    # ADR 0050: a spec still asking for a removed fit option fails loudly; the
    # value that meant "off" (e.g. an old spec's `init: random`) passes.
    mod = _run_exp(monkeypatch)
    mod.build_gated_pc_args({**_base_eff(), "init": "random",
                             "anchor_scope": "closure"}, "/out")
    mod.build_gated_pc_args({**_base_eff(), "weight_y": 0.0,
                             "head_l2": 0.01, "localize_head": True}, "/out")
    for key, val in (("init", "spectral"), ("spectral_bg_anchors", 8), ("weight_y", 50.0),
                     ("anchor_scope", "frontier"), ("alpha_init", "equalized"),
                     ("count_transform", "binary")):
        with pytest.raises(ValueError, match="ADR 0050"):
            mod.build_gated_pc_args({**_base_eff(), key: val}, "/out")
