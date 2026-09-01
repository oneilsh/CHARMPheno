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
           "with_dag_head": True, "num_partitions": 96}
    args = mod.build_gated_pc_args(eff, "/out")
    assert args[args.index("--disease") + 1] == "rare6"
    assert args[args.index("--weight-y") + 1] == "80.0"
    assert args[args.index("--num-partitions") + 1] == "96"
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


def test_head_config_round_trips_frontmatter_to_shim_estimator(monkeypatch):
    """Full config chain: frontmatter-style dict -> build_gated_pc_args ->
    gated_pc_cloud.parse_args -> shim estimator -> the shim builds an engine with
    the head hypers (head_l2 ridge, newton optimizer) carried through."""
    mod = _run_exp(monkeypatch)
    eff = {**_base_eff(), "head_l2": 0.01, "head_optimizer": "newton", "K": 4}
    args = mod.build_gated_pc_args(eff, "/out")
    assert args[args.index("--head-l2") + 1] == "0.01"

    cloud = str(Path(mod.__file__).resolve().parent.parent / "analysis" / "cloud")
    if cloud not in sys.path:
        sys.path.insert(0, cloud)
    import gated_pc_cloud
    from spark_vi.mllib.topic.pc import _build_model_and_config
    parsed = gated_pc_cloud.parse_args(args)
    assert parsed.head_l2 == 0.01 and parsed.head_optimizer == "newton"
    parsed._C = 1
    est = gated_pc_cloud._build_pc_estimator(parsed, weight_y=50.0, gated=False)
    assert est.getOrDefault("headL2") == 0.01
    model, _ = _build_model_and_config(est, vocab_size=8)
    assert model.head_l2 == 0.01 and model.head_optimizer == "newton"


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


def test_mondo_completeness_route(monkeypatch):
    """model_class=mondo_completeness validates, resolves to the completeness driver,
    and builds argv (cdr/billing from env + out-dir + top-unplaced)."""
    mod = _run_exp(monkeypatch)
    mod.validate_frontmatter({
        "id": 87, "slug": "mondo-completeness",
        "cohort": "population_rare_priority", "model_class": "mondo_completeness"})
    assert mod.build_fit_driver_path({"model_class": "mondo_completeness"}) \
        == "analysis/cloud/mondo_completeness_cloud.py"
    args = mod.build_fit_args(
        {"model_class": "mondo_completeness", "top_unplaced": 100}, "/out")
    assert args[args.index("--cdr") + 1] == "proj.ds"
    assert args[args.index("--billing") + 1] == "bill"
    assert args[args.index("--out") + 1] == "/out"
    assert args[args.index("--top-unplaced") + 1] == "100"


def test_mondo_hierarchy_route(monkeypatch):
    """model_class=mondo_hierarchy validates, resolves to the hierarchy driver, and
    builds argv (cdr/billing from env + out-dir + power/reduction knobs)."""
    mod = _run_exp(monkeypatch)
    mod.validate_frontmatter({
        "id": 88, "slug": "mondo-hierarchy",
        "cohort": "population_rare_priority", "model_class": "mondo_hierarchy"})
    assert mod.build_fit_driver_path({"model_class": "mondo_hierarchy"}) \
        == "analysis/cloud/mondo_hierarchy_cloud.py"
    args = mod.build_fit_args(
        {"model_class": "mondo_hierarchy", "min_positives": 100, "tpn": 1}, "/out")
    assert args[args.index("--cdr") + 1] == "proj.ds"
    assert args[args.index("--out") + 1] == "/out"
    assert args[args.index("--min-positives") + 1] == "100"
    assert args[args.index("--tpn") + 1] == "1"


def test_mondo_dag_source_flags_thread(monkeypatch):
    """dag_source=mondo + mondo_branch/min_positives frontmatter -> the mondo CLI
    flags -> the driver parses them (the whole-Mondo / template-branch fit path)."""
    mod = _run_exp(monkeypatch)
    eff = {**_base_eff(), "dag_source": "mondo",
           "mondo_branch": "MONDO:0004995", "min_positives": 100,
           "extra_domains": "measurement,drug", "window_mode": "lookback",
           "label_mask_mode": "closure", "localize_head": True}
    args = mod.build_gated_pc_args(eff, "/out")
    assert args[args.index("--dag-source") + 1] == "mondo"
    assert args[args.index("--mondo-branch") + 1] == "MONDO:0004995"
    assert args[args.index("--min-positives") + 1] == "100"

    # defaults omit the mondo flags (SNOMED path unchanged).
    base = mod.build_gated_pc_args(_base_eff(), "/out")
    assert "--dag-source" not in base and "--mondo-branch" not in base

    cloud = str(Path(mod.__file__).resolve().parent.parent / "analysis" / "cloud")
    if cloud not in sys.path:
        sys.path.insert(0, cloud)
    import gated_pc_cloud
    parsed = gated_pc_cloud.parse_args(args)
    assert parsed.dag_source == "mondo"
    assert parsed.mondo_branch == "MONDO:0004995"
    assert parsed.min_positives == 100


def test_dag_collapse_flag_threads(monkeypatch):
    """exp 0109: `dag_collapse: true` front matter -> --dag-collapse -> the driver
    parses it -> the corpus spec carries it (which is what forks the cache key).
    Absent/false front matter must not emit the flag at all: exp 0104's record run
    has to reproduce byte-identically."""
    mod = _run_exp(monkeypatch)
    eff = {**_base_eff(), "dag_source": "mondo", "min_positives": 100,
           "window_mode": "lookback", "dag_collapse": True}
    args = mod.build_gated_pc_args(eff, "/out")
    assert "--dag-collapse" in args

    assert "--dag-collapse" not in mod.build_gated_pc_args(_base_eff(), "/out")
    assert "--dag-collapse" not in mod.build_gated_pc_args(
        {**eff, "dag_collapse": False}, "/out")

    cloud = str(Path(mod.__file__).resolve().parent.parent / "analysis" / "cloud")
    if cloud not in sys.path:
        sys.path.insert(0, cloud)
    import gated_pc_cloud
    parsed = gated_pc_cloud.parse_args(args)
    assert parsed.dag_collapse is True
    spec = gated_pc_cloud.multidomain_corpus_spec(parsed, ("drug",))
    assert spec["dag_collapse"] is True
    # ...and the default driver invocation still says off, on both DAG sources.
    off = gated_pc_cloud.parse_args(mod.build_gated_pc_args(
        {**eff, "dag_collapse": False}, "/out"))
    assert off.dag_collapse is False
    assert gated_pc_cloud.multidomain_corpus_spec(off, ("drug",))["dag_collapse"] \
        is False


def test_dag_collapse_is_mondo_only_in_the_spec(monkeypatch):
    """The reduction names Mondo CLASS nodes; on the SNOMED path the spec pins it
    False so a stray flag cannot split that cache."""
    mod = _run_exp(monkeypatch)
    cloud = str(Path(mod.__file__).resolve().parent.parent / "analysis" / "cloud")
    if cloud not in sys.path:
        sys.path.insert(0, cloud)
    import gated_pc_cloud
    parsed = gated_pc_cloud.parse_args(mod.build_gated_pc_args(
        {**_base_eff(), "window_mode": "lookback", "dag_collapse": True}, "/out"))
    assert parsed.dag_source == "snomed" and parsed.dag_collapse is True
    assert gated_pc_cloud.multidomain_corpus_spec(parsed, ())["dag_collapse"] is False


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

    # ...even when the flag is explicitly asked for: it is not an option here.
    with_flag = gated_pc_cloud.parse_args(mod.build_gated_pc_args(
        {**eff, "dag_collapse": True}, "/out"))
    assert with_flag.dag_collapse is True
    assert gated_pc_cloud.multidomain_corpus_spec(
        with_flag, ("drug",))["dag_collapse"] is False


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


def test_localize_head_flag_threads(monkeypatch):
    """localize_head frontmatter -> --localize-head -> driver parses it -> the shim
    estimator carries localizeHead=True."""
    mod = _run_exp(monkeypatch)
    on = mod.build_gated_pc_args({**_base_eff(), "localize_head": True}, "/out")
    assert "--localize-head" in on
    off = mod.build_gated_pc_args(_base_eff(), "/out")
    assert "--localize-head" not in off
    cloud = str(Path(mod.__file__).resolve().parent.parent / "analysis" / "cloud")
    if cloud not in sys.path:
        sys.path.insert(0, cloud)
    import gated_pc_cloud
    parsed = gated_pc_cloud.parse_args(on)
    assert parsed.localize_head is True


def test_preindex_closure_flag_threads(monkeypatch):
    """E1: `preindex_closure: true` front matter -> --preindex-closure -> the
    driver parses it -> the corpus spec carries it (which is what forks the cache
    key, so a bundle WITH the column can never be served for one without).

    Absent/false front matter must not emit the flag at all: every existing
    experiment, 0104's and 0109's record runs included, has to keep reproducing
    byte-identically."""
    mod = _run_exp(monkeypatch)
    eff = {**_base_eff(), "dag_source": "mondo_native", "min_positives": 100,
           "window_mode": "lookback", "preindex_closure": True}
    args = mod.build_gated_pc_args(eff, "/out")
    assert "--preindex-closure" in args

    assert "--preindex-closure" not in mod.build_gated_pc_args(_base_eff(), "/out")
    assert "--preindex-closure" not in mod.build_gated_pc_args(
        {**eff, "preindex_closure": False}, "/out")

    cloud = str(Path(mod.__file__).resolve().parent.parent / "analysis" / "cloud")
    if cloud not in sys.path:
        sys.path.insert(0, cloud)
    import gated_pc_cloud
    parsed = gated_pc_cloud.parse_args(args)
    assert parsed.preindex_closure is True
    spec = gated_pc_cloud.multidomain_corpus_spec(parsed, ("drug",))
    assert spec["preindex_closure"] is True
    # ...and it lands in the KEY, which is the whole point of the fork.
    off_spec = dict(spec, preindex_closure=False)
    assert (gated_pc_cloud.multidomain_cache_key(spec)
            != gated_pc_cloud.multidomain_cache_key(off_spec))

    off = gated_pc_cloud.parse_args(mod.build_gated_pc_args(
        {**eff, "preindex_closure": False}, "/out"))
    assert off.preindex_closure is False
    assert gated_pc_cloud.multidomain_corpus_spec(
        off, ("drug",))["preindex_closure"] is False


def test_preindex_closure_is_mondo_only_in_the_spec(monkeypatch):
    """The column is built by re-running the MONDO attestation provider over the
    feature window, and only the Mondo paths construct one driver-side — so on the
    SNOMED path the spec pins it False and a stray flag cannot split that cache."""
    mod = _run_exp(monkeypatch)
    cloud = str(Path(mod.__file__).resolve().parent.parent / "analysis" / "cloud")
    if cloud not in sys.path:
        sys.path.insert(0, cloud)
    import gated_pc_cloud
    parsed = gated_pc_cloud.parse_args(mod.build_gated_pc_args(
        {**_base_eff(), "window_mode": "lookback", "preindex_closure": True},
        "/out"))
    assert parsed.dag_source == "snomed" and parsed.preindex_closure is True
    assert gated_pc_cloud.multidomain_corpus_spec(
        parsed, ())["preindex_closure"] is False


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
