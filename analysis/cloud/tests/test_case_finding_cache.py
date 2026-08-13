"""Tests for the CaseFindingBundle write-through cache (piece 3)."""
import datetime as dt


def _tiny_bundle(spark):
    from charmpheno.omop.case_finding_assembly import assemble_from_events
    from charmpheno.omop.condition_dag import build_condition_dag
    from charmpheno.omop.doc_spec import PatientCohortDocSpec
    edges = [(100, 200), (100, 300), (200, 400)]
    before = build_condition_dag(edges, anchor=100, node_ids=[200, 300, 400],
                                 names={100: "dm", 200: "T2", 300: "T1", 400: "T2r"})
    rows = []
    for pid in range(20):
        rows.append((pid, 200, "diabetes", dt.date(2015, 1, 1)))
        rows.append((pid, 999, "diabetes", dt.date(2015, 2, 1)))
    for pid in range(100, 120):
        rows.append((pid, 888, "general", dt.date(2016, 1, 1)))
        rows.append((pid, 777, "general", dt.date(2016, 2, 1)))
    ev = spark.createDataFrame(
        rows, ["person_id", "concept_id", "source_cohort",
               "condition_era_start_date"])
    return assemble_from_events(
        ev, before, doc_spec=PatientCohortDocSpec(min_doc_length=0),
        min_n=1, holdout_frac=0.3, split_salt=20260716,
        vocab_size=50, min_df=1, min_patient_count=1, n_bg=2, tpn=1)


def test_bundle_cache_key_sensitive_and_stable():
    from _case_finding_cache import compute_bundle_cache_key
    base = dict(source_table="condition_era", person_mod=10, vocab_size=5000,
                min_df=20, min_patient_count=20, doc_min_length=0,
                prior_obs_days=365, window_days=365, disease="diabetes", min_n=50,
                holdout_frac=0.2, split_salt=20260716, n_bg=2, tpn=1, cdr="p.d")
    k0 = compute_bundle_cache_key(**base)
    assert k0 == compute_bundle_cache_key(**base)              # stable
    for field, val in [("disease", "rare6"), ("min_n", 25), ("holdout_frac", 0.3),
                       ("person_mod", 20), ("vocab_size", 3000), ("n_bg", 3),
                       ("tpn", 2)]:
        assert compute_bundle_cache_key(**{**base, field: val}) != k0


def test_bundle_cache_save_load_round_trip(spark, tmp_path):
    from _case_finding_cache import save, try_load
    bundle = _tiny_bundle(spark)
    uri = f"file://{tmp_path}/cache"
    save(spark, bundle, uri, "k1")
    loaded = try_load(spark, uri, "k1")
    assert loaded is not None
    # python fields restored with int keys
    assert loaded.parent_int == bundle.parent_int
    assert loaded.int2cid == bundle.int2cid
    assert loaded.cid2int == bundle.cid2int
    assert loaded.vocab_map == bundle.vocab_map
    assert loaded.name_by_id == bundle.name_by_id
    assert loaded.ledger["K_nodes"] == bundle.ledger["K_nodes"]
    # DataFrame contents preserved (compare as sets of person_ids per split)
    assert ({r["person_id"] for r in loaded.train_df.collect()}
            == {r["person_id"] for r in bundle.train_df.collect()})
    assert ({r["person_id"] for r in loaded.test_df.collect()}
            == {r["person_id"] for r in bundle.test_df.collect()})


def test_bundle_cache_miss_then_hit(spark, tmp_path):
    from _case_finding_cache import load_or_build_case_finding_bundle
    built = _tiny_bundle(spark)
    calls = {"n": 0}

    def _stub_assemble(spark_, **kw):
        calls["n"] += 1
        return built

    uri = f"file://{tmp_path}/cache2"
    params = dict(source_table="condition_era", person_mod=10, vocab_size=5000,
                  min_df=20, min_patient_count=20, doc_min_length=0,
                  prior_obs_days=365, window_days=365, disease="diabetes", min_n=50,
                  holdout_frac=0.2, split_salt=20260716, n_bg=2, tpn=1, cdr="p.d",
                  billing="bp")
    b1 = load_or_build_case_finding_bundle(
        spark, cache_uri=uri, _assemble_fn=_stub_assemble, **params)
    b2 = load_or_build_case_finding_bundle(
        spark, cache_uri=uri, _assemble_fn=_stub_assemble, **params)
    assert calls["n"] == 1                       # built once, second call is a HIT
    assert b1.parent_int == b2.parent_int == built.parent_int


def test_bundle_cache_write_failure_is_non_fatal(spark, tmp_path, monkeypatch, capsys):
    """A write-through cache failure (e.g. a wrong/unwritable cache_uri like a
    missing GCS bucket) must NOT abort a run that already paid the assembly cost:
    warn and return the in-memory bundle. Regression for the 404-on-save that
    killed exp 0076 after a 170s build."""
    import _case_finding_cache as ccache
    built = _tiny_bundle(spark)

    def _boom(*a, **k):
        raise RuntimeError("404 Not Found: bucket does not exist")

    monkeypatch.setattr(ccache, "save", _boom)
    params = dict(source_table="condition_era", person_mod=10, vocab_size=5000,
                  min_df=20, min_patient_count=20, doc_min_length=0,
                  prior_obs_days=365, window_days=365, disease="diabetes", min_n=50,
                  holdout_frac=0.2, n_bg=2, tpn=1, cdr="p.d", billing="bp")
    out = ccache.load_or_build_case_finding_bundle(
        spark, cache_uri=f"file://{tmp_path}/nope", _assemble_fn=lambda s, **k: built,
        **params)
    assert out is built                          # in-memory bundle returned, no raise
    assert "WARNING" in capsys.readouterr().out  # and the failure was surfaced


def test_bundle_cache_miss_then_hit_without_split_salt(spark, tmp_path):
    """Regression: the driver does NOT pass split_salt (there is no --split-salt),
    so a cached run must not require it. compute_bundle_cache_key defaults it to
    the assembly's _SPLIT_SALT; before that fix this call raised TypeError at
    startup on every `make exp` run with a cache_uri."""
    from _case_finding_cache import load_or_build_case_finding_bundle
    built = _tiny_bundle(spark)
    calls = {"n": 0}

    def _stub_assemble(spark_, **kw):
        calls["n"] += 1
        return built

    uri = f"file://{tmp_path}/cache3"
    # Exactly the kwargs dag_placement_cloud.py passes — NO split_salt.
    params = dict(source_table="condition_era", person_mod=10, vocab_size=5000,
                  min_df=20, min_patient_count=20, doc_min_length=0,
                  prior_obs_days=365, window_days=365, disease="diabetes", min_n=50,
                  holdout_frac=0.2, n_bg=2, tpn=1, cdr="p.d", billing="bp")
    b1 = load_or_build_case_finding_bundle(
        spark, cache_uri=uri, _assemble_fn=_stub_assemble, **params)
    b2 = load_or_build_case_finding_bundle(
        spark, cache_uri=uri, _assemble_fn=_stub_assemble, **params)
    assert calls["n"] == 1                       # miss builds once, then HIT
    assert b1.parent_int == b2.parent_int == built.parent_int
