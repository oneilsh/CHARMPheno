import datetime as dt

import pytest

pyspark = pytest.importorskip("pyspark")

from charmpheno.omop.condition_dag import build_condition_dag


def _events(spark, rows, date_col):
    from pyspark.sql import Row
    return spark.createDataFrame(
        [Row(person_id=p, concept_id=c, **{date_col: "2020-01-01"}) for p, c in rows])


def test_multidomain_bow_emits_n_aligned_per_domain_columns(spark):
    from charmpheno.omop.multi_domain import multidomain_bow, DomainVocabSpec
    from charmpheno.omop.doc_spec import PatientDocSpec
    cond = _events(spark, [(1, 201), (1, 202), (2, 201)], "condition_era_start_date")
    drug = _events(spark, [(1, 900), (3, 901)], "drug_era_start_date")
    obs = _events(spark, [(1, 700), (2, 701)], "observation_date")
    spec = DomainVocabSpec(vocab_size=100, min_df=1, min_patient_count=1)
    df, vms = multidomain_bow([cond, drug, obs], [spec, spec, spec],
                              doc_spec=PatientDocSpec())
    assert len(vms) == 3
    rows = {r["person_id"]: r for r in df.collect()}
    for r in rows.values():                       # every doc has all 3 columns
        assert r["features_0"].size == len(vms[0])
        assert r["features_1"].size == len(vms[1])
        assert r["features_2"].size == len(vms[2])
    assert rows[3]["features_0"].numNonzeros() == 0   # person 3: no conditions
    assert rows[3]["features_1"].numNonzeros() == 1   #           has a drug
    assert rows[2]["features_1"].numNonzeros() == 0   # person 2: no drugs
    for r in rows.values():                       # ids within each domain's [0, V)
        for i in range(3):
            assert all(0 <= j < len(vms[i]) for j in r[f"features_{i}"].indices)


def test_multidomain_bow_length_mismatch_raises(spark):
    from charmpheno.omop.multi_domain import multidomain_bow, DomainVocabSpec
    from charmpheno.omop.doc_spec import PatientDocSpec
    cond = _events(spark, [(1, 201)], "condition_era_start_date")
    spec = DomainVocabSpec(vocab_size=100, min_df=1, min_patient_count=1)
    with pytest.raises(ValueError, match="same length"):
        multidomain_bow([cond], [spec, spec], doc_spec=PatientDocSpec())


def _three_domain_bundle(spark, *, marker_in_obs=False, strip_mode="both"):
    """A CLEAN 3-domain bundle: conditions attest DAG nodes (+ a rides-along
    non-node code); drugs + observations are ordinary tokens in their own
    namespaces. When marker_in_obs, the condition node-marker id 200 is ALSO
    emitted as an OBSERVATION token, to pin the per-domain strip over a
    non-condition domain (the spec's defensive guarantee)."""
    from charmpheno.omop.multi_domain import (
        assemble_multidomain_from_events, DomainVocabSpec)
    from charmpheno.omop.doc_spec import PatientCohortDocSpec
    before = build_condition_dag(
        [(100, 200), (100, 300)], anchor=100, node_ids=[200, 300],
        names={100: "root", 200: "A", 300: "B"})
    cond_rows, drug_rows, obs_rows = [], [], []
    for pid in range(20):
        node = 200 if pid % 2 == 0 else 300
        cond_rows.append((pid, node, "dz", dt.date(2015, 1, 1)))
        cond_rows.append((pid, 999, "dz", dt.date(2015, 2, 1)))     # rides-along
        drug_rows.append((pid, 900 + (pid % 3), "dz", dt.date(2015, 1, 5)))
        obs_rows.append((pid, 700 + (pid % 2), "dz", dt.date(2015, 1, 7)))
        if marker_in_obs:
            obs_rows.append((pid, 200, "dz", dt.date(2015, 1, 8)))  # marker in OBS
    for pid in range(100, 115):
        cond_rows.append((pid, 888, "bg", dt.date(2016, 1, 1)))
        drug_rows.append((pid, 950, "bg", dt.date(2016, 1, 5)))
        obs_rows.append((pid, 750, "bg", dt.date(2016, 1, 7)))
    cond = spark.createDataFrame(
        cond_rows, ["person_id", "concept_id", "source_cohort", "condition_era_start_date"])
    drug = spark.createDataFrame(
        drug_rows, ["person_id", "concept_id", "source_cohort", "drug_era_start_date"])
    obs = spark.createDataFrame(
        obs_rows, ["person_id", "concept_id", "source_cohort", "observation_date"])
    spec = DomainVocabSpec(vocab_size=100, min_df=1, min_patient_count=1)
    return assemble_multidomain_from_events(
        cond, [drug, obs], before, doc_spec=PatientCohortDocSpec(min_doc_length=0),
        min_n=1, vocab_specs=[spec, spec, spec], holdout_frac=0.3,
        strip_mode=strip_mode)


def test_assemble_multidomain_shape_and_condition_only_frontier(spark):
    bundle = _three_domain_bundle(spark)
    cols = set(bundle.train_df.columns)
    assert {"person_id", "doc_id", "features_0", "features_1", "features_2",
            "frontier", "source_cohort"} <= cols
    assert len(bundle.vocab_maps) == 3
    fr = [f for r in bundle.train_df.collect() for f in (r["frontier"] or [])]
    assert fr and max(fr) < len(bundle.parent_int) + 2   # engine node-ids only


def test_multidomain_strip_removes_marker_from_a_noncondition_domain(spark):
    """The defensive guarantee: a node-marker concept-id that (wrongly, per OMOP
    convention) lands in the OBSERVATION vocabulary is still stripped from
    features_2 -- the strip loops over all N vocabs, not just conditions."""
    bundle = _three_domain_bundle(spark, marker_in_obs=True, strip_mode="both")
    # domain 0 = conditions: marker 200 stripped from features_0
    m0 = bundle.vocab_maps[0].get(200)
    assert m0 is not None
    # domain 2 = observation: marker 200 ALSO present here (synthetic) and stripped
    m2 = bundle.vocab_maps[2].get(200)
    assert m2 is not None
    for r in bundle.train_df.collect() + bundle.test_df.collect():
        assert m0 not in set(r["features_0"].indices)
        assert m2 not in set(r["features_2"].indices)   # stripped from observation too
    assert any(r["features_2"].numNonzeros() > 0
               for r in bundle.train_df.collect())       # other obs tokens intact


def test_lookback_feature_frames_splits_all_domains_against_one_index(spark):
    """One shared (condition-derived) index splits every domain into a pre-index
    FEATURE frame; the condition (domain 0) forward window is the LABEL frame. A
    post-index drug/observation event never enters any feature frame."""
    from charmpheno.omop.multi_domain import lookback_feature_frames
    from pyspark.sql import Row
    # index: person 1 indexed 2020-06-01.
    index_df = spark.createDataFrame(
        [Row(person_id=1, index_date=dt.date(2020, 6, 1), source_cohort="dz")])
    cond = spark.createDataFrame([
        Row(person_id=1, concept_id=201, condition_era_start_date=dt.date(2020, 1, 1)),  # pre  -> feature
        Row(person_id=1, concept_id=202, condition_era_start_date=dt.date(2020, 7, 1)),  # post -> label
    ])
    drug = spark.createDataFrame([
        Row(person_id=1, concept_id=900, drug_era_start_date=dt.date(2020, 2, 1)),       # pre  -> feature
        Row(person_id=1, concept_id=901, drug_era_start_date=dt.date(2020, 8, 1)),       # post -> dropped
    ])
    obs = spark.createDataFrame([
        Row(person_id=1, concept_id=700, observation_date=dt.date(2020, 3, 1)),          # pre  -> feature
    ])
    feats, cond_label = lookback_feature_frames(
        [cond, drug, obs], index_df,
        ["condition_era_start_date", "drug_era_start_date", "observation_date"],
        lookback_days=365, label_window_days=365)
    assert len(feats) == 3
    cond_feat_cids = {r["concept_id"] for r in feats[0].collect()}
    drug_feat_cids = {r["concept_id"] for r in feats[1].collect()}
    obs_feat_cids = {r["concept_id"] for r in feats[2].collect()}
    label_cids = {r["concept_id"] for r in cond_label.collect()}
    assert cond_feat_cids == {201}          # pre-index condition only
    assert drug_feat_cids == {900}          # pre-index drug; post-index 901 dropped
    assert obs_feat_cids == {700}
    assert label_cids == {202}              # forward-window condition only
    # every feature frame carries source_cohort (from the index join)
    assert "source_cohort" in feats[1].columns


def test_multidomain_bundle_fits_through_the_gated_shim_and_round_trips(spark, tmp_path):
    """The SP3c<->SP3a seam: a 3-domain bundle fits via GatedLDAEstimator with
    featuresCols=[features_0, features_1, features_2], yields a per-domain dict
    lambda over 3 domains, and the saved VIResult round-trips."""
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    from spark_vi.io.export import save_result, load_result
    bundle = _three_domain_bundle(spark)
    est = GatedLDAEstimator(
        featuresCols=["features_0", "features_1", "features_2"], labelCol="frontier",
        parent=bundle.parent_int, nBg=2, tpn=1, maxIter=2, seed=0)
    model = est.fit(bundle.train_df)
    lam = model.result.global_params["lambda"]
    assert isinstance(lam, dict) and set(lam) == {0, 1, 2}
    assert model.result.metadata["domains"] == [len(vm) for vm in bundle.vocab_maps]
    save_result(model.result, tmp_path / "fit")
    loaded = load_result(tmp_path / "fit")
    assert isinstance(loaded.global_params["lambda"], dict)
    assert loaded.metadata["domains"] == [len(vm) for vm in bundle.vocab_maps]


# --------------------------------------------------------------------------- #
# exp 0111 WP-C behavior-preservation oracles: the index_date passthrough is    #
# purely additive, and the assembler's index_df / doc_spec are pure seams.      #
# --------------------------------------------------------------------------- #
def test_lookback_feature_frames_carries_index_date_additively(spark):
    """The R5.2 passthrough is ADDITIVE. On the current one-index-per-person shape,
    each frame equals the pre-WP-C output except for the added index_date column —
    same columns otherwise, same rows, same values — and index_date carries the
    correct per-person anchor. Dropping the column reproduces the old contract; the
    add is exactly what an EpisodeDocSpec keys on."""
    from charmpheno.omop.multi_domain import lookback_feature_frames
    from pyspark.sql import Row
    index_df = spark.createDataFrame([
        Row(person_id=1, index_date=dt.date(2020, 6, 1), source_cohort="dz"),
        Row(person_id=2, index_date=dt.date(2021, 3, 1), source_cohort="general"),
    ])
    cond = spark.createDataFrame([
        Row(person_id=1, concept_id=201, condition_era_start_date=dt.date(2020, 1, 1)),  # pre -> feat
        Row(person_id=1, concept_id=202, condition_era_start_date=dt.date(2020, 7, 1)),  # post -> label
        Row(person_id=2, concept_id=203, condition_era_start_date=dt.date(2021, 2, 1)),  # pre -> feat
    ])
    drug = spark.createDataFrame([
        Row(person_id=1, concept_id=900, drug_era_start_date=dt.date(2020, 2, 1)),        # pre -> feat
        Row(person_id=1, concept_id=901, drug_era_start_date=dt.date(2020, 8, 1)),        # post -> dropped
    ])
    feats, cond_label = lookback_feature_frames(
        [cond, drug], index_df, ["condition_era_start_date", "drug_era_start_date"],
        lookback_days=365, label_window_days=365)
    cond_feat, drug_feat = feats

    anchors = {r["person_id"]: r["index_date"] for r in index_df.collect()}
    for frame in (cond_feat, drug_feat, cond_label):
        assert "index_date" in frame.columns
        for r in frame.collect():
            assert r["index_date"] == anchors[r["person_id"]]

    # Additivity: with index_date dropped, the frames are exactly the old shape.
    assert set(cond_feat.drop("index_date").columns) == {
        "person_id", "concept_id", "condition_era_start_date", "source_cohort"}
    assert set(drug_feat.drop("index_date").columns) == {
        "person_id", "concept_id", "drug_era_start_date", "source_cohort"}
    assert {(r["person_id"], r["concept_id"], r["source_cohort"])
            for r in cond_feat.collect()} == {(1, 201, "dz"), (2, 203, "general")}
    assert {(r["person_id"], r["concept_id"]) for r in drug_feat.collect()} == {(1, 900)}
    assert {(r["person_id"], r["concept_id"]) for r in cond_label.collect()} == {(1, 202)}


def test_assemble_multidomain_case_finding_corpus_wpc_seams(monkeypatch):
    """WP-C injection-seam oracle (R7.1). external REQUIRES index_df and uses it
    verbatim; population/disease REJECT a non-None index_df; doc_spec=None
    reproduces today's PatientCohortDocSpec(min_doc_length=doc_min_length)
    byte-for-byte and a passed spec is forwarded as-is. BigQuery and the real
    lookback/assemble are stubbed — only the seam wiring runs, so this needs no
    Spark and no CDR."""
    import charmpheno.omop as omop_pkg
    from charmpheno.omop import multi_domain as md
    from charmpheno.omop import cohorts
    from charmpheno.omop.doc_spec import PatientCohortDocSpec

    monkeypatch.setattr(omop_pkg, "load_omop_bigquery", lambda **kw: object())

    captured = {}

    def _fake_lookback(domain_raws, index_df, date_cols, *, lookback_days,
                       label_window_days):
        captured["index_df"] = index_df
        return [object() for _ in domain_raws], object()  # (feature_frames, cond_label)
    monkeypatch.setattr(md, "lookback_feature_frames", _fake_lookback)

    def _fake_assemble(cond0, extras, before_dag, *, doc_spec, **kw):
        captured["doc_spec"] = doc_spec
        return "BUNDLE"
    monkeypatch.setattr(md, "assemble_multidomain_from_events", _fake_assemble)

    # The self-building index tables must never run on the external path.
    monkeypatch.setattr(cohorts, "case_finding_index_table",
                        lambda *a, **k: pytest.fail("disease index builder ran"))
    monkeypatch.setattr(cohorts, "case_finding_population_index_table",
                        lambda *a, **k: pytest.fail("population index builder ran"))

    common = dict(disease="rare6", cdr="p.d", billing="bp", extra_domains=("drug",),
                  person_mod=1, min_n=0, vocab_size=10, min_df=1, min_patient_count=1,
                  doc_min_length=7, before_dag=object())  # non-None DAG skips the load

    # external REQUIRES index_df
    with pytest.raises(ValueError, match="external.*requires an index_df"):
        md.assemble_multidomain_case_finding_corpus(
            None, index_mode="external", index_df=None, **common)

    # population / disease REJECT a non-None index_df (they own their index)
    with pytest.raises(ValueError, match="population.*does not accept an index_df"):
        md.assemble_multidomain_case_finding_corpus(
            None, index_mode="population", index_df=object(), **common)
    with pytest.raises(ValueError, match="disease.*does not accept an index_df"):
        md.assemble_multidomain_case_finding_corpus(
            None, index_mode="disease", index_df=object(), **common)

    # external uses index_df VERBATIM; doc_spec=None -> today's default spec.
    my_index = object()
    out = md.assemble_multidomain_case_finding_corpus(
        None, index_mode="external", index_df=my_index, doc_spec=None, **common)
    assert out == "BUNDLE"
    assert captured["index_df"] is my_index
    default_spec = captured["doc_spec"]
    assert isinstance(default_spec, PatientCohortDocSpec)
    assert default_spec.name == "patient_cohort"
    assert default_spec.min_doc_length == 7          # == doc_min_length, byte-for-byte

    # a passed doc_spec is forwarded verbatim.
    sentinel = PatientCohortDocSpec(min_doc_length=3)
    md.assemble_multidomain_case_finding_corpus(
        None, index_mode="external", index_df=object(), doc_spec=sentinel, **common)
    assert captured["doc_spec"] is sentinel
