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
