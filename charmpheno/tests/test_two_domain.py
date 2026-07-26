import pytest

pyspark = pytest.importorskip("pyspark")


def _events(spark, rows, date_col):
    # rows: list of (person_id, concept_id). date is a constant in-window day.
    from pyspark.sql import Row
    return spark.createDataFrame(
        [Row(person_id=p, concept_id=c, **{date_col: "2020-01-01"}) for p, c in rows])


def test_two_domain_bow_emits_two_aligned_per_domain_columns(spark):
    from charmpheno.omop.two_domain import two_domain_bow, DomainVocabSpec
    from charmpheno.omop.doc_spec import PatientDocSpec
    # person 1 has both domains; person 2 has only conditions; person 3 only drugs.
    cond = _events(spark, [(1, 201), (1, 202), (2, 201)], "condition_era_start_date")
    drug = _events(spark, [(1, 900), (3, 901)], "drug_era_start_date")
    df, va, vb = two_domain_bow(
        cond, drug, doc_spec=PatientDocSpec(),
        vocab_a=DomainVocabSpec(vocab_size=100, min_df=1, min_patient_count=1),
        vocab_b=DomainVocabSpec(vocab_size=100, min_df=1, min_patient_count=1))
    rows = {r["person_id"]: r for r in df.collect()}
    # every doc has BOTH columns, each a SparseVector over its own vocab size:
    for r in rows.values():
        assert r["features_a"].size == len(va)
        assert r["features_b"].size == len(vb)
    # person 2 (no drugs) has an empty drug vector, not a dropped row:
    assert rows[2]["features_b"].numNonzeros() == 0
    assert 2 in rows and 3 in rows
    # person 3 (no conditions) has an empty condition vector:
    assert rows[3]["features_a"].numNonzeros() == 0
    # ids are within each domain's own [0, V) range:
    for r in rows.values():
        assert all(0 <= i < len(va) for i in r["features_a"].indices)
        assert all(0 <= i < len(vb) for i in r["features_b"].indices)


def test_two_domain_bow_vocab_sizes_are_independent(spark):
    """Per-domain vocab_size caps each domain separately."""
    from charmpheno.omop.two_domain import two_domain_bow, DomainVocabSpec
    from charmpheno.omop.doc_spec import PatientDocSpec
    cond = _events(spark, [(1, 201), (1, 202), (1, 203), (2, 204)],
                   "condition_era_start_date")
    drug = _events(spark, [(1, 900), (2, 901)], "drug_era_start_date")
    df, va, vb = two_domain_bow(
        cond, drug, doc_spec=PatientDocSpec(),
        vocab_a=DomainVocabSpec(vocab_size=2, min_df=1, min_patient_count=1),
        vocab_b=DomainVocabSpec(vocab_size=100, min_df=1, min_patient_count=1))
    assert len(va) == 2 and len(vb) == 2   # condition capped at 2; drug has 2 tokens


import datetime as dt

from charmpheno.omop.condition_dag import build_condition_dag


def _two_domain_bundle(spark, *, strip_mode="both"):
    """A CLEAN two-domain bundle for fit/seam tests: conditions attest DAG
    nodes (+ a rides-along non-node code); drugs are ordinary drug concept-ids
    in a domain DISJOINT from the condition vocabulary. Unlike the strip
    test's fixture below, this one does NOT emit a node-marker id as a drug
    token -- that synthetic overlap exists only to pin the per-domain-strip
    symmetry property, and would be noise in a plain fit smoke test."""
    from charmpheno.omop.two_domain import (
        assemble_two_domain_from_events, DomainVocabSpec)
    from charmpheno.omop.doc_spec import PatientCohortDocSpec
    # anchor 100 -> node 200, node 300 (a 2-node DAG), same shape as the strip
    # test's fixture but without the synthetic marker-in-drugs row.
    before = build_condition_dag(
        [(100, 200), (100, 300)], anchor=100, node_ids=[200, 300],
        names={100: "root", 200: "A", 300: "B"})
    cond_rows, drug_rows = [], []
    for pid in range(20):                       # foreground: attest a node + a drug
        node = 200 if pid % 2 == 0 else 300
        cond_rows.append((pid, node, "dz", dt.date(2015, 1, 1)))
        cond_rows.append((pid, 999, "dz", dt.date(2015, 2, 1)))    # rides-along non-node
        drug_rows.append((pid, 900 + (pid % 3), "dz", dt.date(2015, 1, 5)))
    for pid in range(100, 115):                 # background
        cond_rows.append((pid, 888, "bg", dt.date(2016, 1, 1)))
        drug_rows.append((pid, 950, "bg", dt.date(2016, 1, 5)))
    cond = spark.createDataFrame(
        cond_rows, ["person_id", "concept_id", "source_cohort", "condition_era_start_date"])
    drug = spark.createDataFrame(
        drug_rows, ["person_id", "concept_id", "source_cohort", "drug_era_start_date"])
    return assemble_two_domain_from_events(
        cond, drug, before, doc_spec=PatientCohortDocSpec(min_doc_length=0), min_n=1,
        vocab_a=DomainVocabSpec(vocab_size=100, min_df=1, min_patient_count=1),
        vocab_b=DomainVocabSpec(vocab_size=100, min_df=1, min_patient_count=1),
        holdout_frac=0.3, strip_mode=strip_mode)


def test_assemble_two_domain_bundle_shape_and_per_domain_strip(spark):
    """The bundle exposes two aligned feature columns + a CONDITION-ONLY frontier;
    the per-domain leakage strip removes the condition node-marker ids from
    features_a and leaves features_b (drug) untouched (node markers are condition
    concept-ids -- they define the DAG -- so only the condition domain holds them)."""
    from charmpheno.omop.two_domain import (
        assemble_two_domain_from_events, DomainVocabSpec)
    from charmpheno.omop.doc_spec import PatientCohortDocSpec
    # anchor 100 -> node 200, node 300 (a 2-node DAG). 200/300 ARE the node markers.
    before = build_condition_dag(
        [(100, 200), (100, 300)], anchor=100, node_ids=[200, 300],
        names={100: "root", 200: "A", 300: "B"})
    cond_rows, drug_rows = [], []
    for pid in range(20):                       # foreground: attest a node + a drug
        node = 200 if pid % 2 == 0 else 300
        cond_rows.append((pid, node, "dz", dt.date(2015, 1, 1)))
        cond_rows.append((pid, 999, "dz", dt.date(2015, 2, 1)))    # rides-along non-node
        drug_rows.append((pid, 900 + (pid % 3), "dz", dt.date(2015, 1, 5)))
        # SYNTHETIC (not realistic OMOP -- condition and drug concept-ids are
        # disjoint namespaces): emit the node-marker concept-id 200 as a DRUG
        # token too, so 200 lands in BOTH vocab_map_a AND vocab_map_b. This is
        # the only way to test that the strip LOGIC is symmetric across domains
        # (a hardcode-to-A strip would leave 200 in features_b).
        drug_rows.append((pid, 200, "dz", dt.date(2015, 1, 6)))
    for pid in range(100, 115):                 # background
        cond_rows.append((pid, 888, "bg", dt.date(2016, 1, 1)))
        drug_rows.append((pid, 950, "bg", dt.date(2016, 1, 5)))
    cond = spark.createDataFrame(
        cond_rows, ["person_id", "concept_id", "source_cohort", "condition_era_start_date"])
    drug = spark.createDataFrame(
        drug_rows, ["person_id", "concept_id", "source_cohort", "drug_era_start_date"])
    bundle = assemble_two_domain_from_events(
        cond, drug, before, doc_spec=PatientCohortDocSpec(min_doc_length=0), min_n=1,
        vocab_a=DomainVocabSpec(vocab_size=100, min_df=1, min_patient_count=1),
        vocab_b=DomainVocabSpec(vocab_size=100, min_df=1, min_patient_count=1),
        holdout_frac=0.3, strip_mode="both")
    cols = set(bundle.train_df.columns)
    assert {"person_id", "doc_id", "features_a", "features_b",
            "frontier", "source_cohort"} <= cols
    # frontier ids are engine node-ids from the CONDITION DAG only (2 nodes -> ids in a
    # small range); a drug never appears in a frontier:
    fr = [f for r in bundle.train_df.collect() for f in (r["frontier"] or [])]
    assert fr and max(fr) < len(bundle.parent_int) + 2
    # per-domain strip: the condition marker 200's vocab index is zeroed in features_a
    # for every doc; features_b is untouched (its drug tokens remain).
    a200 = bundle.vocab_map_a.get(200)
    assert a200 is not None
    for r in bundle.train_df.collect() + bundle.test_df.collect():
        assert a200 not in set(r["features_a"].indices)     # stripped from conditions
    # per-domain-ness: 200 is ALSO a drug token here (synthetic), so it lands in
    # vocab_map_b; a symmetric per-domain strip zeroes it in features_b too, while
    # a strip hardcoded to features_a would leave it. This pins the load-bearing
    # property (the strip is per-domain, not hardcoded to A).
    b200 = bundle.vocab_map_b.get(200)
    assert b200 is not None
    for r in bundle.train_df.collect() + bundle.test_df.collect():
        assert b200 not in set(r["features_b"].indices)     # stripped from drugs too
    assert any(r["features_b"].numNonzeros() > 0
               for r in bundle.train_df.collect())          # other drugs intact


def test_two_domain_bundle_fits_through_the_gated_shim_and_round_trips(spark, tmp_path):
    """The SP3b<->SP3a seam: a two-domain bundle fits via GatedLDAEstimator with
    featuresCols=[features_a, features_b], yields a per-domain dict lambda, and the
    saved VIResult round-trips. Structural (shape + round-trip), not a recovery gate."""
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    from spark_vi.io.export import save_result, load_result
    bundle = _two_domain_bundle(spark)
    est = GatedLDAEstimator(
        featuresCols=["features_a", "features_b"], labelCol="frontier",
        parent=bundle.parent_int, nBg=2, tpn=1, maxIter=2, seed=0)
    model = est.fit(bundle.train_df)
    lam = model.result.global_params["lambda"]
    assert isinstance(lam, dict) and set(lam) == {0, 1}
    assert model.result.metadata["domains"] == [len(bundle.vocab_map_a),
                                                 len(bundle.vocab_map_b)]
    save_result(model.result, tmp_path / "fit")
    loaded = load_result(tmp_path / "fit")
    assert isinstance(loaded.global_params["lambda"], dict)
    assert loaded.metadata["domains"] == [len(bundle.vocab_map_a),
                                          len(bundle.vocab_map_b)]
