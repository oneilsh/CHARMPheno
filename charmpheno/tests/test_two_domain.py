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
