"""Tests for charmpheno.omop.to_bow_dataframe.

Builds a tiny in-memory DataFrame with the canonical OMOP columns and
verifies the bag-of-words conversion produces a clean SparseVector column
plus a deterministic vocab map.
"""
import numpy as np
import pytest
from datetime import date
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DateType


_OMOP_SCHEMA = StructType([
    StructField("person_id", IntegerType(), False),
    StructField("visit_occurrence_id", IntegerType(), False),
    StructField("concept_id", IntegerType(), False),
    StructField("concept_name", StringType(), True),
])


def _tiny_omop_df(spark):
    """Hand-crafted 4-column OMOP fixture with two patients and a known token mix."""
    rows = [
        (1, 100, 4567, "fever"),
        (1, 100, 4567, "fever"),
        (1, 101, 8910, "cough"),
        (2, 200, 4567, "fever"),
        (2, 200, 1234, "rash"),
    ]
    return spark.createDataFrame(rows, schema=_OMOP_SCHEMA)


def test_to_bow_dataframe_returns_sparse_features_per_patient(spark):
    from charmpheno.omop import to_bow_dataframe

    df = _tiny_omop_df(spark)
    bow_df, vocab_map = to_bow_dataframe(df)
    rows = sorted(bow_df.collect(), key=lambda r: r["person_id"])

    assert len(rows) == 2
    assert rows[0]["person_id"] == 1
    assert rows[1]["person_id"] == 2

    idx_to_concept = {v: k for k, v in vocab_map.items()}

    sv1 = rows[0]["features"]
    counts_by_concept_1 = {idx_to_concept[idx]: int(c) for idx, c in zip(sv1.indices, sv1.values)}
    assert counts_by_concept_1 == {4567: 2, 8910: 1}

    sv2 = rows[1]["features"]
    counts_by_concept_2 = {idx_to_concept[idx]: int(c) for idx, c in zip(sv2.indices, sv2.values)}
    assert counts_by_concept_2 == {4567: 1, 1234: 1}


def test_to_bow_dataframe_vocab_map_is_complete_and_contiguous(spark):
    from charmpheno.omop import to_bow_dataframe
    df = _tiny_omop_df(spark)
    _, vocab_map = to_bow_dataframe(df)
    expected_concepts = {4567, 8910, 1234}
    assert set(vocab_map.keys()) == expected_concepts
    indices = sorted(vocab_map.values())
    assert indices == [0, 1, 2]


def test_to_bow_dataframe_vocab_order_is_stable_across_calls(spark):
    """Repeated fits on the same input produce the same vocab map.

    Guards against accidental nondeterminism if the implementation is
    later swapped for something with hash-based bucketing.
    """
    from charmpheno.omop import to_bow_dataframe
    df = _tiny_omop_df(spark)
    _, v1 = to_bow_dataframe(df)
    _, v2 = to_bow_dataframe(df)
    assert v1 == v2
    # The most-frequent concept (4567 appears 3 times) must be index 0;
    # CountVectorizer orders by descending frequency.
    assert v1[4567] == 0


def test_to_bow_dataframe_with_frozen_vocab_preserves_index_assignments(spark):
    """When a vocab list is supplied, index assignments are fixed by the
    list order — not by the input data's term frequencies. A second call
    against a different-distribution corpus produces the same vocab_map.

    This is the contract the eval driver relies on: load vocab from the
    checkpoint, freeze it through the BOW build, and the topic-term
    matrix's column indices stay aligned with whatever corpus you feed.
    """
    from charmpheno.omop import to_bow_dataframe

    df1 = _tiny_omop_df(spark)
    _, vocab_map_1 = to_bow_dataframe(df1)

    # Invert the dict to a positional list (the form saved in
    # VIResult.metadata["vocab"]).
    vocab_list = [None] * len(vocab_map_1)
    for cid, idx in vocab_map_1.items():
        vocab_list[idx] = cid

    # Build a second OMOP fixture with a deliberately different term mix:
    # rash (1234) is now the most-frequent concept, and concept 9999 is
    # entirely outside the frozen vocab and should be dropped.
    rows = [
        (10, 1000, 1234, "rash"),
        (10, 1000, 1234, "rash"),
        (10, 1000, 1234, "rash"),
        (10, 1001, 4567, "fever"),
        (11, 1100, 9999, "out-of-vocab"),
        (11, 1100, 9999, "out-of-vocab"),
    ]
    df2 = spark.createDataFrame(rows, schema=_OMOP_SCHEMA)

    bow_df, vocab_map_2 = to_bow_dataframe(df2, vocab=vocab_list)

    # The frozen path returns the same vocab_map regardless of df2's
    # term distribution.
    assert vocab_map_2 == vocab_map_1

    # Verify the BOW dropped the out-of-vocab concept (9999) and preserved
    # the in-vocab counts at the frozen indices.
    rows_out = {r["person_id"]: r["features"] for r in bow_df.collect()}
    idx_4567 = vocab_map_1[4567]
    idx_1234 = vocab_map_1[1234]
    sv_10 = rows_out[10]
    counts_10 = dict(zip(sv_10.indices, sv_10.values))
    assert counts_10.get(idx_1234, 0) == 3
    assert counts_10.get(idx_4567, 0) == 1

    # Patient 11 only had out-of-vocab terms; nothing of their tokens
    # makes it into the SparseVector.
    sv_11 = rows_out[11]
    assert len(sv_11.indices) == 0


def test_to_bow_dataframe_rejects_vocab_with_conflicting_knobs(spark):
    """Combining a frozen vocab with vocab_size / min_df is ambiguous."""
    from charmpheno.omop import to_bow_dataframe
    df = _tiny_omop_df(spark)
    with pytest.raises(ValueError, match="vocab=<frozen list> is incompatible"):
        to_bow_dataframe(df, vocab=[4567, 8910], min_df=2)
    with pytest.raises(ValueError, match="vocab=<frozen list> is incompatible"):
        to_bow_dataframe(df, vocab=[4567, 8910], vocab_size=500)


def test_to_bow_dataframe_rejects_vocab_with_none_slots(spark):
    """A saved vocab_list with None entries is treated as a malformed
    checkpoint, not silently dropped."""
    from charmpheno.omop import to_bow_dataframe
    df = _tiny_omop_df(spark)
    with pytest.raises(ValueError, match="frozen vocab contains None"):
        to_bow_dataframe(df, vocab=[4567, None, 1234])


def test_min_patient_count_filters_under_patient_doc_spec(spark):
    """Under PatientDocSpec, min_patient_count and min_df should be equivalent
    because each patient is exactly one document."""
    from charmpheno.omop import to_bow_dataframe
    from charmpheno.omop.doc_spec import PatientDocSpec

    # 3 patients. Code 9999 appears in patient 1 only; codes 4567/8910 appear in all three.
    rows = [
        (1, 100, 4567, "fever"),
        (1, 100, 8910, "cough"),
        (1, 100, 9999, "rare-thing"),
        (2, 200, 4567, "fever"),
        (2, 200, 8910, "cough"),
        (3, 300, 4567, "fever"),
        (3, 300, 8910, "cough"),
    ]
    df = spark.createDataFrame(rows, schema=_OMOP_SCHEMA)

    _, vocab_map = to_bow_dataframe(df, doc_spec=PatientDocSpec(), min_patient_count=2)
    # 9999 (1 patient) dropped; 4567 and 8910 (3 patients each) survive.
    assert set(vocab_map.keys()) == {4567, 8910}


def test_min_patient_count_stricter_than_min_df_in_patient_year_mode(spark):
    """Under PatientYearDocSpec, min_patient_count enforces a stricter privacy
    threshold than min_df: one patient can contribute multiple year-docs."""
    from charmpheno.omop import to_bow_dataframe
    from charmpheno.omop.doc_spec import PatientYearDocSpec

    # 2 patients. Code 9999 appears across 3 different years for patient 1 only
    # (so 3 docs but 1 patient). Codes 4567/8910 in both patients.
    rows = [
        # patient 1: 2020, 2021, 2022 (era-replicated; 9999 will appear in 3 docs)
        (1, 100, 9999, "rare-thing-2020", date(2020, 1, 1), date(2022, 12, 31)),
        (1, 100, 4567, "fever", date(2020, 1, 1), date(2020, 12, 31)),
        (1, 100, 8910, "cough", date(2021, 1, 1), date(2021, 12, 31)),
        # patient 2: 2020
        (2, 200, 4567, "fever", date(2020, 1, 1), date(2020, 12, 31)),
        (2, 200, 8910, "cough", date(2020, 1, 1), date(2020, 12, 31)),
    ]
    schema = StructType([
        StructField("person_id", IntegerType(), False),
        StructField("visit_occurrence_id", IntegerType(), False),
        StructField("concept_id", IntegerType(), False),
        StructField("concept_name", StringType(), True),
        StructField("condition_era_start_date", DateType(), False),
        StructField("condition_era_end_date", DateType(), True),
    ])
    df = spark.createDataFrame(rows, schema=schema)

    # min_df=2 alone: 9999 has 3 docs → passes; survives wrongly from a privacy
    # standpoint. min_patient_count=2: 9999 has 1 patient → dropped.
    _, vocab_md = to_bow_dataframe(
        df, doc_spec=PatientYearDocSpec(min_doc_length=0), min_df=2,
    )
    assert 9999 in vocab_md  # min_df=2 alone does not protect

    _, vocab_mpc = to_bow_dataframe(
        df, doc_spec=PatientYearDocSpec(min_doc_length=0), min_patient_count=2,
    )
    assert 9999 not in vocab_mpc  # min_patient_count=2 does


def test_min_patient_count_and_min_df_compose_as_and(spark):
    """min_df and min_patient_count both must hold; either failing drops the
    token."""
    from charmpheno.omop import to_bow_dataframe
    from charmpheno.omop.doc_spec import PatientDocSpec

    # 4 patients. Token 1234 appears in 3 patients, 1 doc each (3 docs, 3 patients).
    # Token 4567 appears in 1 patient, 5 visits (so 1 doc total under PatientDocSpec
    # because PatientDocSpec collapses all visits per patient).
    rows = [
        (1, 100, 1234, "a"),
        (2, 200, 1234, "a"),
        (3, 300, 1234, "a"),
        # patient 4: 5 records of 4567 across visits, but PatientDocSpec → 1 doc, 1 patient.
        (4, 400, 4567, "b"),
        (4, 401, 4567, "b"),
        (4, 402, 4567, "b"),
        (4, 403, 4567, "b"),
        (4, 404, 4567, "b"),
    ]
    df = spark.createDataFrame(rows, schema=_OMOP_SCHEMA)

    _, vocab_map = to_bow_dataframe(
        df, doc_spec=PatientDocSpec(), min_df=2, min_patient_count=2,
    )
    # 1234: 3 docs, 3 patients → passes both.
    # 4567: 1 doc, 1 patient → fails both (and AND-composed filter).
    assert set(vocab_map.keys()) == {1234}


def test_vocab_and_min_patient_count_conflict_raises(spark):
    """When caller passes a frozen vocab, min_patient_count != 1 should raise."""
    from charmpheno.omop import to_bow_dataframe
    from charmpheno.omop.doc_spec import PatientDocSpec

    df = _tiny_omop_df(spark)
    with pytest.raises(ValueError, match="min_patient_count"):
        to_bow_dataframe(df, doc_spec=PatientDocSpec(), vocab=[4567, 8910], min_patient_count=20)
