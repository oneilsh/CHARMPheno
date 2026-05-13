"""Tests for charmpheno.omop.to_bow_dataframe.

Builds a tiny in-memory DataFrame with the canonical OMOP columns and
verifies the bag-of-words conversion produces a clean SparseVector column
plus a deterministic vocab map.
"""
import numpy as np
import pytest
from pyspark.sql.types import StructType, StructField, IntegerType, StringType


def _tiny_omop_df(spark):
    """Hand-crafted 4-column OMOP fixture with two patients and a known token mix."""
    schema = StructType([
        StructField("person_id", IntegerType(), False),
        StructField("visit_occurrence_id", IntegerType(), False),
        StructField("concept_id", IntegerType(), False),
        StructField("concept_name", StringType(), True),
    ])
    rows = [
        (1, 100, 4567, "fever"),
        (1, 100, 4567, "fever"),
        (1, 101, 8910, "cough"),
        (2, 200, 4567, "fever"),
        (2, 200, 1234, "rash"),
    ]
    return spark.createDataFrame(rows, schema=schema)


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
    schema = StructType([
        StructField("person_id", IntegerType(), False),
        StructField("visit_occurrence_id", IntegerType(), False),
        StructField("concept_id", IntegerType(), False),
        StructField("concept_name", StringType(), True),
    ])
    rows = [
        (10, 1000, 1234, "rash"),
        (10, 1000, 1234, "rash"),
        (10, 1000, 1234, "rash"),
        (10, 1001, 4567, "fever"),
        (11, 1100, 9999, "out-of-vocab"),
        (11, 1100, 9999, "out-of-vocab"),
    ]
    df2 = spark.createDataFrame(rows, schema=schema)

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
