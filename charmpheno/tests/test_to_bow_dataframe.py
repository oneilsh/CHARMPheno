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
