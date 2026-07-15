"""Tests for categorical covariate-source diagnostics (covariates.py).

A declared categorical that decodes to a single level silently drops from
the design matrix (formulaic emits zero contrast columns for a constant
factor), which reads downstream as "the covariate did nothing" rather than
"the covariate was never there". exp 0027/0028 hit this when
`person.gender_concept_id` decoded to a single `sex` level. These
diagnostics surface the raw concept-id + decoded-level distribution at build
time so the collapse — and its cause — are visible.
"""
import pytest
from pyspark.sql.types import StructType, StructField, IntegerType, StringType


_PERSON_SCHEMA = StructType([
    StructField("person_id", IntegerType(), False),
    StructField("gender_concept_id", IntegerType(), True),
    StructField("sex", StringType(), True),
])


def _person_df(spark):
    rows = [
        (10, 8532, "F"),
        (20, 8532, "F"),
        (30, 8532, "F"),
        (40, 8507, "M"),
    ]
    return spark.createDataFrame(rows, schema=_PERSON_SCHEMA)


def test_categorical_level_counts_sorted_desc_and_truncated(spark):
    from charmpheno.omop.covariates import categorical_level_counts

    df = _person_df(spark)
    counts = categorical_level_counts(df, ["gender_concept_id", "sex"], top_n=10)
    # sorted by count descending, ties broken by string(value) ascending
    assert counts["gender_concept_id"] == [(8532, 3), (8507, 1)]
    assert counts["sex"] == [("F", 3), ("M", 1)]


def test_categorical_level_counts_respects_top_n(spark):
    from charmpheno.omop.covariates import categorical_level_counts

    df = _person_df(spark)
    counts = categorical_level_counts(df, ["gender_concept_id"], top_n=1)
    assert counts["gender_concept_id"] == [(8532, 3)]


def test_categorical_level_counts_skips_missing_columns(spark):
    from charmpheno.omop.covariates import categorical_level_counts

    df = _person_df(spark)
    counts = categorical_level_counts(df, ["sex", "race"], top_n=10)
    assert "race" not in counts
    assert counts["sex"] == [("F", 3), ("M", 1)]


def test_single_level_categoricals_identifies_collapsed_columns(spark):
    from charmpheno.omop.covariates import (
        categorical_level_counts,
        single_level_categoricals,
    )

    rows = [(1, 8532, "F"), (2, 8532, "F"), (3, 8532, "F")]
    df = spark.createDataFrame(rows, schema=_PERSON_SCHEMA)
    counts = categorical_level_counts(df, ["gender_concept_id", "sex"], top_n=10)
    assert counts["sex"] == [("F", 3)]
    assert single_level_categoricals(counts) == ["gender_concept_id", "sex"]
