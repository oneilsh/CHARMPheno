"""Canonical OMOP shape + validator.

The canonical-columns set was relaxed in 2026-05-12 (ADR 0018) to allow
condition_era as a loader source. visit_occurrence_id moved from
required-canonical to optional-but-typed; condition_*_date columns are
likewise optional-but-typed when present.
"""
import pytest


def test_canonical_columns_are_required_three():
    from charmpheno.omop.schema import CANONICAL_COLUMNS

    assert CANONICAL_COLUMNS == (
        "person_id", "concept_id", "concept_name",
    )


def test_validate_accepts_canonical_dataframe(spark):
    from charmpheno.omop.schema import validate

    df = spark.createDataFrame(
        [(1, 5, "diabetes"), (2, 6, "asthma")],
        schema="person_id INT, concept_id INT, concept_name STRING",
    )
    validate(df)  # no exception


def test_validate_accepts_canonical_with_visit_occurrence_id(spark):
    """Loaders that emit visit_occurrence_id (the condition_occurrence
    path) still pass validation; the column is type-checked when present."""
    from charmpheno.omop.schema import validate

    df = spark.createDataFrame(
        [(1, 100, 5, "diabetes"), (2, 101, 6, "asthma")],
        schema="person_id INT, visit_occurrence_id INT, concept_id INT, concept_name STRING",
    )
    validate(df)  # no exception


def test_validate_rejects_missing_required_column(spark):
    from charmpheno.omop.schema import validate

    df = spark.createDataFrame(
        [(1, "diabetes")],
        schema="person_id INT, concept_name STRING",
    )
    with pytest.raises(ValueError, match="missing.*concept_id"):
        validate(df)


def test_validate_accepts_missing_visit_occurrence_id(spark):
    """The condition_era loader path emits no visit_occurrence_id; absence
    is not an error (only wrong-type-when-present is)."""
    from charmpheno.omop.schema import validate

    df = spark.createDataFrame(
        [(1, 5, "diabetes")],
        schema="person_id INT, concept_id INT, concept_name STRING",
    )
    validate(df)  # no exception


def test_validate_rejects_wrong_type(spark):
    from charmpheno.omop.schema import validate

    df = spark.createDataFrame(
        [("not-an-int", 5, "diabetes")],
        schema="person_id STRING, concept_id INT, concept_name STRING",
    )
    with pytest.raises(ValueError, match=r"person_id.*wrong type.*StringType"):
        validate(df)


def test_validate_rejects_wrong_type_for_optional_column(spark):
    """Optional columns are still type-checked when present."""
    from charmpheno.omop.schema import validate

    df = spark.createDataFrame(
        [(1, "v100", 5, "diabetes")],
        schema="person_id INT, visit_occurrence_id STRING, concept_id INT, concept_name STRING",
    )
    with pytest.raises(ValueError, match=r"visit_occurrence_id.*wrong type"):
        validate(df)


def test_validate_accepts_long_int_columns(spark):
    """BigQuery/pandas-default int columns arrive as LongType — must validate."""
    from charmpheno.omop.schema import validate

    df = spark.createDataFrame(
        [(1, 5, "diabetes"), (2, 6, "asthma")],
        schema="person_id LONG, concept_id LONG, concept_name STRING",
    )
    validate(df)  # must not raise


def test_validate_allows_extra_columns_by_default(spark):
    """Common case: real loaders include date columns or similar. Extras
    are allowed; required columns must be present with right types."""
    from charmpheno.omop.schema import validate

    df = spark.createDataFrame(
        [(1, 5, "diabetes", "2024-01-01")],
        schema="person_id INT, concept_id INT, concept_name STRING, visit_date STRING",
    )
    validate(df)


def test_validate_accepts_era_columns(spark):
    """The condition_era loader path emits era_start_date and era_end_date
    as date columns; validator accepts them as the optional types they are."""
    import datetime as dt

    from charmpheno.omop.schema import validate

    df = spark.createDataFrame(
        [(1, 5, "diabetes", dt.date(2010, 1, 15), dt.date(2018, 7, 3))],
        schema=(
            "person_id INT, concept_id INT, concept_name STRING, "
            "condition_era_start_date DATE, condition_era_end_date DATE"
        ),
    )
    validate(df)
