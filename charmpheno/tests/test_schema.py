"""Canonical OMOP shape + validator."""
import pytest


def test_canonical_columns_are_exactly_four():
    from charmpheno.omop.schema import CANONICAL_COLUMNS

    assert CANONICAL_COLUMNS == (
        "person_id", "visit_occurrence_id", "concept_id", "concept_name",
    )


def test_validate_accepts_canonical_dataframe(spark):
    from charmpheno.omop.schema import validate

    df = spark.createDataFrame(
        [(1, 100, 5, "diabetes"), (2, 101, 6, "asthma")],
        schema="person_id INT, visit_occurrence_id INT, concept_id INT, concept_name STRING",
    )
    validate(df)  # no exception


def test_validate_rejects_missing_column(spark):
    from charmpheno.omop.schema import validate

    df = spark.createDataFrame(
        [(1, 5, "diabetes")],
        schema="person_id INT, concept_id INT, concept_name STRING",
    )
    with pytest.raises(ValueError, match="missing.*visit_occurrence_id"):
        validate(df)


def test_validate_rejects_wrong_type(spark):
    from charmpheno.omop.schema import validate

    df = spark.createDataFrame(
        [("not-an-int", 100, 5, "diabetes")],
        schema="person_id STRING, visit_occurrence_id INT, concept_id INT, concept_name STRING",
    )
    with pytest.raises(ValueError, match="person_id.*STRING|person_id.*type"):
        validate(df)


def test_validate_allows_extra_columns_by_default(spark):
    """Common case: real loaders may include a date column or similar. We
    don't reject extras; we just require the canonical four are present
    with right types."""
    from charmpheno.omop.schema import validate

    df = spark.createDataFrame(
        [(1, 100, 5, "diabetes", "2024-01-01")],
        schema="person_id INT, visit_occurrence_id INT, concept_id INT, concept_name STRING, visit_date STRING",
    )
    validate(df)
