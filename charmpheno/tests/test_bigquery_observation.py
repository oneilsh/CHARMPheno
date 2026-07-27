def test_observation_select_cols_declares_point_event_shape():
    from charmpheno.omop.bigquery import _observation_select_cols
    cols, extra = _observation_select_cols()
    assert cols == ("person_id", "concept_id", "observation_date")
    assert extra == ("observation_date",)   # point event: a single date, no end/span


def test_observation_is_a_supported_source_table_and_concept_type():
    from charmpheno.omop import bigquery as bq
    assert "observation" in bq._SUPPORTED_SOURCE_TABLES
    assert "observation" in bq._SUPPORTED_CONCEPT_TYPES


def test_observation_rejects_cohort_filtering():
    # The existing fast-fail: cohort filtering needs a condition source_table (the
    # index date is condition-derived). observation is a feature-only domain.
    import pytest
    from charmpheno.omop.bigquery import load_omop_bigquery
    with pytest.raises(ValueError, match="cohort filtering requires a condition"):
        load_omop_bigquery(spark=None, cdr_dataset="p.d", billing_project="b",
                           source_table="observation", cohort="population_glp1")
