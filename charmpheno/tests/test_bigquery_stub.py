"""BigQuery loader is a stub during bootstrap; ensure it fails loudly."""
import pytest


def test_load_omop_bigquery_raises_not_implemented(spark):
    from charmpheno.omop.bigquery import load_omop_bigquery

    with pytest.raises(NotImplementedError, match="follow-on spec"):
        load_omop_bigquery(spark=spark, cdr_dataset="any.dataset")


def test_load_omop_bigquery_has_expected_signature():
    import inspect

    from charmpheno.omop.bigquery import load_omop_bigquery

    sig = inspect.signature(load_omop_bigquery)
    expected = {"spark", "cdr_dataset", "concept_types", "limit"}
    assert expected.issubset(sig.parameters.keys())
