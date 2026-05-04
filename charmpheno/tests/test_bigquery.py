"""Unit + cluster-marked tests for charmpheno.omop.bigquery.

The unit tests cover argument validation and don't talk to BigQuery — they
run in the default loop. End-to-end behavior against a real CDR is covered
by the cluster-marked smoke (manual via `make test-cluster`).
"""
import os

import pytest

from charmpheno.omop.bigquery import load_omop_bigquery


def test_rejects_malformed_cdr_dataset(spark):
    with pytest.raises(ValueError, match="<project>.<dataset>"):
        load_omop_bigquery(
            spark=spark,
            cdr_dataset="not-fully-qualified",
            billing_project="some-project",
        )


def test_rejects_unsupported_concept_types(spark):
    with pytest.raises(NotImplementedError, match="not supported in v1"):
        load_omop_bigquery(
            spark=spark,
            cdr_dataset="proj.ds",
            billing_project="some-project",
            concept_types=("condition", "drug"),
        )


def test_rejects_zero_or_negative_sample_mod(spark):
    with pytest.raises(ValueError, match="person_sample_mod"):
        load_omop_bigquery(
            spark=spark,
            cdr_dataset="proj.ds",
            billing_project="some-project",
            person_sample_mod=0,
        )


@pytest.mark.cluster
def test_smoke_against_real_cdr(spark):
    """Reads a tiny slice from the workspace CDR; requires env + connector."""
    cdr = os.environ.get("WORKSPACE_CDR")
    billing = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not (cdr and billing):
        pytest.skip("WORKSPACE_CDR / GOOGLE_CLOUD_PROJECT not set")

    df = load_omop_bigquery(
        spark=spark,
        cdr_dataset=cdr,
        billing_project=billing,
        person_sample_mod=10000,  # aggressive sampling for a test
    ).limit(10)

    # Schema is the contract; rows may be sparse at extreme sampling.
    assert set(df.columns) >= {"person_id", "visit_occurrence_id",
                                "concept_id", "concept_name"}
    df.collect()
