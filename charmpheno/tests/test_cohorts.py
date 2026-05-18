"""Unit tests for charmpheno.omop.cohorts.

Data-path tests against a real CDR are deferred to the cluster smoke;
this file covers the validation surface only.
"""
import pytest

from charmpheno.omop.cohorts import (
    SUPPORTED_COHORTS,
    apply_cohort,
)


def test_supported_cohorts_includes_first_cancer_year():
    assert "first_cancer_year" in SUPPORTED_COHORTS


def test_supported_cohorts_includes_first_pregnancy_year():
    assert "first_pregnancy_year" in SUPPORTED_COHORTS


def test_apply_cohort_rejects_unknown_name():
    with pytest.raises(ValueError, match="not supported"):
        apply_cohort(
            cond_df=None,           # never reached: validation fires first
            cohort="not_a_cohort",
            spark=None,
            cdr_dataset="proj.ds",
            billing_project="bp",
            date_col="condition_start_date",
        )
