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


def test_supported_cohorts_includes_first_dementia_year():
    assert "first_dementia_year" in SUPPORTED_COHORTS


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


def test_supported_cohorts_includes_cancer_or_dementia():
    from charmpheno.omop.cohorts import SUPPORTED_COHORTS
    assert "cancer_or_dementia" in SUPPORTED_COHORTS


def test_cohort_metadata_has_cancer_or_dementia():
    from charmpheno.omop.cohorts import COHORT_METADATA
    assert "cancer_or_dementia" in COHORT_METADATA


def test_combine_cohorts_tags_and_unions_keeping_comorbid(spark):
    from charmpheno.omop.cohorts import _combine_cohorts
    cancer = spark.createDataFrame([(1, 10), (2, 20)], ["person_id", "concept_id"])
    dementia = spark.createDataFrame([(2, 30), (3, 40)], ["person_id", "concept_id"])
    out = _combine_cohorts(cancer, dementia)
    rows = {(r["person_id"], r["source_cohort"]) for r in out.collect()}
    # person 2 is comorbid -> appears under BOTH labels (no dedup).
    assert rows == {(1, "cancer"), (2, "cancer"), (2, "dementia"), (3, "dementia")}
    assert out.count() == 4
