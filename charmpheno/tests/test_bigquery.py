"""Unit + cluster-marked tests for charmpheno.omop.bigquery.

The unit tests cover argument validation and don't talk to BigQuery — they
run in the default loop. End-to-end behavior against a real CDR is covered
by the cluster-marked smoke (manual via `make test-cluster`).
"""
import os

import pytest

from charmpheno.omop.bigquery import load_omop_bigquery


def test_decode_sex_maps_standard_concepts_and_does_not_conflate_unknown(spark):
    """gender_concept_id -> sex must map 8507->M, 8532->F, and everything else
    (Unknown 8551, Other 8521, No-matching 0, null) to 'Unknown' — NOT silently
    to 'F'. Conflating unknowns with Female makes the sex covariate a constant
    when gender data is absent (the exp-0027 symptom: sex collapsed to F)."""
    from pyspark.sql import functions as F
    from charmpheno.omop.bigquery import decode_sex

    rows = [(8507,), (8532,), (8551,), (8521,), (0,), (None,)]
    df = spark.createDataFrame(rows, ["gender_concept_id"])
    got = {
        r["gender_concept_id"]: r["sex"]
        for r in df.withColumn("sex", decode_sex(F.col("gender_concept_id"))).collect()
    }
    assert got[8507] == "M"
    assert got[8532] == "F"
    assert got[8551] == "Unknown"   # OMOP Unknown, not Female
    assert got[8521] == "Unknown"   # OMOP Other, not Female
    assert got[0] == "Unknown"      # No matching concept
    assert got[None] == "Unknown"   # null gender, not Female


def test_decode_sex_from_name_is_vocabulary_agnostic(spark):
    """Decoding sex from the gender *concept name* must cover standard OMOP
    ('MALE'/'FEMALE') and dataset-specific vocabularies alike (AoU uses
    45878463 'Female' / 45880669 'Male'), and must NOT be fooled by AoU's
    aggregated PPI concept 2000000002 whose name contains 'man'/'woman' as
    substrings ('not man only, not woman only ...') — that maps to 'Unknown'.
    Exact-token matching, whitespace/case tolerant."""
    from pyspark.sql import functions as F
    from charmpheno.omop.bigquery import decode_sex_from_name

    rows = [
        ("FEMALE",), ("MALE",),          # standard OMOP 8532 / 8507 names
        ("Female",), ("Male",),          # AoU 45878463 / 45880669 names
        ("Woman",), ("Man",),            # PPI-style gender identity
        ("Not man only, not woman only, prefer not to answer",),  # AoU 2000000002
        ("No matching concept",),        # concept_id 0
        (None,),                         # null / no concept row
        (" female ",),                   # whitespace + case tolerance
    ]
    df = spark.createDataFrame(rows, ["gender_concept_name"])
    got = {
        (r["gender_concept_name"] or "<null>"): r["sex"]
        for r in df.withColumn(
            "sex", decode_sex_from_name(F.col("gender_concept_name"))
        ).collect()
    }
    assert got["FEMALE"] == "F"
    assert got["MALE"] == "M"
    assert got["Female"] == "F"
    assert got["Male"] == "M"
    assert got["Woman"] == "F"
    assert got["Man"] == "M"
    assert got["Not man only, not woman only, prefer not to answer"] == "Unknown"
    assert got["No matching concept"] == "Unknown"
    assert got["<null>"] == "Unknown"
    assert got[" female "] == "F"


def test_filter_known_sex_keeps_only_binary_sex(spark):
    """filter_known_sex drops rows whose decoded sex is not M/F (Unknown,
    Other, null), leaving only the binary-sex analysis population."""
    from charmpheno.omop.bigquery import filter_known_sex

    rows = [(1, "M"), (2, "F"), (3, "Unknown"), (4, "M"), (5, None), (6, "F")]
    df = spark.createDataFrame(rows, ["person_id", "sex"])
    kept = {r["person_id"] for r in filter_known_sex(df).collect()}
    assert kept == {1, 2, 4, 6}


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
            concept_types=("condition", "procedure"),
        )


def test_rejects_zero_or_negative_sample_mod(spark):
    with pytest.raises(ValueError, match="person_sample_mod"):
        load_omop_bigquery(
            spark=spark,
            cdr_dataset="proj.ds",
            billing_project="some-project",
            person_sample_mod=0,
        )


def test_rejects_unknown_cohort(spark):
    with pytest.raises(ValueError, match="cohort"):
        load_omop_bigquery(
            spark=spark,
            cdr_dataset="proj.ds",
            billing_project="some-project",
            cohort="not_a_real_cohort",
        )


def test_drug_concept_type_and_drug_era_source_are_supported():
    """drug/drug_era must pass validation (they raised NotImplementedError/ValueError
    before). We can't hit BigQuery in a unit test, so we assert the validation gate
    opens -- the read failure is a DIFFERENT, later error (no live spark.read)."""
    import pytest
    from charmpheno.omop import bigquery as bq
    assert "drug" in bq._SUPPORTED_CONCEPT_TYPES
    assert "drug_era" in bq._SUPPORTED_SOURCE_TABLES
    # A rejected concept type still raises NotImplementedError, unchanged:
    with pytest.raises(NotImplementedError, match="procedure"):
        bq.load_omop_bigquery(spark=object(), cdr_dataset="p.d", billing_project="b",
                              concept_types=("procedure",))


def test_drug_era_column_normalization_is_declared():
    """The drug_era branch must normalize to (person_id, concept_id, dates) -- the
    same event shape conditions use -- so the downstream window/doc-spec machinery
    is unchanged. We assert the branch's declared output columns via a small pure
    helper `_drug_era_select_cols` (extracted so it is testable without a read)."""
    from charmpheno.omop.bigquery import _drug_era_select_cols
    cols, extra = _drug_era_select_cols()
    # concept_id is the aliased drug_concept_id; dates carried through:
    assert "person_id" in cols and "concept_id" in cols
    assert extra == ("drug_era_start_date", "drug_era_end_date")


def test_rejects_cohort_with_non_condition_source_table():
    """cohort filtering hardcodes a condition-derived index date
    (condition_era_start_date for the era path); passing cohort= with
    source_table='drug_era' must fast-fail with a named ValueError here,
    not fall through to apply_cohort and crash later with a cryptic
    AnalysisException about a missing column. Uses a real SUPPORTED_COHORTS
    name so this exercises the NEW source_table+cohort guard, not the
    pre-existing unknown-cohort check."""
    from charmpheno.omop.bigquery import load_omop_bigquery
    from charmpheno.omop.cohorts import SUPPORTED_COHORTS

    assert "first_cancer_year" in SUPPORTED_COHORTS
    with pytest.raises(ValueError, match="condition source_table"):
        load_omop_bigquery(
            spark=object(),
            cdr_dataset="p.d",
            billing_project="b",
            concept_types=("drug",),
            source_table="drug_era",
            cohort="first_cancer_year",
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
