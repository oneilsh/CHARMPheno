"""Tests for charmpheno.omop.doc_spec.

Each spec is exercised on a small in-memory OMOP-like DataFrame; the
key assertions are (a) the doc_id column is shaped correctly, (b)
person_id is preserved on every output row (so downstream consumers
needing patient-level identity can still recover it from the BOW
DataFrame), and (c) the manifest round-trips through
DocSpec.from_manifest.
"""
from __future__ import annotations

import datetime as dt

import pytest

from charmpheno.omop.doc_spec import (
    DocSpec,
    PatientDocSpec,
    PatientYearDocSpec,
    doc_spec_from_cli,
)


def _events_with_eras(spark):
    """Three patients, with explicit start/end era dates."""
    return spark.createDataFrame(
        [
            # Person 1: HTN era spans 2010-2012 (3 years).
            (1, 100, "htn", dt.date(2010, 5, 1), dt.date(2012, 11, 30)),
            # Person 1: T2DM era 2015-2015 (1 year).
            (1, 200, "t2dm", dt.date(2015, 3, 10), dt.date(2015, 7, 20)),
            # Person 2: Allergy era 2018-2019 (2 years).
            (2, 300, "allergy", dt.date(2018, 1, 1), dt.date(2019, 6, 30)),
            # Person 3: Single-day era in 2020.
            (3, 400, "fracture", dt.date(2020, 4, 15), dt.date(2020, 4, 15)),
        ],
        schema=(
            "person_id INT, concept_id INT, concept_name STRING, "
            "condition_era_start_date DATE, condition_era_end_date DATE"
        ),
    )


def test_patient_doc_spec_one_doc_per_person(spark):
    df = _events_with_eras(spark)
    spec = PatientDocSpec()
    out = spec.derive_docs(df).collect()
    # PatientDocSpec doesn't replicate rows; same count in/out.
    assert len(out) == 4
    # doc_id = stringified person_id.
    by_doc = {}
    for row in out:
        by_doc.setdefault(row["doc_id"], []).append(row["person_id"])
    assert sorted(by_doc.keys()) == ["1", "2", "3"]
    # person_id preserved on every row.
    assert all(by_doc[k][0] == int(k) for k in by_doc)


def test_patient_doc_spec_manifest_roundtrip():
    spec = PatientDocSpec(min_doc_length=5)
    manifest = spec.manifest()
    assert manifest == {"name": "patient", "min_doc_length": 5}
    restored = DocSpec.from_manifest(manifest)
    assert isinstance(restored, PatientDocSpec)
    assert restored.min_doc_length == 5


def test_patient_year_replicates_across_era_span(spark):
    df = _events_with_eras(spark)
    spec = PatientYearDocSpec(min_doc_length=0)  # don't filter for the test
    out = spec.derive_docs(df).collect()
    # Person 1 HTN era: 3 rows (2010, 2011, 2012).
    # Person 1 T2DM era: 1 row (2015).
    # Person 2 allergy era: 2 rows (2018, 2019).
    # Person 3 fracture era: 1 row (2020).
    # Total: 7 rows.
    assert len(out) == 7
    # Check doc_id shape: "{person_id}:{year}".
    doc_ids = {row["doc_id"] for row in out}
    expected = {
        "1:2010", "1:2011", "1:2012",       # HTN era span
        "1:2015",                             # T2DM era
        "2:2018", "2:2019",                  # Allergy era span
        "3:2020",                             # Fracture
    }
    assert doc_ids == expected
    # person_id preserved on every row.
    for row in out:
        assert row["doc_id"].startswith(f"{row['person_id']}:")


def test_patient_year_without_replication_uses_start_year_only(spark):
    df = _events_with_eras(spark)
    spec = PatientYearDocSpec(replicate_eras=False, min_doc_length=0)
    out = spec.derive_docs(df).collect()
    # Without replication, each era contributes one row (its start year).
    assert len(out) == 4
    doc_ids = {row["doc_id"] for row in out}
    # All eras anchored to their start year.
    assert doc_ids == {"1:2010", "1:2015", "2:2018", "3:2020"}


def test_patient_year_manifest_roundtrip():
    spec = PatientYearDocSpec(
        min_doc_length=42, replicate_eras=False,
        date_start_col="custom_start", date_end_col="custom_end",
    )
    manifest = spec.manifest()
    assert manifest == {
        "name": "patient_year",
        "min_doc_length": 42,
        "replicate_eras": False,
        "date_start_col": "custom_start",
        "date_end_col": "custom_end",
    }
    restored = DocSpec.from_manifest(manifest)
    assert isinstance(restored, PatientYearDocSpec)
    assert restored.min_doc_length == 42
    assert restored.replicate_eras is False
    assert restored.date_start_col == "custom_start"
    assert restored.date_end_col == "custom_end"


def test_patient_year_errors_on_missing_columns(spark):
    """If the events frame lacks the era date columns, fail fast with a
    clear message rather than producing garbage doc_ids."""
    df = spark.createDataFrame(
        [(1, 100, "htn")],
        schema="person_id INT, concept_id INT, concept_name STRING",
    )
    spec = PatientYearDocSpec()
    with pytest.raises(ValueError, match="condition_era_start_date"):
        spec.derive_docs(df)


def test_from_manifest_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown DocSpec"):
        DocSpec.from_manifest({"name": "patient_decade"})


def test_doc_spec_from_cli_dispatches_to_registered_classes():
    spec_patient = doc_spec_from_cli("patient")
    assert isinstance(spec_patient, PatientDocSpec)
    assert spec_patient.min_doc_length == 0  # the class default

    spec_year = doc_spec_from_cli("patient_year")
    assert isinstance(spec_year, PatientYearDocSpec)
    assert spec_year.min_doc_length == 30  # the class default for year-binning

    # Override via CLI.
    spec_overridden = doc_spec_from_cli("patient", min_doc_length=10)
    assert spec_overridden.min_doc_length == 10


def test_doc_spec_from_cli_rejects_unknown_name():
    with pytest.raises(ValueError, match="--doc-unit"):
        doc_spec_from_cli("patient_decade")


def test_clips_future_end_dates(spark):
    """A pathological future-dated end_date shouldn't blow up the array."""
    far_future = dt.date(9999, 12, 31)
    df = spark.createDataFrame(
        [(1, 100, "htn", dt.date(2024, 1, 1), far_future)],
        schema=(
            "person_id INT, concept_id INT, concept_name STRING, "
            "condition_era_start_date DATE, condition_era_end_date DATE"
        ),
    )
    spec = PatientYearDocSpec(min_doc_length=0)
    out = spec.derive_docs(df).collect()
    # End date is clipped to current year; we don't get 8000 rows.
    current_year = dt.date.today().year
    expected_count = current_year - 2024 + 1
    assert len(out) == expected_count, (
        f"future-dated end should clip to current year, got {len(out)} rows "
        f"(expected {expected_count})"
    )


def test_patient_year_null_end_date_emits_single_year(spark):
    """NULL end_date with a valid start_date should coalesce to a single-
    year span at the start year — not silently expand into a multi-year
    span via F.least's null-skipping semantics (the pre-fix behavior).
    """
    df = spark.createDataFrame(
        [
            (1, 100, "htn", dt.date(2015, 3, 1), None),
            (1, 200, "t2dm", dt.date(2017, 1, 1), dt.date(2017, 12, 31)),
        ],
        schema=(
            "person_id INT, concept_id INT, concept_name STRING, "
            "condition_era_start_date DATE, condition_era_end_date DATE"
        ),
    )
    spec = PatientYearDocSpec(min_doc_length=0)
    out = spec.derive_docs(df).collect()
    # HTN era with NULL end → single year at 2015 (not 2015..today).
    # T2DM era with full year span → also single year at 2017.
    doc_ids = sorted(row["doc_id"] for row in out)
    assert doc_ids == ["1:2015", "1:2017"]


def test_patient_year_null_start_date_drops_row(spark):
    """A row with NULL start_date has no anchor year and should be dropped
    explicitly (rather than falling out of F.explode silently)."""
    df = spark.createDataFrame(
        [
            (1, 100, "htn", dt.date(2015, 3, 1), dt.date(2015, 12, 31)),
            (1, 200, "t2dm", None, dt.date(2017, 12, 31)),
            (1, 300, "mystery", None, None),
        ],
        schema=(
            "person_id INT, concept_id INT, concept_name STRING, "
            "condition_era_start_date DATE, condition_era_end_date DATE"
        ),
    )
    spec = PatientYearDocSpec(min_doc_length=0)
    out = spec.derive_docs(df).collect()
    # Only the HTN row survives; both NULL-start rows are dropped.
    assert len(out) == 1
    assert out[0]["doc_id"] == "1:2015"


def test_patient_year_null_start_date_drops_row_no_replication(spark):
    """The NULL-start-date drop applies equally under replicate_eras=False
    — otherwise the row would survive with a NULL year_active and produce
    doc_id='person_id:null'."""
    df = spark.createDataFrame(
        [
            (1, 100, "htn", dt.date(2015, 3, 1), dt.date(2015, 12, 31)),
            (1, 200, "t2dm", None, dt.date(2017, 12, 31)),
        ],
        schema=(
            "person_id INT, concept_id INT, concept_name STRING, "
            "condition_era_start_date DATE, condition_era_end_date DATE"
        ),
    )
    spec = PatientYearDocSpec(min_doc_length=0, replicate_eras=False)
    out = spec.derive_docs(df).collect()
    assert len(out) == 1
    assert out[0]["doc_id"] == "1:2015"
