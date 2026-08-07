"""Unit + cluster-marked tests for charmpheno.omop.bigquery.

The unit tests cover argument validation and don't talk to BigQuery — they
run in the default loop. End-to-end behavior against a real CDR is covered
by the cluster-marked smoke (manual via `make test-cluster`).
"""
import datetime as dt
import os

import pytest

from charmpheno.omop.bigquery import load_omop_bigquery


class _FakeReader:
    """Stand-in for ``spark.read`` that serves a synthetic DataFrame per OMOP
    table, dispatching on the ``table`` option's final path component.

    Lets the fused-loader tests exercise the real union/concept-join code in
    ``load_omop_bigquery`` — via its internal ``_read(table)`` seam — without a
    BigQuery round-trip. The connector chain
    ``.format(...).option("table", "<proj>.<ds>.<name>").option(...).load()``
    is mirrored: every ``option`` is a no-op except ``table``, whose bare
    ``<name>`` selects the frame ``load()`` returns.
    """

    def __init__(self, tables):
        self._tables = tables
        self._table = None

    def format(self, _fmt):
        return self

    def option(self, key, value):
        if key == "table":
            self._table = value.split(".")[-1]
        return self

    def load(self):
        return self._tables[self._table]


def _patch_bq(monkeypatch, spark, tables):
    """Route ``spark.read`` to a `_FakeReader` over ``tables`` for one test.

    Patches the ``SparkSession.read`` property at the class level so the
    loader's ``spark.read.format("bigquery")...`` chain resolves to synthetic
    frames; monkeypatch restores the real property at teardown, so the
    session-scoped Spark fixture is unaffected for other tests.
    """
    monkeypatch.setattr(
        type(spark), "read",
        property(lambda self: _FakeReader(tables)),
        raising=True,
    )


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
    # "measurement" is deliberately NOT in the fused loader's supported set
    # (condition/drug/procedure); an unknown domain must still raise.
    with pytest.raises(NotImplementedError, match="not supported in v1"):
        load_omop_bigquery(
            spark=spark,
            cdr_dataset="proj.ds",
            billing_project="some-project",
            concept_types=("condition", "measurement"),
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


def _synthetic_domain_tables(spark):
    """Per-table synthetic OMOP frames for the fused-loader tests.

    condition_occurrence carries a concept_id==0 row (OMOP "no matching
    concept") that the loader must drop, and repeats concept 100 so counts are
    non-trivial. drug_era / procedure_occurrence use disjoint concept_ids
    (globally unique across OMOP domains), so the fused union is a plain
    concept-id union.
    """
    condition_occurrence = spark.createDataFrame(
        [
            (1, 900, 100, dt.date(2020, 1, 1)),
            (1, 901, 100, dt.date(2020, 2, 1)),   # repeat of concept 100
            (2, 902, 100, dt.date(2021, 1, 1)),
            (1, 903, 0, dt.date(2020, 3, 1)),     # concept 0 -> dropped
        ],
        ["person_id", "visit_occurrence_id", "condition_concept_id",
         "condition_start_date"],
    )
    drug_era = spark.createDataFrame(
        [
            (1, 200, dt.date(2020, 1, 5)),
            (2, 201, dt.date(2021, 6, 1)),
        ],
        ["person_id", "drug_concept_id", "drug_era_start_date"],
    )
    procedure_occurrence = spark.createDataFrame(
        [
            (1, 300, dt.date(2020, 4, 1)),
            (2, 300, dt.date(2021, 7, 1)),
        ],
        ["person_id", "procedure_concept_id", "procedure_date"],
    )
    concept = spark.createDataFrame(
        [
            (100, "fever"),
            (200, "aspirin"),
            (201, "ibuprofen"),
            (300, "appendectomy"),
        ],
        ["concept_id", "concept_name"],
    )
    return {
        "condition_occurrence": condition_occurrence,
        "drug_era": drug_era,
        "procedure_occurrence": procedure_occurrence,
        "concept": concept,
    }


def test_fused_multidomain_unions_all_domains_into_one_stream(spark, monkeypatch):
    """concept_types=(condition, drug, procedure) reads each domain's fact
    table and UNIONs them into one flat concept_id stream (person_id,
    concept_id, concept_name, event_date) — the single fused vocabulary a
    Hughes-style bag-of-concepts needs. The concept 0 row is dropped."""
    _patch_bq(monkeypatch, spark, _synthetic_domain_tables(spark))

    omop = load_omop_bigquery(
        spark=spark,
        cdr_dataset="proj.ds",
        billing_project="some-project",
        concept_types=("condition", "drug", "procedure"),
    )

    assert set(omop.columns) == {
        "person_id", "concept_id", "concept_name", "event_date",
    }
    rows = omop.collect()
    # 3 condition rows (concept 0 dropped) + 2 drug + 2 procedure = 7.
    assert len(rows) == 7
    concept_ids = sorted(r["concept_id"] for r in rows)
    assert concept_ids == [100, 100, 100, 200, 201, 300, 300]
    # All three domains fused into one stream, names resolved from `concept`.
    names = {r["concept_id"]: r["concept_name"] for r in rows}
    assert names == {100: "fever", 200: "aspirin", 201: "ibuprofen",
                     300: "appendectomy"}
    # concept 0 ("no matching concept") never reaches the fused stream.
    assert 0 not in {r["concept_id"] for r in rows}


def test_fused_two_domains_union_counts(spark, monkeypatch):
    """A two-domain fuse (condition + drug) unions exactly those domains'
    rows — procedure is not pulled in when not requested."""
    _patch_bq(monkeypatch, spark, _synthetic_domain_tables(spark))

    omop = load_omop_bigquery(
        spark=spark,
        cdr_dataset="proj.ds",
        billing_project="some-project",
        concept_types=("condition", "drug"),
    )
    concept_ids = sorted(r["concept_id"] for r in omop.collect())
    # 3 condition (concept 0 dropped) + 2 drug; no procedure concept 300.
    assert concept_ids == [100, 100, 100, 200, 201]


def test_condition_only_default_keeps_legacy_schema(spark, monkeypatch):
    """concept_types=("condition",) is unchanged: it keeps the domain-specific
    date column and visit_occurrence_id, and does NOT reshape to the fused
    event_date schema."""
    _patch_bq(monkeypatch, spark, _synthetic_domain_tables(spark))

    omop = load_omop_bigquery(
        spark=spark,
        cdr_dataset="proj.ds",
        billing_project="some-project",
        concept_types=("condition",),
    )
    cols = set(omop.columns)
    assert "condition_start_date" in cols
    assert "visit_occurrence_id" in cols
    assert "event_date" not in cols
    # Same concept-0 drop and name join as before.
    concept_ids = sorted(r["concept_id"] for r in omop.collect())
    assert concept_ids == [100, 100, 100]


def test_fused_rejects_cohort_combo(spark):
    """Cohort windowing is condition-index-derived; combining it with a fused
    multi-domain load raises rather than silently ignoring the cohort."""
    with pytest.raises(ValueError, match="single-domain condition load"):
        load_omop_bigquery(
            spark=spark,
            cdr_dataset="proj.ds",
            billing_project="some-project",
            concept_types=("condition", "drug"),
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
