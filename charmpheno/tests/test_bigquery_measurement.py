"""Tests for the value-aware measurement domain in load_omop_bigquery.

Ported from the hybrid-domain branch and adapted to THIS branch's design, where
measurement is a ``concept_type`` (load_omop_bigquery(concept_types=("measurement",)))
rather than a ``source_table``. The cascade behavior test drives ``_load_measurement``
through a stub ``_read`` (the reason it takes ``_read`` as a parameter) so it
exercises the real Spark tokenization against in-memory `measurement` / `concept`
frames without a BigQuery read.
"""
import datetime as dt

import pytest


def test_measurement_select_cols_declares_value_cascade_inputs():
    from charmpheno.omop.bigquery import _measurement_select_cols
    raw, extra = _measurement_select_cols()
    assert raw == ("person_id", "measurement_concept_id", "value_as_number",
                   "value_as_concept_id", "range_low", "range_high",
                   "measurement_date")
    assert extra == ("measurement_date",)


def test_measurement_is_a_supported_concept_type():
    from charmpheno.omop import bigquery as bq
    assert "measurement" in bq._SUPPORTED_CONCEPT_TYPES


def test_measurement_rejects_cohort_filtering():
    # cohort filtering needs a condition source_table (index date is
    # condition-derived); measurement is a feature-only domain. Fast-fails before
    # any read, so spark=None is fine.
    from charmpheno.omop.bigquery import load_omop_bigquery
    with pytest.raises(ValueError, match="cohort filtering is not supported for measurement"):
        load_omop_bigquery(spark=None, cdr_dataset="p.d", billing_project="b",
                           concept_types=("measurement",), cohort="population_glp1")


def test_measurement_cannot_be_fused_with_other_domains():
    # measurement's synthetic tokens would be null-joined away by the generic
    # fused concept-name join, so it must be loaded single-domain. Fast-fails.
    from charmpheno.omop.bigquery import load_omop_bigquery
    with pytest.raises(ValueError, match="cannot be fused"):
        load_omop_bigquery(spark=None, cdr_dataset="p.d", billing_project="b",
                           concept_types=("condition", "measurement"))


def _measurement_frame(spark):
    from pyspark.sql.types import (StructType, StructField, LongType,
                                   DoubleType, DateType)
    d = dt.date(2020, 1, 1)
    schema = StructType([
        StructField("person_id", LongType()),
        StructField("measurement_concept_id", LongType()),
        StructField("value_as_number", DoubleType()),
        StructField("value_as_concept_id", LongType()),
        StructField("range_low", DoubleType()),
        StructField("range_high", DoubleType()),
        StructField("measurement_date", DateType()),
    ])
    rows = [
        (1, 3016723, 6.8, None, 3.5, 5.0, d),   # range -> high
        (2, 3016723, 4.0, None, 3.5, 5.0, d),   # range -> normal
        (3, 3016723, 2.0, None, 3.5, 5.0, d),   # range -> low
        (4, 5000, None, 9191, None, None, d),   # coded -> pos (Positive)
        (5, 6000, None, 7777, None, None, d),   # coded junk (Yellow) -> presence
        (6, 6000, 12.0, None, None, None, d),   # numeric, no range/code -> presence
        (7, 0, 1.0, None, 0.5, 2.0, d),         # concept_id 0 -> dropped
    ]
    return spark.createDataFrame(rows, schema)


def _concept_frame(spark):
    from pyspark.sql.types import StructType, StructField, LongType, StringType
    schema = StructType([
        StructField("concept_id", LongType()),
        StructField("concept_name", StringType()),
    ])
    rows = [
        (3016723, "Creatinine [Mass/volume] in Serum or Plasma"),
        (5000, "HIV 1 Ab [Presence] in Serum"),
        (6000, "Urine color"),
        (9191, "Positive"),
        (7777, "Yellow"),
    ]
    return spark.createDataFrame(rows, schema)


def test_load_measurement_cascade_end_to_end(spark):
    from charmpheno.omop.bigquery import _load_measurement, _FUSED_EVENT_DATE
    from charmpheno.omop import measurement_tokens as mt
    from charmpheno.omop.schema import validate

    frames = {"measurement": _measurement_frame(spark),
              "concept": _concept_frame(spark)}
    out = _load_measurement(lambda t: frames[t], person_sample_mod=None)
    validate(out)  # canonical shape
    assert _FUSED_EVENT_DATE in out.columns  # fused schema (event_date, not measurement_date)

    got = {r["person_id"]: (int(r["concept_id"]), r["concept_name"])
           for r in out.collect()}
    assert 7 not in got  # concept_id 0 dropped
    assert len(got) == 6

    # range branch: high / normal / low, all off the SAME real concept
    assert mt.decode_token(got[1][0]) == (3016723, mt.STATE_RANGE_HIGH)
    assert got[1][1].endswith("[high]")
    assert mt.decode_token(got[2][0]) == (3016723, mt.STATE_RANGE_NORMAL)
    assert mt.decode_token(got[3][0]) == (3016723, mt.STATE_RANGE_LOW)
    # coded branch: allowlisted "Positive" -> pos
    assert mt.decode_token(got[4][0]) == (5000, mt.STATE_CODED_POS)
    assert got[4][1].endswith("[pos]")
    # junk coded value ("Yellow") -> presence; numeric-without-range -> presence
    assert mt.decode_token(got[5][0]) == (6000, mt.STATE_PRESENCE)
    assert mt.decode_token(got[6][0]) == (6000, mt.STATE_PRESENCE)
    assert got[6][1].endswith("[measured]")


def test_load_measurement_person_sampling(spark):
    from charmpheno.omop.bigquery import _load_measurement
    frames = {"measurement": _measurement_frame(spark),
              "concept": _concept_frame(spark)}
    # keep only person_id % 2 == 0 -> persons 2,4,6 (person 7 dropped for concept 0)
    out = _load_measurement(lambda t: frames[t], person_sample_mod=2)
    assert {r["person_id"] for r in out.collect()} == {2, 4, 6}
