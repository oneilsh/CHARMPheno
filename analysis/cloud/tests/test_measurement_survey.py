"""Tests for the pure measurement-survey helpers (no Spark, no BQ).

Mirrors the anchor_selection.py split: everything Spark-free lives in
measurement_survey.py and is exercised here without the pyspark-importing
conftest (import path added inline so the test runs standalone).
"""
import sys
from pathlib import Path

_CLOUD = str(Path(__file__).resolve().parent.parent)
if _CLOUD not in sys.path:
    sys.path.insert(0, _CLOUD)


def test_apply_floor_drops_subfloor_and_counts_suppression():
    from measurement_survey import apply_floor

    rows = [
        {"id": 1, "n_persons": 100},
        {"id": 2, "n_persons": 49},   # below floor 50
        {"id": 3, "n_persons": 50},   # exactly floor -> kept
        {"id": 4, "n_persons": None},  # null -> treated as 0 -> dropped
    ]
    kept, n_sup, sup_total = apply_floor(rows, "n_persons", 50)
    assert [r["id"] for r in kept] == [1, 3]
    assert n_sup == 2
    assert sup_total == 49  # 49 + 0


def test_apply_floor_missing_key_is_suppressed():
    from measurement_survey import apply_floor

    kept, n_sup, _ = apply_floor([{"id": 1}], "n_persons", 20)
    assert kept == [] and n_sup == 1


def test_safe_div_guards_zero_and_none():
    from measurement_survey import safe_div

    assert safe_div(5, 10) == 0.5
    assert safe_div(5, 0) == 0.0
    assert safe_div(5, None) == 0.0


def test_derive_concept_summary_fractions_and_abn_mix():
    from measurement_survey import derive_concept_summary

    agg = {
        "concept_id": 3004501, "n_rows": 1000, "n_persons": 400,
        "n_val_number": 900, "n_val_concept": 100, "n_unit": 950,
        "n_range": 800, "n_operator": 50,
        "n_feasible": 800, "n_low": 80, "n_high": 160,
        "n_distinct_units": 2, "top_unit_n": 900,
    }
    s = derive_concept_summary(agg)
    assert s["pct_val_number"] == 0.9
    assert s["pct_val_concept"] == 0.1
    assert s["pct_range"] == 0.8
    assert s["pct_feasible"] == 0.8
    assert s["top_unit_share"] == 0.9
    # abnormality mix is over the FEASIBLE denominator (800): 80 low, 160 high,
    # 560 normal.
    assert s["frac_low"] == 80 / 800
    assert s["frac_high"] == 160 / 800
    assert abs(s["frac_normal"] - 560 / 800) < 1e-12
    assert abs(s["frac_low"] + s["frac_normal"] + s["frac_high"] - 1.0) < 1e-12


def test_derive_concept_summary_zero_rows_is_safe():
    from measurement_survey import derive_concept_summary

    s = derive_concept_summary({"n_rows": 0, "n_feasible": 0})
    assert s["pct_val_number"] == 0.0
    assert s["frac_low"] == 0.0  # no divide-by-zero


def test_classify_representation_cascade_order():
    from measurement_survey import classify_representation

    # range viable -> range-abnormality wins even if a coded value also present
    assert classify_representation(
        {"pct_feasible": 0.7, "pct_val_concept": 0.9, "pct_val_number": 0.9}
    ) == "range-abnormality"
    # no range, coded value viable -> value-concept
    assert classify_representation(
        {"pct_feasible": 0.1, "pct_val_concept": 0.8, "pct_val_number": 0.9}
    ) == "value-concept"
    # only numeric viable -> needs binning
    assert classify_representation(
        {"pct_feasible": 0.1, "pct_val_concept": 0.1, "pct_val_number": 0.8}
    ) == "numeric-needs-binning"
    # nothing viable -> presence-only
    assert classify_representation(
        {"pct_feasible": 0.1, "pct_val_concept": 0.1, "pct_val_number": 0.1}
    ) == "presence-only"


def test_summarize_representation_mix_weights_by_persons():
    from measurement_survey import summarize_representation_mix

    summaries = [
        {"pct_feasible": 0.9, "pct_val_concept": 0, "pct_val_number": 0.9,
         "n_persons": 300},
        {"pct_feasible": 0.9, "pct_val_concept": 0, "pct_val_number": 0.9,
         "n_persons": 200},
        {"pct_feasible": 0.0, "pct_val_concept": 0.0, "pct_val_number": 0.0,
         "n_persons": 10},
    ]
    mix = summarize_representation_mix(summaries, weight_key="n_persons")
    assert mix["range-abnormality"]["n_concepts"] == 2
    assert mix["range-abnormality"]["weight"] == 500
    assert mix["presence-only"]["n_concepts"] == 1
    assert mix["presence-only"]["weight"] == 10
