"""Local-Spark tests for the PC antidepressant driver's pre-index windowing and
the BOW -> dense-X bridge, on synthetic DataFrames (no BigQuery).

Uses the shared ``spark`` fixture (local[2], conftest.py). The pre-index
windowing reuses the committed ``charmpheno.omop.cohorts.lookback_feature_label_events``
(same helper the driver calls); the bridge test drives the driver's
``collect_bow_aligned`` on a real ``to_bow_dataframe`` output so the dense matrix
is rebuilt from genuine ``features`` SparseVector rows.
"""
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pytest

_CLOUD = str(Path(__file__).resolve().parents[1])
if _CLOUD not in sys.path:
    sys.path.insert(0, _CLOUD)
# Make the sibling charmpheno package importable (as its own venv would), the
# same way test_lda_driver_cohort_docunit.py does.
_CHARMPHENO_PKG_ROOT = str(Path(__file__).resolve().parents[3] / "charmpheno")
if _CHARMPHENO_PKG_ROOT not in sys.path:
    sys.path.insert(0, _CHARMPHENO_PKG_ROOT)

import pc_antidepressant_cloud as drv  # noqa: E402


def _d(s):
    return dt.date.fromisoformat(s)


def test_pre_index_windowing_restricts_to_lookback(spark):
    """Feature events land in [index - lookback_days, index); the index-day event
    and later events do NOT (they are the forward/label side)."""
    from charmpheno.omop.cohorts import lookback_feature_label_events

    # p1 index 2020-06-01; lookback 365 => feature window [2019-06-02, 2020-06-01).
    events = spark.createDataFrame(
        [
            ("p1", 11, _d("2019-01-01")),   # before lookback start -> excluded
            ("p1", 12, _d("2019-07-01")),   # inside feature window -> kept
            ("p1", 13, _d("2020-05-31")),   # inside feature window (< index) -> kept
            ("p1", 14, _d("2020-06-01")),   # == index -> forward side, NOT a feature
            ("p1", 15, _d("2020-08-01")),   # after index -> forward side
        ],
        ["person_id", "concept_id", "event_date"],
    )
    index_df = spark.createDataFrame(
        [("p1", _d("2020-06-01"), "mdd_antidepressant")],
        ["person_id", "index_date", "source_cohort"],
    )

    feature, label = lookback_feature_label_events(
        events, index_df, date_col="event_date",
        lookback_days=365, label_window_days=365,
    )
    feat_concepts = {r["concept_id"] for r in feature.collect()}
    label_concepts = {r["concept_id"] for r in label.collect()}

    assert feat_concepts == {12, 13}
    assert label_concepts == {14, 15}
    # index_date now rides through (exp 0111 R5.2 passthrough); source_cohort too.
    # This driver reads only the feature concepts, so the extra column is inert here.
    assert "index_date" in feature.columns
    assert "source_cohort" in feature.columns
    assert {r["index_date"] for r in feature.collect()} == {_d("2020-06-01")}


def test_pre_index_windowing_is_per_person(spark):
    """Each person's window is anchored on their own index_date."""
    from charmpheno.omop.cohorts import lookback_feature_label_events

    events = spark.createDataFrame(
        [
            ("p1", 1, _d("2020-05-01")),    # p1 feature (index 2020-06-01)
            ("p2", 2, _d("2020-05-01")),    # p2: before its lookback -> excluded
            ("p2", 3, _d("2021-02-01")),    # p2 feature (index 2021-03-01)
        ],
        ["person_id", "concept_id", "event_date"],
    )
    index_df = spark.createDataFrame(
        [
            ("p1", _d("2020-06-01"), "mdd_antidepressant"),
            ("p2", _d("2021-03-01"), "mdd_antidepressant"),
        ],
        ["person_id", "index_date", "source_cohort"],
    )
    feature, _ = lookback_feature_label_events(
        events, index_df, date_col="event_date",
        lookback_days=90, label_window_days=90,
    )
    got = {(r["person_id"], r["concept_id"]) for r in feature.collect()}
    # p1: 2020-05-01 in [2020-03-03, 2020-06-01) -> kept. p2: 2020-05-01 is far
    # before [2020-12-01, 2021-03-01) -> dropped; 2021-02-01 -> kept.
    assert got == {("p1", 1), ("p2", 3)}


def test_collect_bow_aligned_matches_real_to_bow_output(spark):
    """The driver's bridge rebuilds a dense X whose per-person rows equal the
    real to_bow_dataframe counts, aligned to the collected person order."""
    from charmpheno.omop import to_bow_dataframe
    from charmpheno.omop.doc_spec import PatientDocSpec

    # p1: 100 x2, 200 x1 ; p2: 200, 300 ; p3: 100
    events = spark.createDataFrame(
        [
            ("p1", 100), ("p1", 100), ("p1", 200),
            ("p2", 200), ("p2", 300),
            ("p3", 100),
        ],
        ["person_id", "concept_id"],
    )
    bow_df, vocab_map = to_bow_dataframe(
        events, doc_spec=PatientDocSpec(), token_col="concept_id",
        vocab_size=None, min_df=1, min_patient_count=1,
    )
    V = len(vocab_map)
    X, person_order = drv.collect_bow_aligned(bow_df, V)

    assert X.shape == (3, V)
    assert set(person_order) == {"p1", "p2", "p3"}
    row = {pid: X[i] for i, pid in enumerate(person_order)}
    # Columns are vocab_map[concept_id]; assert via the map, order-agnostic.
    def count(pid, cid):
        return row[pid][vocab_map[cid]]
    assert count("p1", 100) == 2 and count("p1", 200) == 1
    assert count("p2", 200) == 1 and count("p2", 300) == 1
    assert count("p3", 100) == 1
    # Absent concepts are zero.
    assert count("p2", 100) == 0 and count("p3", 200) == 0


def test_bridge_then_labels_are_row_aligned(spark):
    """End-to-end (minus BQ): the same person_order threads the bridge and the
    label assembly, so X row d and (y, mask) row d describe the same patient."""
    from charmpheno.omop import to_bow_dataframe
    from charmpheno.omop.doc_spec import PatientDocSpec

    events = spark.createDataFrame(
        [("p1", 100), ("p2", 200), ("p3", 300)],
        ["person_id", "concept_id"],
    )
    bow_df, vocab_map = to_bow_dataframe(
        events, doc_spec=PatientDocSpec(), token_col="concept_id",
        vocab_size=None, min_df=1, min_patient_count=1,
    )
    X, person_order = drv.collect_bow_aligned(bow_df, len(vocab_map))

    outcome_by_person = {
        "p1": ("sertraline", True),
        "p2": ("bupropion", False),
        "p3": ("sertraline", True),
    }
    drug_order = drv.stable_drug_order(
        [outcome_by_person[p][0] for p in person_order],
        reference=("sertraline", "bupropion"),
    )
    y, mask = drv.assemble_multitask_labels(outcome_by_person, person_order, drug_order)

    assert X.shape[0] == y.shape[0] == mask.shape[0] == 3
    # Exactly one observed cell per row.
    np.testing.assert_array_equal(mask.sum(axis=1), [1, 1, 1])
    # Spot-check one patient by its row index in person_order.
    for i, pid in enumerate(person_order):
        drug, worked = outcome_by_person[pid]
        col = drug_order.index(drug)
        assert mask[i, col] == 1
        assert y[i, col] == (1.0 if worked else 0.0)
