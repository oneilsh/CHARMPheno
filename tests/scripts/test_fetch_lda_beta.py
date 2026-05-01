"""Tests for scripts/fetch_lda_beta.py."""
import numpy as np
import pandas as pd
import pytest


def test_parse_topic_metadata_extracts_rank_u_h_c():
    from fetch_lda_beta import parse_topic_metadata

    out = parse_topic_metadata("T-58 (U 0.5%, H 0.91, C -0.5)")
    assert out == {
        "topic_id": 58,
        "usage_pct": 0.5,
        "coherence_h": 0.91,
        "baseline_delta_c": -0.5,
    }


def test_parse_topic_metadata_handles_negative_h_and_extra_whitespace():
    from fetch_lda_beta import parse_topic_metadata

    out = parse_topic_metadata("T-1  ( U 1.23%,  H -0.07,  C 2.5 )")
    assert out["topic_id"] == 1
    assert out["usage_pct"] == 1.23
    assert out["coherence_h"] == -0.07
    assert out["baseline_delta_c"] == 2.5


def test_parse_topic_metadata_rejects_malformed_string():
    from fetch_lda_beta import parse_topic_metadata

    with pytest.raises(ValueError):
        parse_topic_metadata("T-1 (no parens here)")


def test_topic_metadata_from_names_dedupes_and_sorts():
    from fetch_lda_beta import topic_metadata_from_names

    names = pd.Series([
        "T-3 (U 0.1%, H 0.5, C 0.0)",
        "T-1 (U 0.9%, H 0.7, C 0.2)",
        "T-3 (U 0.1%, H 0.5, C 0.0)",   # duplicate
        "T-2 (U 0.4%, H 0.6, C 0.1)",
    ])
    out = topic_metadata_from_names(names)
    assert list(out["topic_id"]) == [1, 2, 3]
    np.testing.assert_allclose(out["usage_pct"], [0.9, 0.4, 0.1])
    np.testing.assert_allclose(out["coherence_h"], [0.7, 0.6, 0.5])
    np.testing.assert_allclose(out["baseline_delta_c"], [0.2, 0.1, 0.0])


def test_top_k_filter_keeps_top_weights_per_topic():
    """Per-topic, keep only the K highest-weight concepts and renormalize."""
    from fetch_lda_beta import top_k_per_topic_and_renormalize

    # Two topics, four concepts each, unbalanced weights.
    rows = pd.DataFrame({
        "topic_id": [0, 0, 0, 0, 1, 1, 1, 1],
        "concept_id": [10, 11, 12, 13, 20, 21, 22, 23],
        "concept_name": ["a", "b", "c", "d", "e", "f", "g", "h"],
        "term_weight": [0.5, 0.3, 0.15, 0.05, 0.1, 0.2, 0.3, 0.4],
    })

    out = top_k_per_topic_and_renormalize(rows, top_k=2)

    # Two topics survive; two rows per topic.
    assert set(out["topic_id"].unique()) == {0, 1}
    counts = out.groupby("topic_id").size()
    assert counts.loc[0] == 2 and counts.loc[1] == 2

    # Topic 0 keeps 10 (0.5) and 11 (0.3).
    topic0 = out[out["topic_id"] == 0].sort_values("term_weight", ascending=False)
    assert list(topic0["concept_id"]) == [10, 11]

    # Topic 1 keeps 23 (0.4) and 22 (0.3).
    topic1 = out[out["topic_id"] == 1].sort_values("term_weight", ascending=False)
    assert list(topic1["concept_id"]) == [23, 22]

    # Each topic's kept weights sum to 1.0 after renormalization.
    sums = out.groupby("topic_id")["term_weight"].sum()
    np.testing.assert_allclose(sums.values, [1.0, 1.0], atol=1e-9)


def test_top_k_filter_small_k_zero_rejects():
    """top_k=0 is nonsensical and should raise."""
    from fetch_lda_beta import top_k_per_topic_and_renormalize

    rows = pd.DataFrame({
        "topic_id": [0, 0],
        "concept_id": [1, 2],
        "concept_name": ["a", "b"],
        "term_weight": [0.5, 0.5],
    })
    with pytest.raises(ValueError):
        top_k_per_topic_and_renormalize(rows, top_k=0)


def test_top_k_filter_handles_fewer_rows_than_k():
    """If a topic has fewer rows than K, keep all of them."""
    from fetch_lda_beta import top_k_per_topic_and_renormalize

    rows = pd.DataFrame({
        "topic_id": [0, 0, 1],
        "concept_id": [10, 11, 20],
        "concept_name": ["a", "b", "c"],
        "term_weight": [0.3, 0.7, 1.0],
    })
    out = top_k_per_topic_and_renormalize(rows, top_k=10)
    # Topic 0 keeps both rows; topic 1 keeps its one row.
    assert len(out) == 3
    sums = out.groupby("topic_id")["term_weight"].sum().sort_index()
    np.testing.assert_allclose(sums.values, [1.0, 1.0], atol=1e-9)
