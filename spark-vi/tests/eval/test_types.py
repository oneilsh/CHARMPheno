"""Tests for spark_vi.eval.topic.types."""
from __future__ import annotations

import numpy as np
import pytest


def test_coherence_report_constructs_and_is_frozen():
    from spark_vi.eval.topic.types import CoherenceReport

    report = CoherenceReport(
        per_topic_npmi=np.array([0.1, 0.2, 0.3]),
        top_term_indices=np.array([[0, 1], [1, 2], [2, 3]]),
        topic_indices=np.array([0, 1, 2]),
        n_holdout_docs=100,
        top_n=2,
        mean=0.2,
        median=0.2,
        stdev=float(np.std([0.1, 0.2, 0.3], ddof=0)),
        min=0.1,
        max=0.3,
    )
    assert report.per_topic_npmi.shape == (3,)
    assert report.top_term_indices.shape == (3, 2)
    assert report.topic_indices.shape == (3,)
    assert report.n_holdout_docs == 100
    assert report.top_n == 2
    assert report.mean == pytest.approx(0.2)
    assert report.min == pytest.approx(0.1)
    assert report.max == pytest.approx(0.3)


def test_coherence_report_is_immutable():
    from spark_vi.eval.topic.types import CoherenceReport

    report = CoherenceReport(
        per_topic_npmi=np.array([0.1]),
        top_term_indices=np.array([[0]]),
        topic_indices=np.array([0]),
        n_holdout_docs=1,
        top_n=1,
        mean=0.1, median=0.1, stdev=0.0, min=0.1, max=0.1,
    )
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        report.mean = 0.5  # type: ignore[misc]
