# spark-vi/tests/test_topic_diagnostics.py
"""Tests for the shared topic-word per-iteration numerics helper."""
from __future__ import annotations

import numpy as np

from spark_vi.models.topic.diagnostics import topic_word_summary


def test_row_sums_peak_mass_fraction():
    lam = np.array([[1.0, 3.0, 0.0, 0.0],   # row sum 4, peak 3/4
                    [2.0, 2.0, 2.0, 2.0]])   # row sum 8, peak 2/8
    s = topic_word_summary(lam, top_n=2)
    np.testing.assert_allclose(s["row_sums"], [4.0, 8.0])
    np.testing.assert_allclose(s["peak"], [0.75, 0.25])
    np.testing.assert_allclose(s["mass_fraction"], [4.0 / 12.0, 8.0 / 12.0])


def test_top_indices_and_probs_match_manual_argsort():
    rng = np.random.default_rng(0)
    lam = rng.gamma(2.0, 1.0, size=(3, 6))
    top_n = 3
    s = topic_word_summary(lam, top_n=top_n)
    topics = lam / lam.sum(axis=1, keepdims=True)
    for k in range(3):
        want = topics[k].argsort()[::-1][:top_n]
        np.testing.assert_array_equal(s["top_indices"][k], want)
        np.testing.assert_allclose(s["top_probs"][k], topics[k][want])
    assert s["top_indices"].shape == (3, top_n)
    assert s["top_probs"].shape == (3, top_n)


def test_top_n_larger_than_vocab_is_clamped():
    lam = np.array([[1.0, 2.0]])
    s = topic_word_summary(lam, top_n=10)
    assert s["top_indices"].shape == (1, 2)   # clamped to V


def test_zero_row_is_safe():
    # A topic with zero mass must not divide-by-zero (1e-12 guard).
    lam = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 2.0]])
    s = topic_word_summary(lam, top_n=2)
    assert np.all(np.isfinite(s["peak"]))
    assert np.all(np.isfinite(s["top_probs"]))
