"""Unit tests for stm_sigma_diagnostic (per-topic eta-variance / spectrum report)
and lda_concentration_readout (per-document topic-concentration summary for an
in-memory theta matrix).

Imports analysis._eval_common via the repo root (the package seam the cloud
eval driver imports from). Pure-numpy helper, no Spark.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

from analysis._eval_common import lda_concentration_readout, stm_sigma_diagnostic


def test_identifies_runaway_topic_and_block():
    # Topic 1 has by far the largest variance -> the "runaway".
    Sigma = np.diag([2.0, 500.0, 3.0]).astype(float)
    labels = {0: None, 1: "cancer", 2: "dementia"}  # None -> background

    report = stm_sigma_diagnostic(Sigma, labels=labels, top_k=3)

    assert report is not None
    # Names the max-variance topic as the runaway, with its block and value.
    assert "runaway = topic 1 [cancer]" in report
    assert "5.000e+02" in report
    # Background label rendered for the None entry when it appears in the ranking.
    assert "[background]" in report


def test_reports_eigen_spectrum_min_max_only():
    # Known eigen-spectrum; the full-matrix condition number and max
    # off-diagonal correlation are a reporting artifact (their cross-block
    # entries never enter the fit) and are no longer surfaced here.
    Sigma = np.array([
        [4.0, 0.0, 3.0],
        [0.0, 1.0, 0.0],
        [3.0, 0.0, 4.0],
    ])
    report = stm_sigma_diagnostic(Sigma, labels=None, top_k=3)

    # eig of [[4,3],[3,4]] are 7 and 1, plus the isolated 1 -> min=1, max=7
    assert "eig[min=1" in report
    assert "max=7" in report
    assert "cond=" not in report
    assert "offdiag" not in report


def test_returns_none_for_non_square_or_1d():
    assert stm_sigma_diagnostic(np.ones(5), labels=None) is None
    assert stm_sigma_diagnostic(np.ones((3, 4)), labels=None) is None
    assert stm_sigma_diagnostic(None, labels=None) is None


def test_lda_concentration_readout_shape():
    # Row 0: one-hot over 4 topics -> (top_mass, eff_topics) = (1.0, 1.0).
    # Row 1: uniform over 4 topics -> (top_mass, eff_topics) = (0.25, 4.0).
    theta_arr = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.25, 0.25, 0.25, 0.25],
    ])

    summary = lda_concentration_readout(theta_arr)

    assert set(summary.keys()) == {"n_docs", "top_mass", "eff_topics"}
    assert summary["n_docs"] == theta_arr.shape[0]
    # mean of (1.0, 0.25) and (1.0, 4.0) respectively.
    assert summary["top_mass"]["mean"] == pytest.approx((1.0 + 0.25) / 2)
    assert summary["eff_topics"]["mean"] == pytest.approx((1.0 + 4.0) / 2)


def test_lda_concentration_readout_empty_raises():
    with pytest.raises(ValueError):
        lda_concentration_readout(np.zeros((0, 5)))
