"""Unit tests for stm_sigma_diagnostic (per-topic eta-variance / spectrum report).

Imports analysis._eval_common via the repo root (the package seam the cloud
eval driver imports from). Pure-numpy helper, no Spark.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

from analysis._eval_common import stm_sigma_diagnostic


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


def test_reports_spectrum_condition_and_offdiag():
    # Strong 0<->2 correlation; known eigen-spectrum.
    Sigma = np.array([
        [4.0, 0.0, 3.0],
        [0.0, 1.0, 0.0],
        [3.0, 0.0, 4.0],
    ])
    report = stm_sigma_diagnostic(Sigma, labels=None, top_k=3)

    # max |off-diag corr| = 3 / sqrt(4*4) = 0.75
    assert "0.75" in report
    # eig of [[4,3],[3,4]] are 7 and 1, plus the isolated 1 -> cond = 7/1 = 7
    assert "cond=7" in report


def test_returns_none_for_non_square_or_1d():
    assert stm_sigma_diagnostic(np.ones(5), labels=None) is None
    assert stm_sigma_diagnostic(np.ones((3, 4)), labels=None) is None
    assert stm_sigma_diagnostic(None, labels=None) is None
