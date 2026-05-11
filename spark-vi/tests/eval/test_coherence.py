"""Tests for spark_vi.eval.topic.coherence."""
from __future__ import annotations

import math

import pytest


def test_npmi_pair_independent_returns_zero():
    """If p(i,j) == p(i)*p(j), NPMI = 0."""
    from spark_vi.eval.topic.coherence import _npmi_pair
    # p_i = p_j = 0.5, p_ij = 0.25 -> log(1)/-log(0.25) = 0
    assert _npmi_pair(p_i=0.5, p_j=0.5, p_ij=0.25) == pytest.approx(0.0)


def test_npmi_pair_perfect_cooccurrence_returns_one():
    """If p(i,j) == p(i) == p(j), NPMI = 1 (always co-occur)."""
    from spark_vi.eval.topic.coherence import _npmi_pair
    # p_i = p_j = p_ij = 0.5
    # numerator: log(0.5 / 0.25) = log(2)
    # denominator: -log(0.5) = log(2)
    # ratio = 1
    assert _npmi_pair(p_i=0.5, p_j=0.5, p_ij=0.5) == pytest.approx(1.0)


def test_npmi_pair_zero_cooccurrence_returns_minus_one():
    """Roder et al. 2015 convention: NPMI = -1 when p(i,j) = 0."""
    from spark_vi.eval.topic.coherence import _npmi_pair
    assert _npmi_pair(p_i=0.5, p_j=0.5, p_ij=0.0) == -1.0


def test_npmi_pair_anti_correlated_is_negative():
    """If pair appears less than independence would predict, NPMI < 0."""
    from spark_vi.eval.topic.coherence import _npmi_pair
    # p_i = p_j = 0.5, p_ij = 0.1 << 0.25 (independence baseline)
    result = _npmi_pair(p_i=0.5, p_j=0.5, p_ij=0.1)
    assert result < 0.0
    assert result > -1.0  # not the zero-cooccur sentinel


def test_npmi_pair_handles_small_probabilities():
    """No log-of-zero NaN/inf for tiny but non-zero p_ij."""
    from spark_vi.eval.topic.coherence import _npmi_pair
    result = _npmi_pair(p_i=0.01, p_j=0.01, p_ij=1e-6)
    # Independence baseline: 0.01 * 0.01 = 1e-4; p_ij = 1e-6 << baseline.
    # Anti-correlated; bounded above by 1 and below by -1; not NaN.
    assert math.isfinite(result)
    assert -1.0 <= result <= 1.0
