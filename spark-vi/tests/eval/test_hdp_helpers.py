"""Tests for spark_vi.eval.topic.hdp_helpers."""
from __future__ import annotations

import numpy as np
import pytest


def test_top_k_used_topics_returns_correct_length_mask():
    from spark_vi.eval.topic.hdp_helpers import top_k_used_topics
    # T=5 corpus-level sticks. u, v are length T-1 = 4. The last topic carries
    # the residual stick mass.
    u = np.array([1.0, 1.0, 1.0, 1.0])
    v = np.array([1.0, 1.0, 1.0, 1.0])
    mask = top_k_used_topics(u=u, v=v, k=3)
    assert mask.shape == (5,)
    assert mask.dtype == bool
    assert mask.sum() == 3


def test_top_k_used_topics_selects_largest_expected_betas():
    """E[beta_t] computed from GEM stick-breaking; mask picks the top-K by E[beta]."""
    from spark_vi.eval.topic.hdp_helpers import top_k_used_topics
    # Construct (u, v) so that the first stick takes ~half the mass:
    #   E[V_1] = u_1 / (u_1 + v_1) = 9/10
    #   E[beta_1] = E[V_1] = 0.9
    #   subsequent sticks split the remaining 0.1.
    u = np.array([9.0, 1.0, 1.0, 1.0])
    v = np.array([1.0, 1.0, 1.0, 1.0])
    mask = top_k_used_topics(u=u, v=v, k=2)
    # Topic 0 dominates; topic 0 must be in the mask.
    assert mask[0]
    # The remaining 0.1 is split mostly into topic 1, then 2, then 3, then 4 (residual).
    assert mask[1]
    assert mask.sum() == 2


def test_top_k_used_topics_k_must_not_exceed_T():
    from spark_vi.eval.topic.hdp_helpers import top_k_used_topics
    u = np.array([1.0, 1.0])
    v = np.array([1.0, 1.0])  # T = 3
    with pytest.raises(ValueError, match="k"):
        top_k_used_topics(u=u, v=v, k=4)


def test_top_k_used_topics_validates_u_v_same_length():
    from spark_vi.eval.topic.hdp_helpers import top_k_used_topics
    u = np.array([1.0, 1.0, 1.0])
    v = np.array([1.0, 1.0])
    with pytest.raises(ValueError, match="length"):
        top_k_used_topics(u=u, v=v, k=2)
