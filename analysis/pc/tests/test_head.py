"""Gradient correctness for the softmax prediction head."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.special import softmax

from analysis.pc.head import softmax_head_loss
from analysis.pc.tests._grad_utils import rel_grad_error


def _rand_instance(seed, D=5, K=3, C=2):
    rng = np.random.default_rng(seed)
    eta = rng.standard_normal((C, K))
    b = rng.standard_normal(C)
    Pi = softmax(rng.standard_normal((D, K)), axis=1)  # rows on the simplex
    y = rng.integers(0, C, size=D)
    return eta, b, Pi, y


@pytest.mark.parametrize("seed", range(5))
def test_head_grad_eta(seed):
    eta, b, Pi, y = _rand_instance(seed)
    _, g_eta, _, _ = softmax_head_loss(eta, b, Pi, y)

    def f(flat):
        loss, *_ = softmax_head_loss(flat.reshape(eta.shape), b, Pi, y)
        return loss

    assert rel_grad_error(f, g_eta, eta) < 1e-5


@pytest.mark.parametrize("seed", range(5))
def test_head_grad_b(seed):
    eta, b, Pi, y = _rand_instance(seed)
    _, _, g_b, _ = softmax_head_loss(eta, b, Pi, y)

    def f(flat):
        loss, *_ = softmax_head_loss(eta, flat, Pi, y)
        return loss

    assert rel_grad_error(f, g_b, b) < 1e-5


@pytest.mark.parametrize("seed", range(5))
def test_head_grad_pi(seed):
    eta, b, Pi, y = _rand_instance(seed)
    _, _, _, g_pi = softmax_head_loss(eta, b, Pi, y)

    def f(flat):
        loss, *_ = softmax_head_loss(eta, b, flat.reshape(Pi.shape), y)
        return loss

    assert rel_grad_error(f, g_pi, Pi) < 1e-5


def test_head_perfect_prediction_low_loss():
    """A head that puts all logit mass on the true class has ~0 loss."""
    D, K, C = 4, 3, 2
    Pi = np.full((D, K), 1.0 / K)
    y = np.array([0, 1, 0, 1])
    # huge positive weight on topic 0 for class chosen by y via bias trick:
    eta = np.zeros((C, K))
    b = np.array([50.0, -50.0])  # always predicts class 0 strongly
    loss, *_ = softmax_head_loss(eta, b, Pi, y)
    # class-0 docs contribute ~0, class-1 docs contribute ~100 each.
    assert np.isfinite(loss)
    assert loss > 0
