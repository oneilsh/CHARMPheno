"""Gradient correctness + prior handling for the generative term."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.special import softmax

from analysis.pc.generative import generative_neg_loglik
from analysis.pc.tests._grad_utils import rel_grad_error


def _rand_instance(seed, D=6, V=8, K=3):
    rng = np.random.default_rng(seed)
    beta = softmax(rng.standard_normal((K, V)), axis=1)   # rows on simplex
    Pi = softmax(rng.standard_normal((D, K)), axis=1)     # rows on simplex
    X = rng.integers(0, 5, size=(D, V)).astype(np.float64)
    return beta, Pi, X


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("alpha", [1.0, 0.5, 1.7])
def test_gen_grad_beta(seed, alpha):
    beta, Pi, X = _rand_instance(seed)
    _, g_beta, _ = generative_neg_loglik(beta, Pi, X, alpha)

    def f(flat):
        val, *_ = generative_neg_loglik(flat.reshape(beta.shape), Pi, X, alpha)
        return val

    assert rel_grad_error(f, g_beta, beta) < 1e-5


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("alpha", [1.0, 0.5, 1.7])
def test_gen_grad_pi(seed, alpha):
    beta, Pi, X = _rand_instance(seed)
    _, _, g_pi = generative_neg_loglik(beta, Pi, X, alpha)

    def f(flat):
        val, *_ = generative_neg_loglik(beta, flat.reshape(Pi.shape), X, alpha)
        return val

    assert rel_grad_error(f, g_pi, Pi) < 1e-5


def test_alpha_one_has_no_prior():
    """alpha == 1 => Dirichlet(1) is uniform => no prior term, no Pi-prior grad."""
    beta, Pi, X = _rand_instance(0)
    val1, gb1, gp1 = generative_neg_loglik(beta, Pi, X, alpha=1.0)
    # Data-only reference computed independently.
    M = Pi @ beta
    data_val = -np.sum(X * np.log(M))
    assert np.isclose(val1, data_val)

    # With alpha != 1 the value and the Pi-gradient must differ (prior active),
    # while the beta-gradient is unchanged (prior does not touch beta).
    val2, gb2, gp2 = generative_neg_loglik(beta, Pi, X, alpha=0.5)
    assert not np.isclose(val1, val2)
    assert np.allclose(gb1, gb2)
    assert not np.allclose(gp1, gp2)
