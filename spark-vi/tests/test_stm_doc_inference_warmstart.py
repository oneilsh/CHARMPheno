from __future__ import annotations

import numpy as np

from spark_vi.models.topic.stm import _stm_doc_inference
from spark_vi.models.topic._linalg import safe_inverse


def _setup(K=4, V=12, reference=None):
    beta = np.full((K, V), 1e-3)
    blk = V // K
    for k in range(K):
        beta[k, k * blk:(k + 1) * blk] += 2.0
    beta /= beta.sum(axis=1, keepdims=True)
    lam = beta * (500.0 * V) + 0.01
    from scipy.special import digamma
    expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))
    Gamma = np.zeros((1, K))
    R = np.eye(K)
    allowed = np.arange(K, dtype=np.int64)
    Sigma_inv_allowed = (1.0 / 4.0) * safe_inverse(R)
    indices = np.array([0, 1, blk, blk + 1, 2 * blk], dtype=np.int64)
    counts = np.array([3.0, 2.0, 5.0, 1.0, 4.0])
    x = np.array([1.0])
    return dict(indices=indices, counts=counts, expElogbeta=expElogbeta,
                Gamma=Gamma, Sigma_inv_allowed=Sigma_inv_allowed, x=x,
                allowed=allowed, reference=reference)


def test_eta_init_none_matches_explicit_zeros():
    kw = _setup()
    a = _stm_doc_inference(**kw)
    b = _stm_doc_inference(**kw, eta_init=np.zeros(4))
    assert np.allclose(a[0][kw["allowed"]], b[0][kw["allowed"]], atol=1e-8)


def test_warm_start_reaches_same_mode():
    kw = _setup()
    cold = _stm_doc_inference(**kw)
    warm = _stm_doc_inference(**kw, eta_init=cold[0])   # seed from the solution
    # tol tracks L-BFGS gtol=1e-4: cold/warm halt within ~2e-5 of each other (warm is closer to the true optimum)
    assert np.allclose(cold[0][kw["allowed"]], warm[0][kw["allowed"]], atol=5e-5)


def test_warm_start_reaches_same_mode_reference():
    kw = _setup(reference=0)
    cold = _stm_doc_inference(**kw)
    warm = _stm_doc_inference(**kw, eta_init=cold[0])
    # tol tracks L-BFGS gtol=1e-4: cold/warm halt within ~2e-5 of each other (warm is closer to the true optimum)
    assert np.allclose(cold[0][kw["allowed"]], warm[0][kw["allowed"]], atol=5e-5)
