from __future__ import annotations

import math
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import digamma

from spark_vi.models.topic._linalg import safe_inverse
from spark_vi.mllib.topic.stm import _stm_doc_inference_tprior
from spark_vi.models.topic.stm import _stm_doc_inference


def _setup(K=4, V=12, reference=None):
    beta = np.full((K, V), 1e-3)
    blk = V // K
    for k in range(K):
        beta[k, k * blk:(k + 1) * blk] += 2.0
    beta /= beta.sum(axis=1, keepdims=True)
    lam = beta * (500.0 * V) + 0.01
    expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))
    Gamma = np.zeros((1, K))
    R = np.eye(K)
    allowed = np.arange(K, dtype=np.int64)
    indices = np.array([0, 1, blk, blk + 1, 2 * blk], dtype=np.int64)
    counts = np.array([3.0, 2.0, 5.0, 1.0, 4.0])
    x = np.array([1.0])
    return dict(indices=indices, counts=counts, expElogbeta=expElogbeta,
                Gamma=Gamma, Rinv_allowed=safe_inverse(R), x=x,
                allowed=allowed, reference=reference), R


def test_nu_inf_reproduces_single_gaussian_solve():
    kw, R = _setup()
    c = 4.0
    eta_t, sd_t, _, n_em = _stm_doc_inference_tprior(**kw, c=c, nu=math.inf)
    assert sd_t == 1.0 and n_em == 1
    Sigma_inv_allowed = (1.0 / c) * kw["Rinv_allowed"]
    eta_g, _, _ = _stm_doc_inference(
        indices=kw["indices"], counts=kw["counts"], expElogbeta=kw["expElogbeta"],
        Gamma=kw["Gamma"], Sigma_inv_allowed=Sigma_inv_allowed, x=kw["x"],
        allowed=kw["allowed"], reference=kw["reference"],
    )
    al = kw["allowed"]
    assert np.allclose(eta_t[al], eta_g[al], atol=1e-8)


def test_sd_update_matches_brute_force_at_fixed_eta():
    kw, R = _setup()
    c, nu = 4.0, 5.0
    # One EM sweep so we have a converged eta at sd=1, then check the sd mode.
    eta_t, sd_t, _, _ = _stm_doc_inference_tprior(
        **kw, c=c, nu=nu, sd_max_iter=1)      # single sd update
    al = kw["allowed"]
    mu = kw["Gamma"].T @ kw["x"]               # (K,) here zero-mean
    diff = eta_t[al] - mu[al]
    q_R = float(diff @ kw["Rinv_allowed"] @ diff)
    K_free = len(al)                           # reference=None
    # brute force: maximize the conditional log-posterior of s given eta
    def neg_logpost(s):
        if s <= 0:
            return np.inf
        return (0.5 * K_free * math.log(s) + q_R / (2.0 * s * c)
                + (nu / 2.0 + 1.0) * math.log(s) + (nu / 2.0) / s)
    opt = minimize_scalar(neg_logpost, bounds=(1e-4, 50.0), method="bounded")
    closed = (nu + q_R / c) / (nu + K_free + 2.0)
    assert abs(closed - opt.x) < 1e-3
    assert abs(sd_t - closed) < 1e-6


def test_em_converges_to_fixed_point():
    kw, R = _setup()
    c, nu = 4.0, 5.0
    eta_t, sd_t, _, n_em = _stm_doc_inference_tprior(
        **kw, c=c, nu=nu, sd_max_iter=50, sd_tol=1e-8)
    assert n_em < 50                            # stopped early = converged
    al = kw["allowed"]
    mu = kw["Gamma"].T @ kw["x"]
    diff = eta_t[al] - mu[al]
    q_R = float(diff @ kw["Rinv_allowed"] @ diff)
    K_free = len(al)
    assert abs(sd_t - (nu + q_R / c) / (nu + K_free + 2.0)) < 1e-5


def test_warm_start_invariance():
    kw, R = _setup()
    c, nu = 4.0, 5.0
    cold = _stm_doc_inference_tprior(**kw, c=c, nu=nu)
    warm = _stm_doc_inference_tprior(**kw, c=c, nu=nu, eta_init=cold[0], sd_init=cold[1])
    al = kw["allowed"]
    assert np.allclose(cold[0][al], warm[0][al], atol=1e-5)
    assert abs(cold[1] - warm[1]) < 1e-5
