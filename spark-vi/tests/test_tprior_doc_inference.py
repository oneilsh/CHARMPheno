from __future__ import annotations

import math
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import digamma

from spark_vi.models.topic._linalg import safe_inverse
from spark_vi.mllib.topic.stm import _stm_doc_inference_tprior
from spark_vi.models.topic.stm import _stm_doc_inference


def _setup(K=4, V=12, reference=None, R=None):
    beta = np.full((K, V), 1e-3)
    blk = V // K
    for k in range(K):
        beta[k, k * blk:(k + 1) * blk] += 2.0
    beta /= beta.sum(axis=1, keepdims=True)
    lam = beta * (500.0 * V) + 0.01
    expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))
    Gamma = np.zeros((1, K))
    if R is None:
        R = np.eye(K)
    allowed = np.arange(K, dtype=np.int64)
    indices = np.array([0, 1, blk, blk + 1, 2 * blk], dtype=np.int64)
    counts = np.array([3.0, 2.0, 5.0, 1.0, 4.0])
    x = np.array([1.0])
    return dict(indices=indices, counts=counts, expElogbeta=expElogbeta,
                Gamma=Gamma, Rinv_allowed=safe_inverse(R), x=x,
                allowed=allowed, reference=reference), R


def _equicorr(K, rho):
    """Equicorrelation matrix: 1 on the diagonal, rho off-diagonal (a valid
    correlation matrix for -1/(K-1) < rho < 1)."""
    return (1.0 - rho) * np.eye(K) + rho * np.ones((K, K))


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
    al = kw["allowed"]
    mu = kw["Gamma"].T @ kw["x"]               # (K,) here zero-mean
    # The single sd-update inside the function is the IG mode of the eta solved
    # at the INPUT sd (=sd_init=1). The returned eta_t is NOT that eta: the
    # closing eta-solve re-solves at the updated sd, decoupling eta_t from the
    # sd-update's input. So to validate the sd-update formula against the eta it
    # actually consumed, reconstruct that input eta directly (cold solve at sd=1,
    # same max_iter/tol defaults the function uses => bit-identical).
    Sigma_inv_1 = (1.0 / (1.0 * c)) * kw["Rinv_allowed"]
    eta_1, _, _ = _stm_doc_inference(
        indices=kw["indices"], counts=kw["counts"], expElogbeta=kw["expElogbeta"],
        Gamma=kw["Gamma"], Sigma_inv_allowed=Sigma_inv_1, x=kw["x"],
        allowed=al, reference=kw["reference"],
    )
    _, sd_t, _, _ = _stm_doc_inference_tprior(
        **kw, c=c, nu=nu, sd_max_iter=1)      # single sd update
    diff = eta_1[al] - mu[al]
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


def test_returned_pair_is_self_consistent_correlated_R_small_nu():
    # Realistic t-prior regime: correlated (non-identity) R and a small nu
    # drawn from the actual diagnostic grid (nu=2.5). The strong invariant
    # from the closing eta-solve: the returned eta_hat is EXACTLY the Laplace
    # argmax at the returned sd (up to lbfgs_tol), so re-solving _stm_doc_inference
    # at Sigma_inv=(1/(sd*c))*Rinv_allowed reproduces it. This holds by
    # construction regardless of how fast the sd sequence contracted.
    rho = 0.5
    K = 4
    R = _equicorr(K, rho)
    assert not np.allclose(R, np.eye(K))          # guard: R really is correlated
    kw, R = _setup(K=K, R=R)
    c, nu = 4.0, 2.5
    lbfgs_tol = 1e-4
    eta_t, sd_t, _, _ = _stm_doc_inference_tprior(**kw, c=c, nu=nu, lbfgs_tol=lbfgs_tol)
    al = kw["allowed"]
    Sigma_inv_allowed = (1.0 / (sd_t * c)) * kw["Rinv_allowed"]
    eta_re, _, _ = _stm_doc_inference(
        indices=kw["indices"], counts=kw["counts"], expElogbeta=kw["expElogbeta"],
        Gamma=kw["Gamma"], Sigma_inv_allowed=Sigma_inv_allowed, x=kw["x"],
        tol=lbfgs_tol, allowed=al, reference=kw["reference"], eta_init=eta_t,
    )
    # The closing eta-solve makes eta_t the argmax at sd_t (to lbfgs_tol), so
    # re-solving _stm_doc_inference from eta_t re-starts AT the fixed point: the
    # solver's first gradient check already passes and it returns eta_t
    # unchanged. Agreement is therefore to solver-fixed-point precision (~0),
    # far tighter than lbfgs_tol. A tight bound (1e-8) is what makes this a real
    # regression guard: the pre-fix one-step-lag returned an eta solved at the
    # PRIOR sd, which under this correlated-R/small-nu regime re-solves ~3e-5
    # away -- above 1e-8, so this assertion would catch a reintroduced lag.
    assert np.allclose(eta_t[al], eta_re[al], atol=1e-8)


def test_nu_inf_reproduces_single_gaussian_solve_reference0():
    # Production (exp 0048) fits with reference_topic=True -> reference=0, but
    # every other test in this file uses reference=None. Cover the pinned-
    # reference path explicitly: nu=inf must still nest the plain Gaussian
    # solve, now with topic 0 pinned at eta=0 and K_free = |allowed| - 1.
    kw, R = _setup(reference=0)
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
    assert eta_t[0] == 0.0


def test_returned_pair_is_self_consistent_correlated_R_small_nu_reference0():
    # reference=0 variant of test_returned_pair_is_self_consistent_correlated_R_small_nu
    # (correlated R, nu=2.5): the pinned-reference s_d step uses the FULL
    # allowed-set q_R (including the reference's (0 - mu_ref) deviation) and
    # K_free = |allowed| - 1 (see _stm_doc_inference_tprior's docstring). The
    # same closing-eta-solve invariant must still hold with the reference
    # pinned: re-solving _stm_doc_inference(..., reference=0) at the returned
    # sd reproduces the returned eta_hat.
    #
    # The pinned sub-problem (K_free = 3 here) converges its EM sd-sequence
    # more slowly than the unconstrained one at the library's DEFAULT
    # sd_tol=1e-4/sd_max_iter=10 -- empirically the default leaves a ~7e-5
    # residual (still within the documented "up to lbfgs_tol" bound, since
    # 7e-5 < lbfgs_tol=1e-4, but not tight enough for a 1e-8 regression
    # guard). Tightening sd_tol/sd_max_iter here (not a production default
    # change -- just this test's exercise of the invariant) lets the EM
    # sequence actually reach the fixed point, so the closing-eta-solve
    # invariant can be checked at high precision.
    rho = 0.5
    K = 4
    R = _equicorr(K, rho)
    assert not np.allclose(R, np.eye(K))          # guard: R really is correlated
    kw, R = _setup(K=K, R=R, reference=0)
    c, nu = 4.0, 2.5
    lbfgs_tol = 1e-4
    eta_t, sd_t, _, _ = _stm_doc_inference_tprior(
        **kw, c=c, nu=nu, lbfgs_tol=lbfgs_tol, sd_tol=1e-6, sd_max_iter=30)
    al = kw["allowed"]
    assert eta_t[0] == 0.0                        # reference stays pinned at 0
    Sigma_inv_allowed = (1.0 / (sd_t * c)) * kw["Rinv_allowed"]
    eta_re, _, _ = _stm_doc_inference(
        indices=kw["indices"], counts=kw["counts"], expElogbeta=kw["expElogbeta"],
        Gamma=kw["Gamma"], Sigma_inv_allowed=Sigma_inv_allowed, x=kw["x"],
        tol=lbfgs_tol, allowed=al, reference=kw["reference"], eta_init=eta_t,
    )
    assert np.allclose(eta_t[al], eta_re[al], atol=1e-8)


def test_warm_start_invariance_correlated_R_small_nu():
    # Same correlated-R / small-nu regime; cold vs warm-restarted-from-cold.
    # Two different inits converge the sd sequence to within O(sd_tol) of the
    # same fixed point, and the closing eta-solve is a continuous function of
    # sd, so the returned etas agree to O(sd_tol) too -- NOT machine epsilon.
    # We therefore assert at a tolerance derived from sd_tol, not 1e-12.
    rho = 0.5
    K = 4
    R = _equicorr(K, rho)
    kw, R = _setup(K=K, R=R)
    c, nu = 4.0, 2.5
    sd_tol = 1e-4
    cold = _stm_doc_inference_tprior(**kw, c=c, nu=nu, sd_tol=sd_tol)
    warm = _stm_doc_inference_tprior(**kw, c=c, nu=nu, sd_tol=sd_tol,
                                     eta_init=cold[0], sd_init=cold[1])
    al = kw["allowed"]
    # sd converged to O(sd_tol) of the fixed point from either init -> the two
    # returned sds sit within a couple of tol-sized steps of each other.
    assert abs(cold[1] - warm[1]) < 2.0 * sd_tol
    # eta is the argmax at sd; d(eta)/d(sd) is O(1) here, so an O(sd_tol)
    # spread in sd maps to an O(sd_tol) spread in eta. Allow a small constant
    # factor for the L-BFGS gradient tolerance on top.
    assert np.allclose(cold[0][al], warm[0][al], atol=5.0 * sd_tol)
