"""Gradient correctness, lambda=0 reduction, semi-supervised asymmetry, and a
tiny end-to-end optimize for the assembled PC objective."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.special import softmax

from analysis.pc.generative import generative_neg_loglik
from analysis.pc.objective import pack_params, unpack_params, pc_objective
from analysis.pc.tests._grad_utils import rel_grad_error


def _rand_setup(seed, D=6, V=8, K=3, C=2, n_labeled=3):
    rng = np.random.default_rng(seed)
    w = 0.5 * rng.standard_normal((K, V))
    u = 0.5 * rng.standard_normal((D, K))
    eta = 0.5 * rng.standard_normal((C, K))
    b = 0.5 * rng.standard_normal(C)
    X = rng.integers(0, 5, size=(D, V)).astype(np.float64)
    y = rng.integers(0, C, size=D)
    labeled_mask = np.zeros(D, dtype=bool)
    labeled_mask[rng.choice(D, size=n_labeled, replace=False)] = True
    flat = pack_params(w, u, eta, b)
    return dict(flat=flat, X=X, y=y, labeled_mask=labeled_mask, K=K, C=C,
               D=D, V=V, w=w, u=u, eta=eta, b=b, rng=rng)


def test_pack_unpack_roundtrip():
    s = _rand_setup(0)
    w, u, eta, b = unpack_params(s["flat"], K=s["K"], V=s["V"], D=s["D"], C=s["C"])
    assert np.allclose(w, s["w"])
    assert np.allclose(u, s["u"])
    assert np.allclose(eta, s["eta"])
    assert np.allclose(b, s["b"])


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("alpha", [1.0, 0.7])
def test_objective_grad(seed, alpha):
    s = _rand_setup(seed)
    lam = 2.0

    def f(flat):
        val, _ = pc_objective(flat, X=s["X"], y=s["y"],
                              labeled_mask=s["labeled_mask"], K=s["K"], C=s["C"],
                              lam=lam, alpha=alpha)
        return val

    _, grad = pc_objective(s["flat"], X=s["X"], y=s["y"],
                           labeled_mask=s["labeled_mask"], K=s["K"], C=s["C"],
                           lam=lam, alpha=alpha)
    assert rel_grad_error(f, grad, s["flat"]) < 1e-5


@pytest.mark.parametrize("seed", range(4))
def test_lambda_zero_reduces_to_generative(seed):
    s = _rand_setup(seed)
    kw = dict(X=s["X"], labeled_mask=s["labeled_mask"], K=s["K"], C=s["C"],
              lam=0.0, alpha=0.9)

    val0, grad0 = pc_objective(s["flat"], y=s["y"], **kw)

    # (a) Changing y leaves value and gradient unchanged when lam=0.
    y2 = (s["y"] + 1) % s["C"]
    val1, grad1 = pc_objective(s["flat"], y=y2, **kw)
    assert np.isclose(val0, val1)
    assert np.allclose(grad0, grad1)

    # value equals the pure generative objective evaluated at (beta, Pi)
    w, u, eta, b = unpack_params(s["flat"], K=s["K"], V=s["V"], D=s["D"], C=s["C"])
    beta = softmax(w, axis=1)
    Pi = softmax(u, axis=1)
    gen_val, _, _ = generative_neg_loglik(beta, Pi, s["X"], alpha=0.9)
    assert np.isclose(val0, gen_val)

    # (b) grad w.r.t. eta and b is exactly 0 at lam=0.
    n_w = s["K"] * s["V"]
    n_u = s["D"] * s["K"]
    n_eta = s["C"] * s["K"]
    g_eta = grad0[n_w + n_u: n_w + n_u + n_eta]
    g_b = grad0[n_w + n_u + n_eta:]
    assert np.all(g_eta == 0.0)
    assert np.all(g_b == 0.0)


@pytest.mark.parametrize("seed", range(4))
def test_semisupervised_asymmetry(seed):
    """Flipping a labeled doc to unlabeled removes its PRED gradient share.

    An unlabeled doc's grad-u block is identical whether lam=0 or lam>0 and
    regardless of its (ignored) label. A labeled doc's grad-u block moves with
    lam. Flipping a labeled doc off the mask makes its block match the lam=0
    (generative-only) block.
    """
    s = _rand_setup(seed)
    D, K, V = s["D"], s["K"], s["V"]
    n_w = K * V

    def grad_u_block(mask, lam):
        _, g = pc_objective(s["flat"], X=s["X"], y=s["y"], labeled_mask=mask,
                            K=K, C=s["C"], lam=lam, alpha=1.0)
        return g[n_w:n_w + D * K].reshape(D, K)

    gen_only = grad_u_block(s["labeled_mask"], lam=0.0)
    full = grad_u_block(s["labeled_mask"], lam=3.0)

    labeled_idx = np.where(s["labeled_mask"])[0]
    unlabeled_idx = np.where(~s["labeled_mask"])[0]

    # Unlabeled docs: gradient identical with or without lambda.
    for d in unlabeled_idx:
        assert np.allclose(gen_only[d], full[d])
    # At least one labeled doc actually picks up a PRED contribution.
    assert any(not np.allclose(gen_only[d], full[d]) for d in labeled_idx)

    # Flip the first labeled doc to unlabeled -> its block reverts to gen-only.
    flipped = s["labeled_mask"].copy()
    d0 = labeled_idx[0]
    flipped[d0] = False
    partial = grad_u_block(flipped, lam=3.0)
    assert np.allclose(partial[d0], gen_only[d0])
    # Other still-labeled docs keep their PRED contribution.
    for d in labeled_idx[1:]:
        assert np.allclose(partial[d], full[d])


def test_tiny_optimize_converges():
    s = _rand_setup(3, D=8, V=6, K=2, C=2, n_labeled=5)

    def fun(flat):
        return pc_objective(flat, X=s["X"], y=s["y"],
                            labeled_mask=s["labeled_mask"], K=s["K"], C=s["C"],
                            lam=1.0, alpha=1.0)

    v0, _ = fun(s["flat"])
    res = minimize(fun, s["flat"], jac=True, method="L-BFGS-B",
                   options=dict(maxiter=200))
    assert np.isfinite(res.fun)
    assert res.fun < v0  # objective decreased
