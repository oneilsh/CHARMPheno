"""Assembler for the PC Lagrangian and the flat-vector pack/unpack helpers that
``scipy.optimize.minimize`` consumes.

The optimizer works in the UNCONSTRAINED space (w, u, eta, b); this module maps
those through softmax to the constrained (beta, Pi), calls the factored
generative and head terms, assembles

    L = GEN(all docs) + lam * PRED(labeled docs),

and pushes the returned d/dbeta and d/dPi gradients back through the softmax
Jacobians to land gradients in (w, u) space, then flattens everything.

Flat-vector layout (a single contiguous float64 vector):

    [ w (K*V) | u (D*K) | eta (C*K) | b (C) ]

The doc-topic block is stored jointly (all D docs, one row of u per doc) rather
than per-doc so the whole state is one vector for L-BFGS-B; per-doc slices are
recovered by reshaping.

Softmax vector-Jacobian product (for s = softmax(a), upstream grad g_s):

    g_a = s * (g_s - sum(g_s * s))

applied row-wise to beta (over V) and to Pi (over K). eta and b are already
unconstrained, so their gradients pass through untouched (scaled by lam).
"""
from __future__ import annotations

import numpy as np
from scipy.special import softmax

from analysis.pc.generative import generative_neg_loglik
from analysis.pc.head import softmax_head_loss


def pack_params(
    w: np.ndarray, u: np.ndarray, eta: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """Flatten (w, u, eta, b) into the single contiguous vector L-BFGS-B sees."""
    return np.concatenate([
        np.asarray(w, dtype=np.float64).ravel(),
        np.asarray(u, dtype=np.float64).ravel(),
        np.asarray(eta, dtype=np.float64).ravel(),
        np.asarray(b, dtype=np.float64).ravel(),
    ])


def unpack_params(
    flat: np.ndarray, *, K: int, V: int, D: int, C: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of :func:`pack_params`. Returns views/reshapes (w, u, eta, b)."""
    flat = np.asarray(flat, dtype=np.float64)
    n_w, n_u, n_eta = K * V, D * K, C * K
    i = 0
    w = flat[i:i + n_w].reshape(K, V); i += n_w
    u = flat[i:i + n_u].reshape(D, K); i += n_u
    eta = flat[i:i + n_eta].reshape(C, K); i += n_eta
    b = flat[i:i + C]
    return w, u, eta, b


def _softmax_vjp(s: np.ndarray, g_s: np.ndarray) -> np.ndarray:
    """Row-wise softmax vector-Jacobian product: given s = softmax(a) (rows over
    the last axis) and upstream gradient g_s (d L / d s), return d L / d a.

    J = diag(s) - s s^T per row; g_a = s * (g_s - (g_s . s)).
    """
    dot = np.sum(g_s * s, axis=1, keepdims=True)
    return s * (g_s - dot)


def pc_objective(
    params_flat: np.ndarray,
    *,
    X: np.ndarray,
    y: np.ndarray,
    labeled_mask: np.ndarray,
    K: int,
    C: int,
    lam: float,
    alpha: float,
) -> tuple[float, np.ndarray]:
    """Value and flat gradient of the PC Lagrangian at ``params_flat``.

    Args:
        params_flat:  packed (w, u, eta, b); see module layout.
        X:            (D, V) nonnegative counts (ALL docs).
        y:            (D,) integer labels; entries where ``labeled_mask`` is
                      False are ignored (may be any placeholder).
        labeled_mask: (D,) bool; True for docs in the labeled subset L.
        K, C:         number of topics / classes.
        lam:          scalar PC multiplier (>= 0); scales PRED. lam == 0 makes
                      the objective (value and gradient) independent of y, eta, b.
        alpha:        symmetric Dirichlet concentration on pi (alpha == 1 => no prior).

    Returns:
        (value, grad_flat) with grad_flat in the same layout as params_flat,
        ready for ``scipy.optimize.minimize(..., jac=True, method="L-BFGS-B")``.

    Assembly:
        beta = softmax(w) (rows over V);  Pi = softmax(u) (rows over K).
        GEN over ALL docs; PRED = lam * head-loss over LABELED docs only.
        d/dbeta comes only from GEN. d/dPi = GEN's grad everywhere, plus
        lam * head's grad on the labeled rows (semi-supervised asymmetry).
        eta/b gradients come only from lam * PRED (=> exactly 0 when lam == 0).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    labeled_mask = np.asarray(labeled_mask, dtype=bool)
    D, V = X.shape

    w, u, eta, b = unpack_params(params_flat, K=K, V=V, D=D, C=C)
    beta = softmax(w, axis=1)          # (K, V) topic-word simplex rows
    Pi = softmax(u, axis=1)            # (D, K) doc-topic simplex rows

    # --- Generative term over ALL docs ---
    gen_val, grad_beta, grad_Pi = generative_neg_loglik(beta, Pi, X, alpha)
    value = gen_val

    grad_eta = np.zeros_like(eta)
    grad_b = np.zeros_like(b)

    # --- Prediction constraint over LABELED docs only, scaled by lam ---
    if lam != 0.0 and labeled_mask.any():
        Pi_L = Pi[labeled_mask]
        y_L = y[labeled_mask]
        pred_loss, ge, gb, gPi_L = softmax_head_loss(eta, b, Pi_L, y_L)
        value += lam * pred_loss
        grad_eta = lam * ge
        grad_b = lam * gb
        # Only labeled rows of Pi pick up the PRED gradient (asymmetry).
        grad_Pi[labeled_mask] += lam * gPi_L

    # --- Map constrained grads back to unconstrained (w, u) via softmax VJP ---
    grad_w = _softmax_vjp(beta, grad_beta)
    grad_u = _softmax_vjp(Pi, grad_Pi)

    grad_flat = pack_params(grad_w, grad_u, grad_eta, grad_b)
    return float(value), grad_flat
