"""Supervised softmax prediction head for the PC objective.

The head predicts a document's class from its topic frequencies pi_d (= the
expected token-topic assignment z-bar_d, a point on the K-simplex), NOT from a
separate latent. One weight vector per class:

    logits_d = eta @ pi_d + b           (C,)   eta in R^{C x K}, b in R^C
    yhat_d   = softmax(logits_d)        (C,)   predicted class distribution
    loss_d   = -log yhat_d[y_d]         multiclass cross-entropy (log-loss)

This module is the reusable head: the same function scores whatever base model
supplies pi (flat LDA here; a gated/DAG base later), which is why it takes pi as
data rather than owning any generative state.

Everything is a pure function ``(params, data) -> (value, grads)``.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp, softmax


def softmax_head_loss(
    eta: np.ndarray,
    b: np.ndarray,
    Pi: np.ndarray,
    y: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Total multiclass cross-entropy of softmax(Pi @ eta.T + b) vs labels y.

    Args:
        eta: (C, K) class-by-topic weights.
        b:   (C,)   per-class bias.
        Pi:  (D, K) topic frequencies for the docs being scored (rows on the
             simplex). In the PC objective these are the *labeled* docs only.
        y:   (D,)   integer class labels in {0..C-1}.

    Returns:
        loss:     scalar, sum over docs of -log p(y_d | pi_d).
        grad_eta: (C, K) d loss / d eta.
        grad_b:   (C,)   d loss / d b.
        grad_Pi:  (D, K) d loss / d pi_d — returned so the label can flow into
                  pi during joint optimization (this is what makes the head
                  "constrain" the representation rather than sit on top of it).

    Math (with logits Z = Pi @ eta.T + b, shape (D, C), and P = softmax_c(Z)):
        loss        = sum_d [ logsumexp_c Z_dc - Z_{d, y_d} ]
        dloss/dZ_dc = P_dc - 1[c == y_d]                         =: G (D, C)
        dloss/deta  = G.T @ Pi          (C, K)
        dloss/db    = G.sum(axis=0)     (C,)
        dloss/dPi   = G @ eta           (D, K)
    """
    eta = np.asarray(eta, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    Pi = np.asarray(Pi, dtype=np.float64)
    y = np.asarray(y)

    D = Pi.shape[0]
    if D == 0:
        # No labeled docs: zero loss, zero grads (keeps the assembler branch-free).
        return 0.0, np.zeros_like(eta), np.zeros_like(b), np.zeros_like(Pi)

    Z = Pi @ eta.T + b                      # (D, C) class logits
    lse = logsumexp(Z, axis=1)              # (D,) numerically stable normalizer
    correct = Z[np.arange(D), y]            # (D,) logit of the true class
    loss = float(np.sum(lse - correct))

    P = softmax(Z, axis=1)                  # (D, C) predicted class probs
    G = P.copy()
    G[np.arange(D), y] -= 1.0               # (D, C) dloss/dZ

    grad_eta = G.T @ Pi                     # (C, K)
    grad_b = G.sum(axis=0)                  # (C,)
    grad_Pi = G @ eta                       # (D, K)

    return loss, grad_eta, grad_b, grad_Pi
