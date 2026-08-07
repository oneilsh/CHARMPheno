"""Generative (unsupervised) term of the PC objective: the flat-LDA data
neg-log-likelihood under a point estimate of the doc-topic proportions, plus a
symmetric Dirichlet(alpha) neg-log-prior on those proportions.

For each document d with count row x_d over vocab V and topic frequencies pi_d
on the K-simplex, and topic-word rows beta_k on the V-simplex, the per-token
word probability marginalizes the topic:

    p(word = v | d) = sum_k pi_dk beta_kv = (Pi @ beta)_dv

so, writing M = Pi @ beta (D, V), the generative term is

    GEN = -sum_d sum_v x_dv log M_dv                 # data neg-log-lik (ALL docs)
          - (alpha - 1) sum_d sum_k log pi_dk        # Dirichlet(alpha) neg-log-prior

This is the term the semi-supervised asymmetry keeps identical for labeled and
unlabeled docs — the label never corrupts the word likelihood.

Modeling notes:
  * The Dirichlet log-normalizer -log B(alpha) is constant in (beta, pi) and is
    dropped: it shifts the objective by a constant and has zero gradient, so it
    changes neither the optimizer's trajectory nor the check_grad comparison.
  * alpha == 1  =>  Dirichlet(1) is uniform  =>  the (alpha - 1) factor is 0 and
    there is no prior term (and no prior gradient on pi). Handled by the factor
    naturally; special-cased only to skip the work.
  * log(0) guard: M = Pi @ beta is strictly positive whenever beta > 0 (rows
    from a softmax always are), so no floor is needed for the intended inputs.
    As a defensive measure for degenerate hand-supplied beta/pi, M is floored at
    a tiny ``_M_FLOOR``; this never triggers for softmax-parametrized inputs so
    it does not perturb the checked gradients.
"""
from __future__ import annotations

import numpy as np

# Only guards genuinely degenerate inputs (a beta column that is exactly 0 in
# every topic). Softmax-parametrized beta is strictly positive, so for the PC
# objective this floor is inert and leaves gradients exact.
_M_FLOOR = 1e-300


def generative_neg_loglik(
    beta: np.ndarray,
    Pi: np.ndarray,
    X: np.ndarray,
    alpha: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Generative neg-log-lik + Dirichlet neg-log-prior, and its gradients.

    Args:
        beta:  (K, V) topic-word rows on the simplex.
        Pi:    (D, K) doc-topic rows on the simplex (point estimate of theta).
        X:     (D, V) nonnegative word counts.
        alpha: symmetric Dirichlet concentration (scalar). alpha == 1 => no prior.

    Returns:
        val:       scalar GEN value.
        grad_beta: (K, V) d GEN / d beta.
        grad_Pi:   (D, K) d GEN / d pi.

    Math (M = Pi @ beta, R = -X / M):
        data      = -sum(X * log M)
        dR/dM     : R = dData/dM = -X / M
        dData/dbeta = Pi.T @ R          (K, V)
        dData/dPi   = R @ beta.T        (D, K)
        prior     = -(alpha - 1) * sum(log Pi)
        dPrior/dPi  = -(alpha - 1) / Pi (D, K)   (no beta dependence)
    """
    beta = np.asarray(beta, dtype=np.float64)
    Pi = np.asarray(Pi, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    alpha = float(alpha)

    M = Pi @ beta                                  # (D, V) per-token word probs
    M = np.maximum(M, _M_FLOOR)                    # inert for softmax inputs

    data_val = -np.sum(X * np.log(M))
    R = -X / M                                     # (D, V) = dData/dM
    grad_beta = Pi.T @ R                           # (K, V)
    grad_Pi = R @ beta.T                           # (D, K)

    if alpha != 1.0:
        # Dirichlet(alpha) neg-log-prior on each pi_d (normalizer dropped, const).
        data_val += -(alpha - 1.0) * np.sum(np.log(Pi))
        grad_Pi = grad_Pi - (alpha - 1.0) / Pi

    return float(data_val), grad_beta, grad_Pi
