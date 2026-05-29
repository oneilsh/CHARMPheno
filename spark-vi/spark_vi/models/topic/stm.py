"""OnlineSTM: prevalence-only Structural Topic Model as a VIModel.

Generative model:
    For each topic k:  β_k ~ Dirichlet(η)                  # shared, unchanged from LDA
    For each doc d:
        η_d ~ N(Γ x_d, Σ)                                  # logistic-normal prior on logit-θ
        θ_d = softmax(η_d)
        For each token n in d:
            z_dn ~ Categorical(θ_d)
            w_dn ~ Categorical(β_{z_dn})

Variational family:
    q(β_k)  = Dirichlet(λ_k)                               # unchanged from LDA
    q(η_d)  = N(μ_d, ν_d) with K-diagonal ν_d              # Laplace approximation
    q(z_dn) = collapsed via Lee/Seung trick                # unchanged from LDA

Per-doc inference is two-step Laplace (ADR 0023):
    Step (a): L-BFGS finds the MAP point η̂_d.
    Step (b): Analytic Hessian at η̂_d gives ν_d = (-H)⁻¹.

M-step:
    β:  natural-gradient SVI on λ                          # unchanged from LDA
    Γ:  ridge regression on aggregated XᵀX, Xᵀμ; ρ-blended # stochastic-EM
    Σ:  K-diagonal sample covariance of residuals + diag(ν_d); ρ-blended

References:
    Roberts, Stewart, Airoldi 2016. "A Model of Text for Experimentation in the
        Social Sciences." JASA.
    docs/superpowers/specs/2026-05-29-stm-prevalence-design.md
    docs/decisions/0023-stm-inference-two-step-laplace-stochastic-em.md
"""
from __future__ import annotations

from functools import partial
from typing import Any, Iterable

import numpy as np
from scipy.optimize import minimize
from scipy.special import digamma, gammaln

from spark_vi.core.model import VIModel
from spark_vi.models.topic.types import STMDocument


def _stm_neg_log_joint(
    eta: np.ndarray,
    *,
    indices: np.ndarray,
    counts: np.ndarray,
    expElogbeta: np.ndarray,
    Gamma: np.ndarray,
    Sigma_diag: np.ndarray,
    x: np.ndarray,
) -> float:
    """Negative log joint at η for a single doc.

    f(η) = -Σ_w n_dw · log(p^T expElogβ_·w) + ½(η - Γx)^T Σ⁻¹(η - Γx)
           where p = softmax(η)
    """
    # Data term — Jensen lower bound using expElogβ (matches LDA's phi_norm).
    p = _softmax(eta)
    eb_d = expElogbeta[:, indices]                         # (K, n_unique)
    q_w = eb_d.T @ p + 1e-100                              # (n_unique,)
    data_term = -float(np.sum(counts * np.log(q_w)))
    # Prior term — Gaussian with diagonal Σ.
    diff = eta - Gamma.T @ x                               # (K,)
    prior_term = 0.5 * float(np.sum(diff * diff / Sigma_diag))
    return data_term + prior_term


def _stm_neg_log_joint_grad(
    eta: np.ndarray,
    *,
    indices: np.ndarray,
    counts: np.ndarray,
    expElogbeta: np.ndarray,
    Gamma: np.ndarray,
    Sigma_diag: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Gradient of _stm_neg_log_joint at η.

    ∇f(η) = N_d · p - Σ_w n_dw · φ_w + Σ⁻¹(η - Γx)
            where N_d = Σ_w n_dw  and  φ_w = (p ⊙ β_·w) / (p^T β_·w)
    """
    p = _softmax(eta)
    eb_d = expElogbeta[:, indices]                         # (K, n_unique)
    q_w = eb_d.T @ p + 1e-100                              # (n_unique,)
    # phi_w (K, n_unique): φ_wk = p_k * expElogβ_kw / q_w. Per-token responsibility.
    phi = (eb_d * p[:, None]) / q_w[None, :]               # (K, n_unique)
    N_d = float(np.sum(counts))
    data_grad = N_d * p - phi @ counts                     # (K,)
    diff = eta - Gamma.T @ x                               # (K,)
    prior_grad = diff / Sigma_diag                         # (K,)
    return data_grad + prior_grad


def _softmax(eta: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    m = eta.max()
    exp = np.exp(eta - m)
    return exp / exp.sum()


def _stm_neg_log_joint_hessian(
    eta: np.ndarray,
    *,
    indices: np.ndarray,
    counts: np.ndarray,
    expElogbeta: np.ndarray,
    Gamma: np.ndarray,
    Sigma_diag: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Hessian of _stm_neg_log_joint at η.

    H(η) = N_d · (diag(p) - p p^T) - Σ_w n_dw · (diag(φ_w) - φ_w φ_w^T) + Σ⁻¹
    """
    K = eta.shape[0]
    p = _softmax(eta)
    eb_d = expElogbeta[:, indices]                         # (K, n_unique)
    q_w = eb_d.T @ p + 1e-100                              # (n_unique,)
    phi = (eb_d * p[:, None]) / q_w[None, :]               # (K, n_unique)
    N_d = float(np.sum(counts))

    # N_d · (diag(p) - p p^T)
    H_data_pos = N_d * (np.diag(p) - np.outer(p, p))
    # Σ_w n_dw · (diag(φ_w) - φ_w φ_w^T) — accumulate weighted by counts.
    # diag part: Σ_w n_dw · diag(φ_w) = diag(φ @ counts)
    diag_term = np.diag(phi @ counts)
    # outer part: Σ_w n_dw · φ_w φ_w^T = (φ * counts) @ φ.T
    outer_term = (phi * counts[None, :]) @ phi.T
    H_data_neg = diag_term - outer_term

    H_prior = np.diag(1.0 / Sigma_diag)
    return H_data_pos - H_data_neg + H_prior


def _stm_doc_inference(
    *,
    indices: np.ndarray,
    counts: np.ndarray,
    expElogbeta: np.ndarray,
    Gamma: np.ndarray,
    Sigma_diag: np.ndarray,
    x: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Per-doc Laplace approximation: L-BFGS for MAP, analytic Hessian for ν_d.

    Cold-start at η = 0 (uniform θ after softmax). Stateless across outer
    iterations — preserves the local_update contract that mini-batch
    sampling assumes (ADR 0023).

    Returns:
        eta_hat:  (K,) the MAP point.
        nu_d:     (K, K) Laplace covariance = (-H)⁻¹ at η̂.
        n_iter:   inner L-BFGS iterations consumed.
    """
    K = expElogbeta.shape[0]
    eta0 = np.zeros(K, dtype=np.float64)
    common = dict(
        indices=indices, counts=counts, expElogbeta=expElogbeta,
        Gamma=Gamma, Sigma_diag=Sigma_diag, x=x,
    )
    f = partial(_stm_neg_log_joint, **common)
    g = partial(_stm_neg_log_joint_grad, **common)
    result = minimize(f, x0=eta0, jac=g, method="L-BFGS-B",
                      options={"maxiter": max_iter, "gtol": tol})
    eta_hat = result.x
    H = _stm_neg_log_joint_hessian(eta_hat, **common)
    nu_d = np.linalg.inv(H)
    return eta_hat, nu_d, int(result.nit)
