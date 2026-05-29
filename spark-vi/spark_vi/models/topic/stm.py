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
    Step (b): Analytic Hessian of the negative log joint at η̂_d gives ν_d = H⁻¹.

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
        nu_d:     (K, K) Laplace covariance = H⁻¹ at η̂, where H is the Hessian
                  of the negative log joint (positive-definite).
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


class OnlineSTM(VIModel):
    """Online STM (prevalence-only) fittable by VIRunner.

    Per-doc inference is two-step Laplace (ADR 0023): L-BFGS to find η̂_d,
    analytic Hessian at η̂_d for ν_d. β stays Dirichlet-conjugate; Γ and Σ
    are hyperparameters of the prior on η, learned by closed-form M-step
    ρ-blended in mini-batch (stochastic-EM).

    random_seed controls the λ initialization. Per-doc inference is
    cold-started at η=0 deterministically regardless of seed.
    """

    def __init__(
        self,
        K: int,
        vocab_size: int,
        P: int,
        eta: float | None = None,
        sigma_init: float = 1.0,
        sigma_ridge: float = 1e-6,
        lbfgs_max_iter: int = 50,
        lbfgs_tol: float = 1e-4,
        gamma_shape: float = 100.0,
        random_seed: int | None = None,
    ) -> None:
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        if vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1, got {vocab_size}")
        if P < 1:
            raise ValueError(f"P must be >= 1, got {P}")
        if eta is None:
            eta = 1.0 / K
        if eta <= 0:
            raise ValueError(f"eta must be > 0, got {eta}")
        if sigma_init <= 0:
            raise ValueError(f"sigma_init must be > 0, got {sigma_init}")
        if sigma_ridge < 0:
            raise ValueError(f"sigma_ridge must be >= 0, got {sigma_ridge}")
        if lbfgs_max_iter < 1:
            raise ValueError(f"lbfgs_max_iter must be >= 1, got {lbfgs_max_iter}")
        if lbfgs_tol <= 0:
            raise ValueError(f"lbfgs_tol must be > 0, got {lbfgs_tol}")
        if gamma_shape <= 0:
            raise ValueError(f"gamma_shape must be > 0, got {gamma_shape}")

        self.K = int(K)
        self.V = int(vocab_size)
        self.P = int(P)
        self.eta = float(eta)
        self.sigma_init = float(sigma_init)
        self.sigma_ridge = float(sigma_ridge)
        self.lbfgs_max_iter = int(lbfgs_max_iter)
        self.lbfgs_tol = float(lbfgs_tol)
        self.gamma_shape = float(gamma_shape)
        self.random_seed = None if random_seed is None else int(random_seed)

    def initialize_global(self, data_summary: Any | None) -> dict[str, np.ndarray]:
        """Random Gamma init for λ (same shape as LDA); Γ = 0; Σ = sigma_init."""
        if self.random_seed is None:
            lam = np.random.gamma(
                shape=self.gamma_shape, scale=1.0 / self.gamma_shape,
                size=(self.K, self.V),
            )
        else:
            rng = np.random.default_rng(self.random_seed)
            lam = rng.gamma(
                shape=self.gamma_shape, scale=1.0 / self.gamma_shape,
                size=(self.K, self.V),
            )
        return {
            "lambda": lam,
            "eta": np.array(self.eta),
            "Gamma": np.zeros((self.P, self.K), dtype=np.float64),
            "Sigma": np.full(self.K, self.sigma_init, dtype=np.float64),
        }

    def get_metadata(self) -> dict[str, Any]:
        return {"K": self.K, "V": self.V, "P": self.P}

    def local_update(
        self,
        rows: Iterable[STMDocument],
        global_params: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """E-step on one Spark partition.

        For each doc: run _stm_doc_inference to get (η̂_d, ν_d). Accumulate:
        - lambda_stats: K×V suff-stats for β SVI step (same shape as LDA)
        - XtX, XtMu: P×P and P×K cross-products for Γ ridge regression
        - residual_diag_stat: K-vector for diagonal Σ sample covariance
        - doc_loglik_sum: data log-lik term at MAP (for ELBO)
        - doc_eta_kl_sum: KL(N(η̂, ν_d) || N(Γx, Σ)) (for ELBO)
        - n_docs scalar
        """
        lam = global_params["lambda"]
        Gamma = global_params["Gamma"]
        Sigma_diag = global_params["Sigma"]
        expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))

        K, V = self.K, self.V
        P = self.P

        lambda_stats = np.zeros((K, V), dtype=np.float64)
        XtX = np.zeros((P, P), dtype=np.float64)
        XtMu = np.zeros((P, K), dtype=np.float64)
        residual_diag = np.zeros(K, dtype=np.float64)
        doc_loglik = 0.0
        doc_eta_kl = 0.0
        n_docs = 0

        log_Sigma_diag = np.log(Sigma_diag)

        for doc in rows:
            eta_hat, nu_d, _ = _stm_doc_inference(
                indices=doc.indices, counts=doc.counts,
                expElogbeta=expElogbeta,
                Gamma=Gamma, Sigma_diag=Sigma_diag, x=doc.x,
                max_iter=self.lbfgs_max_iter, tol=self.lbfgs_tol,
            )
            p = _softmax(eta_hat)
            eb_d = expElogbeta[:, doc.indices]
            q_w = eb_d.T @ p + 1e-100
            phi = (eb_d * p[:, None]) / q_w[None, :]   # (K, n_unique)

            # λ suff-stats: phi · counts directly (phi already incorporates
            # expElogbeta; update_global skips the expElogbeta multiplication
            # for the STM path — see Task 5).
            sstats_row = phi * doc.counts[None, :]
            lambda_stats[:, doc.indices] += sstats_row

            # Regression sufficient stats.
            XtX += np.outer(doc.x, doc.x)
            XtMu += np.outer(doc.x, eta_hat)
            # Residual diag for Σ: (η̂ - Γx)² + diag(ν_d).
            resid = eta_hat - Gamma.T @ doc.x
            residual_diag += resid * resid + np.diag(nu_d)

            # ELBO terms.
            doc_loglik += float(np.sum(doc.counts * np.log(q_w)))
            # KL(N(η̂, ν_d) || N(Γx, Σ)) closed form with K-diagonal Σ:
            # ½(tr(Σ⁻¹ ν_d) + (η̂ - Γx)ᵀ Σ⁻¹ (η̂ - Γx) - K + log|Σ| - log|ν_d|)
            tr_term = float(np.sum(np.diag(nu_d) / Sigma_diag))
            quad_term = float(np.sum(resid * resid / Sigma_diag))
            sign, logdet_nu = np.linalg.slogdet(nu_d)
            logdet_Sigma = float(np.sum(log_Sigma_diag))
            doc_eta_kl += 0.5 * (tr_term + quad_term - K + logdet_Sigma - logdet_nu)

            n_docs += 1

        return {
            "lambda_stats": lambda_stats,
            "XtX": XtX,
            "XtMu": XtMu,
            "residual_diag_stat": residual_diag,
            "doc_loglik_sum": np.array(doc_loglik),
            "doc_eta_kl_sum": np.array(doc_eta_kl),
            "n_docs": np.array(float(n_docs)),
        }

    def update_global(
        self,
        global_params: dict[str, np.ndarray],
        target_stats: dict[str, np.ndarray],
        learning_rate: float,
    ) -> dict[str, np.ndarray]:
        """M-step: apply the natural-gradient update with stepsize rho_t. (Stub for Task 4.)"""
        raise NotImplementedError("update_global implemented in Task 4")
