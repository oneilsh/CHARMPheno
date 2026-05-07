"""Empirical-Bayes optimization steps for concentration hyperparameters.

Two families live here:

  - **Newton steps for Dirichlet concentrations** (LDA's α and η). The
    Dirichlet log-partition (gammaln of a sum) makes the ELBO non-quadratic
    in the concentration, so Newton iterates with a structured Hessian. α
    is asymmetric/vector with a diagonal-plus-rank-1 Hessian inverted in
    closed form via Sherman-Morrison (Blei 2003 A.2/A.4.2). η is the
    symmetric scalar case (Hoffman 2010 §3.4).

  - **Closed-form M-step for Beta(1, β) concentrations** (HDP's γ and α).
    The Beta(1, β) ELBO contribution is N·log β + (β−1)·S, linear in β
    after the log term — derivative N/β + S has a single root at
    β* = -N/S. No Hessian, no iteration. Caller applies ρ_t damping.

All helpers stay pure: they return raw steps / closed-form maximizers.
The caller applies SVI ρ_t damping and the post-step floor (typically
clip to [1e-3, ∞)) at the call site, which keeps the math testable
against synthetic data without simulating the full SVI loop.

References:
  Blei, Ng, Jordan 2003. Latent Dirichlet Allocation. JMLR.
  Hoffman, Blei, Bach 2010. Online learning for LDA. NIPS.
  Blei, Jordan 2006. Variational inference for Dirichlet process mixtures.
"""
from __future__ import annotations

import numpy as np
from scipy.special import digamma, polygamma


def alpha_newton_step(
    alpha: np.ndarray,
    e_log_theta_sum_scaled: np.ndarray,
    D: float,
) -> np.ndarray:
    """One Newton step for asymmetric Dirichlet α (LDA).

    Per Blei, Ng, Jordan 2003 Appendix A.4.2 (which applies the linear-time
    structured-Hessian Newton from Appendix A.2). The ELBO part depending on α is
        L(α) = D · [log Γ(Σ_k α_k) − Σ_k log Γ(α_k)]
             + Σ_d Σ_k (α_k − 1) E[log θ_dk]
    with gradient
        g_k = D · [ψ(Σ_j α_j) − ψ(α_k)] + Σ_d E[log θ_dk]
    and Hessian (diagonal-plus-rank-1)
        H = c · 1·1ᵀ − diag(d_k)
    where c = D · ψ′(Σα), d_k = D · ψ′(α_k).

    The matrix-inversion lemma applied to this structured Hessian (Blei A.2,
    eq. 10) gives Δα = −H⁻¹·g in closed-form O(K):
        Δα_k = (g_k − b) / d_k
        b    = Σ_j(g_j/d_j) / (Σ_j 1/d_j − 1/c)
    (Note: the Hessian formula in Blei A.4.2's printed text has a transcription
    sign error — the corrected derivation gives the negative-definite H above,
    and matches MLlib's OnlineLDAOptimizer.updateAlpha.)

    Inputs:
      alpha:                    (K,) current α vector.
      e_log_theta_sum_scaled:   (K,) corpus-scaled Σ_d E[log θ_dk]
                                (i.e. batch sum × D / |batch|).
      D:                        corpus size (used in g, c, d_k).

    Returns:
      Δα: (K,) raw Newton step. Caller does the ρ_t damping and the
        post-step floor — keeping this function pure makes it trivial
        to unit-test against synthetic data.
    """
    alpha_sum = alpha.sum()
    g = D * (digamma(alpha_sum) - digamma(alpha)) + e_log_theta_sum_scaled
    c = D * polygamma(1, alpha_sum)
    d = D * polygamma(1, alpha)

    sum_g_over_d = (g / d).sum()
    sum_1_over_d = (1.0 / d).sum()
    b = sum_g_over_d / (sum_1_over_d - 1.0 / c)

    return (g - b) / d


def eta_newton_step(
    eta: float,
    e_log_phi_sum: float,
    K: int,
    V: int,
) -> float:
    """One Newton step for symmetric scalar Dirichlet η (LDA topic-word).

    Per Hoffman, Blei, Bach 2010 §3.4. The ELBO part depending on η is
        L(η) = K · log Γ(V·η) − K·V · log Γ(η)
             + (η − 1) · Σ_t Σ_v E[log φ_tv]
    with scalar gradient and Hessian
        g(η) = K·V · [ψ(V·η) − ψ(η)] + Σ_t Σ_v E[log φ_tv]
        H(η) = K·V² · ψ′(V·η) − K·V · ψ′(η)
    Newton step Δη = −g/H.

    Reusable across models — HDP's η on Dirichlet(η · 1_V) topic-word
    prior has the exact same form (T topics instead of K, otherwise
    identical).

    Inputs:
      eta:           current η scalar.
      e_log_phi_sum: Σ_t Σ_v E[log φ_tv] from current λ (typically
                     (digamma(lam) − digamma(lam.sum(axis=1, keepdims=True))).sum()).
                     NOTE: K, V are the topic / vocab dimensions of the model,
                     NOT scale factors — η does not depend on corpus size D.
      K:             number of topics.
      V:             vocabulary size.

    Returns:
      Δη: raw Newton step (float). Caller applies ρ_t damping and the
        post-step floor (parallel to alpha_newton_step's caller contract).
    """
    g = K * V * (digamma(V * eta) - digamma(eta)) + e_log_phi_sum
    h = K * V * V * polygamma(1, V * eta) - K * V * polygamma(1, eta)
    return -g / h


def beta_concentration_closed_form(
    n: float,
    s_log_one_minus: float,
) -> float:
    """Closed-form maximizer of the Beta(1, β) concentration likelihood.

    Used for any scalar concentration parameter on a `Beta(1, β)` prior over
    independent stick-breaking factors — HDP's γ (corpus stick) and α (doc
    stick) are both this shape.

    Setup. Each Beta(1, β) factor contributes
        log p(W | β)  =  log β  +  (β − 1) · log(1 − W)
    to the prior. Summed across N independent factors and taken as the
    expectation under the variational posterior q(W),
        L(β)  =  N · log β  +  (β − 1) · S       where  S  =  Σ E_q[log(1 − W)]
    which is the only β-dependent term in the ELBO. (The −β prefactor on
    the log-(1−W) term and the +log β prefactor are the only pieces not
    constant in β; the rest of Beta(1, β)'s normalizing constant is β-free
    because Γ(1) = 1.)

    Derivative:    L′(β)  =  N/β  +  S
    Setting to 0:  β*     =  −N / S

    S is always negative (sum of logs of values in (0, 1)), and N > 0, so
    β* > 0 is automatic. The second derivative L″(β) = −N/β² < 0 confirms
    β* is the unique maximum. No Hessian inversion, no iteration.

    Why this works for HDP but not LDA's α: LDA's α prior is a
    *Dirichlet*, whose log-partition function is gammaln(Σ α_k) — the sum
    inside gammaln makes the ELBO non-quadratic in α and forces Newton.
    HDP's γ and α are scalar Beta(1, β) concentrations, whose log-partition
    in β is just log β — quadratic enough that the M-step is closed.

    Use sites in HDP:
      - γ (corpus stick): N = T − 1, S = Σ_t [ψ(v_t) − ψ(u_t + v_t)]
        from the corpus Beta posteriors q(W_t) = Beta(u_t, v_t).
      - α (doc stick):    N = D · (K − 1), S = Σ_d Σ_k [ψ(b_jk) − ψ(a_jk + b_jk)]
        accumulated across the (corpus-scaled) minibatch from q(V_jk) =
        Beta(a_jk, b_jk).

    Caller applies ρ_t damping (β_new = (1−ρ)·β_old + ρ·β*) and the
    post-step floor (clip to ≥ 1e-3) at the call site.

    Reference:
      Blei, Jordan 2006. Variational inference for Dirichlet process
      mixtures. Bayesian Analysis. (Same closed form for the DP-mixture
      stick concentration; HDP inherits the structure.)
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    if s_log_one_minus >= 0:
        raise ValueError(
            f"s_log_one_minus must be < 0 (sum of log(1−W) over W ∈ (0,1)), "
            f"got {s_log_one_minus}"
        )
    return -n / s_log_one_minus
