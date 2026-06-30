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
    q(η_d)  = N(μ_d, ν_d) with full (K,K) ν_d              # Laplace approximation
    q(z_dn) = collapsed via Lee/Seung trick                # unchanged from LDA

Per-doc inference is two-step Laplace (ADR 0023):
    Step (a): L-BFGS finds the MAP point η̂_d.
    Step (b): Analytic Hessian of the negative log joint at η̂_d gives ν_d = H⁻¹.

M-step:
    β:  natural-gradient SVI on λ                          # unchanged from LDA
    Γ:  ridge regression on aggregated XᵀX, Xᵀμ; ρ-blended # stochastic-EM
    Σ:  full (K,K) sample covariance of residuals + ν_d scatter; ρ-blended

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
from spark_vi.models.topic._linalg import nearest_spd, safe_inverse
from spark_vi.models.topic.types import STMDocument


def _softmax(eta: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    m = eta.max()
    exp = np.exp(eta - m)
    return exp / exp.sum()


def prior_topic_proportions(Gamma: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Covariate-implied (prior) topic proportions for one document: softmax(Γᵀ x).

    Γᵀx is the prior mean of η_d under the logistic-normal prior
    η_d ~ N(Γᵀ x_d, Σ); pushing it through softmax gives the topic mix the
    document is expected to have from its covariates alone, before any token
    evidence. Averaging this over the corpus yields the α-equivalent
    ``(1/D) Σ_d softmax(Γᵀ x_d)`` the dashboard reports as the default topic
    proportion.

    Γ is (P, K), x is (P,); the result is a length-K probability vector.
    """
    return _softmax(Gamma.T @ x)


def corpus_mean_topic_proportions(Gamma: np.ndarray, X: np.ndarray) -> np.ndarray:
    """α-equivalent: mean over the corpus of per-doc prior proportions.

    Computes ``(1/D) Σ_d softmax(Γᵀ x_d)`` for the design matrix X (D, P).
    Because softmax is nonlinear, this is NOT softmax of the mean covariate
    row — every document's covariates must be pushed through individually and
    then averaged. Returns a length-K probability vector (the mean of
    probability vectors is itself a probability vector).
    """
    eta = X @ Gamma                      # (D, K): row d is Γᵀ x_d
    m = eta.max(axis=1, keepdims=True)
    exp = np.exp(eta - m)
    proportions = exp / exp.sum(axis=1, keepdims=True)
    return proportions.mean(axis=0)


def corpus_mean_topic_proportions_gated(Gamma, X, groups_per_doc, partition):
    """Gating-aware corpus-mean prior proportions: (1/D) Σ_d softmax_allowed(Γᵀ x_d).

    For each document, the softmax is taken over that document's ALLOWED topics
    only (background ∪ its group's foreground, per partition.allowed_indices);
    disallowed topics are exactly 0. So a foreground topic's corpus-mean
    prevalence reflects only its group's share. Γ is (P, K), X is (D, P),
    groups_per_doc is a length-D sequence of frozenset[str].
    """
    import numpy as np
    Gamma = np.asarray(Gamma, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    K = Gamma.shape[1]
    eta = X @ Gamma                                   # (D, K)
    acc = np.zeros(K, dtype=np.float64)
    for d in range(X.shape[0]):
        allowed = partition.allowed_indices(groups_per_doc[d])
        e = eta[d, allowed]
        e = e - e.max()
        p = np.exp(e)
        p = p / p.sum()
        acc[allowed] += p
    return acc / max(X.shape[0], 1)


def _stm_neg_log_joint(
    eta: np.ndarray,
    *,
    indices: np.ndarray,
    counts: np.ndarray,
    expElogbeta: np.ndarray,
    Gamma: np.ndarray,
    Sigma_inv: np.ndarray,
    x: np.ndarray,
) -> float:
    """Negative log joint at η for a single doc.

    f(η) = -Σ_w n_dw · log(p^T expElogβ_·w) + ½(η - Γx)^T Σ⁻¹(η - Γx)
           where p = softmax(η) and Σ⁻¹ is the full (n×n) precision matrix
    """
    # Data term — Jensen lower bound using expElogβ (matches LDA's phi_norm).
    p = _softmax(eta)
    eb_d = expElogbeta[:, indices]                         # (K, n_unique)
    q_w = eb_d.T @ p + 1e-100                              # (n_unique,)
    data_term = -float(np.sum(counts * np.log(q_w)))
    # Prior term — Gaussian with full precision matrix Σ⁻¹.
    diff = eta - Gamma.T @ x                               # (K,)
    prior_term = 0.5 * float(diff @ Sigma_inv @ diff)
    return data_term + prior_term


def _stm_neg_log_joint_grad(
    eta: np.ndarray,
    *,
    indices: np.ndarray,
    counts: np.ndarray,
    expElogbeta: np.ndarray,
    Gamma: np.ndarray,
    Sigma_inv: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Gradient of _stm_neg_log_joint at η.

    ∇f(η) = N_d · p - Σ_w n_dw · φ_w + Σ⁻¹(η - Γx)
            where N_d = Σ_w n_dw, φ_w = (p ⊙ β_·w) / (p^T β_·w),
            and Σ⁻¹ is the full (n×n) precision matrix
    """
    p = _softmax(eta)
    eb_d = expElogbeta[:, indices]                         # (K, n_unique)
    q_w = eb_d.T @ p + 1e-100                              # (n_unique,)
    # phi_w (K, n_unique): φ_wk = p_k * expElogβ_kw / q_w. Per-token responsibility.
    phi = (eb_d * p[:, None]) / q_w[None, :]               # (K, n_unique)
    N_d = float(np.sum(counts))
    data_grad = N_d * p - phi @ counts                     # (K,)
    diff = eta - Gamma.T @ x                               # (K,)
    prior_grad = Sigma_inv @ diff                          # (K,)
    return data_grad + prior_grad


def _stm_neg_log_joint_hessian(
    eta: np.ndarray,
    *,
    indices: np.ndarray,
    counts: np.ndarray,
    expElogbeta: np.ndarray,
    Gamma: np.ndarray,
    Sigma_inv: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Hessian of _stm_neg_log_joint at η.

    H(η) = N_d · (diag(p) - p p^T) - Σ_w n_dw · (diag(φ_w) - φ_w φ_w^T) + Σ⁻¹
           where Σ⁻¹ is the full (n×n) precision matrix
    """
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

    H_prior = Sigma_inv
    return H_data_pos - H_data_neg + H_prior


# Condition-number cap used only when a non-PD Hessian must be repaired before
# inversion (see _spd_inverse). Caps the inverse's largest eigenvalue at
# 1 / (cap * lambda_max), bounding variance in flat/indefinite directions.
_HESSIAN_COND_CAP = 1e-10


def _spd_inverse(H: np.ndarray) -> np.ndarray:
    """Inverse of a Hessian that is *meant* to be positive-definite.

    The per-doc neg-log-joint is NOT globally convex — its data term is a
    difference of two log-sum-exp functions — so the Hessian H at the L-BFGS
    point can fail to be positive-definite when the prior is weak (large Σ) or
    L-BFGS stopped short of the true mode. A raw np.linalg.inv would then return
    a non-SPD "covariance" with negative variances, silently corrupting the
    residual_outer_stat scatter (which adds the ν_d sub-block) and the Gaussian
    KL (whose slogdet would flip sign).

    Fast path: if H is PD (Cholesky succeeds) return inv(H) unchanged — the
    overwhelmingly common case, bit-for-bit identical to the prior code.
    Repair path: floor H's eigenvalues at a small positive value (condition-
    number cap) and rebuild the inverse, guaranteeing an SPD result with
    bounded variance. Mirrors the Hessian "nugget" the reference stm R package
    adds before inversion.
    """
    try:
        np.linalg.cholesky(H)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(0.5 * (H + H.T))
        floor = max(w.max() * _HESSIAN_COND_CAP, 1e-12)
        w = np.maximum(w, floor)
        return (V * (1.0 / w)) @ V.T
    return np.linalg.inv(H)


def _stm_doc_inference(
    *,
    indices: np.ndarray,
    counts: np.ndarray,
    expElogbeta: np.ndarray,
    Gamma: np.ndarray,
    Sigma_inv: np.ndarray,
    x: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-4,
    allowed: np.ndarray | None = None,
    reference: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Per-doc Laplace approximation, optionally restricted to an allowed topic
    set and optionally with one topic pinned to eta=0 (the reference).

    When allowed is None, optimizes over all K topics. When allowed is a sorted
    index array, L-BFGS runs only on those topics; disallowed topics are filled
    with eta=-inf (theta exactly 0) and nu_d=0.

    When reference is a topic id (which must be in allowed), that topic's eta is
    held at 0 and only the other allowed topics are optimized. softmax([nu, 0])
    removes the translation degeneracy. Fixing a coordinate to a constant makes
    the reduced gradient/Hessian the corresponding sub-block of the full ones,
    so we reuse the existing grad/Hessian and just delete the reference row/col.
    The reference stays a real topic: its exp(0)=1 is in the softmax denominator
    and it still contributes to the data term. In the returned arrays the
    reference has eta_hat=0 (finite) and nu_d row/col exactly 0 (it is pinned, so
    it carries no posterior variance).
    """
    K = expElogbeta.shape[0]
    if allowed is None:
        allowed = np.arange(K, dtype=np.int64)
    sub_expElogbeta = expElogbeta[allowed]
    sub_Gamma = Gamma[:, allowed]
    # NOTE(Task 5): replace precision-slice with marginal sub-block inv(Sigma_AA).
    # Slicing the full precision is the *conditional* (gated) form; for non-gated
    # `allowed` is all topics, so the slice is the full precision unchanged. The
    # gated marginal-vs-conditional correction is Task 5.
    sub_Sigma_inv = Sigma_inv[np.ix_(allowed, allowed)]
    n_sub = allowed.shape[0]
    common = dict(
        indices=indices, counts=counts, expElogbeta=sub_expElogbeta,
        Gamma=sub_Gamma, Sigma_inv=sub_Sigma_inv, x=x,
    )

    if reference is None:
        # Canonical path — unchanged from before.
        eta0 = np.zeros(n_sub, dtype=np.float64)
        f = partial(_stm_neg_log_joint, **common)
        g = partial(_stm_neg_log_joint_grad, **common)
        result = minimize(f, x0=eta0, jac=g, method="L-BFGS-B",
                          options={"maxiter": max_iter, "gtol": tol})
        sub_eta = result.x
        H = _stm_neg_log_joint_hessian(sub_eta, **common)
        sub_nu = _spd_inverse(H)
        eta_hat = np.full(K, -np.inf, dtype=np.float64)
        eta_hat[allowed] = sub_eta
        nu_d = np.zeros((K, K), dtype=np.float64)
        nu_d[np.ix_(allowed, allowed)] = sub_nu
        return eta_hat, nu_d, int(result.nit)

    # Reference parameterization: pin `reference` at eta=0, optimize the rest.
    if reference not in allowed:
        raise ValueError(
            f"reference={reference} is not in allowed={list(allowed)}; "
            "the reference topic must be a member of the allowed set"
        )
    ref_pos = int(np.searchsorted(allowed, reference))
    free = np.array([i for i in range(n_sub) if i != ref_pos], dtype=np.int64)

    def _full(nu_free: np.ndarray) -> np.ndarray:
        eta_sub = np.zeros(n_sub, dtype=np.float64)   # reference position stays 0
        eta_sub[free] = nu_free
        return eta_sub

    def f_free(nu_free: np.ndarray) -> float:
        return _stm_neg_log_joint(_full(nu_free), **common)

    def g_free(nu_free: np.ndarray) -> np.ndarray:
        return _stm_neg_log_joint_grad(_full(nu_free), **common)[free]

    result = minimize(f_free, x0=np.zeros(free.shape[0], dtype=np.float64),
                      jac=g_free, method="L-BFGS-B",
                      options={"maxiter": max_iter, "gtol": tol})
    eta_sub = _full(result.x)
    H_full = _stm_neg_log_joint_hessian(eta_sub, **common)   # (n_sub, n_sub)
    H_free = H_full[np.ix_(free, free)]
    sub_nu_free = _spd_inverse(H_free)

    eta_hat = np.full(K, -np.inf, dtype=np.float64)
    eta_hat[allowed] = eta_sub                  # reference -> 0, free -> nu
    nu_d = np.zeros((K, K), dtype=np.float64)
    free_topics = allowed[free]
    nu_d[np.ix_(free_topics, free_topics)] = sub_nu_free
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
        sigma_prior_scale: float | None = None,
        sigma_prior_count: float = 0.0,
        lbfgs_max_iter: int = 50,
        lbfgs_tol: float = 1e-4,
        gamma_shape: float = 100.0,
        random_seed: int | None = None,
        topic_blocks=None,
        reference_topic: bool = True,  # default-on (validated, insight 0030); pass False for the legacy full-K path
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
        if sigma_prior_scale is not None and sigma_prior_scale <= 0:
            raise ValueError(f"sigma_prior_scale must be > 0, got {sigma_prior_scale}")
        if sigma_prior_count < 0:
            raise ValueError(f"sigma_prior_count must be >= 0, got {sigma_prior_count}")
        if lbfgs_max_iter < 1:
            raise ValueError(f"lbfgs_max_iter must be >= 1, got {lbfgs_max_iter}")
        if lbfgs_tol <= 0:
            raise ValueError(f"lbfgs_tol must be > 0, got {lbfgs_tol}")
        if gamma_shape <= 0:
            raise ValueError(f"gamma_shape must be > 0, got {gamma_shape}")
        if topic_blocks is not None and topic_blocks.K != int(K):
            raise ValueError(
                f"topic_blocks.K ({topic_blocks.K}) != K ({K})")
        if reference_topic and K < 2:
            raise ValueError(
                f"reference_topic requires K >= 2 (need a free topic besides "
                f"the reference), got K={K}")

        self.K = int(K)
        self.V = int(vocab_size)
        self.P = int(P)
        self.eta = float(eta)
        self.sigma_init = float(sigma_init)
        self.sigma_ridge = float(sigma_ridge)
        self.sigma_prior_scale = None if sigma_prior_scale is None else float(sigma_prior_scale)
        self.sigma_prior_count = float(sigma_prior_count)
        self.lbfgs_max_iter = int(lbfgs_max_iter)
        self.lbfgs_tol = float(lbfgs_tol)
        self.gamma_shape = float(gamma_shape)
        self.random_seed = None if random_seed is None else int(random_seed)
        self.topic_blocks = topic_blocks
        self.reference_topic = bool(reference_topic)

    def _effective_partition(self):
        """The real partition, or an implicit all-background one when None."""
        if self.topic_blocks is not None:
            return self.topic_blocks
        from spark_vi.models.topic.partition import TopicBlockPartition
        return TopicBlockPartition(group_var="", background_k=self.K, foreground=())

    def _reference_index(self) -> int | None:
        """Global topic id held at eta=0 when reference_topic is on, else None.

        Topic 0 is always the first background topic (TopicBlockPartition lays
        background out first and enforces background_k >= 1), so it is in EVERY
        document's allowed set — the only place a single global reference can
        live when Gamma/Sigma are shared across docs. softmax(eta) is invariant
        to adding a constant to all coordinates; pinning one to 0 removes that
        translation degeneracy, which otherwise lets eta drift to the
        softmax-saturation boundary and Sigma blow up (insight 0029).
        """
        return 0 if self.reference_topic else None

    def initialize_global(self, data_summary: Any | None) -> dict[str, np.ndarray]:
        """Init λ; Γ = 0; Σ = sigma_init.

        Default (data_summary is None): random-gamma λ exactly as LDA — left
        byte-for-byte unchanged so the existing suite stays green.

        Opt-in spectral init: when data_summary carries a "spectral_beta" KxV
        topic-word matrix (from spark_vi.models.topic.spectral_init), seed
        λ = spectral_beta * gamma_shape instead of random gamma. This makes the
        β posterior start at a deterministic, data-driven anchor-word estimate,
        curing the sigma_init-dependent collapse/blow-up of random init
        (insight 0029). Γ and Σ are untouched (Γ = 0, Σ = sigma_init).
        """
        if data_summary is not None and "spectral_beta" in data_summary:
            beta0 = np.asarray(data_summary["spectral_beta"], dtype=np.float64)
            return {
                "lambda": beta0 * self.gamma_shape,
                "eta": np.array(self.eta),
                "Gamma": np.zeros((self.P, self.K), dtype=np.float64),
                "Sigma": np.eye(self.K, dtype=np.float64) * self.sigma_init,
            }
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
            "Sigma": np.eye(self.K, dtype=np.float64) * self.sigma_init,
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
        - residual_outer_stat: (K,K) outer-product scatter for full Σ sample cov
        - n_pairs_stat: (K,K) per-pair support count (docs where both topics free)
        - n_docs_per_topic: K-vector per-topic doc count (drives the Γ lazy rule)
        - doc_loglik_sum: data log-lik term at MAP (for ELBO)
        - doc_eta_kl_sum: KL(N(η̂, ν_d) || N(Γx, Σ)) (for ELBO)
        - n_docs scalar
        """
        lam = global_params["lambda"]
        Gamma = global_params["Gamma"]
        Sigma = global_params["Sigma"]
        Sigma_inv = safe_inverse(Sigma)
        expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))

        K, V = self.K, self.V
        P = self.P

        part = self._effective_partition()
        G = len(part.foreground)
        group_order = part.groups  # tuple of labels in block order

        lambda_stats = np.zeros((K, V), dtype=np.float64)
        XtX = np.zeros((P, P), dtype=np.float64)            # all-doc cross-product
        XtX_groups = np.zeros((G, P, P), dtype=np.float64)  # per-group cross-product
        XtMu = np.zeros((P, K), dtype=np.float64)
        residual_outer = np.zeros((K, K), dtype=np.float64)
        n_pairs = np.zeros((K, K), dtype=np.float64)
        n_docs_per_topic = np.zeros(K, dtype=np.float64)
        doc_loglik = 0.0
        doc_eta_kl = 0.0
        n_docs = 0

        ref = self._reference_index()

        for doc in rows:
            allowed = part.allowed_indices(doc.groups)
            eta_hat, nu_d, _ = _stm_doc_inference(
                indices=doc.indices, counts=doc.counts,
                expElogbeta=expElogbeta,
                Gamma=Gamma, Sigma_inv=Sigma_inv, x=doc.x,
                max_iter=self.lbfgs_max_iter, tol=self.lbfgs_tol,
                allowed=allowed, reference=ref,
            )
            p = _softmax(eta_hat)  # 0 on disallowed
            eb_d = expElogbeta[:, doc.indices]
            q_w = eb_d.T @ p + 1e-100
            phi = (eb_d * p[:, None]) / q_w[None, :]   # (K, n_unique); 0 on disallowed rows
            sstats_row = phi * doc.counts[None, :]
            lambda_stats[:, doc.indices] += sstats_row

            xxT = np.outer(doc.x, doc.x)
            XtX += xxT
            for gi, g in enumerate(group_order):
                if g in doc.groups:
                    XtX_groups[gi] += xxT

            # Prior-side topics for this doc. With a reference topic, exclude it:
            # it is pinned at eta=0, carries no free Gamma column or Sigma entry,
            # and must stay out of the Gamma-regression targets, the Sigma
            # residual, the per-topic doc counts, and the eta KL. Dropping it
            # from XtMu makes its Gamma solve resolve to 0; dropping it from
            # n_docs_per_topic makes update_global's lazy rule leave Sigma[ref]
            # at sigma_init. When ref is None this is exactly `allowed`, so the
            # canonical path is byte-identical.
            if ref is None:
                allowed_free = allowed
            else:
                allowed_free = allowed[allowed != ref]

            # XtMu / residual scatter / counts only over the free prior topics.
            af = allowed_free
            eta_allowed = eta_hat[af]
            XtMu[:, af] += np.outer(doc.x, eta_allowed)
            # resid is the dense over-`af` residual vector (length len(af)).
            resid = eta_allowed - (Gamma.T @ doc.x)[af]
            # Full residual outer-product scatter + the Laplace covariance ν_d
            # sub-block, accumulated into the K×K stat over the free pairs.
            contrib = np.outer(resid, resid) + nu_d[np.ix_(af, af)]
            residual_outer[np.ix_(af, af)] += contrib
            n_pairs[np.ix_(af, af)] += 1.0
            n_docs_per_topic[af] += 1.0

            doc_loglik += float(np.sum(doc.counts * np.log(q_w)))
            # KL(N(η̂, ν_d) || N(Γx, Σ)) over the free prior sub-space only, using
            # the marginal sub-block of Σ over `af` (full-matrix form).
            sub_Sigma = Sigma[np.ix_(af, af)]
            sub_Sigma_inv = safe_inverse(sub_Sigma)
            sub_nu = nu_d[np.ix_(af, af)]
            tr_term = float(np.trace(sub_Sigma_inv @ sub_nu))
            quad_term = float(resid @ sub_Sigma_inv @ resid)
            # sub_nu is SPD by construction (_spd_inverse), so slogdet sign is +1.
            _s, logdet_nu = np.linalg.slogdet(sub_nu)
            _s2, logdet_Sigma = np.linalg.slogdet(sub_Sigma)
            doc_eta_kl += 0.5 * (tr_term + quad_term - len(af) + logdet_Sigma - logdet_nu)
            n_docs += 1

        return {
            "lambda_stats": lambda_stats,
            "XtX": XtX,
            "XtX_groups": XtX_groups,
            "XtMu": XtMu,
            "residual_outer_stat": residual_outer,
            "n_pairs_stat": n_pairs,
            "n_docs_per_topic": n_docs_per_topic,
            "doc_loglik_sum": np.array(doc_loglik),
            "doc_eta_kl_sum": np.array(doc_eta_kl),
            "n_docs": np.array(float(n_docs)),
        }

    SIGMA_FLOOR = 1e-6

    def update_global(
        self,
        global_params: dict[str, np.ndarray],
        target_stats: dict[str, np.ndarray],
        learning_rate: float,
    ) -> dict[str, np.ndarray]:
        """SVI on λ; ρ-blended closed-form M-step on Γ, Σ (ADR 0023).

        λ:  λ_new = (1-ρ)·λ + ρ·(η + lambda_stats)        # natural-gradient SVI
        Γ:  Γ̂ = (XᵀX + ridge·I)⁻¹ XᵀMu                    # ridge OLS on aggregated stats
            Γ_new = (1-ρ)·Γ + ρ·Γ̂                         # stochastic-EM ρ-blend
        Σ:  Σ_target[i,j] = residual_outer_stat[i,j] / n_pairs[i,j]   # full (K,K) sample cov
            Σ_new = nearest_spd((1-ρ)·Σ + ρ·Σ_target + ridge·I, floor=SIGMA_FLOOR)

        The Σ M-step is the plain full-covariance MLE (NO regularizer this task —
        the inverse-Wishart prior + diagonal shrink are Task 4). The per-pair MLE
        is lazy: a pair (i,j) with zero support (no doc had both i,j free this
        batch) keeps its current Σ value, so absent blocks never decay. After the
        ρ-blend, nearest_spd repairs any loss of positive-definiteness from
        stitching the scatter over inconsistent doc subsets.

        Lazy block updates (ADR 0027): any block (background or a group's
        foreground) with zero documents in this minibatch is left unchanged —
        its Γ columns and Σ entries skip the ρ-blend. This keeps a rare group's
        parameters from decaying toward (Γ=0, Σ=floor) in minibatches that miss
        the group. Present blocks and the no-gating path are unaffected.
        """
        lam = global_params["lambda"]
        eta = float(global_params["eta"])
        Gamma = global_params["Gamma"]
        Sigma = global_params["Sigma"]

        # β: SVI natural-gradient step. Note: STM's lambda_stats already
        # incorporates expElogbeta (via phi in local_update), so no extra
        # expElogbeta multiplication here — differs from LDA.
        target_lam = eta + target_stats["lambda_stats"]
        new_lam = (1.0 - learning_rate) * lam + learning_rate * target_lam

        # Γ: block-aware ridge regression on aggregated cross-products.
        part = self._effective_partition()
        XtX = target_stats["XtX"]
        XtX_groups = target_stats["XtX_groups"]
        XtMu = target_stats["XtMu"]
        n_docs_per_topic = target_stats["n_docs_per_topic"]
        ridge_eye = self.sigma_ridge * np.eye(self.P)

        # Lazy block updates (ADR 0027): a block whose group has NO documents in
        # this minibatch carries zero information about its Γ/Σ, so it is left
        # untouched. Defaulting each block's target to the *current* value makes
        # the ρ-blend a no-op for absent blocks (and skips a would-be-singular
        # solve over a zero XtX_groups). Present-block targets are
        # self-normalizing ratios, so they are unaffected; the canonical
        # no-gating and all-groups-present paths are numerically identical to
        # before. This removes the rare-group decay-toward-(Γ=0, Σ=floor) bias
        # that minibatches missing the group would otherwise inflict.
        Gamma_target = Gamma.copy()
        bg = part.background_indices()
        if n_docs_per_topic[bg].any():                # background present this batch
            Gamma_target[:, bg] = np.linalg.solve(XtX + ridge_eye, XtMu[:, bg])
        for gi, g in enumerate(part.groups):
            cols = part.block_indices(g)
            if n_docs_per_topic[cols].any():          # this group present this batch
                Gamma_target[:, cols] = np.linalg.solve(
                    XtX_groups[gi] + ridge_eye, XtMu[:, cols])
        new_Gamma = (1.0 - learning_rate) * Gamma + learning_rate * Gamma_target

        # Σ: full-covariance per-pair MLE. Each entry Σ[i,j] is the mean over the
        # docs where both i and j were free of (resid_i·resid_j + ν_d[i,j]). A
        # pair with no support keeps its current value (lazy rule: target defaults
        # to the current Σ, so the ρ-blend is a no-op there).
        S = target_stats["residual_outer_stat"]
        N = target_stats["n_pairs_stat"]
        Sigma_target = Sigma.copy()
        with np.errstate(invalid="ignore", divide="ignore"):
            mle = np.where(N > 0, S / np.where(N > 0, N, 1.0), Sigma)
        present_pairs = N > 0
        Sigma_target[present_pairs] = mle[present_pairs]
        new_Sigma = (1.0 - learning_rate) * Sigma + learning_rate * Sigma_target
        new_Sigma = nearest_spd(new_Sigma + self.sigma_ridge * np.eye(self.K),
                                floor=self.SIGMA_FLOOR)

        return {
            "lambda": new_lam,
            "eta": global_params["eta"],
            "Gamma": new_Gamma,
            "Sigma": new_Sigma,
        }

    def compute_elbo(
        self,
        global_params: dict[str, np.ndarray],
        aggregated_stats: dict[str, np.ndarray],
    ) -> float:
        """ELBO = doc_loglik_sum - doc_eta_kl_sum - global β KL.

        doc_loglik_sum and doc_eta_kl_sum are aggregated in local_update;
        the global β KL is computed here on the driver from λ, η alone
        (same pattern as OnlineLDA).
        """
        lam = global_params["lambda"]
        eta = float(global_params["eta"])
        K, V = lam.shape
        eta_vec = np.full(V, eta, dtype=np.float64)
        global_kl = 0.0
        for k in range(K):
            global_kl += _dirichlet_kl(lam[k], eta_vec)
        return float(
            float(aggregated_stats["doc_loglik_sum"])
            - float(aggregated_stats["doc_eta_kl_sum"])
            - global_kl
        )


    def infer_local(self, row: STMDocument, global_params: dict[str, np.ndarray]):
        """Single-document MAP inference under fixed global params.

        Pure function of (row, global_params). Uses the same cold-start
        L-BFGS as local_update; returns eta (the MAP point) and theta
        (softmax of eta) for downstream consumers.
        """
        lam = global_params["lambda"]
        Gamma = global_params["Gamma"]
        Sigma_inv = safe_inverse(global_params["Sigma"])
        expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))

        eta_hat, _, _ = _stm_doc_inference(
            indices=row.indices, counts=row.counts,
            expElogbeta=expElogbeta,
            Gamma=Gamma, Sigma_inv=Sigma_inv, x=row.x,
            max_iter=self.lbfgs_max_iter, tol=self.lbfgs_tol,
            reference=self._reference_index(),
        )
        return {"eta": eta_hat, "theta": _softmax(eta_hat)}

    def iteration_summary(self, global_params: dict[str, np.ndarray]) -> str:
        """Compact per-iter view of Γ scale, Σ scale, and λ row-mass spread.

        When topic_blocks is set, appends per-block Σλ mass so operators can
        watch foreground vs background vocabulary absorption separately.
        """
        Gamma = global_params["Gamma"]
        Sigma = global_params["Sigma"]
        sigma_var = np.diag(Sigma)  # per-topic variances (Σ is now (K,K))
        lam = global_params["lambda"]
        lam_row_sums = lam.sum(axis=1)
        base = (
            f"|Γ|[max={np.abs(Gamma).max():.3g} mean={np.abs(Gamma).mean():.3g}], "
            f"Σ[min={sigma_var.min():.3g} max={sigma_var.max():.3g}], "
            f"Σλ_k[min={lam_row_sums.min():.3g} max={lam_row_sums.max():.3g}]"
        )
        if self.topic_blocks is None:
            return base
        part = self.topic_blocks
        bg_mass = float(lam_row_sums[part.background_indices()].sum())
        fg_bits = []
        for g in part.groups:
            fg_bits.append(f"{g}={float(lam_row_sums[part.block_indices(g)].sum()):.3g}")
        return base + f", blocks[bg={bg_mass:.3g} " + " ".join(fg_bits) + "]"

    def iteration_diagnostics(
        self, global_params: dict[str, np.ndarray],
    ) -> dict[str, float | np.ndarray]:
        """Per-iter trajectories of Γ and Σ (small; safe to persist every iter).

        When topic_blocks is set, also includes topic_block_labels: a length-K
        object array with one string label per topic ("background" or the group
        name), in topic-index order.
        """
        diag = {
            "Gamma": np.asarray(global_params["Gamma"]),
            "Sigma": np.asarray(global_params["Sigma"]),
        }
        if self.topic_blocks is not None:
            diag["topic_block_labels"] = np.asarray(
                self.topic_blocks.topic_labels(), dtype=object)
        return diag


def _dirichlet_kl(q_alpha: np.ndarray, p_alpha: np.ndarray) -> float:
    """KL(Dirichlet(q_alpha) || Dirichlet(p_alpha)). Same as in LDA's stm.py uses."""
    qsum = q_alpha.sum()
    psum = p_alpha.sum()
    return float(
        gammaln(qsum) - gammaln(psum)
        - (gammaln(q_alpha) - gammaln(p_alpha)).sum()
        + ((q_alpha - p_alpha) * (digamma(q_alpha) - digamma(qsum))).sum()
    )
