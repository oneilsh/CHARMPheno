"""VanillaLDA: Hoffman 2010 Online LDA as a VIModel.

Generative model for each document d (= one row in the RDD):
    theta_d ~ Dirichlet(alpha · 1_K)
    for n in 1..N_d:
        z_dn ~ Categorical(theta_d)
        w_dn ~ Categorical(beta_{z_dn})

Globally:
    beta_k ~ Dirichlet(eta · 1_V)

Variational mean field:
    q(beta_k) = Dirichlet(lambda_k)         # global, shape (K, V)
    q(theta_d) = Dirichlet(gamma_d)         # local, shape (K,)
    q(z_dn) = Categorical(phi_dn)           # local, collapsed via Lee/Seung 2001

Symbols:
    K           number of topics
    V           vocabulary size
    D           number of documents (corpus_size)
    N_d         total tokens in document d (with repeats)
    lambda      (K, V) global variational Dirichlet for topic-word
    gamma_d     (K,) local variational Dirichlet for doc-topic
    expElogbeta (K, V) precomputed exp(E[log beta_kv]) under q(beta)
    expElogthetad (K,) precomputed exp(E[log theta_dk]) under q(theta_d)
    phi_norm    (n_unique,) implicit phi-normalizer for the Lee/Seung trick
    alpha, eta  symmetric Dirichlet concentrations

References:
    Hoffman, Blei, Bach 2010. Online learning for LDA. NIPS.
    Hoffman, Blei, Wang, Paisley 2013. Stochastic VI. JMLR.
    Lee, Seung 2001. Algorithms for non-negative matrix factorization. NIPS.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from scipy.special import digamma, gammaln, polygamma

from spark_vi.core.model import VIModel
from spark_vi.core.types import BOWDocument


def _cavi_doc_inference(
    indices: np.ndarray,
    counts: np.ndarray,
    expElogbeta: np.ndarray,
    alpha: float | np.ndarray,
    gamma_init: np.ndarray,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Inner CAVI loop for a single document under fixed q(beta).

    Lee/Seung 2001 trick: never materialize the full (K, n_unique) phi
    matrix. Instead carry only gamma_d (K-vector) and phi_norm (n_unique-
    vector). Memory is O(K + n_unique) rather than O(K * n_unique).

    Recurrence (equivalent to explicit phi normalized per token):
        expElogthetad = exp(digamma(gamma) - digamma(gamma.sum()))
        eb_d          = expElogbeta[:, indices]           # (K, n_unique)
        phi_norm      = eb_d.T @ expElogthetad + 1e-100  # (n_unique,)
        gamma_new     = alpha + expElogthetad * (eb_d @ (counts / phi_norm))

    Returns:
        gamma:         (K,) converged variational Dirichlet parameter for theta_d.
        expElogthetad: (K,) exp(E[log theta_d]) at the converged gamma.
        phi_norm:      (n_unique,) implicit phi-normalizer at convergence.
                       Needed for the data-likelihood ELBO term.
        n_iter:        iterations consumed (1..max_iter).
    """
    eb_d = expElogbeta[:, indices]  # (K, n_unique)
    gamma = gamma_init.astype(np.float64, copy=True)

    expElogthetad = np.exp(digamma(gamma) - digamma(gamma.sum()))
    phi_norm = eb_d.T @ expElogthetad + 1e-100

    n_iter = 0
    for it in range(1, max_iter + 1):
        n_iter = it
        prev = gamma.copy()
        # (K, n_unique) @ (n_unique,) -> (K,); elementwise mul with K-vec
        gamma = alpha + expElogthetad * (eb_d @ (counts / phi_norm))
        expElogthetad = np.exp(digamma(gamma) - digamma(gamma.sum()))
        phi_norm = eb_d.T @ expElogthetad + 1e-100
        if np.mean(np.abs(gamma - prev)) < tol:
            break

    return gamma, expElogthetad, phi_norm, n_iter


def _dirichlet_kl(q_alpha: np.ndarray, p_alpha: np.ndarray) -> float:
    """KL(Dirichlet(q_alpha) || Dirichlet(p_alpha)).

    Closed form via gammaln + digamma; both arrays must be K-vectors.
    """
    qsum = q_alpha.sum()
    psum = p_alpha.sum()
    return float(
        gammaln(qsum) - gammaln(psum)
        - (gammaln(q_alpha) - gammaln(p_alpha)).sum()
        + ((q_alpha - p_alpha) * (digamma(q_alpha) - digamma(qsum))).sum()
    )


def _alpha_newton_step(
    alpha: np.ndarray,
    e_log_theta_sum_scaled: np.ndarray,
    D: float,
) -> np.ndarray:
    """One Newton step for asymmetric Dirichlet α.

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


def _eta_newton_step(
    eta: float,
    e_log_phi_sum: float,
    K: int,
    V: int,
) -> float:
    """One Newton step for symmetric scalar Dirichlet η.

    Per Hoffman, Blei, Bach 2010 §3.4. The ELBO part depending on η is
        L(η) = K · log Γ(V·η) − K·V · log Γ(η)
             + (η − 1) · Σ_t Σ_v E[log φ_tv]
    with scalar gradient and Hessian
        g(η) = K·V · [ψ(V·η) − ψ(η)] + Σ_t Σ_v E[log φ_tv]
        H(η) = K·V² · ψ′(V·η) − K·V · ψ′(η)
    Newton step Δη = −g/H.

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
        post-step floor (parallel to _alpha_newton_step's caller contract).
    """
    g = K * V * (digamma(V * eta) - digamma(eta)) + e_log_phi_sum
    h = K * V * V * polygamma(1, V * eta) - K * V * polygamma(1, eta)
    return -g / h


class VanillaLDA(VIModel):
    """Vanilla LDA fittable by VIRunner with mini-batch SVI.

    Hyperparameters match Spark MLlib's pyspark.ml.clustering.LDA defaults
    so head-to-head comparisons are apples-to-apples.
    """

    def __init__(
        self,
        K: int,
        vocab_size: int,
        alpha: float | np.ndarray | None = None,
        eta: float | None = None,
        optimize_alpha: bool = False,
        optimize_eta: bool = False,
        gamma_shape: float = 100.0,
        cavi_max_iter: int = 100,
        cavi_tol: float = 1e-3,
    ) -> None:
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        if vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1, got {vocab_size}")
        if alpha is None:
            alpha = 1.0 / K
        if eta is None:
            eta = 1.0 / K

        # alpha may be a scalar (broadcast to length-K symmetric vector) or
        # a length-K array (asymmetric). Always stored on self as a length-K
        # float64 array so downstream code treats both inputs uniformly.
        alpha_arr = np.asarray(alpha, dtype=np.float64)
        if alpha_arr.ndim == 0:
            alpha_arr = np.full(K, float(alpha_arr), dtype=np.float64)
        elif alpha_arr.ndim != 1 or alpha_arr.shape[0] != K:
            raise ValueError(
                f"alpha must be a scalar or a length-{K} 1-D array, "
                f"got shape {alpha_arr.shape}"
            )
        if (alpha_arr <= 0).any():
            raise ValueError(f"all alpha components must be > 0, got {alpha_arr}")
        if eta <= 0:
            raise ValueError(f"eta must be > 0, got {eta}")
        if gamma_shape <= 0:
            raise ValueError(f"gamma_shape must be > 0, got {gamma_shape}")
        if cavi_max_iter < 1:
            raise ValueError(f"cavi_max_iter must be >= 1, got {cavi_max_iter}")
        if cavi_tol <= 0:
            raise ValueError(f"cavi_tol must be > 0, got {cavi_tol}")

        self.K = int(K)
        self.V = int(vocab_size)
        self.alpha = alpha_arr             # length-K initial α (driver-side starting point)
        self.eta = float(eta)              # scalar initial η
        self.gamma_shape = float(gamma_shape)
        self.cavi_max_iter = int(cavi_max_iter)
        self.cavi_tol = float(cavi_tol)
        self.optimize_alpha = bool(optimize_alpha)
        self.optimize_eta = bool(optimize_eta)

    # Contract methods (filled in over subsequent tasks).

    def initialize_global(self, data_summary: Any | None) -> dict[str, np.ndarray]:
        """Random Gamma(gamma_shape, 1/gamma_shape) init for lambda (K, V),
        plus the initial α (K-vector) and η (scalar) seeded from constructor.

        See VanillaLDA.__init__ for why α is always a length-K array internally.
        Putting α / η in global_params (rather than relying on self) means the
        runner broadcasts and round-trips them like λ; update_global mutates
        them when the optimize_alpha / optimize_eta flags are on.

        gamma_shape=100 trace: see Hoffman 2010's onlineldavb.py line 126.
        """
        lam = np.random.gamma(
            shape=self.gamma_shape,
            scale=1.0 / self.gamma_shape,
            size=(self.K, self.V),
        )
        return {
            "lambda": lam,
            "alpha": self.alpha.copy(),         # defensive copy — runner mutates
            "eta": np.array(self.eta),          # 0-d ndarray for combine_stats type-uniformity
        }

    def local_update(
        self,
        rows: Iterable[BOWDocument],
        global_params: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """E-step on one Spark partition.

        Reads α from global_params so that mid-fit α updates from
        update_global propagate to the next iteration's CAVI without
        needing to re-broadcast self.alpha.

        For each BOWDocument:
          1. Run _cavi_doc_inference to get gamma_d, expElogthetad, phi_norm.
          2. Add the suff-stat row update to lambda_stats[:, indices].
          3. Accumulate the data-likelihood and per-doc Dirichlet-KL terms.
        """
        lam = global_params["lambda"]                                 # (K, V)
        alpha = global_params["alpha"]                                # (K,)
        # Precompute expElogbeta once per partition (shared across docs).
        expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))

        lambda_stats = np.zeros_like(lam)
        doc_loglik_sum = 0.0
        doc_theta_kl_sum = 0.0
        n_docs = 0
        e_log_theta_sum = np.zeros(self.K, dtype=np.float64) if self.optimize_alpha else None

        # gamma_init draws Gamma(gamma_shape, 1/gamma_shape) per doc — same as MLlib.
        for doc in rows:
            # TODO: per-doc reproducibility for MLlib comparisons — derive seed
            # from a per-doc deterministic key (e.g., hash of doc.indices +
            # cfg.random_seed) instead of numpy's global RNG.
            gamma_init = np.random.gamma(
                shape=self.gamma_shape,
                scale=1.0 / self.gamma_shape,
                size=self.K,
            )
            gamma, expElogthetad, phi_norm, _ = _cavi_doc_inference(
                indices=doc.indices,
                counts=doc.counts,
                expElogbeta=expElogbeta,
                alpha=alpha,
                gamma_init=gamma_init,
                max_iter=self.cavi_max_iter,
                tol=self.cavi_tol,
            )

            # Suff-stat row update:
            # outer(expElogthetad, counts/phi_norm) gives (K, n_unique); add to seen cols.
            sstats_row = np.outer(expElogthetad, doc.counts / phi_norm)
            # Safe: BOWDocument guarantees unique indices (no fancy-index aliasing).
            lambda_stats[:, doc.indices] += sstats_row

            # Data-likelihood term: sum_n c_n * log(phi_norm_n).
            # phi_norm has a +1e-100 floor inside _cavi_doc_inference; if the
            # floor triggers (only possible under near-degenerate lambda),
            # log(phi_norm) silently corrupts doc_loglik_sum. Unreachable for
            # typical lambda concentrations; flagging for ELBO debugging.
            doc_loglik_sum += float(np.sum(doc.counts * np.log(phi_norm)))

            # Per-doc Dirichlet KL: KL(q(theta_d) || p(theta_d)).
            doc_theta_kl_sum += _dirichlet_kl(gamma, alpha)
            n_docs += 1

            if self.optimize_alpha:
                # E[log θ_dk] = ψ(γ_dk) − ψ(Σ_j γ_dj) ≡ log(expElogthetad),
                # since _cavi_doc_inference returns expElogthetad = exp of that
                # exact expression at convergence. One log/doc beats two
                # digammas/doc on the Spark hot path.
                e_log_theta_sum += np.log(expElogthetad)

        out: dict[str, np.ndarray] = {
            "lambda_stats": lambda_stats,
            "doc_loglik_sum": np.array(doc_loglik_sum),
            "doc_theta_kl_sum": np.array(doc_theta_kl_sum),
            "n_docs": np.array(float(n_docs)),
        }
        if e_log_theta_sum is not None:
            out["e_log_theta_sum"] = e_log_theta_sum
        return out

    def update_global(
        self,
        global_params: dict[str, np.ndarray],
        target_stats: dict[str, np.ndarray],
        learning_rate: float,
    ) -> dict[str, np.ndarray]:
        """SVI natural-gradient step on λ; optional Newton step on α.

        λ update (always):
            lambda_new = (1 - rho) * lambda
                       + rho * (eta + expElogbeta * target_stats["lambda_stats"])

        α update (only when self.optimize_alpha):
            Δα   = _alpha_newton_step(α, target_stats["e_log_theta_sum"], D=n_docs_scaled)
            α_new = clip(α + rho * Δα, min=1e-3)

        target_stats[*] is already corpus-scaled by the runner per ADR 0005,
        so target_stats["e_log_theta_sum"] is the corpus-equivalent
        Σ_d E[log θ_dk] and target_stats["n_docs"] is the corpus-equivalent D.

        η update (only when self.optimize_eta):
            e_log_phi_sum = Σ_t Σ_v E[log φ_tv] from current λ
                          (NOT from target_stats — this is a global stat).
            Δη   = _eta_newton_step(η, e_log_phi_sum, K, V)
            η_new = clip(η + rho * Δη, min=1e-3)

        Note the asymmetry vs α: η's stat is computable from current global
        state (λ), so no extra return value from local_update. HDP will reuse
        this same pattern for its corpus-stick γ update.

        The expElogbeta multiplication recovers the per-token-per-topic factor
        omitted from local_update's per-doc accumulation: phi_dnk depends on
        both expElogthetad (per-doc, included per-doc) and expElogbeta (per-
        topic-per-vocab, the same across all docs). Factoring expElogbeta out
        of the per-doc sum and applying it once here at the driver matches
        Spark MLlib's OnlineLDAOptimizer ("statsSum *:* expElogbeta.t" before
        updateLambda) and is what makes the natural-gradient direction correct.
        """
        lam = global_params["lambda"]
        alpha = global_params["alpha"]
        eta = global_params["eta"]

        expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))
        target_lam = eta + expElogbeta * target_stats["lambda_stats"]
        new_lam = (1.0 - learning_rate) * lam + learning_rate * target_lam

        if self.optimize_alpha:
            # `_alpha_newton_step`'s closed-form Sherman-Morrison step has a
            # theoretical degeneracy at c == Σ_k 1/d_k (denominator of `b`).
            # Measure-zero in practice; the post-step floor below + ρ-damping
            # keep d_k = D·ψ′(α_k) bounded so the equality is unreachable.
            D = float(target_stats["n_docs"])
            delta_alpha = _alpha_newton_step(
                alpha=alpha,
                e_log_theta_sum_scaled=target_stats["e_log_theta_sum"],
                D=D,
            )
            new_alpha = np.maximum(alpha + learning_rate * delta_alpha, 1e-3)
        else:
            new_alpha = alpha

        if self.optimize_eta:
            # The η stat (Σ_t Σ_v E[log φ_tv]) is computable directly from the
            # *just-updated* λ — unlike α's per-doc stat, no `local_update`
            # contribution is needed. This is the global-stat optimization
            # pattern HDP will reuse for its γ update.
            K, V = new_lam.shape
            e_log_phi_sum = float(
                (digamma(new_lam) - digamma(new_lam.sum(axis=1, keepdims=True))).sum()
            )
            delta_eta = _eta_newton_step(
                eta=float(eta), e_log_phi_sum=e_log_phi_sum, K=K, V=V,
            )
            # 0-d wrap matches initialize_global's `np.array(self.eta)` shape
            # so combine_stats and downstream consumers see the same type
            # regardless of whether η was optimized this iteration.
            new_eta = np.array(max(float(eta) + learning_rate * delta_eta, 1e-3))
        else:
            new_eta = eta

        return {
            "lambda": new_lam,
            "alpha": new_alpha,
            "eta": new_eta,
        }

    def compute_elbo(
        self,
        global_params: dict[str, np.ndarray],
        aggregated_stats: dict[str, np.ndarray],
    ) -> float:
        """ELBO = doc-data-likelihood + doc-level KL + global KL.

        With our sign conventions (KLs subtracted):
            ELBO = doc_loglik_sum
                 - doc_theta_kl_sum
                 - sum_k KL( q(beta_k) || p(beta_k) )

        doc_loglik_sum and doc_theta_kl_sum are aggregated across the
        partition by local_update; the global beta KL is computed here on
        the driver from lambda alone.

        η is read from global_params so an η-optimization update mid-fit
        feeds back into the global KL prior on the next ELBO computation.
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
            - float(aggregated_stats["doc_theta_kl_sum"])
            - global_kl
        )

    def infer_local(self, row: BOWDocument, global_params: dict[str, np.ndarray]):
        """Single-document E-step under fixed global params.

        Pure function of (row, global_params). Reads α from global_params so
        a model trained with optimize_alpha=True transforms with the *trained*
        α rather than the constructor's initial value.
        """
        lam = global_params["lambda"]
        alpha = global_params["alpha"]

        expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))
        gamma_init = np.random.gamma(
            shape=self.gamma_shape,
            scale=1.0 / self.gamma_shape,
            size=self.K,
        )
        gamma, _, _, _ = _cavi_doc_inference(
            indices=row.indices,
            counts=row.counts,
            expElogbeta=expElogbeta,
            alpha=alpha,
            gamma_init=gamma_init,
            max_iter=self.cavi_max_iter,
            tol=self.cavi_tol,
        )
        theta = gamma / gamma.sum()
        return {"gamma": gamma, "theta": theta}

    def iteration_summary(self, global_params: dict[str, np.ndarray]) -> str:
        """Compact per-iter view of α, η, and λ row-mass spread.

        α as min/max/mean: when symmetric all three coincide; when optimized
        the spread shows the asymmetry the empirical-Bayes update has put in.
        η is a scalar. λ row sums (Σ_v λ_kv) span tells you whether topics
        are diverging in mass — useful when chasing topic-collapse vs healthy
        differentiation.

        Per-topic α_k / Σλ_k / peak values are surfaced via the topic-evolution
        logger in analysis/cloud/lda_bigquery_cloud.py, which already iterates
        over k to print top tokens and can prefix the same per-topic stats.
        """
        alpha = np.asarray(global_params["alpha"])
        eta = float(global_params["eta"])
        lam = global_params["lambda"]
        lam_row_sums = lam.sum(axis=1)
        return (
            f"α[min={alpha.min():.4g} max={alpha.max():.4g} mean={alpha.mean():.4g}], "
            f"η={eta:.4g}, "
            f"Σλ_k[min={lam_row_sums.min():.3g} max={lam_row_sums.max():.3g}]"
        )
