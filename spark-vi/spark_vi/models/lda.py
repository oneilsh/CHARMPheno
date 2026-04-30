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
from scipy.special import digamma, gammaln

from spark_vi.core.model import VIModel
from spark_vi.core.types import BOWDocument


def _cavi_doc_inference(
    indices: np.ndarray,
    counts: np.ndarray,
    expElogbeta: np.ndarray,
    alpha: float,
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


class VanillaLDA(VIModel):
    """Vanilla LDA fittable by VIRunner with mini-batch SVI.

    Hyperparameters match Spark MLlib's pyspark.ml.clustering.LDA defaults
    so head-to-head comparisons are apples-to-apples.
    """

    def __init__(
        self,
        K: int,
        vocab_size: int,
        alpha: float | None = None,
        eta: float | None = None,
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
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
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
        self.alpha = float(alpha)
        self.eta = float(eta)
        self.gamma_shape = float(gamma_shape)
        self.cavi_max_iter = int(cavi_max_iter)
        self.cavi_tol = float(cavi_tol)

    # Contract methods (filled in over subsequent tasks).

    def initialize_global(self, data_summary: Any | None) -> dict[str, np.ndarray]:
        """Random Gamma(gamma_shape, 1/gamma_shape) init for lambda (K, V).

        gamma_shape=100 (MLlib default) gives draws tightly concentrated near 1;
        this is the variational analog of an "uninformative" topic-word prior
        with a small amount of symmetry-breaking noise.
        """
        lam = np.random.gamma(
            shape=self.gamma_shape,
            scale=1.0 / self.gamma_shape,
            size=(self.K, self.V),
        )
        return {"lambda": lam}

    def local_update(
        self,
        rows: Iterable[BOWDocument],
        global_params: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """E-step on one Spark partition.

        For each BOWDocument:
          1. Run _cavi_doc_inference to get gamma_d, expElogthetad, phi_norm.
          2. Add the suff-stat row update to lambda_stats[:, indices].
          3. Accumulate the data-likelihood and per-doc Dirichlet-KL terms.
        """
        lam = global_params["lambda"]                                 # (K, V)
        # Precompute expElogbeta once per partition (shared across docs).
        expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))

        lambda_stats = np.zeros_like(lam)
        doc_loglik_sum = 0.0
        doc_theta_kl_sum = 0.0
        n_docs = 0

        alpha_vec = np.full(self.K, self.alpha, dtype=np.float64)
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
                alpha=self.alpha,
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
            doc_theta_kl_sum += _dirichlet_kl(gamma, alpha_vec)
            n_docs += 1

        return {
            "lambda_stats": lambda_stats,
            "doc_loglik_sum": np.array(doc_loglik_sum),
            "doc_theta_kl_sum": np.array(doc_theta_kl_sum),
            "n_docs": np.array(float(n_docs)),
        }

    def update_global(
        self,
        global_params: dict[str, np.ndarray],
        target_stats: dict[str, np.ndarray],
        learning_rate: float,
    ) -> dict[str, np.ndarray]:
        """SVI natural-gradient step:

            lambda_new = (1 - rho) * lambda
                       + rho * (eta + expElogbeta * target_stats["lambda_stats"])

        The expElogbeta multiplication recovers the per-token-per-topic factor
        omitted from local_update's per-doc accumulation: phi_dnk depends on
        both expElogthetad (per-doc, included per-doc) and expElogbeta (per-
        topic-per-vocab, the same across all docs). Factoring expElogbeta out
        of the per-doc sum and applying it once here at the driver matches
        Spark MLlib's OnlineLDAOptimizer ("statsSum *:* expElogbeta.t" before
        updateLambda) and is what makes the natural-gradient direction correct.

        target_stats["lambda_stats"] is already pre-scaled by corpus_size /
        batch_size in mini-batch mode (per ADR 0005). expElogbeta is computed
        from the same lambda local_update saw, so the reference frame is
        consistent.
        """
        lam = global_params["lambda"]
        expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))
        target_lam = self.eta + expElogbeta * target_stats["lambda_stats"]
        new_lam = (1.0 - learning_rate) * lam + learning_rate * target_lam
        return {"lambda": new_lam}

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
        """
        lam = global_params["lambda"]
        K, V = lam.shape
        eta_vec = np.full(V, self.eta, dtype=np.float64)
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

        Pure function of (row, global_params) — must not read self for
        post-fit state. Returns:
          gamma: (K,) variational Dirichlet parameter for theta_d.
          theta: (K,) normalized E[theta_d] = gamma / gamma.sum().
        """
        lam = global_params["lambda"]
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
            alpha=self.alpha,
            gamma_init=gamma_init,
            max_iter=self.cavi_max_iter,
            tol=self.cavi_tol,
        )
        theta = gamma / gamma.sum()
        return {"gamma": gamma, "theta": theta}
