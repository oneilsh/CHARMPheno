"""Online HDP topic model — Wang/Paisley/Blei 2011, AISTATS.

Implements the algorithm via the spark_vi `VIModel` contract: per-doc CAVI
on workers (`local_update`), natural-gradient SVI step on the driver
(`update_global`).

Notation. We follow Wang's reference-code convention (also used by intel-spark):
  T = corpus-level truncation (paper's K)
  K = doc-level truncation    (paper's T)
The AISTATS paper inverts these letters; see
docs/architecture/TOPIC_STATE_MODELING.md "Notation" for the rationale.

References:
  - Wang, Paisley, Blei (2011). "Online Variational Inference for the
    Hierarchical Dirichlet Process." AISTATS. Eqs 15-18 give doc-CAVI;
    Eqs 22-27 give the natural-gradient SVI step.
  - Wang's reference Python implementation:
    https://github.com/blei-lab/online-hdp (onlinehdp.py).
  - intel-spark TopicModeling Scala port (cited for confirmation only;
    we explicitly diverge from its driver-side `chunk.collect()` E-step).
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from scipy.special import digamma, gammaln

from spark_vi.core.model import VIModel


# ---------------------------------------------------------------------------
# Module-private math helpers (pure functions, easy to unit-test in isolation).
# ---------------------------------------------------------------------------


def _log_normalize_rows(M: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise log-normalize.

    Returns log(softmax(M, axis=1)). Subtracts the per-row max before
    exponentiating to avoid overflow on large positive entries; for very
    negative entries np.exp underflows to 0, which is benign.
    """
    row_max = M.max(axis=1, keepdims=True)
    shifted = M - row_max
    log_norm = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return shifted - log_norm


def _expect_log_sticks(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sethuraman-style E[log stick] expectation.

    Inputs `a, b` are length (T-1,) for the corpus stick or (K-1,) for a
    doc stick. Returns a length-(len(a)+1) vector — the trailing entry
    handles the truncation: q(stick_last = 1) = 1, so E[log W_last] = 0
    and only the cumulative E[log(1 - W_<last)] contributes.

    For β'_k ~ Beta(a_k, b_k):
      E[log W_k]      = digamma(a_k) - digamma(a_k + b_k)
      E[log(1 - W_k)] = digamma(b_k) - digamma(a_k + b_k)
      E[log β_k]      = E[log W_k] + sum_{l<k} E[log(1 - W_l)]
    """
    dig_sum = digamma(a + b)
    Elog_W = digamma(a) - dig_sum         # length (T-1,)
    Elog_1mW = digamma(b) - dig_sum       # length (T-1,)

    out = np.zeros(len(a) + 1, dtype=np.float64)
    out[:-1] = Elog_W
    out[1:] += np.cumsum(Elog_1mW)
    return out


def _beta_kl(
    a: np.ndarray,
    b: np.ndarray,
    *,
    prior_a: float | np.ndarray,
    prior_b: float | np.ndarray,
) -> np.ndarray:
    """KL[Beta(a, b) ‖ Beta(prior_a, prior_b)], elementwise.

    Closed form:
      KL = log B(α0, β0) - log B(α, β)
         + (α - α0) * (ψ(α) - ψ(α + β))
         + (β - β0) * (ψ(β) - ψ(α + β))
    where B is the Beta function (B(x, y) = Γ(x)Γ(y)/Γ(x+y)).

    Returns a length-len(a) vector; broadcast `prior_a` / `prior_b` from a
    scalar if needed (used for corpus prior Beta(1, gamma) where prior_a is
    scalar and prior_b is scalar gamma).
    """
    pa = np.broadcast_to(np.asarray(prior_a, dtype=np.float64), a.shape)
    pb = np.broadcast_to(np.asarray(prior_b, dtype=np.float64), a.shape)

    log_B_prior = gammaln(pa) + gammaln(pb) - gammaln(pa + pb)
    log_B_post = gammaln(a) + gammaln(b) - gammaln(a + b)

    dig_sum = digamma(a + b)
    return (
        log_B_prior - log_B_post
        + (a - pa) * (digamma(a) - dig_sum)
        + (b - pb) * (digamma(b) - dig_sum)
    )


def _doc_e_step(
    *,
    indices: np.ndarray,
    counts: np.ndarray,
    Elogbeta_doc: np.ndarray,
    Elog_sticks_corpus: np.ndarray,
    alpha: float,
    K: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    warmup: int = 3,
) -> dict[str, np.ndarray | float]:
    """Per-document coordinate ascent for HDP variational posterior.

    Implements paper Eqs 15-18 (Wang/Paisley/Blei 2011). The inner loop
    rotates through four blocks:
      var_phi   (K, T)   — q(c_jk = t):  doc-atom k → corpus-atom t
      phi       (Wt, K)  — q(z_jn = k):  word n → doc-atom k
      a, b      (K-1,)   — q(π'_jt) = Beta(a, b)  (doc stick params)
    until the per-doc ELBO converges within `tol` (relative).

    Args:
      indices: vocab IDs for the unique words in this document, length Wt.
      counts: integer counts per unique word, length Wt.
      Elogbeta_doc: precomputed E[log φ_t,w] for the words in this doc,
        shape (T, Wt). Caller supplies this from the broadcast Elogbeta.
      Elog_sticks_corpus: E[log β] under the corpus stick variational
        posterior, length T.
      alpha: doc-level stick concentration (paper's α0).
      K: doc-level truncation level.
      max_iter: hard cap on coordinate-ascent iterations.
      tol: relative ELBO convergence threshold.
      warmup: skip the prior-correction terms in var_phi / phi updates for
        this many iterations. Wang's empirical stability trick.

    Returns:
      Dict with keys: a, b, phi, var_phi, log_phi, log_var_phi, plus the
      four ELBO scalars: doc_loglik, doc_z_term, doc_c_term, doc_stick_kl.
    """
    Wt = len(indices)
    T = Elogbeta_doc.shape[0]
    counts_col = counts[:, None]

    # Initialize per-doc state.
    a = np.ones(K - 1, dtype=np.float64)
    b = alpha * np.ones(K - 1, dtype=np.float64)
    phi = np.full((Wt, K), 1.0 / K, dtype=np.float64)
    Elog_sticks_doc = _expect_log_sticks(a, b)  # (K,)

    # log_phi / log_var_phi populated inside the loop; declared here so the
    # final returned dict can reference them after the loop ends.
    log_phi = np.log(phi)
    log_var_phi = np.zeros((K, T), dtype=np.float64)
    var_phi = np.zeros((K, T), dtype=np.float64)

    # Invariant across the CAVI iters; cached to avoid recomputing per-iter.
    weighted_Elogbeta = Elogbeta_doc * counts_col.T  # (T, Wt)

    prev_elbo = -np.inf
    doc_loglik = doc_z_term = doc_c_term = 0.0
    doc_stick_kl = 0.0

    for it in range(max_iter):
        # 1) var_phi update — paper Eq 17. Shape (K, T).
        # E[log p(c_jk | β')] = Elog_sticks_corpus[t]
        # E[log p(w_jn | φ_k)] · counts contributes via phi.T @ (Elogbeta_doc * counts).T
        log_var_phi = phi.T @ weighted_Elogbeta.T  # (K, T)
        if it >= warmup:
            log_var_phi = log_var_phi + Elog_sticks_corpus[None, :]
        log_var_phi = _log_normalize_rows(log_var_phi)
        var_phi = np.exp(log_var_phi)

        # 2) phi update — paper Eq 18. Shape (Wt, K).
        log_phi = (var_phi @ Elogbeta_doc).T  # (Wt, K)
        if it >= warmup:
            log_phi = log_phi + Elog_sticks_doc[None, :]
        log_phi = _log_normalize_rows(log_phi)
        phi = np.exp(log_phi)

        # 3) a, b update — paper Eqs 15-16.
        phi_w = phi * counts_col  # (Wt, K)
        phi_sum = phi_w.sum(axis=0)  # (K,)
        a = 1.0 + phi_sum[: K - 1]
        # b[t] = α + Σ_{s>t} phi_sum[s]
        b = alpha + np.cumsum(phi_sum[1:][::-1])[::-1]
        Elog_sticks_doc = _expect_log_sticks(a, b)

        # 4) Compute doc ELBO and check convergence.
        # Term naming follows paper Eq 14 decomposition:
        #   doc_c_term = E[log p(c | β')] + H(q(c))    = Σ (Elog_sticks_corpus - log_var_phi) · var_phi
        #   doc_z_term = E[log p(z | π)]  + H(q(z))    = Σ_n count_n · Σ_k (Elog_sticks_doc - log_phi) · phi
        #   doc_loglik = E[log p(w | z, c, φ)]         = Σ phi.T · (var_phi @ (Elogbeta_doc * counts))
        #   doc_stick_kl = KL[q(π') ‖ p(π')]           — subtracted
        #
        # Note: doc_c_term sums over K doc-atoms (no count weighting — one c per atom).
        # doc_z_term sums over N word tokens; phi[n,k] is per-unique-type, so it must
        # be multiplied by counts_col to recover the per-token contribution.
        doc_c_term = float(np.sum((Elog_sticks_corpus[None, :] - log_var_phi) * var_phi))
        doc_z_term = float(np.sum((Elog_sticks_doc[None, :] - log_phi) * phi * counts_col))
        data_part = var_phi @ weighted_Elogbeta  # (K, Wt)
        doc_loglik = float(np.sum(phi.T * data_part))
        doc_stick_kl = float(_beta_kl(a, b, prior_a=1.0, prior_b=alpha).sum())

        elbo = doc_loglik + doc_z_term + doc_c_term - doc_stick_kl

        # Don't allow early-exit before warmup completes — the iter < warmup branch
        # omits the prior-correction terms, so we need at least one full post-warmup
        # iter (it == warmup) before the relative-ELBO test is meaningful.
        if it > warmup and abs(elbo - prev_elbo) / max(abs(prev_elbo), 1.0) < tol:
            break
        prev_elbo = elbo

    return {
        "a": a, "b": b,
        "phi": phi, "var_phi": var_phi,
        "log_phi": log_phi, "log_var_phi": log_var_phi,
        "doc_loglik": doc_loglik,
        "doc_z_term": doc_z_term,
        "doc_c_term": doc_c_term,
        "doc_stick_kl": doc_stick_kl,
    }


class OnlineHDP(VIModel):
    """Online Hierarchical Dirichlet Process topic model.

    Implements Wang/Paisley/Blei 2011 stochastic VI for the HDP via the
    spark_vi VIModel contract: per-doc CAVI on workers (`local_update`),
    natural-gradient SVI step on the driver (`update_global`).

    Args:
      T: corpus-level truncation. Upper bound on the number of topics
        the model can discover; effective topic count is typically much
        smaller (the inactive corpus sticks shrink toward 0).
      K: doc-level truncation. Upper bound on topics per document.
        Should be much smaller than T — clinical visits typically span
        a handful of phenotypes, not hundreds.
      vocab_size: V, number of distinct word IDs the model handles.
      alpha: doc-level stick concentration (paper's α0). Higher → more
        topics per doc.
      gamma: corpus-level stick concentration. Higher → more discovered
        topics overall.
      eta: symmetric Dirichlet concentration for the topic-word prior.
      gamma_shape: shape parameter for the Gamma init of λ. Default 100
        matches VanillaLDA (Hoffman 2010 onlineldavb.py).
      cavi_max_iter: hard cap on doc-CAVI iterations per doc.
      cavi_tol: relative ELBO convergence threshold for doc-CAVI early
        termination.
    """

    def __init__(
        self,
        T: int,
        K: int,
        vocab_size: int,
        *,
        alpha: float = 1.0,
        gamma: float = 1.0,
        eta: float = 0.01,
        gamma_shape: float = 100.0,
        cavi_max_iter: int = 100,
        cavi_tol: float = 1e-4,
    ) -> None:
        if T < 2:
            raise ValueError(f"T must be >= 2 (need T-1 sticks), got {T}")
        if K < 2:
            raise ValueError(f"K must be >= 2 (need K-1 sticks), got {K}")
        if vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1, got {vocab_size}")
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        if gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {gamma}")
        if eta <= 0:
            raise ValueError(f"eta must be > 0, got {eta}")
        if gamma_shape <= 0:
            raise ValueError(f"gamma_shape must be > 0, got {gamma_shape}")
        if cavi_max_iter < 1:
            raise ValueError(f"cavi_max_iter must be >= 1, got {cavi_max_iter}")
        if cavi_tol <= 0:
            raise ValueError(f"cavi_tol must be > 0, got {cavi_tol}")

        self.T = int(T)
        self.K = int(K)
        self.V = int(vocab_size)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.eta = float(eta)
        self.gamma_shape = float(gamma_shape)
        self.cavi_max_iter = int(cavi_max_iter)
        self.cavi_tol = float(cavi_tol)

    # Stub methods filled in by Tasks 7-12.
    def initialize_global(self, data_summary: Any | None) -> dict[str, np.ndarray]:
        """Random Gamma(gamma_shape, 1/gamma_shape) init for λ (T, V).

        Departs from Wang's reference (which uses Gamma(1, 1) · D · 100 /
        (T·V) − η) — that scale-then-cancel-η is undocumented and not
        derived. Match-LDA is the validated choice; gamma_shape=100
        traces back to Hoffman 2010 onlineldavb.py line 126.

        Corpus sticks (u, v) start at the prior Beta(1, γ): u = 1, v = γ.
        Paper-following init. Wang's reference uses v = [T-1, T-2, ..., 1]
        ("make a uniform at beginning") which is an undocumented bias
        toward low topic indices; we don't reproduce it.
        """
        lam = np.random.gamma(
            shape=self.gamma_shape,
            scale=1.0 / self.gamma_shape,
            size=(self.T, self.V),
        )
        u = np.ones(self.T - 1, dtype=np.float64)
        v = np.full(self.T - 1, self.gamma, dtype=np.float64)
        return {"lambda": lam, "u": u, "v": v}

    def local_update(
        self,
        rows: Iterable[Any],
        global_params: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError("OnlineHDP is being built; see Task 8.")

    def update_global(
        self,
        global_params: dict[str, np.ndarray],
        target_stats: dict[str, np.ndarray],
        learning_rate: float,
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError("OnlineHDP is being built; see Task 9.")
