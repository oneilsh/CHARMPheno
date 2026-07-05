"""Local (pure numpy) diagnostic for topic-concentration recovery.

Plants synthetic documents at a KNOWN per-document topic concentration over a
shared-term topic matrix (topics share a vocabulary pool so inference must
actually disambiguate them, not just latch onto disjoint signature words),
then measures how faithfully STM inference (at a Sigma scale ``c``) and LDA
inference (at a Dirichlet ``alpha``) recover it.

"Concentration" here means how peaked a document's topic distribution theta
is: top_mass = max_k theta_k, and eff_topics = 1 / sum_k theta_k^2 (the
inverse-Simpson index / Hill number of order 2; Hill 1973, "Diversity and
Evenness: A Unifying Notation and Its Consequences", Ecology 54(2); Jost 2006,
"Entropy and diversity", Oikos 113(2)). Both are reused, not re-derived, from
spark_vi.eval.topic.concentration (doc_concentration / lda_concentration_readout).

This module is pure numpy + scipy, no Spark and no mllib imports, so it can
run entirely in-process. It is the planting + recovery core for a 3-task
diagnostic: this task builds planting/recovery; a companion task adds a
held-out-likelihood gold standard; a third runs the factorial sweep over
mechanisms/levels/scales.
"""
from __future__ import annotations

import numpy as np
from scipy.special import digamma

from spark_vi.eval.topic.concentration import lda_concentration_readout
from spark_vi.inference.concentration_optimization import alpha_newton_step
from spark_vi.models.topic.lda import _cavi_doc_inference
from spark_vi.models.topic.stm import _softmax, _stm_doc_inference
from spark_vi.models.topic.types import STMDocument


def make_shared_beta(
    K: int, V: int, *, pool_frac: float = 0.5, shared_mass: float = 0.5, seed: int = 0,
) -> np.ndarray:
    """Build a (K, V) topic-term matrix whose topics SHARE a common vocabulary
    pool, so recovering per-document concentration requires actually
    disambiguating overlapping topics rather than reading off a disjoint
    signature word. Mirrors the overlap construction in
    tests/_stm_synth.py:synthetic_gated_corpus_overlap.

    Layout: the first C = round(pool_frac * V) term ids are the shared pool
    (every topic draws mass there); the remaining V - C terms are split into K
    contiguous per-topic signature blocks. Topic k puts `shared_mass` of its
    probability on random weights over the pool and `1 - shared_mass` on
    random weights over its own signature block; the row is then normalized
    to sum to 1.
    """
    rng = np.random.default_rng(seed)
    C = round(pool_frac * V)
    sig_v = V - C
    sig = max(1, sig_v // K)

    beta = np.zeros((K, V), dtype=np.float64)
    for k in range(K):
        pool_w = rng.random(C)
        pool_w = pool_w / pool_w.sum() if pool_w.sum() > 0 else pool_w
        beta[k, :C] = shared_mass * pool_w

        lo = C + k * sig
        hi = lo + sig if k < K - 1 else V
        sig_w = rng.random(hi - lo)
        sig_w = sig_w / sig_w.sum() if sig_w.sum() > 0 else sig_w
        beta[k, lo:hi] = (1.0 - shared_mass) * sig_w

    beta /= beta.sum(axis=1, keepdims=True)
    return beta


def plant_corpus(
    beta: np.ndarray, *, D: int, doc_len: int, mechanism: str, level: float, seed: int = 0,
) -> tuple[list, np.ndarray]:
    """Plant D documents at a known per-document topic concentration over
    `beta` (K, V). Returns (docs, theta_true) where docs is a list of
    STMDocument and theta_true is (D, K) -- the MEASURED per-document theta
    used to generate tokens (ground truth for recovery, not the `level` knob
    itself).

    mechanism == "logistic_normal": eta ~ N(0, level * I_K), theta =
      softmax(eta). Larger level -> peakier.
    mechanism == "dirichlet": theta ~ Dirichlet(level * ones(K)). Smaller
      level -> peakier.

    Tokens: doc_len draws ~ Categorical(theta @ beta), aggregated to
    (indices, counts) via np.unique. Each doc is a non-gated STMDocument with
    a dummy length-1 covariate x=[1.0] and empty groups.
    """
    if mechanism not in ("logistic_normal", "dirichlet"):
        raise ValueError(f"plant_corpus: unknown mechanism {mechanism!r}")

    rng = np.random.default_rng(seed)
    K, V = beta.shape

    if mechanism == "logistic_normal":
        eta = rng.normal(loc=0.0, scale=np.sqrt(level), size=(D, K))
        theta_true = np.array([_softmax(eta[d]) for d in range(D)])
    else:
        theta_true = rng.dirichlet(np.full(K, level), size=D)

    docs = []
    for d in range(D):
        word_probs = theta_true[d] @ beta
        word_probs = word_probs / word_probs.sum()
        toks = rng.choice(V, size=doc_len, p=word_probs)
        u, c = np.unique(toks, return_counts=True)
        docs.append(STMDocument(
            indices=u.astype(np.int32),
            counts=c.astype(np.float64),
            length=doc_len,
            x=np.array([1.0]),
            groups=frozenset(),
        ))
    return docs, theta_true


def stm_recover_theta(
    docs: list, beta: np.ndarray, *, c: float, max_iter: int = 200, tol: float = 1e-6,
) -> np.ndarray:
    """Recover (D, K) theta_hat via per-document STM MAP inference under a
    mean-0, non-gated prior N(0, c * I_K). Calls _stm_doc_inference per doc
    with expElogbeta=beta (the plain probability beta, used LINEARLY by the
    STM data term -- NOT logged), Gamma=zeros((1, K)), x=[1.0],
    Sigma_inv_allowed=(1/c)*I_K, allowed=None, reference=None. theta_hat =
    softmax(eta_hat).
    """
    K = beta.shape[0]
    Gamma = np.zeros((1, K))
    x = np.array([1.0])
    Sigma_inv_allowed = (1.0 / c) * np.eye(K)

    theta_hat = np.zeros((len(docs), K))
    for d, doc in enumerate(docs):
        eta_hat, _, _ = _stm_doc_inference(
            indices=doc.indices, counts=doc.counts,
            expElogbeta=beta,
            Gamma=Gamma, Sigma_inv_allowed=Sigma_inv_allowed, x=x,
            max_iter=max_iter, tol=tol,
            allowed=None, reference=None,
        )
        theta_hat[d] = _softmax(eta_hat)
    return theta_hat


def lda_recover_theta(
    docs: list, beta: np.ndarray, *, alpha, max_iter: int = 100, tol: float = 1e-3,
) -> np.ndarray:
    """Recover (D, K) theta_hat via LDA CAVI at a FIXED Dirichlet alpha
    (scalar or length-K) and beta fixed to the given matrix. theta_hat =
    gamma / gamma.sum() per document.
    """
    K = beta.shape[0]
    theta_hat = np.zeros((len(docs), K))
    for d, doc in enumerate(docs):
        gamma_init = np.full(K, 100.0 / K)
        gamma, _, _, _ = _cavi_doc_inference(
            doc.indices, doc.counts, beta, alpha, gamma_init, max_iter, tol,
        )
        theta_hat[d] = gamma / gamma.sum()
    return theta_hat


def lda_optimize_alpha(
    docs: list, beta: np.ndarray, K: int, *, n_iter: int = 100, alpha_init: float = 1.0,
    tol: float = 1e-4,
) -> np.ndarray:
    """Optimize the LDA Dirichlet alpha on the planted corpus with beta
    FROZEN to the true topic-term matrix (isolates the concentration
    hyperparameter from topic learning).

    Mirrors OnlineLDA.update_global's alpha path (spark_vi/models/topic/lda.py
    ~359-370): each iteration runs _cavi_doc_inference per doc at the CURRENT
    alpha, accumulates e_log_theta_sum = sum_d E[log theta_d] computed as
    digamma(gamma) - digamma(gamma.sum()) DIRECTLY from gamma -- NOT via
    log(expElogthetad). Production (lda.py:291-301) explicitly avoids the
    exp(...)-then-log(...) round trip: it underflows to log(0) = -inf whenever
    any gamma_dk is small enough that digamma(gamma_dk) <~ -700 (e.g. once
    alpha has been pushed down near its floor), which then propagates -inf
    into every component of the next Newton step via alpha_newton_step's
    rank-1 Hessian coupling. Matching that direct-digamma form here avoids the
    same underflow. Then takes one alpha_newton_step with D = number of docs
    (the full corpus is processed each iteration, so the corpus-scale factor
    D/|batch| production applies to minibatches is exactly 1 here -- no
    additional scaling needed).
    """
    D = len(docs)
    alpha = np.full(K, float(alpha_init))

    for _ in range(n_iter):
        e_log_theta_sum = np.zeros(K)
        for doc in docs:
            gamma_init = np.full(K, 100.0 / K)
            gamma, _, _, _ = _cavi_doc_inference(
                doc.indices, doc.counts, beta, alpha, gamma_init, 100, 1e-3,
            )
            e_log_theta_sum += digamma(gamma) - digamma(gamma.sum())

        delta_alpha = alpha_newton_step(
            alpha=alpha, e_log_theta_sum_scaled=e_log_theta_sum, D=float(D),
        )
        new_alpha = np.maximum(alpha + delta_alpha, 1e-3)
        if np.max(np.abs(new_alpha - alpha)) < tol:
            alpha = new_alpha
            break
        alpha = new_alpha

    return alpha


def corpus_concentration_summary(theta_matrix: np.ndarray) -> dict:
    """(top_mass, eff_topics) percentile/mean summary over a (D, K) theta
    matrix. Thin re-export of lda_concentration_readout for convenience."""
    return lda_concentration_readout(theta_matrix)


def heldout_split(
    doc: STMDocument, *, holdout_frac: float = 0.3, seed: int,
) -> tuple[STMDocument, np.ndarray, np.ndarray] | None:
    """Split ONE document's tokens into a visible half and a held-out half.

    Expands (indices, counts) to a token multiset (each index repeated by its
    count), shuffles with a seeded rng, holds out round(holdout_frac *
    n_tokens) tokens, and re-aggregates both halves back to (indices, counts)
    form. Returns (visible_doc, held_indices, held_counts) where visible_doc
    is a new STMDocument (same x/groups; indices/counts/length reflect only
    the visible half) and held_indices/held_counts are numpy arrays for the
    held-out half.

    Guards: a document with < 2 tokens, or one whose visible half would be
    empty, returns None so the caller can skip it (there is nothing to infer
    theta_hat from, or nothing held out to score).
    """
    n_tokens = int(round(float(doc.counts.sum())))
    if n_tokens < 2:
        return None

    rng = np.random.default_rng(seed)
    tokens = np.repeat(doc.indices, doc.counts.astype(np.int64))
    rng.shuffle(tokens)

    n_held = round(holdout_frac * n_tokens)
    held_tokens = tokens[:n_held]
    visible_tokens = tokens[n_held:]
    if visible_tokens.size == 0:
        return None

    v_idx, v_counts = np.unique(visible_tokens, return_counts=True)
    if held_tokens.size > 0:
        h_idx, h_counts = np.unique(held_tokens, return_counts=True)
    else:
        h_idx = np.array([], dtype=np.int32)
        h_counts = np.array([], dtype=np.float64)

    visible_doc = STMDocument(
        indices=v_idx.astype(np.int32),
        counts=v_counts.astype(np.float64),
        length=int(v_counts.sum()),
        x=doc.x,
        groups=doc.groups,
    )
    return visible_doc, h_idx.astype(np.int32), h_counts.astype(np.float64)


def _predictive_loglik(
    theta_hat: np.ndarray, beta: np.ndarray, held_indices: np.ndarray, held_counts: np.ndarray,
) -> float:
    """Held-out log-likelihood of held_indices/held_counts under the
    predictive token distribution `theta_hat @ beta` (length V). The 1e-12
    floor guards log(0) for a held-out term with zero predicted mass. Returns
    the SUM over held-out tokens in this one document (the caller averages
    per token across the whole corpus).
    """
    pred = theta_hat @ beta
    return float(np.sum(held_counts * np.log(pred[held_indices] + 1e-12)))


def stm_heldout_ll(
    docs: list, beta: np.ndarray, *, c: float, holdout_frac: float = 0.3, seed: int = 0,
    max_iter: int = 200, tol: float = 1e-6,
) -> float:
    """Mean PER-TOKEN held-out log-likelihood for STM inference at Sigma
    scale c. For each doc: split into visible/held halves (split seed derived
    as seed + doc index, independent of c so a sweep over c sees the
    identical split -- see sweep_heldout), recover theta_hat from the VISIBLE
    doc only via stm_recover_theta (beta fixed), then score
    _predictive_loglik on the held-out half. Returns the corpus-wide total
    held-out log-likelihood divided by the corpus-wide total held-out token
    count (mean per token, so it is comparable across knobs and corpus
    sizes). Docs where heldout_split returns None (too short) are skipped.
    """
    total_ll = 0.0
    total_tokens = 0
    for i, doc in enumerate(docs):
        split = heldout_split(doc, holdout_frac=holdout_frac, seed=seed + i)
        if split is None:
            continue
        visible_doc, held_indices, held_counts = split
        if held_counts.size == 0:
            continue
        theta_hat = stm_recover_theta([visible_doc], beta, c=c, max_iter=max_iter, tol=tol)[0]
        total_ll += _predictive_loglik(theta_hat, beta, held_indices, held_counts)
        total_tokens += int(held_counts.sum())
    return total_ll / total_tokens


def lda_heldout_ll(
    docs: list, beta: np.ndarray, *, alpha, holdout_frac: float = 0.3, seed: int = 0,
    max_iter: int = 100, tol: float = 1e-3,
) -> float:
    """Mean PER-TOKEN held-out log-likelihood for LDA inference at Dirichlet
    alpha. Same protocol as stm_heldout_ll, but recovers theta_hat via
    lda_recover_theta (beta fixed) instead of stm_recover_theta.
    """
    total_ll = 0.0
    total_tokens = 0
    for i, doc in enumerate(docs):
        split = heldout_split(doc, holdout_frac=holdout_frac, seed=seed + i)
        if split is None:
            continue
        visible_doc, held_indices, held_counts = split
        if held_counts.size == 0:
            continue
        theta_hat = lda_recover_theta(
            [visible_doc], beta, alpha=alpha, max_iter=max_iter, tol=tol,
        )[0]
        total_ll += _predictive_loglik(theta_hat, beta, held_indices, held_counts)
        total_tokens += int(held_counts.sum())
    return total_ll / total_tokens


def sweep_heldout(
    docs: list, beta: np.ndarray, *, method: str, knobs: list, holdout_frac: float = 0.3,
    seed: int = 0,
) -> dict:
    """Sweep held-out log-likelihood over a list of concentration knobs.

    method == "stm": knobs are Sigma scales c, scored via stm_heldout_ll.
    method == "lda": knobs are Dirichlet alphas, scored via lda_heldout_ll.

    The SAME seed (and thus, per heldout_split's seed = seed + doc index
    convention, the identical per-doc visible/held split) is used for every
    knob, so the sweep is a controlled comparison in which only the
    inference knob varies. Returns {"lls": {knob: mean_ll, ...},
    "argmax_knob": <knob with max mean_ll>}.
    """
    if method == "stm":
        def score(knob):
            return stm_heldout_ll(docs, beta, c=knob, holdout_frac=holdout_frac, seed=seed)
    elif method == "lda":
        def score(knob):
            return lda_heldout_ll(docs, beta, alpha=knob, holdout_frac=holdout_frac, seed=seed)
    else:
        raise ValueError(f"sweep_heldout: unknown method {method!r}")

    lls = {knob: score(knob) for knob in knobs}
    argmax_knob = max(lls, key=lls.get)
    return {"lls": lls, "argmax_knob": argmax_knob}
