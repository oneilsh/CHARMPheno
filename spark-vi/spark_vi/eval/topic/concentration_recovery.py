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
from scipy.optimize import linear_sum_assignment
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


# ---------------------------------------------------------------------------
# CO-FIT-beta extension (CR-4).
#
# The frozen-beta primitives above (make_shared_beta ... sweep_heldout) plant
# over a KNOWN beta and FREEZE it at truth during recovery, so they isolate
# concentration inference from topic learning. That leaves one question open
# (insight 0038 "What this does NOT claim"): when each model LEARNS its own
# beta from the documents, does LDA's Dirichlet document-sparsity pressure
# carve a SHARPER, more document-specific beta -- raising per-document top_mass
# (peakier patients) -- while STM's logistic-normal stays more blended? If so,
# the real-data STM-vs-LDA peakiness gap (0038: STM 0.269 vs LDA 0.513) is a
# beta-co-adaptation effect, not an alpha-inference artifact.
#
# The helpers below add full-batch (non-Spark) co-fitting of beta for both
# families, reusing the SAME per-doc E-step primitives (_cavi_doc_inference,
# _stm_doc_inference) the frozen path and the production models use, plus a
# permutation-invariant beta-recovery metric and a beta-sharpness readout.
# They are strictly additive: the frozen-beta path and its tests are untouched.
# ---------------------------------------------------------------------------


def stm_cofit_beta(
    train_docs: list, K: int, V: int, *, c: float, eta: float | None = None,
    n_em_iter: int = 60, seed: int = 0, lbfgs_max_iter: int = 50, lbfgs_tol: float = 1e-4,
) -> np.ndarray:
    """Co-fit the STM topic-word matrix beta by full-batch variational EM under
    a FIXED logistic-normal prior N(0, c * I_K).

    This is the co-fit analog of stm_recover_theta: same per-doc E-step
    (_stm_doc_inference with Gamma=0, Sigma_inv = (1/c) * I_K, allowed=None,
    reference=None -- the non-gated, non-reference contract), but instead of
    freezing beta at truth it LEARNS beta. Each EM sweep:

      E-step: for every train doc infer the MAP eta_hat, form
              theta = softmax(eta_hat), and accumulate the LDA/STM
              suff-stat phi * counts into lambda_stats (K, V).
      M-step: full-batch conjugate update of the Dirichlet posterior on beta,
              lambda = eta + lambda_stats  (the rho=1 / batch-size special case
              of OnlineSTM.update_global's SVI step target_lam = eta +
              lambda_stats, which -- unlike LDA -- already folds expElogbeta
              into lambda_stats via phi in local_update; see stm.py).

    Only beta (lambda) is learned here; Sigma is HELD at c * I so the swept
    knob c is a pure concentration prior, exactly as in the frozen-beta sweep.
    Returns the posterior-mean topic-word matrix beta_hat = lambda /
    lambda.sum(axis=1) (K, V), a proper stochastic matrix suitable for both
    the predictive scoring (theta @ beta_hat) and the beta-recovery metric.

    eta (the symmetric Dirichlet prior on beta rows) defaults to 1/K, matching
    OnlineSTM's default. Reference: Roberts, Stewart, Airoldi 2016 (STM);
    Hoffman, Blei, Bach 2010 (the online-VB M-step this batch-specializes).
    """
    if eta is None:
        eta = 1.0 / K
    rng = np.random.default_rng(seed)
    lam = rng.gamma(shape=100.0, scale=1.0 / 100.0, size=(K, V))
    Gamma = np.zeros((1, K))
    x = np.array([1.0])
    Sigma_inv = (1.0 / c) * np.eye(K)

    for _ in range(n_em_iter):
        expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))
        lambda_stats = np.zeros((K, V), dtype=np.float64)
        for doc in train_docs:
            eta_hat, _, _ = _stm_doc_inference(
                indices=doc.indices, counts=doc.counts, expElogbeta=expElogbeta,
                Gamma=Gamma, Sigma_inv_allowed=Sigma_inv, x=x,
                max_iter=lbfgs_max_iter, tol=lbfgs_tol, allowed=None, reference=None,
            )
            p = _softmax(eta_hat)
            eb_d = expElogbeta[:, doc.indices]
            q_w = eb_d.T @ p + 1e-100
            phi = (eb_d * p[:, None]) / q_w[None, :]
            lambda_stats[:, doc.indices] += phi * doc.counts[None, :]
        lam = eta + lambda_stats

    return lam / lam.sum(axis=1, keepdims=True)


def lda_cofit_beta(
    train_docs: list, K: int, V: int, *, alpha, eta: float | None = None,
    n_em_iter: int = 60, seed: int = 0, optimize_alpha: bool = False,
    cavi_max_iter: int = 100, cavi_tol: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Co-fit the LDA topic-word matrix beta by full-batch variational EM at a
    Dirichlet document prior alpha (scalar or length-K).

    Co-fit analog of lda_recover_theta: same per-doc CAVI E-step
    (_cavi_doc_inference) but learning beta. Each EM sweep:

      E-step: per-doc CAVI -> (gamma, expElogthetad, phi_norm); accumulate
              lambda_stats via outer(expElogthetad, counts / phi_norm)
              (Lee/Seung 2001 collapsed-phi trick).
      M-step: lambda = eta + expElogbeta * lambda_stats  -- the rho=1 /
              batch special case of OnlineLDA.update_global (note LDA applies
              expElogbeta at the M-step, unlike STM).

    When optimize_alpha is True, alpha is additionally re-optimized each sweep
    by one alpha_newton_step on the corpus-summed E[log theta] (Blei 2003
    A.4.2 empirical-Bayes update), floored at 1e-3 -- the full-batch analog of
    OnlineLDA(optimize_alpha=True) and of lda_optimize_alpha (frozen-beta),
    but with beta LEARNING alongside. This is the head-to-head with real-data
    LDA, which co-fits topics AND optimizes alpha.

    Returns (beta_hat, alpha) where beta_hat = lambda / lambda.sum(axis=1)
    (K, V) is the posterior-mean topic-word matrix and alpha is the final
    length-K Dirichlet prior (unchanged from the input when optimize_alpha is
    False). eta defaults to 1/K (OnlineLDA default).
    """
    if eta is None:
        eta = 1.0 / K
    alpha_arr = np.asarray(alpha, dtype=np.float64)
    if alpha_arr.ndim == 0:
        alpha_arr = np.full(K, float(alpha_arr))
    rng = np.random.default_rng(seed)
    lam = rng.gamma(shape=100.0, scale=1.0 / 100.0, size=(K, V))

    for _ in range(n_em_iter):
        expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))
        lambda_stats = np.zeros((K, V), dtype=np.float64)
        e_log_theta_sum = np.zeros(K, dtype=np.float64)
        for doc in train_docs:
            gamma_init = np.full(K, 100.0 / K)
            gamma, expElogthetad, phi_norm, _ = _cavi_doc_inference(
                doc.indices, doc.counts, expElogbeta, alpha_arr, gamma_init,
                cavi_max_iter, cavi_tol,
            )
            lambda_stats[:, doc.indices] += np.outer(expElogthetad, doc.counts / phi_norm)
            if optimize_alpha:
                e_log_theta_sum += digamma(gamma) - digamma(gamma.sum())
        lam = eta + expElogbeta * lambda_stats
        if optimize_alpha:
            delta_alpha = alpha_newton_step(
                alpha=alpha_arr, e_log_theta_sum_scaled=e_log_theta_sum,
                D=float(len(train_docs)),
            )
            alpha_arr = np.maximum(alpha_arr + delta_alpha, 1e-3)

    return lam / lam.sum(axis=1, keepdims=True), alpha_arr


def beta_recovery_error(beta_true: np.ndarray, beta_hat: np.ndarray) -> dict:
    """Permutation-invariant topic-recovery error between a planted beta_true
    (K, V) and a recovered beta_hat (K, V).

    Topic labels are arbitrary (the model has no way to know which recovered
    row corresponds to which planted row), so a raw row-by-row comparison is
    meaningless. Solve the optimal one-to-one topic assignment first -- the
    linear assignment / bipartite-matching problem, minimum-cost perfect
    matching on the K x K cost matrix C_ij = 1 - cosine(beta_true_i,
    beta_hat_j) -- via the Hungarian algorithm (Kuhn 1955, "The Hungarian
    method for the assignment problem", Naval Research Logistics Quarterly
    2(1); solved here by scipy.optimize.linear_sum_assignment, which uses the
    Jonker-Volgenant / Crouse 2016 shortest-augmenting-path variant). Then
    report, over the matched pairs:

      mean_l1:  mean over topics of the L1 distance ||beta_true_i -
                beta_hat_match(i)||_1 (in [0, 2]); 0 = exact recovery.
      mean_cos_dist: mean over topics of 1 - cosine similarity (in [0, 2]);
                the quantity the assignment minimizes.

    Returns {"mean_l1": ..., "mean_cos_dist": ..., "row_ind": [...],
    "col_ind": [...]} with the matched index arrays for reproducibility.
    """
    tn = beta_true / (np.linalg.norm(beta_true, axis=1, keepdims=True) + 1e-300)
    hn = beta_hat / (np.linalg.norm(beta_hat, axis=1, keepdims=True) + 1e-300)
    cost = 1.0 - tn @ hn.T                                  # (K, K) cosine distance
    row_ind, col_ind = linear_sum_assignment(cost)
    l1 = np.abs(beta_true[row_ind] - beta_hat[col_ind]).sum(axis=1)
    cos_dist = cost[row_ind, col_ind]
    return {
        "mean_l1": float(l1.mean()),
        "mean_cos_dist": float(cos_dist.mean()),
        "row_ind": row_ind.tolist(),
        "col_ind": col_ind.tolist(),
    }


def beta_sharpness(beta: np.ndarray, *, top_k: int = 10) -> dict:
    """How PEAKED each topic's word distribution is, averaged over topics.

    Two complementary readouts, both means over the K topic rows of a (K, V)
    stochastic matrix:

      top_k_mass:  mean over topics of the summed probability of each topic's
                   top_k highest-probability terms (in [0, 1]); higher = a
                   sharper topic that concentrates on a few words.
      eff_vocab:   mean over topics of the inverse-Simpson index over the
                   vocabulary, 1 / sum_v beta_kv^2 (Hill number of order 2;
                   Hill 1973, Jost 2006 -- the same diversity number the
                   per-document eff_topics uses, applied to the term axis).
                   LOWER = a sharper topic (fewer effective terms).

    top_k_mass and eff_vocab move in opposite directions with sharpness, so a
    genuine LDA-vs-STM beta-sharpening effect shows up as HIGHER top_k_mass AND
    LOWER eff_vocab for the sharper model -- reporting both guards against a
    top_k artifact.
    """
    sorted_desc = np.sort(beta, axis=1)[:, ::-1]
    top_k_mass = float(sorted_desc[:, :top_k].sum(axis=1).mean())
    eff_vocab = float((1.0 / np.sum(beta * beta, axis=1)).mean())
    return {"top_k_mass": top_k_mass, "eff_vocab": eff_vocab}


def sweep_heldout_cofit(
    train_docs: list, test_docs: list, K: int, V: int, *, method: str, knobs: list,
    n_em_iter: int, holdout_frac: float = 0.3, seed: int = 0,
) -> dict:
    """Co-fit-beta analog of sweep_heldout: for each concentration knob, LEARN
    beta on train_docs at that knob, then score document-completion held-out
    predictive-LL on the (disjoint) test_docs under the LEARNED beta.

    Training beta on train and scoring completion on unseen test documents is
    the standard leakage-free topic-model evaluation (document completion;
    Wallach, Murray, Salakhutdinov, Mimno 2009, "Evaluation methods for topic
    models", ICML; Asuncion, Welling, Smyth, Teh 2009). The argmax knob is the
    held-out-LL-calibrated concentration -- the same gold standard insight 0038
    validated on frozen beta, now applied with beta co-fit.

    method == "stm": knobs are Sigma scales c (stm_cofit_beta + stm_heldout_ll).
    method == "lda": knobs are Dirichlet alphas (lda_cofit_beta + lda_heldout_ll).

    Returns {"lls": {knob: mean_test_ll}, "argmax_knob": <best>,
    "beta_hat": {knob: beta_hat (K, V)}} -- the per-knob learned beta is
    returned so the caller can read sharpness / recovery at the argmax without
    re-fitting.
    """
    lls: dict = {}
    betas: dict = {}
    for knob in knobs:
        if method == "stm":
            beta_hat = stm_cofit_beta(train_docs, K, V, c=knob, n_em_iter=n_em_iter, seed=seed)
            ll = stm_heldout_ll(test_docs, beta_hat, c=knob, holdout_frac=holdout_frac, seed=seed)
        elif method == "lda":
            beta_hat, _ = lda_cofit_beta(train_docs, K, V, alpha=knob, n_em_iter=n_em_iter, seed=seed)
            ll = lda_heldout_ll(test_docs, beta_hat, alpha=knob, holdout_frac=holdout_frac, seed=seed)
        else:
            raise ValueError(f"sweep_heldout_cofit: unknown method {method!r}")
        lls[knob] = ll
        betas[knob] = beta_hat
    argmax_knob = max(lls, key=lls.get)
    return {"lls": lls, "argmax_knob": argmax_knob, "beta_hat": betas}
