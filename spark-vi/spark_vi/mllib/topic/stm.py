"""StreamingSTM: MLlib-shim estimator for OnlineSTM.

Two input paths:
  (A) Caller supplies a pre-built `covariates` DenseVector column and
      a list of covariate names. No formulaic dependency required.
  (B) Caller supplies a `covariate_formula` string + a covariate
      DataFrame. Requires the `formula` extra: pip install spark-vi[formula].

Path B is implemented via `covariate_formula`; see `_resolve_model_spec_from_pandas`
and `_formula.fit_model_spec`.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from spark_vi.models.topic.stm import _stm_doc_inference, prior_topic_proportions


def _formula_mentions(group_var: str, covariate_names: list[str]) -> bool:
    """True if group_var appears as a factor in any design-column name,
    e.g. group_var='source_cohort' matches 'C(source_cohort)[T.dementia]'."""
    needle = f"({group_var})"
    return any(needle in name or name == group_var for name in covariate_names)


def corpus_mean_topic_proportions_rdd(cov_rdd, Gamma: np.ndarray, depth: int = 2):
    """Distributed α-equivalent: (1/D) Σ_d softmax(Γᵀ x_d) over an RDD.

    ``cov_rdd`` is an RDD of length-P covariate vectors (bare numpy arrays —
    no person_id, honoring the spark-vi layering rule). Mirrors the engine's
    mapPartitions+treeReduce idiom (see ``core/runner.py``): each partition
    accumulates a (K-vector sum, count) locally and the tree-reduce combines
    them, so only a K-vector and a scalar ever reach the driver. Scales to any
    D and any covariate cardinality — continuous covariates included.

    Γ is broadcast via the Spark-safe default-arg closure convention. Returns
    a length-K probability vector.
    """
    sc = cov_rdd.context
    bcast = sc.broadcast(Gamma)

    def _local(rows, _bcast=bcast):
        G = _bcast.value
        acc = np.zeros(G.shape[1], dtype=np.float64)
        n = 0
        for x in rows:
            acc += prior_topic_proportions(G, np.asarray(x, dtype=np.float64))
            n += 1
        return [(acc, n)]

    def _combine(a, b):
        return a[0] + b[0], a[1] + b[1]

    sum_vec, count = cov_rdd.mapPartitions(_local).treeReduce(_combine, depth=depth)
    if count == 0:
        raise ValueError("corpus_mean_topic_proportions_rdd: empty covariate RDD")
    return sum_vec / count


def corpus_mean_topic_proportions_gated_rdd(
    cov_group_rdd, Gamma: np.ndarray, partition, depth: int = 2,
):
    """Distributed gated α-equivalent: (1/D) Σ_d softmax_allowed(Γᵀ x_d).

    The gating-aware counterpart of ``corpus_mean_topic_proportions_rdd``: each
    document's softmax is taken over its ALLOWED topic set only
    (``partition.allowed_indices(groups)`` — background ∪ its groups' foreground
    blocks), so a foreground topic's corpus-mean prevalence reflects only its
    group's share and disallowed topics contribute exactly 0. Distributed via the
    same mapPartitions+treeReduce idiom — only a K-vector + count reach the
    driver — so it scales where the driver-side numpy
    ``corpus_mean_topic_proportions_gated`` (which needs the full design matrix
    collected) does not. Output is identical to that pure-numpy version.

    ``cov_group_rdd`` is an RDD of ``(x, groups)`` pairs: ``x`` a length-P
    covariate vector (bare array — no person_id, honoring the spark-vi layering
    rule) and ``groups`` a frozenset[str] of the doc's gating-group labels. Γ and
    the (small, frozen) partition are broadcast. Returns a length-K probability
    vector.
    """
    sc = cov_group_rdd.context
    g_bcast = sc.broadcast(np.asarray(Gamma, dtype=np.float64))
    p_bcast = sc.broadcast(partition)

    def _local(rows, _g=g_bcast, _p=p_bcast):
        G = _g.value
        part = _p.value
        acc = np.zeros(G.shape[1], dtype=np.float64)
        n = 0
        for x, groups in rows:
            allowed = part.allowed_indices(groups)
            e = (np.asarray(x, dtype=np.float64) @ G)[allowed]
            e = e - e.max()
            p = np.exp(e)
            p = p / p.sum()
            acc[allowed] += p
            n += 1
        return [(acc, n)]

    def _combine(a, b):
        return a[0] + b[0], a[1] + b[1]

    sum_vec, count = cov_group_rdd.mapPartitions(_local).treeReduce(_combine, depth=depth)
    if count == 0:
        raise ValueError("corpus_mean_topic_proportions_gated_rdd: empty covariate RDD")
    return sum_vec / count


def _welford_update(n, mean, M2, eta_hat, allowed):
    """Per-topic streaming-mean/variance update (Welford 1962, as formalized by
    Chan, Golub & LeVeque 1979 "Updating Formulae and a Pairwise Algorithm for
    Computing Sample Variances") for ONE new observation eta_hat, restricted to
    the doc's allowed (gated) topic set.

    n, mean, M2 are length-K arrays of per-topic running count / mean / sum of
    squared deviations. Only the `allowed` topics are touched — a topic a doc
    does not allow (eta_hat = -inf there) must not perturb that topic's
    accumulator at all, not even with a zero, or a background-only doc would
    silently pollute a foreground topic's variance.

    Returns the updated (n, mean, M2) triples (new arrays; inputs untouched).
    """
    n = n.copy()
    mean = mean.copy()
    M2 = M2.copy()
    x = eta_hat[allowed]
    n[allowed] += 1.0
    delta = x - mean[allowed]
    mean[allowed] += delta / n[allowed]
    delta2 = x - mean[allowed]
    M2[allowed] += delta * delta2
    return n, mean, M2


def _welford_combine(a, b):
    """Parallel combine of two per-topic Welford accumulators (Chan, Golub &
    LeVeque 1979, section 1.3 "Updating Formulae and a Pairwise Algorithm for
    Computing Sample Variances", the pairwise/parallel formula generalizing
    Welford's 1962 online update): for each topic k independently,

        n_ab   = n_a + n_b
        delta  = mean_b - mean_a
        mean_ab = mean_a + delta * n_b / n_ab        (n_ab > 0)
        M2_ab   = M2_a + M2_b + delta^2 * n_a * n_b / n_ab

    Elementwise over length-K arrays; topics with n_a == n_b == 0 stay at
    n=0, mean=0, M2=0 (division guarded).
    """
    n_a, mean_a, M2_a = a
    n_b, mean_b, M2_b = b
    n_ab = n_a + n_b
    safe_n_ab = np.where(n_ab > 0, n_ab, 1.0)
    delta = mean_b - mean_a
    mean_ab = mean_a + delta * (n_b / safe_n_ab)
    M2_ab = M2_a + M2_b + delta * delta * (n_a * n_b / safe_n_ab)
    mean_ab = np.where(n_ab > 0, mean_ab, 0.0)
    M2_ab = np.where(n_ab > 0, M2_ab, 0.0)
    return n_ab, mean_ab, M2_ab


def _welford_variance(n, M2, *, reference=None):
    """Finalize per-topic variance from Welford accumulators: M2_k/(n_k-1)
    where n_k > 1, else 0. The reference topic (eta pinned to 0 for every
    doc) is forced to exactly 0 for cleanliness, even though its natural
    variance is already ~0 by construction."""
    var = np.where(n > 1, M2 / np.maximum(n - 1.0, 1.0), 0.0)
    if reference is not None:
        var = var.copy()
        var[reference] = 0.0
    return var


def corpus_eta_variance_gated(
    docs, global_params, partition, *,
    lbfgs_max_iter=50, lbfgs_tol=1e-4, reference=None,
):
    """Driver-side (in-memory) empirical per-topic variance of document η
    (the logistic-normal generative concentration scale): for each doc, runs
    the same per-doc Laplace E-step as ``infer_local`` to get η̂ over the
    doc's ALLOWED (gated) topic set, then accumulates a per-topic streaming
    mean + M2 (Welford; see ``_welford_update``) over documents, skipping
    topics the doc does not allow (η = -inf there). A topic allowed by zero
    documents, and the reference topic (η pinned to 0), get variance 0.

    Gating: a foreground topic's variance reflects only the documents in its
    own group — background-only documents never touch a foreground topic's
    accumulator, because their allowed set (``partition.allowed_indices``)
    excludes it entirely.

    This is the between-document η spread — how much real documents actually
    spread out in η-space — used downstream to rescale the unit-diagonal
    correlation matrix Σ (ADR 0034) into a generative covariance: the fitted
    correlation captures topic co-movement direction, but its unit-diagonal
    convention discards scale, so generated documents need this empirical
    per-topic variance to be realistically (rather than maximally) concentrated.

    ``docs`` is an in-memory list of STMDocument. ``global_params`` is the
    dict with "lambda" (K,V), "Gamma" (P,K), "Sigma" (K,K) — exactly what
    ``infer_local`` reads. ``partition`` is a TopicBlockPartition (or the
    implicit all-background one). Mirrors how ``corpus_mean_topic_proportions_gated``
    (numpy) sits beside its ``_rdd`` counterpart: useful for tests and small
    corpora; for a live cluster corpus use ``corpus_eta_variance_gated_rdd``.
    """
    from spark_vi.models.topic._linalg import safe_inverse
    from scipy.special import digamma

    lam = np.asarray(global_params["lambda"], dtype=np.float64)
    Gamma = np.asarray(global_params["Gamma"], dtype=np.float64)
    Sigma = np.asarray(global_params["Sigma"], dtype=np.float64)
    K = lam.shape[0]
    expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))

    n = np.zeros(K, dtype=np.float64)
    mean = np.zeros(K, dtype=np.float64)
    M2 = np.zeros(K, dtype=np.float64)
    subprec_cache: dict[tuple, np.ndarray] = {}

    for doc in docs:
        allowed = partition.allowed_indices(doc.groups)
        key = tuple(allowed.tolist())
        Sigma_inv_allowed = subprec_cache.get(key)
        if Sigma_inv_allowed is None:
            Sigma_inv_allowed = safe_inverse(Sigma[np.ix_(allowed, allowed)])
            subprec_cache[key] = Sigma_inv_allowed
        eta_hat, _, _ = _stm_doc_inference(
            indices=doc.indices, counts=doc.counts,
            expElogbeta=expElogbeta,
            Gamma=Gamma, Sigma_inv_allowed=Sigma_inv_allowed, x=doc.x,
            max_iter=lbfgs_max_iter, tol=lbfgs_tol,
            allowed=allowed, reference=reference,
        )
        n, mean, M2 = _welford_update(n, mean, M2, eta_hat, allowed)

    return _welford_variance(n, M2, reference=reference)


def corpus_eta_variance_gated_rdd(
    doc_rdd, global_params, partition, *,
    lbfgs_max_iter=50, lbfgs_tol=1e-4, reference=None, depth=2,
):
    """Distributed empirical per-topic variance of document η (the logistic-
    normal generative concentration scale). For each STMDocument, runs the same
    per-doc Laplace E-step as ``infer_local`` to get η̂_d over the doc's ALLOWED
    (gated) topic set, then accumulates a per-topic streaming mean + M2
    (Welford) over documents, skipping topics the doc does not allow
    (η = -inf). Returns a length-K vector of per-topic variances; a topic
    allowed by zero documents, and the reference topic (η pinned to 0), get
    variance 0.

    Gating: a foreground topic's variance reflects only its own group's
    documents (``partition.allowed_indices(doc.groups)``); a background-only
    document's allowed set excludes every foreground topic, so it never
    contributes to one.

    Distributed via the same mapPartitions + treeReduce idiom as
    ``corpus_mean_topic_proportions_gated_rdd``: each partition runs the full
    per-doc E-step and accumulates a local (n, mean, M2) triple of length-K
    arrays; the tree-reduce combines partitions pairwise via the parallel
    Welford formula (Chan, Golub & LeVeque 1979) — only three K-vectors per
    partition ever cross the network, never the documents or per-doc η̂/ν_d.
    ``global_params`` and ``partition`` are broadcast via the Spark-safe
    default-arg closure convention (see ``corpus_mean_topic_proportions_gated_rdd``).

    This is the between-document η spread used to rescale the unit-diagonal
    fitted correlation Σ (ADR 0034) into a generative covariance downstream: the
    fitted correlation is unit-diagonal by construction (variance pinned to 1
    for fitting stability), so a consumer that wants to *generate* documents
    with realistic concentration needs this empirical per-topic variance to
    rescale it, rather than drawing from an over-diffuse unit-variance prior.

    ``doc_rdd`` is an RDD of STMDocument. Returns a length-K numpy array.
    """
    sc = doc_rdd.context
    gp_bcast = sc.broadcast({
        k: np.asarray(v, dtype=np.float64) for k, v in global_params.items()
    })
    p_bcast = sc.broadcast(partition)

    def _local(rows, _gp=gp_bcast, _p=p_bcast):
        from spark_vi.models.topic._linalg import safe_inverse
        from scipy.special import digamma

        gp = _gp.value
        part = _p.value
        lam = gp["lambda"]
        Gamma = gp["Gamma"]
        Sigma = gp["Sigma"]
        K = lam.shape[0]
        expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))

        n = np.zeros(K, dtype=np.float64)
        mean = np.zeros(K, dtype=np.float64)
        M2 = np.zeros(K, dtype=np.float64)
        subprec_cache: dict[tuple, np.ndarray] = {}
        n_docs = 0

        for doc in rows:
            allowed = part.allowed_indices(doc.groups)
            key = tuple(allowed.tolist())
            Sigma_inv_allowed = subprec_cache.get(key)
            if Sigma_inv_allowed is None:
                Sigma_inv_allowed = safe_inverse(Sigma[np.ix_(allowed, allowed)])
                subprec_cache[key] = Sigma_inv_allowed
            eta_hat, _, _ = _stm_doc_inference(
                indices=doc.indices, counts=doc.counts,
                expElogbeta=expElogbeta,
                Gamma=Gamma, Sigma_inv_allowed=Sigma_inv_allowed, x=doc.x,
                max_iter=lbfgs_max_iter, tol=lbfgs_tol,
                allowed=allowed, reference=reference,
            )
            n, mean, M2 = _welford_update(n, mean, M2, eta_hat, allowed)
            n_docs += 1
        return [(n, mean, M2, n_docs)]

    def _combine(a, b):
        n_ab, mean_ab, M2_ab = _welford_combine(a[:3], b[:3])
        return n_ab, mean_ab, M2_ab, a[3] + b[3]

    n, mean, M2, n_docs = doc_rdd.mapPartitions(_local).treeReduce(_combine, depth=depth)
    if n_docs == 0:
        raise ValueError("corpus_eta_variance_gated_rdd: empty document RDD")
    return _welford_variance(n, M2, reference=reference)


def _pool_scale(n, M2, nu_sum, nu_count, *, reference):
    """Pool the per-topic law-of-total-variance totals into ONE scalar scale c.

    Law of total variance (Weiss 2005, "A Course in Probability", thm. 4.4.7):
    for a topic k, the marginal spread of eta decomposes as

        Var(eta_k) = Var_d(E[eta_k | d]) + E_d(Var[eta_k | d])
                   = (between-doc variance of the posterior mode eta_hat_k)
                     + (mean over docs of the Laplace posterior variance nu_d_k).

    The first term is the finalized Welford variance M2_k/(n_k-1); the second is
    nu_sum_k / nu_count_k. Their sum is topic k's total generative variance.

    The single pooled scale is the MEAN of those totals over the FREE, OBSERVED
    topics: the reference topic (eta pinned to 0, ~0 total) is excluded so its
    near-zero does not drag the mean down, and any topic seen by fewer than 2
    documents (n_k < 2, Welford variance undefined) is excluded so a NaN cannot
    corrupt c. Pooling to a single number is the runaway-safe estimator: a
    per-topic free diagonal reopened the insight-0033 variance runaway (a
    low-ess topic's noise self-amplified), whereas a low-ess topic's noise
    averaged against every other topic's -- with beta frozen -- cannot.

    Returns (c, totals) where totals is the length-K per-topic total variance
    (0 where a topic is excluded from the pool), for introspection/tests.
    """
    K = n.shape[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        between = np.where(n > 1, M2 / np.maximum(n - 1.0, 1.0), 0.0)
        within = np.where(nu_count > 0, nu_sum / np.maximum(nu_count, 1.0), 0.0)
    totals = between + within
    in_pool = n >= 2.0
    if reference is not None:
        in_pool = in_pool.copy()
        in_pool[reference] = False
    totals = np.where(in_pool, totals, 0.0)
    n_pool = int(in_pool.sum())
    if n_pool == 0:
        raise ValueError(
            "corpus_eta_scale: no free topic was observed by >= 2 documents; "
            "cannot pool a generative-variance scale")
    c = float(totals[in_pool].mean())
    return c, totals


def corpus_eta_scale_gated(
    docs, global_params, partition, *,
    reference=None, max_iter=15, tol=0.02,
    lbfgs_max_iter=50, lbfgs_tol=1e-4,
    _return_trace=False, _return_totals=False,
):
    """Driver-side (in-memory) iterated pooled generative-variance scale c for
    the STM generative covariance Sigma_gen = c*R, with beta and R FROZEN.

    R = ``global_params["Sigma"]`` is the fit's unit-diagonal block-wise
    correlation (ADR 0034). Its unit diagonal discards the generative concentration
    scale, so a faithful generative simulator needs a single variance level c such
    that Sigma_gen = c*R produces documents that spread in eta-space like the real
    corpus. This estimates that ONE scalar by an iterated pooled EM: each round runs
    the same per-doc Laplace E-step as ``infer_local`` under the prior Sigma = c*R,
    accumulates -- per free, observed topic -- both terms of the law of total
    variance Var(eta)=Var_d(E[eta|d])+E_d(Var[eta|d]) (the between-doc variance of
    the posterior mode eta_hat via Welford, plus the mean per-doc Laplace posterior
    variance nu_d), pools those per-topic totals to one scalar over the free
    observed topics, and updates c toward the self-consistent value.

    Why iterate: one E-step under the unit prior underestimates the scale
    (posterior modes shrink toward the prior mean); re-broadcasting the prior as
    c*R each round lets c climb to the self-consistent level. Empirically converges
    in ~5-10 rounds and stays bounded because beta is frozen and c is a single
    pooled number -- a low-ess topic's noise is averaged against every other
    topic's, so it cannot self-amplify the way a per-topic free diagonal did
    (insight 0033 variance runaway). It under-corrects modestly (Laplace
    posterior-variance bias); that is expected and acceptable.

    ``docs`` is an in-memory list of STMDocument; ``global_params`` is the dict with
    "lambda", "Gamma", "Sigma" (= R); ``partition`` is a TopicBlockPartition (or the
    implicit all-background one). The prior precision each round is
    ``(1/c) * safe_inverse(R[allowed])`` -- ``safe_inverse(R[allowed])`` is
    INDEPENDENT of c, so it is computed once per distinct allowed set (cached by
    ``tuple(allowed)``) and merely rescaled by 1/c. Returns the converged scalar c.
    For a live cluster corpus use ``corpus_eta_scale_gated_rdd``.
    """
    from spark_vi.models.topic._linalg import safe_inverse
    from scipy.special import digamma

    lam = np.asarray(global_params["lambda"], dtype=np.float64)
    Gamma = np.asarray(global_params["Gamma"], dtype=np.float64)
    Sigma = np.asarray(global_params["Sigma"], dtype=np.float64)   # R (unit-diag)
    K = lam.shape[0]
    expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))

    # safe_inverse(R[allowed]) is c-independent: cache by allowed-set key, once.
    rinv_cache: dict[tuple, np.ndarray] = {}
    allowed_cache: dict[int, np.ndarray] = {}

    c = 1.0
    trace: list[float] = []
    totals = np.zeros(K, dtype=np.float64)
    for _ in range(max_iter):
        n = np.zeros(K, dtype=np.float64)
        mean = np.zeros(K, dtype=np.float64)
        M2 = np.zeros(K, dtype=np.float64)
        nu_sum = np.zeros(K, dtype=np.float64)
        nu_count = np.zeros(K, dtype=np.float64)
        inv_c = 1.0 / c
        for di, doc in enumerate(docs):
            allowed = allowed_cache.get(di)
            if allowed is None:
                allowed = partition.allowed_indices(doc.groups)
                allowed_cache[di] = allowed
            key = tuple(allowed.tolist())
            rinv = rinv_cache.get(key)
            if rinv is None:
                rinv = safe_inverse(Sigma[np.ix_(allowed, allowed)])
                rinv_cache[key] = rinv
            eta_hat, nu_d, _ = _stm_doc_inference(
                indices=doc.indices, counts=doc.counts,
                expElogbeta=expElogbeta,
                Gamma=Gamma, Sigma_inv_allowed=inv_c * rinv, x=doc.x,
                max_iter=lbfgs_max_iter, tol=lbfgs_tol,
                allowed=allowed, reference=reference,
            )
            n, mean, M2 = _welford_update(n, mean, M2, eta_hat, allowed)
            nu_diag = np.diag(nu_d)
            nu_sum[allowed] += nu_diag[allowed]
            nu_count[allowed] += 1.0
        c_new, totals = _pool_scale(n, M2, nu_sum, nu_count, reference=reference)
        trace.append(c_new)
        if abs(c_new - c) <= tol * c:
            c = c_new
            break
        c = c_new

    if _return_trace:
        return trace
    if _return_totals:
        return totals
    return float(c)


def corpus_eta_scale_gated_rdd(
    doc_rdd, global_params, partition, *,
    reference=None, max_iter=15, tol=0.02,
    sample_fraction=None, sample_seed=0,
    lbfgs_max_iter=50, lbfgs_tol=1e-4, depth=2,
):
    """Distributed iterated pooled generative-variance scale c for the STM
    generative covariance Sigma_gen = c*R, with beta and R FROZEN (see
    ``corpus_eta_scale_gated`` for the full derivation; this is its Spark
    counterpart, byte-for-byte the same math on a distributed corpus).

    Each OUTER round re-broadcasts the current scale c and runs ONE distributed
    E-step pass: each partition runs the same per-doc Laplace E-step as
    ``infer_local`` under the prior Sigma = c*R (precision ``(1/c) *
    safe_inverse(R[allowed])``), accumulating a local per-topic Welford triple
    (n, mean, M2) of the posterior mode eta_hat PLUS a per-topic (nu_sum,
    nu_count) of the Laplace posterior variance nu_d. The tree-reduce combines
    partitions pairwise (parallel Welford, Chan/Golub/LeVeque 1979, for the
    between-doc term; plain sums for the nu term) -- only five K-vectors per
    partition ever cross the network, never the documents. The driver pools the
    per-topic law-of-total-variance totals to a single scalar over the free,
    observed topics and updates c until ``abs(c_new - c) <= tol*c``.

    ``safe_inverse(R[allowed])`` is c-independent, so it is cached per distinct
    allowed set inside each partition and only rescaled by 1/c each round.
    Convergence is logged per round on the driver (``[eta_scale] iter ...``) so
    the export shows the trace.

    ``sample_fraction``: a single pooled scalar needs only a sample. When set,
    the rdd is sampled (without replacement) and cached ONCE before the loop and
    every round iterates on that sample -- keeping the iterated cost near a single
    full pass. When None, the full rdd is cached before the loop (it is traversed
    up to max_iter times). ``global_params`` and ``partition`` are broadcast via
    the Spark-safe default-arg closure convention; the scalar c is re-broadcast
    each round. Returns the converged scalar c.
    """
    from pyspark import StorageLevel

    sc = doc_rdd.context
    gp_bcast = sc.broadcast({
        k: np.asarray(v, dtype=np.float64) for k, v in global_params.items()
    })
    p_bcast = sc.broadcast(partition)

    # Sample (a scalar needs only a sample) or take the full rdd, and cache once:
    # the loop traverses it up to max_iter times.
    if sample_fraction is not None:
        work_rdd = doc_rdd.sample(
            withReplacement=False, fraction=sample_fraction, seed=sample_seed)
    else:
        work_rdd = doc_rdd
    work_rdd = work_rdd.persist(StorageLevel.MEMORY_AND_DISK)

    try:
        c = 1.0
        for it in range(max_iter):
            c_bcast = sc.broadcast(float(c))

            def _local(rows, _gp=gp_bcast, _p=p_bcast, _c=c_bcast):
                from spark_vi.models.topic._linalg import safe_inverse
                from scipy.special import digamma

                gp = _gp.value
                part = _p.value
                inv_c = 1.0 / _c.value
                lam = gp["lambda"]
                Gamma = gp["Gamma"]
                Sigma = gp["Sigma"]     # R (unit-diagonal correlation)
                K = lam.shape[0]
                expElogbeta = np.exp(
                    digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))

                n = np.zeros(K, dtype=np.float64)
                mean = np.zeros(K, dtype=np.float64)
                M2 = np.zeros(K, dtype=np.float64)
                nu_sum = np.zeros(K, dtype=np.float64)
                nu_count = np.zeros(K, dtype=np.float64)
                rinv_cache: dict[tuple, np.ndarray] = {}
                n_docs = 0

                for doc in rows:
                    allowed = part.allowed_indices(doc.groups)
                    key = tuple(allowed.tolist())
                    rinv = rinv_cache.get(key)
                    if rinv is None:
                        rinv = safe_inverse(Sigma[np.ix_(allowed, allowed)])
                        rinv_cache[key] = rinv
                    eta_hat, nu_d, _ = _stm_doc_inference(
                        indices=doc.indices, counts=doc.counts,
                        expElogbeta=expElogbeta,
                        Gamma=Gamma, Sigma_inv_allowed=inv_c * rinv, x=doc.x,
                        max_iter=lbfgs_max_iter, tol=lbfgs_tol,
                        allowed=allowed, reference=reference,
                    )
                    n, mean, M2 = _welford_update(n, mean, M2, eta_hat, allowed)
                    nu_diag = np.diag(nu_d)
                    nu_sum[allowed] += nu_diag[allowed]
                    nu_count[allowed] += 1.0
                    n_docs += 1
                return [(n, mean, M2, nu_sum, nu_count, n_docs)]

            def _combine(a, b):
                n_ab, mean_ab, M2_ab = _welford_combine(a[:3], b[:3])
                return (n_ab, mean_ab, M2_ab,
                        a[3] + b[3], a[4] + b[4], a[5] + b[5])

            n, mean, M2, nu_sum, nu_count, n_docs = (
                work_rdd.mapPartitions(_local).treeReduce(_combine, depth=depth))
            c_bcast.destroy()
            if n_docs == 0:
                raise ValueError("corpus_eta_scale_gated_rdd: empty document RDD")

            c_new, _ = _pool_scale(n, M2, nu_sum, nu_count, reference=reference)
            print(f"[eta_scale] iter {it}: c={c_new:.4f}")
            if abs(c_new - c) <= tol * c:
                c = c_new
                break
            c = c_new
        return float(c)
    finally:
        work_rdd.unpersist(blocking=False)


class StreamingSTM:
    """Streaming-VI estimator for OnlineSTM with DataFrame input.

    Constructor enforces that the caller supplies enough information
    to determine P (covariate dimension) — either via covariate_names
    (Path A) or covariate_formula (Path B).
    """

    def __init__(
        self,
        K: int,
        features_col: str = "features",
        covariates_col: str | None = None,
        covariate_names: list[str] | None = None,
        covariate_formula: str | None = None,
        covariate_df: Any | None = None,
        join_key: str | None = None,
        max_levels: int = 10_000,
        sigma_init: float = 1.0,
        sigma_ridge: float = 1e-6,
        min_pair_support: int = 1,
        lbfgs_max_iter: int = 50,
        lbfgs_tol: float = 1e-4,
        random_seed: int | None = None,
        reference_topic: bool = True,
        estimate_sigma_diagonal: bool = False,
        spectral_init: bool = True,
        spectral_method: str = "dense",          # "dense" | "scalable"
        spectral_d: int | None = None,           # scalable projection dim; None => ~1000
        spectral_min_doc_freq: int = 5,          # scalable absolute doc-frequency floor
        topic_blocks=None,
        doc_group_col: str | None = None,
    ) -> None:
        # Path A vs B validation.
        path_a = covariates_col is not None and covariate_names is not None
        path_b = covariate_formula is not None
        if not (path_a or path_b):
            raise ValueError(
                "StreamingSTM requires either (covariates_col + covariate_names) "
                "for Path A, or covariate_formula for Path B."
            )
        if path_a and path_b:
            raise ValueError("Use either Path A or Path B, not both.")

        self.K = int(K)
        self.features_col = features_col

        if path_a:
            if not covariate_names:
                raise ValueError("covariate_names must be non-empty for Path A.")
            self.covariates_col = covariates_col
            self.covariate_names = list(covariate_names)
            self.P = len(self.covariate_names)
            self.covariate_formula = None
        else:
            # Path B: uses formulaic ModelSpec for covariate resolution.
            self.covariates_col = "covariates"
            self.covariate_formula = covariate_formula
            self.covariate_df = covariate_df
            self.join_key = join_key
            self.max_levels = max_levels
            self.covariate_names = None       # set during fit
            self.P = None                     # set during fit

        self.sigma_init = sigma_init
        self.sigma_ridge = sigma_ridge
        self.min_pair_support = int(min_pair_support)
        self.lbfgs_max_iter = lbfgs_max_iter
        self.lbfgs_tol = lbfgs_tol
        self.random_seed = random_seed
        self.reference_topic = bool(reference_topic)
        self.estimate_sigma_diagonal = bool(estimate_sigma_diagonal)
        self.spectral_init = bool(spectral_init)
        if spectral_method not in {"dense", "scalable"}:
            raise ValueError(
                f"spectral_method must be 'dense' or 'scalable', got {spectral_method!r}")
        self.spectral_method = spectral_method
        self.spectral_d = spectral_d
        self.spectral_min_doc_freq = int(spectral_min_doc_freq)

        self.topic_blocks = topic_blocks
        self.doc_group_col = doc_group_col
        if topic_blocks is not None:
            if doc_group_col is None:
                raise ValueError(
                    "topic_blocks requires doc_group_col (the column naming each "
                    "document's gating group).")
            if self.covariate_names is not None and \
                    _formula_mentions(topic_blocks.group_var, self.covariate_names):
                raise ValueError(
                    f"group_var {topic_blocks.group_var!r} must not also appear in "
                    f"the prevalence formula (foreground regression would be "
                    f"rank-deficient); remove it from the formula terms.")
        elif doc_group_col is not None:
            raise ValueError("doc_group_col set without topic_blocks.")

    def fit(
        self,
        dataset,
        *,
        max_iter: int = 20,
        subsampling_rate: float = 0.2,
        tau0: float = 64.0,
        kappa: float = 0.7,
        save_interval: int | None = None,
        checkpoint_dir: str | None = None,
        on_iteration=None,
        resume_from: str | None = None,
    ) -> "STMModel":
        """Fit OnlineSTM via VIRunner on a DataFrame with features + covariates columns.

        The input DataFrame must have the configured `features_col` (SparseVector)
        and `covariates_col` (DenseVector). Vocab size is discovered from the
        first features row.

        Parameters:
            dataset: Spark DataFrame with `features_col` and `covariates_col`.
            max_iter: maximum number of SVI iterations.
            subsampling_rate: fraction of documents per mini-batch (maps to
                VIConfig.mini_batch_fraction).
            tau0: Robbins-Monro delay parameter (maps to
                VIConfig.learning_rate_tau0).
            kappa: Robbins-Monro decay exponent (maps to
                VIConfig.learning_rate_kappa).
            save_interval: if set, checkpoint every N iterations
                (requires checkpoint_dir).
            checkpoint_dir: directory for periodic checkpoints.
            on_iteration: optional per-iteration callback
                fn(iter_num, global_params, elbo_trace).
            resume_from: optional path to a previously-saved STMModel dir. When
                set, VIRunner loads its global_params + n_iterations and
                continues the Robbins-Monro schedule from there; max_iter is
                then ADDITIONAL iterations on top of the loaded count. The
                resumed corpus/covariate shapes (V, P) must match the loaded
                params, so resume only with the same corpus + formula.
        """
        from pyspark import StorageLevel

        from spark_vi.core.config import VIConfig
        from spark_vi.core.runner import VIRunner
        from spark_vi.mllib.topic._common import _vector_to_stm_document
        from spark_vi.models.topic.stm import OnlineSTM

        if self.covariate_names is None:
            raise ValueError(
                "StreamingSTM.fit requires covariate_names to be set. For Path A "
                "supply covariate_names at construction; for Path B call "
                "_resolve_model_spec_from_pandas first."
            )

        # Path B re-guard: covariate_names not known at construction time,
        # so re-run the formula check now that they are resolved.
        if self.topic_blocks is not None and \
                _formula_mentions(self.topic_blocks.group_var, self.covariate_names):
            raise ValueError(
                f"group_var {self.topic_blocks.group_var!r} must not also appear "
                f"in the prevalence formula.")

        first = dataset.select(self.features_col).head(1)
        if not first:
            raise ValueError("Cannot fit on an empty DataFrame.")
        vocab_size = first[0][0].size

        model = OnlineSTM(
            K=self.K,
            vocab_size=vocab_size,
            P=self.P,
            sigma_init=self.sigma_init,
            sigma_ridge=self.sigma_ridge,
            min_pair_support=self.min_pair_support,
            lbfgs_max_iter=self.lbfgs_max_iter,
            lbfgs_tol=self.lbfgs_tol,
            random_seed=self.random_seed,
            topic_blocks=self.topic_blocks,
            reference_topic=self.reference_topic,
            estimate_sigma_diagonal=self.estimate_sigma_diagonal,
        )

        # VIConfig uses learning_rate_tau0/kappa and mini_batch_fraction;
        # checkpoint_interval + checkpoint_dir must be both set or both None.
        checkpoint_kwargs: dict = {}
        if save_interval is not None and checkpoint_dir is not None:
            checkpoint_kwargs = {
                "checkpoint_interval": save_interval,
                "checkpoint_dir": checkpoint_dir,
            }

        config = VIConfig(
            max_iterations=max_iter,
            learning_rate_tau0=tau0,
            learning_rate_kappa=kappa,
            mini_batch_fraction=subsampling_rate if subsampling_rate < 1.0 else None,
            **checkpoint_kwargs,
        )

        features_col = self.features_col
        covariates_col = self.covariates_col
        group_col = self.doc_group_col
        select_cols = [features_col, covariates_col]
        if group_col is not None:
            select_cols.append(group_col)
        rdd = (
            dataset.select(*select_cols).rdd
            .map(lambda row: _vector_to_stm_document(
                {c: row[i] for i, c in enumerate(select_cols)},
                features_col=features_col,
                covariates_col=covariates_col,
                group_col=group_col,
            ))
        )
        rdd = rdd.persist(StorageLevel.MEMORY_AND_DISK)
        rdd.count()

        # Optional spectral (anchor-word) β seed (insight 0029, ADR 0031/0032).
        # spectral_method selects dense (exact V×V on the driver, the validated
        # default) vs scalable (random-projection sketch for large vocabularies).
        # Dense path: collect the (already-materialized) docs to the driver and
        # run the anchor-word init, handing the engine a deterministic, data-
        # driven β via data_summary={"spectral_beta": KxV}. `initialize_global`
        # seeds λ = β·gamma_shape from it instead of random gamma, curing the
        # sigma_init-dependent collapse/blow-up. Skipped when resuming, where the
        # runner restores λ from the checkpoint and never calls initialize_global.
        # Fine at the cancer scale (V≈3691, ~11k docs → ~18s/109MB); the large-V
        # scalable rewrite (distributed co-occurrence + random projection) is a
        # separate arc (ADR 0032).
        data_summary = None
        if self.spectral_init and resume_from is None:
            partition = model._effective_partition()
            if self.spectral_method == "scalable":
                from spark_vi.models.topic.spectral_init_scalable import (
                    scalable_spectral_init_beta,
                )
                beta0 = scalable_spectral_init_beta(
                    rdd, partition, vocab_size,
                    d=self.spectral_d,
                    seed=self.random_seed or 0,
                    min_doc_freq=self.spectral_min_doc_freq,
                )
            else:  # "dense" — current exact path, the default
                from spark_vi.models.topic.spectral_init import spectral_init_beta
                docs = rdd.collect()
                beta0 = spectral_init_beta(docs, partition, vocab_size)
            data_summary = {"spectral_beta": beta0}

        runner = VIRunner(model, config=config)
        try:
            result = runner.fit(
                rdd, data_summary=data_summary,
                on_iteration=on_iteration, resume_from=resume_from,
            )
        finally:
            rdd.unpersist(blocking=False)

        metadata = dict(result.metadata)
        if self.topic_blocks is not None:
            metadata.setdefault("topic_block_spec", self.topic_blocks.to_dict())
        # Provenance: record the opt-in hardening knobs that produced this fit.
        # Not load-bearing for the current export path (the dashboard prevalence
        # helpers use softmax(Gamma^T x), already correct under a reference fit
        # because Gamma[:, 0] = 0); persisted so a reloaded model's provenance is
        # complete and a future inference path can re-pin.
        metadata.setdefault("stm_hardening", {
            "reference_topic": self.reference_topic,
            "min_pair_support": self.min_pair_support,
            "spectral_init": self.spectral_init,
            "spectral_method": self.spectral_method,
            "estimate_sigma_diagonal": self.estimate_sigma_diagonal,
        })
        return STMModel(
            global_params=result.global_params,
            metadata=metadata,
            model_spec=getattr(self, "model_spec", None),
            covariate_names=list(self.covariate_names),
            n_iterations=result.n_iterations,
            elbo_trace=list(result.elbo_trace),
            converged=result.converged,
            diagnostic_traces=dict(result.diagnostic_traces),
            topic_blocks=self.topic_blocks,
        )

    def _resolve_model_spec_from_pandas(self, covariate_pdf):
        """Resolve P and covariate_names from a pre-collected pandas covariate DataFrame.

        Used by tests and by the in-memory Path-B construction. Production
        .fit() invocations against Spark DataFrames will use the
        schema-frame Spark discovery path instead (see _formula.fit_model_spec_from_spark and ADR 0024).
        """
        from spark_vi.mllib.topic._formula import fit_model_spec
        spec, names = fit_model_spec(self.covariate_formula, covariate_pdf)
        self.model_spec = spec
        self.covariate_names = names
        self.P = len(names)


class STMModel:
    """Fitted MLlib-shim STM model. Wraps OnlineSTM's global params + ModelSpec.

    Persistence layout under <model_dir> (VIResult-compatible):
        manifest.json           # metadata + elbo_trace (written by save_result)
        params/<name>.npy       # one file per global_param key
        model_spec.pkl          # formulaic ModelSpec (pickle sidecar)
        covariate_names.json    # list of covariate name strings (sidecar)
    """

    def __init__(
        self,
        global_params: dict[str, np.ndarray],
        metadata: dict[str, Any],
        model_spec: Any,
        covariate_names: list[str],
        n_iterations: int = 0,
        elbo_trace: list[float] | None = None,
        converged: bool = False,
        diagnostic_traces: dict | None = None,
        topic_blocks=None,
    ) -> None:
        self.global_params = global_params
        self.metadata = metadata
        self.model_spec = model_spec
        self.covariate_names = covariate_names
        # Resume state: n_iterations + elbo_trace are persisted so a later
        # fit(resume_from=...) continues the Robbins-Monro counter (rho_t
        # depends on the loaded iteration count) instead of restarting at t=0.
        self.n_iterations = n_iterations
        self.elbo_trace = list(elbo_trace) if elbo_trace is not None else []
        self.converged = converged
        self.diagnostic_traces = (
            dict(diagnostic_traces) if diagnostic_traces is not None else {}
        )
        self.topic_blocks = topic_blocks

    def save(self, out_dir: Path) -> None:
        from spark_vi.core.result import VIResult
        from spark_vi.io.export import save_result

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Wrap state as a VIResult so the canonical saver handles the
        # standard layout (manifest.json, params/*.npy, traces/).
        # load_result in build_dashboard_cloud.py expects this layout.
        result = VIResult(
            global_params=self.global_params,
            metadata=dict(self.metadata),
            elbo_trace=list(self.elbo_trace),
            n_iterations=self.n_iterations,
            converged=self.converged,
            # STM's per-iter diagnostics (2-D Gamma + 1-D Sigma + topic-block
            # labels) now round-trip: save_result persists ndarrays of any rank
            # as traces/<name>.npy of shape (n_iter, *dims). These are small
            # aggregate params (P*K + K floats per iter), safe to carry.
            diagnostic_traces=dict(self.diagnostic_traces),
        )
        save_result(result, out_dir)
        # Derived sidecar: correlation matrix R_ij = Sigma_ij / sqrt(Sigma_ii Sigma_jj).
        # Written to params/correlation.npy for downstream / dashboard consumers.
        # NOTE: STMModel.load does NOT restore this file — R is re-derivable from
        # the round-tripped Sigma via topic_correlation(). This sidecar is for
        # consumers that want R without re-loading and re-deriving the model.
        from spark_vi.models.topic._linalg import topic_correlation
        np.save(
            out_dir / "params" / "correlation.npy",
            topic_correlation(self.global_params["Sigma"]),
        )
        # Sidecars: formulaic ModelSpec + covariate names list.
        with (out_dir / "model_spec.pkl").open("wb") as f:
            pickle.dump(self.model_spec, f)
        (out_dir / "covariate_names.json").write_text(
            json.dumps(self.covariate_names)
        )

    @classmethod
    def load(cls, in_dir: Path) -> "STMModel":
        from spark_vi.io.export import load_result
        from spark_vi.models.topic.partition import TopicBlockPartition

        in_dir = Path(in_dir)
        result = load_result(in_dir)
        with (in_dir / "model_spec.pkl").open("rb") as f:
            spec = pickle.load(f)
        covariate_names = json.loads(
            (in_dir / "covariate_names.json").read_text()
        )
        spec_dict = result.metadata.get("topic_block_spec")
        topic_blocks = (
            TopicBlockPartition.from_dict(spec_dict) if spec_dict else None)
        return cls(
            global_params=result.global_params,
            metadata=dict(result.metadata),
            model_spec=spec,
            covariate_names=covariate_names,
            n_iterations=result.n_iterations,
            elbo_trace=list(result.elbo_trace),
            converged=result.converged,
            diagnostic_traces=dict(result.diagnostic_traces),
            topic_blocks=topic_blocks,
        )
