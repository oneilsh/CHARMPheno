"""MLlib Estimator / Model for OnlineSTM (the structural topic model), plus the
distributed STM evaluation helpers (corpus prevalence, concentration, Sigma scale).

``OnlineSTMEstimator`` / ``OnlineSTMModel`` are MLlib stages like the LDA, HDP and
DAG-gated shims (ADR 0056): Params, ``transform``, MLlib persistence. Covariates are
columns of the input DataFrame, given either as a pre-built covariate vector column
with its term names (no extra dependency) or as an R-style formula over the
DataFrame's columns (the ``formula`` extra: pip install spark-vi[formula]; see
``spark_vi.mllib.topic._formula``).
"""
from __future__ import annotations

import dataclasses
import json
import logging
import re

import numpy as np
from pyspark import keyword_only
from pyspark.ml.base import Estimator, Model
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasFeaturesCol, HasMaxIter, HasSeed

log = logging.getLogger(__name__)

from spark_vi.eval.topic.concentration import ConcentrationAcc
from spark_vi.models.topic.stm import _stm_doc_inference, prior_topic_proportions
from spark_vi.core.broadcast import destroy_broadcast as _destroy_broadcast
from spark_vi.mllib._common import (
    _PersistenceParams, _ResultStagePersistence, apply_persistence_params,
)


# Vocabulary size at/above which spectral_method="auto" routes from the dense
# (exact V×V co-occurrence on the driver) path to the scalable random-projection
# path. The dense matrix is V×V float64 = 8·V² bytes ≈ 0.8 GB at V=10,000; the
# threshold sits near a ~1 GB single-matrix driver footprint (peak is a small
# multiple with per-group matrices). Heuristic memory guard, not a correctness
# boundary — both paths are correct (ADR 0037, superseding ADR 0032's
# explicit-knob-only clause). Tunable.
SPECTRAL_AUTO_VOCAB_THRESHOLD = 10_000


def resolve_spectral_method(
    method: str, vocab_size: int,
    threshold: int = SPECTRAL_AUTO_VOCAB_THRESHOLD,
) -> str:
    """Resolve spectral_method='auto' to 'dense'/'scalable' by vocabulary size.

    'auto' picks dense below `threshold` (exact, validated) and scalable at or
    above it (random-projection, bounded driver memory; ADR 0037). Explicit
    'dense'/'scalable' pass through unchanged — the threshold never overrides an
    explicit choice. Pure and side-effect-free so the routing is unit-testable
    without a Spark fit.
    """
    if method == "auto":
        return "scalable" if vocab_size >= threshold else "dense"
    return method


def _formula_mentions(group_var: str, covariate_names: list[str]) -> bool:
    """True if group_var appears as a factor in any design-column name,
    e.g. group_var='group' matches 'C(group)[T.b]'."""
    needle = f"({group_var})"
    return any(needle in name or name == group_var for name in covariate_names)


def corpus_mean_topic_proportions_rdd(cov_rdd, Gamma: np.ndarray, depth: int = 2):
    """Distributed α-equivalent: (1/D) Σ_d softmax(Γᵀ x_d) over an RDD.

    ``cov_rdd`` is an RDD of length-P covariate vectors (bare numpy arrays —
    no row ids, honoring the spark-vi layering rule). Mirrors the engine's
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

    try:
        sum_vec, count = cov_rdd.mapPartitions(_local).treeReduce(_combine, depth=depth)
        if count == 0:
            raise ValueError("corpus_mean_topic_proportions_rdd: empty covariate RDD")
        return sum_vec / count
    finally:
        # The treeReduce has returned, so nothing references the broadcast any
        # more: destroy (not unpersist) to also free the driver-side pickle —
        # the broadcast temp-file (ENOSPC) leak class.
        _destroy_broadcast(bcast)


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
    covariate vector (bare array — no row ids, honoring the spark-vi layering
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

    try:
        sum_vec, count = cov_group_rdd.mapPartitions(_local).treeReduce(
            _combine, depth=depth)
        if count == 0:
            raise ValueError(
                "corpus_mean_topic_proportions_gated_rdd: empty covariate RDD")
        return sum_vec / count
    finally:
        # Destroy (not unpersist) once the treeReduce has returned — unpersist
        # would leave the driver-side pickle behind (the broadcast temp-file ENOSPC leak class).
        _destroy_broadcast(g_bcast)
        _destroy_broadcast(p_bcast)


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

    try:
        n, mean, M2, n_docs = doc_rdd.mapPartitions(_local).treeReduce(
            _combine, depth=depth)
        if n_docs == 0:
            raise ValueError("corpus_eta_variance_gated_rdd: empty document RDD")
        return _welford_variance(n, M2, reference=reference)
    finally:
        # Destroy (not unpersist) once the treeReduce has returned — unpersist
        # would leave the driver-side pickle behind (the broadcast temp-file ENOSPC leak class).
        _destroy_broadcast(gp_bcast)
        _destroy_broadcast(p_bcast)


def _gated_mode_theta(eta_hat: np.ndarray, allowed: np.ndarray, K: int) -> np.ndarray:
    """Doc mode topic-proportion vector theta from eta_hat, restricted to the
    doc's allowed (gated) topic set: softmax(eta_hat[allowed]) placed at those
    indices, zeros elsewhere. ``_stm_doc_inference`` leaves non-allowed
    eta_hat entries meaningless (-inf by construction), so the softmax must be
    taken over ``allowed`` only, never over the full K-length eta_hat."""
    z = eta_hat[allowed]
    z = z - z.max()
    w = np.exp(z)
    theta = np.zeros(K, dtype=np.float64)
    theta[allowed] = w / w.sum()
    return theta


def corpus_concentration_stm(
    docs, global_params, partition, *,
    lbfgs_max_iter=50, lbfgs_tol=1e-4, reference=None, n_bins=50,
) -> dict:
    """Per-document topic-concentration summary (mode-based) for a gated STM.

    For each STMDocument runs the same per-doc Laplace E-step as
    ``corpus_eta_variance_gated`` to get eta_hat_d, forms the gated mode
    theta_d = softmax(eta_hat_d over the doc's allowed topics), and
    accumulates (top_mass, eff_topics) into a ConcentrationAcc (see
    ``spark_vi.eval.topic.concentration``). Returns ConcentrationAcc.summary().

    ``n_bins`` sets the histogram resolution; eff_topics bins span [1, K].

    NOTE: this is the posterior-MODE concentration (comparable to LDA's
    gamma/theta point estimate); a draw-based variant (sampling
    eta ~ N(eta_hat, nu_d)) is a future enrichment.
    """
    from spark_vi.models.topic._linalg import safe_inverse
    from scipy.special import digamma

    lam = np.asarray(global_params["lambda"], dtype=np.float64)
    Gamma = np.asarray(global_params["Gamma"], dtype=np.float64)
    Sigma = np.asarray(global_params["Sigma"], dtype=np.float64)
    K = lam.shape[0]
    expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))

    acc = ConcentrationAcc.zeros(n_bins, eff_max=float(K))
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
        theta = _gated_mode_theta(eta_hat, allowed, K)
        acc.add(theta)

    return acc.summary()


def corpus_concentration_stm_rdd(
    doc_rdd, global_params, partition, *,
    lbfgs_max_iter=50, lbfgs_tol=1e-4, reference=None, n_bins=50, depth=2,
) -> dict:
    """Distributed per-document topic-concentration summary (mode-based) for a
    gated STM. For each STMDocument, runs the same per-doc Laplace E-step as
    ``corpus_eta_variance_gated_rdd`` to get eta_hat_d over the doc's ALLOWED
    (gated) topic set, forms the gated mode theta_d = softmax(eta_hat_d over
    allowed) via ``_gated_mode_theta``, and accumulates (top_mass, eff_topics)
    into a ConcentrationAcc per partition.

    Distributed via the same mapPartitions + treeReduce idiom as
    ``corpus_eta_variance_gated_rdd``: each partition builds one
    ConcentrationAcc (histograms + streaming sums, all length-n_bins arrays
    plus scalars); the tree-reduce combines them via ``ConcentrationAcc.combine``
    (functional, so it is a safe treeReduce combiner). Only the small
    accumulator objects cross the network, never the documents or per-doc
    eta_hat/nu_d. ``global_params`` and ``partition`` are broadcast via the
    Spark-safe default-arg closure convention (see
    ``corpus_eta_variance_gated_rdd``).

    ``doc_rdd`` is an RDD of STMDocument. Returns ConcentrationAcc.summary().
    Raises ValueError if the reduced document count is 0.
    """
    sc = doc_rdd.context
    gp_bcast = sc.broadcast({
        k: np.asarray(v, dtype=np.float64) for k, v in global_params.items()
    })
    p_bcast = sc.broadcast(partition)

    def _local(rows, _gp=gp_bcast, _p=p_bcast):
        from spark_vi.eval.topic.concentration import ConcentrationAcc
        from spark_vi.models.topic._linalg import safe_inverse
        from scipy.special import digamma

        gp = _gp.value
        part = _p.value
        lam = gp["lambda"]
        Gamma = gp["Gamma"]
        Sigma = gp["Sigma"]
        K = lam.shape[0]
        expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))

        acc = ConcentrationAcc.zeros(n_bins, eff_max=float(K))
        subprec_cache: dict[tuple, np.ndarray] = {}

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
            theta = _gated_mode_theta(eta_hat, allowed, K)
            acc.add(theta)
        return [acc]

    try:
        acc = doc_rdd.mapPartitions(_local).treeReduce(
            lambda a, b: a.combine(b), depth=depth
        )
        if acc.n == 0:
            raise ValueError("corpus_concentration_stm_rdd: empty document RDD")
        return acc.summary()
    finally:
        # Destroy (not unpersist) once the treeReduce has returned — unpersist
        # would leave the driver-side pickle behind (the broadcast temp-file ENOSPC leak class).
        _destroy_broadcast(gp_bcast)
        _destroy_broadcast(p_bcast)


def corpus_theta_gated_rdd(
    doc_rdd, global_params, partition, *,
    reference=None, scale=1.0, sample_cap=200_000, seed=0,
    lbfgs_max_iter=50, lbfgs_tol=1e-4,
) -> np.ndarray:
    """Per-document gated MAP theta over the corpus, at the fitted Sigma, as an
    (N_sampled, K) array collected to the driver for histogramming.

    Runs the SAME per-doc gated Laplace E-step as ``corpus_concentration_stm_rdd``
    (``_stm_doc_inference`` -> gated mode ``_gated_mode_theta``) but instead of
    folding each theta into a ``ConcentrationAcc`` it COLLECTS the raw K-vectors
    so the caller can build a per-doc theta ("topic mass distribution")
    histogram of its own. This is the STM analog of LDA's per-doc gamma.

    Inference is under the E-step prior eta ~ N(mu, Sigma_prior) with
    ``Sigma_prior = scale * Sigma_fitted`` (``Sigma_fitted =
    global_params["Sigma"]``, the unit-diagonal correlation R of ADR 0034).
    ``scale`` is the GENERATION-SCALE multiplier on the fitted Sigma: at
    ``scale=1.0`` (the default) the prior is the raw fit and this is inference
    under the fitted model (c == 1) -- byte-identical to the pre-``scale``
    behavior. At the calibrated held-out ``eta_scale`` (c ~ 4.6) the prior
    penalty is 1/scale, so eta_hat is more data-driven and theta_hat is PEAKIER,
    giving the honest per-doc concentration instead of the over-diffuse
    unit-scale fit (unit-prior per-doc outputs are not measurements). ``scale`` is a passed-in calibrated value, not a
    heuristic constant. Inference is on the FULL document -- NOT the held-out
    split the generative-scale sweep (``corpus_heldout_scale_sweep_gated_rdd``)
    uses. A foreground topic outside a doc's allowed set is structurally EXACTLY
    0 (the hard gating mask, via ``_gated_mode_theta``); those zeros are honest
    and land in the histogram's lowest bin.

    The corpus is down-sampled to at most ``sample_cap`` documents (Bernoulli
    ``RDD.sample`` at ``frac = min(1, sample_cap / N)``) BEFORE collection: a
    distribution needs a large sample, not every doc, and collecting an (N, K)
    array to the driver must be bounded. ``sample_cap=200_000`` is a heuristic
    driver-memory bound, not a literature value. N, N_sampled and the fraction
    are logged (project rule: no silent cap).

    ``global_params`` and ``partition`` are broadcast via the Spark-safe
    default-arg closure convention (see ``corpus_concentration_stm_rdd``).

    Raises ValueError if the reduced document count is 0.
    """
    n_docs = doc_rdd.count()
    if n_docs == 0:
        raise ValueError("corpus_theta_gated_rdd: empty document RDD")

    frac = min(1.0, float(sample_cap) / float(n_docs))
    sampled = doc_rdd if frac >= 1.0 else doc_rdd.sample(False, frac, seed)

    sc = doc_rdd.context
    gp_bcast = sc.broadcast({
        k: np.asarray(v, dtype=np.float64) for k, v in global_params.items()
    })
    p_bcast = sc.broadcast(partition)
    scale = float(scale)

    def _local(rows, _gp=gp_bcast, _p=p_bcast, _scale=scale):
        from scipy.special import digamma

        from spark_vi.models.topic._linalg import safe_inverse

        gp = _gp.value
        part = _p.value
        lam = gp["lambda"]
        Gamma = gp["Gamma"]
        Sigma = gp["Sigma"]
        K = lam.shape[0]
        expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))

        subprec_cache: dict[tuple, np.ndarray] = {}
        for doc in rows:
            allowed = part.allowed_indices(doc.groups)
            key = tuple(allowed.tolist())
            Sigma_inv_allowed = subprec_cache.get(key)
            if Sigma_inv_allowed is None:
                # Sigma_prior = scale * Sigma_fitted -> prior precision is
                # safe_inverse(scale * Sigma[allowed]). At scale=1.0 this is
                # byte-identical to safe_inverse(Sigma[allowed]).
                Sigma_inv_allowed = safe_inverse(
                    _scale * Sigma[np.ix_(allowed, allowed)])
                subprec_cache[key] = Sigma_inv_allowed
            eta_hat, _, _ = _stm_doc_inference(
                indices=doc.indices, counts=doc.counts,
                expElogbeta=expElogbeta,
                Gamma=Gamma, Sigma_inv_allowed=Sigma_inv_allowed, x=doc.x,
                max_iter=lbfgs_max_iter, tol=lbfgs_tol,
                allowed=allowed, reference=reference,
            )
            yield _gated_mode_theta(eta_hat, allowed, K)

    try:
        theta_rows = sampled.mapPartitions(_local).collect()
        n_sampled = len(theta_rows)
        if n_sampled == 0:
            # frac>0 but the Bernoulli draw emptied every partition (tiny corpus /
            # tiny cap): fall back to the full corpus so we never return an empty
            # histogram sample.
            theta_rows = doc_rdd.mapPartitions(_local).collect()
            n_sampled = len(theta_rows)
        log.info(
            "corpus_theta_gated_rdd: sampled N'=%d of N=%d docs (frac=%.4f, "
            "cap=%d) for per-doc theta histogram.",
            n_sampled, n_docs, frac, sample_cap)
        return np.asarray(theta_rows, dtype=np.float64)
    finally:
        # Destroy (not unpersist) once the collect(s) have returned — unpersist
        # would leave the driver-side pickle behind (the broadcast temp-file ENOSPC leak class).
        _destroy_broadcast(gp_bcast)
        _destroy_broadcast(p_bcast)


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
    (the variance runaway). It under-corrects modestly (Laplace
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
            _destroy_broadcast(c_bcast)
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
        # The round loop is done, so nothing references these any more: destroy
        # (not unpersist) also frees the driver-side pickle of each — the
        # ENOSPC leak class, and gp_bcast carries a full lambda.
        _destroy_broadcast(gp_bcast)
        _destroy_broadcast(p_bcast)


def corpus_heldout_scale_sweep_gated(
    docs, global_params, partition, *, c_grid,
    holdout_frac=0.3, reference=None, seed=0,
    lbfgs_max_iter=50, lbfgs_tol=1e-4,
) -> dict:
    """Driver-side (in-memory) held-out predictive-LL sweep over the STM
    generative covariance scale c, for a GATED fit.

    This is the gated, distributed-ready counterpart of the LOCAL, non-gated
    diagnostic validated in simulation
    (``spark_vi.eval.topic.heldout``): held-out within-document
    token prediction, swept over a concentration knob, recovers the true
    generative concentration. Here the knob is the scalar scale c in
    Sigma_gen = c*R (R the fit's diagonal-normalized correlation, ADR 0034/0036
    -- see ``corpus_eta_scale_gated``), and inference/scoring use the GATED
    machinery: eta_d ~ N(Gamma^T x_d, Sigma) restricted to the doc's allowed
    (background ∪ own group) topic set.

    For each doc, ONCE (reused across every c): split its tokens into a
    visible half and a held-out half (``heldout_split``, seeded independent
    of c so the sweep is a controlled comparison -- see that docstring for
    the split protocol and its skip guards for short docs).

    For each c, on the SAME split: infer eta_hat from the VISIBLE tokens via
    the same per-doc Laplace E-step as ``infer_local``, under prior precision
    ``(1/c) * safe_inverse(R[allowed])`` (R[allowed] is c-independent, so it
    is cached once per distinct allowed set). Note the deliberate split of
    roles: INFERENCE uses ``expElogbeta`` (exp-digamma of lambda, matching the
    fit's own E-step data term); SCORING the held-out predictive probability
    uses ``beta_prob`` (lambda-normalized, i.e. E[beta]) via
    ``_predictive_loglik`` -- conflating the two would silently miscalibrate
    the recovered scale.

    Returns ``{"lls": {c: mean_per_token_ll for c in c_grid}, "argmax_c": c,
    "n_docs": n}`` -- mean_per_token_ll is the corpus-wide held-out
    log-likelihood total divided by the corpus-wide held-out token count (so
    it is comparable across c and corpus size), and n_docs counts documents
    that actually contributed (docs skipped by ``heldout_split`` -- too few
    tokens, or an empty visible half -- do not count).

    ``docs`` is an in-memory list of STMDocument; ``global_params`` is the
    dict with "lambda" (K,V), "Gamma" (P,K), "Sigma" (K,K, the fit's
    correlation R); ``partition`` is a TopicBlockPartition (or the implicit
    all-background one). For a live cluster corpus use
    ``corpus_heldout_scale_sweep_gated_rdd``.
    """
    from scipy.special import digamma

    from spark_vi.eval.topic.heldout import (
        _predictive_loglik, heldout_split,
    )
    from spark_vi.models.topic._linalg import safe_inverse

    lam = np.asarray(global_params["lambda"], dtype=np.float64)
    Gamma = np.asarray(global_params["Gamma"], dtype=np.float64)
    Sigma = np.asarray(global_params["Sigma"], dtype=np.float64)
    K = lam.shape[0]
    lam_rowsum = lam.sum(axis=1, keepdims=True)
    expElogbeta = np.exp(digamma(lam) - digamma(lam_rowsum))   # inference term
    beta_prob = lam / lam_rowsum                               # scoring term: E[beta]

    d = np.diag(Sigma)
    R = Sigma / np.sqrt(np.outer(d, d))   # correlation; c is applied to THIS

    rinv_cache: dict[tuple, np.ndarray] = {}
    sum_ll = {c: 0.0 for c in c_grid}
    n_tokens = {c: 0 for c in c_grid}
    n_docs = 0

    for i, doc in enumerate(docs):
        split = heldout_split(doc, holdout_frac=holdout_frac, seed=seed + i)
        if split is None:
            continue
        visible_doc, held_indices, held_counts = split
        if held_counts.size == 0:
            continue

        allowed = partition.allowed_indices(doc.groups)
        key = tuple(allowed.tolist())
        Rinv_allowed = rinv_cache.get(key)
        if Rinv_allowed is None:
            Rinv_allowed = safe_inverse(R[np.ix_(allowed, allowed)])
            rinv_cache[key] = Rinv_allowed

        n_docs += 1
        n_held = int(held_counts.sum())
        for c in c_grid:
            Sigma_inv_allowed = (1.0 / c) * Rinv_allowed
            eta_hat, _, _ = _stm_doc_inference(
                indices=visible_doc.indices, counts=visible_doc.counts,
                expElogbeta=expElogbeta,
                Gamma=Gamma, Sigma_inv_allowed=Sigma_inv_allowed, x=doc.x,
                max_iter=lbfgs_max_iter, tol=lbfgs_tol,
                allowed=allowed, reference=reference,
            )
            theta_hat = _gated_mode_theta(eta_hat, allowed, K)
            sum_ll[c] += _predictive_loglik(theta_hat, beta_prob, held_indices, held_counts)
            n_tokens[c] += n_held

    lls = {c: sum_ll[c] / n_tokens[c] for c in c_grid}
    argmax_c = max(lls, key=lls.get)
    return {"lls": lls, "argmax_c": argmax_c, "n_docs": n_docs}


def corpus_heldout_scale_sweep_gated_rdd(
    doc_rdd, global_params, partition, *, c_grid,
    holdout_frac=0.3, reference=None, seed=0,
    lbfgs_max_iter=50, lbfgs_tol=1e-4, depth=2,
) -> dict:
    """Distributed held-out predictive-LL sweep over the STM generative
    covariance scale c, for a GATED fit (see ``corpus_heldout_scale_sweep_gated``
    for the full derivation; this is its Spark counterpart, byte-for-byte the
    same math on a distributed corpus).

    ``doc_rdd.zipWithIndex()`` gives each document the SAME index it would
    have under ``enumerate(docs)`` in the numpy path (zipWithIndex preserves
    the RDD's element order, and ``sc.parallelize(docs, n)`` preserves the
    input list's order across partitions), so ``heldout_split``'s per-doc seed
    (``seed + index``) reproduces the identical visible/held split as the
    numpy oracle -- required for numpy/RDD parity, since the split is a
    random draw that must line up doc-for-doc, not just partition-for-partition.

    Distributed via the same mapPartitions + treeReduce idiom as
    ``corpus_eta_scale_gated_rdd``: each partition runs the full per-doc
    split + per-c Laplace E-step + held-out scoring, accumulating a local
    dict of per-c (sum_ll, n_held_tokens) plus a doc count; the tree-reduce
    sums those elementwise across partitions -- only the small per-c totals
    cross the network, never the documents or per-doc eta_hat. The driver
    divides each c's summed LL by its summed token count (mean per token) and
    takes the argmax. ``global_params``, ``partition``, and ``c_grid`` are
    broadcast via the Spark-safe default-arg closure convention; helpers are
    imported inside ``_local`` so the closure is picklable on workers.

    Raises ValueError if the reduced document count is 0.
    """
    sc = doc_rdd.context
    gp_bcast = sc.broadcast({
        k: np.asarray(v, dtype=np.float64) for k, v in global_params.items()
    })
    p_bcast = sc.broadcast(partition)
    c_list = list(c_grid)
    c_bcast = sc.broadcast(c_list)

    def _local(rows, _gp=gp_bcast, _p=p_bcast, _cg=c_bcast):
        from scipy.special import digamma

        from spark_vi.eval.topic.heldout import (
            _predictive_loglik, heldout_split,
        )
        from spark_vi.models.topic._linalg import safe_inverse

        gp = _gp.value
        part = _p.value
        c_grid_local = _cg.value
        lam = gp["lambda"]
        Gamma = gp["Gamma"]
        Sigma = gp["Sigma"]
        K = lam.shape[0]
        lam_rowsum = lam.sum(axis=1, keepdims=True)
        expElogbeta = np.exp(digamma(lam) - digamma(lam_rowsum))
        beta_prob = lam / lam_rowsum

        d = np.diag(Sigma)
        R = Sigma / np.sqrt(np.outer(d, d))

        rinv_cache: dict[tuple, np.ndarray] = {}
        acc = {c: [0.0, 0] for c in c_grid_local}
        n_docs = 0

        for doc, idx in rows:
            split = heldout_split(doc, holdout_frac=holdout_frac, seed=seed + idx)
            if split is None:
                continue
            visible_doc, held_indices, held_counts = split
            if held_counts.size == 0:
                continue

            allowed = part.allowed_indices(doc.groups)
            key = tuple(allowed.tolist())
            Rinv_allowed = rinv_cache.get(key)
            if Rinv_allowed is None:
                Rinv_allowed = safe_inverse(R[np.ix_(allowed, allowed)])
                rinv_cache[key] = Rinv_allowed

            n_docs += 1
            n_held = int(held_counts.sum())
            for c in c_grid_local:
                Sigma_inv_allowed = (1.0 / c) * Rinv_allowed
                eta_hat, _, _ = _stm_doc_inference(
                    indices=visible_doc.indices, counts=visible_doc.counts,
                    expElogbeta=expElogbeta,
                    Gamma=Gamma, Sigma_inv_allowed=Sigma_inv_allowed, x=doc.x,
                    max_iter=lbfgs_max_iter, tol=lbfgs_tol,
                    allowed=allowed, reference=reference,
                )
                theta_hat = _gated_mode_theta(eta_hat, allowed, K)
                acc[c][0] += _predictive_loglik(
                    theta_hat, beta_prob, held_indices, held_counts)
                acc[c][1] += n_held

        return [(acc, n_docs)]

    def _combine(a, b):
        acc_a, n_a = a
        acc_b, n_b = b
        merged = {
            c: (acc_a[c][0] + acc_b[c][0], acc_a[c][1] + acc_b[c][1])
            for c in acc_a
        }
        return merged, n_a + n_b

    try:
        acc, n_docs = (
            doc_rdd.zipWithIndex().mapPartitions(_local)
            .treeReduce(_combine, depth=depth)
        )
        if n_docs == 0:
            raise ValueError("corpus_heldout_scale_sweep_gated_rdd: empty document RDD")

        lls = {c: acc[c][0] / acc[c][1] for c in c_list}
        argmax_c = max(lls, key=lls.get)
        return {"lls": lls, "argmax_c": argmax_c, "n_docs": n_docs}
    finally:
        # Destroy (not unpersist) once the treeReduce has returned — unpersist
        # would leave the driver-side pickle behind (the broadcast temp-file ENOSPC leak class).
        _destroy_broadcast(gp_bcast)
        _destroy_broadcast(p_bcast)
        _destroy_broadcast(c_bcast)


def smooth_scale_log_quadratic(lls: dict, *, window_radius: int = 2) -> dict:
    """Reduce a held-out-LL-vs-scale grid (``corpus_heldout_scale_sweep_gated{,_rdd}``'s
    ``"lls"``) to a smoothed c* by local quadratic interpolation in log c.

    The held-out LL curve is a broad, flat SHELF (differences ~0.001-0.01 nats
    across c in roughly [2, 12], within resampling noise): a raw argmax over a
    coarse grid on a curve that flat is a quantized, jittery point estimate
    (it drove a 5 -> 12 -> 8 refit wander in practice). This function instead
    fits a quadratic to a small WINDOW of grid points around the raw argmax,
    in u = ln(c) (c is a multiplicative scale, so log space is where the
    curve is locally well-approximated by a parabola and where grid spacing
    is uniform, see ``C_GRID`` at the call sites) and reports:
      - c_star: the smoothed vertex (sub-grid, noise-averaged), or the raw
        argmax if the window is not concave (see below);
      - curvature_q and a delta-method SE, so the caller can tell a sharp
        interior peak from a flat shelf instead of silently trusting a
        jittery grid-argmax.

    ``lls`` maps scale c > 0 -> mean per-token held-out LL (as returned by the
    sweep). Returns a dict (see below); never raises on a well-formed input.

    Algorithm (parabolic/quadratic interpolation to locate an extremum from
    sampled function values is the classic Numerical-Recipes / Brent
    ingredient -- Press, Teukolsky, Vetterling & Flannery, *Numerical
    Recipes*, 3rd ed., Section 10.2 (parabolic interpolation), the same
    interpolation step used inside Brent's method (Brent, R. P., 1973,
    *Algorithms for Minimization without Derivatives*, ch. 5). The vertex
    standard error is the standard delta method: propagate the polyfit
    parameter covariance through u* = -p1/(2 p2), then c = exp(u*)):

    1. Sort the grid; u_i = ln(c_i), y_i = lls[c_i]. If fewer than 3 points
       are supplied there is nothing to fit a quadratic to -- return the raw
       argmax with interior=False, curvature_q=0.0, se_log_c=se_c=None.
    2. i* = index of the grid's max y (the raw argmax). Take a WINDOW of grid
       indices [i*-window_radius, i*+window_radius], clipped to the grid,
       widening (growing the radius) if clipping left fewer than 4 points
       (can happen when i* sits at or near a grid edge) -- 4, not the bare
       minimum of 3, so the quadratic fit has >= 1 residual degree of
       freedom to scale a covariance from (a 3-point window exactly
       determines a quadratic with zero residual; ``np.polyfit(...,
       cov=True)`` cannot scale a covariance in that case -- see step 3).
       ``window_radius`` (default 2) and the grid bounds used by the callers
       ([0.5, 32]) are HEURISTIC choices, not literature values: they trade
       off "enough points to fit a stable quadratic" against "local enough
       that a broad non-parabolic curve doesn't bias the vertex."
    3. Fit p2*u^2 + p1*u + p0 to the window via
       ``np.polyfit(u_win, y_win, 2, cov=True)``. A concave-down parabola
       (a genuine interior MAX) has p2 < 0. If the window still has < 4
       points (a total grid smaller than 4), the covariance can't be scaled;
       fall back to the unscaled point fit and report se_log_c=se_c=None
       (an honest "can't estimate uncertainty" rather than a fabricated SE).
    4. If p2 < 0: u_star = -p1/(2 p2). If u_star falls within
       [min(u_grid), max(u_grid)] it is INTERIOR (interior=True); otherwise
       clip to the nearer grid endpoint and mark interior=False (the fitted
       vertex extrapolated past the data -- don't ship an extrapolated
       scale). c_star = exp(u_star); curvature_q = 2*p2 (< 0, more negative =
       sharper peak = more identifiable).
       SE via the delta method: with J = [d(u*)/dp1, d(u*)/dp2] =
       [-1/(2 p2), p1/(2 p2^2)] and cov_12 the 2x2 (p1, p2) block of the
       polyfit covariance, se_log_c = sqrt(J @ cov_12 @ J.T); se_c =
       c_star * se_log_c (first-order propagation of c = exp(u)).
    5. If p2 >= 0 (the window is flat or convex -- not a real max; this is
       what a monotone-rising/falling boundary window looks like, since the
       "peak" is only a grid edge, not a genuine interior vertex): fall back
       to c_star = the raw grid argmax c, interior=False, curvature_q = 2*p2
       (>= 0), se_log_c = se_c = None (no vertex to attach a SE to).

    Returns
    -------
    dict with keys:
      method: "log_quadratic"
      c_star: float -- the smoothed (or fallback) point estimate
      log_c_star: float -- ln(c_star)
      curvature_q: float -- 2*p2 of the local quadratic fit (< 0 = concave/
        peaked, >= 0 = flat-or-convex/not a real max, 0.0 in the degenerate
        <3-point case)
      se_log_c: float | None -- delta-method SE of log_c_star, None if not
        an interior concave fit
      se_c: float | None -- se_log_c propagated to c-scale, None likewise
      interior: bool -- True iff the fitted vertex is concave AND falls
        strictly within the grid's [min, max]
      grid_argmax_c: float -- the raw grid argmax, for comparison/fallback
      window_c: list[float] -- the c's actually used in the quadratic fit
    """
    items = sorted(lls.items(), key=lambda kv: kv[0])
    cs = np.array([c for c, _ in items], dtype=np.float64)
    ys = np.array([y for _, y in items], dtype=np.float64)
    n = cs.size

    i_star = int(np.argmax(ys))
    grid_argmax_c = float(cs[i_star])

    if n < 3:
        return {
            "method": "log_quadratic",
            "c_star": grid_argmax_c,
            "log_c_star": float(np.log(grid_argmax_c)),
            "curvature_q": 0.0,
            "se_log_c": None,
            "se_c": None,
            "interior": False,
            "grid_argmax_c": grid_argmax_c,
            "window_c": cs.tolist(),
        }

    us = np.log(cs)

    # Window around the raw argmax, widened symmetrically until it holds
    # >= 4 points (only needed near a grid edge, where a fixed-radius clip
    # can otherwise leave fewer). >= 4 rather than the bare minimum of 3
    # (3 points exactly determine a quadratic with zero residual, and
    # np.polyfit(..., cov=True) cannot scale a covariance from a zero-dof
    # fit) so there is at least one residual degree of freedom to estimate
    # the SE from -- unless the whole grid is already in the window and
    # still short of 4 (tiny grids), in which case we stop widening and
    # fall back to an unscaled (finite-difference-only) covariance below.
    radius = window_radius
    while True:
        lo = max(0, i_star - radius)
        hi = min(n - 1, i_star + radius)
        if hi - lo + 1 >= 4 or (lo == 0 and hi == n - 1):
            break
        radius += 1
    u_win = us[lo:hi + 1]
    y_win = ys[lo:hi + 1]
    c_win = cs[lo:hi + 1]

    try:
        p, cov = np.polyfit(u_win, y_win, 2, cov=True)
    except ValueError:
        # Degenerate window (e.g. a grid with < 4 total points, so no
        # residual d.o.f. is available to scale the covariance): still fit
        # the point estimate, but the SE is genuinely unavailable -- report
        # it as None rather than fabricating one.
        p = np.polyfit(u_win, y_win, 2)
        cov = None
    p2, p1, _p0 = p

    u_min, u_max = us[0], us[-1]

    if p2 < 0:
        u_star = -p1 / (2.0 * p2)
        curvature_q = 2.0 * p2
        if u_min <= u_star <= u_max:
            interior = True
        else:
            interior = False
            u_star = u_max if u_star > u_max else u_min
        c_star = float(np.exp(u_star))

        if cov is None:
            se_log_c = None
            se_c = None
        else:
            # Delta method: propagate the (p1, p2) block of the polyfit
            # covariance through u* = -p1/(2 p2).
            J = np.array([-1.0 / (2.0 * p2), p1 / (2.0 * p2 ** 2)])
            cov_12 = cov[0:2, 0:2][::-1, ::-1]  # polyfit orders p as [p2,p1,p0]
            var_log_c = float(J @ cov_12 @ J.T)
            se_log_c = float(np.sqrt(var_log_c)) if var_log_c > 0 else 0.0
            se_c = c_star * se_log_c
    else:
        interior = False
        curvature_q = 2.0 * p2
        c_star = grid_argmax_c
        se_log_c = None
        se_c = None

    return {
        "method": "log_quadratic",
        "c_star": c_star,
        "log_c_star": float(np.log(c_star)),
        "curvature_q": float(curvature_q),
        "se_log_c": se_log_c,
        "se_c": se_c,
        "interior": bool(interior),
        "grid_argmax_c": grid_argmax_c,
        "window_c": c_win.tolist(),
    }


def _topic_blocks_converter(value):
    """``topicBlocks`` is held as canonical JSON (a TopicBlockPartition's
    ``to_dict``), so it round-trips through MLlib's Params metadata; a
    TopicBlockPartition, a dict or a JSON string are accepted."""
    from spark_vi.models.topic.partition import TopicBlockPartition
    if value is None:
        return None
    if isinstance(value, TopicBlockPartition):
        d = value.to_dict()
    elif isinstance(value, dict):
        d = TopicBlockPartition.from_dict(value).to_dict()
    elif isinstance(value, str):
        d = TopicBlockPartition.from_dict(json.loads(value)).to_dict()
    else:
        raise TypeError(f"topicBlocks takes a TopicBlockPartition, a dict or a JSON "
                        f"string, not {type(value).__name__}")
    return json.dumps(d, sort_keys=True)


def _spectral_method_converter(value):
    value = TypeConverters.toString(value)
    if value not in {"auto", "dense", "scalable"}:
        raise ValueError(
            f"spectralMethod must be 'auto', 'dense', or 'scalable', got {value!r}")
    return value


# Param defaults, shared by the Estimator, the Model and their __init__ signatures.
# The SVI schedule defaults (learningOffset 64, learningDecay 0.7, subsamplingRate
# 0.2, maxIter 20) are the STM fit's own, not MLlib LDA's.
_STM_DEFAULTS = dict(
    k=10, maxIter=20, featuresCol="features",
    topicDistributionCol="topicDistribution",
    covariatesCol="covariates", covariateFormula="", maxLevels=10_000,
    learningOffset=64.0, learningDecay=0.7, subsamplingRate=0.2,
    sigmaInit=1.0, sigmaRidge=1e-6, minPairSupport=1,
    lbfgsMaxIter=50, lbfgsTol=1e-4,
    referenceTopic=True, estimateSigmaDiagonal=False, estimateGlobalScale=False,
    globalScaleStepCap=1.2, sigmaDiagonalPin=1.0,
    spectralInit=True, spectralMethod="auto", spectralMinDocFreq=5,
    docGroupCol="",
)


class _OnlineSTMParams(HasFeaturesCol, HasMaxIter, HasSeed, _PersistenceParams):
    """Shared Param surface for OnlineSTMEstimator and OnlineSTMModel (the
    MLlib ``_LDAParams`` pattern, as the LDA / HDP / DAG-gated shims).

    Covariates come from columns of the input DataFrame, one of two ways:

    - a pre-built covariate vector column (``covariatesCol``, a DenseVector per
      row) with its term names (``covariateNames``, length P);
    - an R-style formula over the DataFrame's columns (``covariateFormula``, the
      optional ``formula`` extra), whose categorical levels are discovered with
      Spark (ADR 0024) and whose design matrix is built per partition.

    Persistence Params (saveInterval, saveDir, resumeFrom) come from
    ``_PersistenceParams``."""

    k = Param(Params._dummy(), "k",
              "number of topics (with topicBlocks: background_k plus the foreground "
              "block sizes); must be >= 1 (>= 2 with referenceTopic)",
              typeConverter=TypeConverters.toInt)
    topicDistributionCol = Param(
        Params._dummy(), "topicDistributionCol",
        "output column of transform: each document's topic proportions theta, the "
        "mode of its gated logistic-normal posterior (softmax of the MAP eta over "
        "the document's allowed topics; exactly 0 on topics its gate excludes)",
        typeConverter=TypeConverters.toString)
    covariatesCol = Param(
        Params._dummy(), "covariatesCol",
        "pre-built covariate vector column (a DenseVector of length P per row); "
        "used with covariateNames, ignored when covariateFormula is set",
        typeConverter=TypeConverters.toString)
    covariateNames = Param(
        Params._dummy(), "covariateNames",
        "the P covariate term names, in the order of covariatesCol's entries; on a "
        "fitted model from a formula, the formula's design-column names",
        typeConverter=TypeConverters.toListString)
    covariateFormula = Param(
        Params._dummy(), "covariateFormula",
        "R-style prevalence formula over the DataFrame's columns (e.g. "
        "'~ C(sex) + age'); empty (default) = use covariatesCol + covariateNames. "
        "Needs the 'formula' extra; stateful transforms (splines, scale) are "
        "rejected (ADR 0022). Rows with a null in any formula column are dropped "
        "at fit and get a null topicDistribution at transform",
        typeConverter=TypeConverters.toString)
    categoricalCols = Param(
        Params._dummy(), "categoricalCols",
        "formula columns to treat as categorical; unset (default) = the first "
        "argument of every C(...) plus every string or boolean column (ADR 0024)",
        typeConverter=TypeConverters.toListString)
    maxLevels = Param(
        Params._dummy(), "maxLevels",
        "largest number of distinct levels a categorical formula column may have "
        "(fit fails above it; bin the column instead)",
        typeConverter=TypeConverters.toInt)
    learningOffset = Param(
        Params._dummy(), "learningOffset",
        "tau0 in the Robbins-Monro step rho_t = (tau0 + t)^-kappa",
        typeConverter=TypeConverters.toFloat)
    learningDecay = Param(
        Params._dummy(), "learningDecay",
        "kappa in the Robbins-Monro step rho_t = (tau0 + t)^-kappa",
        typeConverter=TypeConverters.toFloat)
    subsamplingRate = Param(
        Params._dummy(), "subsamplingRate",
        "fraction of the corpus sampled per mini-batch (>= 1.0 = full batch)",
        typeConverter=TypeConverters.toFloat)
    sigmaInit = Param(
        Params._dummy(), "sigmaInit",
        "initial diagonal of the topic covariance Sigma",
        typeConverter=TypeConverters.toFloat)
    sigmaRidge = Param(
        Params._dummy(), "sigmaRidge",
        "ridge added to Sigma's diagonal in the M-step, for a well-conditioned inverse",
        typeConverter=TypeConverters.toFloat)
    minPairSupport = Param(
        Params._dummy(), "minPairSupport",
        "documents that must share a topic pair before Sigma's off-diagonal entry "
        "for the pair is estimated (below it the entry stays 0)",
        typeConverter=TypeConverters.toInt)
    lbfgsMaxIter = Param(
        Params._dummy(), "lbfgsMaxIter",
        "max L-BFGS iterations of the per-document Laplace E-step (fit and transform)",
        typeConverter=TypeConverters.toInt)
    lbfgsTol = Param(
        Params._dummy(), "lbfgsTol",
        "L-BFGS tolerance of the per-document Laplace E-step",
        typeConverter=TypeConverters.toFloat)
    referenceTopic = Param(
        Params._dummy(), "referenceTopic",
        "pin topic 0's eta at 0 (removes softmax's translation degeneracy; ADR 0031)",
        typeConverter=TypeConverters.toBoolean)
    estimateSigmaDiagonal = Param(
        Params._dummy(), "estimateSigmaDiagonal",
        "estimate Sigma's diagonal instead of pinning it (ADR 0034)",
        typeConverter=TypeConverters.toBoolean)
    estimateGlobalScale = Param(
        Params._dummy(), "estimateGlobalScale",
        "estimate one global scale c of Sigma = c * R (ADR 0036)",
        typeConverter=TypeConverters.toBoolean)
    globalScaleStepCap = Param(
        Params._dummy(), "globalScaleStepCap",
        "largest multiplicative change of the global scale per iteration",
        typeConverter=TypeConverters.toFloat)
    sigmaDiagonalPin = Param(
        Params._dummy(), "sigmaDiagonalPin",
        "constant generative scale c the diagonal is pinned to (Sigma = c * R); 1.0 = "
        "the unit-diagonal pin. Exclusive with the two estimate* Params",
        typeConverter=TypeConverters.toFloat)
    spectralInit = Param(
        Params._dummy(), "spectralInit",
        "seed beta with the anchor-word spectral initialization (skipped on resume)",
        typeConverter=TypeConverters.toBoolean)
    spectralMethod = Param(
        Params._dummy(), "spectralMethod",
        "'auto' (dense below a vocabulary of 10,000 terms, scalable at or above it), "
        "'dense' (exact V x V co-occurrence on the driver) or 'scalable' (random "
        "projection; ADR 0037)",
        typeConverter=_spectral_method_converter)
    spectralD = Param(
        Params._dummy(), "spectralD",
        "projection dimension of the scalable spectral init; unset = about 1,000",
        typeConverter=TypeConverters.toInt)
    spectralMinDocFreq = Param(
        Params._dummy(), "spectralMinDocFreq",
        "document-frequency floor for anchor-word candidates (scalable init)",
        typeConverter=TypeConverters.toInt)
    topicBlocks = Param(
        Params._dummy(), "topicBlocks",
        "gated topic layout (a TopicBlockPartition, or its to_dict as a dict or JSON "
        "string): background topics every document may use, plus one foreground "
        "block per group, open only to documents in that group. Unset (default) = "
        "no gating. Needs docGroupCol",
        typeConverter=_topic_blocks_converter)
    docGroupCol = Param(
        Params._dummy(), "docGroupCol",
        "column naming each document's gating group(s) (a value or an array of "
        "values); required with topicBlocks. The group must not also appear in "
        "the covariates",
        typeConverter=TypeConverters.toString)

    def _get_or_none(self, name):
        p = self.getParam(name)
        return self.getOrDefault(p) if self.isDefined(p) else None

    def _topic_block_partition(self):
        """The ``topicBlocks`` Param as a TopicBlockPartition, or None."""
        from spark_vi.models.topic.partition import TopicBlockPartition
        tb = self._get_or_none("topicBlocks")
        return TopicBlockPartition.from_dict(json.loads(tb)) if tb else None

    def _uses_formula(self) -> bool:
        return bool(self.getOrDefault("covariateFormula"))

    def _check_covariate_config(self) -> None:
        """Raise on an inconsistent covariate / gating configuration (one of the two
        covariate sources, a group column exactly when there are topic blocks, the
        group not among the covariates). Called at construction when Params are given
        and again at fit."""
        names = self._get_or_none("covariateNames")
        if self._uses_formula():
            if names is not None:
                raise ValueError("Use either covariateFormula, or covariatesCol + "
                                 "covariateNames, not both.")
        elif not names:
            raise ValueError(
                "OnlineSTMEstimator needs its covariates: either covariateFormula, "
                "or covariatesCol + a non-empty covariateNames.")
        part = self._topic_block_partition()
        group_col = self.getOrDefault("docGroupCol")
        if part is not None and not group_col:
            raise ValueError("topicBlocks requires docGroupCol (the column naming "
                             "each document's gating group).")
        if part is None and group_col:
            raise ValueError("docGroupCol set without topicBlocks.")
        if part is not None and names is not None and \
                _formula_mentions(part.group_var, names):
            raise ValueError(
                f"group_var {part.group_var!r} must not also appear in the "
                f"prevalence formula (foreground regression would be rank-deficient); "
                f"remove it from the covariates.")
        if part is not None and self._uses_formula() and \
                re.search(rf"\b{re.escape(part.group_var)}\b",
                          self.getOrDefault("covariateFormula")):
            raise ValueError(
                f"group_var {part.group_var!r} must not also appear in the "
                f"prevalence formula (foreground regression would be rank-deficient).")


def _copy_params(src, dst) -> None:
    """Copy every set and default Param value of ``src`` onto ``dst`` (both carry
    ``_OnlineSTMParams``), so a fitted model reports the configuration that made it."""
    for param in src.params:
        if src.isSet(param):
            dst._set(**{param.name: src.getOrDefault(param)})
        elif src.hasDefault(param):
            dst._setDefault(**{param.name: src.getOrDefault(param)})


def _encode_formula_rows(spec, variables, rows):
    """Design-matrix rows for a chunk of Rows: ``(X, ok)`` where ``ok[i]`` says row
    i had every formula variable non-null and ``X`` holds the encoded rows of the
    ok ones, in order. Nulls are screened here, not left to formulaic's NA-drop,
    which would silently misalign the matrix with the rows."""
    import pandas as pd
    from spark_vi.mllib.topic._formula import apply_model_spec
    ok = [all(r[v] is not None for v in variables) for r in rows]
    kept = [r for r, good in zip(rows, ok) if good]
    if not kept:
        return np.zeros((0, 0)), ok
    pdf = pd.DataFrame({v: [r[v] for r in kept] for v in variables})
    return apply_model_spec(spec, pdf), ok


# Rows per formulaic materialization on an executor: bounds a partition's pandas
# frame while amortizing formulaic's per-call overhead (a few ms).
_ENCODE_CHUNK = 2048


def _chunks(it, n):
    buf = []
    for row in it:
        buf.append(row)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def _groups_of(val):
    """A row's gating group value as the frozenset an STMDocument holds (the same
    rule as ``_vector_to_stm_document``)."""
    if val is None:
        return frozenset()
    return (frozenset(str(v) for v in val) if isinstance(val, (list, tuple))
            else frozenset({str(val)}))


class OnlineSTMEstimator(_OnlineSTMParams, Estimator):
    """MLlib Estimator for the structural topic model (OnlineSTM: prevalence
    covariates through a logistic-normal prior, optional topic gating), fitted by
    streaming variational inference on Spark.

    The input DataFrame holds a bag-of-words vector column (``featuresCol``) and the
    covariates (see ``_OnlineSTMParams``); with ``topicBlocks``, also a gating group
    column (``docGroupCol``). ``fit`` returns an ``OnlineSTMModel``. The Estimator
    itself is not persisted (as the LDA / HDP Estimators; ADR 0055)."""

    @keyword_only
    def __init__(
        self,
        *,
        k: int = 10,
        maxIter: int = 20,
        seed: int | None = None,
        featuresCol: str = "features",
        topicDistributionCol: str = "topicDistribution",
        covariatesCol: str = "covariates",
        covariateNames: list[str] | None = None,
        covariateFormula: str = "",
        categoricalCols: list[str] | None = None,
        maxLevels: int = 10_000,
        learningOffset: float = 64.0,
        learningDecay: float = 0.7,
        subsamplingRate: float = 0.2,
        sigmaInit: float = 1.0,
        sigmaRidge: float = 1e-6,
        minPairSupport: int = 1,
        lbfgsMaxIter: int = 50,
        lbfgsTol: float = 1e-4,
        referenceTopic: bool = True,
        estimateSigmaDiagonal: bool = False,
        estimateGlobalScale: bool = False,
        globalScaleStepCap: float = 1.2,
        sigmaDiagonalPin: float = 1.0,
        spectralInit: bool = True,
        spectralMethod: str = "auto",
        spectralD: int | None = None,
        spectralMinDocFreq: int = 5,
        topicBlocks=None,
        docGroupCol: str = "",
        # _PersistenceParams kwargs: they MUST appear here for kwarg construction
        # (see that mixin's docstring).
        saveInterval: int = -1,
        saveDir: str = "",
        resumeFrom: str = "",
    ) -> None:
        super().__init__()
        self._setDefault(**_STM_DEFAULTS)
        self._set_persistence_defaults()
        # Diagnostic-only iteration callback (not a Param: callables are not
        # MLlib-serializable).
        self._on_iteration = None
        kwargs = {k: v for k, v in self._input_kwargs.items() if v is not None}
        self.setParams(**kwargs)
        # Fail early on an inconsistent covariate / gating configuration given at
        # construction; a bare Estimator configured later is checked at fit.
        if any(n in kwargs for n in ("covariateNames", "covariateFormula",
                                     "topicBlocks", "docGroupCol")):
            self._check_covariate_config()

    @keyword_only
    def setParams(self, **kwargs) -> "OnlineSTMEstimator":
        """Standard MLlib pattern: set any subset of Params after construction."""
        return self._set(**{k: v for k, v in kwargs.items() if v is not None})

    def setOnIteration(self, fn) -> "OnlineSTMEstimator":
        """Register a per-iteration diagnostic callback for the next fit:
        ``fn(iter_num, global_params, elbo_trace)``, on the driver, in the fit's hot
        path; it must not mutate global_params. Not persisted."""
        self._on_iteration = fn
        return self

    def _resolve_formula(self, dataset):
        """Discover the formula's categorical levels with Spark and fit its ModelSpec
        on the schema-frame (ADR 0024). Returns ``(spec, names, variables, state)``,
        ``state`` the JSON-able spec that the fitted model persists (ADR 0056)."""
        from spark_vi.mllib.topic._formula import (
            build_schema_frame, covariate_spec_state, fit_model_spec_on_schema_frame,
            formula_variables, split_formula_variables, validate_formula,
        )
        formula = self.getOrDefault("covariateFormula")
        validate_formula(formula)
        types = {f.name: f.dataType.simpleString() for f in dataset.schema.fields}
        categorical, continuous = split_formula_variables(formula, types)
        explicit = self._get_or_none("categoricalCols")
        if explicit is not None:
            variables = formula_variables(formula)
            unknown = [c for c in explicit if c not in variables]
            if unknown:
                raise ValueError(f"categoricalCols {unknown} are not formula columns "
                                 f"({variables}).")
            categorical = list(explicit)
            continuous = [v for v in variables if v not in categorical]
        variables = sorted(categorical + continuous)
        non_null = dataset.dropna(subset=variables) if variables else dataset
        schema_pdf = build_schema_frame(non_null, categorical, continuous,
                                        int(self.getOrDefault("maxLevels")))
        spec, names = fit_model_spec_on_schema_frame(formula, schema_pdf, categorical)
        state = covariate_spec_state(formula, schema_pdf, names, categorical)
        return spec, list(names), variables, state

    def _fit(self, dataset) -> "OnlineSTMModel":
        from pyspark import StorageLevel

        from spark_vi.core.config import VIConfig
        from spark_vi.core.runner import VIRunner
        from spark_vi.mllib.topic._common import (
            _vector_to_bow_document, _vector_to_stm_document,
        )
        from spark_vi.models.topic.stm import OnlineSTM
        from spark_vi.models.topic.types import STMDocument

        self._check_covariate_config()
        g = self.getOrDefault
        features_col = g("featuresCol")
        partition = self._topic_block_partition()
        group_col = g("docGroupCol") or None
        seed = g("seed") if self.isSet("seed") else None

        first = dataset.select(features_col).head(1)
        if not first:
            raise ValueError("Cannot fit on an empty DataFrame.")
        vocab_size = first[0][0].size

        spec_state = None
        if self._uses_formula():
            spec, names, variables, spec_state = self._resolve_formula(dataset)
            if partition is not None and \
                    _formula_mentions(partition.group_var, names):
                raise ValueError(
                    f"group_var {partition.group_var!r} must not also appear in "
                    f"the prevalence formula.")
        else:
            names = list(g("covariateNames"))

        model = OnlineSTM(
            K=int(g("k")), vocab_size=vocab_size, P=len(names),
            sigma_init=g("sigmaInit"), sigma_ridge=g("sigmaRidge"),
            min_pair_support=int(g("minPairSupport")),
            lbfgs_max_iter=g("lbfgsMaxIter"), lbfgs_tol=g("lbfgsTol"),
            random_seed=seed, topic_blocks=partition,
            reference_topic=bool(g("referenceTopic")),
            estimate_sigma_diagonal=bool(g("estimateSigmaDiagonal")),
            estimate_global_scale=bool(g("estimateGlobalScale")),
            global_scale_step_cap=float(g("globalScaleStepCap")),
            sigma_diagonal_pin=float(g("sigmaDiagonalPin")),
        )
        rate = float(g("subsamplingRate"))
        config = VIConfig(
            max_iterations=int(g("maxIter")),
            learning_rate_tau0=float(g("learningOffset")),
            learning_rate_kappa=float(g("learningDecay")),
            mini_batch_fraction=rate if rate < 1.0 else None,
        )
        config, resume_path = apply_persistence_params(self, config)

        if spec_state is None:
            cov_col = g("covariatesCol")
            select_cols = [features_col, cov_col] + ([group_col] if group_col else [])
            rdd = dataset.select(*select_cols).rdd.map(
                lambda row: _vector_to_stm_document(
                    {c: row[i] for i, c in enumerate(select_cols)},
                    features_col=features_col, covariates_col=cov_col,
                    group_col=group_col))
        else:
            # Formula path: encode each partition's covariates with the broadcast
            # spec (small; formulaic materializes a chunk at a time), dropping rows
            # with a null formula column (logged below).
            n_in = dataset.count()
            select_cols = [features_col] + variables + ([group_col] if group_col
                                                        else [])
            spec_bcast = dataset.sparkSession.sparkContext.broadcast(spec)

            def _docs(rows, _spec=spec_bcast, _vars=tuple(variables)):
                for chunk in _chunks(rows, _ENCODE_CHUNK):
                    X, ok = _encode_formula_rows(_spec.value, _vars, chunk)
                    j = 0
                    for row, good in zip(chunk, ok):
                        if not good:
                            continue
                        bow = _vector_to_bow_document(row[features_col])
                        yield STMDocument(
                            indices=bow.indices, counts=bow.counts, length=bow.length,
                            x=np.asarray(X[j], dtype=np.float64),
                            groups=(_groups_of(row[group_col]) if group_col
                                    else frozenset()))
                        j += 1

            rdd = dataset.select(*select_cols).rdd.mapPartitions(_docs)
        rdd = rdd.persist(StorageLevel.MEMORY_AND_DISK)
        n_docs = rdd.count()
        if spec_state is not None and n_docs < n_in:
            log.warning("OnlineSTMEstimator: dropped %d of %d rows with a null "
                        "covariate-formula column.", n_in - n_docs, n_in)

        # Optional spectral (anchor-word) beta seed. spectralMethod selects dense
        # (exact V x V on the driver, the validated default) vs scalable (random-
        # projection sketch for large vocabularies). The seed reaches the engine as
        # data_summary={"spectral_beta": KxV}: initialize_global seeds lambda from it
        # instead of random gamma, curing the sigmaInit-dependent collapse /
        # blow-up. Skipped when resuming (the runner restores lambda and never calls
        # initialize_global). "auto" resolves by vocabulary size (ADR 0037),
        # unconditionally, so the metadata records what would run even when
        # spectralInit is off.
        requested_method = g("spectralMethod")
        resolved_spectral_method = resolve_spectral_method(requested_method, vocab_size)
        if requested_method == "auto" and resolved_spectral_method == "scalable":
            log.warning(
                "spectralMethod='auto': vocab_size %d >= %d threshold; routing to "
                "the scalable random-projection init (dense V×V co-occurrence would "
                "be ~%.1f GB on the driver). Set spectralMethod='dense' to force "
                "the exact path.",
                vocab_size, SPECTRAL_AUTO_VOCAB_THRESHOLD,
                8.0 * vocab_size * vocab_size / 1e9)

        data_summary = None
        if g("spectralInit") and resume_path is None:
            eff_partition = model._effective_partition()
            if resolved_spectral_method == "scalable":
                from spark_vi.models.topic.spectral_init_scalable import (
                    scalable_spectral_init_beta,
                )
                beta0 = scalable_spectral_init_beta(
                    rdd, eff_partition, vocab_size,
                    d=self._get_or_none("spectralD"),
                    seed=seed or 0,
                    min_doc_freq=int(g("spectralMinDocFreq")),
                )
            else:
                from spark_vi.models.topic.spectral_init import spectral_init_beta
                beta0 = spectral_init_beta(rdd.collect(), eff_partition, vocab_size)
            data_summary = {"spectral_beta": beta0}

        runner = VIRunner(model, config=config)
        try:
            result = runner.fit(
                rdd, data_summary=data_summary,
                on_iteration=self._on_iteration, resume_from=resume_path,
            )
        finally:
            rdd.unpersist(blocking=False)
            # The formula spec's broadcast outlives the fit, not the first count:
            # a persisted partition that is evicted (or lost with an executor) is
            # recomputed from the lineage, which encodes with the spec again.
            if spec_state is not None:
                _destroy_broadcast(spec_bcast)

        metadata = dict(result.metadata)
        if partition is not None:
            metadata.setdefault("topic_block_spec", partition.to_dict())
        # Provenance: the hardening knobs that produced this fit (ADR 0031, 0034,
        # 0036, 0037); spectral_method is what ran, spectral_method_requested what
        # was asked for.
        metadata.setdefault("stm_hardening", {
            "reference_topic": bool(g("referenceTopic")),
            "min_pair_support": int(g("minPairSupport")),
            "spectral_init": bool(g("spectralInit")),
            "spectral_method": resolved_spectral_method,
            "spectral_method_requested": requested_method,
            "estimate_sigma_diagonal": bool(g("estimateSigmaDiagonal")),
            "estimate_global_scale": bool(g("estimateGlobalScale")),
            "global_scale_step_cap": float(g("globalScaleStepCap")),
            "sigma_diagonal_pin": float(g("sigmaDiagonalPin")),
        })
        metadata["covariate_names"] = list(names)
        result = dataclasses.replace(result, metadata=metadata)

        out = OnlineSTMModel(result, covariate_spec=spec_state)
        _copy_params(self, out)
        out._set(covariateNames=list(names))
        return out


class OnlineSTMModel(_OnlineSTMParams, _ResultStagePersistence, Model):
    """A fitted structural topic model: the trained VIResult (lambda, Gamma, Sigma,
    ...) plus every Param of the Estimator that produced it.

    ``transform`` appends each document's topic proportions (``topicDistributionCol``)
    from its words and covariates, by the same per-document Laplace E-step the fit
    uses, cold-started, at the fitted Sigma.

    Persistence is MLlib's (``save`` / ``write().overwrite().save`` / ``load``, and
    inside a ``PipelineModel``; ADR 0055, 0056)::

        <path>/metadata/        Params (MLlib's layout)
        <path>/result/          the VIResult (spark_vi.io.export format)
        <path>/covariate_spec.json   formula models only: the formula and its
                                     schema-frame, from which the ModelSpec is rebuilt
                                     on load (no pickle)

    A bare VIResult directory (a ``saveDir`` checkpoint) does not ``load``; wrap it,
    ``OnlineSTMModel(load_result(dir))`` (Params at their defaults; set
    ``covariateNames`` before ``transform``)."""

    _expected_model_class = "OnlineSTM"
    _SPEC_FILE = "covariate_spec.json"

    def __init__(self, result, covariate_spec: dict | None = None) -> None:
        super().__init__()
        self._result = result
        self._covariate_spec = covariate_spec
        self._model_spec = None
        self._setDefault(**_STM_DEFAULTS)
        self._set_persistence_defaults()
        tb = (result.metadata or {}).get("topic_block_spec")
        if tb:
            self._setDefault(topicBlocks=_topic_blocks_converter(tb))
        names = (result.metadata or {}).get("covariate_names")
        if names:
            self._setDefault(covariateNames=list(names))
        if covariate_spec is not None:
            self._setDefault(covariateFormula=covariate_spec["formula"])

    @property
    def result(self):
        """The trained VIResult (global_params, metadata, elbo_trace, n_iterations, ...)."""
        return self._result

    @property
    def covariateSpec(self) -> dict | None:
        """The formula model's JSON-able covariate spec (formula, schema-frame,
        design-column names), or None for a model fitted on a covariate column."""
        return self._covariate_spec

    def _formula_spec(self):
        if self._model_spec is None and self._covariate_spec is not None:
            from spark_vi.mllib.topic._formula import model_spec_from_state
            self._model_spec = model_spec_from_state(self._covariate_spec)
        return self._model_spec

    def encodeCovariates(self, pdf) -> np.ndarray:
        """The design matrix (rows, P) of a pandas frame under this model's formula
        (raises on a categorical level the fit did not see). Formula models only."""
        spec = self._formula_spec()
        if spec is None:
            raise ValueError("This model was fitted on a covariate column, not a "
                             "formula.")
        from spark_vi.mllib.topic._formula import apply_model_spec
        return apply_model_spec(spec, pdf)

    def topicBlockPartition(self):
        """The gated topic layout as a TopicBlockPartition, or None. (Not named
        ``topicBlocks``: that is the Param, ADR 0012 on accessor / Param name
        collisions.)"""
        return self._topic_block_partition()

    def vocabSize(self) -> int:
        return int(self._result.global_params["lambda"].shape[1])

    def topicCorrelation(self) -> np.ndarray:
        """(K, K) topic correlation R = Sigma_ij / sqrt(Sigma_ii Sigma_jj) (Blei &
        Lafferty 2007); the reference topic's row is the inert identity row."""
        from spark_vi.models.topic._linalg import topic_correlation
        return topic_correlation(self._result.global_params["Sigma"])

    def describeTopics(self, maxTermsPerTopic: int = 10):
        """DataFrame of (topic, termIndices, termWeights): each topic's top terms by
        E[beta] (lambda row-normalized), as ``OnlineLDAModel.describeTopics``."""
        from pyspark.sql import SparkSession
        from pyspark.sql.types import (
            ArrayType, DoubleType, IntegerType, StructField, StructType,
        )
        if maxTermsPerTopic < 1:
            raise ValueError(f"maxTermsPerTopic must be >= 1, got {maxTermsPerTopic}")
        lam = self._result.global_params["lambda"]
        beta = lam / lam.sum(axis=1, keepdims=True)
        m = min(maxTermsPerTopic, beta.shape[1])
        rows = []
        for k in range(beta.shape[0]):
            order = np.argsort(beta[k])[::-1][:m]
            rows.append((int(k), [int(i) for i in order],
                         [float(beta[k, i]) for i in order]))
        schema = StructType([
            StructField("topic", IntegerType(), False),
            StructField("termIndices", ArrayType(IntegerType(), False), False),
            StructField("termWeights", ArrayType(DoubleType(), False), False),
        ])
        return SparkSession.builder.getOrCreate().createDataFrame(rows, schema=schema)

    # --- persistence: the VIResult plus the formula's JSON spec ---------------
    def _save_data(self, path: str) -> None:
        from spark_vi.io import fs
        super()._save_data(path)
        if self._covariate_spec is not None:
            fs.write_text(fs.join(path, self._SPEC_FILE),
                          json.dumps(self._covariate_spec, indent=1))

    @classmethod
    def _load_data(cls, path: str, metadata: dict):
        from spark_vi.io import fs
        model = super()._load_data(path, metadata)
        spec_path = fs.join(path, cls._SPEC_FILE)
        if fs.exists(spec_path):
            model._covariate_spec = json.loads(fs.read_text(spec_path))
        return model

    # --- transform -------------------------------------------------------------
    def _transform(self, dataset):
        """Append ``topicDistributionCol``: per document, theta = softmax of the MAP
        eta over its allowed topics (``_gated_mode_theta``), from the cold-started
        Laplace E-step at the fitted Gamma and Sigma, with topic 0 pinned when the
        model was fitted with ``referenceTopic``. A row with a null features vector
        or a null covariate gets a null; a categorical level the fit never saw
        fails the job (apply_model_spec), rather than encode silently as the
        reference level.

        The (K, V) E[log beta] table and the (K, K) Sigma go to the executors as
        memory-mapped files (``ship_array``: one copy per executor, not one per
        Python worker); Gamma (P, K) and the scalars are broadcast."""
        from pyspark.ml.linalg import DenseVector, VectorUDT
        from pyspark.sql.types import StructField, StructType
        from scipy.special import digamma

        from spark_vi.core.broadcast import load_shipped, ship_array
        from spark_vi.mllib.topic._common import _vector_to_bow_document
        from spark_vi.models.topic.partition import TopicBlockPartition

        g = self.getOrDefault
        gp = self._result.global_params
        lam = np.asarray(gp["lambda"], dtype=np.float64)
        K = lam.shape[0]
        expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))
        features_col = g("featuresCol")
        out_col = g("topicDistributionCol")
        part = self._topic_block_partition() or TopicBlockPartition(
            group_var="", background_k=K, foreground=())
        group_col = g("docGroupCol") or None
        if self._topic_block_partition() is not None and not group_col:
            raise ValueError("A gated model's transform needs docGroupCol.")
        spec = self._formula_spec()
        if spec is not None:
            from spark_vi.mllib.topic._formula import formula_variables
            variables = tuple(formula_variables(self._covariate_spec["formula"]))
            cov_col = None
        else:
            variables = ()
            cov_col = g("covariatesCol")
        # The reference pin is a property of the fit: read it from the result's
        # provenance when present (a wrapped bare VIResult has Params at defaults).
        hardening = (self._result.metadata or {}).get("stm_hardening") or {}
        reference_on = bool(hardening.get("reference_topic", g("referenceTopic")))

        sc = dataset.sparkSession.sparkContext
        eb_token = ship_array(sc, expElogbeta, "stm_expElogbeta")
        sigma_token = ship_array(sc, np.asarray(gp["Sigma"], dtype=np.float64),
                                 "stm_Sigma")
        del expElogbeta
        bcast = sc.broadcast({
            "Gamma": np.asarray(gp["Gamma"], dtype=np.float64),
            "part": part, "K": K, "spec": spec,
            "reference": 0 if reference_on else None,
            "lbfgs_max_iter": int(g("lbfgsMaxIter")), "lbfgs_tol": float(g("lbfgsTol")),
        })

        def _theta_rows(rows, _b=bcast, _eb=eb_token, _sg=sigma_token,
                        _vars=variables, _cov=cov_col, _grp=group_col,
                        _fc=features_col):
            from spark_vi.models.topic._linalg import safe_inverse
            p = _b.value
            eb = load_shipped(_eb)
            Sigma = load_shipped(_sg)
            cache: dict = {}
            for chunk in _chunks(rows, _ENCODE_CHUNK):
                if p["spec"] is not None:
                    X, ok = _encode_formula_rows(p["spec"], _vars, chunk)
                    xs, j = [], 0
                    for good in ok:
                        xs.append(X[j] if good else None)
                        j += good
                else:
                    xs = [None if r[_cov] is None else r[_cov] for r in chunk]
                for row, x in zip(chunk, xs):
                    feats = row[_fc]
                    if x is None or feats is None:
                        yield tuple(row) + (None,)
                        continue
                    bow = _vector_to_bow_document(feats)
                    allowed = p["part"].allowed_indices(
                        _groups_of(row[_grp]) if _grp else frozenset())
                    key = tuple(allowed.tolist())
                    prec = cache.get(key)
                    if prec is None:
                        prec = safe_inverse(np.asarray(Sigma[np.ix_(allowed, allowed)]))
                        cache[key] = prec
                    eta_hat, _, _ = _stm_doc_inference(
                        indices=bow.indices, counts=bow.counts, expElogbeta=eb,
                        Gamma=p["Gamma"], Sigma_inv_allowed=prec,
                        x=np.asarray(x, dtype=np.float64),
                        max_iter=p["lbfgs_max_iter"], tol=p["lbfgs_tol"],
                        allowed=allowed, reference=p["reference"],
                    )
                    yield tuple(row) + (
                        DenseVector(_gated_mode_theta(eta_hat, allowed, p["K"])),)

        schema = StructType(list(dataset.schema.fields)
                            + [StructField(out_col, VectorUDT(), True)])
        # The broadcast and the shipped files live as long as the returned
        # DataFrame's lineage (ContextCleaner reclaims the broadcast on GC; do NOT
        # destroy it here, the DataFrame is lazy).
        return dataset.sparkSession.createDataFrame(
            dataset.rdd.mapPartitions(_theta_rows), schema=schema)
