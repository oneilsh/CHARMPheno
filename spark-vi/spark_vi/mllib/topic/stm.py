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
import logging
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

from spark_vi.eval.topic.concentration import ConcentrationAcc
from spark_vi.models.topic.stm import _stm_doc_inference, prior_topic_proportions


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


def _stm_doc_inference_tprior(
    *, indices, counts, expElogbeta, Gamma, Rinv_allowed, x, c, nu,
    allowed, reference=None, eta_init=None, sd_init=1.0,
    lbfgs_max_iter=50, lbfgs_tol=1e-4, sd_max_iter=10, sd_tol=1e-4,
):
    """Per-document joint MAP (eta, s_d) under the multivariate-t prior
    eta_d | s_d ~ N(mu_d, s_d*c*R), s_d ~ Inverse-Gamma(nu/2, nu/2)
    (design doc 2026-07-10-tprior-per-document-scale-design.md).

    Explicit EM / coordinate ascent: the eta-step is the existing gated Laplace
    solve (``_stm_doc_inference``) at prior precision (1/(s_d*c))*Rinv_allowed,
    warm-started across sweeps; the s_d-step is the closed-form Inverse-Gamma
    mode  s_d = (nu + q_R/c)/(nu + K_free + 2)  with
    q_R = (eta-mu)^T Rinv_allowed (eta-mu) over the allowed set (reference pinned
    at 0) and K_free = |allowed| - (1 if reference else 0). nu=inf recovers the
    single Gaussian solve at s_d=1 (nesting). Returns (eta_hat full-K, s_d float,
    nu_d Laplace cov from the final eta-step, n_em sweeps).

    The returned pair is a joint fixed point to tolerance, independent of how
    fast the s_d sequence contracts: within each sweep the eta-step is solved at
    the CURRENT sd, then the sd-step updates from that eta, so at loop exit
    eta_hat is one sd-step stale (it was solved at the pre-update sd). We repair
    this with ONE closing eta-solve at the final returned sd after the s_d
    sequence terminates -- warm-started from the last eta, so ~1 L-BFGS
    iteration. After it, eta_hat is exactly the Laplace argmax at the returned
    sd (up to lbfgs_tol), and sd is the IG-mode of the immediately-prior
    eta ~= eta_hat, so (eta_hat, sd) is mutually consistent to tolerance
    regardless of contraction speed. n_em counts the s_d sweeps (not the closing
    solve)."""
    allowed = np.asarray(allowed, dtype=np.int64)
    mu_allowed = (Gamma[:, allowed].T @ x)
    K_free = int(allowed.shape[0] - (1 if reference is not None else 0))

    if not math.isfinite(nu):
        Sigma_inv_allowed = (1.0 / c) * Rinv_allowed
        eta_hat, nu_d, _ = _stm_doc_inference(
            indices=indices, counts=counts, expElogbeta=expElogbeta,
            Gamma=Gamma, Sigma_inv_allowed=Sigma_inv_allowed, x=x,
            max_iter=lbfgs_max_iter, tol=lbfgs_tol,
            allowed=allowed, reference=reference, eta_init=eta_init,
        )
        return eta_hat, 1.0, nu_d, 1

    sd = float(sd_init)
    eta_warm = eta_init
    eta_hat = nu_d = None
    n_em = 0
    for n_em in range(1, sd_max_iter + 1):
        Sigma_inv_allowed = (1.0 / (sd * c)) * Rinv_allowed
        eta_hat, nu_d, _ = _stm_doc_inference(
            indices=indices, counts=counts, expElogbeta=expElogbeta,
            Gamma=Gamma, Sigma_inv_allowed=Sigma_inv_allowed, x=x,
            max_iter=lbfgs_max_iter, tol=lbfgs_tol,
            allowed=allowed, reference=reference, eta_init=eta_warm,
        )
        eta_warm = eta_hat
        diff = eta_hat[allowed] - mu_allowed
        q_R = float(diff @ Rinv_allowed @ diff)
        sd_new = (nu + q_R / c) / (nu + K_free + 2.0)
        converged = abs(sd_new - sd) < sd_tol
        sd = sd_new
        if converged:
            break

    # Closing eta-solve at the final sd: makes (eta_hat, sd) jointly consistent
    # even when the s_d sequence stopped one step before eta caught up (or hit
    # sd_max_iter). Warm-started from the last eta, so ~1 L-BFGS iteration.
    Sigma_inv_allowed = (1.0 / (sd * c)) * Rinv_allowed
    eta_hat, nu_d, _ = _stm_doc_inference(
        indices=indices, counts=counts, expElogbeta=expElogbeta,
        Gamma=Gamma, Sigma_inv_allowed=Sigma_inv_allowed, x=x,
        max_iter=lbfgs_max_iter, tol=lbfgs_tol,
        allowed=allowed, reference=reference, eta_init=eta_warm,
    )
    return eta_hat, sd, nu_d, n_em


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

    acc = doc_rdd.mapPartitions(_local).treeReduce(
        lambda a, b: a.combine(b), depth=depth
    )
    if acc.n == 0:
        raise ValueError("corpus_concentration_stm_rdd: empty document RDD")
    return acc.summary()


def gated_infer_theta(
    global_params, partition, *, c, reference=None,
    lbfgs_max_iter=50, lbfgs_tol=1e-4,
):
    """Build a gated `infer_theta(indices, counts, groups, x=None) -> theta`
    closure at generative scale c, for use as the model-specific inference
    callback the general concentration-heterogeneity diagnostic
    (``spark_vi.eval.topic.concentration_heterogeneity``) needs.

    The core diagnostic's `infer_theta(indices, counts) -> theta` signature
    is model-agnostic and takes no notion of gating; a GATED document also
    carries a `groups` field that determines its allowed (background ∪ own
    group) topic set (``partition.allowed_indices``), so this adapter's
    returned callable takes one more positional argument than the core's.
    `x` (the doc's covariate vector, Gamma^T x is the prior mean) is
    optional and defaults to an all-zero vector when omitted -- callers with
    real per-doc covariates (the normal case) should pass ``doc.x``.

    Runs the SAME per-doc gated Laplace E-step as
    ``corpus_heldout_scale_sweep_gated`` at a fixed scale c: prior precision
    ``Sigma_inv_allowed = (1/c) * safe_inverse(R[allowed])``, R the fit's
    diagonal-normalized correlation from ``global_params["Sigma"]`` (ADR
    0034/0036). ``R[allowed]`` is cached per distinct allowed set (a
    corpus typically has few distinct group combinations, so this cache is
    small and reused across every document sharing a group). The MAP
    eta_hat is converted to the K-length display theta via
    ``_gated_mode_theta`` (allowed topics filled, disallowed exactly 0).

    ``global_params`` is the dict with "lambda" (K,V), "Gamma" (P,K), "Sigma"
    (K,K, the fit's correlation R); ``partition`` is a TopicBlockPartition (or
    the implicit all-background one).
    """
    from scipy.special import digamma

    from spark_vi.models.topic._linalg import safe_inverse

    lam = np.asarray(global_params["lambda"], dtype=np.float64)
    Gamma = np.asarray(global_params["Gamma"], dtype=np.float64)
    Sigma = np.asarray(global_params["Sigma"], dtype=np.float64)
    K = lam.shape[0]
    lam_rowsum = lam.sum(axis=1, keepdims=True)
    expElogbeta = np.exp(digamma(lam) - digamma(lam_rowsum))

    d = np.diag(Sigma)
    R = Sigma / np.sqrt(np.outer(d, d))

    rinv_cache: dict[tuple, np.ndarray] = {}

    def infer_theta(indices, counts, groups, x=None):
        allowed = partition.allowed_indices(groups)
        key = tuple(allowed.tolist())
        Rinv_allowed = rinv_cache.get(key)
        if Rinv_allowed is None:
            Rinv_allowed = safe_inverse(R[np.ix_(allowed, allowed)])
            rinv_cache[key] = Rinv_allowed
        Sigma_inv_allowed = (1.0 / c) * Rinv_allowed
        if x is None:
            x = np.zeros(Gamma.shape[0], dtype=np.float64)
        eta_hat, _, _ = _stm_doc_inference(
            indices=indices, counts=counts,
            expElogbeta=expElogbeta,
            Gamma=Gamma, Sigma_inv_allowed=Sigma_inv_allowed, x=x,
            max_iter=lbfgs_max_iter, tol=lbfgs_tol,
            allowed=allowed, reference=reference,
        )
        return _gated_mode_theta(eta_hat, allowed, K)

    return infer_theta


def corpus_concentration_heterogeneity_gated(
    docs, global_params, partition, *, c, reference=None,
    lbfgs_max_iter=50, lbfgs_tol=1e-4,
) -> dict:
    """Driver-side (in-memory) gated concentration-heterogeneity diagnostic.

    Numpy oracle / testing counterpart of ``corpus_concentration_heterogeneity_rdd``
    (see that function for the distributed version and the full derivation).
    Runs the general per-doc raw-vs-dedup burstiness diagnostic
    (``spark_vi.eval.topic.concentration_heterogeneity``) with theta supplied
    by the gated MAP E-step at scale c (``gated_infer_theta``), reusing the
    SAME skip guard (total < 2 tokens or a single unique token) and the SAME
    aggregation (``summarize_concentration_heterogeneity``) that
    ``concentration_raw_vs_dedup`` uses -- this function does not reimplement
    either, it only supplies the per-doc theta.

    ``docs`` is an in-memory list of STMDocument. Returns
    ``summarize_concentration_heterogeneity``'s summary dict plus
    ``{"sample_frac": None, "c": c}`` (n_docs/n_skipped are already part of
    that summary).
    """
    from spark_vi.eval.topic.concentration import doc_concentration
    from spark_vi.eval.topic.concentration_heterogeneity import (
        _json_safe, dedup_counts, doc_burstiness,
        summarize_concentration_heterogeneity,
    )

    infer_theta = gated_infer_theta(
        global_params, partition, c=c, reference=reference,
        lbfgs_max_iter=lbfgs_max_iter, lbfgs_tol=lbfgs_tol,
    )

    top_mass_raw: list[float] = []
    top_mass_dedup: list[float] = []
    eff_topics_raw: list[float] = []
    eff_topics_dedup: list[float] = []
    repeat_fraction: list[float] = []
    n_skipped = 0

    for doc in docs:
        indices = np.asarray(doc.indices)
        counts = np.asarray(doc.counts, dtype=np.float64)
        burst = doc_burstiness(indices, counts)
        if burst["total"] < 2.0 or burst["unique"] <= 1:
            n_skipped += 1
            continue

        theta_raw = infer_theta(indices, counts, doc.groups, doc.x)
        theta_dedup = infer_theta(indices, dedup_counts(counts), doc.groups, doc.x)
        top_raw, eff_raw = doc_concentration(theta_raw)
        top_dedup, eff_dedup = doc_concentration(theta_dedup)

        top_mass_raw.append(top_raw)
        top_mass_dedup.append(top_dedup)
        eff_topics_raw.append(eff_raw)
        eff_topics_dedup.append(eff_dedup)
        repeat_fraction.append(burst["repeat_fraction"])

    summary = summarize_concentration_heterogeneity(
        top_mass_raw=np.array(top_mass_raw, dtype=np.float64),
        top_mass_dedup=np.array(top_mass_dedup, dtype=np.float64),
        eff_topics_raw=np.array(eff_topics_raw, dtype=np.float64),
        eff_topics_dedup=np.array(eff_topics_dedup, dtype=np.float64),
        repeat_fraction=np.array(repeat_fraction, dtype=np.float64),
        n_skipped=n_skipped,
    )
    summary["sample_frac"] = None
    summary["c"] = c
    return _json_safe(summary)


def corpus_concentration_heterogeneity_rdd(
    doc_rdd, global_params, partition, *, c, reference=None,
    sample_frac=None, seed=0,
    lbfgs_max_iter=50, lbfgs_tol=1e-4,
) -> dict:
    """Distributed gated concentration-heterogeneity diagnostic.

    Makes the general per-doc raw-vs-dedup burstiness diagnostic
    (``spark_vi.eval.topic.concentration_heterogeneity.concentration_raw_vs_dedup``)
    runnable on a distributed gated-STM corpus, by running the per-doc gated
    MAP E-step (``gated_infer_theta``, the same machinery
    ``corpus_heldout_scale_sweep_gated_rdd`` uses) on workers and reusing the
    core's aggregation (``summarize_concentration_heterogeneity``) on the
    driver -- see those two functions for the diagnostic's math and the
    gated inference derivation, respectively; nothing here reimplements
    either.

    ``sample_frac`` (optional): the diagnostic does not need the full
    corpus, so callers may subsample first (``doc_rdd.sample(False,
    sample_frac, seed)``) for cost control on a large corpus.

    Distributed via ``mapPartitions``: each partition builds ONE
    ``gated_infer_theta`` closure (so ``R[allowed]`` is cached once per
    partition, not per document) and, for every document, applies the SAME
    skip guard as the core (total token count < 2 or a single unique
    token -- see ``concentration_raw_vs_dedup``'s docstring), computing the
    per-doc 5-tuple (top_mass_raw, top_mass_dedup, eff_topics_raw,
    eff_topics_dedup, repeat_fraction) for surviving documents and `None`
    for skipped ones. Only these small per-doc scalars -- never the
    documents, theta vectors, or per-doc eta_hat -- are collected to the
    driver, which then calls ``summarize_concentration_heterogeneity`` on
    the assembled arrays (identical aggregation to the driver-side/numpy
    ``corpus_concentration_heterogeneity_gated``).

    ``global_params`` and ``partition`` are broadcast via the same
    Spark-safe default-arg-closure convention
    ``corpus_heldout_scale_sweep_gated_rdd`` uses; helpers are imported
    inside the closure so it is picklable on workers.

    ``doc_rdd`` is an RDD of STMDocument. Returns
    ``summarize_concentration_heterogeneity``'s summary dict plus
    ``{"sample_frac": sample_frac, "c": c}`` (n_docs/n_skipped are already
    part of that summary). Raises ValueError if the (possibly sampled)
    RDD collects zero documents.
    """
    work_rdd = doc_rdd
    if sample_frac is not None:
        work_rdd = work_rdd.sample(False, sample_frac, seed)

    sc = work_rdd.context
    gp_bcast = sc.broadcast({
        k: np.asarray(v, dtype=np.float64) for k, v in global_params.items()
    })
    p_bcast = sc.broadcast(partition)

    def _local(rows, _gp=gp_bcast, _p=p_bcast):
        from spark_vi.eval.topic.concentration import doc_concentration
        from spark_vi.eval.topic.concentration_heterogeneity import (
            dedup_counts, doc_burstiness,
        )
        from spark_vi.mllib.topic.stm import gated_infer_theta

        gp = _gp.value
        part = _p.value
        infer_theta = gated_infer_theta(
            gp, part, c=c, reference=reference,
            lbfgs_max_iter=lbfgs_max_iter, lbfgs_tol=lbfgs_tol,
        )

        out = []
        for doc in rows:
            indices = np.asarray(doc.indices)
            counts = np.asarray(doc.counts, dtype=np.float64)
            burst = doc_burstiness(indices, counts)
            if burst["total"] < 2.0 or burst["unique"] <= 1:
                out.append(None)
                continue
            theta_raw = infer_theta(indices, counts, doc.groups, doc.x)
            theta_dedup = infer_theta(
                indices, dedup_counts(counts), doc.groups, doc.x
            )
            top_raw, eff_raw = doc_concentration(theta_raw)
            top_dedup, eff_dedup = doc_concentration(theta_dedup)
            out.append((top_raw, top_dedup, eff_raw, eff_dedup, burst["repeat_fraction"]))
        return out

    collected = work_rdd.mapPartitions(_local).collect()
    if len(collected) == 0:
        raise ValueError("corpus_concentration_heterogeneity_rdd: empty document RDD")

    n_skipped = sum(1 for item in collected if item is None)
    good = [item for item in collected if item is not None]

    from spark_vi.eval.topic.concentration_heterogeneity import (
        _json_safe, summarize_concentration_heterogeneity,
    )

    if good:
        arr = np.array(good, dtype=np.float64)
        top_mass_raw, top_mass_dedup = arr[:, 0], arr[:, 1]
        eff_topics_raw, eff_topics_dedup = arr[:, 2], arr[:, 3]
        repeat_fraction = arr[:, 4]
    else:
        top_mass_raw = top_mass_dedup = eff_topics_raw = eff_topics_dedup = (
            repeat_fraction
        ) = np.array([], dtype=np.float64)

    summary = summarize_concentration_heterogeneity(
        top_mass_raw=top_mass_raw,
        top_mass_dedup=top_mass_dedup,
        eff_topics_raw=eff_topics_raw,
        eff_topics_dedup=eff_topics_dedup,
        repeat_fraction=repeat_fraction,
        n_skipped=n_skipped,
    )
    summary["sample_frac"] = sample_frac
    summary["c"] = c
    return _json_safe(summary)


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
    so the caller can build the dashboard's per-doc theta ("topic mass
    distribution") histogram via ``charmpheno.export.compute_theta_aggregates``.
    This is the STM analog of LDA's per-doc gamma the dashboard already
    histograms.

    Inference is under the E-step prior eta ~ N(mu, Sigma_prior) with
    ``Sigma_prior = scale * Sigma_fitted`` (``Sigma_fitted =
    global_params["Sigma"]``, the unit-diagonal correlation R of ADR 0034).
    ``scale`` is the GENERATION-SCALE multiplier on the fitted Sigma: at
    ``scale=1.0`` (the default) the prior is the raw fit and this is inference
    under the fitted model (c == 1) -- byte-identical to the pre-``scale``
    behavior. At the calibrated held-out ``eta_scale`` (c ~ 4.6) the prior
    penalty is 1/scale, so eta_hat is more data-driven and theta_hat is PEAKIER,
    giving the honest per-doc concentration instead of the over-diffuse
    unit-scale fit (the project's "unit-prior per-doc outputs are not
    measurements" result). ``scale`` is a passed-in calibrated value, not a
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


def corpus_heldout_scale_sweep_gated(
    docs, global_params, partition, *, c_grid,
    holdout_frac=0.3, reference=None, seed=0,
    lbfgs_max_iter=50, lbfgs_tol=1e-4,
    marginalize=False, n_samples=64,
) -> dict:
    """Driver-side (in-memory) held-out predictive-LL sweep over the STM
    generative covariance scale c, for a GATED fit.

    This is the gated, distributed-ready counterpart of the LOCAL, non-gated
    diagnostic validated by insight 0038
    (``spark_vi.eval.topic.concentration_recovery``): held-out within-document
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

    ``marginalize`` (default False, so existing callers are byte-for-byte
    unaffected): when True, per-doc/per-c scoring is routed through the
    Laplace-sample posterior-predictive instead of the MAP theta_hat
    plug-in. The per-doc E-step already returns a Laplace covariance nu_d
    over the free (allowed, non-reference) topics alongside the MAP mode
    eta_hat (Blei & Lafferty 2007); ``laplace_theta_samples`` draws
    ``n_samples`` theta from that N(eta_hat, nu_d) restricted to the doc's
    ``allowed`` set (with the reference fixed at 0 and disallowed topics
    exactly 0), and ``marginalized_predictive_loglik`` scores the held-out
    tokens by the log-of-average (not average-of-log) predictive -- the
    ordering that removes the MAP plug-in bias (Wallach, Murray,
    Salakhutdinov, Mimno 2009, "Evaluation methods for topic models", ICML).
    The sample rng is created FRESH as ``np.random.default_rng(seed + i)``
    at the sampling point inside the per-c loop (i = the doc's enumerate
    index): every c value for a given doc reuses the same standard-normal
    draws (common random numbers -- only c, hence nu_d, varies), matching
    ``heldout_split``'s split-seed discipline and required for numpy/RDD
    parity (``corpus_heldout_scale_sweep_gated_rdd``).

    ``docs`` is an in-memory list of STMDocument; ``global_params`` is the
    dict with "lambda" (K,V), "Gamma" (P,K), "Sigma" (K,K, the fit's
    correlation R); ``partition`` is a TopicBlockPartition (or the implicit
    all-background one). For a live cluster corpus use
    ``corpus_heldout_scale_sweep_gated_rdd``.
    """
    from scipy.special import digamma

    from spark_vi.eval.topic.concentration_recovery import (
        _predictive_loglik, heldout_split,
        laplace_theta_samples, marginalized_predictive_loglik,
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
            eta_hat, nu_d, _ = _stm_doc_inference(
                indices=visible_doc.indices, counts=visible_doc.counts,
                expElogbeta=expElogbeta,
                Gamma=Gamma, Sigma_inv_allowed=Sigma_inv_allowed, x=doc.x,
                max_iter=lbfgs_max_iter, tol=lbfgs_tol,
                allowed=allowed, reference=reference,
            )
            if marginalize:
                th = laplace_theta_samples(
                    eta_hat, nu_d, allowed, K, reference=reference,
                    n_samples=n_samples, rng=np.random.default_rng(seed + i),
                )
                sum_ll[c] += marginalized_predictive_loglik(
                    th, beta_prob, held_indices, held_counts)
            else:
                theta_hat = _gated_mode_theta(eta_hat, allowed, K)
                sum_ll[c] += _predictive_loglik(theta_hat, beta_prob, held_indices, held_counts)
            n_tokens[c] += n_held

    lls = {c: sum_ll[c] / n_tokens[c] for c in c_grid}
    argmax_c = max(lls, key=lls.get)
    return {"lls": lls, "argmax_c": argmax_c, "n_docs": n_docs}


def _nu_key(nu):
    return "inf" if not math.isfinite(nu) else float(nu)


def _c_sweep_at_nu(
    docs, *, expElogbeta, beta_prob, Gamma, R, Rinv_cache, partition, c_grid, nu,
    holdout_frac, reference, seed, K, lbfgs_max_iter, lbfgs_tol,
    sd_max_iter, sd_tol,
):
    """One 1-D held-out c-sweep at a fixed nu. Returns ({c: mean_per_token_ll},
    argmax_c).

    Each (doc, c) pair is solved from a COLD start (eta_init=None), matching
    ``corpus_heldout_scale_sweep_gated`` byte-for-byte at nu=inf (the load-
    bearing nesting check, ``test_nu_inf_column_matches_gaussian_sweep``).
    An earlier version warm-started eta across c within each doc; empirically
    (verified against this exact synthetic corpus) that changes the recovered
    per-doc mode by O(1e-3-1e-2) in held-out log-lik, NOT just solver-tolerance
    noise -- it persists at gtol as tight as 1e-8, so it lands in a genuinely
    different stationary point. The per-doc data term here is a log-MIXTURE,
    -log(softmax(eta)^T expElogbeta_w) (the direct predictive-probability form
    ``_predictive_loglik`` scores against, not the Jensen/variational surrogate
    sum_k p_k log(beta_k) that is guaranteed concave), so it is not globally
    concave in eta and a warm start is not guaranteed to reach the same basin
    as a cold start. Cold-starting every (doc, c) trades some redundant
    L-BFGS work for a result that is reproducible and consistent with the
    Gaussian sweep by construction rather than by luck."""
    from spark_vi.eval.topic.concentration_recovery import (
        _predictive_loglik, heldout_split,
    )
    from spark_vi.models.topic._linalg import safe_inverse

    sum_ll = {c: 0.0 for c in c_grid}
    n_tok = {c: 0 for c in c_grid}
    for i, doc in enumerate(docs):
        split = heldout_split(doc, holdout_frac=holdout_frac, seed=seed + i)
        if split is None:
            continue
        visible_doc, held_indices, held_counts = split
        if held_counts.size == 0:
            continue
        allowed = partition.allowed_indices(doc.groups)
        key = tuple(allowed.tolist())
        Rinv_allowed = Rinv_cache.get(key)
        if Rinv_allowed is None:
            Rinv_allowed = safe_inverse(R[np.ix_(allowed, allowed)])
            Rinv_cache[key] = Rinv_allowed
        n_held = int(held_counts.sum())
        for c in c_grid:
            eta_hat, _sd, _nu_d, _n = _stm_doc_inference_tprior(
                indices=visible_doc.indices, counts=visible_doc.counts,
                expElogbeta=expElogbeta, Gamma=Gamma, Rinv_allowed=Rinv_allowed,
                x=doc.x, c=c, nu=nu, allowed=allowed, reference=reference,
                eta_init=None, lbfgs_max_iter=lbfgs_max_iter,
                lbfgs_tol=lbfgs_tol, sd_max_iter=sd_max_iter, sd_tol=sd_tol,
            )
            theta_hat = _gated_mode_theta(eta_hat, allowed, K)
            sum_ll[c] += _predictive_loglik(theta_hat, beta_prob, held_indices, held_counts)
            n_tok[c] += n_held
    lls = {c: (sum_ll[c] / n_tok[c] if n_tok[c] else float("-inf")) for c in c_grid}
    argmax_c = max(lls, key=lls.get)
    return lls, argmax_c


def _count_contributing(docs, partition, holdout_frac, seed):
    from spark_vi.eval.topic.concentration_recovery import heldout_split
    n = 0
    for i, doc in enumerate(docs):
        split = heldout_split(doc, holdout_frac=holdout_frac, seed=seed + i)
        if split is None:
            continue
        _v, _hi, hc = split
        if hc.size == 0:
            continue
        n += 1
    return n


def corpus_tprior_scale_sweep_gated(
    docs, global_params, partition, *, c_grid, nu_grid,
    holdout_frac=0.3, drift_fracs=(0.2, 0.3, 0.5), reference=None, seed=0,
    lbfgs_max_iter=50, lbfgs_tol=1e-4, sd_max_iter=10, sd_tol=1e-4,
) -> dict:
    """Driver-side 2-D held-out (c, nu) sweep for the multivariate-t per-document
    scale diagnostic (design doc 2026-07-10-tprior-per-document-scale-design.md).

    Mirrors ``corpus_heldout_scale_sweep_gated`` (same heldout_split, same
    inference-vs-scoring role split, same short-doc skips). Emits the (c, nu)
    grid + argmax, the f-drift readout (c* across drift_fracs at nu=inf vs
    nu=nu*), and the s_d readout (sd and sd*c* quantiles at (c*, nu*), inferred
    on full docs). Both readouts emit numbers only, no verdicts. nu=inf column
    reproduces the Gaussian sweep (nesting). See the RDD sibling for cluster use.
    """
    from scipy.special import digamma
    from spark_vi.eval.topic.concentration_heterogeneity import _json_safe

    lam = np.asarray(global_params["lambda"], dtype=np.float64)
    Gamma = np.asarray(global_params["Gamma"], dtype=np.float64)
    Sigma = np.asarray(global_params["Sigma"], dtype=np.float64)
    K = lam.shape[0]
    lam_rowsum = lam.sum(axis=1, keepdims=True)
    expElogbeta = np.exp(digamma(lam) - digamma(lam_rowsum))
    beta_prob = lam / lam_rowsum
    d = np.diag(Sigma)
    R = Sigma / np.sqrt(np.outer(d, d))
    Rinv_cache: dict[tuple, np.ndarray] = {}

    common = dict(
        expElogbeta=expElogbeta, beta_prob=beta_prob, Gamma=Gamma, R=R,
        Rinv_cache=Rinv_cache, partition=partition, c_grid=list(c_grid),
        holdout_frac=holdout_frac, reference=reference, seed=seed, K=K,
        lbfgs_max_iter=lbfgs_max_iter, lbfgs_tol=lbfgs_tol,
        sd_max_iter=sd_max_iter, sd_tol=sd_tol,
    )

    # --- 2-D grid at holdout_frac ---
    grid = []
    lls_by_nu = {}
    for nu in nu_grid:
        lls, _ = _c_sweep_at_nu(docs, nu=nu, **common)
        lls_by_nu[_nu_key(nu)] = lls
        for c in c_grid:
            grid.append({"c": float(c), "nu": _nu_key(nu), "ll": float(lls[c])})
    best = max(grid, key=lambda r: r["ll"])
    c_star = best["c"]
    nu_star = math.inf if best["nu"] == "inf" else float(best["nu"])

    # --- drift readout: c*(f) at nu=inf vs nu=nu* ---
    def _c_star_at(nu, frac):
        d2 = dict(common); d2["holdout_frac"] = frac
        _, argmax_c = _c_sweep_at_nu(docs, nu=nu, **d2)
        return float(argmax_c)

    gaussian = [{"frac": float(f), "c_star": _c_star_at(math.inf, f)} for f in drift_fracs]
    tprior = [{"frac": float(f), "c_star": _c_star_at(nu_star, f)} for f in drift_fracs]
    def _spread(rows):
        cs = [r["c_star"] for r in rows]
        return float(max(cs) - min(cs))

    # --- s_d readout at (c*, nu*), full docs (no split) ---
    from spark_vi.models.topic._linalg import safe_inverse
    sd_vals = []
    for doc in docs:
        allowed = partition.allowed_indices(doc.groups)
        key = tuple(allowed.tolist())
        Rinv_allowed = Rinv_cache.get(key)
        if Rinv_allowed is None:
            Rinv_allowed = safe_inverse(R[np.ix_(allowed, allowed)])
            Rinv_cache[key] = Rinv_allowed
        _eta, sd, _nu_d, _n = _stm_doc_inference_tprior(
            indices=doc.indices, counts=doc.counts, expElogbeta=expElogbeta,
            Gamma=Gamma, Rinv_allowed=Rinv_allowed, x=doc.x, c=c_star, nu=nu_star,
            allowed=allowed, reference=reference,
            lbfgs_max_iter=lbfgs_max_iter, lbfgs_tol=lbfgs_tol,
            sd_max_iter=sd_max_iter, sd_tol=sd_tol,
        )
        sd_vals.append(sd)
    sd_arr = np.asarray(sd_vals, dtype=np.float64)
    def _q(a):
        ps = np.quantile(a, [0.10, 0.25, 0.50, 0.75, 0.90])
        return {"p10": float(ps[0]), "p25": float(ps[1]), "p50": float(ps[2]),
                "p75": float(ps[3]), "p90": float(ps[4])}

    n_docs = _count_contributing(docs, partition, holdout_frac, seed)

    out = {
        "grid": grid,
        "argmax": {"c": c_star, "nu": _nu_key(nu_star), "ll": best["ll"]},
        "n_docs": n_docs,
        "drift": {"fracs": [float(f) for f in drift_fracs],
                  "gaussian": gaussian, "tprior": tprior,
                  "gaussian_spread": _spread(gaussian),
                  "tprior_spread": _spread(tprior)},
        "sd_readout": {"n_docs": int(sd_arr.size),
                       "sd_quantiles": _q(sd_arr),
                       "sd_c_quantiles": _q(sd_arr * c_star)},
    }
    return _json_safe(out)


def corpus_heldout_scale_sweep_gated_rdd(
    doc_rdd, global_params, partition, *, c_grid,
    holdout_frac=0.3, reference=None, seed=0,
    lbfgs_max_iter=50, lbfgs_tol=1e-4, depth=2,
    marginalize=False, n_samples=64,
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

    ``marginalize`` (default False, existing callers unaffected) mirrors
    ``corpus_heldout_scale_sweep_gated``'s option byte-for-byte: when True,
    per-doc/per-c scoring routes through ``laplace_theta_samples`` +
    ``marginalized_predictive_loglik`` instead of the MAP theta_hat plug-in
    (see that function's docstring for the full derivation and citations).
    The sample rng is ``np.random.default_rng(seed + idx)`` with ``idx`` the
    SAME ``zipWithIndex`` index used for ``heldout_split`` above -- common
    random numbers across c for a given doc, and the doc-for-doc alignment
    with the numpy path's ``seed + i`` (``i`` = ``enumerate(docs)`` index)
    that numpy/RDD parity depends on.

    Raises ValueError if the reduced document count is 0.
    """
    sc = doc_rdd.context
    gp_bcast = sc.broadcast({
        k: np.asarray(v, dtype=np.float64) for k, v in global_params.items()
    })
    p_bcast = sc.broadcast(partition)
    c_list = list(c_grid)
    c_bcast = sc.broadcast(c_list)

    def _local(rows, _gp=gp_bcast, _p=p_bcast, _cg=c_bcast,
               _marginalize=marginalize, _n_samples=n_samples):
        from scipy.special import digamma

        from spark_vi.eval.topic.concentration_recovery import (
            _predictive_loglik, heldout_split,
            laplace_theta_samples, marginalized_predictive_loglik,
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
                eta_hat, nu_d, _ = _stm_doc_inference(
                    indices=visible_doc.indices, counts=visible_doc.counts,
                    expElogbeta=expElogbeta,
                    Gamma=Gamma, Sigma_inv_allowed=Sigma_inv_allowed, x=doc.x,
                    max_iter=lbfgs_max_iter, tol=lbfgs_tol,
                    allowed=allowed, reference=reference,
                )
                if _marginalize:
                    th = laplace_theta_samples(
                        eta_hat, nu_d, allowed, K, reference=reference,
                        n_samples=_n_samples, rng=np.random.default_rng(seed + idx),
                    )
                    acc[c][0] += marginalized_predictive_loglik(
                        th, beta_prob, held_indices, held_counts)
                else:
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

    acc, n_docs = (
        doc_rdd.zipWithIndex().mapPartitions(_local).treeReduce(_combine, depth=depth)
    )
    if n_docs == 0:
        raise ValueError("corpus_heldout_scale_sweep_gated_rdd: empty document RDD")

    lls = {c: acc[c][0] / acc[c][1] for c in c_list}
    argmax_c = max(lls, key=lls.get)
    return {"lls": lls, "argmax_c": argmax_c, "n_docs": n_docs}


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
        estimate_global_scale: bool = False,
        global_scale_step_cap: float = 1.2,
        sigma_diagonal_pin: float = 1.0,
        spectral_init: bool = True,
        spectral_method: str = "auto",           # "auto" | "dense" | "scalable"
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
        self.estimate_global_scale = bool(estimate_global_scale)
        self.global_scale_step_cap = float(global_scale_step_cap)
        # Pin Sigma_ii to a constant generative scale c (Sigma = c*R); default 1.0
        # is the standard unit-diagonal pin. Validated (mutual exclusion with the
        # estimate_* flags, c > 0) by OnlineSTM at fit() time. See OnlineSTM for
        # the full ADR 0034 / ADR 0036 rationale.
        self.sigma_diagonal_pin = float(sigma_diagonal_pin)
        self.spectral_init = bool(spectral_init)
        if spectral_method not in {"auto", "dense", "scalable"}:
            raise ValueError(
                "spectral_method must be 'auto', 'dense', or 'scalable', got "
                f"{spectral_method!r}")
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
            estimate_global_scale=self.estimate_global_scale,
            global_scale_step_cap=self.global_scale_step_cap,
            sigma_diagonal_pin=self.sigma_diagonal_pin,
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
        # Resolve spectral_method="auto" to dense/scalable by vocab size (ADR 0037);
        # explicit dense/scalable pass through. Resolved unconditionally so metadata
        # records what would run even when spectral_init is off.
        resolved_spectral_method = resolve_spectral_method(
            self.spectral_method, vocab_size)
        if self.spectral_method == "auto" and resolved_spectral_method == "scalable":
            log.warning(
                "spectral_method='auto': vocab_size %d >= %d threshold; routing to "
                "the scalable random-projection init (dense V×V co-occurrence would "
                "be ~%.1f GB on the driver). Pass spectral_method='dense' to force "
                "the exact path.",
                vocab_size, SPECTRAL_AUTO_VOCAB_THRESHOLD,
                8.0 * vocab_size * vocab_size / 1e9)

        data_summary = None
        if self.spectral_init and resume_from is None:
            partition = model._effective_partition()
            if resolved_spectral_method == "scalable":
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
            # Resolved method = what actually ran (auto -> dense/scalable);
            # requested = what the caller asked for (ADR 0037).
            "spectral_method": resolved_spectral_method,
            "spectral_method_requested": self.spectral_method,
            "estimate_sigma_diagonal": self.estimate_sigma_diagonal,
            "estimate_global_scale": self.estimate_global_scale,
            "global_scale_step_cap": self.global_scale_step_cap,
            "sigma_diagonal_pin": self.sigma_diagonal_pin,
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
