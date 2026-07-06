"""COLD (reference) single-document predictive-gain oracle for a fitted STM.

For a fitted gated STM, the "predictive gain" of topic k for a document d is
the leave-one-topic-out contrast

    Delta_k = LL(allowed) - LL(allowed \\ {k})

where LL(A) is the document's held-out per-token predictive log-likelihood
under the mode topic-proportion estimate theta_hat obtained by restricting
inference to the topic set A. Delta_k measures how much held-out predictive
power topic k contributes to THIS document specifically: a document that
draws no support from topic k loses (almost) nothing when k is removed from
its allowed set (the "auto-floor" property -- see
tests/test_predictive_gain.py for a hand-checkable disjoint-vocab fixture
demonstrating it), while a document whose tokens are actually explained by
k loses real held-out likelihood when k is taken away.

This module is the COLD, brute-force reference implementation: for a
document with |allowed| topics it reruns the full per-doc Laplace E-step
|allowed| additional times (once per ablation), which is deliberately simple
and exact. It is the correctness oracle a later, algebraically-downdated
fast path is validated against -- it introduces zero new numerics, only
reusing the existing E-step/scoring primitives in their established
conventions (mirroring ``corpus_heldout_scale_sweep_gated`` in
spark_vi/mllib/topic/stm.py, which assembles the same primitives for a
scale-grid sweep instead of a topic-ablation loop).

Two conventions matter and are easy to get backwards:

  - INFERENCE vs SCORING. ``expElogbeta`` (exp-digamma of the variational
    lambda) is the data term the fit's own E-step uses and MUST be what
    ``_stm_doc_inference`` sees; ``beta_prob`` (lambda-normalized, i.e.
    E[beta]) is the actual predictive word distribution and MUST be what
    ``_predictive_loglik`` scores against. Swapping them silently
    miscalibrates every Delta_k.
  - SCALE ENTERS VIA CORRELATION, NOT RAW SIGMA. Per ADR 0034/0036, the
    fit's Sigma is normalized to a correlation R = Sigma / sqrt(outer(d, d))
    (d = diag(Sigma)) and the scalar generative scale c multiplies R's
    inverse: Sigma_inv_allowed = (1/c) * safe_inverse(R[allowed, allowed]).
    In current bundles Sigma already has a unit diagonal (R == Sigma), but
    the normalization is done anyway for consistency with every other c-scale
    consumer in this codebase.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import digamma

from spark_vi.eval.topic.concentration_recovery import _predictive_loglik, heldout_split
from spark_vi.mllib.topic.stm import _gated_mode_theta
from spark_vi.models.topic._linalg import safe_inverse
from spark_vi.models.topic.stm import _stm_doc_inference


@dataclass
class DocGain:
    """Per-document leave-one-topic-out predictive-gain result.

    allowed:     the doc's allowed topic indices (from the partition),
                 sorted int64 array, length n_allowed.
    delta:       Delta_k in nats for each position in ``allowed`` (same
                 order/length as ``allowed``): held-out LL with the full
                 allowed set minus held-out LL with topic k ablated.
    ll_full:     held-out predictive log-likelihood (sum over held tokens)
                 under the full allowed set.
    n_held:      number of held-out tokens actually scored.
    theta_full:  length-K mode topic-proportion vector (zeros off
                 ``allowed``), from the full-allowed-set inference.
    dedup_delta: Delta recomputed with held-out counts capped at 1 (i.e.
                 scored against the held-out VOCABULARY rather than the
                 held-out token multiset) -- a variant that down-weights
                 documents with one dominant repeated held-out word.
    """
    allowed: np.ndarray
    delta: np.ndarray
    ll_full: float
    n_held: int
    theta_full: np.ndarray
    dedup_delta: np.ndarray


def doc_predictive_gain(
    doc, global_params, partition, *, c, reference=None,
    holdout_frac=0.3, seed, lbfgs_max_iter=50, lbfgs_tol=1e-4,
) -> DocGain | None:
    """Leave-one-topic-out held-out predictive gain for ONE document (COLD).

    ``global_params`` is the dict with "lambda" (K,V), "Gamma" (P,K), "Sigma"
    (K,K, the fit's correlation R -- see module docstring). ``partition`` is
    a TopicBlockPartition (or the implicit all-background one) used to
    compute the doc's allowed topic set. ``c`` is the scalar STM generative
    covariance scale (Sigma_gen = c*R).

    Splits the document's tokens once via ``heldout_split`` (seeded,
    independent of any topic ablation). Returns None if the split is
    degenerate (mirrors ``corpus_heldout_scale_sweep_gated``'s skip rule: too
    few tokens, or an empty visible or held half) -- there is nothing to
    infer theta_hat from, or nothing held out to score.

    Infers the full-allowed-set mode theta_hat ONCE, scores it, then for
    each position p (topic k = allowed[p]) re-infers theta with topic k
    removed from the allowed set and rescopes. Removing the pinned
    ``reference`` topic is undefined (``_stm_doc_inference`` requires
    reference in allowed) and a document with a single allowed topic has no
    contrast to make; both cases set Delta_k = 0 without any extra
    inference. ``dedup_delta`` reuses the SAME inferred theta_full/theta_k
    (dedup changes only the held-out counts used for scoring, never
    inference) so it costs one extra re-score per ablation, not one extra
    optimization.

    This is O(|allowed|) full L-BFGS inferences per document -- deliberately
    simple and exact; it is the reference a downdated fast path (Task 4) is
    validated against, not the production hot path.
    """
    lam = np.asarray(global_params["lambda"], dtype=np.float64)
    Gamma = np.asarray(global_params["Gamma"], dtype=np.float64)
    Sigma = np.asarray(global_params["Sigma"], dtype=np.float64)
    K = lam.shape[0]
    lam_rowsum = lam.sum(axis=1, keepdims=True)
    expElogbeta = np.exp(digamma(lam) - digamma(lam_rowsum))   # inference term
    beta_prob = lam / lam_rowsum                                # scoring term: E[beta]

    d = np.diag(Sigma)
    R = Sigma / np.sqrt(np.outer(d, d))   # correlation; c is applied to THIS

    split = heldout_split(doc, holdout_frac=holdout_frac, seed=seed)
    if split is None:
        return None
    visible_doc, held_indices, held_counts = split
    if held_counts.size == 0:
        return None

    allowed = partition.allowed_indices(doc.groups)
    n_held = int(held_counts.sum())

    Rinv_allowed = safe_inverse(R[np.ix_(allowed, allowed)])
    Sigma_inv_allowed = (1.0 / c) * Rinv_allowed
    eta_hat, _, _ = _stm_doc_inference(
        indices=visible_doc.indices, counts=visible_doc.counts,
        expElogbeta=expElogbeta, Gamma=Gamma, Sigma_inv_allowed=Sigma_inv_allowed,
        x=doc.x, max_iter=lbfgs_max_iter, tol=lbfgs_tol,
        allowed=allowed, reference=reference,
    )
    theta_full = _gated_mode_theta(eta_hat, allowed, K)
    ll_full = _predictive_loglik(theta_full, beta_prob, held_indices, held_counts)

    held_ones = np.ones_like(held_counts)
    ll_full_dedup = _predictive_loglik(theta_full, beta_prob, held_indices, held_ones)

    n_allowed = allowed.shape[0]
    delta = np.zeros(n_allowed, dtype=np.float64)
    dedup_delta = np.zeros(n_allowed, dtype=np.float64)

    for p in range(n_allowed):
        k = allowed[p]
        allowed_k = np.delete(allowed, p)
        if allowed_k.size == 0 or (reference is not None and k == reference):
            # Undefined (pinned reference) or no contrast (single-topic doc).
            delta[p] = 0.0
            dedup_delta[p] = 0.0
            continue

        Rinv_k = safe_inverse(R[np.ix_(allowed_k, allowed_k)])
        Sigma_inv_k = (1.0 / c) * Rinv_k
        eta_k, _, _ = _stm_doc_inference(
            indices=visible_doc.indices, counts=visible_doc.counts,
            expElogbeta=expElogbeta, Gamma=Gamma, Sigma_inv_allowed=Sigma_inv_k,
            x=doc.x, max_iter=lbfgs_max_iter, tol=lbfgs_tol,
            allowed=allowed_k, reference=reference,
        )
        theta_k = _gated_mode_theta(eta_k, allowed_k, K)
        ll_k = _predictive_loglik(theta_k, beta_prob, held_indices, held_counts)
        delta[p] = ll_full - ll_k

        ll_k_dedup = _predictive_loglik(theta_k, beta_prob, held_indices, held_ones)
        dedup_delta[p] = ll_full_dedup - ll_k_dedup

    return DocGain(
        allowed=allowed, delta=delta, ll_full=ll_full, n_held=n_held,
        theta_full=theta_full, dedup_delta=dedup_delta,
    )
