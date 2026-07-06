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


def _infer_theta(
    visible_doc, allowed, expElogbeta, Gamma, R, c, x, reference, K,
    max_iter, tol,
) -> np.ndarray:
    """Infer the gated mode topic-proportion vector theta over ``allowed``.

    Shared by ``doc_predictive_gain`` (the real oracle) and ``null_delta``
    (the permuted-topic null) so both call the SAME sequence -- form
    Sigma_inv_allowed from the correlation R at scale c, run the per-doc
    Laplace E-step restricted to ``allowed`` (with ``reference`` pinned if
    given), then collapse eta_hat to a gated mode theta -- with no room for
    the two paths to drift apart. Only the ``expElogbeta`` passed in differs
    between the two callers (real vs. permuted); everything else is
    identical.
    """
    Rinv = safe_inverse(R[np.ix_(allowed, allowed)])
    Sigma_inv_allowed = (1.0 / c) * Rinv
    eta_hat, _, _ = _stm_doc_inference(
        indices=visible_doc.indices, counts=visible_doc.counts,
        expElogbeta=expElogbeta, Gamma=Gamma, Sigma_inv_allowed=Sigma_inv_allowed,
        x=x, max_iter=max_iter, tol=tol,
        allowed=allowed, reference=reference,
    )
    return _gated_mode_theta(eta_hat, allowed, K)


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

    theta_full = _infer_theta(
        visible_doc, allowed, expElogbeta, Gamma, R, c, doc.x, reference, K,
        lbfgs_max_iter, lbfgs_tol,
    )
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

        theta_k = _infer_theta(
            visible_doc, allowed_k, expElogbeta, Gamma, R, c, doc.x, reference, K,
            lbfgs_max_iter, lbfgs_tol,
        )
        ll_k = _predictive_loglik(theta_k, beta_prob, held_indices, held_counts)
        delta[p] = ll_full - ll_k

        ll_k_dedup = _predictive_loglik(theta_k, beta_prob, held_indices, held_ones)
        dedup_delta[p] = ll_full_dedup - ll_k_dedup

    return DocGain(
        allowed=allowed, delta=delta, ll_full=ll_full, n_held=n_held,
        theta_full=theta_full, dedup_delta=dedup_delta,
    )


def null_delta(
    doc, global_params, partition, *, c, reference=None,
    holdout_frac=0.3, seed, n_perm=4, rng_seed=0, lbfgs_max_iter=50, lbfgs_tol=1e-4,
) -> np.ndarray | None:
    """Permuted-topic NULL BAND for ONE document: what Delta does a topic
    that explains NOTHING produce?

    There is no hard zero threshold for "topic k is present in document d".
    Even a topic with zero true held-out support produces a small nonzero
    Delta_k, because the Gaussian prior regularizes the MAP estimate (see
    the module docstring's auto-floor discussion and
    tests/test_predictive_gain.py). So presence must be judged against the
    distribution of Delta a topic with NO real word-signature produces, not
    against 0 -- that distribution is this null band.

    For each of ``n_perm`` samples this: (1) seeded-picks a topic k_i from
    the doc's allowed set (never the pinned ``reference`` -- ablating the
    reference is undefined, same guard as ``doc_predictive_gain``); (2)
    shuffles k_i's row of lambda across the vocabulary, destroying its
    learned word identities while preserving its row-sum/normalization (a
    pure relabeling of which word means what to that topic, not a change to
    its concentration); (3) reruns EXACTLY the same infer-over-allowed /
    score / ablate-k_i / infer-over-allowed-minus-k_i / score sequence
    ``doc_predictive_gain`` uses for a real topic (via the shared
    ``_infer_theta`` helper), against the SAME held-out split (same
    ``seed``/``holdout_frac``) so the null samples are directly comparable
    to the real Delta on this document; (4) records ll_full - ll_k as one
    null sample.

    Determinism is via ``rng = np.random.default_rng(rng_seed + i)`` for
    each perm index i -- a fresh, explicitly-seeded generator per sample,
    NOT numpy's global RNG state (``np.random.seed``/``np.random.*``) -- so
    ``null_delta`` is reproducible regardless of what other code has done to
    global numpy random state before or after this call.

    Mirrors ``doc_predictive_gain``'s degenerate-split rule exactly: returns
    None if ``heldout_split`` returns None, or the held half is empty.
    Otherwise returns a length-``n_perm`` float array. A sample is:
      - ``np.nan`` if ``allowed`` has no non-reference topic to permute
        (undefined -- e.g. a single-topic doc where that topic IS the
        reference); the corpus-level aggregator drops NaNs when it pools
        per-doc samples into the null band.
      - ``0.0`` if, after choosing k_i, ablating it leaves an empty allowed
        set (no contrast possible -- same as the single-allowed-topic case
        in ``doc_predictive_gain``).
      - otherwise the permuted topic's Delta_k_i against the real allowed
        set, which collapses toward 0 because k_i's beta row is now noise --
        that collapse is the point: it calibrates what "nothing" looks like.

    ``n_perm`` is deliberately small (default 4): each sample feeds a
    corpus-level null band, not a per-document estimate, so this is O(1)
    extra full ablations per document, not O(|allowed|).
    """
    lam = np.asarray(global_params["lambda"], dtype=np.float64)
    Gamma = np.asarray(global_params["Gamma"], dtype=np.float64)
    Sigma = np.asarray(global_params["Sigma"], dtype=np.float64)
    K, V = lam.shape

    d = np.diag(Sigma)
    R = Sigma / np.sqrt(np.outer(d, d))   # correlation; c is applied to THIS

    split = heldout_split(doc, holdout_frac=holdout_frac, seed=seed)
    if split is None:
        return None
    visible_doc, held_indices, held_counts = split
    if held_counts.size == 0:
        return None

    allowed = partition.allowed_indices(doc.groups)

    samples = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        # Seeded per-sample RNG: determinism comes from (rng_seed + i), not
        # from numpy's global random state.
        rng = np.random.default_rng(rng_seed + i)

        draw_pos = int(rng.integers(allowed.size))
        k_i = allowed[draw_pos]
        if reference is not None and k_i == reference:
            # A permuted reference topic is undefined (same as Task 1's
            # ablation guard) -- deterministically advance to the next
            # allowed, non-reference topic rather than re-drawing from rng
            # (keeps the rng's subsequent permutation draw reproducible
            # regardless of where the reference happened to land).
            k_i = None
            for step in range(1, allowed.size):
                cand = allowed[(draw_pos + step) % allowed.size]
                if cand != reference:
                    k_i = cand
                    break
            if k_i is None:
                # allowed has no non-reference topic to permute at all.
                samples[i] = np.nan
                continue

        # Shuffle k_i's beta row across the vocabulary: word identities are
        # scrambled, but the row is still a permutation of the SAME values,
        # so its normalization (row-sum for expElogbeta's digamma terms,
        # E[beta] for beta_prob) is preserved -- only what k_i "means" is
        # destroyed, not its concentration.
        perm = rng.permutation(V)
        lam_perm = lam.copy()
        lam_perm[k_i] = lam[k_i][perm]

        lam_perm_rowsum = lam_perm.sum(axis=1, keepdims=True)
        expElogbeta_perm = np.exp(digamma(lam_perm) - digamma(lam_perm_rowsum))
        beta_prob_perm = lam_perm / lam_perm_rowsum

        theta_full = _infer_theta(
            visible_doc, allowed, expElogbeta_perm, Gamma, R, c, doc.x, reference, K,
            lbfgs_max_iter, lbfgs_tol,
        )
        ll_full = _predictive_loglik(theta_full, beta_prob_perm, held_indices, held_counts)

        pos = int(np.nonzero(allowed == k_i)[0][0])
        allowed_k = np.delete(allowed, pos)
        if allowed_k.size == 0:
            # No contrast possible (k_i was the sole allowed topic).
            samples[i] = 0.0
            continue

        theta_k = _infer_theta(
            visible_doc, allowed_k, expElogbeta_perm, Gamma, R, c, doc.x, reference, K,
            lbfgs_max_iter, lbfgs_tol,
        )
        ll_k = _predictive_loglik(theta_k, beta_prob_perm, held_indices, held_counts)
        samples[i] = ll_full - ll_k

    return samples
