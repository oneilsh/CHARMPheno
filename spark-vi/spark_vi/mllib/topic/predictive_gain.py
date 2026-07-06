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

import logging
from dataclasses import dataclass

import numpy as np
from scipy.special import digamma

from spark_vi.eval.topic.concentration_recovery import _predictive_loglik, heldout_split
from spark_vi.mllib.topic.stm import _gated_mode_theta
from spark_vi.models.topic._linalg import safe_inverse
from spark_vi.models.topic.stm import _stm_doc_inference, _stm_doc_newton_polish

log = logging.getLogger(__name__)


def _smoothed_predictive_loglik(
    theta, beta_prob, held_idx, held_cnt, *, marginal, eps,
) -> float:
    """Held-out predictive log-likelihood under a background-smoothed
    predictive distribution -- the SCORING-side fix for the 1e-12 log-floor
    contamination (Fable's endorsed "marginal back-off").

    Scores against the mixture

        p_S(w) = (1 - eps) * (theta @ beta_prob)(w) + eps * m_w

    where ``m_w`` (``marginal``) is the corpus unigram (a length-V token
    probability vector, sums to 1) and ``eps`` is the length-aware mixing
    weight. Removing the only topic that explained a held-out token no longer
    sends its predicted mass to a hard 1e-12 floor (which made Delta_k blow up
    by tens of nats per exclusive bursty token); instead the token falls back
    to a background probability eps*m_w > 0, so Delta_k becomes a BOUNDED
    log-lift of the full model over that background baseline. Each per-token
    log-ratio is then bounded by log(p_full(w) / (eps * m_w)).

    This is the standard length-aware Dirichlet-smoothed predictive: mixing a
    fitted distribution with a background/collection unigram under a small
    pseudo-count is the linear-interpolation / Jelinek-Mercer form of Bayesian
    smoothing for held-out language-model scoring (Zhai & Lafferty 2004, "A
    study of smoothing methods for language models applied to ad hoc
    information retrieval"; MacKay & Peto 1995, "A hierarchical Dirichlet
    language model"). The mixing weight eps = lambda / (lambda + n) is exactly
    the Dirichlet posterior-predictive weight on the prior given n observed
    tokens.

    INFERENCE IS UNCHANGED: theta_hat is still inferred from the visible half
    by the fit's own E-step. Smoothing enters ONLY here, where held-out tokens
    are scored. The 1e-300 term is numeric underflow safety only (eps*m_w is
    already strictly positive for any in-corpus held token, m_w > 0)."""
    pred = (1.0 - eps) * (theta @ beta_prob) + eps * marginal   # length V
    return float(np.sum(held_cnt * np.log(pred[held_idx] + 1e-300)))


def _make_scorer(marginal, smoothing_lambda, visible_doc):
    """Build the single held-out scorer used by EVERY scoring path in a
    predictive-gain call (full LL, every ablation LL, dedup, and the null
    permutations) so they all share the SAME (eps, marginal).

    Returns a callable ``score(theta, beta_prob, held_idx, held_cnt) -> float``.

    When ``marginal is None`` this is the historical unsmoothed
    ``_predictive_loglik`` (1e-12 floor) -- the backward-compatible path the
    existing tests pin. When a ``marginal`` (length-V corpus unigram) is
    supplied, it is the background-smoothed scorer with the length-aware
    weight eps = smoothing_lambda / (smoothing_lambda + n_visible), where
    n_visible = the visible-token count (the amount of data theta_hat was
    inferred from). Thin documents -> larger eps -> lean on the marginal, the
    intended uncertainty behavior."""
    if marginal is None:
        def score(theta, beta_prob, held_idx, held_cnt):
            return _predictive_loglik(theta, beta_prob, held_idx, held_cnt)
        return score

    marg = np.asarray(marginal, dtype=np.float64)
    n_visible = float(visible_doc.counts.sum())
    eps = float(smoothing_lambda) / (float(smoothing_lambda) + n_visible)

    def score(theta, beta_prob, held_idx, held_cnt):
        return _smoothed_predictive_loglik(
            theta, beta_prob, held_idx, held_cnt, marginal=marg, eps=eps,
        )
    return score


def _sigma_inv_scaled(R, allowed, c) -> np.ndarray:
    """Marginal prior precision over ``allowed`` at generative scale c:
    (1/c) * safe_inverse(R[allowed, allowed]). The one place scale enters, per
    the module docstring (Sigma_gen = c*R). Shared by every inference call
    (cold full, cold ablation, and the warm-start downdate) so the scale is
    formed identically everywhere."""
    return (1.0 / c) * safe_inverse(R[np.ix_(allowed, allowed)])


def _infer_eta_theta(
    visible_doc, allowed, expElogbeta, Gamma, R, c, x, reference, K,
    max_iter, tol,
) -> tuple[np.ndarray, np.ndarray]:
    """Cold-solve the gated mode over ``allowed``; return (eta_hat, theta).

    Forms Sigma_inv_allowed from the correlation R at scale c, runs the COLD
    per-doc Laplace E-step (``_stm_doc_inference``) restricted to ``allowed``
    (with ``reference`` pinned if given), and collapses eta_hat to a gated mode
    theta. ``eta_hat`` is the full-K MAP point (finite on ``allowed``, -inf
    off it) that the fast downdate path reuses as its Newton warm start; the
    COLD path only needs ``theta``. Both are returned so the two paths share
    one solve and cannot drift apart."""
    Sigma_inv_allowed = _sigma_inv_scaled(R, allowed, c)
    eta_hat, _, _ = _stm_doc_inference(
        indices=visible_doc.indices, counts=visible_doc.counts,
        expElogbeta=expElogbeta, Gamma=Gamma, Sigma_inv_allowed=Sigma_inv_allowed,
        x=x, max_iter=max_iter, tol=tol,
        allowed=allowed, reference=reference,
    )
    return eta_hat, _gated_mode_theta(eta_hat, allowed, K)


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
    identical. Thin wrapper over ``_infer_eta_theta`` that discards eta_hat
    (byte-identical to the historical inline computation)."""
    return _infer_eta_theta(
        visible_doc, allowed, expElogbeta, Gamma, R, c, x, reference, K,
        max_iter, tol,
    )[1]


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
    holdout_frac=0.3, seed, marginal=None, smoothing_lambda=1.0,
    lbfgs_max_iter=50, lbfgs_tol=1e-4,
    fast=False, high_mass_bound=0.2, max_fast_steps=3,
) -> DocGain | None:
    """Leave-one-topic-out held-out predictive gain for ONE document.

    ``fast`` selects the ablation optimizer. ``fast=False`` (default) is the
    COLD oracle: a full cold L-BFGS re-inference per ablated topic (byte-
    identical to before this parameter existed -- the Task-1 tests pin it).
    ``fast=True`` is the warm-start one-Newton-step DOWNDATE: the full mode is
    still solved once cold (for the warm start and theta_full), but each
    ablation is polished with ``_stm_doc_newton_polish`` (a few Newton steps
    warm-started at the full mode) instead of a cold re-solve -- WITH a safety
    fallback: if the warm-start Newton stalls (zero accepted steps -- the warm
    start is in a different basin than the oracle's η=0 start, which happens
    when a dropped high-mass topic forces the objective into its non-convex/
    pathological region), that ablation falls back to the EXACT cold solve, so
    it is byte-identical to the oracle there. The two paths differ ONLY in how
    ``theta_k`` is obtained (warm Newton, or cold on a stall/on fast=False); the
    held-out scoring, reference/empty-allowed skip rules, and dedup reuse are
    shared. ``fast`` is an APPROXIMATION of ``fast=False`` -- see
    ``_stm_doc_newton_polish`` and ``predictive_gain_downdate_audit`` for the
    cold-vs-fast agreement basis.
    ``high_mass_bound`` (default 0.2) and ``max_fast_steps`` (default 3) are
    HEURISTICS (see the ablation loop): a topic with theta_full[k] above the
    bound gets up to ``max_fast_steps`` Newton steps (its removal forces a
    large, locally non-convex move that one step under-captures); a low-mass
    topic gets a single step (removing it barely perturbs the mode).

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

    SCORING: when ``marginal`` (a length-V corpus unigram probability vector)
    is supplied, every held-out score uses the background-smoothed predictive
    p_S(w) = (1 - eps)*(theta@beta)(w) + eps*m_w (see
    ``_smoothed_predictive_loglik``) with the length-aware weight
    eps = smoothing_lambda / (smoothing_lambda + n_visible). This retires the
    1e-12 log-floor: ablating the sole topic that explained a token no longer
    blows Delta_k up by tens of nats -- Delta_k becomes a BOUNDED log-lift over
    the background baseline, and an irrelevant/permuted topic scores Delta ~= 0
    by construction (a meaningful origin). When ``marginal is None`` (default)
    the historical unsmoothed ``_predictive_loglik`` floor path is used, so the
    Task-1..4 fixtures are byte-identical. INFERENCE (theta_hat) is UNCHANGED
    either way -- smoothing is scoring-only.
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

    # ONE scorer, used for every LL below (full, each ablation, dedup): either
    # the background-smoothed mixture (marginal supplied) or the historical
    # floored _predictive_loglik (marginal=None).
    score = _make_scorer(marginal, smoothing_lambda, visible_doc)

    allowed = partition.allowed_indices(doc.groups)
    n_held = int(held_counts.sum())

    if fast:
        # Solve the full mode ONCE (cold); its eta_hat is the Newton warm start
        # reused by every ablation's downdate below.
        eta_hat_full, theta_full = _infer_eta_theta(
            visible_doc, allowed, expElogbeta, Gamma, R, c, doc.x, reference, K,
            lbfgs_max_iter, lbfgs_tol,
        )
    else:
        theta_full = _infer_theta(
            visible_doc, allowed, expElogbeta, Gamma, R, c, doc.x, reference, K,
            lbfgs_max_iter, lbfgs_tol,
        )
    ll_full = score(theta_full, beta_prob, held_indices, held_counts)

    held_ones = np.ones_like(held_counts)
    ll_full_dedup = score(theta_full, beta_prob, held_indices, held_ones)

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

        if fast:
            # Warm-start Newton downdate from the full mode over allowed\{k}.
            # HEURISTIC trigger: high-mass topics (theta_full[k] > bound) get up
            # to max_fast_steps steps to capture the large softmax-renormalization
            # their removal forces; low-mass topics get a single (near-exact) step.
            n_steps = max_fast_steps if theta_full[k] > high_mass_bound else 1
            eta_k, n_accepted = _stm_doc_newton_polish(
                indices=visible_doc.indices, counts=visible_doc.counts,
                expElogbeta=expElogbeta, Gamma=Gamma,
                Sigma_inv_allowed=_sigma_inv_scaled(R, allowed_k, c),
                x=doc.x, allowed=allowed_k, reference=reference,
                eta_warm=eta_hat_full, max_steps=n_steps, tol=lbfgs_tol,
            )
            if n_accepted == 0:
                # STALL: the warm start sits in a region where no damped Newton
                # step descends -- the signature of a dropped high-mass topic
                # pushing the ablated objective into the non-convex/steep region
                # where the cold L-BFGS oracle itself terminates at a start-
                # dependent early point (line-search abort). A warm start in a
                # DIFFERENT basin than the oracle's η=0 start cannot reproduce
                # the oracle there, so fall back to the EXACT cold solve (making
                # the fast path byte-identical to the oracle on exactly these
                # ablations). Empirically ALL cold-vs-fast discrepancies above
                # ~3e-4 are stalls; whenever the polish makes any progress it
                # tracks the oracle to <1e-3 (see predictive_gain_downdate_audit
                # and the fast-vs-cold agreement test).
                theta_k = _infer_theta(
                    visible_doc, allowed_k, expElogbeta, Gamma, R, c, doc.x,
                    reference, K, lbfgs_max_iter, lbfgs_tol,
                )
            else:
                theta_k = _gated_mode_theta(eta_k, allowed_k, K)
        else:
            theta_k = _infer_theta(
                visible_doc, allowed_k, expElogbeta, Gamma, R, c, doc.x, reference, K,
                lbfgs_max_iter, lbfgs_tol,
            )
        ll_k = score(theta_k, beta_prob, held_indices, held_counts)
        delta[p] = ll_full - ll_k

        ll_k_dedup = score(theta_k, beta_prob, held_indices, held_ones)
        dedup_delta[p] = ll_full_dedup - ll_k_dedup

    return DocGain(
        allowed=allowed, delta=delta, ll_full=ll_full, n_held=n_held,
        theta_full=theta_full, dedup_delta=dedup_delta,
    )


def null_delta(
    doc, global_params, partition, *, c, reference=None,
    holdout_frac=0.3, seed, n_perm=4, rng_seed=0, marginal=None, smoothing_lambda=1.0,
    lbfgs_max_iter=50, lbfgs_tol=1e-4,
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

    # SAME smoothed (or floored) scorer the real Delta uses, so the null is
    # measured on the identical scale (see doc_predictive_gain / _make_scorer).
    score = _make_scorer(marginal, smoothing_lambda, visible_doc)

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
        ll_full = score(theta_full, beta_prob_perm, held_indices, held_counts)

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
        ll_k = score(theta_k, beta_prob_perm, held_indices, held_counts)
        samples[i] = ll_full - ll_k

    return samples


# --------------------------------------------------------------------------
# Task 3: corpus-level aggregation (numpy + distributed twins)
# --------------------------------------------------------------------------
#
# corpus_predictive_gain_gated / corpus_predictive_gain_gated_rdd fold the
# per-document DocGain (doc_predictive_gain) and permuted null (null_delta)
# into per-topic arrays over a whole corpus. They mirror the
# numpy-in-memory/`_rdd`-distributed twin pattern used throughout this
# codebase (e.g. corpus_eta_variance_gated{,_rdd},
# corpus_heldout_scale_sweep_gated{,_rdd} in spark_vi/mllib/topic/stm.py):
# the numpy path is the reference for small/test corpora, the `_rdd` path is
# the mapPartitions(_local).treeReduce(_combine) distributed twin, and a
# numpy<->RDD parity test proves they compute the identical thing.
#
# DESIGN DECISIONS (see docstrings below for the full derivation; both are
# flagged for Phase-2 confirmation in the task report):
#
#   1. PRESENCE is a PER-DOCUMENT PAIRED permutation test: each document's
#      real Delta_k is compared to THAT SAME document's own permuted-null
#      samples (from null_delta, run against the identical held-out split),
#      not to a pooled corpus-wide null. This controls for the document's
#      own length/composition (a paired design), is a standard one-sided
#      permutation-test construction (reject if the observed statistic
#      exceeds the max of n_perm null draws -> approx level 1/(n_perm+1)),
#      is single-pass (no second corpus traversal to first pool nulls then
#      re-score presence), and does not depend on the provisional
#      prominence_range bin edges at all. The POOLED corpus null_band
#      (mean/std/histogram/p95 over every doc's finite null samples) is
#      ALSO returned, for the frontend's noise-floor display and Phase-2
#      inspection -- but it is descriptive, not what presence is tested
#      against.
#   2. prominence_range=(-1.0, 10.0) is PROVISIONAL: Delta_k's natural scale
#      (nats of held-out per-document log-likelihood) is not yet calibrated
#      against a real fitted corpus. observed_delta_range is returned
#      precisely so a Phase-2 caller can recalibrate the histogram edges
#      from real numbers rather than trusting this placeholder.


@dataclass
class _PredGainAcc:
    """Per-topic (+ two pooled scalars) accumulator for the corpus
    predictive-gain aggregation. Every array field is length K (indexed by
    topic id) except ``prominence_hist`` (K x n_bins) and ``null_hist``
    (n_bins,). Built once per partition (numpy path: once for the whole
    corpus) and folded across partitions by ``_combine_pred_gain_acc``, which
    is a PURE function (returns a new accumulator, mutates neither input) so
    it is safe to use as ``RDD.treeReduce``'s combiner.

    Fields, and what streams into them per document per allowed topic k
    (``dg`` = this doc's DocGain, ``d_k`` = dg.delta at k's position, ``L`` =
    the doc's full token length, ``sum_j`` = sum(dg.delta) over ALL of the
    doc's allowed topics):

      sum_gain[k]   += d_k                        (-> mean_gain, depth_num)
      depth_den[k]  += sum_j                       (the doc's TOTAL held-out
                       structure attributed to EVERY one of its allowed
                       topics -- see corpus_predictive_gain_gated's
                       docstring for why this, not a per-topic sum, is the
                       correct "depth" denominator)
      dedup_sum[k]  += dg.dedup_delta at k
      count_k[k]    += 1                           (WITHIN-GROUP: a doc only
                       reaches this loop for topics in ITS OWN allowed set,
                       so a foreground topic's count_k is automatically
                       restricted to its own group's documents -- no extra
                       cross-group masking needed anywhere in this module)
      Slen/SdL/SLL/Sdd[k] += L / d_k*L / L*L / d_k*d_k  (streaming sums for
                       the length<->Delta Pearson correlation, finalized
                       once at the end -- see _finalize_pred_gain)
      prominence_hist[k, bin(d_k)] += 1            (per-topic Delta_k
                       histogram; the aggregate distribution that replaces
                       a theta-hat histogram for the predictive-gain view)
      presence_count[k] += 1 IFF the doc had at least one finite null sample
                       AND d_k beats that doc's OWN null max (paired test,
                       decision #1 above); a doc with NO finite null sample
                       contributes to count_k but never to presence_count.
                       presence[k] = presence_count[k] / count_k[k] (see
                       _finalize_pred_gain) -- count_k is the FULL denominator
                       (not restricted to docs whose null could be
                       evaluated); a doc whose null band could not be
                       evaluated (e.g. no non-reference topic to permute)
                       stays in count_k but can never reach presence_count,
                       so it effectively counts AGAINST presence (as if k
                       were "not present" in that doc).
      obs_delta_min/max: running min/max of every d_k ever seen (-> the
                       returned observed_delta_range, decision #2 above).

    Independently, the POOLED corpus null distribution accumulates every
    FINITE null sample from EVERY document (not just one per doc) into
    null_sum/null_sqsum/null_count (mean/std) and null_hist (its own
    histogram over the same bin edges) -- purely descriptive (see decision
    #1), never used to decide presence.
    """
    K: int
    n_bins: int
    sum_gain: np.ndarray
    depth_den: np.ndarray
    dedup_sum: np.ndarray
    Slen: np.ndarray
    SdL: np.ndarray
    SLL: np.ndarray
    Sdd: np.ndarray
    count_k: np.ndarray
    presence_count: np.ndarray
    presence_count_pos: np.ndarray
    prominence_hist: np.ndarray
    null_hist: np.ndarray
    null_sum: float = 0.0
    null_sqsum: float = 0.0
    null_count: int = 0
    n_docs: int = 0
    obs_delta_min: float = float("inf")
    obs_delta_max: float = float("-inf")

    @staticmethod
    def zeros(K: int, n_bins: int) -> "_PredGainAcc":
        return _PredGainAcc(
            K=K, n_bins=n_bins,
            sum_gain=np.zeros(K, dtype=np.float64),
            depth_den=np.zeros(K, dtype=np.float64),
            dedup_sum=np.zeros(K, dtype=np.float64),
            Slen=np.zeros(K, dtype=np.float64),
            SdL=np.zeros(K, dtype=np.float64),
            SLL=np.zeros(K, dtype=np.float64),
            Sdd=np.zeros(K, dtype=np.float64),
            count_k=np.zeros(K, dtype=np.int64),
            presence_count=np.zeros(K, dtype=np.int64),
            presence_count_pos=np.zeros(K, dtype=np.int64),
            prominence_hist=np.zeros((K, n_bins), dtype=np.int64),
            null_hist=np.zeros(n_bins, dtype=np.int64),
        )


def _combine_pred_gain_acc(a: _PredGainAcc, b: _PredGainAcc) -> _PredGainAcc:
    """Functional (treeReduce-safe) merge of two accumulators: elementwise
    sum for every array/count/scalar, min/max for the observed-Delta range.
    Mutates neither ``a`` nor ``b``."""
    return _PredGainAcc(
        K=a.K, n_bins=a.n_bins,
        sum_gain=a.sum_gain + b.sum_gain,
        depth_den=a.depth_den + b.depth_den,
        dedup_sum=a.dedup_sum + b.dedup_sum,
        Slen=a.Slen + b.Slen,
        SdL=a.SdL + b.SdL,
        SLL=a.SLL + b.SLL,
        Sdd=a.Sdd + b.Sdd,
        count_k=a.count_k + b.count_k,
        presence_count=a.presence_count + b.presence_count,
        presence_count_pos=a.presence_count_pos + b.presence_count_pos,
        prominence_hist=a.prominence_hist + b.prominence_hist,
        null_hist=a.null_hist + b.null_hist,
        null_sum=a.null_sum + b.null_sum,
        null_sqsum=a.null_sqsum + b.null_sqsum,
        null_count=a.null_count + b.null_count,
        n_docs=a.n_docs + b.n_docs,
        obs_delta_min=min(a.obs_delta_min, b.obs_delta_min),
        obs_delta_max=max(a.obs_delta_max, b.obs_delta_max),
    )


def _clamped_bin(value: float, edges: np.ndarray, n_bins: int) -> int:
    """Bin index of ``value`` over ``edges`` (length n_bins+1), clamped into
    [0, n_bins-1]: values at or below the lowest edge land in bin 0, values
    at or above the highest edge land in bin n_bins-1 (``np.digitize``
    against the INTERIOR edges already does this -- it has no notion of
    "out of range" -- the explicit ``np.clip`` is a defensive belt-and-
    braces guard, e.g. against a NaN slipping through)."""
    idx = int(np.digitize(value, edges[1:-1]))
    return int(np.clip(idx, 0, n_bins - 1))


def _accumulate_doc(
    acc: _PredGainAcc, doc, idx: int, global_params, partition, *,
    c, reference, holdout_frac, seed, n_perm, bin_edges, n_bins,
    marginal=None, smoothing_lambda=1.0,
    fast=False, high_mass_bound=0.2, max_fast_steps=3,
) -> _PredGainAcc:
    """Fold ONE document's predictive-gain contribution into ``acc``
    (mutated in place -- this is the per-partition-local accumulation step,
    never shared across partitions/workers, so in-place mutation here is
    safe and cheap; only ``_combine_pred_gain_acc``, which crosses the
    treeReduce boundary, must be side-effect-free).

    Shared by BOTH the numpy driver (``corpus_predictive_gain_gated``, which
    calls this once per ``enumerate(docs)``) and the RDD ``_local`` worker
    function (once per ``doc_rdd.zipWithIndex()`` row) so the per-document
    logic cannot drift between the two twins -- ``idx`` is exactly
    ``seed + idx`` fed to ``doc_predictive_gain``/``null_delta`` as their
    per-doc seed, and MUST be the same index a document gets under
    ``enumerate(docs)`` in the numpy path for numpy<->RDD parity (see
    ``corpus_heldout_scale_sweep_gated_rdd``'s docstring for why
    ``zipWithIndex`` on an un-sampled, order-preserving RDD reproduces this).

    Degenerate-split documents (``doc_predictive_gain`` returns None) are
    skipped entirely -- not counted anywhere, not even ``n_docs``.
    """
    dg = doc_predictive_gain(
        doc, global_params, partition, c=c, reference=reference,
        holdout_frac=holdout_frac, seed=seed + idx,
        marginal=marginal, smoothing_lambda=smoothing_lambda,
        fast=fast, high_mass_bound=high_mass_bound, max_fast_steps=max_fast_steps,
    )
    if dg is None:
        return acc

    nulls = null_delta(
        doc, global_params, partition, c=c, reference=reference,
        holdout_frac=holdout_frac, seed=seed + idx,
        n_perm=n_perm, rng_seed=seed + idx,
        marginal=marginal, smoothing_lambda=smoothing_lambda,
    )
    if nulls is not None:
        finite = nulls[np.isfinite(nulls)]
    else:
        finite = np.empty(0, dtype=np.float64)
    null_ok = finite.size > 0
    thr = float(np.max(finite)) if null_ok else None

    L = int(doc.counts.sum())
    sum_j = float(np.sum(dg.delta))

    for p in range(dg.allowed.shape[0]):
        k = int(dg.allowed[p])
        d_k = float(dg.delta[p])

        acc.sum_gain[k] += d_k
        acc.depth_den[k] += sum_j
        acc.dedup_sum[k] += float(dg.dedup_delta[p])
        acc.count_k[k] += 1

        acc.Slen[k] += L
        acc.SdL[k] += d_k * L
        acc.SLL[k] += L * L
        acc.Sdd[k] += d_k * d_k

        b = _clamped_bin(d_k, bin_edges, n_bins)
        acc.prominence_hist[k, b] += 1

        if null_ok and d_k > thr:
            acc.presence_count[k] += 1

        # Primary presence (smoothed metric): Delta_k > 0 == the doc's
        # held-out tokens are better explained WITH topic k than by the
        # background baseline alone. With the marginal back-off, 0 is a
        # meaningful origin (an irrelevant/permuted topic scores ~0).
        if d_k > 0.0:
            acc.presence_count_pos[k] += 1

        if d_k < acc.obs_delta_min:
            acc.obs_delta_min = d_k
        if d_k > acc.obs_delta_max:
            acc.obs_delta_max = d_k

    for v in finite:
        vf = float(v)
        acc.null_sum += vf
        acc.null_sqsum += vf * vf
        acc.null_count += 1
        b = _clamped_bin(vf, bin_edges, n_bins)
        acc.null_hist[b] += 1

    acc.n_docs += 1
    return acc


def _hist_percentile(hist: np.ndarray, edges: np.ndarray, q: float) -> float:
    """Linear-interpolated ``q``-percentile (0 < q < 1) from a binned
    histogram (``hist`` length n_bins, ``edges`` length n_bins+1): locate the
    bin containing the ``q``-th quantile via cumulative counts, then
    linearly interpolate within that bin's width under a uniform-within-bin
    density assumption. Returns NaN if the histogram is empty (total count
    0)."""
    hist = np.asarray(hist, dtype=np.float64)
    total = float(hist.sum())
    if total <= 0:
        return float("nan")
    target = q * total
    cum = np.cumsum(hist)
    bin_idx = int(np.searchsorted(cum, target, side="left"))
    bin_idx = min(bin_idx, hist.shape[0] - 1)
    prev_cum = float(cum[bin_idx - 1]) if bin_idx > 0 else 0.0
    bin_count = float(hist[bin_idx])
    lo, hi = float(edges[bin_idx]), float(edges[bin_idx + 1])
    if bin_count <= 0:
        return lo
    frac = min(max((target - prev_cum) / bin_count, 0.0), 1.0)
    return lo + frac * (hi - lo)


def _finalize_pred_gain(acc: _PredGainAcc, bin_edges: np.ndarray) -> dict:
    """Shared driver-side finalize for both ``corpus_predictive_gain_gated``
    and ``corpus_predictive_gain_gated_rdd``: turns the reduced
    ``_PredGainAcc`` totals into the returned per-topic arrays + null_band
    summary. See those functions' docstrings for field semantics."""
    count_k = acc.count_k
    denom = np.maximum(count_k, 1)

    mean_gain = np.where(count_k > 0, acc.sum_gain / denom, np.nan)

    depth_num = acc.sum_gain.copy()
    depth_den = acc.depth_den
    # depth is a SHARE of total predictive structure, only interpretable
    # when that total (depth_den = Sigma_d Sigma_j Delta_j) is positive;
    # Delta can be negative, so depth_den can be <= 0, which would yield a
    # negative or exploded "share" rather than a real fraction -- guard
    # with `> 0` (not `!= 0`) so depth[k] is nan wherever depth_den[k] <= 0.
    depth = np.where(
        depth_den > 0, depth_num / np.where(depth_den > 0, depth_den, 1.0), np.nan,
    )

    # PRIMARY presence = fraction of a topic's docs with Delta_k > 0 (beats
    # the background baseline) -- the operating definition under the smoothed
    # score. presence_vs_null keeps the paired permuted-null fraction as a
    # VALIDATION diagnostic: once the null collapses to ~0 (see null_band) the
    # two should agree on real data.
    presence = np.where(count_k > 0, acc.presence_count_pos / denom, np.nan)
    presence_vs_null = np.where(count_k > 0, acc.presence_count / denom, np.nan)

    # Streaming Pearson correlation of per-doc Delta_k with document length L,
    # from sufficient statistics (never a per-doc ratio): guard n<2 and
    # zero-variance (den<=0) to NaN.
    n_f = count_k.astype(np.float64)
    num = n_f * acc.SdL - acc.Slen * acc.sum_gain
    var_len = n_f * acc.SLL - acc.Slen ** 2
    var_d = n_f * acc.Sdd - acc.sum_gain ** 2
    den_sq = var_len * var_d
    den = np.sqrt(np.maximum(den_sq, 0.0))
    valid = (count_k >= 2) & (den > 0)
    length_corr = np.where(valid, num / np.where(den == 0, 1.0, den), np.nan)

    dedup_mean_gain = np.where(count_k > 0, acc.dedup_sum / denom, np.nan)

    if acc.null_count > 0:
        null_mean = acc.null_sum / acc.null_count
        null_var = max(0.0, acc.null_sqsum / acc.null_count - null_mean ** 2)
        null_std = float(np.sqrt(null_var))
        p95 = _hist_percentile(acc.null_hist, bin_edges, 0.95)
    else:
        null_mean = float("nan")
        null_std = float("nan")
        p95 = float("nan")

    null_band = {
        "mean": float(null_mean),
        "std": float(null_std),
        "n": int(acc.null_count),
        "hist": acc.null_hist.tolist(),
        "p95": float(p95),
    }

    return {
        "mean_gain": mean_gain,
        "depth": depth,
        "depth_num": depth_num,
        "depth_den": depth_den,
        "presence": presence,
        "presence_vs_null": presence_vs_null,
        "prominence_hist": acc.prominence_hist,
        "prominence_bin_edges": bin_edges,
        "length_corr": length_corr,
        "dedup_mean_gain": dedup_mean_gain,
        "null_band": null_band,
        "count_k": count_k,
        "n_docs": acc.n_docs,
        "observed_delta_range": (float(acc.obs_delta_min), float(acc.obs_delta_max)),
    }


def corpus_predictive_gain_gated(
    docs, global_params, partition, *, c, reference=None,
    holdout_frac=0.5, seed=0, n_perm=4, n_bins=50, prominence_range=(-1.0, 10.0),
    marginal=None, smoothing_lambda=1.0,
    fast=False, high_mass_bound=0.2, max_fast_steps=3,
) -> dict:
    """Driver-side (in-memory) corpus aggregation of the per-document COLD
    predictive gain (``doc_predictive_gain``) and permuted null
    (``null_delta``) into per-topic summary arrays, for a gated STM.

    For each document (``enumerate(docs)``, per-doc seed ``seed + i``): runs
    ``doc_predictive_gain`` to get Delta_k for every topic k in the doc's
    allowed set (skipping the doc entirely -- not even counted in
    ``n_docs`` -- if the held-out split is degenerate), and ``null_delta``
    (SAME ``seed + i``, so it scores against the IDENTICAL held-out split)
    to get that doc's own permuted-null sample band. See ``tests/
    test_predictive_gain.py``'s ``TestCorpusPredictiveGainGated`` for a
    hand-checkable gated-corpus fixture.

    Returned per-topic (length K) arrays:
      mean_gain        Sigma_d Delta_k / count_k  -- average held-out gain
                        topic k contributes, over documents that allow it.
      depth, depth_num, depth_den
                        depth = depth_num/depth_den = (Sigma_d Delta_k) /
                        (Sigma_d Sigma_j Delta_j), i.e. topic k's SHARE of
                        the total held-out predictive structure attributed
                        across all of k's documents' allowed topics. This is
                        a ratio of SUMS (never an average of per-document
                        ratios Delta_k/sum_j) -- summing first pools evidence
                        across documents before dividing, so one document
                        with an unusually small sum_j cannot blow up its
                        depth contribution the way a per-doc-then-averaged
                        ratio would. depth_num/depth_den are also returned
                        directly so a caller (or test) can verify the
                        division was formed this way. depth is defined only
                        where depth_den > 0 (the doc-pooled total predictive
                        structure is positive); Delta can be negative, so
                        depth_den can be zero or negative, and depth[k] is
                        nan there rather than a negative or exploded share.
      presence          presence[k] = presence_count[k] / count_k[k]: the
                        fraction of topic k's documents (the FULL count_k,
                        not a null-evaluable subset) whose Delta_k beats
                        THAT SAME document's own permuted-null maximum -- a
                        PER-DOCUMENT PAIRED permutation test (see the module-
                        level "DESIGN DECISIONS" comment above this
                        function for the full derivation and why this,
                        rather than testing against the pooled corpus null
                        band, is correct). Documents with no finite null
                        sample (e.g. a single-allowed-topic doc where that
                        topic is the pinned reference) remain in the
                        count_k denominator but can NEVER enter
                        presence_count -- so a null-unevaluable document
                        effectively counts AGAINST presence, as if topic k
                        were judged "not present" in it.
      prominence_hist   (K, n_bins) histogram of per-doc Delta_k over
                        ``prominence_range`` -- the aggregate Delta
                        distribution, replacing a theta-hat histogram for
                        this predictive-gain view.
      prominence_bin_edges
                        length n_bins+1 edges shared by every topic's
                        histogram AND the pooled null_hist.
      length_corr       per-topic Pearson correlation of per-doc Delta_k
                        with document token length, from streaming
                        sufficient statistics (never a per-doc ratio); NaN
                        if fewer than 2 documents or zero variance.
      dedup_mean_gain   like mean_gain but using dg.dedup_delta (held-out
                        counts capped at 1 -- see ``doc_predictive_gain``).
      null_band         POOLED corpus null summary (mean, std, n, hist,
                        p95) over every document's finite null samples --
                        descriptive only (see decision #1 above); NOT what
                        ``presence`` is tested against.
      count_k           number of documents that allow topic k (the
                        WITHIN-GROUP denominator: a foreground topic k of
                        group g only appears in ``allowed`` for group-g
                        documents, so count_k[k] is automatically restricted
                        to group g -- no separate cross-group masking is
                        applied or needed anywhere in this module).
      n_docs            number of documents that actually contributed (docs
                        skipped by a degenerate held-out split do not
                        count).
      observed_delta_range
                        (min, max) Delta_k actually observed across the
                        whole corpus -- returned because
                        ``prominence_range`` is PROVISIONAL (Delta's natural
                        nats scale is not yet calibrated against a real
                        fitted corpus; see decision #2 above): a Phase-2
                        caller recalibrates the histogram edges from these
                        real numbers.

    ``docs`` is an in-memory list of STMDocument; ``global_params`` is the
    dict with "lambda" (K,V), "Gamma" (P,K), "Sigma" (K,K, the fit's
    correlation R); ``partition`` is a TopicBlockPartition (or the implicit
    all-background one). For a live cluster corpus use
    ``corpus_predictive_gain_gated_rdd``. Raises ValueError if no document
    contributes (``n_docs == 0``).
    """
    K = partition.K
    bin_edges = np.linspace(prominence_range[0], prominence_range[1], n_bins + 1)
    acc = _PredGainAcc.zeros(K, n_bins)

    for i, doc in enumerate(docs):
        _accumulate_doc(
            acc, doc, i, global_params, partition, c=c, reference=reference,
            holdout_frac=holdout_frac, seed=seed, n_perm=n_perm,
            bin_edges=bin_edges, n_bins=n_bins,
            marginal=marginal, smoothing_lambda=smoothing_lambda,
            fast=fast, high_mass_bound=high_mass_bound, max_fast_steps=max_fast_steps,
        )

    if acc.n_docs == 0:
        raise ValueError("corpus_predictive_gain_gated: empty document corpus")

    return _finalize_pred_gain(acc, bin_edges)


def corpus_predictive_gain_gated_rdd(
    doc_rdd, global_params, partition, *, c, reference=None,
    holdout_frac=0.5, seed=0, sample_cap=200_000, n_perm=4, n_bins=50,
    prominence_range=(-1.0, 10.0), depth=2,
    marginal=None, smoothing_lambda=1.0,
    fast=False, high_mass_bound=0.2, max_fast_steps=3,
) -> dict:
    """Distributed corpus aggregation of the per-document COLD predictive
    gain and permuted null into per-topic summary arrays, for a gated STM
    (see ``corpus_predictive_gain_gated`` for the full field-by-field
    derivation; this is its Spark counterpart, byte-for-byte the same math
    on a distributed corpus when ``sample_cap=None`` -- see the numpy<->RDD
    parity test in ``tests/test_predictive_gain.py``).

    ``doc_rdd.zipWithIndex()`` gives each document the SAME index it would
    have under ``enumerate(docs)`` in the numpy path (zipWithIndex preserves
    the RDD's element order, and ``sc.parallelize(docs, n)`` preserves the
    input list's order across partitions), so ``doc_predictive_gain``'s and
    ``null_delta``'s per-doc seed (``seed + index``) reproduce the identical
    held-out splits as the numpy oracle -- required for parity, exactly the
    convention ``corpus_heldout_scale_sweep_gated_rdd`` uses.

    Distributed via the same ``mapPartitions(_local).treeReduce(_combine)``
    idiom as ``corpus_eta_variance_gated_rdd`` /
    ``corpus_heldout_scale_sweep_gated_rdd``: each partition folds its rows
    into a local ``_PredGainAcc`` (via ``_accumulate_doc``, the SAME
    per-document logic the numpy path uses); the tree-reduce combines
    partitions pairwise via ``_combine_pred_gain_acc`` (a pure elementwise
    sum/min/max over small length-K / (K, n_bins) arrays -- never the
    documents or per-doc DocGain/null samples cross the network).
    ``global_params`` and ``partition`` are broadcast via the Spark-safe
    default-arg closure convention; ``doc_predictive_gain``/``null_delta``/
    ``_accumulate_doc``/``_PredGainAcc`` live in this SAME module, so
    ``_local`` references them directly (no in-function import needed for
    them -- only for helpers imported from OTHER modules, which
    ``doc_predictive_gain``/``null_delta`` already do at this module's own
    top level).

    ``sample_cap`` bounds the number of documents actually processed (this
    function is O(|allowed|) full L-BFGS solves per document, so unlike a
    cheap collect-and-histogram pass, the SWEEP OVER DOCUMENTS itself is the
    expensive part worth capping): if ``sample_cap`` is not None and the
    corpus has more than ``sample_cap`` documents, it is Bernoulli-sampled
    (``RDD.sample(False, frac)``) down to approximately ``sample_cap``
    documents BEFORE ``zipWithIndex``/processing; N (pre-sample count),
    the target N' (~= sample_cap), and frac are logged (project rule: no
    silent cap). ``sample_cap=None`` disables sampling entirely -- required
    for numpy<->RDD parity, since sampling breaks the doc-for-doc
    correspondence the seed-parity argument above depends on.

    Raises ValueError if the (possibly sampled) document count that actually
    contributes is 0.
    """
    if sample_cap is not None:
        n_total = doc_rdd.count()
        if n_total > sample_cap:
            frac = float(sample_cap) / float(n_total)
            log.info(
                "corpus_predictive_gain_gated_rdd: sampling N=%d -> N'~=%d "
                "(frac=%.4f, cap=%d)", n_total, sample_cap, frac, sample_cap,
            )
            doc_rdd = doc_rdd.sample(False, frac, seed)
        else:
            log.info(
                "corpus_predictive_gain_gated_rdd: N=%d <= sample_cap=%d, "
                "no sampling", n_total, sample_cap,
            )

    sc = doc_rdd.context
    gp_bcast = sc.broadcast({
        k: np.asarray(v, dtype=np.float64) for k, v in global_params.items()
    })
    p_bcast = sc.broadcast(partition)
    # marginal is a length-V probability vector; broadcast it like global_params
    # (None stays None -> the unsmoothed backward-compat path on the workers).
    marg_bcast = sc.broadcast(
        None if marginal is None else np.asarray(marginal, dtype=np.float64)
    )
    K = partition.K
    bin_edges = np.linspace(prominence_range[0], prominence_range[1], n_bins + 1)

    def _local(rows, _gp=gp_bcast, _p=p_bcast, _marg=marg_bcast):
        gp = _gp.value
        part = _p.value
        marg = _marg.value
        acc = _PredGainAcc.zeros(K, n_bins)
        for doc, doc_idx in rows:
            _accumulate_doc(
                acc, doc, doc_idx, gp, part, c=c, reference=reference,
                holdout_frac=holdout_frac, seed=seed, n_perm=n_perm,
                bin_edges=bin_edges, n_bins=n_bins,
                marginal=marg, smoothing_lambda=smoothing_lambda,
                fast=fast, high_mass_bound=high_mass_bound,
                max_fast_steps=max_fast_steps,
            )
        return [acc]

    acc = (
        doc_rdd.zipWithIndex().mapPartitions(_local)
        .treeReduce(_combine_pred_gain_acc, depth=depth)
    )
    if acc.n_docs == 0:
        raise ValueError("corpus_predictive_gain_gated_rdd: empty document RDD")

    return _finalize_pred_gain(acc, bin_edges)


def predictive_gain_downdate_audit(
    docs, global_params, partition, *, c, reference=None,
    holdout_frac=0.5, seed=0, marginal=None, smoothing_lambda=1.0,
    high_mass_bound=0.2, max_fast_steps=3,
) -> dict:
    """Real-data cold-vs-fast discrepancy audit for the downdate approximation.

    This is the real-data cold-solve check the fast path's validation basis
    rests on (design doc §7): the synthetic agreement tests show the warm-start
    Newton downdate (``fast=True``) matches the COLD oracle (``fast=False``) on
    hand-built fixtures, but the guarantee that it still holds on a REAL fitted
    corpus can only come from cold-solving actual documents and comparing. This
    function does exactly that on a small in-memory sample and returns a
    per-topic AGGREGATE of the discrepancy -- it is called during the export
    phase and LOGGED. It ships only |Delta_cold - Delta_fast| summaries, never
    any per-document delta, so it is ToS-safe (no row-level quantities leave).

    ``docs`` is a small in-memory list of STMDocument (the export layer samples
    the corpus RDD and collects a handful to the driver). For each document it
    runs BOTH ``doc_predictive_gain(..., fast=False)`` (the cold oracle) and
    ``doc_predictive_gain(..., fast=True, ...)`` against the SAME held-out split
    (same per-doc seed ``seed + i``), so both produce the identical ``allowed``
    set and their ``delta`` vectors align position-for-position. It accumulates
    the per-topic absolute discrepancy across documents; a document skipped by
    either path (degenerate split -> None) contributes nothing.

    Returns a dict:
      max_abs_discrepancy   length-K: per-topic max over audited docs of
                            |Delta_cold[k] - Delta_fast[k]| (nan for topics no
                            audited doc allowed).
      mean_abs_discrepancy  length-K: per-topic mean of the same (nan for
                            unaudited topics).
      max_abs_overall       float: the single largest discrepancy over all
                            (doc, topic) pairs -- the headline number to log
                            and threshold on.
      n_docs_audited        int: number of documents both paths scored.
    """
    K = partition.K
    max_abs = np.zeros(K, dtype=np.float64)
    sum_abs = np.zeros(K, dtype=np.float64)
    count = np.zeros(K, dtype=np.int64)
    max_abs_overall = 0.0
    n_docs_audited = 0

    for i, doc in enumerate(docs):
        dg_cold = doc_predictive_gain(
            doc, global_params, partition, c=c, reference=reference,
            holdout_frac=holdout_frac, seed=seed + i,
            marginal=marginal, smoothing_lambda=smoothing_lambda, fast=False,
        )
        dg_fast = doc_predictive_gain(
            doc, global_params, partition, c=c, reference=reference,
            holdout_frac=holdout_frac, seed=seed + i,
            marginal=marginal, smoothing_lambda=smoothing_lambda, fast=True,
            high_mass_bound=high_mass_bound, max_fast_steps=max_fast_steps,
        )
        if dg_cold is None or dg_fast is None:
            continue

        # Same split+seed => identical allowed sets => positions align.
        for p in range(dg_cold.allowed.shape[0]):
            k = int(dg_cold.allowed[p])
            disc = abs(float(dg_cold.delta[p]) - float(dg_fast.delta[p]))
            if disc > max_abs[k]:
                max_abs[k] = disc
            sum_abs[k] += disc
            count[k] += 1
            if disc > max_abs_overall:
                max_abs_overall = disc
        n_docs_audited += 1

    seen = count > 0
    max_abs_discrepancy = np.where(seen, max_abs, np.nan)
    mean_abs_discrepancy = np.where(seen, sum_abs / np.maximum(count, 1), np.nan)

    return {
        "max_abs_discrepancy": max_abs_discrepancy,
        "mean_abs_discrepancy": mean_abs_discrepancy,
        "max_abs_overall": float(max_abs_overall),
        "n_docs_audited": int(n_docs_audited),
    }
