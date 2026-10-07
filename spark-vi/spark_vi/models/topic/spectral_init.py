"""Block-aware spectral (anchor-word) initialization for OnlineSTM.

OnlineSTM under random-gamma initialization is unstable: depending on
sigma_init it either collapses topics or lets Σ blow up to ~1e10 (insight
0029). The cure is a deterministic, data-driven β seed via the anchor-word
algorithm (Arora, Ge, Halpern, Mimno, Moitra, Sontag, Wu, Zhu 2013, "A
Practical Algorithm for Topic Modeling with Provable Guarantees", ICML).

The classic anchor-word recipe, on the word-by-word same-document
co-occurrence matrix Q:
  1. find K "anchor" words — words that (nearly) occur in a single topic, so
     their Q rows span the convex hull of all word rows;
  2. express every word's Q row as a non-negative convex combination of the
     anchor rows → P(topic | word);
  3. Bayes-flip with the word marginal → P(word | topic) = β.

This module adds a *block-aware* twist for gated STM (TopicBlockPartition).
The background block is recovered globally on the pooled Q. Each group's
foreground anchors are then found on that group's *within-group* Q — where a
rare group's phenotype is undiluted by the majority — while *deflating*
against the already-chosen background anchors (passed as ``seed_rows`` so the
greedy search spans away from them but never returns them). This is the seam
that lets a minority arm's foreground topic land its planted phenotype at
init, before any EM. Non-gated fitting routes through an all-background
partition (background_k = K, no groups), so step 1 alone reproduces a global
single-pass anchor-word init — one code path, no special case.

Domain-agnostic: integer token ids only, no OMOP/EHR vocabulary.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from spark_vi.models.topic.domains import validate_domain_bounds


def word_cooccurrence(docs, V: int) -> np.ndarray:
    """V×V normalized same-document word co-occurrence matrix Q.

    For each document with unique token ids and counts ``n`` over length
    ``L = Σ n``, the empirical probability that two *distinct* token draws (one
    drawn, then a second drawn without replacement) land on words i and j is
    ``(outer(n, n) − diag(n)) / (L · (L − 1))``: outer(n, n) counts ordered
    pairs with replacement, subtracting diag(n) removes the i==i self-pairs
    that "without replacement" forbids, and L·(L−1) is the number of ordered
    distinct pairs. Averaging this per-doc estimator over all documents and
    renormalizing to sum 1 gives Q, the joint P(word_1 = i, word_2 = j).

    Single-token docs (L < 2) carry no co-occurrence and contribute nothing —
    correct, not a special case. Returns a dense (V, V) float64 array summing
    to 1 (0 only in the degenerate corpus with no multi-token document).
    """
    Q = np.zeros((V, V), dtype=np.float64)
    n_docs = 0
    for d in docs:
        L = float(d.counts.sum())
        if L < 2:
            continue
        idx = np.asarray(d.indices, dtype=np.int64)
        n = np.asarray(d.counts, dtype=np.float64)
        block = (np.outer(n, n) - np.diag(n)) / (L * (L - 1.0))
        Q[np.ix_(idx, idx)] += block
        n_docs += 1
    if n_docs:
        Q /= n_docs
    total = Q.sum()
    if total > 0:
        Q /= total
    return Q


def _row_normalize(Q: np.ndarray) -> np.ndarray:
    """Row-stochastic view of Q: row i ∝ Q[i] (zero rows stay zero).

    The anchor geometry lives on the conditional rows Q̄_i = P(word_2 | word_1 = i),
    which is what makes anchor rows the vertices of the convex hull the other
    rows live inside.
    """
    rs = Q.sum(axis=1, keepdims=True)
    rs_safe = np.where(rs > 0, rs, 1.0)
    return Q / rs_safe


def _domain_candidate_mask(marginal, min_marginal_frac, domain_bounds):
    """Boolean 'eligible to be an anchor' mask, floored WITHIN each domain.

    The candidate floor (find_anchors docstring) keeps sub-promille noise words
    from being picked as spurious hull vertices. On a multi-domain joint Q the
    pooled mean is dominated by the densest domain, so a sparser domain's real
    anchors fall below the pooled bar and no anchor ever comes from it (spec:
    'domain imbalance is the most likely thing to silently break init'). The fix
    is to compare each word only to the mean nonzero marginal of ITS OWN domain.

    domain_bounds is a strictly-increasing cumulative-offset sequence starting at
    0 and ending at V; domain d spans [domain_bounds[d], domain_bounds[d+1]).
    None -> a single pooled domain, reproducing the original pooled floor exactly.
    It is VALIDATED here (`domains.validate_domain_bounds`), which is what makes
    every `find_anchors` caller -- including `spectral_init_beta` and
    `gated_init.spectral_block_aligned_lambda` -- raise on malformed bounds
    instead of silently leaving the uncovered tail of the vocabulary ineligible
    to anchor.
    """
    V = marginal.shape[0]
    domain_bounds = validate_domain_bounds(domain_bounds, V)
    mask = np.zeros(V, dtype=bool)
    for lo, hi in zip(domain_bounds[:-1], domain_bounds[1:]):
        seg = marginal[lo:hi]
        pos = seg > 0
        thr = min_marginal_frac * seg[pos].mean() if pos.any() else 0.0
        mask[lo:hi] = seg >= thr
    return mask


def find_anchors(Q: np.ndarray, n: int, *, seed_rows=None,
                 min_marginal_frac: float = 1.0,
                 domain_bounds=None, preferred=None) -> list[int]:
    """Greedy farthest-point anchor selection on the row-normalized rows of Q.

    Gram–Schmidt "pivoted QR" geometry (Arora et al. 2013, Algorithm 4): build
    an orthonormal basis of the span of already-chosen rows; the next anchor is
    the word whose row has the largest residual norm after projecting out that
    span. This finds, greedily, the rows that are most extreme / most nearly
    pure — the convex-hull vertices.

    Candidate restriction (anchor-word fragility cure): a rare/noise word that
    happened to co-occur in only one or two documents has a degenerate, near-
    pure Q row and would be picked as a spurious "vertex" ahead of a genuine
    topic word whose row is a true mixture. Every practical anchor-word
    implementation restricts anchor *candidates* to words with enough document
    mass. Here a word may anchor only if its Q marginal (row sum) is at least
    ``min_marginal_frac`` times the mean nonzero marginal. The default of 1.0
    (at least average frequency) cleanly separates real phenotype words
    (above-average co-occurrence mass) from sub-promille noise words and is what
    makes a minority group's foreground phenotype, not corpus noise, surface as
    its anchor. Words below the bar still get β rows recovered against the
    chosen anchors; they just cannot *be* anchors.

    ``seed_rows`` (optional word ids) pre-seeds the spanned basis with those
    rows WITHOUT returning them. This is the deflation seam: when finding a
    group's foreground anchors we seed with the background anchors so the search
    spans *away* from the shared background and toward the group-specific
    phenotype rows. Seeds that are (near) linearly dependent on the existing
    basis are skipped, never erroring.

    ``domain_bounds`` (optional cumulative column-offset sequence, e.g. a
    concatenated multi-domain vocab ``[domain_0 ; domain_1 ; ...]``) floors the
    candidate marginal WITHIN each domain instead of over the pooled Q. On a
    joint Q built from domains of very different density (spec risk: domain
    imbalance is the most likely thing to silently break init), the pooled mean
    is dominated by the densest domain and a sparser domain's real anchors never
    clear the bar — no anchor is ever drawn from it. Flooring per-domain fixes
    this while leaving the greedy hull-vertex geometry itself untouched. Default
    ``None`` reproduces the single pooled-domain floor exactly (see
    ``_domain_candidate_mask``). Bounds that do not cover [0, V) exactly once
    raise ValueError (``domains.validate_domain_bounds``): the previous silent
    behavior returned FEWER anchors than asked for, with every column past the
    last bound permanently ineligible.

    ``preferred`` (optional word ids) is the GUIDED-ANCHOR seam (spec
    2026-10-05, HPO-guided spectral anchors): the same greedy search runs in two
    stages over ONE shared basis — stage 1 may pick only from ``preferred`` (∩
    the candidate floor), stage 2 fills whatever stage 1 could not from the full
    candidate pool. Why a set and not a weight: the point is to let an outside
    source of meaning (a node's HPO profile) decide WHICH words may anchor its
    block, while the co-occurrence geometry still decides which of those are the
    vertices and the recovery still decides what the topic IS — a set changes
    the pool, never the criterion, so there is no strength to tune. ``None`` or
    an empty set reproduces the single-stage search exactly (stage 2 alone IS
    today's loop).

    Returns the ``n`` newly chosen anchor word ids in selection order.
    """
    Qbar = _row_normalize(Q)
    V = Qbar.shape[0]
    norms = (Qbar * Qbar).sum(axis=1)            # squared row norms

    marginal = Q.sum(axis=1)
    candidate = _domain_candidate_mask(marginal, min_marginal_frac, domain_bounds)

    pref = _preferred_mask(preferred, V)
    stages = ([candidate & pref] if pref is not None else []) + [candidate]
    return greedy_anchors(Qbar, norms, stages, n, seed_rows=seed_rows)


_GREEDY_EPS = 1e-12


def greedy_anchors(Qbar, norms, stages, n, *, seed_rows=None, eps=_GREEDY_EPS):
    """Greedy pivoted-QR anchor selection over the rows of ``Qbar`` — the one
    search both the dense (`find_anchors`) and the projected
    (`spectral_init_scalable.find_anchors_projected`) paths run.

    Geometry (Arora et al. 2013, Alg. 4): keep an orthonormal basis of the
    chosen rows' residuals; the next anchor is the eligible row with the
    largest residual norm after projecting out that span. ``stages`` is a list
    of boolean (V,) pools searched in order over ONE shared basis (the guided
    two-stage search: a preferred pool first, then the open pool); ``n`` is the
    total to return; ``seed_rows`` pre-span the basis (deflation) without being
    returned; a row with ``norms <= eps`` never anchors.

    Why this form. The earlier form re-projected every candidate row against
    the whole basis at every step — O(n^2 V d) Python-level dots — which is
    free for eight background anchors and does not finish in a day for the
    1,200-topic shared background of exp 0134. This form never touches the
    (V, d) matrix beyond one GEMV per basis vector: the basis ``B`` (d, k) is
    the state, a row's residual is formed only when it is chosen
    (``r = q - B (B^T q)``, done twice for re-orthogonalization), and every
    row's squared residual norm is kept incrementally as
    ``res_i -= (q_i . b)^2`` — exact in exact arithmetic because the basis is
    orthonormal, so the argmax is the same one the reprojection form takes up
    to floating-point ties. Seeds are spanned in ONE batch (an SVD of the seed
    rows: the residual against a span does not depend on which basis spans it)
    instead of one rank-1 update each, so a node seeded with 1,200 background
    anchors pays one GEMM, not 1,200 passes.

    Returns the anchor ids in selection order (fewer than ``n`` only when every
    pool is exhausted of distinct directions).
    """
    Qbar = np.asarray(Qbar, dtype=np.float64)
    V, d = Qbar.shape
    norms = np.asarray(norms, dtype=np.float64)
    res = norms.copy()                              # squared residual norms
    chosen = np.zeros(V, dtype=bool)
    seeds = np.asarray([int(x) for x in (seed_rows or [])], dtype=np.int64)
    cap = min(d, int(n) + int(seeds.size)) + 1
    B = np.zeros((d, cap), dtype=np.float64)
    k = 0

    def _proj_sub(v):
        # residual of one row against the current basis, re-orthogonalized once
        if k == 0:
            return v.copy()
        Bk = B[:, :k]
        r = v - Bk @ (Bk.T @ v)
        return r - Bk @ (Bk.T @ r)

    def _append(bvec):
        nonlocal k
        if k >= B.shape[1]:
            return
        B[:, k] = bvec
        k += 1
        proj = Qbar @ bvec
        res[:] -= proj * proj
        np.maximum(res, 0.0, out=res)

    if seeds.size:
        chosen[seeds] = True
        S = Qbar[seeds]                             # (k_seed, d)
        if S.shape[0] and np.any(S):
            # span the seeds in one batch: right-singular vectors with a
            # non-negligible singular value are an orthonormal basis of the span
            _, sv, vt = np.linalg.svd(S, full_matrices=False)
            keep = sv > eps * max(1.0, float(sv[0]))
            for bvec in vt[keep]:
                _append(bvec)

    anchors: list[int] = []
    eligible = norms > eps
    for pool in stages:
        pool = np.asarray(pool, dtype=bool)
        while len(anchors) < n:
            mask = pool & eligible & ~chosen
            if not mask.any():                      # pool exhausted
                break
            best = int(np.argmax(np.where(mask, res, -np.inf)))
            anchors.append(best)
            chosen[best] = True
            r = _proj_sub(Qbar[best])
            nrm = float(np.sqrt(r @ r))
            if nrm > eps:
                _append(r / nrm)
            else:
                res[best] = 0.0
    return anchors


def _preferred_mask(preferred, V: int):
    """Boolean (V,) mask of the preferred ids, or None when there are none.
    Ids outside [0, V) are ignored (a vocab index from a different bundle is
    a caller bug the anchor search should not crash on; the builder counts
    them)."""
    if preferred is None:
        return None
    ids = np.asarray([int(i) for i in preferred], dtype=np.int64)
    ids = ids[(ids >= 0) & (ids < V)]
    if ids.size == 0:
        return None
    mask = np.zeros(V, dtype=bool)
    mask[ids] = True
    return mask


def recover_beta(Q: np.ndarray, anchors, rows=None) -> np.ndarray:
    """Recover an ``len(anchors)×V`` β (P(word|topic)) from anchors via NNLS.

    For each word w, solve a non-negative least squares fit of its row-normalized
    Q row onto the anchor rows: ``min_{c >= 0} || A^T c − Q̄_w ||`` where A is the
    (n_anchors, V) matrix of anchor rows. Normalizing c to sum 1 gives
    P(topic | word = w) — the convex weights placing w inside the anchor
    simplex. Bayes-flip with the word marginal p_w = Q.sum(axis=1):
        P(word=w | topic=k) ∝ P(topic=k | word=w) · p_w
    then renormalize each topic row to a probability distribution over words.

    ``rows`` optionally restricts which word rows are fit (the rest get zero
    weight); used so a foreground recovery only sees the group's own vocabulary
    support. Default: all V words.
    """
    Qbar = _row_normalize(Q)
    p_w = Q.sum(axis=1)                            # word marginal
    V = Qbar.shape[0]
    n_topics = len(anchors)
    A = Qbar[list(anchors)]                        # (n_topics, V)
    A_T = A.T                                      # (V, n_topics): solve A_T c = Q̄_w

    if rows is None:
        rows = range(V)
    rows = list(rows)

    # P(topic | word): one convex-weight vector per fitted word.
    topic_given_word = np.zeros((V, n_topics), dtype=np.float64)
    for w in rows:
        c, _ = nnls(A_T, Qbar[w])
        s = c.sum()
        if s > 0:
            topic_given_word[w] = c / s
        # else: word has no co-occurrence support -> left at zero, contributes
        # nothing to any topic (it carries no signal).

    # Bayes flip + renormalize rows -> P(word | topic).
    beta = (topic_given_word * p_w[:, None]).T     # (n_topics, V)
    row_sums = beta.sum(axis=1, keepdims=True)
    # An anchor whose column collapsed to all-zero weights (degenerate corpus)
    # falls back to a uniform row so β stays a valid stochastic matrix.
    zero = (row_sums[:, 0] <= 0)
    if zero.any():
        beta[zero] = 1.0 / V
        row_sums[zero, 0] = 1.0
    beta = beta / row_sums
    return beta


def spectral_init_beta(docs, partition, V: int, *, domain_bounds=None) -> np.ndarray:
    """Block-aware K×V β seed in ``partition`` slot order.

    Step 1 (background): pooled Q over all docs → ``background_k`` anchors →
    recover those rows into ``partition.background_indices()``.

    Step 2 (per group): for each group g, restrict to that group's docs, build
    the within-group Q_g, find ``len(block_indices(g))`` foreground anchors with
    the background anchors passed as ``seed_rows`` (deflation), and recover those
    rows from Q_g into ``partition.block_indices(g)``.

    Non-gated partitions (background_k = K, no foreground groups) execute step 1
    only and produce exactly what a global single-pass anchor-word init would —
    the degenerate, identical case.

    ``domain_bounds`` (optional; spec's multi-domain joint-Q construction) is
    passed straight through to EVERY ``find_anchors`` call this makes — the
    background pooled-Q call in step 1 and each group's within-group Q_g call in
    step 2 — so a sparser domain's anchors clear the per-domain candidate floor
    instead of being swamped by the pooled mean (see ``find_anchors`` /
    ``_domain_candidate_mask``). ``word_cooccurrence`` already builds the joint Q
    over the concatenated multi-domain vocab unchanged; only this floor threading
    is needed. Default ``None`` reproduces current single-pooled-domain behavior
    byte-for-byte. Validated up front (`domains.validate_domain_bounds`) so a
    malformed sequence fails before the O(V^2) co-occurrence pass rather than
    inside the first ``find_anchors`` call.

    No in-repo caller passes ``domain_bounds`` here yet: this is the BLOCK-AWARE
    STM entry point, while the multi-domain gated arc seeds through
    ``gated_init.multidomain_spectral_lambda``. The passthrough is kept because a
    multi-domain STM would need exactly it, and it is covered by
    ``test_spectral_init_beta_threads_domain_bounds_to_both_anchor_passes`` (which
    asserts both anchor passes receive it), not left as untested surface.
    """
    if domain_bounds is not None:
        domain_bounds = validate_domain_bounds(domain_bounds, V)
    K = partition.K
    beta = np.zeros((K, V), dtype=np.float64)

    # Step 1: background block on pooled Q.
    Q_all = word_cooccurrence(docs, V)
    bg_anchors = find_anchors(Q_all, partition.background_k, domain_bounds=domain_bounds)
    bg_beta = recover_beta(Q_all, bg_anchors)
    bg_idx = partition.background_indices()
    # bg_beta has one row per found anchor; if find_anchors fell short (corpus
    # too degenerate to yield background_k distinct directions), only fill what
    # we have and leave the rest as zero rows (the hook never sees a NaN).
    n_bg = min(len(bg_idx), bg_beta.shape[0])
    beta[bg_idx[:n_bg]] = bg_beta[:n_bg]

    # Step 2: each group's foreground on within-group Q, deflated vs background.
    #
    # Deflation happens twice, both against bg_anchors:
    #   (a) selection — find_anchors(seed_rows=bg_anchors) spans the search away
    #       from the shared background so the chosen foreground anchors are the
    #       group-specific phenotype words, not background.
    #   (b) recovery — recover_beta is run with the background anchors INCLUDED
    #       alongside the foreground anchors. Within a group's docs the foreground
    #       words still co-occur heavily with background words; letting the NNLS
    #       place that shared mass on the background anchor columns leaves the
    #       foreground topic rows concentrated on the phenotype. We keep only the
    #       foreground rows (the trailing len(fg_anchors) of the recovered block).
    for g in partition.groups:
        fg_idx = partition.block_indices(g)
        docs_g = [d for d in docs if g in d.groups]
        Q_g = word_cooccurrence(docs_g, V)
        fg_anchors = find_anchors(Q_g, len(fg_idx), seed_rows=bg_anchors,
                                  domain_bounds=domain_bounds)
        if not fg_anchors:
            continue
        combined = list(bg_anchors) + list(fg_anchors)
        combined_beta = recover_beta(Q_g, combined)
        fg_beta = combined_beta[len(bg_anchors):]          # drop the background rows
        n_fg = min(len(fg_idx), fg_beta.shape[0])
        beta[fg_idx[:n_fg]] = fg_beta[:n_fg]

    return beta


def split_domains(beta, domain_bounds):
    """Split a joint K×V β into per-domain row-renormalized bases.

    Under the shared-topic multi-domain model a token drawn in domain 0 and a
    token drawn in domain 1 from one document share the same θ, so the joint
    co-occurrence factors as Q_01 = (B_0)ᵀ A (B_1) (spec) and ONE anchor defines
    the topic across both domains. After recover_beta returns the joint β over
    the concatenated vocab, slicing each topic row at the domain boundaries and
    renormalizing each slice to sum 1 gives the per-domain P(word | topic)
    matrices — the MixEHR-style bases (β^0, β^1) that share topic identity
    (Halpern, Horng, Choi, Sontag, JAMIA 2016, anchor-and-learn corroboration).

    domain_bounds: strictly-increasing cumulative offsets [0, ..., V], validated
    against β's own column count (`domains.validate_domain_bounds`) -- bounds
    ending short of V used to silently DROP the uncovered trailing columns from
    the returned blocks. Returns a list of (K, V_d) row-stochastic matrices in
    domain order. A topic that never expresses a domain (all-zero slice) falls
    back to a uniform row there so each returned matrix stays a valid stochastic
    matrix.
    """
    domain_bounds = validate_domain_bounds(domain_bounds, np.shape(beta)[1])
    out = []
    for lo, hi in zip(domain_bounds[:-1], domain_bounds[1:]):
        sub = beta[:, lo:hi].copy()
        rs = sub.sum(axis=1, keepdims=True)
        zero = (rs[:, 0] <= 0)
        if zero.any():
            sub[zero] = 1.0 / (hi - lo)
            rs[zero, 0] = 1.0
        out.append(sub / rs)
    return out
