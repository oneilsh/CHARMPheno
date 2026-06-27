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


def find_anchors(Q: np.ndarray, n: int, *, seed_rows=None,
                 min_marginal_frac: float = 1.0) -> list[int]:
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

    Returns the ``n`` newly chosen anchor word ids in selection order.
    """
    Qbar = _row_normalize(Q)
    V = Qbar.shape[0]
    norms = (Qbar * Qbar).sum(axis=1)            # squared row norms

    marginal = Q.sum(axis=1)
    pos = marginal > 0
    thr = min_marginal_frac * marginal[pos].mean() if pos.any() else 0.0
    candidate = marginal >= thr                   # eligible to BE an anchor

    basis: list[np.ndarray] = []                  # orthonormal residual basis
    EPS = 1e-12

    def project_out(vec: np.ndarray) -> np.ndarray:
        r = vec.copy()
        for b in basis:
            r = r - (r @ b) * b
        return r

    def add_to_basis(row_id: int) -> None:
        r = project_out(Qbar[row_id])
        nrm = np.sqrt(r @ r)
        if nrm > EPS:
            basis.append(r / nrm)

    if seed_rows is not None:
        for s in seed_rows:
            add_to_basis(int(s))

    anchors: list[int] = []
    chosen = set(int(s) for s in (seed_rows or []))
    for _ in range(n):
        # residual squared norm after projecting each row onto current basis
        best_id, best_res = -1, -np.inf
        for i in range(V):
            if i in chosen or norms[i] <= EPS or not candidate[i]:
                continue
            r = project_out(Qbar[i])
            res = r @ r
            if res > best_res:
                best_res, best_id = res, i
        if best_id < 0:                           # exhausted distinct directions
            break
        anchors.append(best_id)
        chosen.add(best_id)
        add_to_basis(best_id)
    return anchors


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


def spectral_init_beta(docs, partition, V: int) -> np.ndarray:
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
    """
    K = partition.K
    beta = np.zeros((K, V), dtype=np.float64)

    # Step 1: background block on pooled Q.
    Q_all = word_cooccurrence(docs, V)
    bg_anchors = find_anchors(Q_all, partition.background_k)
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
        fg_anchors = find_anchors(Q_g, len(fg_idx), seed_rows=bg_anchors)
        if not fg_anchors:
            continue
        combined = list(bg_anchors) + list(fg_anchors)
        combined_beta = recover_beta(Q_g, combined)
        fg_beta = combined_beta[len(bg_anchors):]          # drop the background rows
        n_fg = min(len(fg_idx), fg_beta.shape[0])
        beta[fg_idx[:n_fg]] = fg_beta[:n_fg]

    return beta
