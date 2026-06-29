"""Scalable (random-projection) foundation for anchor-word spectral init.

The dense anchor-word recipe (``spectral_init.word_cooccurrence``) materializes
the V×V same-document co-occurrence matrix Q on the driver — O(V²), ~80GB at
V=100k. ADR 0032 scales it by RANDOM PROJECTION: we never form V×V, instead we
project the co-occurrence rows onto a d-dimensional Gaussian sketch R (V×d) and
accumulate only the V×d projected matrix QR, plus the V-vectors p_w (word
marginal = true Q row sums) and df_w (document frequency). Anchor finding and
β-recovery run on the sketch in a SEPARATE later task; this module builds only
the projection + the distributed accumulation pass.

The crux is the per-doc rank-1 structure. For a doc with support ``idx``, counts
``n``, and length ``L = Σ n`` (only L ≥ 2 docs contribute), the dense per-doc
co-occurrence block is

    block = (outer(n, n) − diag(n)) / (L · (L − 1))          # (|idx|, |idx|)

Projecting onto R restricted to the support, with ``s = Σ_j n_j · R[idx_j]`` (one
d-vector per doc), the contribution to projected row i is rank-1 plus a diagonal
self-pair correction:

    block @ R[idx] row i  =  n_i · (s − R[idx_i]) / (L · (L − 1))

The ``− R[idx_i]`` term is exactly the ``− diag(n)`` self-pair removal carried
into the projected space (the i==j column contributes ``n_i² R[idx_i]`` from
outer(n,n) and ``− n_i R[idx_i]`` from diag(n); factoring n_i leaves ``s − R[idx_i]``).
The true Q row-sum (word marginal) needs no projection — it is the scalar

    p_w_contrib[i] = n_i · (L − 1) / (L · (L − 1)) = n_i / L  # = block.sum(axis=1)[i]

(only ONE self-pair count of n_i is removed by −diag(n), so the marginal is n_i/L).

R is generated DETERMINISTICALLY PER ROW (no broadcast of a V×d matrix):
``R[j] = default_rng(SeedSequence([seed, j])).standard_normal(d)``. This makes
the projected result reproducible regardless of how the corpus is partitioned —
two executors that both touch word j regenerate the identical row.

Domain-agnostic: integer token ids only, no OMOP/EHR vocabulary.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls


def default_projection_dim(K: int, V: int, eps: float = 0.1) -> int:
    """Johnson–Lindenstrauss target dimension ``max(K, ceil(eps^-2 · ln V))``.

    The JL lemma says a random Gaussian projection into ``ceil(eps^-2 · ln V)``
    dimensions preserves pairwise distances among V points to within (1 ± eps),
    which is what keeps the anchor geometry (the convex-hull vertices of the
    co-occurrence rows) intact in the sketch. We floor at K because the recovery
    needs at least K independent directions to place K anchors.
    """
    return max(int(K), math.ceil(eps ** -2 * math.log(V)))


def _r_rows(indices, seed: int, d: int, cache: dict | None = None) -> np.ndarray:
    """Per-row Gaussian sketch rows ``R[indices]`` — shape ``(len(indices), d)``.

    Row j is ``default_rng(SeedSequence([seed, j])).standard_normal(d)``,
    independent of every other row and of the partitioning. ``cache`` (an
    optional bounded dict, word id -> row) memoizes rows within a partition so a
    word that appears in many docs is regenerated once.
    """
    idx = np.asarray(indices, dtype=np.int64)
    out = np.empty((idx.shape[0], d), dtype=np.float64)
    for k, j in enumerate(idx):
        ji = int(j)
        if cache is not None and ji in cache:
            out[k] = cache[ji]
            continue
        row = np.random.default_rng(
            np.random.SeedSequence([int(seed), ji])
        ).standard_normal(d)
        out[k] = row
        if cache is not None:
            cache[ji] = row
    return out


def _project_doc(indices, counts, R_rows):
    """Rank-1 projected co-occurrence + word-marginal contributions for one doc.

    Pure numpy. ``indices``/``counts`` are the doc's support and counts;
    ``R_rows`` is that support's sketch rows (``_r_rows(indices, ...)``, shape
    ``(|idx|, d)``). Returns ``(qr_contrib, pw_contrib)``:

        qr_contrib[i] = n_i · (s − R[i]) / (L · (L − 1))     # shape (|idx|, d)
        pw_contrib[i] = n_i / L                              # shape (|idx|,)

    with ``s = Σ_j n_j · R[j]``. Caller must skip docs with L < 2 (they carry no
    co-occurrence; this function would divide by zero).
    """
    n = np.asarray(counts, dtype=np.float64)
    L = float(n.sum())
    denom = L * (L - 1.0)
    s = (n[:, None] * R_rows).sum(axis=0)                 # (d,)
    qr_contrib = (n[:, None] * (s[None, :] - R_rows)) / denom
    # block.sum(axis=1)[i] = (n_i·L − n_i) / (L·(L−1)) = n_i / L. Only ONE self-
    # pair count of n_i is removed by −diag(n) (the δ_ij term), not n_i², so the
    # marginal is n_i/L, NOT n_i·(L − n_i)/(L·(L−1)).
    pw_contrib = (n * (L - 1.0)) / denom                  # == n / L
    return qr_contrib, pw_contrib


@dataclass
class ProjectedCoocResult:
    """Output of the distributed projected-co-occurrence pass.

    pooled_QR: (V, d) summed projected co-occurrence over all docs (L ≥ 2).
    group_QR:  group label -> (V, d) summed projected co-occurrence over that
               group's docs (a doc in group g adds its SAME qr_contrib to both
               pooled_QR and group_QR[g], fusing the dense path's separate
               per-group Q passes).
    p_w:       (V,) summed word marginal (true Q row sums), float64.
    df_w:      (V,) document frequency over L ≥ 2 docs, int64.
    group_p_w: group label -> (V,) summed WITHIN-GROUP word marginal over that
               group's docs. The gated foreground recovery needs the within-group
               marginal (the dense path uses Q_g.sum(axis=1)); the pooled p_w is
               the wrong normalizer there. A doc in group g adds its SAME
               pw_contrib to both p_w and group_p_w[g].
    group_df_w:group label -> (V,) within-group document frequency over L ≥ 2 docs.
    """
    pooled_QR: np.ndarray
    group_QR: dict
    p_w: np.ndarray
    df_w: np.ndarray
    group_p_w: dict
    group_df_w: dict


def projected_cooccurrence_rdd(
    rdd, partition, V: int, d: int, seed: int, *, depth: int = 2,
    dtype=np.float32,
) -> ProjectedCoocResult:
    """One distributed pass: pooled + per-group projected co-occurrence, p_w, df_w.

    ``rdd`` is an RDD of ``STMDocument``. Mirrors the engine's
    mapPartitions+treeReduce+broadcast-via-default-arg idiom
    (cf. ``corpus_mean_topic_proportions_gated_rdd``): each partition accumulates
    its V×d / V slabs locally, regenerating sketch rows per word (bounded cache),
    and the tree-combine sums slabs elementwise — only the accumulators ever reach
    the driver, never a V×V matrix.

    A doc in group g (g ∈ ``partition.groups``) adds its SAME ``qr_contrib`` to
    both ``pooled_QR`` and ``group_QR[g]``. The (V, d) accumulators use ``dtype``
    (default float32, for memory at large V); p_w is float64 and df_w int64.

    Normalization: results are left on the SUMMED scale (Σ_docs, no division by
    n_docs or grand total). Downstream anchor-finding row-normalizes Q's sketch,
    so any global positive scalar cancels — averaging here would only lose
    precision and force a second statistic (n_docs) across the wire.
    """
    sc = rdd.context
    p_bcast = sc.broadcast(partition)
    groups = tuple(partition.groups)

    def _local(docs, _p=p_bcast, _V=V, _d=d, _seed=seed, _dtype=dtype,
               _groups=groups):
        part = _p.value
        pooled = np.zeros((_V, _d), dtype=_dtype)
        group_QR = {g: np.zeros((_V, _d), dtype=_dtype) for g in _groups}
        group_p_w = {g: np.zeros(_V, dtype=np.float64) for g in _groups}
        group_df_w = {g: np.zeros(_V, dtype=np.int64) for g in _groups}
        p_w = np.zeros(_V, dtype=np.float64)
        df_w = np.zeros(_V, dtype=np.int64)
        known = set(_groups)
        r_cache: dict = {}
        n_docs = 0
        for doc in docs:
            n = np.asarray(doc.counts, dtype=np.float64)
            L = float(n.sum())
            if L < 2:
                continue
            idx = np.asarray(doc.indices, dtype=np.int64)
            R_rows = _r_rows(idx, _seed, _d, cache=r_cache)
            qr, pw = _project_doc(idx, n, R_rows)
            qr = qr.astype(_dtype)
            pooled[idx] += qr
            p_w[idx] += pw
            df_w[idx] += 1
            for g in doc.groups:
                if g in known:
                    group_QR[g][idx] += qr
                    group_p_w[g][idx] += pw
                    group_df_w[g][idx] += 1
            n_docs += 1
        return [(pooled, group_QR, p_w, df_w, group_p_w, group_df_w, n_docs)]

    def _combine(a, b):
        ga, gb = a[1], b[1]
        merged = {g: ga[g] + gb[g] for g in ga}
        gpa, gpb = a[4], b[4]
        merged_pw = {g: gpa[g] + gpb[g] for g in gpa}
        gda, gdb = a[5], b[5]
        merged_df = {g: gda[g] + gdb[g] for g in gda}
        return (
            a[0] + b[0], merged, a[2] + b[2], a[3] + b[3],
            merged_pw, merged_df, a[6] + b[6],
        )

    (pooled, group_QR, p_w, df_w,
     group_p_w, group_df_w, _n) = rdd.mapPartitions(_local).treeReduce(
        _combine, depth=depth
    )
    return ProjectedCoocResult(
        pooled_QR=pooled, group_QR=group_QR, p_w=p_w, df_w=df_w,
        group_p_w=group_p_w, group_df_w=group_df_w,
    )


def _row_normalize_projected(QR: np.ndarray, p_w: np.ndarray) -> np.ndarray:
    """Projected analog of dense ``_row_normalize``: ``QR[i] / p_w[i]``.

    The dense ``spectral_init._row_normalize`` divides Q row i by its sum, which
    is exactly the word marginal ``p_w[i]``. In the sketch the row ``QR[i]`` can
    have negative entries and its *sum* is NOT the marginal (R has zero-mean
    Gaussian columns), so we cannot recover the conditional row by dividing by
    ``QR[i].sum()``. But ``QR`` and ``p_w`` come off the same pass on the same
    summed scale, so ``QR[i] / p_w[i]`` is the exact projected image
    ``(Q[i] / p_w[i]) @ R`` of the dense conditional row ``Qbar[i]``. A word with
    no co-occurrence support (``p_w[i] == 0``) yields a zero row, mirroring the
    dense zero-row convention.
    """
    p_safe = np.where(p_w > 0, p_w, 1.0)
    Qbar = QR / p_safe[:, None]
    Qbar[p_w <= 0] = 0.0
    return Qbar


_EPS = 1e-12


def find_anchors_projected(QR: np.ndarray, p_w: np.ndarray, df_w: np.ndarray,
                           n: int, *, seed_rows=None,
                           min_doc_freq: int = 5) -> list[int]:
    """Greedy pivoted-QR anchor selection on the p_w-row-normalized sketch.

    Same geometry as dense ``spectral_init.find_anchors`` (Gram–Schmidt: keep an
    orthonormal basis of the chosen rows' residuals; the next anchor is the
    eligible word whose normalized row has the largest residual norm after
    projecting out that basis), but operating on the projected conditional rows
    ``Qbar_proj = QR / p_w[:,None]`` (see ``_row_normalize_projected``) instead of
    the dense ``Qbar``.

    Candidate floor (ADR 0032): a word may BE an anchor only if its absolute
    document frequency ``df_w[i] >= min_doc_freq``. This replaces the dense
    version's mean-relative ``min_marginal_frac`` floor, which over-excludes
    rare-but-pure phenotype words (a minority arm's phenotype is below the corpus
    mean marginal yet appears in plenty of that arm's docs). The dense
    ``norm > EPS`` guard is kept (a degenerate all-zero row is never an anchor).

    ``seed_rows`` (optional word ids) pre-seeds the basis (deflation) WITHOUT
    being returned. Returns the ``n`` newly chosen anchor ids in selection order.
    """
    Qbar = _row_normalize_projected(QR, p_w)
    V = Qbar.shape[0]
    norms = (Qbar * Qbar).sum(axis=1)            # squared row norms
    df = np.asarray(df_w)
    candidate = df >= min_doc_freq               # eligible to BE an anchor

    basis: list[np.ndarray] = []                  # orthonormal residual basis

    def project_out(vec: np.ndarray) -> np.ndarray:
        r = vec.copy()
        for b in basis:
            r = r - (r @ b) * b
        return r

    def add_to_basis(row_id: int) -> None:
        r = project_out(Qbar[row_id])
        nrm = np.sqrt(r @ r)
        if nrm > _EPS:
            basis.append(r / nrm)

    if seed_rows is not None:
        for s in seed_rows:
            add_to_basis(int(s))

    anchors: list[int] = []
    chosen = set(int(s) for s in (seed_rows or []))
    for _ in range(n):
        best_id, best_res = -1, -np.inf
        for i in range(V):
            if i in chosen or norms[i] <= _EPS or not candidate[i]:
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


def recover_beta_projected(QR: np.ndarray, p_w: np.ndarray, anchors,
                           rows=None) -> np.ndarray:
    """Per-word NNLS β recovery in projected space — analog of dense ``recover_beta``.

    Driver-side loop (sufficient at the cancer scale V≈3691; a distributed
    Spark-map version is a future enhancement). For each word w, solve
    ``min_{c >= 0} || A^T c − Qbar_proj[w] ||`` where ``A = Qbar_proj[anchors]``
    (the projected conditional anchor rows). NNLS is valid on arbitrary-sign A/b —
    non-negativity is only on the weights c. Normalizing c to sum 1 gives
    P(topic | word = w); a word with no support (all-zero c) is left at zero.
    Bayes-flip with the marginal ``p_w`` and renormalize each topic row; an
    all-zero topic row falls back to uniform ``1/V`` (mirrors the dense guard).
    """
    Qbar = _row_normalize_projected(QR, p_w)
    V = Qbar.shape[0]
    n_topics = len(anchors)
    A = Qbar[list(anchors)]                        # (n_topics, d)
    A_T = A.T                                       # (d, n_topics): solve A_T c = Qbar_w

    if rows is None:
        rows = range(V)
    rows = list(rows)

    topic_given_word = np.zeros((V, n_topics), dtype=np.float64)
    for w in rows:
        c, _ = nnls(A_T, Qbar[w])
        s = c.sum()
        if s > 0:
            topic_given_word[w] = c / s
        # else: no support -> left at zero (carries no signal).

    beta = (topic_given_word * p_w[:, None]).T     # (n_topics, V)
    row_sums = beta.sum(axis=1, keepdims=True)
    zero = (row_sums[:, 0] <= 0)
    if zero.any():
        beta[zero] = 1.0 / V
        row_sums[zero, 0] = 1.0
    beta = beta / row_sums
    return beta


def scalable_spectral_init_beta(
    rdd, partition, V: int, *, d: int | None = None, eps: float = 0.1,
    seed: int = 0, min_doc_freq: int = 5,
) -> np.ndarray:
    """Block-aware K×V β seed in ``partition`` slot order — scalable analog.

    Mirrors the dense ``spectral_init.spectral_init_beta`` orchestration exactly,
    substituting the projected primitives so the whole co-occurrence pass is a
    SINGLE distributed sketch (never a V×V matrix on the driver):

    Step 1 (background): pooled projected sketch over all docs → ``background_k``
    anchors on the pooled sketch → recover those rows into
    ``partition.background_indices()``.

    Step 2 (per group): for each group g, use that group's WITHIN-GROUP sketch
    (``group_QR[g]`` with within-group marginal ``group_p_w[g]`` /
    ``group_df_w[g]``) — where a rare group's phenotype is undiluted by the
    majority. Find ``len(block_indices(g))`` foreground anchors with the background
    anchors passed as ``seed_rows`` (deflation in selection), recover the combined
    background+foreground anchors against the within-group sketch (deflation in
    recovery), and keep only the trailing foreground rows into
    ``partition.block_indices(g)``.

    Non-gated partitions (background_k = K, no foreground groups) execute step 1
    only — the dense degenerate case, one code path, no special case.

    ``d`` defaults to ``default_projection_dim(partition.K, V, eps)``. Short-fill
    guards match the dense path: if anchor-finding falls short, only the found
    rows are placed and the rest stay zero (the hook never sees a NaN).
    """
    K = partition.K
    if d is None:
        d = default_projection_dim(K, V, eps)
    beta = np.zeros((K, V), dtype=np.float64)

    # ONE distributed pass: pooled + per-group projected sketches, marginals, df.
    res = projected_cooccurrence_rdd(rdd, partition, V, d, seed)

    # Step 1: background block on the pooled sketch.
    bg_anchors = find_anchors_projected(
        res.pooled_QR, res.p_w, res.df_w, partition.background_k,
        min_doc_freq=min_doc_freq,
    )
    bg_beta = recover_beta_projected(res.pooled_QR, res.p_w, bg_anchors)
    bg_idx = partition.background_indices()
    # Short-fill guard: fill only what find_anchors_projected returned.
    n_bg = min(len(bg_idx), bg_beta.shape[0])
    beta[bg_idx[:n_bg]] = bg_beta[:n_bg]

    # Step 2: each group's foreground on its within-group sketch, deflated vs bg.
    for g in partition.groups:
        fg_idx = partition.block_indices(g)
        fg_anchors = find_anchors_projected(
            res.group_QR[g], res.group_p_w[g], res.group_df_w[g], len(fg_idx),
            seed_rows=bg_anchors, min_doc_freq=min_doc_freq,
        )
        if not fg_anchors:
            continue
        combined = list(bg_anchors) + list(fg_anchors)
        combined_beta = recover_beta_projected(
            res.group_QR[g], res.group_p_w[g], combined
        )
        fg_beta = combined_beta[len(bg_anchors):]      # drop the background rows
        n_fg = min(len(fg_idx), fg_beta.shape[0])
        beta[fg_idx[:n_fg]] = fg_beta[:n_fg]

    return beta
