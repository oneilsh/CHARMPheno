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
    """
    pooled_QR: np.ndarray
    group_QR: dict
    p_w: np.ndarray
    df_w: np.ndarray


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
            n_docs += 1
        return [(pooled, group_QR, p_w, df_w, n_docs)]

    def _combine(a, b):
        ga, gb = a[1], b[1]
        merged = {g: ga[g] + gb[g] for g in ga}
        return (a[0] + b[0], merged, a[2] + b[2], a[3] + b[3], a[4] + b[4])

    pooled, group_QR, p_w, df_w, _n = rdd.mapPartitions(_local).treeReduce(
        _combine, depth=depth
    )
    return ProjectedCoocResult(
        pooled_QR=pooled, group_QR=group_QR, p_w=p_w, df_w=df_w
    )
