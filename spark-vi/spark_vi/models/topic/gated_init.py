"""Pluggable init strategies for GatedOnlineLDA.

A strategy is `f(data_summary, lay, V) -> (K, V) lambda`, called by
GatedOnlineLDA.initialize_global when init != "random". "random" is the validated default
(the DAG gate supplies identifiability) and lives in OnlineLDA.initialize_global, so it is
NOT in this registry.

`spectral_block_aligned_lambda` generalizes OnlineLDA.spectral_init.spectral_init_beta's
background->foreground init to a multi-level DAG: each node is recovered by anchor-word
spectral recovery (Arora et al. 2013) DEFLATED against its already-recovered closure-ancestor
anchors, so it must run in FORWARD topological order (ancestors first — a node can only be
deflated against ancestors already recovered). This is a documented OPTIONAL strategy: on the
synthetic plants it did not improve the gated fit (the gate already breaks symmetry) and could
regress shallow nodes when a recovered seed row was imperfect — see the design spec's prototype
findings. Kept for the real-DAG A/B harness and as the extension point for future strategies
(e.g. phenotype-profile seeding)."""
from __future__ import annotations

import logging

import numpy as np

from spark_vi.models.topic.dag_placement import _as_counts
from spark_vi.models.topic.spectral_init import (
    word_cooccurrence, find_anchors, recover_beta,
)

logger = logging.getLogger(__name__)


def _union_closure(front, lay):
    s = set()
    for f in front:
        for u in lay.closure(f):
            if u != 0:
                s.add(u)
    return s


def spectral_block_aligned_lambda(data_summary, lay, V, *, scale: float = 200.0) -> np.ndarray:
    """Forward-topological block-aligned spectral lambda seed.

    data_summary carries {"train_docs": [token-id arrays], "train_labels": [node id or
    frontier set per doc]}. Returns a (K, V) lambda = block-aligned beta * scale.

    Step 1 (background): pooled Q over all docs -> n_bg anchors -> background block.
    Step 2 (each node, ancestors-first by lay.depth): docs training node u = those with u in
    the union of their frontier closures; find tpn anchors on the within-node Q_u deflated
    against background + u's already-recovered proper-ancestor anchors (seed_rows AND
    include-then-drop in recover_beta), recover into u's block."""
    if not (isinstance(data_summary, dict)
            and "train_docs" in data_summary and "train_labels" in data_summary):
        raise ValueError(
            "spectral init requires data_summary={'train_docs':..., 'train_labels':...}"
        )
    train_docs = data_summary["train_docs"]
    train_labels = data_summary["train_labels"]
    counted = [_as_counts(d) for d in train_docs]
    fronts = [set(y) if hasattr(y, "__iter__") else {int(y)} for y in train_labels]
    trains = [_union_closure(f, lay) for f in fronts]

    beta = np.zeros((lay.K, V))
    Q_all = word_cooccurrence(counted, V)
    bg_anchors = find_anchors(Q_all, lay.n_bg)
    bg_beta = recover_beta(Q_all, bg_anchors)
    for i in range(min(lay.n_bg, bg_beta.shape[0])):
        beta[i] = bg_beta[i]

    node_anchors: dict[int, list] = {}
    for u in sorted(lay.nodes, key=lambda x: (lay.depth(x), x)):   # forward topological
        docs_u = [counted[d] for d in range(len(counted)) if u in trains[d]]
        if not docs_u:
            logger.warning(
                "spectral_block_aligned_lambda: node %s has zero training docs; "
                "its block stays at the 1e-9 floor (uninitialized).", u,
            )
            continue
        Q_u = word_cooccurrence(docs_u, V)
        anc = [a for a in lay.closure(u) if a not in (u, 0)]
        seed = list(bg_anchors) + [a for p in anc for a in node_anchors.get(p, [])]
        fg_anchors = find_anchors(Q_u, lay.tpn, seed_rows=seed)
        if not fg_anchors:
            logger.warning(
                "spectral_block_aligned_lambda: node %s found no anchors "
                "(sparse/degenerate co-occurrence); its block stays at the "
                "1e-9 floor (uninitialized).", u,
            )
            continue
        node_anchors[u] = list(fg_anchors)
        combined_beta = recover_beta(Q_u, list(seed) + list(fg_anchors))
        fg_beta = combined_beta[len(seed):]
        for j, idx in enumerate(lay.block[u]):
            if j < fg_beta.shape[0]:
                beta[idx] = fg_beta[j]

    beta = beta + 1e-9                                    # keep lambda strictly positive
    return beta * float(scale)


INIT_STRATEGIES = {"spectral": spectral_block_aligned_lambda}


class _GroupDoc:
    """Minimal doc view for projected_cooccurrence_rdd: token support/counts plus
    the DAG node groups this doc trains (its frontier closure, root 0 dropped)."""
    __slots__ = ("indices", "counts", "groups")

    def __init__(self, indices, counts, groups):
        self.indices = indices
        self.counts = counts
        self.groups = groups


class _NodeGroups:
    """Minimal partition view for projected_cooccurrence_rdd: it reads only
    `.groups` (the set of node ids that get a per-group sketch)."""
    __slots__ = ("groups",)

    def __init__(self, groups):
        self.groups = groups


def scalable_block_aligned_lambda(rdd, lay, V, *, d: int | None = None,
                                  seed: int = 0, min_doc_freq: int = 5,
                                  scale: float = 200.0) -> np.ndarray:
    """Distributed random-projection analogue of `spectral_block_aligned_lambda`.

    `rdd` is an RDD of GatedBOWDocument. Never forms a driver V×V matrix (ADR
    0032). Runs a STREAMING sequence of projected co-occurrence passes so the
    driver holds only ONE (V, d) sketch at a time — O(V·d) driver memory,
    INDEPENDENT of the node count. (Accumulating every node's sketch in a single
    pass instead costs O(n_nodes·V·d) on the driver, which overflows
    spark.driver.maxResultSize once the DAG has more than a handful of nodes.)

    A doc trains node u iff u is in its frontier closure (root 0 dropped), so the
    per-node co-occurrence Q_u equals the POOLED sketch of the sub-RDD filtered to
    those docs — this is what lets each node use its own one-slab pass:

    Step 1 (background): pooled sketch over ALL docs (one pass) → n_bg anchors →
    background block.
    Step 2 (each node u, ancestors-first by lay.depth): pooled sketch over the
    sub-RDD of docs training u (one pass = Q_u's sketch); find tpn anchors on it
    deflated against background + u's already-recovered proper-ancestor anchors
    (seed_rows), recover the combined anchors, keep the trailing foreground rows
    into u's block. Anchor-word spectral recovery (Arora et al. 2013); the random
    projection preserves the residual-norm geometry the greedy anchor search needs
    (Johnson–Lindenstrauss).

    Returns a (K, V) λ = block-aligned β * scale, the same contract as the dense
    function (a drop-in seed). Numerically identical to the single-pass
    all-groups accumulation — same doc set and same per-word sketch rows per
    node, just materialized one node at a time."""
    from pyspark import StorageLevel
    from spark_vi.models.topic.spectral_init_scalable import (
        projected_cooccurrence_rdd, find_anchors_projected,
        recover_beta_projected, default_projection_dim,
    )
    if d is None:
        d = default_projection_dim(lay.K, V)

    lay_b = rdd.context.broadcast(lay)

    def _to_group(doc, _lay=lay_b):
        L = _lay.value
        groups = set()
        for f in doc.frontier:
            for u in L.closure(f):
                if u != 0:
                    groups.add(u)
        # frozenset so each node's `_u in gd.groups` filter is O(1); computed once
        # here and cached so the per-node passes never recompute closures.
        return _GroupDoc(doc.indices, doc.counts, frozenset(groups))

    group_rdd = rdd.map(_to_group).persist(StorageLevel.MEMORY_AND_DISK)
    no_groups = _NodeGroups(())          # pooled-only passes (no per-group slabs)
    try:
        group_rdd.count()                # materialize the cache once

        beta = np.zeros((lay.K, V), dtype=np.float64)

        # Step 1: background block on the pooled sketch over ALL docs.
        pooled = projected_cooccurrence_rdd(group_rdd, no_groups, V, d, seed)
        bg_anchors = find_anchors_projected(
            pooled.pooled_QR, pooled.p_w, pooled.df_w, lay.n_bg,
            min_doc_freq=min_doc_freq)
        bg_beta = recover_beta_projected(pooled.pooled_QR, pooled.p_w, bg_anchors)
        for i in range(min(lay.n_bg, bg_beta.shape[0])):
            beta[i] = bg_beta[i]

        # Step 2: each node, ancestors first, its OWN filtered one-slab pass.
        node_anchors: dict[int, list] = {}
        for u in sorted(lay.nodes, key=lambda x: (lay.depth(x), x)):
            rdd_u = group_rdd.filter(lambda gd, _u=u: _u in gd.groups)
            res_u = projected_cooccurrence_rdd(rdd_u, no_groups, V, d, seed)
            if int(res_u.df_w.sum()) == 0:
                logger.warning(
                    "scalable_block_aligned_lambda: node %s has zero training "
                    "docs; its block stays at the 1e-9 floor (uninitialized).", u)
                continue
            anc = [a for a in lay.closure(u) if a not in (u, 0)]
            seed_rows = list(bg_anchors) + [a for p in anc
                                            for a in node_anchors.get(p, [])]
            fg_anchors = find_anchors_projected(
                res_u.pooled_QR, res_u.p_w, res_u.df_w, lay.tpn,
                seed_rows=seed_rows, min_doc_freq=min_doc_freq)
            if not fg_anchors:
                logger.warning(
                    "scalable_block_aligned_lambda: node %s found no anchors "
                    "(sparse/degenerate sketch); its block stays at the floor.", u)
                continue
            node_anchors[u] = list(fg_anchors)
            combined_beta = recover_beta_projected(
                res_u.pooled_QR, res_u.p_w, list(seed_rows) + list(fg_anchors))
            fg_beta = combined_beta[len(seed_rows):]
            for j, idx in enumerate(lay.block[u]):
                if j < fg_beta.shape[0]:
                    beta[idx] = fg_beta[j]
    finally:
        group_rdd.unpersist(blocking=False)

    beta = beta + 1e-9                                   # strictly positive λ
    return beta * float(scale)
