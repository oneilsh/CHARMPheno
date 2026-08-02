"""Pluggable init strategies for GatedOnlineLDA.

A strategy is `f(data_summary, lay, V) -> (K, V) lambda`, called by
GatedOnlineLDA.initialize_global when init != "random". "random" is the validated default
(the DAG gate supplies identifiability) and lives in OnlineLDA.initialize_global, so it is
NOT in this registry.

`spectral_block_aligned_lambda` generalizes OnlineLDA.spectral_init.spectral_init_beta's
background->foreground init to a multi-level DAG: each node is recovered by anchor-word
spectral recovery (Arora et al. 2013) DEFLATED against its already-recovered relatives. The
`topo_order` knob picks which relatives: "forward" (default) recovers nodes ancestors-first
and deflates each against its proper ANCESTORS (a node's topic = its increment over its
ancestors); "reverse" recovers leaves-first and deflates each against its proper DESCENDANTS
(leaves claim their full signal first; an ancestor's topic = the residual after its
descendants). Either way a node can only be deflated against relatives already recovered, so
the iteration order is depth-monotonic in the chosen direction (see `_node_order_and_relatives`).
This is a documented OPTIONAL strategy: on the
synthetic plants it did not improve the gated fit (the gate already breaks symmetry) and could
regress shallow nodes when a recovered seed row was imperfect — see the design spec's prototype
findings. Kept for the real-DAG A/B harness and as the extension point for future strategies
(e.g. phenotype-profile seeding)."""
from __future__ import annotations

import logging
import os

import numpy as np

from spark_vi.models.topic.dag_placement import _as_counts
from spark_vi.models.topic.spectral_init import (
    word_cooccurrence, find_anchors, recover_beta,
)

logger = logging.getLogger(__name__)


ANCHOR_SCOPES = ("closure", "frontier")

# Pseudo-count magnitude of a spectral lambda seed. Anchor recovery (Arora et al.
# 2013) supplies the topic SHAPE; this is the seed's total mass per topic, not a
# recovered quantity. It is the single source of truth for every spectral seed
# path -- the dense single-domain seed, the multi-domain dense seed, the scalable
# seed, AND the mllib shim's scalable->dict conversion -- so the dense and scalable
# multi-domain routes cannot drift to different seed strengths (SP3a whole-branch
# review Minor 1).
SPECTRAL_LAMBDA_SCALE = 200.0


def _validate_anchor_scope(anchor_scope):
    if anchor_scope not in ANCHOR_SCOPES:
        raise ValueError(
            f"anchor_scope must be one of {ANCHOR_SCOPES}, got {anchor_scope!r}")


TOPO_ORDERS = ("forward", "reverse")


def _validate_topo_order(topo_order):
    if topo_order not in TOPO_ORDERS:
        raise ValueError(
            f"topo_order must be one of {TOPO_ORDERS}, got {topo_order!r}")


def _node_order_and_relatives(lay, topo_order):
    """(ordered node list, relatives(u)) for the deflation loop.

    forward: nodes ascending depth (ancestors first); each node is deflated against its
    proper ANCESTORS (closure minus self/root) — a node's topic is its increment over its
    ancestors. reverse: nodes descending depth (leaves first); each node is deflated against
    its proper DESCENDANTS (subtree minus self) — leaves claim their full signal first and an
    ancestor's topic is the residual after its descendants. A proper descendant always has
    strictly greater longest-path depth than its ancestor, so descending-depth order
    guarantees every descendant is recovered before the node (the mirror of the forward
    ancestors-first guarantee)."""
    _validate_topo_order(topo_order)
    if topo_order == "forward":
        order = sorted(lay.nodes, key=lambda x: (lay.depth(x), x))

        def relatives(u):
            return [a for a in lay.closure(u) if a not in (u, 0)]
    else:
        order = sorted(lay.nodes, key=lambda x: (lay.depth(x), x), reverse=True)

        def relatives(u):
            return list(lay.descendants(u))     # already proper; 0 is never a descendant

    return order, relatives


def _union_closure(front, lay):
    s = set()
    for f in front:
        for u in lay.closure(f):
            if u != 0:
                s.add(u)
    return s


def _anchor_node_set(front, lay, anchor_scope):
    """Nodes whose anchor sketch this doc feeds, under `anchor_scope`.

    "closure" (default): the doc trains every node in its frontier's closure
    (ancestors included) — a leaf doc contributes to its parents' anchors too.
    "frontier": only the most-specific attested nodes (the frontier itself, root
    dropped) — so a node's anchors come only from docs where it is the deepest
    attested node, never from a descendant's docs. Background is handled
    separately (empty set here = a background doc in both scopes)."""
    if anchor_scope == "closure":
        return _union_closure(front, lay)
    return {int(u) for u in front if u != 0}          # "frontier"


def spectral_block_aligned_lambda(data_summary, lay, V, *, scale: float = SPECTRAL_LAMBDA_SCALE,
                                  anchor_scope: str = "closure",
                                  topo_order: str = "forward",
                                  domain_bounds=None) -> np.ndarray:
    """Block-aligned spectral lambda seed (topological, direction set by `topo_order`).

    data_summary carries {"train_docs": [token-id arrays], "train_labels": [node id or
    frontier set per doc]}. Returns a (K, V) lambda = block-aligned beta * scale.

    Step 1 (background): pooled Q over the background doc set -> n_bg anchors ->
    background block.
    Step 2 (each node, in the topo order set by `topo_order`): docs training node u = those
    in u's anchor node set; find tpn anchors on the within-node Q_u deflated against
    background + u's already-recovered relative anchors (seed_rows AND include-then-drop in
    recover_beta), recover into u's block.

    `topo_order` picks the deflation direction (see `_node_order_and_relatives`): "forward"
    (default) processes nodes ancestors-first and deflates u against its proper ANCESTORS
    (u's topic = its increment over its ancestors); "reverse" processes leaves-first and
    deflates u against its proper DESCENDANTS (leaves claim their full signal first; u's
    topic = the residual after its descendants). The "isolate u's own signal" guarantee for
    a given relative holds only when that relative has its own recovered anchors (see the
    anchor_scope corollary below — under "reverse" the same caveat applies to descendants
    that got no own docs).

    `anchor_scope` controls which docs feed each anchor set (see `_anchor_node_set`):
    "closure" (default, legacy) pools background over ALL docs and trains node u from every
    doc with u in its closure; "frontier" pools background over ONLY background docs (empty
    frontier) and trains node u from ONLY docs where u is the most-specific attested node —
    so anchor selection (a max-residual / information-content search) cannot let background
    steal a foreground node's defining word, nor a parent steal a child's, at any depth. In
    "frontier" mode an internal node whose patients always carry a subtype gets no
    node-specific docs and stays at the floor (informative: no distinct own-signal) — and,
    as a corollary, a descendant is then NOT deflated against that ancestor's increment
    (node_anchors[p] is absent), so its block can absorb the un-anchored ancestor signal;
    the "isolate u's own increment" guarantee holds only for ancestors that have their own
    frontier docs.

    `domain_bounds` (optional cumulative offsets [0, V_0, V_0+V_1, ..., V] over a
    concatenated multi-domain vocabulary) is passed to EVERY `find_anchors` call this makes —
    the pooled background call in step 1 and each node's within-node call in step 2 — so each
    domain's candidate floor is computed against its OWN mean marginal (see
    `spectral_init.find_anchors` / `_domain_candidate_mask`). It is LOAD-BEARING for
    multi-domain seeds: without it the denser domain dominates anchor selection, a node's
    sparse-domain slice is under-seeded, and the E/M leaves that node's topic dead at the
    prior (insight 0066; recovery 0.005 vs 0.675 at random_seed=0). Default `None` = one
    pooled domain, byte-identical to the single-domain behavior;
    `multidomain_spectral_lambda` is the multi-domain entry point that derives these bounds
    from per-domain sizes. Malformed bounds raise (`domains.validate_domain_bounds`).

    NOTE: only this DENSE path takes `domain_bounds` — the distributed twin
    `scalable_block_aligned_lambda` does not take it and (measured) still recovers
    every node on a well-specified multi-domain plant, because recovery there is
    carried by the gated EM rather than the seed. See that function's docstring and
    `test_scalable_init_recovers_every_identifying_signal`
    (spark-vi/tests/test_gated_lda.py). CAVEAT on the "0.005 vs 0.675" figure above:
    it comes from insight 0066's BACKGROUND-STARVED plant; insight 0067 later showed
    the dense floor's apparent value is largely a degenerate-plant artifact, so read
    "LOAD-BEARING" as "load-bearing on that degenerate plant", not unconditionally."""
    _validate_anchor_scope(anchor_scope)
    if not (isinstance(data_summary, dict)
            and "train_docs" in data_summary and "train_labels" in data_summary):
        raise ValueError(
            "spectral init requires data_summary={'train_docs':..., 'train_labels':...}"
        )
    train_docs = data_summary["train_docs"]
    train_labels = data_summary["train_labels"]
    counted = [_as_counts(d) for d in train_docs]
    fronts = [set(y) if hasattr(y, "__iter__") else {int(y)} for y in train_labels]
    trains = [_anchor_node_set(f, lay, anchor_scope) for f in fronts]

    beta = np.zeros((lay.K, V))
    # Background doc set: all docs ("closure") vs only background docs ("frontier").
    if anchor_scope == "closure":
        bg_docs = counted
    else:
        bg_docs = [counted[d] for d in range(len(counted)) if not trains[d]]
    if not bg_docs:
        logger.warning(
            "spectral_block_aligned_lambda: no background docs under anchor_scope="
            "%r; background block stays at the 1e-9 floor.", anchor_scope)
        bg_anchors = []
    else:
        Q_all = word_cooccurrence(bg_docs, V)
        bg_anchors = find_anchors(Q_all, lay.n_bg, domain_bounds=domain_bounds)
        bg_beta = recover_beta(Q_all, bg_anchors)
        for i in range(min(lay.n_bg, bg_beta.shape[0])):
            beta[i] = bg_beta[i]

    node_anchors: dict[int, list] = {}
    order, relatives = _node_order_and_relatives(lay, topo_order)
    for u in order:
        docs_u = [counted[d] for d in range(len(counted)) if u in trains[d]]
        if not docs_u:
            logger.warning(
                "spectral_block_aligned_lambda: node %s has zero training docs; "
                "its block stays at the 1e-9 floor (uninitialized).", u,
            )
            continue
        Q_u = word_cooccurrence(docs_u, V)
        anc = relatives(u)
        seed = list(bg_anchors) + [a for p in anc for a in node_anchors.get(p, [])]
        fg_anchors = find_anchors(Q_u, lay.tpn, seed_rows=seed, domain_bounds=domain_bounds)
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


def multidomain_spectral_lambda(data_summary, lay, domains, *, scale: float = SPECTRAL_LAMBDA_SCALE,
                                anchor_scope: str = "closure",
                                topo_order: str = "forward") -> dict:
    """Per-domain dict-lambda spectral seed for the multi-domain gated model.

    Runs the block-aligned anchor recipe (spectral_block_aligned_lambda) on the
    joint co-occurrence over the concatenated vocab [domain 0; domain 1; ...] WITH
    the per-domain candidate floor threaded through anchor selection (domain_bounds
    from `domains`), then splits the block-aligned joint beta into per-domain
    row-normalized matrices (spectral_init.split_domains) scaled into lambda_m.

    The per-domain floor is load-bearing: without it the denser domain dominates
    anchor selection and a node's sparse-domain slice is under-seeded, which the
    E/M then leaves as topic-death (a node topic stuck at the uniform prior; see
    insight 0066). With it, a node can anchor on its sparse-domain word, which
    then defines the topic across BOTH domains via the within-document cross-block
    co-occurrence (Q_01) — the MixEHR shared-theta tie (Li et al. 2020).

    `domains` = per-domain vocab sizes [V_0, V_1, ...]; V = sum(domains). Returns
    {m: (K, V_m)} lambda, each block a scaled per-domain distribution + a tiny
    positive floor. data_summary carries {'train_docs':..., 'train_labels':...} as
    for spectral_block_aligned_lambda."""
    from spark_vi.models.topic.domains import domains_to_bounds
    from spark_vi.models.topic.spectral_init import split_domains
    V = int(sum(domains))
    bounds = domains_to_bounds(domains).tolist()
    beta_joint = spectral_block_aligned_lambda(
        data_summary, lay, V, scale=1.0, anchor_scope=anchor_scope,
        topo_order=topo_order, domain_bounds=bounds)          # (K, V), rows joint distributions
    per_domain = split_domains(beta_joint, bounds)            # each row-normalized within its domain
    return {m: per_domain[m] * float(scale) + 1e-9 for m in range(len(domains))}


INIT_STRATEGIES = {"spectral": spectral_block_aligned_lambda}

# Multi-domain counterparts, keyed by the SAME strategy names. A strategy needs a
# separate entry because its multi-domain form has a different signature and return
# type (per-domain sizes in, a per-domain dict lambda {m: (K, V_m)} out), so
# INIT_STRATEGIES entries are not usable as-is. Keeping it a registry rather than
# hard-wiring the one implementation means a future strategy that is added to
# INIT_STRATEGIES but has no multi-domain form raises in GatedOnlineLDA.
# initialize_global instead of silently running the spectral recipe under its name.
MULTIDOMAIN_INIT_STRATEGIES = {"spectral": multidomain_spectral_lambda}


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
                                  scale: float = SPECTRAL_LAMBDA_SCALE,
                                  anchor_scope: str = "closure",
                                  topo_order: str = "forward") -> np.ndarray:
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
    Step 2 (each node u, in the topo order set by `topo_order`): pooled sketch over
    the sub-RDD of docs training u (one pass = Q_u's sketch); find tpn anchors on
    it deflated against background + u's already-recovered relative anchors
    (seed_rows), recover the combined anchors, keep the trailing foreground rows
    into u's block. Anchor-word spectral recovery (Arora et al. 2013); the random
    projection preserves the residual-norm geometry the greedy anchor search needs
    (Johnson–Lindenstrauss).

    `topo_order` picks the deflation direction, mirroring the dense
    `spectral_block_aligned_lambda` (see `_node_order_and_relatives`): "forward"
    (default) processes nodes ancestors-first and deflates u against its proper
    ANCESTORS (u's topic = its increment over its ancestors); "reverse" processes
    leaves-first and deflates u against its proper DESCENDANTS (leaves claim their
    full signal first; u's topic = the residual after its descendants).

    Returns a (K, V) λ = block-aligned β * scale, the same contract as the dense
    function (a drop-in seed). Numerically identical to the single-pass
    all-groups accumulation — same doc set and same per-word sketch rows per
    node, just materialized one node at a time (`filter` preserves partition
    membership and order, so the float32 accumulation is identical, not merely
    equivalent; a future repartition of `group_rdd` would break that).

    COST: this issues `n_nodes + 1` sequential passes, each a full scan of the
    cached corpus — fine for tens of nodes, but O(n_nodes) passes is slow for
    hundreds. The bounded-memory batching variant (recover B node slabs per pass
    via a non-empty `_NodeGroups(batch)`, batched within a depth level so no node
    shares a batch with an ancestor) is the follow-up for large DAGs.

    `anchor_scope` mirrors the dense function: "closure" (default) trains each
    node's sketch from every doc in its frontier closure and background from ALL
    docs; "frontier" trains node u only from docs where u is the most-specific
    attested node (u in gd.groups := the frontier itself) and background only from
    empty-frontier docs — so anchor selection cannot let background or a parent
    steal a descendant's defining word (see `_anchor_node_set`).

    UNLIKE `spectral_block_aligned_lambda`, this function takes NO `domain_bounds`
    and carries no per-domain candidate floor — and, measured, a multi-domain fit
    seeded here still recovers every node. `find_anchors_projected`'s candidate
    gate is an ABSOLUTE per-word document-frequency floor (`df_w >= min_doc_freq`,
    ADR 0032, adopted because the dense function's MEAN-RELATIVE marginal floor
    over-excludes rare-but-pure words under the sketch), not a pooled-mean-relative
    one — so there is no pooled mean for a denser domain to dominate.

    What is SETTLED, and what is not (corrected 2026-07-26 after a fuller test —
    an earlier version of this docstring overclaimed):
    `test_scalable_init_recovers_every_identifying_signal`
    (spark-vi/tests/test_gated_lda.py) fits the b_only-node two-domain plant from
    THIS seed across 8 projection draws and asserts post-EM recovery of every
    node's IDENTIFYING per-domain signal above a floor — and it passes. What is
    NOT claimed: that this seed matches or beats the dense+floor seed cell-for-cell
    (an 8-seed comparison showed the scalable seed is itself FRAGILE — an
    exclusive-node signal can be 0.0 at init on some draws), nor that the dense
    per-domain floor is redundant in general. The honest reading: on a
    WELL-SPECIFIED plant, recovery is carried by the gated EM, which refines a
    fragile scalable seed back up — not by the seed's own quality or by a floor
    analogue (consistent with insight 0067, where the dense floor's apparent value
    was itself a degenerate-plant artifact). The arc's SP3 blocker-1 question ("is
    the production scalable init adequate for a multi-domain fit?") is answered YES
    by that recovery gate; the stronger "immune / needs no floor, proven by a
    dense comparison" framing was a premature reading of a 3-seed lucky draw and is
    withdrawn."""
    from pyspark import StorageLevel
    from spark_vi.models.topic.spectral_init_scalable import (
        projected_cooccurrence_rdd, find_anchors_projected,
        recover_beta_projected, default_projection_dim,
        _row_normalize_projected,
    )
    _validate_anchor_scope(anchor_scope)
    if d is None:
        d = default_projection_dim(lay.K, V)

    lay_b = rdd.context.broadcast(lay)

    def _to_group(doc, _lay=lay_b, _scope=anchor_scope):
        L = _lay.value
        # frozenset so each node's `_u in gd.groups` filter is O(1); computed once
        # here and cached so the per-node passes never recompute the node set.
        return _GroupDoc(doc.indices, doc.counts,
                         frozenset(_anchor_node_set(doc.frontier, L, _scope)))

    group_rdd = rdd.map(_to_group).persist(StorageLevel.MEMORY_AND_DISK)
    no_groups = _NodeGroups(())          # pooled-only passes (no per-group slabs)
    try:
        group_rdd.count()                # materialize the cache once

        beta = np.zeros((lay.K, V), dtype=np.float64)

        # Step 1: background block. "closure" pools over ALL docs; "frontier" pools
        # over only background docs (empty node set) so foreground can't seed bg.
        bg_rdd = (group_rdd if anchor_scope == "closure"
                  else group_rdd.filter(lambda gd: len(gd.groups) == 0))
        pooled = projected_cooccurrence_rdd(bg_rdd, no_groups, V, d, seed)
        bg_anchors = find_anchors_projected(
            pooled.pooled_QR, pooled.p_w, pooled.df_w, lay.n_bg,
            min_doc_freq=min_doc_freq)
        bg_beta = recover_beta_projected(pooled.pooled_QR, pooled.p_w, bg_anchors)
        for i in range(min(lay.n_bg, bg_beta.shape[0])):
            beta[i] = bg_beta[i]

        # Opt-in diagnostic (no effect on lambda): dump per-node effective rank
        # (data-driven K_v) off each node's own sketch. Runs only when
        # CHARM_PROBE_EFFRANK is set; CHARM_PROBE_EFFRANK_MAX overrides max_probe.
        probe_effrank = bool(os.environ.get("CHARM_PROBE_EFFRANK"))
        effrank_reports: dict[int, dict] = {}
        if probe_effrank:
            from spark_vi.models.topic.effective_rank import effective_rank_report
            try:
                _probe_max = int(os.environ.get("CHARM_PROBE_EFFRANK_MAX", "40"))
            except ValueError:
                _probe_max = 40

        # Step 2: each node, in `topo_order`, its OWN filtered one-slab pass.
        node_anchors: dict[int, list] = {}
        order, relatives = _node_order_and_relatives(lay, topo_order)
        for u in order:
            rdd_u = group_rdd.filter(lambda gd, _u=u: _u in gd.groups)
            res_u = projected_cooccurrence_rdd(rdd_u, no_groups, V, d, seed)
            if int(res_u.df_w.sum()) == 0:
                logger.warning(
                    "scalable_block_aligned_lambda: node %s has zero training "
                    "docs; its block stays at the 1e-9 floor (uninitialized).", u)
                continue
            anc = relatives(u)
            seed_rows = list(bg_anchors) + [a for p in anc
                                            for a in node_anchors.get(p, [])]
            if probe_effrank:
                # PROGRESSIVE (parent-)deflation: deflate node u's sketch against
                # background + u's already-recovered ANCESTOR anchors (the SAME
                # seed_rows the fit uses for u), so the rank measures u's phenotypic
                # INCREMENT over its ancestors -- NOT its whole comorbidity load
                # (which is shared background/parent structure and would otherwise
                # inflate every node's rank to the ceiling). Forward topo order
                # guarantees ancestors are recovered before u, so their anchors are
                # in node_anchors. (Reverse order deflates against descendants; the
                # increment interpretation follows topo_order either way.)
                _qbar_u = _row_normalize_projected(res_u.pooled_QR, res_u.p_w)
                _rep_u = effective_rank_report(
                    _qbar_u, _probe_max, seed_rows=seed_rows, tau=0.01)
                _rep_u["n_docs"] = int(res_u.n_docs)   # diversity-vs-volume readout
                effrank_reports[u] = _rep_u
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

        if probe_effrank:
            from spark_vi.models.topic.effective_rank import (
                log_effrank_table, save_effrank_sidecar,
            )
            print("", flush=True)
            log_effrank_table(effrank_reports, n_nodes=len(lay.nodes),
                              k_uniform=len(lay.nodes) * lay.tpn,
                              printer=lambda s: print(s, flush=True))
            # Persist a sidecar (node id -> report) so a post-fit readout can join
            # names + doc counts without re-parsing logs or re-running the fit. The
            # path is provided by the driver via CHARM_PROBE_EFFRANK_OUT (the fit's
            # out_dir); log-only if unset.
            _out = os.environ.get("CHARM_PROBE_EFFRANK_OUT")
            if _out:
                try:
                    save_effrank_sidecar(effrank_reports, _out)
                    print(f"[effrank] wrote sidecar -> {_out}", flush=True)
                except OSError as e:
                    print(f"[effrank] sidecar write failed ({e})", flush=True)
    finally:
        group_rdd.unpersist(blocking=False)

    beta = beta + 1e-9                                   # strictly positive λ
    return beta * float(scale)
