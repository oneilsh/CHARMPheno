"""Data-driven gate seeds for the DAG-gated LDA engine (`DagGatedOnlineLDA`).

The engine's lambda starts from its random Gamma draw; under the leaves-first gate
schedule (`set_gate_schedule`) each node block is additionally seeded from the
documents its gate admits, the moment it opens (and, with `leaves=True`, the leaf
blocks at iteration 0). This module builds those seed rows in one distributed pass
over the cached document RDD (ADR 0047 closure discipline). The DAG spectral
(anchor-word) init that used to live here was removed by ADR 0050; the flat
anchor-word init for the STM stays in `spectral_init.py`.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _destroy_broadcast(bcast) -> None:
    """Fully release a broadcast: executor blocks, the driver block-manager copy,
    AND the driver-local pickled temp file.

    ``unpersist`` frees only the executor copies, so a broadcast that is merely
    unpersisted (or never cleaned) leaves one driver-side pickle behind — the leak
    class that filled the master's disk and killed a run with ENOSPC (exp 0104; see
    ``core/runner._destroy_broadcast``). Cleanup must never outrank the work it
    follows, so failures degrade to unpersist and a log line rather than raising."""
    try:
        bcast.destroy()
    except Exception as exc:
        try:
            bcast.unpersist(blocking=False)
        except Exception:
            pass
        logger.warning("broadcast destroy fell back to unpersist: %s", exc)


def _seed_reduce_depth(n_rows: int, V: int) -> int:
    """treeReduce depth for a (n_rows, V) float32 accumulator: the driver receives
    about P^(1/depth) partials, so a whole-Mondo-sized array (hundreds of MB) needs a
    deeper tree than a branch's few MB."""
    return 4 if n_rows * V * 4 > 50_000_000 else 2


def gate_open_seed_counts(rdd, lay, V: int, *, mode: str = "subtree",
                          own_idx: dict | None = None, leaves: bool = False):
    """The leaves-first schedule's open-time seed: a (K, V) array whose rows for each
    LATE-OPENING block (node height >= 1) hold the summed word counts of the
    documents whose gate admits that block — u in closure(frontier), i.e. a frontier
    node in u's subtree, following every child edge on a multi-parent DAG — and
    zeros elsewhere (leaves and background open at iteration 0 and need no seed).

    Why counts, not a fit: the seed only has to make a block competitive when it
    opens (`GatedOnlineLDA.set_gate_schedule`); the gated EM then strips what the
    already-sharp children and background explain. The cost is one pass of the
    E-step's own sparse scatter without CAVI. ADR 0047: the layout rides a broadcast,
    per-partition accumulators are allocated executor-side, the reduction identity is
    a None sentinel, and the broadcast is destroyed after the pass.
    `rdd` items carry `.indices`, `.counts`, `.frontier` (GatedPCDocument /
    GatedBOWDocument).

    `mode="balanced"` (exp 0160): a parent with one dominant child would otherwise
    open looking like that child (exp 0154: intrinsic cardiomyopathy took HCM's
    words). The row is instead the EQUAL-weight mean of each child's normalized
    subtree word distribution plus one share for the documents coded at the parent
    itself (frontier contains it), so a word specific to one child weighs about
    1/(children + 1) and a word all children share keeps full weight. `own_idx`
    ({node: [vocab idx]}, the native code map's attesting codes) removes every
    proper descendant's own codes from the parent's row before normalization. This
    accumulates float32 (rows, V) arrays per partition — the late nodes' children and
    the late nodes only — with a deeper reduction tree when they are large, so a
    whole-Mondo pass stays at about 200 MB a task and well under 1 GB on the driver.

    `leaves=True` (exp 0162) also fills the LEAF rows (height 0) with the word counts
    of the documents each leaf's gate admits, its own patients, for the estimator to
    apply at iteration 0. With every node block seeded from its own documents, the
    spectral init has nothing left to do under the schedule (`init: random`), and its
    per-node batched scans (hours at whole-Mondo) are skipped."""
    if mode not in ("subtree", "balanced"):
        raise ValueError(f"unknown seed mode {mode!r}")
    if mode == "balanced":
        return _balanced_seed(rdd, lay, V, own_idx or {}, leaves=leaves)
    late = [u for u in lay.nodes if lay.height(u) >= 1 or leaves]
    row_of = {u: i for i, u in enumerate(late)}
    out = np.zeros((lay.K, V), dtype=np.float64)
    if not late:
        return out
    lay_b = rdd.context.broadcast(lay)
    rows_b = rdd.context.broadcast(row_of)

    def _part(docs, _lay=lay_b, _rows=rows_b):
        L, R = _lay.value, _rows.value
        acc = None
        for d in docs:
            if not d.frontier:
                continue
            us = {u for f in d.frontier for u in L.closure(f) if u in R}
            if not us:
                continue
            if acc is None:
                acc = np.zeros((len(R), V), dtype=np.float32)
            idx = np.asarray(d.indices, dtype=np.int64)
            cnt = np.asarray(d.counts, dtype=np.float32)
            for u in us:
                acc[R[u], idx] += cnt
        yield acc

    def _comb(a, b):
        if a is None:
            return b
        if b is None:
            return a
        a += b
        return a

    try:
        tot = rdd.mapPartitions(_part).treeReduce(
            _comb, depth=_seed_reduce_depth(len(late), V))
    finally:
        _destroy_broadcast(lay_b)
        _destroy_broadcast(rows_b)
    if tot is not None:
        for u, i in row_of.items():
            out[lay.block[u]] = tot[i]
    return out


def _balanced_seed(rdd, lay, V: int, own_idx: dict, *, leaves: bool = False):
    """`gate_open_seed_counts(mode="balanced")`: one pass accumulating, per node, the
    word counts of its subtree's documents (S) and of the documents coded at the node
    itself (O); the driver combines them child-balanced. Same ADR 0047 discipline."""
    nodes = list(lay.nodes)
    out = np.zeros((lay.K, V), dtype=np.float64)
    late = [u for u in nodes if lay.height(u) >= 1]
    leaf = [u for u in nodes if lay.height(u) == 0] if leaves else []
    if not late and not leaf:
        return out
    # Only the rows the seed reads: S (subtree counts) for the children of late
    # nodes (and every leaf when `leaves`), O (own coded documents) for the late
    # nodes. float32 accumulators.
    kids = sorted({c for u in late for c in lay.children.get(u, []) if c != 0}
                  | set(leaf))
    row_of = {u: i for i, u in enumerate(kids)}           # S rows
    orow_of = {u: i for i, u in enumerate(late)}          # O rows
    lay_b = rdd.context.broadcast(lay)
    rows_b = rdd.context.broadcast((row_of, orow_of))

    def _part(docs, _lay=lay_b, _rows=rows_b):
        L, (R, Ro) = _lay.value, _rows.value
        acc = None
        for d in docs:
            if not d.frontier:
                continue
            if acc is None:
                acc = (np.zeros((len(R), V), dtype=np.float32),
                       np.zeros((len(Ro), V), dtype=np.float32))
            idx = np.asarray(d.indices, dtype=np.int64)
            cnt = np.asarray(d.counts, dtype=np.float32)
            for u in {u for f in d.frontier for u in L.closure(f) if u in R}:
                acc[0][R[u], idx] += cnt
            for f in d.frontier:
                if f in Ro:
                    acc[1][Ro[f], idx] += cnt
        yield acc

    def _comb(a, b):
        if a is None:
            return b
        if b is None:
            return a
        a[0][:] += b[0]
        a[1][:] += b[1]
        return a

    try:
        tot = rdd.mapPartitions(_part).treeReduce(
            _comb, depth=_seed_reduce_depth(len(kids) + len(late), V))
    finally:
        _destroy_broadcast(lay_b)
        _destroy_broadcast(rows_b)
    if tot is None:
        return out
    S, O = (t.astype(np.float64) for t in tot)

    def norm(v):
        t = v.sum()
        return v / t if t > 0 else None

    for u in late:
        drop = set()
        for c in lay.descendants(u):
            drop.update(int(i) for i in own_idx.get(c, ()))
        drop = np.asarray(sorted(i for i in drop if 0 <= i < V), dtype=np.int64)
        shares = []
        for c in lay.children.get(u, []):
            if c in row_of:
                v = S[row_of[c]].copy()
                v[drop] = 0.0
                n = norm(v)
                if n is not None:
                    shares.append(n)
        v = O[orow_of[u]].copy()
        v[drop] = 0.0
        n = norm(v)
        if n is not None:
            shares.append(n)
        if shares:
            out[lay.block[u]] = np.mean(shares, axis=0)
    for u in leaf:
        v = S[row_of[u]]
        if v.sum() > 0:
            out[lay.block[u]] = v
    return out
