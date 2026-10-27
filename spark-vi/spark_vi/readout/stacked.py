"""The closure product over the per-node heads (spec 2026-10-06 stacked readout).

Under a closure label mask node c's head is fit on the rows inside its parent's
closure with siblings as negatives, so σ(z_c) ≈ P(c | parent(c)). The product down
the closure,

    log P_stack(c | x) = Σ_{a ∈ closure(c)} log σ(z_a(x)),

closure = c plus every ancestor including the root, each counted once (diamond-safe),
gives P(child) ≤ P(parent) by construction and carries the root head's
case-vs-background prior down the DAG -- provided the root head saw the background
(the caller observes the root on every train row). Pure numpy; the DAG is a
`{child: [parents]}` map over label indices [0, C).
"""
from __future__ import annotations

import numpy as np


def dag_closure_parents(parent_int, C):
    """Length-C list of parent-LABEL-index lists over the [0, C) engine-id label
    space (root and any parentless node -> []): `parent_int` (root omitted)
    densified over range(C)."""
    return [[int(p) for p in parent_int.get(c, [])] for c in range(C)]


def closure_matrix(parent_int, C):
    """(C, C) float64 `M[c, a] = 1` iff `a ∈ closure(c)` — c itself plus every
    ancestor, each ONCE under diamonds, root included.

    A memoized DFS over the direct-parent lists. `parent_int` is the bundle's
    `{child: [parents]}` over engine ids."""
    C = int(C)
    parents = dag_closure_parents(parent_int, C)
    clo = [None] * C

    def _closure(node):
        if clo[node] is None:
            acc = {node}
            for p in parents[node]:
                if 0 <= p < C and p != node:
                    acc |= _closure(p)
            clo[node] = acc
        return clo[node]

    M = np.zeros((C, C), dtype=np.float64)
    for c in range(C):
        for a in _closure(c):
            M[c, a] = 1.0
    return M


def stacked_proba(proba, M, *, chunk_rows=8192, floor=1e-12):
    """`P_stack = exp(log(proba) @ M.T)` — the closure product per (doc, node).

    Row-chunked so the (D, C) log matrix never exists in full beside the float32
    input (at whole-Mondo, D_te×C float64 is ~2 GB). `floor` clips the input
    probabilities before the log: a degenerate-negative node's constant fallback
    is exactly 0.0, and log(0) would turn every descendant's product into 0
    (a constant column) through a single ancestor the solver refused to fit.
    Returns float32 (the lean collect's own dtype)."""
    proba = np.asarray(proba)
    D, C = proba.shape
    MT = np.asarray(M, dtype=np.float64).T
    out = np.empty((D, C), dtype=np.float32)
    for s in range(0, D, int(chunk_rows)):
        blk = np.log(np.clip(proba[s:s + chunk_rows].astype(np.float64),
                             floor, 1.0))
        out[s:s + chunk_rows] = np.exp(blk @ MT)
    return out


