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

from spark_vi.dag.ops import closure_matrix  # noqa: F401  (re-exported: the product's M)


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


