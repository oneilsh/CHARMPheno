"""DAG/ontology-structured additive mean-offset layer for the gated PG-STM (v1).

A document's mean logits gain an additive term summed over its ancestral closure in
an is-a DAG: mu_d = Gamma^T x_d + sum_{u in closure(v_d)} eta_u. v1 realizes eta_u as a
MEAN SHIFT only (nodes own no new topics), so the model is the existing PG-STM with an
augmented covariate w_d = [x_d ; closure_indicator_d] and coefficient [Gamma ; B], plus
a depth-scaled ridge penalty on B. See docs/superpowers/plans/2026-07-13-dag-ontology-
pg-stm-model-core.md and the spec it implements.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


class DagGate:
    """An is-a DAG over group nodes. Node 0 is the root (background). Parent indices
    must be strictly less than the child index (topological order => acyclic). Closures
    are SETS, so a diamond's shared ancestor is counted once. Depth is the shortest
    root distance (used by the depth-scaled offset prior)."""

    def __init__(self, parents: Sequence[Sequence[int]]):
        self.parents: list[tuple[int, ...]] = [tuple(int(p) for p in ps) for ps in parents]
        self.n_nodes = len(self.parents)
        if self.n_nodes == 0 or self.parents[0] != ():
            raise ValueError("node 0 must be the root with no parents")
        for u, ps in enumerate(self.parents):
            for p in ps:
                if not (0 <= p < u):
                    raise ValueError(f"parent {p} of node {u} must satisfy 0 <= p < {u}")
        self._anc: list[frozenset[int]] = []
        for u, ps in enumerate(self.parents):
            acc: set[int] = set()
            for p in ps:
                acc.add(p)
                acc |= self._anc[p]
            self._anc.append(frozenset(acc))
        self.depth = self._compute_depth()

    def _compute_depth(self) -> np.ndarray:
        depth = np.zeros(self.n_nodes, dtype=np.int64)
        for u in range(self.n_nodes):
            if self.parents[u]:
                depth[u] = 1 + min(depth[p] for p in self.parents[u])
        return depth

    def ancestors(self, u: int) -> frozenset[int]:
        return self._anc[u]

    def closure(self, nodes) -> frozenset[int]:
        out: set[int] = set()
        for v in nodes:
            out.add(int(v))
            out |= self._anc[int(v)]
        return frozenset(out)

    def closure_indicator(self, nodes) -> np.ndarray:
        z = np.zeros(self.n_nodes, dtype=np.float64)
        for u in self.closure(nodes):
            z[u] = 1.0
        return z

    def dump(self) -> list[dict]:
        return [{"node": u, "depth": int(self.depth[u]), "parents": list(self.parents[u])}
                for u in range(self.n_nodes)]


def offset_penalty(P, dag, *, gamma_ridge, lam_base, gamma_depth):
    """(P + n_nodes,) ridge penalty: ``gamma_ridge`` on each covariate row, and
    ``lam_base * (1 + depth[u]) ** gamma_depth`` on each node-offset row. Depth-scaling
    (deeper => larger penalty) encodes "prefer general explanations, specialize only on
    evidence" (a structural, inspectable shrinkage; not an inference hyperparameter).
    A node whose closure-indicator column is never active is pulled to 0 by its penalty."""
    pen = np.empty(int(P) + dag.n_nodes, dtype=np.float64)
    pen[:P] = float(gamma_ridge)
    pen[P:] = float(lam_base) * (1.0 + dag.depth.astype(np.float64)) ** float(gamma_depth)
    return pen


def dag_offset_ridge(WtW, WtM, *, penalty):
    """Penalized moment-form ridge: solve (WtW + diag(penalty)) C = WtM. Generalizes
    pg_gamma_ridge_moments' scalar ridge to a per-row penalty vector, so covariate and
    node-offset rows are shrunk independently (depth-scaled). WtW is (P+U, P+U), WtM is
    (P+U, K-1)."""
    WtW = np.asarray(WtW, dtype=np.float64)
    WtM = np.asarray(WtM, dtype=np.float64)
    return np.linalg.solve(WtW + np.diag(np.asarray(penalty, dtype=np.float64)), WtM)
