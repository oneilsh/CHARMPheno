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

import numpy as np

from spark_vi.models.topic.dag_placement import _as_counts
from spark_vi.models.topic.spectral_init import (
    word_cooccurrence, find_anchors, recover_beta,
)


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
            continue
        Q_u = word_cooccurrence(docs_u, V)
        anc = [a for a in lay.closure(u) if a not in (u, 0)]
        seed = list(bg_anchors) + [a for p in anc for a in node_anchors.get(p, [])]
        fg_anchors = find_anchors(Q_u, lay.tpn, seed_rows=seed)
        if not fg_anchors:
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
