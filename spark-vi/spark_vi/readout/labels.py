"""Per-node readout targets from a document's DAG labels (spec 2026-10-30 R6).

The heads (`readout.heads.fit_heads`) read a dense target per document: `label (C,)`,
which nodes the document belongs to, and `labelMask (C,)`, which nodes it is evidence
for. This module builds both from what a user has -- the document's labels in their
own node ids -- and a learned node set (`spark_vi.dag.nodeset.NodeSet`), so the dense
vectors are no longer an input contract.

Output index space. Index 0 is the synthetic ROOT (the "any label" head: positive on
every foreground document, negative on background), indices 1..N the kept nodes in
the node set's engine order (`NodeSet.user_ids[i]` is index i's node). The index
space therefore equals the DAG-gated topic model's engine ids, so heads fit on the
model's node set align with its topic blocks, and `readout.evaluate` / `stacked`
keep reading column 0 as the root.

The label. A document's labels land on the kept node set (`FrontierFn.landing`: a
kept label stays, a pruned or spliced one moves to its nearest kept ancestors) and
the target is the CLOSURE of the landed nodes:

    label[c] = 1  iff  c is a landed node or an ancestor of one (root included).

So a document labelled at a leaf is a positive for every node above it, which is
what makes node c's head a "c or anything under c" classifier, and an empty or null
label array (a background document) is the all-zero label.

The mask (`mask_mode`, the observation policy):

- `"full"` (default): every node is observed on every document; a background
  document is a negative everywhere. Coded absence is read as true absence.
- `"closure"`: observe only the label's closure (the positives) and the DAG
  siblings of each positive (the other children of its parents: the near-boundary
  negatives); distant nodes stay unobserved. A background document observes
  nothing. Under this policy node c's head is fit on the rows inside its parents'
  closure, so sigma(z_c) estimates P(c | parent), which is what the closure product
  (`readout.stacked`) multiplies.

Both policies are the ones the clinical assembler applied to its dense columns
before this module existed, on the same engine-id node space.

`NodeLabeler` is small and picklable (dicts and tuples, no arrays: ADR 0047): the
Spark path broadcasts it and calls it per row; the local path calls `dense` on a
list of label arrays and hands the result to `LocalReadoutData.from_arrays`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spark_vi.dag.nodeset import (_UNKNOWN_CAP, UNKNOWN_LABEL_POLICIES, _coerce_id,
                                  _iter_labels, _unknown_error)

LABEL_MASK_MODES = ("full", "closure")


@dataclass(frozen=True)
class NodeLabeler:
    """A document's labels (user ids) -> its per-node target over indices 0..N.

    `landing`: user id -> tuple of indices it lands on (every DAG node is a key);
    `closure`: `closure[i]`, the indices of i and all its ancestors (root included);
    `siblings`: `siblings[i]`, the children of i's parents (i included; empty for the
    root); `mask_mode` and `unknown_labels` as in the module docstring and
    `spark_vi.dag.nodeset`."""
    landing: dict
    closure: tuple
    siblings: tuple
    mask_mode: str = "full"
    unknown_labels: str = "error"

    @property
    def C(self) -> int:
        return len(self.closure)

    @classmethod
    def from_node_set(cls, ns, *, mask_mode="full", unknown_labels=None):
        """From a learned `NodeSet`. `unknown_labels` defaults to the node set's own
        policy (`ns.frontier.unknown_labels`)."""
        if mask_mode not in LABEL_MASK_MODES:
            raise ValueError(f"labelMaskMode must be one of {LABEL_MASK_MODES}; got "
                             f"{mask_mode!r}")
        policy = ns.frontier.unknown_labels if unknown_labels is None else unknown_labels
        if policy not in UNKNOWN_LABEL_POLICIES:
            raise ValueError(f"unknownLabels must be one of {UNKNOWN_LABEL_POLICIES}; "
                             f"got {policy!r}")
        n = len(ns.user_ids) - 1
        parents = {0: []}
        parents.update({int(c): [int(p) for p in ps]
                        for c, ps in ns.engine_parents.items()})
        children = {i: [] for i in range(n + 1)}
        for c, ps in parents.items():
            for p in ps:
                children[p].append(c)
        clo: dict = {}

        def _closure(i):
            if i not in clo:
                acc = {i}
                for p in parents[i]:
                    acc |= _closure(p)
                clo[i] = frozenset(acc)
            return clo[i]

        # Engine ids are topological (parents before children), so a forward sweep
        # keeps the recursion depth at one level.
        for i in range(n + 1):
            _closure(i)
        sib = tuple(tuple(sorted({s for p in parents[i] for s in children[p]}))
                    for i in range(n + 1))
        return cls(landing={k: tuple(int(e) for e in v)
                            for k, v in ns.frontier.landing.items()},
                   closure=tuple(tuple(sorted(clo[i])) for i in range(n + 1)),
                   siblings=sib, mask_mode=mask_mode, unknown_labels=policy)

    def active(self, labels) -> tuple:
        """The sorted indices whose label is 1: the closure of the landed labels
        (empty for a background document)."""
        landed, unknown = set(), []
        for lab in _iter_labels(labels):
            try:
                tgt = self.landing.get(_coerce_id(lab, what="label"))
            except TypeError:
                tgt = None
            if tgt is None:
                unknown.append(lab)
            else:
                landed.update(tgt)
        if unknown and self.unknown_labels == "error":
            raise _unknown_error(unknown[:_UNKNOWN_CAP], len(unknown))
        act = set()
        for e in landed:
            act.update(self.closure[e])
        return tuple(sorted(act))

    def observed(self, active) -> tuple | None:
        """The observed indices for an active set; None means every index."""
        if self.mask_mode == "full":
            return None
        obs = set(active)
        for c in active:
            obs.update(self.siblings[c])
        return tuple(sorted(obs))

    def __call__(self, labels):
        """`(positive indices, observed indices or None for all)`."""
        act = self.active(labels)
        return act, self.observed(act)

    def dense(self, docs_labels):
        """`(label (D, C), mask (D, C))` float64 for a list of per-document label
        arrays: the local backend's input (`LocalReadoutData.from_arrays`)."""
        docs_labels = list(docs_labels)
        C = self.C
        label = np.zeros((len(docs_labels), C))
        mask = np.zeros((len(docs_labels), C))
        for d, labs in enumerate(docs_labels):
            act, obs = self(labs)
            label[d, list(act)] = 1.0
            if obs is None:
                mask[d] = 1.0
            else:
                mask[d, list(obs)] = 1.0
        return label, mask
