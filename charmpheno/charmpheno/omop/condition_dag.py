"""Build the anchor-first SNOMED label DAG from OMOP concept_ancestor.

Pure transformation over a small in-memory edge list in concept-id space.
`ConditionDag.user_dag()` hands it to the DAG-gated estimator in the estimator's
user-id contract (spec 2026-10-30 R1): `{concept_id: [parent concept_ids]}`, int ids
(OMOP concept ids are ints, and the bundle's `labels` column is `array<bigint>`).
Which nodes get a topic block (the attestation prune that used to live here,
`prune_by_attestation`, and the engine renumbering `to_engine`) is the estimator's,
learned from the training documents (`minDocsPerNode`).

See docs/superpowers/specs/2026-07-15-condition-dag-builder-design.md."""
from collections import defaultdict


class ConditionDag:
    """Multi-parent condition DAG in concept-id space, rooted at `anchor`. `parents` maps a
    non-anchor concept id to its list of parent concept ids (the anchor has no entry)."""

    def __init__(self, parents, anchor, names=None):
        self.parents = {c: sorted(set(ps)) for c, ps in parents.items() if c != anchor}
        self.anchor = anchor
        self.names = dict(names or {})
        self._depth = {}

    def nodes(self):
        return {self.anchor} | set(self.parents.keys())

    def children(self):
        ch = defaultdict(list)
        for c, ps in self.parents.items():
            for p in ps:
                ch[p].append(c)
        return {p: sorted(cs) for p, cs in ch.items()}

    def depth(self, cid, _stack=()):
        """Longest path length from the anchor to `cid` (anchor = 0). Memoized; cycle-guarded."""
        if cid in self._depth:
            return self._depth[cid]
        ps = [p for p in self.parents.get(cid, []) if p != cid and p not in _stack]
        d = 0 if (cid == self.anchor or not ps) else 1 + max(self.depth(p, _stack + (cid,)) for p in ps)
        self._depth[cid] = d
        return d

    def user_dag(self, *, synthetic_root=None):
        """The DAG in the estimator's user-id contract: `{node: [parents]}` over every
        node, `[]` for a top-level node (the estimator hangs it from its own synthetic
        root).

        `synthetic_root` names an anchor that is NOT a real concept (the multi-disease
        forest root, `case_finding_assembly._FOREST_ROOT_CID`): it is dropped, and its
        children become top-level. A real anchor (a single-disease DAG) stays a node,
        top-level, with a topic block of its own; the estimator's root is above it."""
        out = {}
        for n in sorted(self.nodes()):
            if synthetic_root is not None and n == synthetic_root:
                continue
            out[n] = sorted(p for p in self.parents.get(n, [])
                            if synthetic_root is None or p != synthetic_root)
        return out


def build_condition_dag(edges, anchor, node_ids, names=None):
    """From min-sep-1 (ancestor, descendant) edges restricted to `node_ids` (standard-condition
    descendants incl. the anchor), assemble the multi-parent parent map. A node with no in-set
    parent (orphan) attaches to the anchor so the DAG is connected and rooted."""
    nodeset = set(node_ids) | {anchor}
    parents = defaultdict(list)
    for a, d in edges:
        if a in nodeset and d in nodeset and a != d:
            parents[d].append(a)
    orphans = set()
    for c in nodeset:
        if c != anchor and c not in parents:
            parents[c] = [anchor]
            orphans.add(c)                              # no in-set parent -> attached to anchor
    dag = ConditionDag(parents, anchor, {c: (names or {}).get(c, str(c)) for c in nodeset})
    # Observability: a node orphaned here is indistinguishable in structure from a genuine depth-1
    # child, so surface the set — a large/unexpected orphan count signals an upstream edges/node_ids
    # extraction problem (e.g. a real parent filtered out as non-standard), not a real depth-1 node.
    dag.orphans = orphans
    return dag
