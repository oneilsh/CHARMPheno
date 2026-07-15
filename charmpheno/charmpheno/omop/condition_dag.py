"""Build the anchor-first label DAG for hierarchical placement from OMOP concept_ancestor.

Pure transformation over a small in-memory edge list in concept-id space, plus an attestation
prune (the principled size cap: a node no patient populates has no learnable topic) and a pruning
ledger that makes the granularity cost of pruning visible. `ConditionDag.to_engine()` remaps to the
integer `{child: [parents]}` map consumed by spark_vi.models.topic.dag_placement.DagLayout.

See docs/superpowers/specs/2026-07-15-condition-dag-builder-design.md."""
from collections import defaultdict, Counter


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


def build_condition_dag(edges, anchor, node_ids, names=None):
    """From min-sep-1 (ancestor, descendant) edges restricted to `node_ids` (standard-condition
    descendants incl. the anchor), assemble the multi-parent parent map. A node with no in-set
    parent (orphan) attaches to the anchor so the DAG is connected and rooted."""
    nodeset = set(node_ids) | {anchor}
    parents = defaultdict(list)
    for a, d in edges:
        if a in nodeset and d in nodeset and a != d:
            parents[d].append(a)
    for c in nodeset:
        if c != anchor and c not in parents:
            parents[c] = [anchor]
    return ConditionDag(parents, anchor, {c: (names or {}).get(c, str(c)) for c in nodeset})
