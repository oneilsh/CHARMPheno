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


def prune_by_attestation(dag, counts, min_n):
    """Drop every non-anchor node with fewer than `min_n` attesting patients; rewire each surviving
    node to its nearest surviving ancestors (transitive walk up past dropped nodes). The anchor is
    never dropped. This is the principled size cap: a node no cohort patient populates cannot have a
    learnable topic."""
    keep = {n for n in dag.nodes() if n == dag.anchor or counts.get(n, 0) >= min_n}
    new_parents = {}
    for c in keep:
        if c == dag.anchor:
            continue
        surv, seen, stack = set(), set(), list(dag.parents.get(c, []))
        while stack:
            p = stack.pop()
            if p in seen:
                continue
            seen.add(p)
            if p in keep:
                surv.add(p)
            else:
                stack.extend(dag.parents.get(p, []))
        new_parents[c] = sorted(surv) if surv else [dag.anchor]
    return ConditionDag(new_parents, dag.anchor, dag.names)


def pruning_ledger(before, after, counts, *, cohort_frontiers=None):
    """A receipt for what pruning discarded. Structural stats need only the two DAGs + counts:
    kept/dropped totals, breakdown by (pre-prune) depth, resulting K (= engine topic-count driver),
    and the smallest kept count. When `cohort_frontiers` (per-patient most-specific attested
    concept-id sets) is supplied, also report the coarsening rate (fraction of patients whose
    most-specific node was pruned, so their frontier rolled up) and the mean depth drop for them."""
    kept = after.nodes()
    dropped = before.nodes() - kept
    led = {"kept": len(kept), "dropped": len(dropped), "K_nodes": len(kept),
           "kept_by_depth": dict(sorted(Counter(before.depth(n) for n in kept).items())),
           "dropped_by_depth": dict(sorted(Counter(before.depth(n) for n in dropped).items())),
           "min_count_kept": min((counts.get(n, 0) for n in kept if n != before.anchor), default=0)}
    if cohort_frontiers is not None:
        coarsened, drops = 0, []
        for fr in cohort_frontiers:
            dfr = [c for c in fr if c in dropped]
            if dfr:
                coarsened += 1
                worst = max(before.depth(c) for c in dfr)
                aft = max((after.depth(c) for c in fr if c in kept), default=0)
                drops.append(worst - aft)
        n = len(cohort_frontiers)
        led["coarsening_rate"] = coarsened / n if n else 0.0
        led["mean_depth_drop"] = (sum(drops) / len(drops)) if drops else 0.0
    return led
