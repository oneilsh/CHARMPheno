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

    def __init__(self, parents, anchor, names=None, protected=None):
        self.parents = {c: sorted(set(ps)) for c, ps in parents.items() if c != anchor}
        self.anchor = anchor
        self.names = dict(names or {})
        # Nodes exempt from attestation pruning even below min_n — deliberate
        # structural scaffolding (disease anchors + inserted class nodes) that must
        # survive to provide the hierarchy/pooling even when rarely coded directly.
        self.protected = set(protected or ())
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

    def to_engine(self):
        """Remap concept ids to contiguous engine ids: anchor -> 0 (root), descendants -> 1..N in
        (depth, cid) order. Returns (parent_int, int2cid, cid2int); `parent_int` is the
        `{child: [parents]}` map that spark_vi's DagLayout consumes directly."""
        order = sorted((n for n in self.nodes() if n != self.anchor),
                       key=lambda c: (self.depth(c), c))
        cid2int = {self.anchor: 0}
        for i, c in enumerate(order, start=1):
            cid2int[c] = i
        int2cid = {i: c for c, i in cid2int.items()}
        parent_int = {cid2int[c]: [cid2int[p] for p in ps] for c, ps in self.parents.items()}
        return parent_int, int2cid, cid2int


def build_condition_dag(edges, anchor, node_ids, names=None, protected=None):
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
    dag = ConditionDag(parents, anchor,
                       {c: (names or {}).get(c, str(c)) for c in nodeset},
                       protected=protected)
    # Observability: a node orphaned here is indistinguishable in structure from a genuine depth-1
    # child, so surface the set — a large/unexpected orphan count signals an upstream edges/node_ids
    # extraction problem (e.g. a real parent filtered out as non-standard), not a real depth-1 node.
    dag.orphans = orphans
    return dag


def prune_by_attestation(dag, counts, min_n):
    """Drop every non-anchor node with fewer than `min_n` attesting patients; rewire each surviving
    node to its nearest surviving ancestors (transitive walk up past dropped nodes). The anchor is
    never dropped. This is the principled size cap: a node no cohort patient populates cannot have a
    learnable topic."""
    keep = {n for n in dag.nodes()
            if n == dag.anchor or n in dag.protected or counts.get(n, 0) >= min_n}
    # Rewire each survivor to its nearest surviving ancestors — the SAME walk the pruning ledger
    # uses to report where patients land, so the two can never disagree (single source of truth).
    new_parents = {c: sorted(_nearest_surviving_ancestors(dag, c, keep))
                   for c in keep if c != dag.anchor}
    return ConditionDag(new_parents, dag.anchor, dag.names, protected=dag.protected)


def _nearest_surviving_ancestors(dag, node, keep):
    """The kept nodes a dropped `node` rewires up to (transitive walk past dropped ancestors) — the
    same landing set `prune_by_attestation` reattaches its descendants to."""
    surv, seen, stack = set(), set(), list(dag.parents.get(node, []))
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        if p in keep:
            surv.add(p)
        else:
            stack.extend(dag.parents.get(p, []))
    return surv or {dag.anchor}


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
                # where the patient actually lands after pruning: surviving frontier members plus
                # the nearest surviving ancestors its dropped members rewire up to (NOT the anchor).
                landing = {c for c in fr if c in kept}
                for c in dfr:
                    landing |= _nearest_surviving_ancestors(before, c, kept)
                # measure both depths in the ORIGINAL-ontology (before) frame, so the drop reads as
                # "levels rolled up in the original ontology" and isn't confounded by the landing
                # node itself losing ancestors elsewhere in the prune.
                worst = max(before.depth(c) for c in dfr)
                landed = max((before.depth(a) for a in landing), default=0)
                drops.append(worst - landed)
        n = len(cohort_frontiers)
        led["coarsening_rate"] = coarsened / n if n else 0.0
        led["mean_depth_drop"] = (sum(drops) / len(drops)) if drops else 0.0
    return led
