"""Generic label-DAG operations over `{child: [parents]}` (or `{parent: [children]}`) maps.

Closure, descendants, depth, the transitive reduction of an order restricted to a
kept node set, and where an attestation of a dropped node lands. The node ids are any
hashables; nothing here knows what a node means (an ontology-specific builder in
the consuming package calls these).
"""
from __future__ import annotations

import numpy as np


def parent_adjacency(child_adj) -> dict:
    """`{child: [parents]}` from a `{parent: [children]}` adjacency (e.g. an
    ontology's is-a graph). Parents are deduped and sorted so every downstream walk
    is deterministic."""
    out: dict = {}
    for parent, children in child_adj.items():
        for c in children:
            out.setdefault(c, []).append(parent)
    return {c: sorted(set(ps)) for c, ps in out.items()}


def ancestor_closure(term, parent_adj) -> set:
    """The REFLEXIVE is-a closure of `term`: itself plus every transitive parent.
    Set-based, so a diamond is visited once and a (malformed) cycle terminates."""
    out = {term}
    stack = list(parent_adj.get(term, ()))
    while stack:
        p = stack.pop()
        if p in out:
            continue
        out.add(p)
        stack.extend(parent_adj.get(p, ()))
    return out


def _nearest_kept_above(term, kept, parent_adj) -> set:
    """The first kept node on every upward branch from `term` (`term` itself
    excluded): the walk stops climbing a branch at its first kept node."""
    found, seen = set(), set()
    stack = list(parent_adj.get(term, ()))
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        if cur in kept:
            found.add(cur)
        else:
            stack.extend(parent_adj.get(cur, ()))
    return found


def nearest_kept_ancestors(term, kept, parent_adj, memo=None) -> set:
    """Where an attestation of `term` LANDS in a DAG built over `kept`: `{term}`
    when it is kept, else the nearest kept ancestor on each upward branch (the walk
    stops climbing a branch at its first kept node), else `set()`.

    The reflexive twin of `nearest_kept_parents` (the same walk), so an attestation
    lands exactly where the label DAG reattaches its node. Composing it over
    successive prunings is why resolving codes against the FINAL node set is
    correct: nearest kept ancestor of nearest kept ancestor is nearest kept
    ancestor."""
    if term in kept:
        return {term}
    if memo is not None and term in memo:
        return memo[term]
    found = _nearest_kept_above(term, kept, parent_adj)
    if memo is not None:
        memo[term] = found
    return found


def transitive_reduction_parents(kept, parent_adj) -> dict:
    """The TRANSITIVE REDUCTION of the order `parent_adj` defines, restricted to `kept`.

    `nearest_kept_parents` is the first half: for each kept term, the first kept
    node on every upward branch. On a multi-path DAG that is NOT a transitive
    reduction. Worked example (from this module's unit tests):

        A                A, C, D kept; B dropped.
       / \\               D's parents are B and C. Climbing D's LEFT branch stops
      B   C              at A (past the dropped B); the RIGHT branch stops at C.
       \\ /               nearest_kept_parents(D) = {A, C}.
        D

    A is then a parent of both C and D, so C and D are SIBLINGS — and C is an
    ANCESTOR of D. Under closure masking a doc that fires D observes its siblings
    under every parent, C among them, and C is positive on every such doc: an
    all-positive cell and a constant head. It appears exactly when an intermediate
    is dropped, i.e. on every pruned rung of a real ontology.

    So the second half: a parent `p` is REDUNDANT when some other parent `q` of
    the same node has `p` among its own ancestors. Redundant parents are dropped,
    which is what "the transitive reduction deletes the redundant sibling edge"
    means operationally. Ancestry is read from the FULL graph, which is
    sound because the induced order on kept nodes is the restriction of the full
    one: if `p` is an ancestor of `q` in the full graph and both are kept, the edge `p->c`
    carries no information the path `p ... q -> c` does not.

    Multi-parenthood survives untouched wherever the parents are genuinely
    incomparable (orthogonal axes of one ontology, common in real ontologies):
    only comparable pairs are reduced. Returns `{kept term: [kept parents]}`; a term with no kept
    ancestor gets `[]` and attaches to the synthetic root."""
    kept = set(kept)
    nearest = nearest_kept_parents(kept, parent_adj)
    memo: dict = {}

    def strict_ancestors(term):
        if term not in memo:
            memo[term] = ancestor_closure(term, parent_adj) - {term}
        return memo[term]

    out: dict = {}
    for child, parents in nearest.items():
        if len(parents) <= 1:
            out[child] = list(parents)
            continue
        out[child] = sorted(
            p for p in parents
            if not any(p in strict_ancestors(q) for q in parents if q != p))
    return out


def nearest_kept_parents(mapped_ids: set, parent_adj: dict) -> dict:
    """For each mapped term, its nearest ANCESTOR terms that are themselves mapped
    (collapsing unmapped intermediates), over a child->parents adjacency. A term
    with no mapped ancestor gets ``[]`` (attaches to the synthetic root).

    "Nearest" = the first mapped node on every upward path (see
    `transitive_reduction_parents` for why this alone is not the reduction)."""
    return {node: sorted(_nearest_kept_above(node, mapped_ids, parent_adj))
            for node in mapped_ids}


def descendants(child_adj: dict, root) -> set:
    """Proper descendants of ``root`` as a SET, over a ``{parent: [children]}``
    adjacency (matching ``networkx.descendants``). For a `DagLayout`, use its own
    `descendants(u)`, which returns them sorted by (depth, id)."""
    seen: set = set()
    stack = list(child_adj.get(root, ()))
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(child_adj.get(node, ()))
    return seen


def children_and_min_depth(parent_int, C):
    """(children map {node: [children]}, min depth {node: int}) over ids [0, C).

    ``parent_int`` maps child -> [parent ids] (parentless/root -> []). MIN depth is
    the SHORTEST path from the parentless roots (BFS; root = 0). Not
    `DagLayout.depth`, which is the LONGEST path: on a multi-parent DAG a node
    reachable both directly and via a deeper parent has a smaller min depth."""
    from collections import deque
    children = {c: [] for c in range(C)}
    for child in range(C):
        for p in parent_int.get(child, []):
            p = int(p)
            if 0 <= p < C:
                children[p].append(child)
    depth = {c: None for c in range(C)}
    roots = [c for c in range(C) if not parent_int.get(c)]
    dq = deque((r, 0) for r in roots)
    while dq:
        n, d = dq.popleft()
        if depth[n] is not None and depth[n] <= d:
            continue
        depth[n] = d
        for ch in children[n]:
            dq.append((ch, d + 1))
    return children, depth


def parents_over_range(parent_int, C):
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
    parents = parents_over_range(parent_int, C)
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
