"""Generic label-DAG operations over `{child: [parents]}` (or `{parent: [children]}`) maps.

Closure, descendants, depth, the transitive reduction of an order restricted to a
kept node set, and where an attestation of a dropped node lands. The node ids are any
hashables; nothing here knows what a node means (the Mondo-specific build that uses
these is `charmpheno.mondo.native_dag`). Moved with ADR 0051 from the Mondo builders,
the readout's closure product and the fit driver, verbatim except for names.
"""
from __future__ import annotations

import numpy as np


def parent_adjacency(child_adj) -> dict:
    """`{child: [parents]}` from a `{parent: [children]}` adjacency (e.g. Mondo's
    disease graph, `charmpheno.mondo.omop_mapping._disease_child_adjacency`). Parents
    are deduped and sorted so every downstream walk is deterministic."""
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


def nearest_kept_ancestors(term, kept, parent_adj, memo=None) -> set:
    """Where an attestation of `term` LANDS in a DAG built over `kept`: `{term}`
    when it is kept, else the nearest kept ancestor on each upward branch (the walk
    stops climbing a branch at its first kept node), else `set()`.

    The reflexive twin of `nearest_kept_parents`'s walk and of
    `condition_dag._nearest_surviving_ancestors`'s rewire, so an attestation lands
    exactly where the label DAG reattaches its node — the single-source-of-truth
    property the pruning ledger relies on. Composing it over the Hasse and then the
    splice is why resolving codes against the FINAL node set is correct: nearest
    kept ancestor of nearest kept ancestor is nearest kept ancestor."""
    if term in kept:
        return {term}
    if memo is not None and term in memo:
        return memo[term]
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
    if memo is not None:
        memo[term] = found
    return found


def transitive_reduction_parents(kept, parent_adj) -> dict:
    """The TRANSITIVE REDUCTION of the order `parent_adj` defines, restricted to `kept`.

    `nearest_kept_parents` is the first half: for each kept
    term, the first kept node on every upward branch. Its docstring calls that
    "the induced Hasse edges", and on main's dashboard it is — but it is NOT a
    transitive reduction on a multi-path DAG, and the difference is precisely exp
    0104's bug. Worked example, from this module's own unit tests:

        A                A, C, D kept; B unpowered and dropped.
       / \\               D's parents are B and C. Climbing D's LEFT branch stops
      B   C              at A (past the dropped B); the RIGHT branch stops at C.
       \\ /               nearest_kept_parents(D) = {A, C}.
        D

    A is then a parent of both C and D, so C and D are SIBLINGS — and C is an
    ANCESTOR of D. Under closure masking a doc that fires D observes its siblings
    under every parent, C among them, and C is positive on every such doc: an
    all-positive cell and a constant head. That is the 619-category-anchor
    mechanism reconstructed inside the very construction meant to make it
    impossible, and it appears exactly when an intermediate is dropped — i.e. on
    every unpowered rung of a real ontology.

    So the second half: a parent `p` is REDUNDANT when some other parent `q` of
    the same node has `p` among its own ancestors. Redundant parents are dropped,
    which is what "the transitive reduction deletes the redundant sibling edge"
    means operationally. Ancestry is read from the FULL Mondo graph, which is
    sound because the induced order on kept nodes is the restriction of the full
    one: if `p` is an ancestor of `q` in Mondo and both are kept, the edge `p->c`
    carries no information the path `p ... q -> c` does not.

    Multi-parenthood survives untouched wherever the parents are genuinely
    incomparable (the orthogonal-axis case, ~50% of Mondo terms): only comparable
    pairs are reduced. Returns `{kept term: [kept parents]}`; a term with no kept
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


def nearest_kept_parents(
    mapped_ids: set, parent_adj: dict[str, list[str]]
) -> dict[str, list[str]]:
    """For each mapped term, its nearest ANCESTOR terms that are themselves mapped
    (collapsing unmapped Mondo intermediates), over a child->parents adjacency. A
    term with no mapped ancestor gets ``[]`` (attaches to the synthetic root).

    "Nearest" = the first mapped node on every upward path; the search stops
    climbing a branch as soon as it hits a mapped ancestor, so we keep the closest
    mapped parents on each branch (the induced Hasse edges over mapped terms)."""
    out: dict[str, list[str]] = {}
    for node in mapped_ids:
        found: set = set()
        seen: set = set()
        stack = list(parent_adj.get(node, ()))
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            if cur in mapped_ids:
                found.add(cur)          # nearest on this branch; don't climb past it
            else:
                stack.extend(parent_adj.get(cur, ()))
        out[node] = sorted(found)
    return out


def descendants(child_adj: dict[str, list[str]], root: str) -> set[str]:
    """Transitive descendants of ``root`` (breadth-first), matching
    ``networkx.descendants`` over the same parent->child adjacency."""
    seen: set[str] = set()
    stack = list(child_adj.get(root, ()))
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(child_adj.get(node, ()))
    return seen


def children_and_depth(parent_int, C):
    """(children map {node: [children]}, depth {node: int}) over engine ids [0, C).

    ``parent_int`` maps child -> [parent engine-ids] (parentless/root -> []). Depth
    is the BFS distance from the parentless roots (the synthetic forest root is
    depth 0; the disease anchors depth 1; their Mondo/SNOMED subtypes deeper)."""
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
