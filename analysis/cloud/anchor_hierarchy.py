"""Reduce a big ontology DAG to the compact hierarchy induced by a set of anchors.

The Mondo DAG has hundreds of ancestors above any set of disease anchors, so
adding the full transitive ancestor closure as model/layout nodes would blow up
the topic count. But almost all of those ancestors are *uninformative* for a
given anchor set — they are linear-chain pass-throughs or the shared umbrella
root. The only ancestors that carry structure are the **branch points**: terms
that are the common ancestor of two or more anchors (the lowest such term = the
anchor set's natural "class", e.g. connective-tissue disorder, vasculitis).

This module keeps exactly those. For N leaf anchors the number of distinct
non-trivial anchor-coverage sets — hence kept class nodes — is O(N) (≤ N-1 for a
tree), so the compact hierarchy stays the same order as the flat-under-root layout
we use today while adding real intermediate class nodes. It is a pure,
ontology-agnostic graph reduction (takes a child->parents adjacency), so it is
unit-testable without Mondo and reusable for any DAG + terminal set.

The reduced hierarchy serves two purposes:
  1. class structure for within-class ("rank EDS vs Marfan vs scleroderma among
     connective-tissue patients") conditional readouts, and
  2. a DagLayout for a future fit where each class node gets its own topic block
     (partial pooling across its sibling anchors + hierarchical placement).
"""
from __future__ import annotations

from typing import Iterable


def ancestors(parent_adj: dict, node: str, stop: frozenset = frozenset()) -> set:
    """Transitive ancestors of ``node`` (breadth-first) over a child->parents
    adjacency, EXCLUDING any term in ``stop`` and everything above it (stop terms
    are treated as the ceiling — the over-general umbrella roots we don't want as
    classes). ``node`` itself is not included."""
    seen: set = set()
    stack = [p for p in parent_adj.get(node, ()) if p not in stop]
    while stack:
        cur = stack.pop()
        if cur in seen or cur in stop:
            continue
        seen.add(cur)
        stack.extend(p for p in parent_adj.get(cur, ()) if p not in stop)
    return seen


def _coverage(terminals: set, parent_adj: dict, stop: frozenset) -> dict:
    """node -> set of terminals it covers (is an ancestor of, or is)."""
    cov: dict = {}
    for t in terminals:
        for node in ancestors(parent_adj, t, stop) | {t}:
            cov.setdefault(node, set()).add(t)
    return cov


def reduce_to_anchor_hierarchy(
    terminals: Iterable[str],
    parent_adj: dict,
    *,
    stop: Iterable[str] = (),
    min_class_size: int = 2,
    max_class_fraction: float = 1.0,
) -> dict:
    """Compact class hierarchy induced by ``terminals`` over the DAG.

    Parameters
    ----------
    terminals : the anchor node ids (leaves of interest).
    parent_adj : {child: [parents]} adjacency over the ontology (subclass_of).
    stop : umbrella terms to treat as the ceiling (excluded as classes, and their
        ancestors too) — e.g. the Mondo human-disease root(s).
    min_class_size : an internal node is kept as a class only if it covers at
        least this many terminals (default 2 = a genuine branch point).
    max_class_fraction : drop classes covering more than this fraction of all
        terminals (default 1.0 = keep the top class; set <1 to suppress the
        broadest umbrella groupings). A class covering ALL terminals is always
        allowed only when max_class_fraction == 1.0.

    Returns a dict:
      "classes": {class_id: {"members": sorted terminals, "size": n}} — class_id
          is the most-specific ontology term with that exact anchor coverage.
      "parent_of": {node: [parent nodes]} over kept nodes (terminals + classes);
          a node with no kept superset maps to [] (attach to the synthetic root).
      "terminal_class": {terminal: class_id or None} — its most-specific class.
      "n_raw_ancestors": total distinct ancestor terms across all terminals (the
          count we are AVOIDING adding as nodes), for the size report.
      "n_classes": number of kept class nodes.
    """
    terminals = set(terminals)
    stop = frozenset(stop)
    n = len(terminals)
    cov = _coverage(terminals, parent_adj, stop)

    n_raw = sum(1 for node in cov if node not in terminals)
    max_size = n if max_class_fraction >= 1.0 else max(
        min_class_size, int(max_class_fraction * n))

    # Candidate class nodes: internal terms covering [min_class_size, max_size]
    # terminals. (A terminal that is itself an ancestor of another terminal can
    # also be a class, but we keep terminals as leaves and let their coverage
    # be represented by a separate internal rep when one exists.)
    candidates = {
        node: frozenset(members)
        for node, members in cov.items()
        if node not in terminals and min_class_size <= len(members) <= max_size
    }

    # Collapse chains: group candidates by identical coverage set; a linear chain
    # A->B->C over the same anchors yields one class. Representative = the
    # MOST-SPECIFIC node in the group (not an ancestor of any other group member).
    by_cover: dict = {}
    for node, members in candidates.items():
        by_cover.setdefault(members, []).append(node)
    class_of_cover: dict = {}
    for members, nodes in by_cover.items():
        node_set = set(nodes)
        # most-specific = a node none of whose (transitive) descendants-in-group
        # exist; i.e. it is not an ancestor of another group node.
        specific = [x for x in nodes
                    if not any(x in ancestors(parent_adj, y, stop) for y in node_set if y != x)]
        class_of_cover[members] = sorted(specific or nodes)[0]

    # Distinct class coverage sets, ordered so smaller (more specific) come first.
    covers = sorted(by_cover, key=lambda s: (len(s), sorted(s)))
    classes = {class_of_cover[c]: {"members": sorted(c), "size": len(c)} for c in covers}

    def _nearest_superset(cover: frozenset, strict: bool):
        best = None
        for other in covers:
            if other == cover and strict:
                continue
            if cover <= other and (not strict or cover < other):
                if best is None or len(other) < len(best):
                    best = other
        return best

    # parent links among class nodes + terminal -> most-specific class.
    parent_of: dict = {}
    for cover in covers:
        cid = class_of_cover[cover]
        sup = _nearest_superset(cover, strict=True)
        parent_of[cid] = [class_of_cover[sup]] if sup is not None else []
    terminal_class: dict = {}
    for t in terminals:
        sup = _nearest_superset(frozenset({t}), strict=False)
        cid = class_of_cover[sup] if sup is not None else None
        terminal_class[t] = cid
        parent_of[t] = [cid] if cid is not None else []

    return {
        "classes": classes,
        "parent_of": parent_of,
        "terminal_class": terminal_class,
        "n_raw_ancestors": n_raw,
        "n_classes": len(classes),
    }


def hierarchy_to_edges(reduced: dict, root, *, strip_prefix: str = "anchor:") -> list:
    """Convert a reduce_to_anchor_hierarchy result into concept-id (parent, child)
    edges for a DagLayout: root -> top classes, class -> class (multi-parent), and
    class -> anchor; any node with no kept parent attaches directly to ``root``
    (top classes and unclustered anchors alike).

    Terminal ids carry ``strip_prefix`` (e.g. "anchor:76685"); it is stripped and
    the remainder cast to the same type as ``root`` (int for concept-id space).
    Class ids are used as-is (cast to root's type). Returns sorted unique edges.
    """
    cast = type(root)

    def _cid(node):
        s = node[len(strip_prefix):] if node.startswith(strip_prefix) else node
        return cast(s)

    edges = set()
    for node, parents in reduced["parent_of"].items():
        child = _cid(node)
        if parents:
            for p in parents:
                edges.add((_cid(p), child))
        else:
            edges.add((root, child))
    return sorted(edges)
