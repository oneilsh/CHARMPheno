"""Domain-agnostic hierarchical placement engine (integer ids only). Places held-out items in
a label DAG from their features via gated collapsed-Gibbs topic learning (Griffiths & Steyvers
2004) with anchor-word spectral init (Arora et al. 2013). See
docs/superpowers/specs/2026-07-15-anchor-first-hierarchical-case-finding-design.md."""
import numpy as np


class DagLayout:
    """Topic-block layout over a label DAG: `n_bg` shared background topics, then `tpn` topics per
    non-root node. `parent` maps child id -> parent id; the root is id 0 (no entry)."""

    def __init__(self, parent, n_bg=2, tpn=1):
        self.parent = dict(parent)
        self.nodes = sorted(parent.keys())
        self.n_bg = int(n_bg)
        self.tpn = int(tpn)
        self.children = {0: []}
        for c, p in parent.items():
            self.children.setdefault(p, []).append(c)
            self.children.setdefault(c, [])
        for p in self.children:
            self.children[p] = sorted(self.children[p])
        self.block = {u: list(range(n_bg + i * tpn, n_bg + (i + 1) * tpn))
                      for i, u in enumerate(self.nodes)}
        self.K = n_bg + len(self.nodes) * tpn

    def closure(self, v):
        c = [v]
        while v in self.parent:
            v = self.parent[v]
            c.append(v)
        return c[::-1]

    def subtree(self, u):
        out = {u}
        stack = [u]
        while stack:
            x = stack.pop()
            for ch in self.children.get(x, []):
                out.add(ch)
                stack.append(ch)
        return out

    def allowed(self, v):
        al = list(range(self.n_bg))
        for u in self.closure(v):
            if u != 0:
                al += self.block[u]
        return np.array(sorted(al), dtype=int)

    def depth(self, v):
        return len(self.closure(v)) - 1


def label_from_coded(coded_nodes, lay):
    """The item's label from its in-window coded nodes. If they lie on a single root->node path
    (one node is a descendant-or-self of all others), return that deepest node (most-specific).
    Otherwise return the lowest common ancestor (deepest node that is an ancestor-or-self of all)."""
    nodes = list(dict.fromkeys(coded_nodes))
    for cand in nodes:                                   # single-path: cand's closure holds all
        cset = set(lay.closure(cand))
        if all(n in cset for n in nodes):
            return cand
    common = set(lay.closure(nodes[0]))
    for n in nodes[1:]:
        common &= set(lay.closure(n))
    return max(common, key=lay.depth)                    # root (0) is always common


def strip_dag_node_codes(doc, dag_node_codes):
    """Remove every token whose id matches a DAG-node code (leakage strip; evaluation only)."""
    doc = np.asarray(doc)
    if not dag_node_codes:
        return doc
    mask = ~np.isin(doc, np.fromiter(dag_node_codes, dtype=doc.dtype))
    return doc[mask]
