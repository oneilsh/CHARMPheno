"""`DagLayout`: a label DAG mapped onto topic blocks, and the frontier of a coded set.

`DagLayout(parent, n_bg, tpn)` takes a DAG over integer node ids (root 0, given as
`{child: [parents]}`) and lays out the topic space the gated engine reads: a shared
background block, then `tpn` topics per non-root node. `allowed_set(frontier)` is the
gate (background plus every node in the closure of the frontier); `depth`, `height`,
`closure`, `subtree` and `descendants` are the DAG queries the layout and its callers
need. `frontier_from_coded` turns a set of attested nodes into the most-specific ones.

Moved from `models/topic/dag_placement.py` with ADR 0051 (that module's collapsed-Gibbs
oracle went to the test suite). Generic: nothing here knows what the nodes mean.
"""
import numpy as np


class DagLayout:
    """Topic-block layout over a label DAG: `n_bg` shared background topics, then `tpn` topics per
    non-root node. `parent` maps child id -> parent id OR list of parent ids (multi-parent DAG); the
    root is id 0 (no entry). A scalar parent is normalized to a one-element list, so single-parent
    tree maps keep working unchanged."""

    def __init__(self, parent, n_bg=2, tpn=1):
        self.parents = {c: (list(p) if isinstance(p, (list, tuple, set)) else [p])
                        for c, p in parent.items()}
        self.nodes = sorted(self.parents.keys())
        self.n_bg = int(n_bg)
        self.tpn = int(tpn)
        self.children = {0: []}
        for c, ps in self.parents.items():
            self.children.setdefault(c, [])
            for p in ps:
                self.children.setdefault(p, []).append(c)
        for p in self.children:
            self.children[p] = sorted(self.children[p])
        self.block = {u: list(range(n_bg + i * tpn, n_bg + (i + 1) * tpn))
                      for i, u in enumerate(self.nodes)}
        self.K = n_bg + len(self.nodes) * tpn
        self._depth = {}
        self._height = {}

    def depth(self, v, _stack=()):
        """Longest path length from root to v (root = 0). Memoized; cycle-guarded.

        Input parent maps are acyclic by construction (ConditionDag.to_engine), but
        DagLayout is the domain-agnostic public entry and can be built from any integer
        map, so the _stack guard prevents a malformed cyclic map from recursing forever.
        """
        if v in self._depth:
            return self._depth[v]
        ps = [p for p in self.parents.get(v, []) if p != v and p not in _stack]
        d = 0 if not ps else 1 + max(self.depth(p, _stack + (v,)) for p in ps)
        self._depth[v] = d
        return d

    def height(self, v, _stack=()):
        """Longest path DOWN from v to a leaf (leaves = 0). Memoized; cycle-guarded.

        The clock of the leaves-first gate schedule (`DagGatedOnlineLDA.set_gate_schedule`):
        by height, not depth, because leaves sit at very different depths, and a node's
        height is strictly greater than every child's, so on a multi-parent DAG every
        child — a shared one included — opens before any of its parents."""
        if v in self._height:
            return self._height[v]
        ch = [c for c in self.children.get(v, []) if c != v and c not in _stack]
        h = 0 if not ch else 1 + max(self.height(c, _stack + (v,)) for c in ch)
        self._height[v] = h
        return h

    def closure(self, v):
        """All ancestors of v plus v, as a list sorted by (depth, id) so root comes first. For a
        single-parent tree this reproduces the old root..v ordering exactly."""
        seen = set()
        stack = [v]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            for p in self.parents.get(x, []):
                stack.append(p)
        return sorted(seen, key=lambda u: (self.depth(u), u))

    def subtree(self, u):
        out = {u}
        stack = [u]
        while stack:
            x = stack.pop()
            for ch in self.children.get(x, []):
                if ch not in out:
                    out.add(ch)
                    stack.append(ch)
        return out

    def descendants(self, u):
        """Proper descendants of u (every v of which u is a proper ancestor), sorted by
        (depth, id). The mirror of `closure` (ancestors + self): `descendants` reuses the
        children adjacency via `subtree` (u + descendants) and drops u, so there is no
        separate adjacency to keep in sync. A leaf has none; the anchor's descendants are
        every other node. Used by reverse-topological spectral init to deflate a node
        against its already-recovered descendants."""
        return sorted((v for v in self.subtree(u) if v != u),
                      key=lambda v: (self.depth(v), v))

    def allowed(self, v):
        al = list(range(self.n_bg))
        for u in self.closure(v):
            if u != 0:
                al += self.block[u]
        return np.array(sorted(al), dtype=int)

    def cost_report(self, *, vocab_size=None):
        """PRE-FLIGHT size profile for a gated fit over this layout, logged at the
        data-build boundary so a big (e.g. whole-Mondo) fit's shape is visible
        BEFORE compute is committed: K, the DAG fan-out (children per parent) and,
        given `vocab_size`, the lambda size (K * V * 8 bytes). Pure numpy; returns
        (metrics dict, formatted str)."""
        K = self.K
        fan = np.array([len(ch) for ch in self.children.values() if ch] or [0],
                       dtype=float)
        lam = (K * int(vocab_size) * 8) if vocab_size else None
        d = {"K": K, "n_nodes": len(self.nodes),
             "fanout_max": int(fan.max()),
             "fanout_p90": float(np.percentile(fan, 90)),
             "fanout_p99": float(np.percentile(fan, 99)),
             "lambda_bytes": lam}

        def _hb(b):
            for u in ("B", "KB", "MB", "GB", "TB"):
                if b < 1024:
                    return f"{b:.1f}{u}"
                b /= 1024
            return f"{b:.1f}PB"

        lines = [
            "=" * 74,
            f"GATED SIZE PROFILE  K={K} topics "
            f"({self.n_bg} bg + {len(self.nodes)} nodes x {self.tpn} tpn)",
            f"  fan-out (children/parent):  max={int(fan.max())}  "
            f"p90={d['fanout_p90']:.0f}  p99={d['fanout_p99']:.0f}",
        ]
        if lam is not None:
            lines.append(f"  lambda (K*V*8):             {_hb(lam)}")
        lines.append("=" * 74)
        return d, "\n".join(lines)

    def allowed_set(self, frontier, max_height=None):
        """Background ∪ blocks over the union of closures of the frontier nodes (set-valued gate).

        `max_height` (None = no cap, the production gate) keeps only closure nodes of
        height <= max_height: a round of the leaves-first schedule."""
        al = set(range(self.n_bg))
        for f in frontier:
            for u in self.closure(f):
                if u != 0 and (max_height is None or self.height(u) <= max_height):
                    al.update(self.block[u])
        return np.array(sorted(al), dtype=int)


def frontier_from_coded(coded_nodes, lay):
    """The set-valued truth: the most-specific attested nodes = attested nodes with NO attested
    descendant. Drops attested ancestors (same-path -> most-specific), keeps incomparable attested
    nodes as a set (comorbid or contradictory — the DAG cannot tell these apart, so we do not roll
    them up). Returns a frozenset."""
    C = set(coded_nodes)
    return frozenset(c for c in C
                     if not any((c2 != c) and (c2 in lay.subtree(c)) for c2 in C))


