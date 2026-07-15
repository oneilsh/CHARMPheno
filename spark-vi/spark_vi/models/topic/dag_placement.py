"""Domain-agnostic hierarchical placement engine (integer ids only). Places held-out items in
a label DAG from their features via gated collapsed-Gibbs topic learning (Griffiths & Steyvers
2004) with anchor-word spectral init (Arora et al. 2013). See
docs/superpowers/specs/2026-07-15-anchor-first-hierarchical-case-finding-design.md."""
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

    def depth(self, v):
        """Longest path length from root to v (root = 0). Memoized."""
        if v in self._depth:
            return self._depth[v]
        ps = self.parents.get(v, [])
        d = 0 if not ps else 1 + max(self.depth(p) for p in ps)
        self._depth[v] = d
        return d

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

    def allowed(self, v):
        al = list(range(self.n_bg))
        for u in self.closure(v):
            if u != 0:
                al += self.block[u]
        return np.array(sorted(al), dtype=int)

    def allowed_set(self, frontier):
        """Background ∪ blocks over the union of closures of the frontier nodes (set-valued gate)."""
        al = set(range(self.n_bg))
        for f in frontier:
            for u in self.closure(f):
                if u != 0:
                    al.update(self.block[u])
        return np.array(sorted(al), dtype=int)


def label_from_coded(coded_nodes, lay):
    """The item's label from its in-window coded nodes. If they lie on a single root->node path
    (one node is a descendant-or-self of all others), return that deepest node (most-specific).
    Otherwise return the lowest common ancestor (deepest node that is an ancestor-or-self of all)."""
    nodes = list(dict.fromkeys(coded_nodes))
    if not nodes:
        return 0
    for cand in nodes:                                   # single-path: cand's closure holds all
        cset = set(lay.closure(cand))
        if all(n in cset for n in nodes):
            return cand
    common = set(lay.closure(nodes[0]))
    for n in nodes[1:]:
        common &= set(lay.closure(n))
    return max(common, key=lay.depth)                    # root (0) is always common


def frontier_from_coded(coded_nodes, lay):
    """The set-valued truth: the most-specific attested nodes = attested nodes with NO attested
    descendant. Drops attested ancestors (same-path -> most-specific), keeps incomparable attested
    nodes as a set (comorbid or contradictory — the DAG cannot tell these apart, so we do not roll
    them up; multi-frontier is instrumented by evaluate). Returns a frozenset."""
    C = set(coded_nodes)
    return frozenset(c for c in C
                     if not any((c2 != c) and (c2 in lay.subtree(c)) for c2 in C))


def strip_dag_node_codes(doc, dag_node_codes):
    """Remove every token whose id matches a DAG-node code (leakage strip; evaluation only)."""
    doc = np.asarray(doc)
    if not dag_node_codes:
        return doc
    mask = ~np.isin(doc, np.fromiter(dag_node_codes, dtype=doc.dtype))
    return doc[mask]


from types import SimpleNamespace
from spark_vi.models.topic.spectral_init import word_cooccurrence, find_anchors, recover_beta


def _as_counts(doc):
    idx, cnt = np.unique(np.asarray(doc), return_counts=True)
    return SimpleNamespace(indices=idx, counts=cnt.astype(np.float64))


def fit_gated(train_docs, train_labels, lay, V, *, beta_prior=0.02,
              n_iter=150, burn=80, rng=None):
    """Gated topic training by collapsed sampling: each training item is masked to
    allowed(label) = background ∪ blocks along its label's closure, tying topics to nodes
    structurally. Anchor-word spectral init (Arora et al. 2013) seeds beta. Returns posterior-mean
    beta_hat (K, V).

    The per-token conditional uses ONLY the collapsed word-topic factor
    (n_kw + beta_prior) / (n_k + V*beta_prior) of Griffiths & Steyvers (2004), restricted to the
    gated allowed-set. The usual document-topic Dirichlet factor (n_dk + alpha) is intentionally
    omitted: the gate already fixes each document's admissible topics to its label closure, so the
    per-document mixing term is not what we are estimating here — we want the node-tied topic-word
    distributions. This is a supervised topic-word estimator, not the full unsupervised LDA sampler;
    hence there is no `alpha` argument. The returned beta_hat averages the post-burn token counts
    (n_kw + beta_prior) and normalizes once — the averaged-counts point estimate, not the mean of
    per-sweep normalized rows."""
    rng = np.random.default_rng() if rng is None else rng
    K = lay.K
    counted = [_as_counts(d) for d in train_docs]
    Q = word_cooccurrence(counted, V)
    beta0 = recover_beta(Q, find_anchors(Q, K))
    if beta0.shape[0] < K:
        pad = np.full((K - beta0.shape[0], V), 1.0 / V)
        beta0 = np.vstack([beta0, pad])
    beta0 = beta0 + 1e-6
    beta0 /= beta0.sum(1, keepdims=True)
    n_kw = np.zeros((K, V))
    n_k = np.zeros(K)
    # Each label may be a scalar node id or a frontier set (comorbid patient). A comorbid patient
    # trains every block along the union of its frontier's closures — strictly better use of data.
    allowed = [lay.allowed_set(y if hasattr(y, "__iter__") else (y,)) for y in train_labels]
    words = [np.asarray(d, dtype=np.int64) for d in train_docs]
    Z = []
    for d in range(len(train_docs)):
        al = allowed[d]
        w = words[d]
        r = beta0[al][:, w].T
        r = r / r.sum(1, keepdims=True)
        zi = al[(rng.random(len(w))[:, None] < np.cumsum(r, 1)).argmax(1)]
        Z.append(zi)
        np.add.at(n_kw, (zi, w), 1.0)
        for k in zi:
            n_k[k] += 1.0
    Vb = V * beta_prior
    acc = np.zeros((K, V))
    nacc = 0
    for it in range(n_iter):
        for d in range(len(train_docs)):
            al = allowed[d]
            w = words[d]
            zi = Z[d]
            for i in range(len(w)):
                wi = w[i]
                k = zi[i]
                n_kw[k, wi] -= 1.0
                n_k[k] -= 1.0
                p = (n_kw[al, wi] + beta_prior) / (n_k[al] + Vb)
                p /= p.sum()
                knew = al[np.searchsorted(np.cumsum(p), rng.random())]
                zi[i] = knew
                n_kw[knew, wi] += 1.0
                n_k[knew] += 1.0
        if it >= burn:
            acc += n_kw + beta_prior
            nacc += 1
    beta_hat = acc / max(nacc, 1)
    beta_hat /= beta_hat.sum(1, keepdims=True)
    return beta_hat


def profile(doc, beta_hat, lay, *, alpha=0.1, n_iter=60, burn=30, rng=None):
    """Unmasked fold-in (topics fixed) -> per-node affinity = posterior mean mass on each node's
    block. The full profile IS the output; do not collapse to a single node."""
    rng = np.random.default_rng() if rng is None else rng
    K = lay.K
    w = np.asarray(doc, dtype=np.int64)
    ndk = np.zeros(K)
    zi = rng.integers(K, size=len(w))
    for k in zi:
        ndk[k] += 1.0
    acc = np.zeros(K)
    nacc = 0
    for it in range(n_iter):
        for i in range(len(w)):
            wi = w[i]
            k = zi[i]
            ndk[k] -= 1.0
            p = (ndk + alpha) * beta_hat[:, wi]
            s = p.sum()
            p = p / s if s > 0 else np.full(K, 1.0 / K)
            knew = int(np.searchsorted(np.cumsum(p), rng.random()))
            zi[i] = knew
            ndk[knew] += 1.0
        if it >= burn:
            acc += ndk / max(len(w), 1)
            nacc += 1
    th = acc / max(nacc, 1)
    return {u: float(th[lay.block[u]].sum()) for u in lay.nodes}


def _auc(scores, y):
    y = np.asarray(y)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _hops(a, b, lay):
    """Undirected hop distance between two nodes over parent/child edges (BFS)."""
    if a == b:
        return 0
    seen = {a}
    queue = [(a, 0)]
    while queue:
        x, d = queue.pop(0)
        for nb in list(lay.parents.get(x, [])) + lay.children.get(x, []):
            if nb == b:
                return d + 1
            if nb not in seen:
                seen.add(nb)
                queue.append((nb, d + 1))
    return float("inf")


def evaluate(profiles, test_labels, lay):
    """Per-node case-finding AUC (subtree membership), AUC by longest-path depth, and set-valued
    ranking. `test_labels` entries may be frontier sets or scalars (scalars -> singletons). A patient
    is a positive for node u if any of its frontier lies in subtree(u). MRR/top2/mean_hops use the
    BEST (closest) true frontier node. frontier_size_mean and multi_frontier_rate instrument the
    comorbid/contradictory ambiguity (the DAG cannot tell those apart; we surface it, not resolve it).
    Profiles are the graded affinity dicts from `profile`; scoring never collapses to one node."""
    fronts = [set(t) if hasattr(t, "__iter__") else {t} for t in test_labels]
    P = np.array([[pr[u] for u in lay.nodes] for pr in profiles])
    node_auc = {u: _auc(P[:, i], [bool(f & lay.subtree(u)) for f in fronts])
                for i, u in enumerate(lay.nodes)}
    ranks, hops = [], []
    for i, f in enumerate(fronts):
        true_idx = [lay.nodes.index(t) for t in f if t in lay.nodes]   # skip root/unscoreable
        if not true_idx:
            continue
        ranks.append(min(1 + int((P[i] > P[i][j]).sum()) for j in true_idx))   # best (smallest) rank
        pred = lay.nodes[int(np.argmax(P[i]))]
        hops.append(min(_hops(pred, lay.nodes[j], lay) for j in true_idx))
    ranks = np.array(ranks, dtype=float) if ranks else np.array([np.nan])
    by_depth = {}
    for dep in sorted({lay.depth(u) for u in lay.nodes}):
        us = [u for u in lay.nodes if lay.depth(u) == dep]
        by_depth[dep] = float(np.nanmean([node_auc[u] for u in us]))
    return {"node_auc": node_auc, "auc_by_depth": by_depth,
            "mrr": float(np.nanmean(1.0 / ranks)),
            "top2": float(np.nanmean(ranks <= 2)),
            "mean_hops": float(np.mean(hops)) if hops else float("nan"),
            "frontier_size_mean": float(np.mean([len(f) for f in fronts])),
            "multi_frontier_rate": float(np.mean([len(f) > 1 for f in fronts]))}


def _node_topic_mean(beta_hat, lay, u):
    return beta_hat[lay.block[u]].mean(0)


def identifiability_annotation(beta_hat, lay, *, tol=0.9):
    """Post-fit diagnostic: flag WITHIN-STRUCTURE node pairs (siblings, or parent<->child) whose
    learned topic distributions are near-collinear (cosine >= tol) -> hard to separate. Cross-branch
    pairs are never reported; their similarity is a reporting fact, not a structural one."""
    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    pairs = set()
    for u in lay.nodes:                                  # parent<->child
        for c in lay.children.get(u, []):
            pairs.add((u, c))
    for p, kids in lay.children.items():                 # siblings
        for i in range(len(kids)):
            for j in range(i + 1, len(kids)):
                pairs.add((kids[i], kids[j]))
    out = []
    for u, v in pairs:
        c = cos(_node_topic_mean(beta_hat, lay, u), _node_topic_mean(beta_hat, lay, v))
        if c >= tol:
            out.append((u, v, c))
    return out


def render_profile(affinity, lay, *, names=None, true_node=None, width=24):
    """Indented DAG tree with a unicode affinity bar per node (spot-check output, sim and real)."""
    names = names or {}
    lines = []

    def bar(x):
        n = int(round(max(0.0, min(1.0, x)) * width))
        return "█" * n + "▁" * (width - n)

    def walk(v, prefix, is_last):
        if v == 0:
            lines.append(names.get(0, "root"))
        else:
            a = affinity.get(v, 0.0)
            conn = "└─ " if is_last else "├─ "
            mark = "  <- true" if v == true_node else ""
            nm = str(names.get(v, v)).ljust(10)
            lines.append(f"{prefix}{conn}{nm} {bar(a)} {a:0.2f}{mark}")
        kids = lay.children.get(v, [])
        child_prefix = prefix + ("   " if is_last else "│  ") if v != 0 else ""
        for i, c in enumerate(kids):
            walk(c, child_prefix, i == len(kids) - 1)

    walk(0, "", True)
    return "\n".join(lines)
