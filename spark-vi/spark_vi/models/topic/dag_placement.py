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
    hence there is no `alpha` argument."""
    K = lay.K
    counted = [_as_counts(d) for d in train_docs]
    Q = word_cooccurrence(counted, V)
    beta0 = recover_beta(Q, find_anchors(Q, K))
    beta0 = beta0 + 1e-6
    beta0 /= beta0.sum(1, keepdims=True)
    n_kw = np.zeros((K, V))
    n_k = np.zeros(K)
    allowed = [lay.allowed(v) for v in train_labels]
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
    beta_hat = acc / nacc
    beta_hat /= beta_hat.sum(1, keepdims=True)
    return beta_hat


def profile(doc, beta_hat, lay, *, alpha=0.1, n_iter=60, burn=30, rng=None):
    """Unmasked fold-in (topics fixed) -> per-node affinity = posterior mean mass on each node's
    block. The full profile IS the output; do not collapse to a single node."""
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


def evaluate(profiles, test_labels, lay):
    """Per-node case-finding AUC (subtree membership), AUC by depth, and true-node MRR / top-2.
    Profiles are the graded affinity dicts from `profile`; scoring never collapses to one node."""
    P = np.array([[pr[u] for u in lay.nodes] for pr in profiles])
    node_auc = {u: _auc(P[:, i], [t in lay.subtree(u) for t in test_labels])
                for i, u in enumerate(lay.nodes)}
    ranks = []
    for i, t in enumerate(test_labels):
        ti = lay.nodes.index(t)
        ranks.append(1 + int((P[i] > P[i][ti]).sum()))
    ranks = np.array(ranks)
    by_depth = {}
    for dep in sorted({lay.depth(u) for u in lay.nodes}):
        us = [u for u in lay.nodes if lay.depth(u) == dep]
        by_depth[dep] = float(np.nanmean([node_auc[u] for u in us]))
    return {"node_auc": node_auc, "auc_by_depth": by_depth,
            "mrr": float(np.mean(1.0 / ranks)), "top2": float(np.mean(ranks <= 2))}
