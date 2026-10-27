"""The label-DAG layout the gated topic engine reads, plus its collapsed-Gibbs test oracle.

`DagLayout` maps a DAG (integer node ids, root 0) to topic blocks: a shared background
block, then `tpn` topics per node; `allowed_set(frontier)` is the gate. Hashed charmpheno
modules import `DagLayout` and `frontier_from_coded` from here, so they stay at this path
until their move is made as an announced cache drop (ADR 0050/0051).

`fit_gated` / `profile` are the collapsed-Gibbs placement estimator (Griffiths & Steyvers
2004) the variational engine (`DagGatedOnlineLDA`) is validated against; they run in tests
only. The node-placement scoring and evaluation of the first gated model (the
`dag_placement` model class, exps 0052-0069) was removed by ADR 0050. See
docs/superpowers/specs/2026-07-15-anchor-first-hierarchical-case-finding-design.md."""
from types import SimpleNamespace

import numpy as np

from spark_vi.models.topic.spectral_init import word_cooccurrence, find_anchors, recover_beta


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

        The clock of the leaves-first gate schedule (`GatedOnlineLDA.set_gate_schedule`):
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

    def allowed_with_siblings(self, v):
        """`allowed(v)` PLUS the blocks of v's SIBLINGS (the other children of v's
        parents). This is the topic support a LOCALIZED conditional head needs: to rank
        v against the siblings it is scored against (the closure objective's contrast
        set), the head must read their topics, not just v's own block + ancestors —
        without them the wide top level collapses to chance (exp 0089 run 1). Still
        O(depth + local fan-out) ≪ K, so the per-node head stays cheap at whole-Mondo."""
        al = set(int(k) for k in self.allowed(v))
        for p in self.parents.get(v, []):
            for sib in self.children.get(p, []):
                if sib != v and sib != 0:
                    al.update(self.block[sib])
        return np.array(sorted(al), dtype=int)

    def allowed_with_path_cousins(self, v):
        """`allowed(v)` PLUS the sibling blocks of EVERY node on v's root-path — the
        path-cousins, not just v's own siblings. Widens the head's contrast set up the
        whole hierarchy: to rank v against the diseases it competes with at each level, the
        head reads the cousins at every ancestor, not only the immediate siblings — the
        out-of-immediate-neighborhood signal a closure+siblings head misses. Still bounded,
        O(depth * fan-out) << K, so the per-node Fisher (C*S^2) and solve (C*S^3) stay
        localized-cheap and shuffleable — EXACT Newton is kept, unlike a dense full-K head
        whose C*K^2 Fisher blows the driver collect and thrashes on preemptible executors."""
        al = set(int(k) for k in self.allowed(v))
        for a in self.closure(v):
            if a == 0:
                continue
            for p in self.parents.get(a, []):
                for sib in self.children.get(p, []):
                    if sib != 0 and sib != a:
                        al.update(self.block[sib])
        return np.array(sorted(al), dtype=int)

    def allowed_with_path_cousins_kids(self, v):
        """`allowed_with_path_cousins(v)` PLUS v's OWN children's blocks — the subtype signal.
        A patient coded at a subtype of v is v-positive, and its subtype-specific topics predict
        v, so the head reading v's children's blocks captures that. Immediate children only
        (bounded by fan-out); deeper descendants' signal already reaches v's block via the
        closure gate. Still O(depth*fan-out) << K — exact Newton kept."""
        al = set(int(k) for k in self.allowed_with_path_cousins(v))
        for ch in self.children.get(v, []):
            if ch != 0:
                al.update(self.block[ch])
        return np.array(sorted(al), dtype=int)

    def cost_report(self, C, *, vocab_size=None, localized=True):
        """PRE-FLIGHT size/cost profile for a gated-PC fit over this layout — logged at
        the data-build boundary so a big (e.g. whole-Mondo) fit's cost is visible BEFORE
        compute is committed. Reports: fan-out (children/parent), K/C, per-node head
        SUPPORT sizes (with siblings when localized — the quantity that drives per-node
        Newton cost and can blow up on a high-fan-out parent), and the head Fisher
        memory/compute DENSE (C·K²/C·K³) vs LOCALIZED (Σ|s|²/Σ|s|³). Pure numpy; returns
        (metrics dict, formatted str)."""
        K = self.K
        fan = np.array([len(ch) for ch in self.children.values() if ch] or [0],
                       dtype=float)
        # support over the ACTUAL label nodes (root 0 + non-root nodes), robust to a
        # non-contiguous id map; in production ids are contiguous so this == range(C).
        node_ids = [0] + list(self.nodes)
        supp = np.array(
            [len(self.allowed_with_siblings(c) if localized else self.allowed(c))
             for c in node_ids], dtype=float)

        def _q(a, p):
            return float(np.percentile(a, p))

        smax, sarg = int(supp.max()), int(supp.argmax())
        dense_mem, dense_flop = C * K * K * 8, float(C) * K ** 3
        loc_mem, loc_flop = float((supp ** 2).sum()) * 8, float((supp ** 3).sum())
        # The Fisher stat is actually EMITTED/COLLECTED as a padded (C, S, S) block
        # stack (S = smax) — this is the size that hits spark.driver.maxResultSize,
        # so report it alongside the dense wall and the packed Sum|s|^2 floor.
        pad_mem = C * smax * smax * 8
        lam = (K * int(vocab_size) * 8) if vocab_size else None
        d = {"C": C, "K": K, "n_nodes": len(self.nodes),
             "fanout_max": int(fan.max()), "fanout_p90": _q(fan, 90),
             "fanout_p99": _q(fan, 99), "support_max": smax, "support_argmax": sarg,
             "support_p50": _q(supp, 50), "support_p90": _q(supp, 90),
             "support_p99": _q(supp, 99), "dense_head_mem_bytes": dense_mem,
             "dense_head_flop": dense_flop, "localized_head_mem_bytes": loc_mem,
             "collected_head_mem_bytes": pad_mem,
             "localized_head_flop": loc_flop, "per_node_max_flop": float(smax) ** 3,
             "lambda_bytes": lam}

        def _hb(b):
            if b is None:
                return "n/a"
            for u in ("B", "KB", "MB", "GB", "TB"):
                if b < 1024:
                    return f"{b:.1f}{u}"
                b /= 1024
            return f"{b:.1f}PB"

        lines = [
            "=" * 74,
            f"GATED-PC SIZE/COST PROFILE  C={C} labels  K={K} topics "
            f"({self.n_bg} bg + {len(self.nodes)} nodes x {self.tpn} tpn)  "
            f"[head={'localized+siblings' if localized else 'dense'}]",
            f"  fan-out (children/parent):  max={int(fan.max())}  "
            f"p90={_q(fan, 90):.0f}  p99={_q(fan, 99):.0f}",
            f"  head support/node:          max={smax} (node {sarg})  "
            f"p50={_q(supp, 50):.0f}  p90={_q(supp, 90):.0f}  p99={_q(supp, 99):.0f}",
            f"  head Fisher MEMORY:  dense C*K^2={_hb(dense_mem)}   "
            f"collected C*S^2={_hb(pad_mem)} (S={smax})   "
            f"packed Sum|s|^2={_hb(loc_mem)}   "
            f"({dense_mem / max(pad_mem, 1):.0f}x smaller)",
            f"  head SOLVE/iter:     dense C*K^3={dense_flop:.2e}   "
            f"localized Sum|s|^3={loc_flop:.2e}   max node |s|^3={float(smax) ** 3:.2e}",
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
    per-sweep normalized rows.

    ORACLE / VALIDATOR — NOT the production fit path. This collapsed-Gibbs estimator is the reference
    the production SVI engine (DagGatedOnlineLDA, models/topic/dag_gated_lda.py) is validated
    against; no driver calls it. Used in tests only (its runtime — a Python per-token loop — is not tuned for production scale)."""
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
    block. The full profile IS the output; do not collapse to a single node.

    ORACLE / VALIDATOR — the Gibbs-oracle fold-in counterpart of `fit_gated`, NOT the production
    read-out. Test/oracle only."""
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
