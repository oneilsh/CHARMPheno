"""Domain-agnostic hierarchical placement engine (integer ids only). Places held-out items in
a label DAG from their features via gated collapsed-Gibbs topic learning (Griffiths & Steyvers
2004) with anchor-word spectral init (Arora et al. 2013). See
docs/superpowers/specs/2026-07-15-anchor-first-hierarchical-case-finding-design.md."""
from types import SimpleNamespace

import numpy as np
from scipy import stats as _sps
from scipy.sparse import issparse

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


def _routing_rows(lam, lay, *, epsilon=1e-9):
    """Per-node soft responsibility Rnode[u,w] = fraction of code w's total
    topic-probability that lands on node u's topic block. Codes compete across ALL
    topics (background + nodes) with a UNIFORM topic prior (responsibility ∝
    P(w|topic) = λ[k]/Σλ[k]); a code unseen in every topic -> 0 everywhere. This is
    the "explain-away" routing (Pearl 1988; the mixture E-step's soft assignment):
    a comorbid code claimed by a background topic gets ~0 node responsibility, so it
    neither penalizes nor spuriously supports a foreground node. Returns [n_nodes x V]
    in lay.nodes order."""
    lam = np.asarray(lam, dtype=float)
    ptopic = lam / np.maximum(lam.sum(axis=1, keepdims=True), epsilon)  # P(w|topic k)
    rtopic = ptopic / np.maximum(ptopic.sum(axis=0, keepdims=True), epsilon)  # responsibility
    rnode = np.zeros((len(lay.nodes), lam.shape[1]), dtype=float)
    for i, u in enumerate(lay.nodes):
        rnode[i] = rtopic[lay.block[u]].sum(axis=0)
    return rnode


def _lr_logratio_rows(lam, lay, *, alpha, bg, epsilon):
    """Per-node shrunk log-ratio row log[P(w|node u)/bg(w)], stacked [n_nodes x V].

    P(w|node u) = (Σ_{k in block(u)} λ[k,w] + α·bg(w)) / (Σλ(u) + α) — Dirichlet /
    empirical-Bayes smoothing toward the background base rate bg: large α pulls a
    mass-starved node toward bg (under-evidenced and unseen codes -> log-ratio ≈ 0),
    small α trusts the node's own counts. Floored at epsilon so α=0 never yields
    log(0).

    α = inf (the parameter-free limit): as α->∞, log(P/bg) = log((nc/bg + α)/(Σλ + α))
    ~ (nc/bg - Σλ)/α, so the score's RANKING converges (up to the positive 1/α scale,
    which AUC/argmax ignore) to the knob-free direction (nc/bg - Σλ) — a "lift minus
    node-mass" score. This is the limit the α sweep approaches; use it to read the
    readout with no shrinkage hyperparameter to choose."""
    n_nodes = len(lay.nodes)
    logratio = np.zeros((n_nodes, lam.shape[1]))
    for i, u in enumerate(lay.nodes):
        nc = lam[lay.block[u]].sum(axis=0)
        if np.isinf(alpha):
            logratio[i] = nc / bg - nc.sum()          # parameter-free α->∞ limit
        else:
            p_u = (nc + alpha * bg) / (nc.sum() + alpha)
            logratio[i] = np.log(np.maximum(p_u, epsilon) / bg)
    return logratio


def _lr_base_rate(bow, background, epsilon):
    if background is None:
        col = np.asarray(bow.sum(axis=0)).ravel().astype(float)
        bg = col / max(col.sum(), 1.0)
    else:
        bg = np.asarray(background, dtype=float)
    return np.maximum(bg, epsilon)


def lr_placement_scores(bow, lam, lay, *, alpha, background=None, epsilon=1e-9,
                        count_mode="raw", length_normalize=False):
    """Per-node Naive-Bayes log-likelihood-ratio placement score.

    s(i,u) = Σ_w cnt(i,w)·log[P(w|node u)/bg(w)], reading the learned topic-word
    counts λ as class-conditional distributions (P(w|node u) = the node block's λ
    rows summed+normalized+shrunk toward bg; see `_lr_logratio_rows`). Unlike
    θ-mass this does not compete on the simplex, and the log-ratio down-weights
    common codes automatically (idf-for-free). `bow` [n_docs x V] counts (dense or
    scipy.sparse); `background` = base rate (None -> corpus code frequency from
    bow). count_mode 'raw'|'log1p' (saturate repeated codes); length_normalize
    divides by the per-doc token count. Returns [n_docs x n_nodes], columns in
    lay.nodes order."""
    lam = np.asarray(lam, dtype=float)
    bg = _lr_base_rate(bow, background, epsilon)
    logratio = _lr_logratio_rows(lam, lay, alpha=alpha, bg=bg, epsilon=epsilon)
    X = bow
    if count_mode == "log1p":
        if issparse(X):
            X = X.copy(); X.data = np.log1p(X.data)
        else:
            X = np.log1p(X)
    scores = np.asarray(X @ logratio.T, dtype=float)
    if length_normalize:
        tok = np.asarray(bow.sum(axis=1)).ravel().astype(float)
        scores = scores / np.maximum(tok, 1.0)[:, None]
    return scores


def explain_away_placement_scores(bow, lam, lay, *, alpha, background=None,
                                  epsilon=1e-9, count_mode="raw",
                                  length_normalize=False):
    """Explain-away (responsibility-weighted) LR placement score:
    s(i,u) = Σ_w cnt(i,w) · r(u|w) · log[P(w|u)/bg(w)], where r(u|w) is code w's soft
    responsibility on node u's block (_routing_rows). Codes competing to a background
    topic (comorbidities) get r(u|w) ~ 0, so their evidence -- crucially the SMALL
    NEGATIVE log-ratios that make the plain LR penalize comorbidity-heavy patients --
    is suppressed toward 0 instead of docking the node. Same signature/shape as
    lr_placement_scores; the α->∞ lift limit applies to the evidence term, routing is
    α-independent (raw normalized λ). Returns [n_docs x n_nodes], lay.nodes order."""
    lam = np.asarray(lam, dtype=float)
    bg = _lr_base_rate(bow, background, epsilon)
    logratio = _lr_logratio_rows(lam, lay, alpha=alpha, bg=bg, epsilon=epsilon)
    weight = _routing_rows(lam, lay, epsilon=epsilon) * logratio   # Rnode ⊙ logratio
    X = bow
    if count_mode == "log1p":
        if issparse(X):
            X = X.copy(); X.data = np.log1p(X.data)
        else:
            X = np.log1p(X)
    scores = np.asarray(X @ weight.T, dtype=float)
    if length_normalize:
        tok = np.asarray(bow.sum(axis=1)).ravel().astype(float)
        scores = scores / np.maximum(tok, 1.0)[:, None]
    return scores


def explain_away_decompose(bow_row, lam, lay, u, *, alpha, background,
                           epsilon=1e-9, count_mode="raw"):
    """Itemized (w, count, r(u|w), contribution) for
    explain_away_placement_scores(...)[i, node u] (raw, no length-normalization).
    contribution = cnt · r(u|w) · log[P(w|u)/bg(w)]; Σ contribution == that node's
    RAW score (length_normalize divides the placement score by token count after
    this sum, so the two only match when length_normalize=False). r(u|w) is the
    routing weight (0 = the code went to background/another node; ~1 = it belongs
    to u), so the viewer can show WHERE each code routed. Only codes present in
    bow_row are returned, sorted by |contribution| desc."""
    lam = np.asarray(lam, dtype=float)
    bg = np.maximum(np.asarray(background, dtype=float), epsilon)
    i = lay.nodes.index(u)
    logratio = _lr_logratio_rows(lam, lay, alpha=alpha, bg=bg, epsilon=epsilon)[i]
    rnode = _routing_rows(lam, lay, epsilon=epsilon)[i]
    row = np.asarray(bow_row).ravel().astype(float)
    cnt = np.log1p(row) if count_mode == "log1p" else row
    contrib = cnt * rnode * logratio
    out = [(int(w), float(row[w]), float(rnode[w]), float(contrib[w]))
           for w in np.nonzero(row)[0]]
    out.sort(key=lambda t: -abs(t[3]))
    return out


def lr_decompose(bow_row, lam, lay, u, *, alpha, background, epsilon=1e-9,
                 count_mode="raw"):
    """Itemized (w, count, contribution) for lr_placement_scores(...)[i, node u]
    (raw, no length-normalization). Σ contributions == that score. Only codes
    present in bow_row are returned, sorted by |contribution| desc."""
    lam = np.asarray(lam, dtype=float)
    bg = np.maximum(np.asarray(background, dtype=float), epsilon)
    logratio = _lr_logratio_rows(lam, lay, alpha=alpha, bg=bg,
                                 epsilon=epsilon)[lay.nodes.index(u)]
    row = np.asarray(bow_row).ravel().astype(float)
    cnt = np.log1p(row) if count_mode == "log1p" else row
    contrib = cnt * logratio
    out = [(int(w), float(row[w]), float(contrib[w])) for w in np.nonzero(row)[0]]
    out.sort(key=lambda t: -abs(t[2]))
    return out


def lr_auc_sweep(bow, lam, lay, is_fg, *, alpha_grid, background=None,
                 count_mode="raw", length_normalize=False):
    """{alpha: case-vs-background ROC-AUC} of the max-over-nodes LR score, for each
    alpha in alpha_grid. The fork-settler vs the θ-mass detection AUC: LR-AUC ≫
    θ-AUC => signal present but buried (θ-mass was the wrong lens); LR-AUC ≈ θ-AUC
    => signal genuinely absent."""
    y = np.asarray(is_fg, dtype=int)
    out = {}
    for a in alpha_grid:
        s = lr_placement_scores(bow, lam, lay, alpha=float(a), background=background,
                                count_mode=count_mode, length_normalize=length_normalize)
        out[float(a)] = _auc(s.max(axis=1), y)
    return out


def _auc(scores, y):
    """Mann-Whitney (rank-sum) AUC. Ties in `scores` get AVERAGE (mid)ranks
    (scipy.stats.rankdata method='average'), so tied score blocks contribute
    0.5 per positive-negative pair — the correct ROC-AUC. (argsort's distinct
    ranks made a tie block read as 0 or 1 depending on row order.) One-class
    input -> nan."""
    from scipy.stats import rankdata
    y = np.asarray(y)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _average_precision(scores, y):
    """Average precision (area under the precision-recall curve, the step
    definition used by sklearn.metrics.average_precision_score): AP = sum_i
    (R_i - R_{i-1}) * P_i over distinct score thresholds i (descending). Tied
    scores share a threshold, so AP is order-invariant and a constant scorer
    yields AP == prevalence. No positives -> nan."""
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=float)
    n1 = y.sum()
    if n1 == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")          # desc, stable
    s, yy = scores[order], y[order]
    tp_cum = np.cumsum(yy)
    pp_cum = np.arange(1, len(yy) + 1)
    group_end = np.concatenate((s[1:] != s[:-1], [True]))  # last index of each tie group
    ends = np.where(group_end)[0]
    recall = tp_cum[ends] / n1
    precision = tp_cum[ends] / pp_cum[ends]
    r_prev = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - r_prev) * precision))


def _empirical_right_tail_p(values, reference):
    """Right-tail empirical p-value of each `values[i]` against an empirical
    `reference` sample: p = (#{reference >= value} + 1) / (n + 1). The +1/(n+1)
    plug (never 0) bounds the resolution at the reference size and keeps BH
    well-defined. Vectorised via searchsorted on the sorted reference."""
    ref = np.sort(np.asarray(reference, dtype=float))
    v = np.asarray(values, dtype=float)
    n = len(ref)
    if n == 0:
        return np.ones_like(v)
    ge = n - np.searchsorted(ref, v, side="left")     # count of ref >= v
    return (ge + 1.0) / (n + 1.0)


def _fdr_reject(pvals, q, method="bh"):
    """Step-up multiple-testing rejection at false-discovery-rate q.

    method='bh': Benjamini & Hochberg 1995 (JRSS-B 57:289) — reject the k largest
    ranks with p_(i) <= (i/m) q. method='by': Benjamini & Yekutieli 2001 (Ann.
    Statist. 29:1165) — the same with the harmonic penalty c(m)=sum_{i<=m} 1/i,
    valid under arbitrary dependence (conservative). Returns a boolean mask
    aligned to `pvals`."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    c = 1.0 if method == "bh" else float(np.sum(1.0 / np.arange(1, m + 1)))
    thresh = (np.arange(1, m + 1) / m) * (q / c)
    below = ranked <= thresh
    kmax = int(np.max(np.nonzero(below)[0])) + 1 if below.any() else 0
    reject = np.zeros(m, dtype=bool)
    if kmax:
        reject[order[:kmax]] = True
    return reject


def _zib_empirical_gap(values, *, zero_eps=1e-6):
    """Max-CDF-gap between a fitted zero-inflated Beta and the empirical CDF of
    `values` (node-block mass in [0,1]). The mixture is pi0 * 1[x<=0] + (1-pi0) *
    Beta(a,b) with pi0 the mass at <= zero_eps and Beta MLE (scipy) on the
    positive part. Returns a KS-style statistic in [0,1]; nan if degenerate
    (all-zero or <2 positive points). Diagnostic only: it decides whether the
    exportable null (sub-project 2) can be a ~3KB parametric fit or must ship a
    tail-dense empirical grid; the FDR p-values never use it."""
    v = np.sort(np.asarray(values, dtype=float))
    v = np.clip(v, 0.0, 1.0)
    n = len(v)
    if n == 0:
        return float("nan")
    pos = v[v > zero_eps]
    if len(pos) < 2 or np.ptp(pos) == 0.0:
        # <2 positive points, or a constant positive part (zero variance): no
        # Beta is fittable — scipy's method-of-moments would divide by the zero
        # variance and warn. The diagnostic is undefined here; return nan.
        return float("nan")
    pi0 = float(np.mean(v <= zero_eps))
    try:
        a, b, _, _ = _sps.beta.fit(pos, floc=0.0, fscale=1.0)
    except Exception:
        return float("nan")
    # Empirical CDF at each sorted v[i], tie-corrected: plain (i+1)/n over-weights
    # the *position within* a tied block (e.g. the point mass at 0) rather than
    # using the count of all tied members, which spuriously inflates the gap at
    # an atom. searchsorted(..., side="right") counts all values <= v[i].
    emp = np.searchsorted(v, v, side="right") / n
    fit = pi0 + (1.0 - pi0) * _sps.beta.cdf(v, a, b)      # mixture CDF (Beta.cdf(0)=0)
    return float(np.max(np.abs(emp - fit)))


def _assign_length_bins(lengths, ref_lengths, n_bins):
    """Assign each `lengths[i]` to a quantile bin (0..n_bins-1) of the reference
    length distribution `ref_lengths` (the background records). n_bins<=1 returns
    all zeros (the unconditioned null). Ties/degenerate quantiles collapse to
    fewer effective bins, which is harmless (a bin just holds more docs)."""
    lengths = np.asarray(lengths, dtype=float)
    if n_bins <= 1 or len(ref_lengths) == 0:
        return np.zeros(len(lengths), dtype=int)
    edges = np.quantile(np.asarray(ref_lengths, dtype=float),
                        np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    return np.digitize(lengths, edges).astype(int)


def per_node_discoveries(P, is_fg, doc_lengths, *, q_grid,
                         n_length_bins=4, method="bh"):
    """Length-conditioned, background-relative per-node discovery.

    P is the [n_docs x n_nodes] node-block affinity (profile mass per node). For
    each node u and length bin b, the null reference is the background docs'
    node-u mass in bin b; the per-doc p-value is the right-tail empirical p
    against that reference (Efron two-groups empirical null; the background arm is
    the null sample). Benjamini-Hochberg (or BY) is then applied per node column
    across all docs, giving a discovery set at each q in q_grid. Returns pmat, the
    floor mask (p at the 1/(n_ref+1) resolution floor), the per-q discovery masks,
    and the bin ids."""
    P = np.asarray(P, dtype=float)
    is_fg = np.asarray(is_fg, dtype=bool)
    n, n_nodes = P.shape
    ref_lengths = doc_lengths[~is_fg] if (~is_fg).any() else doc_lengths
    bins = _assign_length_bins(doc_lengths, ref_lengths, n_length_bins)
    pmat = np.ones((n, n_nodes))
    floor = np.zeros((n, n_nodes), dtype=bool)
    for b in np.unique(bins):
        in_b = bins == b
        ref_rows = in_b & (~is_fg)
        idx = np.nonzero(in_b)[0]
        for u in range(n_nodes):
            ref = P[ref_rows, u]
            if len(ref) == 0:
                continue
            p = _empirical_right_tail_p(P[idx, u], ref)
            pmat[idx, u] = p
            floor[idx, u] = p <= (1.0 / (len(ref) + 1.0) + 1e-12)
    discoveries = {}
    for q in q_grid:
        mask = np.zeros((n, n_nodes), dtype=bool)
        for u in range(n_nodes):
            mask[:, u] = _fdr_reject(pmat[:, u], q, method)
        discoveries[q] = mask
    return {"pmat": pmat, "floor": floor, "discoveries": discoveries, "bins": bins}


def _bootstrap_ci(P, fronts, lay, node_pos, *, n_boot=500, seed=0, max_docs=5000):
    """Percentile bootstrap 95% CIs for the headline metrics, resampling DOCS
    with replacement (docs are one-per-patient here, so this is a patient-
    clustered bootstrap). Returns {metric: (lo, hi)} for ap_macro, mrr, top2,
    recall_at_1.

    `recall_at_1` here is the SAME frontier-normalized recall the top-level
    `recall_at_k[1]` reports (|top-1 ∩ frontier| / |frontier|), not top-1 accuracy
    — so the CI brackets the metric it names even for multi-node (comorbid)
    frontiers. To keep cost bounded (the per-node AP is recomputed each replicate),
    the doc set is subsampled to at most `max_docs` before bootstrapping when
    larger; the CIs then reflect that subsample (wider, still valid). Fixed seed
    -> resume-stable."""
    rng = np.random.default_rng(seed)
    n = P.shape[0]
    nodes = lay.nodes
    sel = np.arange(n)
    if n > max_docs:
        sel = np.sort(rng.choice(n, size=max_docs, replace=False))
    Ps = P[sel]
    fs = [fronts[j] for j in sel]
    poss = {u: [node_pos[u][j] for j in sel] for u in nodes}
    m = len(sel)

    def _metrics(idx):
        Pb = Ps[idx]
        fb = [fs[j] for j in idx]
        posb = {u: [poss[u][j] for j in idx] for u in nodes}
        aps = [_average_precision(Pb[:, i], posb[u]) for i, u in enumerate(nodes)]
        aps = [a for a in aps if not np.isnan(a)]
        apm = float(np.mean(aps)) if aps else np.nan
        r1, ranks, top2 = [], [], []
        for i, f in enumerate(fb):
            ti = [nodes.index(t) for t in f if t in nodes]
            if not ti:
                continue
            top1 = int(np.argmax(Pb[i]))
            r1.append(len({top1} & set(ti)) / len(ti))     # matches recall_at_k[1]
            rk = min(1 + int((Pb[i] > Pb[i][j]).sum()) for j in ti)
            ranks.append(1.0 / rk)
            top2.append(1.0 if rk <= 2 else 0.0)
        return (apm,
                float(np.mean(ranks)) if ranks else np.nan,
                float(np.mean(top2)) if top2 else np.nan,
                float(np.mean(r1)) if r1 else np.nan)

    draws = {"ap_macro": [], "mrr": [], "top2": [], "recall_at_1": []}
    for _ in range(n_boot):
        idx = rng.integers(0, m, size=m)
        apm, mrr, top2, r1 = _metrics(idx)
        draws["ap_macro"].append(apm); draws["mrr"].append(mrr)
        draws["top2"].append(top2); draws["recall_at_1"].append(r1)
    out = {}
    for k, vals in draws.items():
        v = np.array([x for x in vals if not np.isnan(x)])
        out[k] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) \
            if len(v) else (float("nan"), float("nan"))
    return out


def _detection_metrics(case_score, is_fg, *, sens_targets=(0.80, 0.90, 0.95)):
    """Foreground-vs-background DETECTION: in a background-dominated cohort, can
    the model flag the rare true cases without flooding on false positives?

    This is the deployment metric — most patients are NOT in any rare-disease
    group; we want high affinity for those who should be placed and low affinity
    for those who should not. `case_score` is a per-doc scalar (higher = more
    case-like, e.g. the strongest single disease-node affinity); `is_fg` is the
    boolean truth (the patient truly belongs to >= 1 scoreable disease node).

    Reports discrimination — ROC-AUC and PR-AUC (the latter honest under the
    heavy class imbalance, floored at the foreground prevalence) — and, at each
    target foreground sensitivity, the operating point that achieves it: the
    score threshold, the realized sensitivity, the background false-positive rate
    (fraction of background patients wrongly flagged), the specificity, and the
    precision (PPV — of everyone flagged at this threshold, the fraction that are
    real cases, which the low prevalence makes the demanding number). Either
    class empty -> nan / no operating points."""
    s = np.asarray(case_score, dtype=float)
    y = np.asarray(is_fg, dtype=bool)
    n_fg, n_bg = int(y.sum()), int((~y).sum())
    out = {"auc": _auc(s, y.astype(int)),
           "ap": _average_precision(s, y.astype(int)),
           "prevalence": float(n_fg / len(y)) if len(y) else float("nan"),
           "n_foreground": n_fg, "n_background": n_bg,
           "operating_points": {}}
    if n_fg and n_bg:
        fg, bg = np.sort(s[y]), s[~y]
        for t in sens_targets:
            # Threshold = the k-th largest foreground score, k = ceil(t * n_fg),
            # i.e. the smallest score among the top-t fraction of foreground. No
            # np.quantile interpolation-kwarg version worries. Realized
            # sensitivity is >= t (ties may catch a few more).
            k = max(1, int(np.ceil(t * n_fg)))
            thr = float(fg[n_fg - k])
            sens = float(np.mean(s[y] >= thr))
            fpr = float(np.mean(bg >= thr))
            denom = sens * n_fg + fpr * n_bg
            ppv = float(sens * n_fg / denom) if denom > 0 else float("nan")
            out["operating_points"][f"{t:.2f}"] = {
                "threshold": thr, "sensitivity": sens, "bg_fpr": fpr,
                "specificity": 1.0 - fpr, "precision": ppv}
    return out


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


def evaluate(profiles, test_labels, lay, *, doc_lengths=None,
            fdr_q_grid=(0.05, 0.10, 0.20), n_length_bins=4):
    """Per-node case-finding AUC (subtree membership), AUC by longest-path depth, and set-valued
    ranking. `test_labels` entries may be frontier sets or scalars (scalars -> singletons). A patient
    is a positive for node u if any of its frontier lies in subtree(u). MRR/top2/mean_hops use the
    BEST (closest) true frontier node. frontier_size_mean and multi_frontier_rate instrument the
    comorbid/contradictory ambiguity (the DAG cannot tell those apart; we surface it, not resolve it).
    Profiles are the graded affinity dicts from `profile`; scoring never collapses to one node.

    Tie policy: node AUC/PR use midranks (see `_auc`). MRR/top2 count only nodes with STRICTLY
    greater affinity than the true node (best-rank-among-ties — optimistic, appropriate for a
    set-valued truth). mean_hops uses `argmax` (ties broken by node id).

    The `detection` block (see `_detection_metrics`) is the deployment metric: it scores whether the
    model separates the rare true cases from the background-dominated majority (foreground = any
    scoreable frontier node; background = empty/root-only frontier), with ROC/PR-AUC and, at target
    sensitivities, the background false-positive rate + precision. The node-level AUC/PR above already
    use background docs as the negative class; this block reports the case-vs-background question
    directly, plus the background-block topic mass per class."""
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
    have_ranks = len(ranks) > 0                          # some doc had a rankable true frontier node
    ranks = np.array(ranks, dtype=float) if ranks else np.array([np.nan])
    by_depth = {}
    for dep in sorted({lay.depth(u) for u in lay.nodes}):
        vals = [node_auc[u] for u in lay.nodes if lay.depth(u) == dep]
        # guard the all-nan slice (every node in the depth has an empty positive/negative class,
        # e.g. the degenerate all-root batch) so numpy does not warn on an empty mean
        by_depth[dep] = float(np.nanmean(vals)) if any(not np.isnan(v) for v in vals) else float("nan")
    # Guard mrr AND top2 symmetrically: with no rankable docs, both are nan (not applicable).
    # (np.nan <= 2 is False, so an unguarded top2 would silently read 0.0 instead of nan.)

    # --- PR-AUC (average precision) per node + summaries ---------------------
    node_pos = {u: [bool(f & lay.subtree(u)) for f in fronts] for u in lay.nodes}
    node_ap = {u: _average_precision(P[:, i], node_pos[u])
               for i, u in enumerate(lay.nodes)}
    valid_ap = {u: a for u, a in node_ap.items() if not np.isnan(a)}
    npos = {u: int(np.sum(node_pos[u])) for u in lay.nodes}
    ap_macro = float(np.mean(list(valid_ap.values()))) if valid_ap else float("nan")
    tot_pos = sum(npos[u] for u in valid_ap)
    ap_prevalence_weighted = (
        float(sum(valid_ap[u] * npos[u] for u in valid_ap) / tot_pos)
        if tot_pos else float("nan"))
    # micro AP: pool every (node, doc) pair into one ranking.
    flat_scores = P.reshape(-1)
    # NOTE order: reshape(-1) walks docs-major (row d, then node i); build labels to match.
    flat_labels = np.array([node_pos[lay.nodes[i]][d]
                            for d in range(P.shape[0]) for i in range(P.shape[1])])
    ap_micro = _average_precision(flat_scores, flat_labels)

    # --- recall@k over the FULL set-valued frontier -------------------------
    def _recall_at_k(k):
        rec = []
        for i, f in enumerate(fronts):
            true_idx = [lay.nodes.index(t) for t in f if t in lay.nodes]
            if not true_idx:
                continue
            topk = set(np.argsort(-P[i], kind="mergesort")[:k].tolist())
            rec.append(len(topk & set(true_idx)) / len(true_idx))
        return float(np.mean(rec)) if rec else float("nan")
    recall_at_k = {k: _recall_at_k(k) for k in (1, 2, 3)}

    # --- percentile bootstrap CIs (resample docs = patients; 1 doc/patient) --
    ci = _bootstrap_ci(P, fronts, lay, node_pos)

    # --- foreground-vs-background DETECTION (the deployment metric) ----------
    # A patient is a detection-positive iff it truly belongs to >= 1 scoreable
    # disease node (empty/root-only frontier = background). The case score is the
    # strongest single disease-node affinity (what you would threshold to flag a
    # candidate); disease_mass (total non-background topic mass) is reported as a
    # complementary aggregate score. Background-block mass = 1 - disease_mass
    # (the blocks partition [0,K), see DagLayout), summarized over each class to
    # show that background patients park their mass on the background topics.
    node_set = set(lay.nodes)
    is_fg = np.array([bool(f & node_set) for f in fronts])
    if P.size:
        disease_mass = P.sum(axis=1)
        case_score = P.max(axis=1)
    else:
        disease_mass = np.zeros(len(fronts))
        case_score = np.zeros(len(fronts))
    bg_mass = np.clip(1.0 - disease_mass, 0.0, 1.0)
    n_fg, n_bg = int(is_fg.sum()), int((~is_fg).sum())
    detection = _detection_metrics(case_score, is_fg)
    detection["auc_disease_mass"] = _auc(disease_mass, is_fg.astype(int))
    detection["ap_disease_mass"] = _average_precision(disease_mass, is_fg.astype(int))
    detection["bg_mass_background_mean"] = (
        float(bg_mass[~is_fg].mean()) if n_bg else float("nan"))
    detection["bg_mass_foreground_mean"] = (
        float(bg_mass[is_fg].mean()) if n_fg else float("nan"))

    # --- FDR readout: background-relative, per-node, multiple-testing corrected -
    # Post-hoc on P (node-block mass). Each (patient, node) is its own test; the
    # background docs are the empirical null (Efron two-groups); BH per node.
    node_list = lay.nodes
    lengths = (np.asarray(doc_lengths, dtype=float) if doc_lengths is not None
               else np.ones(len(fronts)))
    nlb = n_length_bins if doc_lengths is not None else 1
    q_grid = list(fdr_q_grid)
    disc = per_node_discoveries(P, is_fg, lengths, q_grid=q_grid,
                                n_length_bins=nlb)
    truth = np.array([[node_pos[u][i] for u in node_list]
                      for i in range(len(fronts))], dtype=bool)   # [n_docs x n_nodes]
    by_q = {}
    for q in q_grid:
        m = disc["discoveries"][q]
        ndisc = int(m.sum())
        tp = int((m & truth).sum())
        total_pos = int(truth.sum())
        by_q[q] = {
            "n_discoveries": ndisc,
            "precision": float(tp / ndisc) if ndisc else float("nan"),
            "recall": float(tp / total_pos) if total_pos else float("nan")}
    # multimorbidity payoff at the middle q, measured on CORRECT captures so it is
    # like-for-like: mean TRUE node-discoveries per truly-multimorbid patient (a
    # patient whose true frontier has >=2 scoreable nodes) vs the argmax true-hit
    # baseline. FDR can credit several true nodes at once (the simplex fix), while
    # argmax credits at most one node/patient (so its true-capture is <=1). The
    # raw total-discovery count (incl. false ones) is reported separately for
    # context, NOT as the headline — comparing a count to a rate would overstate.
    q_mid = q_grid[len(q_grid) // 2]
    mm_rows = np.array([len(f & set(node_list)) >= 2 for f in fronts])
    if mm_rows.any():
        m_mid = disc["discoveries"][q_mid]
        mean_true_disc = float((m_mid & truth)[mm_rows].sum(axis=1).mean())
        mean_total_disc = float(m_mid[mm_rows].sum(axis=1).mean())
        argmax_node = np.argmax(P[mm_rows], axis=1)
        argmax_tp = truth[mm_rows][np.arange(mm_rows.sum()), argmax_node]
        argmax_base = float(argmax_tp.mean())
    else:
        mean_true_disc = mean_total_disc = argmax_base = float("nan")
    gaps = [_zib_empirical_gap(P[~is_fg, u]) for u in range(len(node_list))]
    gaps = [g for g in gaps if not np.isnan(g)]
    fdr_block = {
        "q_grid": q_grid,
        "by_q": by_q,
        "multimorbidity": {
            "mean_true_discoveries_per_multimorbid": mean_true_disc,
            "argmax_true_baseline_per_multimorbid": argmax_base,
            "mean_total_discoveries_per_multimorbid": mean_total_disc},
        "saturation_rate": float(disc["floor"][disc["discoveries"][q_mid]].mean())
            if disc["discoveries"][q_mid].any() else float("nan"),
        "zib_gap_mean": float(np.mean(gaps)) if gaps else float("nan"),
        "zib_gap_max": float(np.max(gaps)) if gaps else float("nan"),
        "n_length_bins_effective": int(len(np.unique(disc["bins"]))),
    }

    return {"node_auc": node_auc, "auc_by_depth": by_depth,
            "mrr": float(np.nanmean(1.0 / ranks)) if have_ranks else float("nan"),
            "top2": float(np.nanmean(ranks <= 2)) if have_ranks else float("nan"),
            "mean_hops": float(np.mean(hops)) if hops else float("nan"),
            "frontier_size_mean": float(np.mean([len(f) for f in fronts])),
            "multi_frontier_rate": float(np.mean([len(f) > 1 for f in fronts])),
            "node_ap": node_ap, "ap_macro": ap_macro, "ap_micro": ap_micro,
            "ap_prevalence_weighted": ap_prevalence_weighted,
            "recall_at_k": recall_at_k, "ci": ci, "detection": detection,
            "fdr": fdr_block}


def _node_topic_mean(beta_hat, lay, u):
    return beta_hat[lay.block[u]].mean(0)


def identifiability_annotation(beta_hat, lay, *, tol=0.9):
    """Post-fit diagnostic: flag WITHIN-STRUCTURE node pairs (siblings, or parent<->child) whose
    learned topic distributions are near-collinear (cosine >= tol) -> hard to separate. Cross-branch
    pairs are never reported; their similarity is a reporting fact, not a structural one."""
    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    pairs = set()
    for c, ps in lay.parents.items():                    # every parent<->child edge
        for p in ps:
            if p != 0:
                pairs.add((min(p, c), max(p, c)))
    for p, kids in lay.children.items():                 # siblings sharing at least one parent
        for i in range(len(kids)):
            for j in range(i + 1, len(kids)):
                pairs.add((min(kids[i], kids[j]), max(kids[i], kids[j])))
    out = []
    for u, v in pairs:
        c = cos(_node_topic_mean(beta_hat, lay, u), _node_topic_mean(beta_hat, lay, v))
        if c >= tol:
            out.append((u, v, c))
    return out


def render_profile(affinity, lay, *, names=None, true_node=None, width=24):
    """Indented DAG tree with a unicode affinity bar per node (spot-check output). A multi-parent
    node is rendered in full ONCE (first encounter); later encounters show a short reference line so
    the tree stays readable and no node's affinity is double-counted visually. `true_node` may be a
    single node id OR an iterable of node ids (a set-valued frontier); EVERY true node is marked."""
    names = names or {}
    if true_node is None:
        true_set = set()
    elif hasattr(true_node, "__iter__"):
        true_set = set(true_node)
    else:
        true_set = {true_node}
    lines = []
    seen = set()

    def bar(x):
        n = int(round(max(0.0, min(1.0, x)) * width))
        return "█" * n + "▁" * (width - n)

    def walk(v, prefix, is_last):
        if v == 0:
            lines.append(names.get(0, "root"))
        else:
            conn = "└─ " if is_last else "├─ "
            nm = str(names.get(v, v)).ljust(10)
            if v in seen:                                 # multi-parent: reference, do not re-render
                lines.append(f"{prefix}{conn}{nm} (^ shared)")
                return
            seen.add(v)
            a = affinity.get(v, 0.0)
            mark = "  <- true" if v in true_set else ""
            lines.append(f"{prefix}{conn}{nm} {bar(a)} {a:0.2f}{mark}")
        kids = lay.children.get(v, [])
        child_prefix = prefix + ("   " if is_last else "│  ") if v != 0 else ""
        for i, c in enumerate(kids):
            walk(c, child_prefix, i == len(kids) - 1)

    walk(0, "", True)
    return "\n".join(lines)
