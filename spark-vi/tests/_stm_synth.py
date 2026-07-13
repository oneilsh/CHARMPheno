"""Synthetic STM corpora (non-gated + gated) + in-process fit + ground-truth
recovery. Domain-agnostic: integer token ids only."""
from __future__ import annotations
import numpy as np
from spark_vi.models.topic.stm import OnlineSTM
from spark_vi.models.topic.types import STMDocument
from spark_vi.models.topic.partition import TopicBlockPartition
from spark_vi.models.topic._linalg import pd_complete


def _block_of(row, *, eps=1e-3):
    return np.where(row > eps)[0]


def planted_recovery(beta_hat, planted_beta, *, thresh=0.5):
    n = 0
    for k in range(planted_beta.shape[0]):
        if beta_hat[:, _block_of(planted_beta[k])].sum(axis=1).max() >= thresh:
            n += 1
    return n


def foreground_recovers_group(beta_hat, partition, group, planted_beta, *,
                              thresh=0.5):
    fg = partition.block_indices(group)
    # planted foreground rows for this group sit in the same slot indices
    for k in fg:
        block = _block_of(planted_beta[k])
        if len(block) and beta_hat[fg][:, block].sum(axis=1).max() >= thresh:
            return True
    return False


def final_sigma_range(gp):
    s = gp["Sigma"]; return float(s.min()), float(s.max())


def synthetic_ehr_corpus(*, K_rare, V, D, doc_len, bg_frac, seed=0):
    rng = np.random.default_rng(seed)
    BG_V = V // 2
    bg = np.full(V, 1e-4); bg[:BG_V] = rng.random(BG_V) + 0.1; bg /= bg.sum()
    bs = (V - BG_V) // K_rare
    planted = np.full((K_rare, V), 1e-4)
    for k in range(K_rare):
        planted[k, BG_V + k * bs: BG_V + (k + 1) * bs] += 1.0
    planted /= planted.sum(axis=1, keepdims=True)
    docs = []
    for _ in range(D):
        k = int(rng.integers(K_rare)); n_bg = int(rng.binomial(doc_len, bg_frac))
        toks = np.concatenate([rng.choice(V, size=n_bg, p=bg),
                               rng.choice(V, size=doc_len - n_bg, p=planted[k])])
        u, c = np.unique(toks, return_counts=True)
        docs.append(STMDocument(indices=u.astype(np.int32),
                                counts=c.astype(np.float64),
                                length=int(c.sum()), x=np.array([1.0])))
    return docs, planted


def synthetic_gated_corpus(*, groups, fg_per_group, bg_k, V, D, doc_len,
                           bg_frac, seed=0):
    rng = np.random.default_rng(seed)
    part = TopicBlockPartition(group_var="g", background_k=bg_k,
                               foreground=tuple((g, fg_per_group) for g in groups))
    K = part.K
    # Vocab layout: background region [0:V//2], then a disjoint region per
    # (group, fg-topic). planted[k] aligns with partition slot k.
    BG_V = V // 2
    rest = V - BG_V
    n_fg = len(groups) * fg_per_group
    fb = rest // max(n_fg, 1)
    planted = np.full((K, V), 1e-4)
    bg_rows = part.background_indices()
    for j, k in enumerate(bg_rows):           # background topics over [0:BG_V]
        planted[k, (j * (BG_V // bg_k)):((j + 1) * (BG_V // bg_k))] += 1.0
    fg_slot = 0
    for g in groups:                          # each group's foreground block
        for k in part.block_indices(g):
            lo = BG_V + fg_slot * fb
            planted[k, lo:lo + fb] += 1.0
            fg_slot += 1
    planted /= planted.sum(axis=1, keepdims=True)
    docs = []
    glist = list(groups)
    for _ in range(D):
        g = glist[int(rng.integers(len(glist)))]
        allowed = part.allowed_indices(frozenset({g}))
        # doc mixes background topics + this group's foreground topics
        bg_topic = bg_rows[int(rng.integers(len(bg_rows)))]
        fg_topics = part.block_indices(g)
        fg_topic = fg_topics[int(rng.integers(len(fg_topics)))]
        n_bg = int(rng.binomial(doc_len, bg_frac))
        toks = np.concatenate([
            rng.choice(V, size=n_bg, p=planted[bg_topic]),
            rng.choice(V, size=doc_len - n_bg, p=planted[fg_topic])])
        u, c = np.unique(toks, return_counts=True)
        docs.append(STMDocument(indices=u.astype(np.int32),
                                counts=c.astype(np.float64), length=int(c.sum()),
                                x=np.array([1.0]), groups=frozenset({g})))
    return docs, planted, part


def topic_support_jaccard(beta, *, eps=1e-3):
    """Mean pairwise Jaccard overlap of topic supports (concepts with prob > eps).

    A separation diagnostic: 0 = every topic uses a disjoint vocabulary (the
    artificial-separation regime of synthetic_gated_corpus); ->1 = topics share
    all terms. The real HF lda_pasc beta measures ~0.35 here."""
    supports = [set(np.where(beta[k] > eps)[0]) for k in range(beta.shape[0])]
    vals = []
    for i in range(len(supports)):
        for j in range(i + 1, len(supports)):
            uni = len(supports[i] | supports[j])
            vals.append(len(supports[i] & supports[j]) / uni if uni else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def synthetic_gated_corpus_overlap(*, groups, fg_per_group, bg_k, V, D, doc_len,
                                   bg_frac, shared_frac=0.5, shared_pool=None,
                                   seed=0):
    """Gated corpus whose topics SHARE vocabulary (calibrated to the real HF
    beta's mean pairwise Jaccard ~0.35), unlike synthetic_gated_corpus's disjoint
    per-topic blocks.

    Vocab layout: a shared common pool [0:C] that EVERY topic samples from (the
    'hypertension shows up everywhere' effect) plus a disjoint signature block per
    topic in [C:V]. `shared_frac` is the probability mass each topic places on the
    shared pool; the pool SIZE defaults to one signature block so Jaccard lands
    ~1/3 regardless of K (Jaccard = C/(C+2*sig)).

    Documents co-activate a fixed SPINE background topic (slot bg[0], present in
    every doc of BOTH groups) + one random other background topic + one of the
    doc's own group's foreground topics. The spine drives a strong background<->A
    and background<->B Sigma coupling while NO doc co-activates an A and a B
    foreground topic, so the A<->B cross-pair is structurally unobserved (free) —
    the block-arrow inconsistency the PD completion must repair."""
    rng = np.random.default_rng(seed)
    part = TopicBlockPartition(group_var="g", background_k=bg_k,
                               foreground=tuple((g, fg_per_group) for g in groups))
    K = part.K
    sig_region = int(round(V * (1.0 - shared_frac)))
    sig = max(1, sig_region // K)
    C = int(shared_pool) if shared_pool is not None else sig   # pool ~ one block
    C = min(C, V - K * sig)                                     # keep blocks in range
    C = max(C, 1)
    planted = np.full((K, V), 1e-4)
    # shared common pool: every topic, per-topic random weights (not identical)
    for k in range(K):
        planted[k, 0:C] += rng.random(C) + 0.1
    # disjoint signature block per topic in [C:V]
    for k in range(K):
        lo = C + k * sig
        planted[k, lo:lo + sig] += 5.0
    planted /= planted.sum(axis=1, keepdims=True)

    bg_rows = part.background_indices()
    spine = bg_rows[0]
    docs = []
    glist = list(groups)
    for _ in range(D):
        g = glist[int(rng.integers(len(glist)))]
        other_bg = bg_rows[int(rng.integers(len(bg_rows)))]
        fg_topics = part.block_indices(g)
        fg = fg_topics[int(rng.integers(len(fg_topics)))]
        n_bg = int(rng.binomial(doc_len, bg_frac))
        n_other = n_bg // 2
        toks = np.concatenate([
            rng.choice(V, size=n_bg - n_other, p=planted[spine]),
            rng.choice(V, size=n_other, p=planted[other_bg]),
            rng.choice(V, size=doc_len - n_bg, p=planted[fg])])
        u, c = np.unique(toks, return_counts=True)
        docs.append(STMDocument(indices=u.astype(np.int32),
                                counts=c.astype(np.float64), length=int(c.sum()),
                                x=np.array([1.0]), groups=frozenset({g})))
    return docs, planted, part


def gated_ln_corpus(*, group_weights, fg_per_group, bg_k, V, D, doc_len,
                    eta_scale: float = 1.0, seed=0):
    """Single-label gated logistic-normal corpus with a KNOWN unit-diagonal Sigma_true.

    Each doc belongs to ONE group (sampled by group_weights, so a minority arm is
    thin); its allowed set is background ∪ that group's foreground block. eta over the
    allowed set ~ N(0, Sigma_true[A,A]); theta = softmax(eta); words ~ theta·beta.
    Planted correlations (bg-bg 0.10, bg-fg 0.25, within-fg 0.30) are made PD via the
    max-det completion of the cross-foreground free block (pd_complete used ONLY to
    build ground truth, not in the fit). Domain-agnostic: integer token ids only.

    ``eta_scale`` multiplies the PLANTED covariance used to draw eta (the true
    generative variance level, e.g. for the refit-dynamics experiment where the
    calibration target is a KNOWN scale != 1); the RETURNED Sigma_true is scaled
    identically, so callers always see the true generative covariance the draws
    actually came from. Default 1.0 reproduces the original unit-scale draws and
    return value byte-for-byte."""
    rng = np.random.default_rng(seed)
    groups = tuple(group_weights)
    part = TopicBlockPartition(group_var="g", background_k=bg_k,
                              foreground=tuple((g, fg_per_group) for g in groups))
    K = part.K
    sig = max(1, (V // 2) // K)
    C = max(1, min(sig, V - K * sig))
    beta = np.full((K, V), 1e-3)
    for k in range(K):
        beta[k, 0:C] += rng.random(C) + 0.1
        lo = C + k * sig
        beta[k, lo:lo + sig] += 5.0
    beta /= beta.sum(axis=1, keepdims=True)

    bg = part.background_indices()
    Sigma_true = np.eye(K); obs = np.eye(K, dtype=bool)
    for a in bg:
        for b in bg:
            if a != b:
                Sigma_true[a, b] = 0.10; obs[a, b] = True
    for a in bg:
        for g in groups:
            for c in part.block_indices(g):
                Sigma_true[a, c] = Sigma_true[c, a] = 0.25
                obs[a, c] = obs[c, a] = True
    for g in groups:
        blk = part.block_indices(g)
        for i in blk:
            for j in blk:
                if i != j:
                    Sigma_true[i, j] = 0.30; obs[i, j] = True
    Sigma_true = pd_complete(Sigma_true, obs)
    Sigma_true = float(eta_scale) * Sigma_true

    gl = list(groups)
    wts = np.array([group_weights[g] for g in gl], float); wts /= wts.sum()
    docs = []
    for _ in range(D):
        g = gl[int(rng.choice(len(gl), p=wts))]
        allowed = sorted(part.allowed_indices(frozenset({g})))
        eta = rng.multivariate_normal(np.zeros(len(allowed)),
                                      Sigma_true[np.ix_(allowed, allowed)])
        theta = np.zeros(K)
        theta[allowed] = np.exp(eta - eta.max()); theta /= theta.sum()
        toks = rng.choice(V, size=doc_len, p=theta @ beta)
        u, c = np.unique(toks, return_counts=True)
        docs.append(STMDocument(indices=u.astype(np.int32),
                                counts=c.astype(np.float64), length=int(c.sum()),
                                x=np.array([1.0]), groups=frozenset({g})))
    return docs, part, Sigma_true, beta


def gated_ln_corpus_overlap(*, group_weights, fg_per_group, bg_k, V, D, doc_len,
                            shared_frac=0.5, eta_scale=1.0, seed=0):
    """Logistic-normal gated corpus with a KNOWN scale AND realistic vocabulary
    overlap-by-MASS -- the regime real EHR data lives in.

    Combines gated_ln_corpus's scaled logistic-normal draws (eta over the doc's
    allowed set ~ N(0, eta_scale*Sigma_true), theta = softmax) with a beta that
    places `shared_frac` of EVERY topic's probability mass on a common shared pool
    [0:C] (the 'common codes appear in every phenotype' effect) and the rest on a
    disjoint per-topic signature block. Unlike gated_ln_corpus (which puts ~89% of
    mass on signatures, so beta trivially separates topics and ABSORBS the
    per-document concentration, leaving the generative scale nothing to do -- see
    feedback_synthetic_vocab_overlap / exp 0040), a high shared-mass fraction keeps
    most tokens ambiguous, so beta CANNOT resolve the topic from tokens alone and
    the prior/scale must do real work. Support-Jaccard lands ~1/3 (pool size = one
    signature block) matching the real HF beta ~0.35; verify per-run with
    topic_support_jaccard.

    shared_frac is the MASS fraction on the shared pool (0.5 = half of every doc's
    expected tokens are ambiguous). eta_scale scales the planted covariance for the
    eta draws (the generative scale to recover); the returned Sigma_true is the
    scaled covariance. Returns docs, part, Sigma_true (= eta_scale*R_true), beta."""
    rng = np.random.default_rng(seed)
    groups = tuple(group_weights)
    part = TopicBlockPartition(group_var="g", background_k=bg_k,
                               foreground=tuple((g, fg_per_group) for g in groups))
    K = part.K
    sig = max(1, V // (K + 1))          # pool size = one sig block -> Jaccard ~1/3
    C = min(sig, V - K * sig)
    C = max(C, 1)
    # beta as probabilities: shared_frac mass spread (per-topic random) over the
    # shared pool [0:C], (1-shared_frac) over the topic's own signature block.
    beta = np.full((K, V), 1e-6)
    for k in range(K):
        w = rng.random(C) + 0.05
        beta[k, 0:C] = shared_frac * (w / w.sum())
        lo = C + k * sig
        s = rng.random(sig) + 0.05
        beta[k, lo:lo + sig] = (1.0 - shared_frac) * (s / s.sum())
    beta /= beta.sum(axis=1, keepdims=True)

    # Same planted correlation structure as gated_ln_corpus, scaled by eta_scale.
    bg = part.background_indices()
    R_true = np.eye(K); obs = np.eye(K, dtype=bool)
    for a in bg:
        for b in bg:
            if a != b:
                R_true[a, b] = 0.10; obs[a, b] = True
    for a in bg:
        for g in groups:
            for c in part.block_indices(g):
                R_true[a, c] = R_true[c, a] = 0.25
                obs[a, c] = obs[c, a] = True
    for g in groups:
        blk = part.block_indices(g)
        for i in blk:
            for j in blk:
                if i != j:
                    R_true[i, j] = 0.30; obs[i, j] = True
    R_true = pd_complete(R_true, obs)
    Sigma_true = float(eta_scale) * R_true

    gl = list(groups)
    wts = np.array([group_weights[g] for g in gl], float); wts /= wts.sum()
    docs = []
    for _ in range(D):
        g = gl[int(rng.choice(len(gl), p=wts))]
        allowed = sorted(part.allowed_indices(frozenset({g})))
        eta = rng.multivariate_normal(np.zeros(len(allowed)),
                                      Sigma_true[np.ix_(allowed, allowed)])
        theta = np.zeros(K)
        theta[allowed] = np.exp(eta - eta.max()); theta /= theta.sum()
        toks = rng.choice(V, size=doc_len, p=theta @ beta)
        u, c = np.unique(toks, return_counts=True)
        docs.append(STMDocument(indices=u.astype(np.int32),
                                counts=c.astype(np.float64), length=int(c.sum()),
                                x=np.array([1.0]), groups=frozenset({g})))
    return docs, part, Sigma_true, beta


def gated_ln_corpus_stick(*, group_weights, fg_per_group, bg_k, V, D, doc_len,
                          rho_bg=0.3, rho_grp=0.3, rho_cross=0.2, eta_scale=1.0,
                          topic_overlap=0.0, seed=0):
    """STICK-NATIVE gated corpus: draw psi ~ N(0, Sigma_true) in the model's own
    (K-1)-dim STICK space and compose theta = gated_theta(psi), so the planted Sigma is
    IDENTIFIED by the likelihood (unlike gated_ln_corpus, which plants eta in SOFTMAX
    space and leaves the stick-space Sigma only weakly identified — the confound behind
    the Task-7 VI-vs-Gibbs xfail). This is the corpus on which VI-vs-Gibbs Sigma
    agreement is a meaningful pass/fail gate and true Sigma-recovery (not just beta) can
    be tested.

    Construction mirrors the model exactly (see pg_stm.stick_layout / gated_theta):

      * Sigma_true is (K-1)x(K-1) in stick space, block-structured like the estimator's
        _assemble_sigma: a shared background-stick block (indices 0..B-2), one
        [gate_g, fg_g] block per group, background<->group cross-terms, and the
        never-co-active group<->group' entries filled by the max-det PD completion
        (pd_complete — used ONLY to build a valid PD ground truth; those entries never
        enter any doc's draw). Unit diagonal, off-diagonals rho_bg / rho_grp / rho_cross,
        scaled by eta_scale.
      * For a group-g doc, the ACTIVE stick sub-vector [psi_bg (B-1), psi_gate,
        psi_fg (m_g-1)] ~ N(0, Sigma_true[active, active]); theta over the doc's allowed
        topics = gated_theta(psi_bg, psi_gate, psi_fg) (background topics 0..B-1 then the
        group's m_g foreground topics), placed into the length-K theta at those global
        indices; tokens ~ Multinomial(doc_len, theta @ beta).

    Domain-agnostic: integer token ids only. Returns docs, part, Sigma_true (stick-space,
    (K-1)x(K-1), == eta_scale * planted correlation), beta."""
    from spark_vi.models.topic.pg_stm import stick_layout, gated_theta

    rng = np.random.default_rng(seed)
    groups = tuple(group_weights)
    part = TopicBlockPartition(group_var="g", background_k=bg_k,
                               foreground=tuple((g, fg_per_group) for g in groups))
    K = part.K
    lay = stick_layout(part)

    # beta: same planted topic-word structure as gated_ln_corpus (a shared common pool
    # [0:C] + a per-topic signature block), so beta is recoverable and topic identity is
    # not degenerate. topic_overlap in [0,1) widens each signature window symmetrically by
    # round(topic_overlap*sig) words into its neighbors, so adjacent topics SHARE
    # vocabulary (realistic overlapping phenotypes); topic_overlap=0 -> disjoint blocks
    # (backward compatible).
    sig = max(1, (V // 2) // K)
    C = max(1, min(sig, V - K * sig))
    extra = int(round(float(topic_overlap) * sig))
    sig_lo, sig_hi = C, C + K * sig
    beta = np.full((K, V), 1e-3)
    for k in range(K):
        beta[k, 0:C] += rng.random(C) + 0.1
        lo = max(C + k * sig - extra, sig_lo)
        hi = min(C + k * sig + sig + extra, sig_hi)
        beta[k, lo:hi] += 5.0
    beta /= beta.sum(axis=1, keepdims=True)

    # Sigma_true in STICK space (dimension K-1), block-structured + PD-completed.
    Ksm1 = K - 1
    bg_sticks = lay["bg_sticks"]
    Sigma_true = np.eye(Ksm1)
    obs = np.eye(Ksm1, dtype=bool)
    for a in bg_sticks:
        for b in bg_sticks:
            if a != b:
                Sigma_true[a, b] = rho_bg; obs[a, b] = True
    for g in groups:
        gl = lay["groups"][g]
        block = np.concatenate([[gl["gate"]], gl["fg_sticks"]]).astype(np.int64)
        for i in block:
            for j in block:
                if i != j:
                    Sigma_true[i, j] = rho_grp; obs[i, j] = True
        for a in bg_sticks:
            for c in block:
                Sigma_true[a, c] = Sigma_true[c, a] = rho_cross
                obs[a, c] = obs[c, a] = True
    Sigma_true = pd_complete(Sigma_true, obs)
    Sigma_true = float(eta_scale) * Sigma_true

    gl_list = list(groups)
    wts = np.array([group_weights[g] for g in gl_list], float); wts /= wts.sum()
    nb = len(bg_sticks)                         # B-1 background sticks
    docs = []
    for _ in range(D):
        g = gl_list[int(rng.choice(len(gl_list), p=wts))]
        active = lay["groups"][g]["active"]     # [bg_sticks, gate_g, fg_g_sticks]
        psi = rng.multivariate_normal(np.zeros(active.shape[0]),
                                      Sigma_true[np.ix_(active, active)])
        psi_bg, psi_gate, psi_fg = psi[:nb], psi[nb], psi[nb + 1:]
        theta_allowed = gated_theta(psi_bg, psi_gate, psi_fg)   # [bg topics, fg topics]
        allowed = np.concatenate([part.background_indices(),
                                  part.block_indices(g)]).astype(np.int64)
        theta = np.zeros(K); theta[allowed] = theta_allowed
        toks = rng.choice(V, size=doc_len, p=theta @ beta)
        u, c = np.unique(toks, return_counts=True)
        docs.append(STMDocument(indices=u.astype(np.int32),
                                counts=c.astype(np.float64), length=int(c.sum()),
                                x=np.array([1.0]), groups=frozenset({g})))
    return docs, part, Sigma_true, beta


def real_beta_from(K, V, *, source=None, seed=0):
    """Topic-word matrix (K,V) for DAG plants. If ``source`` names an export bundle, load
    its beta (realistic overlap by construction). Otherwise (default) synthesize a
    REALISTIC-OVERLAP beta via gated_ln_corpus_stick(topic_overlap=0.6) — the honest
    stand-in until a real bundle is wired; the caller's test docstring must not claim
    transfer from it."""
    if source is not None:
        import numpy as _np
        return _np.load(source)["beta"]
    # borrow a realistic-overlap beta shaped (K, V)
    gw = {"A": 0.5, "B": 0.5}
    fg = max(1, (K - 2) // 2)
    _d, _p, _S, beta = gated_ln_corpus_stick(
        group_weights=gw, fg_per_group=fg, bg_k=K - 2 * fg, V=V, D=4, doc_len=10,
        topic_overlap=0.6, seed=seed)
    return beta[:K] if beta.shape[0] >= K else np.pad(beta, ((0, K - beta.shape[0]), (0, 0)))


def dag_offset_corpus(*, dag, node_offsets, partition, beta, node_of_group,
                      doc_nodes_plan, sigma_true, doc_len, seed):
    """Plant additive node offsets on a DagGate and generate a gated corpus.

    For a document at most-specific node v (anchor = the top-level node on v's root
    path, mapped back to a partition group via ``node_of_group`` inverse), the mean over
    its ACTIVE sticks is mu = sum_{u in closure(v)} node_offsets[u]; psi ~ N(mu[active],
    sigma_true[active,active]); theta = gated_theta(psi split into bg/gate/fg); tokens ~
    Multinomial(doc_len, theta @ beta). ``doc_nodes_plan`` maps node id -> #docs at that
    node. Returns (docs, doc_nodes) with doc.groups = {anchor group} and doc_nodes[d] =
    frozenset({v}). Domain-agnostic (integer ids only)."""
    from spark_vi.models.topic.pg_stm import stick_layout, gated_theta
    rng = np.random.default_rng(seed)
    lay = stick_layout(partition)
    group_of_node = {nid: g for g, nid in node_of_group.items()}
    # anchor(v) = the child-of-root on v's path (the node whose parent chain hits an anchor id)
    anchor_ids = set(node_of_group.values())

    def anchor_of(v):
        chain = [v] + sorted(dag.ancestors(v))
        for c in chain:
            if c in anchor_ids:
                return c
        raise ValueError(f"node {v} has no anchor ancestor")

    docs, doc_nodes = [], []
    nb = len(lay["bg_sticks"])
    for v, n_docs in doc_nodes_plan.items():
        g = group_of_node[anchor_of(v)]
        active = lay["groups"][g]["active"]
        allowed = np.concatenate([partition.background_indices(),
                                  partition.block_indices(g)]).astype(np.int64)
        mu_full = np.zeros(partition.K - 1)
        for u in dag.closure(frozenset({v})):
            mu_full = mu_full + node_offsets[u]
        mu_a = mu_full[active]
        Sa = sigma_true[np.ix_(active, active)]
        for _ in range(n_docs):
            psi = rng.multivariate_normal(mu_a, Sa)
            psi_bg, psi_gate, psi_fg = psi[:nb], psi[nb], psi[nb + 1:]
            theta_allowed = gated_theta(psi_bg, psi_gate, psi_fg)
            theta = np.zeros(partition.K); theta[allowed] = theta_allowed
            toks = rng.choice(partition.beta_dim if hasattr(partition, "beta_dim") else beta.shape[1],
                              size=doc_len, p=theta @ beta)
            u_, c_ = np.unique(toks, return_counts=True)
            docs.append(STMDocument(indices=u_.astype(np.int32), counts=c_.astype(np.float64),
                                    length=int(c_.sum()), x=np.array([1.0]),
                                    groups=frozenset({g})))
            doc_nodes.append(frozenset({v}))
    return docs, doc_nodes


def fit_stm(docs, *, K, V, sigma_init, n_iter=250, batch=None, seed=42,
            partition=None, init_data=None, **model_kwargs):
    m = OnlineSTM(K=K, vocab_size=V, P=1, sigma_init=sigma_init,
                  random_seed=seed, topic_blocks=partition, **model_kwargs)
    gp = m.initialize_global(init_data)
    if batch is None:
        for _ in range(n_iter):
            gp = m.update_global(gp, m.local_update(docs, gp), learning_rate=1.0)
        return gp
    D = len(docs); rng = np.random.default_rng(seed); scale = D / batch
    for t in range(n_iter):
        idx = rng.choice(D, size=batch, replace=False)
        stats = m.local_update([docs[i] for i in idx], gp)
        scaled = {kk: (v * scale if isinstance(v, (np.ndarray, int, float)) else v)
                  for kk, v in stats.items()}
        gp = m.update_global(gp, scaled, learning_rate=(t + 64) ** -0.7)
    return gp
