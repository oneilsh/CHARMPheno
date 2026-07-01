"""Synthetic STM corpora (non-gated + gated) + in-process fit + ground-truth
recovery. Domain-agnostic: integer token ids only."""
from __future__ import annotations
import numpy as np
from spark_vi.models.topic.stm import OnlineSTM
from spark_vi.models.topic.types import STMDocument
from spark_vi.models.topic.partition import TopicBlockPartition


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
