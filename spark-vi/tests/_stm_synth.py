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
