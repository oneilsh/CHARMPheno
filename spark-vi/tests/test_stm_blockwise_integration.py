import numpy as np
from spark_vi.models.topic.stm import OnlineSTM
from _stm_synth import (gated_ln_corpus, planted_recovery,
                        foreground_recovers_group)


def _fit(cls, docs, part, V, beta, n_iter=150, batch=120, seed=42):
    m = cls(K=part.K, vocab_size=V, P=1, random_seed=seed,
            topic_blocks=part, min_pair_support=5, reference_topic=True)
    gp = m.initialize_global({"spectral_beta": beta})     # aligned init
    D = len(docs); rng = np.random.default_rng(seed); scale = D / batch
    max_var = 0.0
    for t in range(n_iter):
        idx = rng.choice(D, size=batch, replace=False)
        stats = m.local_update([docs[i] for i in idx], gp)
        scaled = {k: (v * scale if isinstance(v, np.ndarray) and k.endswith("stat")
                      else v) for k, v in stats.items()}
        gp = m.update_global(gp, scaled, learning_rate=(t + 64) ** -0.7)
        max_var = max(max_var, float(np.diag(gp["Sigma"]).max()))
    return gp, max_var


def test_blockwise_no_runaway_and_recovers_subphenotypes():
    docs, part, Sigma_true, beta = gated_ln_corpus(
        group_weights={"A": 0.92, "B": 0.08}, fg_per_group=4, bg_k=6,
        V=200, D=2000, doc_len=80, seed=0)
    gp, max_var = _fit(OnlineSTM, docs, part, 200, beta)
    # (1) no variance runaway: diagonal pinned at 1 throughout
    assert max_var <= 1.0 + 1e-9, f"variance ran away to {max_var}"
    # (2) sub-phenotype recovery holds, including the thin minority arm
    bhat = gp["lambda"] / gp["lambda"].sum(axis=1, keepdims=True)
    assert planted_recovery(bhat, beta) >= part.K - 1
    assert foreground_recovers_group(bhat, part, "B", beta)
    # (3) within-group correlations are recovered (unit-diagonal => Σ IS R)
    Sig = gp["Sigma"]
    a, b = part.block_indices("A")[0], part.block_indices("A")[1]
    assert Sig[a, b] > 0.05, "within-group A correlation not recovered"


def test_blockwise_marginals_are_pd():
    docs, part, Sigma_true, beta = gated_ln_corpus(
        group_weights={"A": 0.92, "B": 0.08}, fg_per_group=4, bg_k=6,
        V=200, D=2000, doc_len=80, seed=1)
    gp, _ = _fit(OnlineSTM, docs, part, 200, beta)
    Sig = gp["Sigma"]; bg = list(part.background_indices())
    for g in part.groups:
        allowed = sorted(set(bg) | set(part.block_indices(g)))
        assert np.linalg.eigvalsh(Sig[np.ix_(allowed, allowed)]).min() > 0
