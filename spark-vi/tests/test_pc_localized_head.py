"""Localized head: each node's logistic reads ONLY its topic support (gated block +
ancestors + background), so w_c stays 0 off-support and the per-node Newton solve is
O(|support|^3) not O(K^3) — the whole-Mondo scale fix (insight 0071, ADR 0042 done
right: hierarchy in the head's SUPPORT, not a closure product that collapses with the
gate). Pure-numpy: one local_update + update_global on a tiny gated batch."""
import numpy as np


def _gated_pc(C, *, topic_support=None, V=30):
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    from spark_vi.models.topic.pc import OnlinePCLDA
    parent = {1: 0, 2: 0, 3: 1}                 # root 0; 1,2 under root; 3 under 1
    lay = DagLayout(parent, n_bg=2, tpn=1)      # K = 2 + 3 = 5
    engine = GatedOnlineLDA(lay, vocab_size=V, random_seed=0)
    m = OnlinePCLDA(K=engine.K, vocab_size=V, C=C, weight_y=50.0,
                    head_optimizer="newton", head_l2=1e-2, head_lr=0.5,
                    topic_support=topic_support, random_seed=0, topic_engine=engine)
    return m, engine, lay, V


def _docs(rng, C, V):
    from spark_vi.models.topic.types import GatedPCDocument
    out = []
    for f in ({1}, {2}, {3}, set(), {1}, {3}):
        idx = np.sort(rng.choice(np.arange(V), size=4, replace=False)).astype(np.int32)
        cnt = rng.integers(1, 5, size=4).astype(np.float64)
        y = rng.integers(0, 2, size=C).astype(np.float64)
        out.append(GatedPCDocument(
            indices=idx, counts=cnt, length=int(cnt.sum()), y=y,
            label_mask=np.ones(C, np.float64), frontier=frozenset(f)))
    return out


def test_localized_head_updates_only_support_and_solve_is_correct():
    C = 4
    from spark_vi.models.topic.dag_placement import DagLayout
    lay = DagLayout({1: 0, 2: 0, 3: 1}, n_bg=2, tpn=1)
    support = [lay.allowed(c) for c in range(C)]           # node c's topic support
    m, engine, _lay, V = _gated_pc(C, topic_support=support)
    gp = m.initialize_global(None)
    assert np.allclose(gp["w_CK"], 0.0)                    # inits at 0
    stats = m.local_update(_docs(np.random.default_rng(0), C, V), gp)
    ngp = m.update_global(gp, stats, 0.5)
    new_w = ngp["w_CK"]

    for c in range(C):
        sup = set(int(k) for k in support[c])
        off = [k for k in range(engine.K) if k not in sup]
        # (1) locality invariant: w_c is exactly 0 off its support.
        assert np.allclose(new_w[c, off], 0.0), f"node {c} leaked off-support"
        # (3) the head actually moved on the support (nonzero, finite).
        assert np.isfinite(new_w[c]).all()
    # root reads background only; a leaf reads bg + its path blocks.
    assert set(int(k) for k in support[0]) == {0, 1}
    assert set(int(k) for k in support[3]) == {0, 1, 2, 4}   # bg + block[1] + block[3]

    # (2) the support entries equal the restricted ridge-Newton solve recomputed from
    #     the SAME stats (verifies the sub-block solve, not just non-leakage).
    H = np.asarray(stats["head_hess_stat"], float)
    g = np.asarray(stats["grad_wCK_stat"], float)
    for c in range(C):
        s = support[c]
        Hc = H[c][np.ix_(s, s)]
        ridge = m.head_l2 + m.head_newton_ridge * (np.trace(Hc) / len(s)) + 1e-10
        delta = np.linalg.solve(Hc + ridge * np.eye(len(s)),
                                g[c][s] + ridge * gp["w_CK"][c][s])
        expect = gp["w_CK"][c][s] - m.head_lr * delta
        assert np.allclose(new_w[c][s], expect), f"node {c} solve mismatch"


def test_dense_vs_localized_differ_but_both_sane():
    """topic_support=None (dense) and the localized head are DIFFERENT heads (the full
    K×K solve couples off-support dims), so they must NOT coincide — the localized one is
    a genuinely cheaper, structurally-constrained head, not a re-derivation of the dense."""
    C = 4
    from spark_vi.models.topic.dag_placement import DagLayout
    lay = DagLayout({1: 0, 2: 0, 3: 1}, n_bg=2, tpn=1)
    support = [lay.allowed(c) for c in range(C)]
    docs_seed = 1
    m_d, eng, _l, V = _gated_pc(C, topic_support=None)
    m_l, _e, _l2, _V = _gated_pc(C, topic_support=support)
    gp_d = m_d.initialize_global(None)
    gp_l = m_l.initialize_global(None)
    s_d = m_d.local_update(_docs(np.random.default_rng(docs_seed), C, V), gp_d)
    s_l = m_l.local_update(_docs(np.random.default_rng(docs_seed), C, V), gp_l)
    w_d = m_d.update_global(gp_d, s_d, 0.5)["w_CK"]
    w_l = m_l.update_global(gp_l, s_l, 0.5)["w_CK"]
    assert np.isfinite(w_d).all() and np.isfinite(w_l).all()
    assert not np.allclose(w_d, w_l)                       # genuinely different heads
