import numpy as np
from spark_vi.models.topic.types import GatedBOWDocument


def test_gated_bow_document_fields():
    d = GatedBOWDocument(
        indices=np.array([1, 3], dtype=np.int32),
        counts=np.array([2.0, 1.0]),
        length=3,
        frontier=frozenset({4, 5}),
    )
    assert d.indices.tolist() == [1, 3]
    assert d.counts.tolist() == [2.0, 1.0]
    assert d.length == 3
    assert d.frontier == frozenset({4, 5})


def test_gated_bow_document_frontier_defaults_empty():
    d = GatedBOWDocument(indices=np.array([0]), counts=np.array([1.0]), length=1)
    assert d.frontier == frozenset()


from spark_vi.models.topic.dag_placement import DagLayout
from spark_vi.models.topic.gated_lda import GatedOnlineLDA, node_affinity

PARENT = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}


def _lay():
    return DagLayout(PARENT, n_bg=2, tpn=1)


def test_local_update_gates_sstats_to_allowed_rows_only():
    lay = _lay()
    V = 30
    m = GatedOnlineLDA(lay, V, alpha=0.1, eta=0.02, random_seed=0)
    gp = m.initialize_global(None)
    # A doc whose frontier is {3}: allowed = background (0,1) + closure blocks of 3.
    doc = GatedBOWDocument(indices=np.array([5, 6], dtype=np.int32),
                           counts=np.array([2.0, 1.0]), length=3,
                           frontier=frozenset({3}))
    allowed = set(lay.allowed_set(frozenset({3})).tolist())
    out = m.local_update([doc], gp)
    stats = out["lambda_stats"]
    disallowed = [k for k in range(lay.K) if k not in allowed]
    assert np.allclose(stats[disallowed], 0.0)          # gate: zero outside allowed
    assert stats[sorted(allowed)].sum() > 0.0           # allowed rows got mass


def test_local_update_empty_frontier_is_background_only():
    # A labeled background doc (empty frontier) is a known negative and must be gated
    # to the BACKGROUND block only — NOT full-K. Otherwise the large background
    # population trains the node topics and collapses them into generic comorbidity
    # (matches the Gibbs oracle and the gated STM's TopicBlockPartition.allowed_indices).
    lay = _lay()
    V = 30
    m = GatedOnlineLDA(lay, V, alpha=0.1, eta=0.02, random_seed=0)
    gp = m.initialize_global(None)
    doc = GatedBOWDocument(indices=np.array([5, 6], dtype=np.int32),
                           counts=np.array([2.0, 1.0]), length=3,
                           frontier=frozenset())
    out = m.local_update([doc], gp)
    stats = out["lambda_stats"]
    bg = list(range(lay.n_bg))
    node_rows = [k for k in range(lay.K) if k not in bg]
    assert np.allclose(stats[node_rows], 0.0)           # node topics get NOTHING
    assert stats[bg].sum() > 0.0                         # only the background block trained
    assert out["n_docs"] == 1.0


def test_node_affinity_sums_blocks():
    lay = _lay()
    theta = np.zeros(lay.K)
    for u in lay.nodes:
        for k in lay.block[u]:
            theta[k] = 0.1
    aff = node_affinity(theta, lay)
    assert set(aff.keys()) == set(lay.nodes)
    for u in lay.nodes:
        assert np.isclose(aff[u], 0.1 * lay.tpn)


def test_local_update_gamma_init_is_content_deterministic_with_seed():
    # Regression: GatedOnlineLDA.local_update must inherit OnlineLDA's content-deterministic
    # gamma_init seeding (blake2b hash of random_seed + doc content) so a distributed fit is
    # reproducible regardless of Spark partition/executor/iteration order — not draw gamma_init
    # from the plain global RNG, which would make lambda_stats vary run-to-run for the same docs.
    lay = _lay()
    V = 30
    m = GatedOnlineLDA(lay, V, alpha=0.1, eta=0.02, random_seed=0)
    gp = m.initialize_global(None)
    docs = [
        GatedBOWDocument(indices=np.array([5, 6], dtype=np.int32),
                         counts=np.array([2.0, 1.0]), length=3,
                         frontier=frozenset({3})),
        GatedBOWDocument(indices=np.array([1, 2, 7], dtype=np.int32),
                         counts=np.array([1.0, 3.0, 2.0]), length=6,
                         frontier=frozenset()),
    ]
    out1 = m.local_update(docs, gp)
    out2 = m.local_update(docs, gp)
    np.testing.assert_array_equal(out1["lambda_stats"], out2["lambda_stats"])


def test_unknown_init_strategy_raises():
    lay = _lay()
    import pytest
    with pytest.raises(ValueError, match="init"):
        GatedOnlineLDA(lay, 30, init="banana").initialize_global(None)


def test_compute_elbo_is_finite_and_gated_doc_kl_is_positive():
    # FIX 2: the gated local_update must accumulate a real per-doc Dirichlet KL (over the
    # doc's allowed sub-simplex), not the hardcoded 0.0 surrogate, so compute_elbo (and
    # VIRunner's convergence check on the production shim path) is a real variational bound.
    lay = _lay()
    V = 30
    m = GatedOnlineLDA(lay, V, alpha=0.1, eta=0.02, random_seed=0)
    gp = m.initialize_global(None)
    doc = GatedBOWDocument(indices=np.array([5, 6], dtype=np.int32),
                           counts=np.array([2.0, 1.0]), length=3,
                           frontier=frozenset({3}))
    stats = m.local_update([doc], gp)
    assert float(stats["doc_theta_kl_sum"]) > 0.0
    elbo = m.compute_elbo(gp, stats)
    assert np.isfinite(elbo)


from spark_vi.models.topic.dag_placement import fit_gated, profile, evaluate
from _stm_synth import (
    dag_placement_corpus, fit_gated_svi_local, svi_node_profiles,
)


def test_svi_matches_gibbs_placement_single_parent():
    """Placement-based, depth-weighted equivalence gate (NOT beta cosine). Gated SVI must
    match the collapsed-Gibbs oracle on node-AUC-by-depth (deep depth held to the oracle),
    MRR, and top2 within Monte-Carlo tolerance. Prototype: aucD1=aucD2=1.0, mrr 0.914 vs
    Gibbs 0.905 at 200-250 iters."""
    parent = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
    V, doc_len, n_train = 120, 60, 1600
    docs, labels, _ = dag_placement_corpus(
        parent=parent, node_prev={u: 1 for u in range(1, 7)},
        V=V, doc_len=doc_len, seed=1)
    lay = DagLayout(parent, n_bg=2, tpn=1)
    tr_d, tr_l = docs[:n_train], labels[:n_train]
    te_d, te_l = docs[n_train:], labels[n_train:]

    # Gibbs oracle
    beta_g = fit_gated(tr_d, tr_l, lay, V, rng=np.random.default_rng(0))
    ev_g = evaluate([profile(d, beta_g, lay, rng=np.random.default_rng(i))
                     for i, d in enumerate(te_d)], te_l, lay)

    # Gated SVI (random init default)
    bow = [GatedBOWDocument(*_bow(d), frontier=frozenset({int(y)}))
           for d, y in zip(tr_d, tr_l)]
    m = GatedOnlineLDA(lay, V, alpha=0.1, eta=0.02, random_seed=0)
    gp = fit_gated_svi_local(m, bow, n_iter=250)
    ev_s = evaluate(svi_node_profiles(m, gp, te_d, lay), te_l, lay)

    # Depth-weighted: DEEP-node AUC must meet the oracle within tolerance.
    max_depth = max(lay.depth(u) for u in lay.nodes)
    assert ev_s["auc_by_depth"][max_depth] >= ev_g["auc_by_depth"][max_depth] - 0.05
    # Shallow AUC and ranking within Monte-Carlo tolerance of the oracle.
    for dep in ev_g["auc_by_depth"]:
        assert ev_s["auc_by_depth"][dep] >= ev_g["auc_by_depth"][dep] - 0.08
    assert ev_s["mrr"] >= ev_g["mrr"] - 0.06
    assert ev_s["top2"] >= ev_g["top2"] - 0.08


def _bow(tokens):
    idx, cnt = np.unique(np.asarray(tokens), return_counts=True)
    return idx.astype(np.int32), cnt.astype(np.float64), int(cnt.sum())


from _stm_synth import dag_placement_corpus_multi


def test_svi_matches_gibbs_placement_multi_parent():
    """Same placement-based gate on a multi-parent diamond with comorbid (set-valued)
    frontiers. Prototype: aucD1~0.99/aucD2~1.0 both engines, mrr within ~0.01."""
    parent = {1: 0, 2: 0, 3: 0, 4: [1, 2], 5: [1, 3]}
    V, doc_len, n_train = 120, 60, 1900
    docs, labels, _ = dag_placement_corpus_multi(
        parent=parent, leaf_prev={4: 1.0, 5: 1.0}, comorbid_rate=0.25,
        V=V, doc_len=doc_len, seed=1)
    lay = DagLayout(parent, n_bg=2, tpn=1)
    tr_d, tr_l = docs[:n_train], labels[:n_train]
    te_d, te_l = docs[n_train:], labels[n_train:]

    beta_g = fit_gated(tr_d, tr_l, lay, V, rng=np.random.default_rng(0))
    ev_g = evaluate([profile(d, beta_g, lay, rng=np.random.default_rng(i))
                     for i, d in enumerate(te_d)], te_l, lay)

    bow = [GatedBOWDocument(*_bow(d), frontier=f) for d, f in zip(tr_d, tr_l)]
    m = GatedOnlineLDA(lay, V, alpha=0.1, eta=0.02, random_seed=0)
    gp = fit_gated_svi_local(m, bow, n_iter=250)
    ev_s = evaluate(svi_node_profiles(m, gp, te_d, lay), te_l, lay)

    max_depth = max(lay.depth(u) for u in lay.nodes)
    assert ev_s["auc_by_depth"][max_depth] >= ev_g["auc_by_depth"][max_depth] - 0.08
    assert ev_s["mrr"] >= ev_g["mrr"] - 0.08


def test_gated_optimize_alpha_requires_frontier_histogram():
    import pytest
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    lay = DagLayout({1: 0, 2: 0}, n_bg=2, tpn=1)
    with pytest.raises(ValueError, match="frontier_histogram"):
        GatedOnlineLDA(lay, vocab_size=5, optimize_alpha=True)   # no histogram

def test_gated_tied_layout_shapes():
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    lay = DagLayout({1: 0, 2: 0, 3: 1}, n_bg=2, tpn=1)          # 3 nodes
    m = GatedOnlineLDA(lay, vocab_size=5, optimize_alpha=True,
                       frontier_histogram={frozenset({3}): 4, frozenset(): 6})
    # tied layout: [bg, node1, node2, node3]
    assert list(m._block_sizes) == [2, 1, 1, 1]
    # topic_to_tied: topics 0,1 -> 0 (bg); block(1)->1, block(2)->2, block(3)->3
    assert m._topic_to_tied[0] == 0 and m._topic_to_tied[1] == 0
    assert m._topic_to_tied[lay.block[1][0]] == 1
    assert m._topic_to_tied[lay.block[3][0]] == 3

def test_gated_local_update_emits_node_theta_stat():
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    from spark_vi.models.topic.types import GatedBOWDocument
    lay = DagLayout({1: 0, 2: 0}, n_bg=2, tpn=1)                # B = 1 + 2 = 3
    m = GatedOnlineLDA(lay, vocab_size=6, optimize_alpha=True, random_seed=0,
                       frontier_histogram={frozenset({1}): 1, frozenset(): 1})
    gp = m.initialize_global(None)
    docs = [GatedBOWDocument(indices=np.array([0, 3], np.int32),
                             counts=np.array([2.0, 1.0]), length=3,
                             frontier=frozenset({1})),
            GatedBOWDocument(indices=np.array([1, 4], np.int32),
                             counts=np.array([1.0, 1.0]), length=2,
                             frontier=frozenset())]           # background doc
    out = m.local_update(docs, gp)
    stat = out["e_log_theta_node_sum"]
    assert stat.shape == (3,)                                  # [bg, node1, node2]
    # node2 is in neither doc's allowed set (doc1 -> node1 only; doc2 -> bg only)
    assert stat[2] == 0.0
    # background block is in both docs' allowed sets -> nonzero
    assert stat[0] != 0.0

def test_gated_local_update_no_stat_when_alpha_fixed():
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    from spark_vi.models.topic.types import GatedBOWDocument
    lay = DagLayout({1: 0, 2: 0}, n_bg=2, tpn=1)
    m = GatedOnlineLDA(lay, vocab_size=6, random_seed=0)        # optimize_alpha False
    gp = m.initialize_global(None)
    out = m.local_update([GatedBOWDocument(indices=np.array([0], np.int32),
                                           counts=np.array([1.0]), length=1,
                                           frontier=frozenset({1}))], gp)
    assert "e_log_theta_node_sum" not in out


def test_gated_update_global_leaves_alpha_fixed_when_disabled():
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    lay = DagLayout({1: 0, 2: 0}, n_bg=2, tpn=1)
    m = GatedOnlineLDA(lay, vocab_size=6, random_seed=0)          # alpha fixed
    gp = m.initialize_global(None)
    alpha0 = gp["alpha"].copy()
    stats = {"lambda_stats": np.ones((lay.K, 6)), "n_docs": np.array(4.0)}
    gp2 = m.update_global(gp, stats, learning_rate=0.5)
    assert np.array_equal(gp2["alpha"], alpha0)                   # unchanged
    assert not np.array_equal(gp2["lambda"], gp["lambda"])        # lambda still moves

def test_gated_update_global_learns_asymmetric_alpha():
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    lay = DagLayout({1: 0, 2: 0}, n_bg=2, tpn=1)                  # B=3: bg,node1,node2
    hist = {frozenset({1}): 30, frozenset({2}): 30, frozenset(): 40}
    m = GatedOnlineLDA(lay, vocab_size=6, optimize_alpha=True, random_seed=0,
                       frontier_histogram=hist)
    gp = m.initialize_global(None)
    # Make node1 look 'common' (E[log θ] near 0) and node2 'rare' (very negative):
    # tied order [bg, node1, node2].
    stats = {
        "lambda_stats": np.ones((lay.K, 6)),
        "n_docs": np.array(100.0),
        "e_log_theta_node_sum": np.array([-20.0, -2.0, -40.0]),
    }
    gp2 = m.update_global(gp, stats, learning_rate=1.0)
    a_full = gp2["alpha"]
    a_node1 = a_full[lay.block[1][0]]
    a_node2 = a_full[lay.block[2][0]]
    assert a_node1 > a_node2                                      # common node gets larger alpha
    assert a_full.min() >= 1e-3                                   # floor respected
    # tying preserved: all topics in a block share one value
    assert np.allclose(a_full[lay.block[1]], a_node1)


def test_initialize_global_uses_precomputed_spectral_lambda():
    # When data_summary carries a precomputed (K,V) 'spectral_lambda', the model
    # seeds lambda from it directly (the scalable path) instead of running a
    # dense INIT_STRATEGIES strategy.
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA

    lay = DagLayout({1: 0, 2: 0}, n_bg=2, tpn=1)      # K = 4
    V = 6
    m = GatedOnlineLDA(lay, V, init="spectral")
    planted = np.arange(lay.K * V, dtype=np.float64).reshape(lay.K, V) + 1.0
    gp = m.initialize_global({"spectral_lambda": planted})
    assert np.allclose(gp["lambda"], planted)


# --- SP2 Task 1: per-domain dict-lambda storage + assemble/split + init ---

def test_domains_none_is_single_array_unchanged():
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    lay = DagLayout({1: 0, 2: 1, 3: 1}, n_bg=2, tpn=1)
    m = GatedOnlineLDA(lay, vocab_size=12, random_seed=7)          # domains=None
    gp = m.initialize_global(None)
    assert isinstance(gp["lambda"], np.ndarray) and gp["lambda"].shape == (lay.K, 12)


def test_multidomain_init_is_per_domain_dict():
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    lay = DagLayout({1: 0, 2: 1, 3: 1}, n_bg=2, tpn=1)
    m = GatedOnlineLDA(lay, vocab_size=56, domains=[40, 16], random_seed=7)
    gp = m.initialize_global(None)
    lam = gp["lambda"]
    assert set(lam) == {0, 1}
    assert lam[0].shape == (lay.K, 40) and lam[1].shape == (lay.K, 16)


def test_assemble_split_round_trip():
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    lay = DagLayout({1: 0, 2: 1, 3: 1}, n_bg=2, tpn=1)
    m = GatedOnlineLDA(lay, vocab_size=7, domains=[4, 3], random_seed=1)
    lam = {0: np.abs(np.random.default_rng(0).normal(size=(lay.K, 4))) + .1,
           1: np.abs(np.random.default_rng(1).normal(size=(lay.K, 3))) + .1}
    eb = m._assemble_expElogbeta(lam)
    assert eb.shape == (lay.K, 7)
    # each block equals its own full-row normalization
    from scipy.special import digamma
    np.testing.assert_allclose(eb[:, :4], np.exp(digamma(lam[0]) - digamma(lam[0].sum(1, keepdims=True))))
    # split of a concatenated array round-trips the block shapes
    back = m._split_to_domains(eb)
    assert back[0].shape == (lay.K, 4) and back[1].shape == (lay.K, 3)


def test_domains_must_sum_to_vocab():
    import pytest
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    lay = DagLayout({1: 0, 2: 1, 3: 1}, n_bg=2, tpn=1)
    with pytest.raises(ValueError):
        GatedOnlineLDA(lay, vocab_size=56, domains=[40, 10])   # 50 != 56


# --- SP2 Task 2: dict-aware E/M-step + per-domain eta + multi-domain ELBO ---

def test_multidomain_em_refines_seeded_per_domain_betas():
    """The dict-aware multi-domain E/M-step recovers each node's planted per-domain
    signature (>0.4 mass on the planted support in BOTH domains) when its block
    topic is given initial traction — a direct validation of the multi-domain E/M
    math (a broken assemble / split-target / per-domain normalization could not
    reach this).

    Each block topic is seeded with a gentle bump (0.3) on its OWN planted
    per-domain support — a symmetry-break, NOT the answer: the bump is tiny next
    to the ~1.0-per-column Gamma random lambda, so the E/M must still concentrate
    each block topic from ~0.03 raw column mass to the ~0.5 recovered support mass
    (half of each domain's tokens are the shared common pool, absorbed by the
    background block; the other half is the node signature). The traction bump is
    REQUIRED because random init alone is seed-fragile here: a node's block topic
    can suffer topic-death (its signal absorbed by the background block) on some
    seeds (worst-node recovery across seeds 0-4: 0.13/0.51/0.13/0.50/0.50) — a
    known LDA local optimum that the SP1 spectral seed addresses in production
    (see insight 0066). This test isolates E/M correctness from that init
    fragility; the random-init/spectral-seed path is validated separately.
    """
    import numpy as np
    from tests._stm_synth import two_domain_dag_corpus
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    from spark_vi.models.topic.types import GatedBOWDocument
    # Flat DAG: every node is a direct child of the root, so each node's signature
    # is node-specific and undiluted by a shared-parent block. b_only_node=None ->
    # every node has a clean unique signature in BOTH domains.
    parent = {1: 0, 2: 0, 3: 0}
    docs, labels, domain_bounds, pa, pb, slot_of_node, codes = two_domain_dag_corpus(
        parent=parent, node_prev={1: 1., 2: 1., 3: 1.}, V_a=40, V_b=16,
        doc_len=40, seed=5, b_only_node=None)
    lay = DagLayout(parent, n_bg=2, tpn=1); V = domain_bounds[-1]
    gdocs = [GatedBOWDocument(indices=np.unique(d).astype(np.int32),
             counts=np.unique(d, return_counts=True)[1].astype(float), length=len(d),
             frontier=frozenset({int(f)}))
             for d, f in zip(docs[:800], labels[:800])]
    m = GatedOnlineLDA(lay, vocab_size=V, domains=[40, 16], random_seed=0)
    gp = m.initialize_global(None)
    planted = {0: pa, 1: pb}
    # Gentle traction bump on each block topic's OWN planted support (symmetry-break).
    for u in lay.nodes:
        for md in (0, 1):
            support = np.where(planted[md][slot_of_node[u]] > 1e-3)[0]
            for k in lay.block[u]:
                gp["lambda"][md][k, support] += 0.3
    for _ in range(50):
        gp = m.update_global(gp, m.local_update(gdocs, gp), learning_rate=1.0)
    for md in (0, 1):
        lam_m = gp["lambda"][md]
        beta_m = lam_m / lam_m.sum(1, keepdims=True)
        np.testing.assert_allclose(beta_m.sum(1), 1.0)          # valid per-domain distribution
        assert np.isfinite(lam_m).all()
        for u in lay.nodes:
            support = np.where(planted[md][slot_of_node[u]] > 1e-3)[0]
            recovery = float(beta_m[lay.block[u]][:, support].sum(axis=1).max())
            uniform = len(support) / lam_m.shape[1]             # dead-topic baseline (~0.15/0.125)
            assert recovery > 0.4, (u, md, recovery, uniform)   # recovered ~0.5 >> uniform


def test_multidomain_spectral_seed_fixes_topic_death():
    """The multi-domain spectral seed (block-aligned anchors WITH the per-domain
    candidate floor, split per-domain) gives every node's block topic traction in
    BOTH domains, so a fit at random_seed=0 -- the seed that suffers topic-death
    under RANDOM init (insight 0066) -- recovers EVERY node/domain (>0.5 mass on the
    planted support), not just the lucky ones. Without the per-domain floor the
    denser domain dominates anchor selection and a node's sparse-domain slice dies
    (recovery ~0.005); with it, a node anchors on its sparse-domain word which then
    defines the topic across both domains via the Q_01 within-doc tie."""
    import numpy as np
    from tests._stm_synth import two_domain_dag_corpus
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    from spark_vi.models.topic.types import GatedBOWDocument
    parent = {1: 0, 2: 0, 3: 0}
    docs, labels, domain_bounds, pa, pb, slot_of_node, codes = two_domain_dag_corpus(
        parent=parent, node_prev={1: 1., 2: 1., 3: 1.}, V_a=40, V_b=16,
        doc_len=40, seed=5, b_only_node=None)
    lay = DagLayout(parent, n_bg=2, tpn=1); V = domain_bounds[-1]
    ds = {"train_docs": [np.asarray(d) for d in docs[:800]],
          "train_labels": [int(f) for f in labels[:800]]}
    m = GatedOnlineLDA(lay, vocab_size=V, domains=[40, 16], init="spectral", random_seed=0)
    gp = m.initialize_global(ds)
    assert set(gp["lambda"]) == {0, 1}                       # per-domain dict seed
    assert gp["lambda"][0].shape == (lay.K, 40) and gp["lambda"][1].shape == (lay.K, 16)
    gdocs = [GatedBOWDocument(indices=np.unique(d).astype(np.int32),
             counts=np.unique(d, return_counts=True)[1].astype(float), length=len(d),
             frontier=frozenset({int(f)}))
             for d, f in zip(docs[:800], labels[:800])]
    for _ in range(50):
        gp = m.update_global(gp, m.local_update(gdocs, gp), learning_rate=1.0)
    planted = {0: pa, 1: pb}
    for md in (0, 1):
        beta_m = gp["lambda"][md] / gp["lambda"][md].sum(1, keepdims=True)
        for u in lay.nodes:
            support = np.where(planted[md][slot_of_node[u]] > 1e-3)[0]
            recovery = float(beta_m[lay.block[u]][:, support].sum(axis=1).max())
            assert recovery > 0.5, (u, md, recovery)         # spectral: ~0.67-0.72, no topic death


def test_multidomain_svi_planted_bonly_node_recovery():
    """SVI acceptance analogue of SP1 Task 4: a spectral-seeded multi-domain fit
    recovers each node's planted per-domain signature on a NON-FLAT DAG containing
    a `b_only_node` -- a node recoverable from domain 1 ALONE. This is the
    genuinely new case here: both other multi-domain tests in this file use a flat
    DAG (`parent={1:0,2:0,3:0}`) with `b_only_node=None`, where every node's
    signature is undiluted in both domains and no shared-parent ambiguity exists.

    DAG: parent={1: 0, 2: 1} -- node 1 is node 2's non-root parent (required by
    `b_only_node`; the corpus generator raises if the target's only parent is the
    root, which carries no signature block to share). b_only_node=2 makes node
    2's DOMAIN-0 signature block IDENTICAL to node 1's, at the generator's own
    documented default boost (b_only_signal_boost=4, not hand-tuned here).

    HONESTY ABOUT WHAT'S IDENTIFIED (do not over-claim): in DOMAIN 0, node 2's
    planted support IS node 1's planted support by construction, so mass there
    does NOT uniquely identify node 2 -- it is equally consistent with node 1.
    The domain-0 assertions below check for MASS on that (shared) support, not
    for discriminating the two nodes. In DOMAIN 1, node 2's support is exclusive
    to it, so recovery there IS unique identification -- the actual point of the
    "recoverable from domain 1 alone" plant, and it gets the strongest gate below.

    anchor_scope="frontier": this DAG is genuinely hierarchical (node 1 is an
    ancestor of node 2, so EVERY document's frontier closure includes node 1),
    unlike the flat DAGs above where closure(v) == {v} and anchor_scope is moot.
    Verified: under the default "closure" scope this exact corpus/fit produces
    near-uniform-or-worse recovery everywhere (domain 0: ~0.23-0.25 vs uniform
    0.225; domain 1 node 1: ~0.20 vs uniform 0.1875; domain 1 node 2: ~0.006 --
    total topic death) because node 1's anchor sketch pools over every document
    (it is common to all of them), which -- as gated_init.py's own docstring
    documents -- lets a parent's ubiquity get mistaken for background at anchor
    selection. "frontier" (trains each node's sketch only from docs where it is
    the deepest attested node) is the documented fix for exactly this DAG shape,
    applied here via data_summary's existing anchor_scope knob -- a configuration
    choice for a non-flat DAG, not a plant or assertion hack.

    Batch-VB (Hoffman, Blei & Bach 2010) full-batch lr=1.0 for 50 iterations,
    mirroring the other two multi-domain tests. Dead-topic baselines (uniform
    mass on the same support): ~0.225 domain 0, ~0.1875 domain 1. Observed
    recovery, stable across corpus seeds 1-8 (this test pins seed=5): domain 0
    node 1 ~0.50-0.53, node 2 (shared/ambiguous) ~0.47-0.50; domain 1 node 1
    ~0.49-0.50, node 2 (unique) ~0.76-0.82."""
    import numpy as np
    from tests._stm_synth import two_domain_dag_corpus
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    from spark_vi.models.topic.types import GatedBOWDocument
    parent = {1: 0, 2: 1}
    docs, labels, domain_bounds, pa, pb, slot_of_node, codes = two_domain_dag_corpus(
        parent=parent, node_prev={1: 1., 2: 1.}, V_a=40, V_b=16,
        doc_len=40, seed=5, b_only_node=2)
    lay = DagLayout(parent, n_bg=2, tpn=1); V = domain_bounds[-1]
    ds = {"train_docs": [np.asarray(d) for d in docs[:800]],
          "train_labels": [int(f) for f in labels[:800]],
          "anchor_scope": "frontier"}   # non-flat DAG: see docstring for why
    m = GatedOnlineLDA(lay, vocab_size=V, domains=[40, 16], init="spectral", random_seed=0)
    gp = m.initialize_global(ds)
    assert set(gp["lambda"]) == {0, 1}                       # per-domain dict seed
    gdocs = [GatedBOWDocument(indices=np.unique(d).astype(np.int32),
             counts=np.unique(d, return_counts=True)[1].astype(float), length=len(d),
             frontier=frozenset({int(f)}))
             for d, f in zip(docs[:800], labels[:800])]
    for _ in range(50):
        gp = m.update_global(gp, m.local_update(gdocs, gp), learning_rate=1.0)
    planted = {0: pa, 1: pb}
    for md in (0, 1):
        lam_m = gp["lambda"][md]
        beta_m = lam_m / lam_m.sum(1, keepdims=True)
        np.testing.assert_allclose(beta_m.sum(1), 1.0)       # valid per-domain distribution
        assert np.isfinite(lam_m).all()
        for u in lay.nodes:
            support = np.where(planted[md][slot_of_node[u]] > 1e-3)[0]
            recovery = float(beta_m[lay.block[u]][:, support].sum(axis=1).max())
            uniform = len(support) / lam_m.shape[1]          # dead-topic baseline
            assert recovery > 1.5 * uniform, (u, md, recovery, uniform)   # clear of dead-topic
            if u == 2 and md == 1:
                # domain 1 is where b_only_node=2's EXCLUSIVE signature lives --
                # unique identification, the strongest claim this plant supports.
                assert recovery > 0.6, (u, md, recovery, uniform)         # observed ~0.76-0.82
            else:
                assert recovery > 0.4, (u, md, recovery, uniform)         # observed ~0.47-0.53


def test_single_domain_fit_byte_identical():
    """domains=None reproduces the current gated fit exactly (fixed seed): two
    independently-run fits over the same docs/seed must be bit-identical, and the
    lambda representation stays a plain (K, V) ndarray (not a dict)."""
    import numpy as np
    lay = _lay()
    V = 30
    docs = [
        GatedBOWDocument(indices=np.array([5, 6], dtype=np.int32),
                         counts=np.array([2.0, 1.0]), length=3,
                         frontier=frozenset({3})),
        GatedBOWDocument(indices=np.array([1, 2, 7], dtype=np.int32),
                         counts=np.array([1.0, 3.0, 2.0]), length=6,
                         frontier=frozenset()),
    ]

    def run():
        m = GatedOnlineLDA(lay, V, alpha=0.1, eta=0.02, random_seed=0)
        gp = m.initialize_global(None)
        for _ in range(5):
            gp = m.update_global(gp, m.local_update(docs, gp), learning_rate=0.5)
        return gp

    gp1, gp2 = run(), run()
    assert isinstance(gp1["lambda"], np.ndarray)
    np.testing.assert_array_equal(gp1["lambda"], gp2["lambda"])


def test_multidomain_compute_elbo_matches_manual_per_domain_kl():
    """ELBO's global KL term for a per-domain eta must equal the hand-computed
    Sigma_k Sigma_m KL(Dirichlet(lam_m[k]) || Dirichlet(eta_m . 1_{V_m})) — NOT a
    single KL over the naively concatenated vector (Dirichlet KL does not
    decompose that way across a concatenation)."""
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    from spark_vi.models.topic.lda import _dirichlet_kl
    lay = DagLayout({1: 0, 2: 1, 3: 1}, n_bg=2, tpn=1)
    m = GatedOnlineLDA(lay, vocab_size=7, domains=[4, 3], eta=[0.5, 0.2], random_seed=3)
    gp = m.initialize_global(None)
    aggregated_stats = {"doc_loglik_sum": np.array(-12.3), "doc_theta_kl_sum": np.array(1.7)}
    elbo = m.compute_elbo(gp, aggregated_stats)

    manual_kl = 0.0
    eta_by_domain = {0: 0.5, 1: 0.2}
    for md, V_m in enumerate([4, 3]):
        eta_vec = np.full(V_m, eta_by_domain[md])
        for k in range(lay.K):
            manual_kl += _dirichlet_kl(gp["lambda"][md][k], eta_vec)
    expected = (float(aggregated_stats["doc_loglik_sum"])
                - float(aggregated_stats["doc_theta_kl_sum"]) - manual_kl)
    np.testing.assert_allclose(elbo, expected)


def test_multidomain_infer_local_assembles_dict_lambda():
    """infer_local (deployment fold-in) must work off the dict-lambda in multi-domain
    mode: assemble the concatenated expElogbeta and run the same full-K CAVI."""
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    lay = DagLayout({1: 0, 2: 1, 3: 1}, n_bg=2, tpn=1)
    m = GatedOnlineLDA(lay, vocab_size=7, domains=[4, 3], random_seed=2)
    gp = m.initialize_global(None)
    row = GatedBOWDocument(indices=np.array([0, 4], dtype=np.int32),
                           counts=np.array([2.0, 1.0]), length=3)
    out = m.infer_local(row, gp)
    assert out["theta"].shape == (lay.K,)
    np.testing.assert_allclose(out["theta"].sum(), 1.0)
    assert np.isfinite(out["gamma"]).all()


def test_eta_scalar_broadcasts_to_all_domains():
    """A scalar eta under multi-domain applies the SAME concentration to every
    domain block (matches the domains=None default-broadcast behavior)."""
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    lay = DagLayout({1: 0, 2: 1, 3: 1}, n_bg=2, tpn=1)
    m = GatedOnlineLDA(lay, vocab_size=7, domains=[4, 3], eta=0.3, random_seed=0)
    vec = m._eta_vocab_vector()
    assert vec.shape == (7,)
    np.testing.assert_allclose(vec, 0.3)


def test_eta_per_domain_sequence_applies_per_block():
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    lay = DagLayout({1: 0, 2: 1, 3: 1}, n_bg=2, tpn=1)
    m = GatedOnlineLDA(lay, vocab_size=7, domains=[4, 3], eta=[0.5, 0.2], random_seed=0)
    vec = m._eta_vocab_vector()
    np.testing.assert_allclose(vec[:4], 0.5)
    np.testing.assert_allclose(vec[4:], 0.2)


def test_eta_per_domain_sequence_length_mismatch_raises():
    import pytest
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    lay = DagLayout({1: 0, 2: 1, 3: 1}, n_bg=2, tpn=1)
    with pytest.raises(ValueError):
        GatedOnlineLDA(lay, vocab_size=7, domains=[4, 3], eta=[0.5, 0.2, 0.1])


def test_eta_per_domain_sequence_rejects_nonpositive():
    import pytest
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    lay = DagLayout({1: 0, 2: 1, 3: 1}, n_bg=2, tpn=1)
    with pytest.raises(ValueError):
        GatedOnlineLDA(lay, vocab_size=7, domains=[4, 3], eta=[0.5, 0.0])
