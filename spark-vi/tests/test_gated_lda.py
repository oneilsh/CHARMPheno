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

    DAG: parent={1: 0, 2: 1, 3: 1} -- node 1 (family) has two children, subtypes
    2 and 3; b_only_node=3 makes node 3's DOMAIN-0 signature block IDENTICAL to
    node 1's (required: `b_only_node` needs a non-root parent with its own
    signature block to share), while node 3's domain-1 block stays exclusive.
    This 3-node shape -- not a 2-node chain -- is REQUIRED, not merely
    illustrative: it is the only shape with a node (node 2) whose domain-0
    signature is genuinely exclusive to it (a 2-node chain's b_only_node has NO
    domain-0-exclusive row anywhere in the plant, which silently drops the
    domain-0 discrimination check below for every node).

    HONESTY ABOUT WHAT'S IDENTIFIED (do not over-claim): in DOMAIN 0, node 3's
    planted support IS node 1's planted support by construction -- ground truth
    TIES there (both rows put mass 1.0 on the identical 6-column block) -- so
    the ambiguity is SYMMETRIC: node 1's domain-0 topic cannot legitimately be
    required to "beat" node 3's on that support, or vice versa. The
    discrimination check (below) is deliberately SKIPPED for BOTH sides of the
    shared block (domain 0, node 3 AND domain 0, node 1 -- derived from the DAG
    via `lay.parents[B_ONLY]`, not hardcoded); only a mass/magnitude gate
    applies to either. In DOMAIN 1, node 3's support is exclusive to it, so
    recovery there IS unique identification -- the actual point of the
    "recoverable from domain 1 alone" plant, and it gets the strongest gate
    below.

    ROOT CAUSE this test's first version missed, and the reason for `bg_frac`
    (see `two_domain_dag_corpus`'s docstring): the corpus generator previously
    labeled EVERY document with a node -- zero true background documents -- so
    the gated spectral seed's background block (`lay.n_bg`) had no real
    background pool to anchor on. That is NOT "a parent stealing a child's
    word" (no such mechanism exists here): under `anchor_scope="closure"` the
    background pool is drawn from ALL docs, so with none of them truly
    backgroundless the background anchors are free to land on and absorb a
    node's own signature (background-anchor theft); under `"frontier"` the
    background pool is docs with an empty frontier, so with none the
    background block never gets seeded at all (logs a warning, stays at the
    1e-9 floor). `bg_frac=0.2` gives the corpus a genuine background pool.

    Even with real background docs, `anchor_scope` still matters, and it is
    NOT "moot" for a flat DAG either (a flat-DAG corpus with `bg_frac=0.0`
    would ALSO break under "frontier", for the identical reason: zero
    background docs) -- `anchor_scope` always selects both what counts as a
    node's own training docs AND what counts as the background pool
    (`gated_init.py:_anchor_node_set`/`spectral_block_aligned_lambda`).
    "frontier" is used here because it is what is verified to work for this
    hierarchical DAG: node 1 is node 3's ancestor, so under "closure" node 1's
    sketch pools over every document, and -- verified on this exact corpus,
    with `bg_frac=0.2` docs present -- "closure" recovers nodes 1 and 2 well
    (~0.71-0.78) but leaves node 3 (the b_only node) dead in BOTH domains
    (~0.150/~0.125, at the uniform floor) for a residual (non-background)
    reason not further diagnosed here. "frontier" (train each node's sketch
    only from docs where it is the deepest attested node) recovers all three.

    The 800-doc TRAINING SAMPLE below is deliberately a fixed pseudorandom
    subset of the full (foreground-then-background-ordered) 2000-doc corpus,
    not `docs[:800]` -- the background docs are appended at the end of the
    corpus, so an unshuffled prefix slice would contain zero of them and
    silently reproduce the exact failure `bg_frac` exists to fix.

    FIT-DEPENDENCE (this is the assertion the seed alone cannot satisfy --
    so this test is not re-covering ground `test_multidomain_spectral_seed_fixes_topic_death`
    already covers, which only checks seed-independent mass gates): domain 0,
    node 2's magnitude gate (`recovery > 0.4`) is the well-posed one that
    carries this, on a domain-0 support genuinely EXCLUSIVE to node 2 (not the
    shared/tied block -- see HONESTY above). At the spectral seed, BEFORE any
    EM iteration, this gate FAILS on this exact config (recovery=0.2796 <
    0.4), and it PASSES only after the 50-iteration fit (recovery=0.8615).
    (An earlier version of this test instead rested this claim on domain 0,
    node 1's DISCRIMINATION check -- which does flip seed-False to fit-True on
    this config, but that cell sits on the shared/tied support the HONESTY
    section above excludes from discrimination for exactly this reason: a
    tie is not a well-posed thing to require one side to "beat".)

    Batch-VB (Hoffman, Blei & Bach 2010) full-batch lr=1.0 for 50 iterations,
    mirroring the other two multi-domain tests. Dead-topic baselines (uniform
    mass on the same support): 0.15 domain 0, 0.125 domain 1. Observed
    recovery on this exact config (corpus seed=5, sample-permutation seed=0):
    domain 0 node 1=0.938, node 2=0.862, node 3 (shared/ambiguous)=0.641;
    domain 1 node 1=0.969, node 2=0.868, node 3 (unique)=0.984. Stable in sign
    and magnitude across corpus seeds 1-8 (spot-checked, not asserted here)."""
    import numpy as np
    from tests._stm_synth import two_domain_dag_corpus
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    from spark_vi.models.topic.types import GatedBOWDocument
    B_ONLY = 3
    parent = {1: 0, 2: 1, 3: 1}
    docs, labels, domain_bounds, pa, pb, slot_of_node, codes = two_domain_dag_corpus(
        parent=parent, node_prev={1: 1., 2: 1., 3: 1.}, V_a=40, V_b=16,
        doc_len=40, seed=5, b_only_node=B_ONLY, bg_frac=0.2)
    lay = DagLayout(parent, n_bg=2, tpn=1); V = domain_bounds[-1]
    # Fixed pseudorandom sample (NOT docs[:800] -- see docstring) mixing foreground
    # and background docs; labels are ints (foreground) or frozenset() (background).
    sample = np.random.default_rng(0).permutation(len(docs))[:800]
    docs_s = [docs[i] for i in sample]
    labels_s = [labels[i] for i in sample]
    assert any(isinstance(f, frozenset) for f in labels_s)   # sample actually has bg docs
    ds = {"train_docs": [np.asarray(d) for d in docs_s],
          "train_labels": [f if isinstance(f, frozenset) else int(f) for f in labels_s],
          "anchor_scope": "frontier"}   # non-flat DAG + real bg pool: see docstring for why
    m = GatedOnlineLDA(lay, vocab_size=V, domains=[40, 16], init="spectral", random_seed=0)
    gp = m.initialize_global(ds)
    assert set(gp["lambda"]) == {0, 1}                       # per-domain dict seed
    gdocs = [GatedBOWDocument(indices=np.unique(d).astype(np.int32),
             counts=np.unique(d, return_counts=True)[1].astype(float), length=len(d),
             frontier=(f if isinstance(f, frozenset) else frozenset({int(f)})))
             for d, f in zip(docs_s, labels_s)]
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
            if u == B_ONLY and md == 1:
                # domain 1 is where b_only_node=3's EXCLUSIVE signature lives --
                # unique identification, the strongest claim this plant supports.
                assert recovery > 0.6, (u, md, recovery, uniform)         # observed ~0.98
            else:
                # (domain=0, node=2) is the FIT-DEPENDENT cell (see docstring): this
                # gate FAILS at the spectral seed (0.2796) and only PASSES after the
                # 50-iteration fit (0.8615) -- on a domain-0 support genuinely
                # exclusive to node 2, so it is well-posed (unlike a discrimination
                # check on a shared/tied support would be).
                assert recovery > 0.4, (u, md, recovery, uniform)         # observed ~0.64-0.97

            # The domain-0 ambiguity is SYMMETRIC: b_only_node's domain-0 block is
            # made identical to its shared parent's (see two_domain_dag_corpus), so
            # ground truth TIES between the two of them on that support -- neither
            # can legitimately "beat" the other there. Skip discrimination for BOTH
            # sides of the shared block (derived from the DAG, not hardcoded), not
            # just b_only_node itself; the magnitude gate above still applies to both.
            shared_parent = lay.parents[B_ONLY][0]
            shared_ambiguous = (md == 0 and u in (B_ONLY, shared_parent))
            if not shared_ambiguous:
                # Discrimination: the OWNER block must beat every OTHER topic's mass
                # on this support -- not merely accumulate mass itself (mass alone is
                # what the rejected first version of this test checked, and it let a
                # sibling's topic silently outscore the true owner).
                other_topics = [k for k in range(lay.K) if k not in lay.block[u]]
                best_other = float(beta_m[other_topics][:, support].sum(axis=1).max())
                assert recovery > best_other, (u, md, recovery, best_other)


def _per_node_mrr(profiles, labels, lay):
    """Mean reciprocal rank of the TRUE node, broken out BY true node.

    Same rank rule as `dag_placement.evaluate`'s mrr (count only nodes with strictly
    greater affinity, so ties are scored optimistically), but grouped instead of pooled.
    This is the read-out with power against a rank swap between a node and its ancestor:
    a pooled mrr dilutes one node's collapse across every other node's docs, and node-AUC
    cannot see it at all (it ranks DOCS within one node, so it is invariant to a per-node
    affinity SCALE offset -- the exact shape a node/ancestor swap takes)."""
    import numpy as np
    acc = {}
    for pr, y in zip(profiles, labels):
        P = np.array([pr[u] for u in lay.nodes])
        j = lay.nodes.index(int(y))
        acc.setdefault(int(y), []).append(1.0 / (1 + int((P > P[j]).sum())))
    return {u: float(np.mean(v)) for u, v in acc.items()}


def _profile_l1(prof_a, prof_b, lay):
    """Mean L1 distance between two engines' per-doc node-affinity profiles, each
    normalized over the DAG's nodes. Deliberately scale-SENSITIVE: rank correlation and
    node-AUC are both invariant to a per-node scale offset, so neither can see one engine
    concentrating its affinity differently from the other. Range [0, 2]."""
    import numpy as np
    out = []
    for a, b in zip(prof_a, prof_b):
        va = np.array([a[u] for u in lay.nodes]); va /= max(va.sum(), 1e-12)
        vb = np.array([b[u] for u in lay.nodes]); vb /= max(vb.sum(), 1e-12)
        out.append(float(np.abs(va - vb).sum()))
    return float(np.mean(out))


def test_multidomain_svi_matches_gibbs_placement():
    """Multi-domain placement-equivalence gate: the SVI engine (GatedOnlineLDA, mean-field
    variational EM -- Hoffman, Blei & Bach 2010) and the collapsed-Gibbs oracle (`fit_gated`
    -- Griffiths & Steyvers 2004 collapsed sampling, plus the per-domain word-topic factor,
    MixEHR-style: Li, Nair, Lu et al. 2020, Nat. Commun.) must place held-out, UNGATED
    fold-in documents onto the same DAG nodes. Gated on the SPLIT-AVERAGED read-out over 8
    independent train/eval splits, not one split.

    DAG SHAPE (`{1:0, 2:0, 3:1, 4:1, 5:2, 6:2}`) is chosen so the metrics resolve. Two
    depth-1 nodes each with two children: `subtree(1) = {1,3,4}` and `subtree(2) = {2,5,6}`,
    so neither covers the other's labels and depth-1 node-AUC is SCOREABLE (on a DAG whose
    single depth-1 node is every other node's ancestor it is structurally nan, because that
    node has no negative among labeled foreground eval docs). Six nodes give mrr real
    resolution. `b_only_node=3` puts the multi-domain case that motivates the per-domain
    factor INTO THE CORPUS -- node 3's domain-0 signature block is identical to its parent
    node 1's, so node 3 is separable from domain 1 alone -- but this READ-OUT cannot resolve
    it: with the SVI arm switched to `domains=None` (the per-domain factor gone entirely),
    node 3's per-true-node mrr is still 1.000 and every assertion below still passes. The
    test that resolves that case discriminatively is
    `test_multidomain_svi_planted_bonly_node_recovery`, which scores node 3's recovery on its
    planted per-domain signature support. Do not read this gate as evidence about node 3's
    domain asymmetry.

    IDENTIFIABILITY OF THE LABEL (`ancestor_signature_decay=0.5`) -- load-bearing, and the
    reason an earlier version of this gate could not be written honestly. Every document's
    label is its DEEPEST attested node, but the generator's default splits signature draws
    EVENLY over closure(v), so a depth-2 document emits as many of its parent's signature
    tokens as its own. The evidence is then symmetric between a node and its ancestor while
    the label is not, so "the deepest attested node ranks first" is an UNIDENTIFIED
    tie-break -- and the two engines break it in opposite directions (measured on the even
    plant: for docs at node 5, Gibbs put the child ahead 0.2578 vs 0.2450 while SVI put the
    parent ahead 0.4253 vs 0.2796, driving split-averaged mrr to 0.9971 vs 0.8756 with
    per-node mrr pinned at exactly 0.500 -- a deterministic rank-2). The knob's default is
    1.0 and reproduces every other caller's corpus byte-identically; see
    `two_domain_dag_corpus`'s docstring.

    The exact arithmetic at decay=0.5 on THIS config, because the margin is what matters and
    it is not simply "double": doc_len=40 gives n_common=20, so a depth-2 label has
    per = (40-20)//(2*2) = 5 draws of its own block per domain and its one ancestor gets
    max(1, round(5*0.5)) = 2 -- a 2.5x margin. A depth-1 label has per = (40-20)//(2*1) = 10
    and no non-root ancestor, so the knob does not touch it.

    THE BIAS IS OUT-VOTED, NOT REMOVED -- and the margin needed is measurable, so a future
    reader should not treat 0.5 as cosmetic. Sweeping the knob on this DAG (max depth 2, so
    only the k=1 ratio matters; note round() makes decay=0.9 and decay=0.75 produce the
    IDENTICAL corpus, ancestor draws = 4 in both):
        decay 1.00 -> ancestor 5 vs own 5 = 1.0x  -> unidentified; split-averaged mrr 0.9971
                      (Gibbs) vs 0.8756 (SVI), per-node mrr pinned at 0.500
        decay 0.90 / 0.75 -> 4 vs 5 = 1.25x       -> swap PARTIALLY returns: split 0 gives
                      Gibbs mrr 1.0000 vs SVI 0.9133 with node 4 at exactly 0.500, while
                      split 3 is clean (1.0000 / 1.0000)
        decay 0.50 -> 2 vs 5 = 2.5x               -> clean on all 8 splits (this gate)
        decay 0.25 -> 1 vs 5 = 5.0x               -> clean (splits 0, 3 spot-checked)
    So the SVI arm's ancestor-inflation bias (it puts more affinity on an ancestor block than
    the oracle does; see SCOPE below) is still present at decay=0.5 -- it is simply out-voted
    by a 2.5x identifiability margin. It re-emerges at 1.25x.

    Background docs (`bg_frac=0.2`) stay in TRAINING -- they are the pool the gated spectral
    seed's `n_bg` background block anchors on under `anchor_scope="frontier"`, and without
    them the background block is never seeded (see insight 0066 / the generator's docstring)
    -- but are dropped from EVAL, because `evaluate` scores a profile against a true node
    and a background doc's frontier is empty. The split therefore PERMUTES first (the
    generator appends background docs after foreground ones, so a prefix slice would contain
    none), keeps a mixed train prefix, and filters `frozenset()` labels out of the eval pool.
    Measured 32-50 background docs in each split's 200-doc training set, and no
    "no background docs under anchor_scope=" warning on any split.

    `beta_prior=0.02` (Gibbs) and an EXPLICIT `eta=0.02, alpha=0.1` (SVI, rather than the
    K-dependent 1/K default) make both engines see the same Dirichlet concentration.

    MEASURED, all 8 splits, n_train=200 / n_test=300 / SVI n_iter=50 (full-batch lr=1.0):
    Gibbs and SVI both give mrr = 1.0000 and per-true-node mrr = 1.000 for all six nodes on
    every split. auc_by_depth: Gibbs {1: 1.0, 2: 1.0} on every split; SVI depth-2 1.0 on
    every split, depth-1 1.0 on 5 splits and 0.9700 / 0.9868 / 0.9901 on the other three
    (mean 0.9934, spread 0.0300 -- the widest live spread in the gate, and what the depth
    tolerance is sized for). Mean normalized-profile L1 = 0.2356 (spread 0.1534).

    NON-TRIVIALITY -- established by MUTATION, not by headroom. mrr and per-node mrr sit at
    1.0000 for both engines, so the honest question is whether this plant would fail for a
    broken engine at all. It would: swapping only the SVI arm's init to "random" (the
    known-bad multi-domain configuration of insight 0066, where a node anchors nothing and
    its topic dies) gives mrr 0.6943 / 0.7377 / 0.7186 on splits 0 / 3 / 6, depth-1 AUC
    0.5991 / 0.6008 / 0.6319, per-node mrr as low as 0.167, and profile-L1 1.0315 / 1.0542 /
    1.0191. Every assertion below fires by a wide margin under that mutation, so a 1.0-vs-1.0
    agreement here is two engines solving a well-posed problem, not a problem too easy to
    fail. `top2` is EXCLUDED from the gate and reported here as degenerate: it is 1.0000 for
    both engines with spread 0.0000, and structurally so -- a node/ancestor swap moves the
    true node to rank 2, never worse, which `mean(rank <= 2)` cannot see.

    SCOPE -- WHAT THIS GATE DOES NOT CERTIFY. Read this before citing the test. A mutation
    matrix over the SVI arm, 8 splits, everything else held fixed (mean mrr / mean depth-1
    AUC / mean profile-L1, against the oracle's 1.0000 / 1.0000):
        production                1.0000 / 0.9934 / 0.2356   passes
        n_iter=1                  1.0000 / 0.9823 / 0.2344   PASSES
        alpha=1.0                 1.0000 / 1.0000 / 0.3818   PASSES
        eta=0.5                   1.0000 / 0.8404 / 0.3419   fires (depth-AUC, by 0.0796)
        domains=None (SP2 off)    1.0000 / 0.9148 / 0.2142   fires (depth-AUC, by 0.0052)
        init="random"             0.7325 / 0.6161 / 0.9852   fires (everything, by 0.19-0.70)
    The RANKING assertions (mrr and per-true-node mrr) are 1.0000 with a worst per-node gap
    of exactly 0.0000 for EVERY one of those arms except random init. So this gate certifies
    SEED-PLUS-FOLD-IN placement-ranking equivalence; it does NOT certify the per-domain
    variational updates, and it does NOT certify that E/M refinement happened at all -- it
    passes at n_iter=1. On this plant the placement RANKING genuinely does not depend on
    either: the spectral seed already places correctly, and 50 iterations of gated variational
    EM do not change which node wins. That is a finding about the plant and the read-out, not
    a defect to engineer around, and it is why no assertion here was contrived to depend on
    `domains`. Note the `domains=None` arm fires only by 0.0052 on a statistic whose own
    split-to-split spread is 0.0300 -- that is inside the noise and MUST NOT be relied on as
    per-domain coverage.

    The tests that DO carry what this one does not:
      * per-domain E/M being load-bearing end-to-end --
        `test_multidomain_svi_planted_bonly_node_recovery`'s (domain 0, node 2) magnitude
        gate, which FAILS at the spectral seed (recovery 0.2796) and passes only after the
        50-iteration fit (0.8615);
      * the per-domain collapsed denominator --
        `test_fit_gated_per_domain_denominator_sets_the_conditional` in
        `tests/test_dag_placement.py`, mutation-pinned against a pooled denominator.

    TOLERANCES. 0.08 (the upper end of the single-domain sibling gates) two-sided on the
    split-averaged mrr, on each node's split-averaged mrr, and on each depth's
    split-averaged AUC. What it is a tolerance FOR: (a) the stochastic collapsed-Gibbs
    fold-in (`profile`, 60 iters / 30 burn per doc) -- Monte-Carlo noise in the READ-OUT;
    (b) the stochastic SVI fold-in (`infer_local` draws gamma_init from the global RNG);
    (c) finite eval sets (300 docs/split). It is quantified by the measured split-to-split
    spread: the widest live spread in any gated statistic is 0.0300 (SVI depth-1 AUC), so
    0.08 is ~2.7x the observed spread, and the mutation above clears it by 0.25-0.40.
    EIGHT splits, not one and not five: on the pre-decay plant SVI's per-split mrr had sd
    0.0871, which puts the standard error of a FIVE-split mean at 0.039 -- the 5-split mean
    gap there was 0.0750 (inside 0.08) while the 8-split mean gap was 0.1215 (outside), so a
    5-split mean would itself have been a lucky read.

    0.45 on the mean profile L1 (a range-[0,2] statistic) is set FROM the measurement, not
    picked round: worst measured per-split value 0.3043 plus one full observed split-to-split
    range (0.1508 to 0.3043 = 0.1535) = 0.4578, taken down to 0.45. The multiplier is
    therefore "worst split + one full range", the same one-range-of-headroom rule the 0.08
    tolerance uses against its 0.0300 spread. What it CATCHES: gross profile divergence --
    the random-init mutant at mean 0.9852 (per-split max 1.0542), 2.2x the bound. What it
    does NOT catch, measured, so do not claim otherwise: affinity-CONCENTRATION drift. The
    alpha=1.0 mutant sits at mean 0.3818 (max 0.3983) and the eta=0.5 mutant at mean 0.3419,
    both UNDER 0.45 and neither caught by this bound (eta=0.5 is caught, but by the depth-AUC
    assertion, not this one). The bound cannot be tightened to catch them without dropping
    below production's own worst split plus its spread. That the two engines differ in
    concentration at all is expected and is the open calibration finding: SVI's node topics
    retain 0.17-0.51 of their domain-0 mass on the shared common pool where the Gibbs oracle
    retains 0.01-0.23, so for docs at node 2 SVI reports affinity 0.834 against the oracle's
    0.506 (truth ~0.50) -- a level difference, not a placement one.

    RUNTIME: ~92 s measured (8 splits x ~11.5 s; per split `fit_gated` ~6.5 s, Gibbs
    fold-in over 300 docs ~2.7 s, SVI fit ~2.0 s, SVI fold-in ~0.06 s, two `evaluate`
    calls). The Gibbs arms are pure-Python per-token loops and dominate."""
    import numpy as np
    from tests._stm_synth import (
        two_domain_dag_corpus, fit_gated_svi_local, svi_node_profiles)
    from spark_vi.models.topic.dag_placement import DagLayout, fit_gated, profile, evaluate
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    from spark_vi.models.topic.types import GatedBOWDocument

    parent = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
    B_ONLY = 3
    V_a, V_b, doc_len = 60, 28, 40
    n_train, n_test = 200, 300
    SPLITS = range(8)
    docs, labels, domain_bounds, _pa, _pb, _slot, _codes = two_domain_dag_corpus(
        parent=parent, node_prev={u: 1.0 for u in range(1, 7)}, V_a=V_a, V_b=V_b,
        doc_len=doc_len, seed=5, b_only_node=B_ONLY, bg_frac=0.2,
        ancestor_signature_decay=0.5)          # identifies the label; see docstring
    lay = DagLayout(parent, n_bg=2, tpn=1)
    V = domain_bounds[-1]

    def _bow_multi(tokens):
        idx, cnt = np.unique(np.asarray(tokens), return_counts=True)
        return idx.astype(np.int32), cnt.astype(np.float64), int(cnt.sum())

    g_mrr, s_mrr, l1s = [], [], []
    g_node, s_node = {u: [] for u in lay.nodes}, {u: [] for u in lay.nodes}
    depths = sorted({lay.depth(u) for u in lay.nodes})
    g_depth, s_depth = {d: [] for d in depths}, {d: [] for d in depths}

    for split_seed in SPLITS:
        perm = np.random.default_rng(split_seed).permutation(len(docs))
        tr_d = [docs[i] for i in perm[:n_train]]
        tr_raw = [labels[i] for i in perm[:n_train]]
        # Background docs (empty frontier) belong in TRAINING, not eval -- see docstring.
        assert sum(1 for f in tr_raw if isinstance(f, frozenset)) > 0
        tr_l = [f if isinstance(f, frozenset) else int(f) for f in tr_raw]
        test_idx = [i for i in perm[n_train:]
                    if not isinstance(labels[i], frozenset)][:n_test]
        te_d = [docs[i] for i in test_idx]
        te_l = [int(labels[i]) for i in test_idx]
        assert len(te_d) == n_test

        # --- Gibbs oracle (Griffiths & Steyvers 2004; per-domain factor MixEHR-style) ---
        beta_g = fit_gated(tr_d, tr_l, lay, V, beta_prior=0.02,
                           domain_bounds=domain_bounds, rng=np.random.default_rng(0))
        prof_g = [profile(d, beta_g, lay, rng=np.random.default_rng(i))
                  for i, d in enumerate(te_d)]

        # --- Gated SVI (Hoffman, Blei & Bach 2010; spectral seed via data_summary) ---
        bow = [GatedBOWDocument(*_bow_multi(d),
                                frontier=(f if isinstance(f, frozenset)
                                          else frozenset({int(f)})))
               for d, f in zip(tr_d, tr_raw)]
        ds = {"train_docs": [np.asarray(d) for d in tr_d],
              "train_labels": tr_l, "anchor_scope": "frontier"}
        m = GatedOnlineLDA(lay, vocab_size=V, domains=[V_a, V_b], eta=0.02, alpha=0.1,
                           init="spectral", random_seed=0)
        gp = fit_gated_svi_local(m, bow, n_iter=50, data_summary=ds)
        prof_s = svi_node_profiles(m, gp, te_d, lay)

        ev_g, ev_s = evaluate(prof_g, te_l, lay), evaluate(prof_s, te_l, lay)
        g_mrr.append(ev_g["mrr"]); s_mrr.append(ev_s["mrr"])
        l1s.append(_profile_l1(prof_g, prof_s, lay))
        pn_g, pn_s = _per_node_mrr(prof_g, te_l, lay), _per_node_mrr(prof_s, te_l, lay)
        # Every node must draw at least one eval doc, or the per-node loop below would index a
        # missing key. node_prev is 1.0 for all six nodes and n_test is 300, so this holds
        # comfortably (measured minimum over the 8 splits: 40 docs for the thinnest node) --
        # but it is a probabilistic property of the split, so assert it instead of trusting it.
        assert set(pn_g) == set(lay.nodes) == set(pn_s), (sorted(pn_g), sorted(pn_s))
        for u in lay.nodes:
            g_node[u].append(pn_g[u]); s_node[u].append(pn_s[u])
        for d in depths:
            g_depth[d].append(ev_g["auc_by_depth"][d]); s_depth[d].append(ev_s["auc_by_depth"][d])

    TOL = 0.08          # see TOLERANCES in the docstring; ~2.7x the widest measured spread
    mean_g, mean_s = float(np.mean(g_mrr)), float(np.mean(s_mrr))

    # Precondition: the oracle must actually solve the task, or "agreement" is meaningless.
    assert mean_g > 0.95, mean_g
    # Split-AVERAGED overall ranking agreement, two-sided (either engine drifting is a fail).
    assert abs(mean_s - mean_g) <= TOL, (mean_g, mean_s, g_mrr, s_mrr)
    # Per-TRUE-NODE ranking agreement: the assertion with power against a node/ancestor
    # rank swap, which the pooled mrr dilutes and node-AUC cannot see at all.
    for u in lay.nodes:
        gm, sm = float(np.mean(g_node[u])), float(np.mean(s_node[u]))
        assert abs(sm - gm) <= TOL, (u, gm, sm, g_node[u], s_node[u])
    # Node-AUC by depth, EVERY depth (depth 1 is scoreable on this DAG shape -- assert that
    # it is a real number rather than silently averaging a nan away).
    for d in depths:
        gd, sd = float(np.mean(g_depth[d])), float(np.mean(s_depth[d]))
        assert np.isfinite(gd) and np.isfinite(sd), (d, gd, sd)
        assert abs(sd - gd) <= TOL, (d, gd, sd, g_depth[d], s_depth[d])
    # Scale-SENSITIVE profile divergence (rank correlation and node-AUC are both blind to a
    # per-node scale offset). 0.45 = worst measured split (0.3043) + one full observed
    # split-to-split range (0.1535). Catches GROSS divergence (random init: 0.9852); does NOT
    # catch concentration drift (alpha=1.0: 0.3818, eta=0.5: 0.3419) -- see TOLERANCES.
    assert float(np.mean(l1s)) <= 0.45, (float(np.mean(l1s)), l1s)


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


# --- SP2 Task 6: per-domain modality weight omega (theta-only) + v2 seam ---


def _omega_lay():
    """Flat two-node layout: topic 0 = background, 1 = node 1's block, 2 = node 2's."""
    return DagLayout({1: 0, 2: 0}, n_bg=1, tpn=1)


def test_omega_ones_is_identity():
    """omega=None and omega=[1, 1] must reproduce the multi-domain fit EXACTLY
    (bitwise, at a fixed seed): the weight enters the gamma recurrence as a
    multiplication by 1.0, which is exact in IEEE-754, so an all-ones omega is
    not merely close to the unweighted path -- it IS the unweighted path."""
    import numpy as np
    lay = _omega_lay()
    V, doms = 8, [4, 4]
    docs = [
        GatedBOWDocument(indices=np.array([0, 5], dtype=np.int32),
                         counts=np.array([2.0, 3.0]), length=5,
                         frontier=frozenset({1})),
        GatedBOWDocument(indices=np.array([1, 2, 4, 6], dtype=np.int32),
                         counts=np.array([1.0, 4.0, 2.0, 5.0]), length=12,
                         frontier=frozenset({2})),
        GatedBOWDocument(indices=np.array([0, 3, 7], dtype=np.int32),
                         counts=np.array([1.0, 1.0, 2.0]), length=4,
                         frontier=frozenset()),
    ]

    def run(omega):
        m = GatedOnlineLDA(lay, V, domains=doms, omega=omega,
                           alpha=0.1, eta=0.02, random_seed=0)
        gp = m.initialize_global(None)
        stats = m.local_update(docs, gp)
        for _ in range(5):
            gp = m.update_global(gp, m.local_update(docs, gp), learning_rate=0.5)
        return gp, stats

    gp_none, st_none = run(None)
    gp_ones, st_ones = run([1.0, 1.0])
    for md in (0, 1):
        assert np.array_equal(gp_none["lambda"][md], gp_ones["lambda"][md]), md
    # the per-doc sufficient statistics are identical too, not just the fit endpoint
    for key in ("lambda_stats", "doc_loglik_sum", "doc_theta_kl_sum", "n_docs"):
        assert np.array_equal(st_none[key], st_ones[key]), key


def _omega_split_lambda(lay, V_m):
    """Hand-built per-domain lambda: node 1's topic explains domain-0 token 0,
    node 2's topic explains domain-1 token 0. No fit (hence no spectral seed and
    no background-doc requirement) -- the point is the READ-OUT under omega, so
    the topic-word map is planted directly and exactly."""
    import numpy as np
    lam = {0: np.full((lay.K, V_m), 0.01), 1: np.full((lay.K, V_m), 0.01)}
    lam[0][lay.block[1][0], 0] = 10.0          # domain 0 explained by node 1's topic
    lam[1][lay.block[2][0], 0] = 10.0          # domain 1 explained by node 2's topic
    return lam


def test_omega_downweights_domain_shifts_theta():
    """On a held-out doc where domain 1 is HIGH-VOLUME (20 tokens vs 2), a small
    omega_1 shifts theta toward the domain-0-explained block. Directional: the
    node-1/node-2 mass ratio must strictly increase as omega_1 shrinks, because
    omega tempers domain 1's contribution to the gamma (doc-topic) accumulation."""
    import numpy as np
    lay = _omega_lay()
    V_m, V = 4, 8
    lam = _omega_split_lambda(lay, V_m)
    # 2 domain-0 tokens (id 0) vs 20 domain-1 tokens (id V_m + 0): volume says node 2.
    row = GatedBOWDocument(indices=np.array([0, V_m], dtype=np.int32),
                           counts=np.array([2.0, 20.0]), length=22)

    def theta(omega):
        m = GatedOnlineLDA(lay, V, domains=[V_m, V_m], omega=omega, alpha=0.1)
        gp = {"lambda": lam, "alpha": m.alpha.copy(), "eta": np.array(m.eta)}
        np.random.seed(0)                       # same gamma_init for every arm
        return m.infer_local(row, gp)["theta"]

    def ratio(th):
        return float(th[lay.block[1]].sum()) / float(th[lay.block[2]].sum())

    th_ones = theta(None)
    th_down = theta([1.0, 0.05])
    assert ratio(th_ones) < 1.0, ratio(th_ones)          # volume wins at omega = 1
    assert ratio(th_down) > ratio(th_ones), (ratio(th_ones), ratio(th_down))
    assert float(th_down[lay.block[1]].sum()) > float(th_ones[lay.block[1]].sum())
    # monotone in omega_1, not just a two-point flip
    rs = [ratio(theta([1.0, w])) for w in (1.0, 0.5, 0.2, 0.05)]
    assert all(b > a for a, b in zip(rs, rs[1:])), rs


def test_omega_weights_gamma_only_sstats_and_loglik_use_true_counts():
    """omega weights theta and NOTHING else.

    Exact equivalence: for a doc whose tokens all live in domain 1, omega_1 = w
    enters the gamma recurrence as a pure count scaling (phi_norm carries no
    counts at all), so the converged gamma under omega=[1, w] on counts c is
    IDENTICAL to the gamma under omega=None on counts w*c. The lambda sufficient
    statistics and the doc log-likelihood, however, use TRUE counts -- so they
    must come out a factor w SMALLER in the scaled-count run, not equal to it.
    If omega ever leaked into the sstats or into doc_loglik, the two runs would
    agree exactly instead, and this assertion would fail."""
    import numpy as np
    lay = _omega_lay()
    V_m, V, w = 4, 8, 0.4
    idx = np.array([V_m + 0, V_m + 1, V_m + 2], dtype=np.int32)   # domain 1 only
    counts = np.array([5.0, 2.0, 9.0])
    doc = GatedBOWDocument(indices=idx, counts=counts, length=16,
                           frontier=frozenset({1}))
    doc_scaled = GatedBOWDocument(indices=idx, counts=w * counts, length=16,
                                   frontier=frozenset({1}))
    m_om = GatedOnlineLDA(lay, V, domains=[V_m, V_m], omega=[1.0, w], eta=0.02)
    m_pl = GatedOnlineLDA(lay, V, domains=[V_m, V_m], eta=0.02)
    # LOAD-BEARING: both arms must take local_update's random_seed=None branch, which
    # draws gamma_init from the GLOBAL RNG (seeded below) and so gives the two arms an
    # identical init. With random_seed set, the per-doc seed is a hash of doc.counts --
    # and the two arms' counts differ by construction (c vs w*c), so the inits would
    # diverge and the exact factor-w equality below would fail for a reason that has
    # nothing to do with omega. Do not add random_seed= to these constructors.
    assert m_om.random_seed is None and m_pl.random_seed is None
    np.random.seed(3)
    gp = m_om.initialize_global(None)          # shared global params for both arms
    np.random.seed(4)
    st_om = m_om.local_update([doc], gp)
    np.random.seed(4)                          # same gamma_init draw
    st_pl = m_pl.local_update([doc_scaled], gp)
    np.testing.assert_allclose(st_pl["lambda_stats"], w * st_om["lambda_stats"],
                               rtol=1e-12, atol=0)
    np.testing.assert_allclose(float(st_pl["doc_loglik_sum"]),
                               w * float(st_om["doc_loglik_sum"]), rtol=1e-12)
    # ... and the omega run's own sstats reconstruct the TRUE counts: summing the
    # phi-weighted topic mass over a token's column gives back count_n (not w*count_n),
    # since sum_k expElogbeta[k,n] * sstats[k,n] = count_n * (phi_norm - 1e-100)/phi_norm.
    eb = m_om._assemble_expElogbeta(gp["lambda"])
    allowed = lay.allowed_set(doc.frontier)
    sel = np.ix_(allowed, idx)
    recon = (eb[sel] * st_om["lambda_stats"][sel]).sum(axis=0)
    np.testing.assert_allclose(recon, counts, rtol=1e-12)
    # sanity: omega DID move the fit (otherwise the above is vacuous)
    np.random.seed(4)
    st_ref = m_pl.local_update([doc], gp)
    assert not np.allclose(st_om["lambda_stats"], st_ref["lambda_stats"])


def test_omega_leaves_phi_norm_omega_free():
    """phi_norm must stay the omega-FREE function of (gamma, expElogbeta) it is in
    the unweighted recurrence: eb_d.T @ expElogthetad + 1e-100. Checked at the
    converged gamma of a weighted run, so a refactor that folded the per-token
    weight into phi_norm (and thus into the data log-likelihood) would fail here."""
    import numpy as np
    from spark_vi.models.topic.lda import _cavi_doc_inference
    rng = np.random.default_rng(0)
    K, n_unique = 3, 5
    indices = np.array([0, 2, 5, 6, 9], dtype=np.int32)
    counts = np.array([3.0, 1.0, 7.0, 2.0, 4.0])
    eb = rng.gamma(1.0, 1.0, size=(K, 10))
    w_tok = np.array([1.0, 1.0, 0.3, 0.3, 0.3])          # domain-1 tokens downweighted
    gamma, eth, phi_norm, _ = _cavi_doc_inference(
        indices=indices, counts=counts, expElogbeta=eb, alpha=0.1,
        gamma_init=np.full(K, 1.0), max_iter=50, tol=1e-9,
        gamma_count_weight=w_tok)
    assert phi_norm.shape == (n_unique,)
    np.testing.assert_array_equal(phi_norm, eb[:, indices].T @ eth + 1e-100)
    # and the weight is not a no-op on gamma
    g_unw, _, _, _ = _cavi_doc_inference(
        indices=indices, counts=counts, expElogbeta=eb, alpha=0.1,
        gamma_init=np.full(K, 1.0), max_iter=50, tol=1e-9)
    assert not np.allclose(gamma, g_unw)


def test_omega_requires_domains():
    """omega with domains=None is a contradiction (no modality to weight) and must
    raise an explicit ValueError, not be silently ignored."""
    import pytest
    with pytest.raises(ValueError, match="omega"):
        GatedOnlineLDA(_omega_lay(), 8, omega=[1.0, 1.0])


def test_omega_length_mismatch_raises():
    import pytest
    with pytest.raises(ValueError, match="omega"):
        GatedOnlineLDA(_omega_lay(), 8, domains=[4, 4], omega=[1.0, 1.0, 1.0])


def test_omega_rejects_negative_and_nonfinite_in_every_input_form():
    """The finite/nonnegative check must apply to the SCALAR and 0-d forms too, not
    just to sequences: the scalar broadcast is a separate dispatch branch, and if it
    returns before validation a negative omega produces NEGATIVE theta mass, which
    flows into node_affinity as a negative node score and silently corrupts the
    placement ranking (no exception, no warning). Every accepted input form is
    covered here because the scalar dispatch is exactly where that hole opened."""
    import numpy as np
    import pytest
    lay = _omega_lay()
    bad_scalars = [-0.5, np.array(-0.5), np.float64(-0.5),
                   float("nan"), float("inf"), np.array(float("nan"))]
    for val in bad_scalars:
        with pytest.raises(ValueError, match="omega"):
            GatedOnlineLDA(lay, 8, domains=[4, 4], omega=val)
    bad_sequences = [[1.0, -0.5], (1.0, float("nan")), np.array([1.0, float("inf")]),
                     [-1.0, -1.0]]
    for seq in bad_sequences:
        with pytest.raises(ValueError, match="omega"):
            GatedOnlineLDA(lay, 8, domains=[4, 4], omega=seq)
    # 0.0 is legal (drop a domain from theta entirely), in both forms
    np.testing.assert_allclose(
        GatedOnlineLDA(lay, 8, domains=[4, 4], omega=0.0).omega, [0.0, 0.0])
    np.testing.assert_allclose(
        GatedOnlineLDA(lay, 8, domains=[4, 4], omega=[1.0, 0.0]).omega, [1.0, 0.0])


def test_omega_scalar_broadcasts_to_all_domains():
    """A scalar omega applies the same weight to every domain (a global tempering
    of the doc-topic accumulation). A 0-d ndarray must behave like the equivalent
    python float -- np.isscalar(np.array(0.5)) is False, so scalar dispatch here
    uses np.ndim."""
    import numpy as np
    lay = _omega_lay()
    for val in (0.5, np.array(0.5), np.float64(0.5)):
        m = GatedOnlineLDA(lay, 8, domains=[4, 4], omega=val)
        np.testing.assert_allclose(m.omega, [0.5, 0.5])
    assert GatedOnlineLDA(lay, 8, domains=[4, 4]).omega is None
    # every sequence form resolves the same way (a generator must not be consumed
    # by validation before it reaches the array)
    for seq in ([1.0, 0.5], (1.0, 0.5), np.array([1.0, 0.5]), (x for x in (1.0, 0.5))):
        m = GatedOnlineLDA(lay, 8, domains=[4, 4], omega=seq)
        np.testing.assert_allclose(m.omega, [1.0, 0.5])


# --- SP2 Task 7: per-domain theta-contribution instrument + v2 seam ---


def _contrib_docs():
    """Three-doc hand-built two-domain batch (V_m = 4 each, so domain 1 is ids 4..7).

    Hand-built rather than planted: the instrument is an EXACT accounting identity
    on the gamma recurrence, so the per-domain token volumes have to be known
    exactly (4 in domain 0, 14 in domain 1). A planted corpus would only let us
    assert an inequality. Volumes by domain:
        doc 0: 2 + 3     doc 1: 1 + 9     doc 2: 1 + 2   ->  (4, 14)
    """
    return [
        GatedBOWDocument(indices=np.array([0, 5], dtype=np.int32),
                         counts=np.array([2.0, 3.0]), length=5,
                         frontier=frozenset({1})),
        GatedBOWDocument(indices=np.array([1, 4, 6], dtype=np.int32),
                         counts=np.array([1.0, 4.0, 5.0]), length=10,
                         frontier=frozenset({2})),
        GatedBOWDocument(indices=np.array([0, 7], dtype=np.int32),
                         counts=np.array([1.0, 2.0]), length=3,
                         frontier=frozenset()),
    ]


def test_theta_contribution_by_domain_reported():
    """local_update emits the per-domain theta-contribution instrument.

    Length n_domains, nonnegative, and LARGER for the higher-volume domain under
    omega = 1 -- the volume-imbalance read the arc design needs to tune omega at
    all. Under omega = 1 the documented definition reduces to the per-domain token
    volume exactly (the 1e-100 phi_norm guard is below float64 resolution here), so
    the assertion is on the known volumes (4, 14), not on a re-run of the code.
    domains=None must NOT grow the key (single-domain stats dict unchanged)."""
    import numpy as np
    lay = _omega_lay()
    V, doms = 8, [4, 4]
    docs = _contrib_docs()

    m = GatedOnlineLDA(lay, V, domains=doms, alpha=0.1, eta=0.02, random_seed=0)
    gp = m.initialize_global(None)
    st = m.local_update(docs, gp)
    c = np.asarray(st["theta_contribution_by_domain"])
    assert c.shape == (len(doms),), c.shape
    assert np.all(c >= 0.0), c
    assert c[1] > c[0], c
    np.testing.assert_allclose(c, [4.0, 14.0], rtol=1e-12)

    m1 = GatedOnlineLDA(lay, V, alpha=0.1, eta=0.02, random_seed=0)
    st1 = m1.local_update(docs, m1.initialize_global(None))
    assert "theta_contribution_by_domain" not in st1, sorted(st1)


def test_theta_contribution_equals_gamma_increment_from_cavi():
    """The stat IS the per-domain partition of the gamma increment, not a proxy.

    For one document the CAVI recurrence is
        gamma = alpha + expElogtheta_d * (eb_d @ (w * counts / phi_norm))
    so the total evidence mass the tokens add to gamma over the prior is
    sum_k (gamma_k - alpha_k). This test reproduces that sweep independently (same
    gamma_init off the global RNG) and asserts the emitted stat sums to it. Run
    under omega != 1 so the weighted form is the one being checked."""
    import numpy as np
    from spark_vi.models.topic.lda import _cavi_doc_inference
    lay = _omega_lay()
    V_m, V = 4, 8
    doc = GatedBOWDocument(indices=np.array([1, 2, 4, 6], dtype=np.int32),
                           counts=np.array([3.0, 1.0, 7.0, 2.0]), length=13,
                           frontier=frozenset({1}))
    m = GatedOnlineLDA(lay, V, domains=[V_m, V_m], omega=[1.0, 0.3],
                       alpha=0.1, eta=0.02)
    assert m.random_seed is None            # LOAD-BEARING: gamma_init off global RNG
    gp = {"lambda": _omega_split_lambda(lay, V_m),
          "alpha": m.alpha.copy(), "eta": np.array(m.eta)}

    np.random.seed(11)
    st = m.local_update([doc], gp)

    allowed = lay.allowed_set(doc.frontier)
    np.random.seed(11)                      # same draw local_update just consumed
    gamma_init = np.random.gamma(shape=m.gamma_shape, scale=1.0 / m.gamma_shape,
                                 size=len(allowed))
    eb = m._assemble_expElogbeta(gp["lambda"])[allowed]
    a_d = gp["alpha"][allowed]
    w_tok = m._gamma_count_weight(doc.indices)
    _, eth, phi_norm, _ = _cavi_doc_inference(
        indices=doc.indices, counts=doc.counts, expElogbeta=eb, alpha=a_d,
        gamma_init=gamma_init, max_iter=m.cavi_max_iter, tol=m.cavi_tol,
        gamma_count_weight=w_tok)
    gamma_next = a_d + eth * (eb[:, doc.indices] @ (w_tok * doc.counts / phi_norm))
    increment = float((gamma_next - a_d).sum())

    c = np.asarray(st["theta_contribution_by_domain"])
    np.testing.assert_allclose(c.sum(), increment, rtol=1e-12)
    # and the split is by TOKEN domain: ids 1,2 are domain 0, ids 4,6 are domain 1
    np.testing.assert_allclose(c, [3.0 + 1.0, 0.3 * (7.0 + 2.0)], rtol=1e-12)


def test_theta_contribution_responds_to_omega():
    """The instrument measures the OMEGA-WEIGHTED mass it claims, not raw volume.

    Down-weighting domain 1 must scale its reported contribution by exactly that
    weight and leave domain 0's alone; omega_1 = 0 (a domain dropped from theta)
    must report exactly zero. Without this the docstring's definition would be
    unverified prose and the stat could silently be plain token volume."""
    import numpy as np
    lay = _omega_lay()
    V, doms = 8, [4, 4]
    docs = _contrib_docs()

    def contrib(omega):
        m = GatedOnlineLDA(lay, V, domains=doms, omega=omega,
                           alpha=0.1, eta=0.02, random_seed=0)
        gp = m.initialize_global(None)
        return np.asarray(m.local_update(docs, gp)["theta_contribution_by_domain"])

    ref = contrib(None)
    down = contrib([1.0, 0.25])
    np.testing.assert_allclose(down[0], ref[0], rtol=1e-12)
    np.testing.assert_allclose(down[1], 0.25 * ref[1], rtol=1e-12)
    assert down[1] < down[0]                      # the imbalance read flips
    dropped = contrib([1.0, 0.0])
    assert dropped[1] == 0.0 and dropped[0] > 0.0, dropped
    np.testing.assert_allclose(contrib(0.5), 0.5 * ref, rtol=1e-12)


def test_theta_contribution_sums_across_partitions():
    """The stat is additive: the default combine_stats (elementwise sum) must give
    the whole batch's contribution from two partitions' halves, so a distributed
    fit reads the same imbalance a local one does."""
    import numpy as np
    lay = _omega_lay()
    m = GatedOnlineLDA(lay, 8, domains=[4, 4], alpha=0.1, eta=0.02, random_seed=0)
    gp = m.initialize_global(None)
    docs = _contrib_docs()
    whole = m.local_update(docs, gp)
    merged = m.combine_stats(m.local_update(docs[:1], gp), m.local_update(docs[1:], gp))
    np.testing.assert_allclose(merged["theta_contribution_by_domain"],
                               whole["theta_contribution_by_domain"], rtol=1e-12)


def test_v2_seam_per_token_domain_live():
    """_token_domains is the v2 seam: each token's domain, resolved where gamma/phi
    are formed. Asymmetric domain widths so a transposed bound would show; the
    BLOCK BOUNDARIES (last id of domain 0, first id of domain 1) are the point --
    an off-by-one in searchsorted's side= is exactly what this catches."""
    import numpy as np
    import pytest
    lay = _omega_lay()
    m = GatedOnlineLDA(lay, 8, domains=[3, 5])         # bounds [0, 3, 8]
    np.testing.assert_array_equal(m._token_domains(np.arange(8)),
                                  [0, 0, 0, 1, 1, 1, 1, 1])
    assert int(m._token_domains(np.array([2]))[0]) == 0     # last id of domain 0
    assert int(m._token_domains(np.array([3]))[0]) == 1     # first id of domain 1
    assert m._token_domains(np.array([], dtype=np.int32)).size == 0
    # three domains incl. a width-1 block: bounds [0, 2, 5, 6]
    m3 = GatedOnlineLDA(lay, 6, domains=[2, 3, 1])
    np.testing.assert_array_equal(m3._token_domains(np.arange(6)),
                                  [0, 0, 1, 1, 1, 2])
    assert int(m3._token_domains(np.array([5]))[0]) == 2    # last id of the last domain
    # Out of range must be a NAMED ValueError. searchsorted saturates instead of
    # failing (id 8 with domains=[4,4] returns 2), which the omega gather surfaces
    # as an opaque "index 2 is out of bounds" IndexError -- and would silently read
    # the WRONG domain if the weight array ever grew a sentinel entry.
    mo = GatedOnlineLDA(lay, 8, domains=[4, 4], omega=[1.0, 0.5])
    for bad in ([8], [-1], [0, 4, 99]):
        with pytest.raises(ValueError, match="outside the vocabulary"):
            mo._token_domains(np.array(bad, dtype=np.int64))
    bad_doc = GatedBOWDocument(indices=np.array([8], dtype=np.int32),
                               counts=np.array([1.0]), length=1,
                               frontier=frozenset({1}))
    with pytest.raises(ValueError, match="outside the vocabulary"):
        mo.local_update([bad_doc], mo.initialize_global(None))
    # domains=None has no domain axis
    with pytest.raises(ValueError, match="multi-domain"):
        GatedOnlineLDA(lay, 8)._token_domains(np.array([0]))


def test_iteration_summary_multidomain_surfaces_eta_lambda_and_theta_contribution():
    """iteration_summary must SURVIVE multi-domain and surface the instrument.

    The base OnlineLDA implementation does float(global_params["eta"]) and
    lam.sum(axis=1) -- both raise on a per-domain eta sequence / dict lambda, so
    VIRunner (which calls this every iteration) broke outright in multi-domain
    mode. Deleting the override makes this test raise, which is the regression
    guard. The theta-contribution appears only after the first M-step has handed
    the aggregated stat over."""
    import numpy as np
    lay = _omega_lay()
    m = GatedOnlineLDA(lay, 8, domains=[4, 4], eta=[0.02, 0.05],
                       alpha=0.1, random_seed=0)
    gp = m.initialize_global(None)
    s0 = m.iteration_summary(gp)
    assert isinstance(s0, str) and s0 and "\n" not in s0
    assert "θ_contrib_m" not in s0                     # nothing aggregated yet
    assert "η_m=[0.02, 0.05]" in s0, s0
    assert "Σλ_k[m0:" in s0 and "Σλ_k[m1:" in s0, s0

    gp = m.update_global(gp, m.local_update(_contrib_docs(), gp), learning_rate=1.0)
    s1 = m.iteration_summary(gp)
    assert "\n" not in s1
    assert "θ_contrib_m=[4, 14]" in s1, s1        # the (4, 14) volumes, .4g-formatted
    assert "η_m=[0.02, 0.05]" in s1, s1


def test_iteration_summary_domains_none_is_byte_identical_to_base():
    """domains=None must produce the base OnlineLDA string byte-for-byte -- the
    override may only ever branch, never reformat the single-domain line."""
    import numpy as np
    from spark_vi.models.topic.lda import OnlineLDA
    lay = _omega_lay()
    m = GatedOnlineLDA(lay, 8, alpha=0.1, eta=0.02, random_seed=0)
    base = OnlineLDA(K=lay.K, vocab_size=8, alpha=0.1, eta=0.02, random_seed=0)
    gp = m.initialize_global(None)
    assert m.iteration_summary(gp) == base.iteration_summary(gp)
    gp2 = m.update_global(gp, m.local_update(_contrib_docs(), gp), learning_rate=0.5)
    assert m.iteration_summary(gp2) == base.iteration_summary(gp2)
