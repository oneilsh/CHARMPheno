import numpy as np
import pytest
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


from spark_vi.dag import DagLayout
from spark_vi.models.topic.dag_gated_lda import DagGatedOnlineLDA

PARENT = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}


def _lay():
    return DagLayout(PARENT, n_bg=2, tpn=1)


def test_local_update_gates_sstats_to_allowed_rows_only():
    lay = _lay()
    V = 30
    m = DagGatedOnlineLDA(lay, V, alpha=0.1, eta=0.02, random_seed=0)
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
    m = DagGatedOnlineLDA(lay, V, alpha=0.1, eta=0.02, random_seed=0)
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


def test_local_update_gamma_init_is_content_deterministic_with_seed():
    # Regression: DagGatedOnlineLDA.local_update must inherit OnlineLDA's content-deterministic
    # gamma_init seeding (blake2b hash of random_seed + doc content) so a distributed fit is
    # reproducible regardless of Spark partition/executor/iteration order — not draw gamma_init
    # from the plain global RNG, which would make lambda_stats vary run-to-run for the same docs.
    lay = _lay()
    V = 30
    m = DagGatedOnlineLDA(lay, V, alpha=0.1, eta=0.02, random_seed=0)
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


def test_compute_elbo_is_finite_and_gated_doc_kl_is_positive():
    # FIX 2: the gated local_update must accumulate a real per-doc Dirichlet KL (over the
    # doc's allowed sub-simplex), not the hardcoded 0.0 surrogate, so compute_elbo (and
    # VIRunner's convergence check on the production shim path) is a real variational bound.
    lay = _lay()
    V = 30
    m = DagGatedOnlineLDA(lay, V, alpha=0.1, eta=0.02, random_seed=0)
    gp = m.initialize_global(None)
    doc = GatedBOWDocument(indices=np.array([5, 6], dtype=np.int32),
                           counts=np.array([2.0, 1.0]), length=3,
                           frontier=frozenset({3}))
    stats = m.local_update([doc], gp)
    assert float(stats["doc_theta_kl_sum"]) > 0.0
    elbo = m.compute_elbo(gp, stats)
    assert np.isfinite(elbo)


from _gibbs_oracle import fit_gated, profile
from _stm_synth import (
    dag_placement_corpus, fit_gated_svi_local, placement_metrics, svi_node_profiles,
)


@pytest.mark.slow
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
    ev_g = placement_metrics([profile(d, beta_g, lay, rng=np.random.default_rng(i))
                     for i, d in enumerate(te_d)], te_l, lay)

    # Gated SVI (random init default)
    bow = [GatedBOWDocument(*_bow(d), frontier=frozenset({int(y)}))
           for d, y in zip(tr_d, tr_l)]
    m = DagGatedOnlineLDA(lay, V, alpha=0.1, eta=0.02, random_seed=0)
    gp = fit_gated_svi_local(m, bow, n_iter=250)
    ev_s = placement_metrics(svi_node_profiles(m, gp, te_d, lay), te_l, lay)

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


@pytest.mark.slow
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
    ev_g = placement_metrics([profile(d, beta_g, lay, rng=np.random.default_rng(i))
                     for i, d in enumerate(te_d)], te_l, lay)

    bow = [GatedBOWDocument(*_bow(d), frontier=f) for d, f in zip(tr_d, tr_l)]
    m = DagGatedOnlineLDA(lay, V, alpha=0.1, eta=0.02, random_seed=0)
    gp = fit_gated_svi_local(m, bow, n_iter=250)
    ev_s = placement_metrics(svi_node_profiles(m, gp, te_d, lay), te_l, lay)

    max_depth = max(lay.depth(u) for u in lay.nodes)
    assert ev_s["auc_by_depth"][max_depth] >= ev_g["auc_by_depth"][max_depth] - 0.08
    assert ev_s["mrr"] >= ev_g["mrr"] - 0.08


def test_gated_optimize_alpha_requires_frontier_histogram():
    import pytest
    from spark_vi.dag import DagLayout
    from spark_vi.models.topic.dag_gated_lda import DagGatedOnlineLDA
    lay = DagLayout({1: 0, 2: 0}, n_bg=2, tpn=1)
    with pytest.raises(ValueError, match="frontier_histogram"):
        DagGatedOnlineLDA(lay, vocab_size=5, optimize_alpha=True)   # no histogram

def test_gated_tied_layout_shapes():
    import numpy as np
    from spark_vi.dag import DagLayout
    from spark_vi.models.topic.dag_gated_lda import DagGatedOnlineLDA
    lay = DagLayout({1: 0, 2: 0, 3: 1}, n_bg=2, tpn=1)          # 3 nodes
    m = DagGatedOnlineLDA(lay, vocab_size=5, optimize_alpha=True,
                       frontier_histogram={frozenset({3}): 4, frozenset(): 6})
    # tied layout: [bg, node1, node2, node3]
    assert list(m._block_sizes) == [2, 1, 1, 1]
    # topic_to_tied: topics 0,1 -> 0 (bg); block(1)->1, block(2)->2, block(3)->3
    assert m._topic_to_tied[0] == 0 and m._topic_to_tied[1] == 0
    assert m._topic_to_tied[lay.block[1][0]] == 1
    assert m._topic_to_tied[lay.block[3][0]] == 3

def test_gated_local_update_emits_node_theta_stat():
    import numpy as np
    from spark_vi.dag import DagLayout
    from spark_vi.models.topic.dag_gated_lda import DagGatedOnlineLDA
    from spark_vi.models.topic.types import GatedBOWDocument
    lay = DagLayout({1: 0, 2: 0}, n_bg=2, tpn=1)                # B = 1 + 2 = 3
    m = DagGatedOnlineLDA(lay, vocab_size=6, optimize_alpha=True, random_seed=0,
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
    from spark_vi.dag import DagLayout
    from spark_vi.models.topic.dag_gated_lda import DagGatedOnlineLDA
    from spark_vi.models.topic.types import GatedBOWDocument
    lay = DagLayout({1: 0, 2: 0}, n_bg=2, tpn=1)
    m = DagGatedOnlineLDA(lay, vocab_size=6, random_seed=0)        # optimize_alpha False
    gp = m.initialize_global(None)
    out = m.local_update([GatedBOWDocument(indices=np.array([0], np.int32),
                                           counts=np.array([1.0]), length=1,
                                           frontier=frozenset({1}))], gp)
    assert "e_log_theta_node_sum" not in out


def test_gated_update_global_leaves_alpha_fixed_when_disabled():
    import numpy as np
    from spark_vi.dag import DagLayout
    from spark_vi.models.topic.dag_gated_lda import DagGatedOnlineLDA
    lay = DagLayout({1: 0, 2: 0}, n_bg=2, tpn=1)
    m = DagGatedOnlineLDA(lay, vocab_size=6, random_seed=0)          # alpha fixed
    gp = m.initialize_global(None)
    alpha0 = gp["alpha"].copy()
    stats = {"lambda_stats": np.ones((lay.K, 6)), "n_docs": np.array(4.0)}
    gp2 = m.update_global(gp, stats, learning_rate=0.5)
    assert np.array_equal(gp2["alpha"], alpha0)                   # unchanged
    assert not np.array_equal(gp2["lambda"], gp["lambda"])        # lambda still moves

def test_gated_update_global_learns_asymmetric_alpha():
    import numpy as np
    from spark_vi.dag import DagLayout
    from spark_vi.models.topic.dag_gated_lda import DagGatedOnlineLDA
    lay = DagLayout({1: 0, 2: 0}, n_bg=2, tpn=1)                  # B=3: bg,node1,node2
    hist = {frozenset({1}): 30, frozenset({2}): 30, frozenset(): 40}
    m = DagGatedOnlineLDA(lay, vocab_size=6, optimize_alpha=True, random_seed=0,
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


# --- SP2 Task 1: per-domain dict-lambda storage + assemble/split + init ---

def test_domains_none_is_single_array_unchanged():
    import numpy as np
    from spark_vi.dag import DagLayout
    from spark_vi.models.topic.dag_gated_lda import DagGatedOnlineLDA
    lay = DagLayout({1: 0, 2: 1, 3: 1}, n_bg=2, tpn=1)
    m = DagGatedOnlineLDA(lay, vocab_size=12, random_seed=7)          # domains=None
    gp = m.initialize_global(None)
    assert isinstance(gp["lambda"], np.ndarray) and gp["lambda"].shape == (lay.K, 12)


def test_multidomain_init_is_per_domain_dict():
    import numpy as np
    from spark_vi.dag import DagLayout
    from spark_vi.models.topic.dag_gated_lda import DagGatedOnlineLDA
    lay = DagLayout({1: 0, 2: 1, 3: 1}, n_bg=2, tpn=1)
    m = DagGatedOnlineLDA(lay, vocab_size=56, domains=[40, 16], random_seed=7)
    gp = m.initialize_global(None)
    lam = gp["lambda"]
    assert set(lam) == {0, 1}
    assert lam[0].shape == (lay.K, 40) and lam[1].shape == (lay.K, 16)


def test_assemble_split_round_trip():
    import numpy as np
    from spark_vi.dag import DagLayout
    from spark_vi.models.topic.dag_gated_lda import DagGatedOnlineLDA
    lay = DagLayout({1: 0, 2: 1, 3: 1}, n_bg=2, tpn=1)
    m = DagGatedOnlineLDA(lay, vocab_size=7, domains=[4, 3], random_seed=1)
    lam = {0: np.abs(np.random.default_rng(0).normal(size=(lay.K, 4))) + .1,
           1: np.abs(np.random.default_rng(1).normal(size=(lay.K, 3))) + .1}
    eb = m._assemble_expElogbeta(lam)
    assert eb.shape == (lay.K, 7)
    # EACH block equals its OWN full-row normalization -- domain 1 is checked by
    # value too, not just by shape: a bug that normalized only the second block
    # against the wrong row mass is precisely what this test exists to catch, and a
    # shape-only assertion on domain 1 would pass straight through it.
    from scipy.special import digamma
    np.testing.assert_allclose(eb[:, :4], np.exp(digamma(lam[0]) - digamma(lam[0].sum(1, keepdims=True))))
    np.testing.assert_allclose(eb[:, 4:], np.exp(digamma(lam[1]) - digamma(lam[1].sum(1, keepdims=True))))
    # cross-check the type-safety claim: normalizing the CONCATENATED array instead
    # (the failure a single pooled array cannot prevent) gives a different answer.
    pooled = np.exp(digamma(np.concatenate([lam[0], lam[1]], axis=1))
                    - digamma(np.concatenate([lam[0], lam[1]], axis=1).sum(1, keepdims=True)))
    assert not np.allclose(eb, pooled)
    # split of a concatenated array round-trips the block shapes AND the values
    back = m._split_to_domains(eb)
    assert back[0].shape == (lay.K, 4) and back[1].shape == (lay.K, 3)
    np.testing.assert_allclose(back[0], eb[:, :4])
    np.testing.assert_allclose(back[1], eb[:, 4:])


def test_domains_must_sum_to_vocab():
    import pytest
    from spark_vi.dag import DagLayout
    from spark_vi.models.topic.dag_gated_lda import DagGatedOnlineLDA
    lay = DagLayout({1: 0, 2: 1, 3: 1}, n_bg=2, tpn=1)
    with pytest.raises(ValueError):
        DagGatedOnlineLDA(lay, vocab_size=56, domains=[40, 10])   # 50 != 56


# --- SP2 Task 2: dict-aware E/M-step + per-domain eta + multi-domain ELBO ---

@pytest.mark.slow
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
    from spark_vi.dag import DagLayout
    from spark_vi.models.topic.dag_gated_lda import DagGatedOnlineLDA
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
    m = DagGatedOnlineLDA(lay, vocab_size=V, domains=[40, 16], random_seed=0)
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
        m = DagGatedOnlineLDA(lay, V, alpha=0.1, eta=0.02, random_seed=0)
        gp = m.initialize_global(None)
        for _ in range(5):
            gp = m.update_global(gp, m.local_update(docs, gp), learning_rate=0.5)
        return gp

    gp1, gp2 = run(), run()
    assert isinstance(gp1["lambda"], np.ndarray)
    np.testing.assert_array_equal(gp1["lambda"], gp2["lambda"])


def test_multidomain_compute_elbo_matches_manual_per_domain_kl():
    """ELBO's global KL term must equal the hand-computed
    Sigma_k Sigma_m KL(Dirichlet(lam_m[k]) || Dirichlet(eta . 1_{V_m})) — NOT a
    single KL over the naively concatenated vector (Dirichlet KL does not
    decompose that way across a concatenation)."""
    import numpy as np
    from spark_vi.dag import DagLayout
    from spark_vi.models.topic.dag_gated_lda import DagGatedOnlineLDA
    from spark_vi.models.topic.lda import _dirichlet_kl
    lay = DagLayout({1: 0, 2: 1, 3: 1}, n_bg=2, tpn=1)
    m = DagGatedOnlineLDA(lay, vocab_size=7, domains=[4, 3], eta=0.5, random_seed=3)
    gp = m.initialize_global(None)
    aggregated_stats = {"doc_loglik_sum": np.array(-12.3), "doc_theta_kl_sum": np.array(1.7)}
    elbo = m.compute_elbo(gp, aggregated_stats)

    manual_kl = 0.0
    for md, V_m in enumerate([4, 3]):
        eta_vec = np.full(V_m, 0.5)
        for k in range(lay.K):
            manual_kl += _dirichlet_kl(gp["lambda"][md][k], eta_vec)
    expected = (float(aggregated_stats["doc_loglik_sum"])
                - float(aggregated_stats["doc_theta_kl_sum"]) - manual_kl)
    np.testing.assert_allclose(elbo, expected)


def test_multidomain_infer_local_assembles_dict_lambda():
    """infer_local (deployment fold-in) must work off the dict-lambda in multi-domain
    mode: assemble the concatenated expElogbeta and run the same full-K CAVI."""
    import numpy as np
    from spark_vi.dag import DagLayout
    from spark_vi.models.topic.dag_gated_lda import DagGatedOnlineLDA
    lay = DagLayout({1: 0, 2: 1, 3: 1}, n_bg=2, tpn=1)
    m = DagGatedOnlineLDA(lay, vocab_size=7, domains=[4, 3], random_seed=2)
    gp = m.initialize_global(None)
    row = GatedBOWDocument(indices=np.array([0, 4], dtype=np.int32),
                           counts=np.array([2.0, 1.0]), length=3)
    out = m.infer_local(row, gp)
    assert out["theta"].shape == (lay.K,)
    np.testing.assert_allclose(out["theta"].sum(), 1.0)
    assert np.isfinite(out["gamma"]).all()


# --- multi-domain iteration summary / metadata ---


def _flat_lay():
    """Flat two-node layout: topic 0 = background, 1 = node 1's block, 2 = node 2's."""
    return DagLayout({1: 0, 2: 0}, n_bg=1, tpn=1)


def _two_domain_docs():
    """Three-doc hand-built two-domain batch (V_m = 4 each, so domain 1 is ids 4..7)."""
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


def test_iteration_summary_multidomain_surfaces_eta_and_per_domain_lambda():
    """iteration_summary must SURVIVE multi-domain.

    The base OnlineLDA implementation does lam.sum(axis=1), which raises on a
    per-domain dict lambda, so VIRunner (which calls this every iteration) broke
    outright in multi-domain mode. Deleting the override makes this test raise,
    which is the regression guard."""
    lay = _flat_lay()
    m = DagGatedOnlineLDA(lay, 8, domains=[4, 4], eta=0.02, alpha=0.1, random_seed=0)
    gp = m.initialize_global(None)
    s0 = m.iteration_summary(gp)
    assert isinstance(s0, str) and s0 and "\n" not in s0
    assert "η=0.02" in s0, s0
    assert "Σλ_k[m0:" in s0 and "Σλ_k[m1:" in s0, s0
    gp = m.update_global(gp, m.local_update(_two_domain_docs(), gp), learning_rate=1.0)
    s1 = m.iteration_summary(gp)
    assert "\n" not in s1 and "Σλ_k[m1:" in s1


def test_iteration_summary_domains_none_is_byte_identical_to_base():
    """domains=None must produce the base OnlineLDA string byte-for-byte -- the
    override may only ever branch, never reformat the single-domain line."""
    import numpy as np
    from spark_vi.models.topic.lda import OnlineLDA
    lay = _flat_lay()
    m = DagGatedOnlineLDA(lay, 8, alpha=0.1, eta=0.02, random_seed=0)
    base = OnlineLDA(K=lay.K, vocab_size=8, alpha=0.1, eta=0.02, random_seed=0)
    gp = m.initialize_global(None)
    assert m.iteration_summary(gp) == base.iteration_summary(gp)
    gp2 = m.update_global(gp, m.local_update(_two_domain_docs(), gp), learning_rate=0.5)
    assert m.iteration_summary(gp2) == base.iteration_summary(gp2)


# --- final-review fix wave: shared resolvers, init dispatch, empty partition ---


def test_eta_accepts_a_0d_ndarray_like_the_equivalent_float():
    """Every scalar spelling of eta (float, numpy scalar, 0-d array) is the same
    prior in multi-domain mode."""
    import numpy as np
    lay = _flat_lay()
    for val in (0.02, np.float64(0.02), np.array(0.02)):
        m = DagGatedOnlineLDA(lay, 8, domains=[4, 4], eta=val)
        assert float(m.eta) == 0.02


def test_multidomain_get_metadata_carries_reconstruction_constants():
    """A saved multi-domain result must be reconstructable: the per-domain
    widths have to travel in metadata."""
    import numpy as np
    from spark_vi.dag import DagLayout
    from spark_vi.models.topic.dag_gated_lda import DagGatedOnlineLDA
    lay = DagLayout({1: 0, 2: 1, 3: 1}, n_bg=2, tpn=1)
    m = DagGatedOnlineLDA(lay, vocab_size=56, domains=[40, 16], eta=0.02,
                       random_seed=0)
    md = m.get_metadata()
    assert md["K"] == lay.K and md["V"] == 56
    assert md["domains"] == [40, 16]
    # JSON-serializable: metadata is written verbatim into manifest.json.
    import json
    json.dumps(md)


def test_single_domain_get_metadata_unchanged():
    """domains=None must return exactly the base contract, so existing archives
    and consumers see no new keys."""
    from spark_vi.dag import DagLayout
    from spark_vi.models.topic.dag_gated_lda import DagGatedOnlineLDA
    lay = DagLayout({1: 0, 2: 1, 3: 1}, n_bg=2, tpn=1)
    m = DagGatedOnlineLDA(lay, vocab_size=12, random_seed=0)
    assert m.get_metadata() == {"K": lay.K, "V": 12}


def test_optimize_eta_rejected_pins_the_eta_provenance_invariant():
    """Multi-domain reads eta from instance state; single-domain reads it from
    global_params["eta"]. Those two sources are equivalent ONLY because eta
    cannot change during a fit -- optimize_eta is rejected outright. This test
    pins that rejection."""
    import pytest
    from spark_vi.dag import DagLayout
    from spark_vi.models.topic.dag_gated_lda import DagGatedOnlineLDA
    lay = DagLayout({1: 0, 2: 1, 3: 1}, n_bg=2, tpn=1)
    for kwargs in ({}, {"domains": [40, 16]}):
        vocab = 56 if kwargs else 12
        with pytest.raises(NotImplementedError, match="optimize_eta"):
            DagGatedOnlineLDA(lay, vocab_size=vocab, optimize_eta=True, **kwargs)


# --- Production-path optimizer equivalence -----------------------------------


@pytest.mark.slow
def test_vi_runner_full_batch_matches_local_variational_em(spark):
    """PRODUCTION-PATH gate: VIRunner's full-batch fit must reach the same ELBO as
    `fit_gated_svi_local`, the in-memory lr=1.0 variational-EM reference that every
    SVI-vs-Gibbs gate in this file is written against.

    Why this exists. A real defect lived in the production full-batch path for its
    whole life (insight 0074): VIRunner applied the decaying Robbins-Monro step
    rho_t = (tau0 + t + 1)^-kappa UNCONDITIONALLY, so full batches were damped toward
    a DETERMINISTIC target -- the parameters froze short of the fixed point and the
    early stop fired on the vanishing step. Every equivalence gate here validated
    `fit_gated_svi_local` at lr=1.0 and none of them touched VIRunner, so the two
    paths ran different optimizers and nothing noticed. Fixing that instance without
    closing the gap would leave the next one free to happen; this is the gap.

    Not exact parameter equality, deliberately: `local_update` draws its gamma init
    from the global numpy RNG, so partitioning the corpus changes the draw sequence.
    The ELBO is the right currency -- it is what the optimizer optimizes.

    DISCRIMINATION, measured by restoring the unconditional decaying rho and
    re-running this exact setup:

        rho = 1 (fixed):  production converges at iteration 6, ELBO -219689.68,
                          gap to the reference  0.00 nats
        damped (the bug): production runs all 40 iterations WITHOUT converging,
                          ELBO -219895.52 and still climbing ~10 nats/iter,
                          gap to the reference  -205.84 nats

    so the 20-nat tolerance below sits ~10x inside the regression and ~infinitely
    outside the correct case.

    Note what does NOT discriminate, so nobody mistakes it for the gate: placement
    metrics come out IDENTICAL under both schedules on this corpus (mrr 0.841,
    top2 0.809), because a 206-nat ELBO deficit does not move node placement here.
    An earlier draft of this test asserted only placement and would have passed with
    the bug in place. They are kept below as a cheap floor against a grosser
    production-path break, but the ELBO assertion is what closes the gap.
    """
    import numpy as np
    from tests._stm_synth import (dag_placement_corpus, fit_gated_svi_local,
                                  placement_metrics, svi_node_profiles)
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.dag import DagLayout
    from spark_vi.models.topic.dag_gated_lda import DagGatedOnlineLDA
    from spark_vi.models.topic.types import GatedBOWDocument

    parent = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
    V, doc_len, n_train, n_iter = 120, 60, 900, 40
    docs, labels, _ = dag_placement_corpus(
        parent=parent, node_prev={u: 1 for u in range(1, 7)},
        V=V, doc_len=doc_len, seed=1)
    lay = DagLayout(parent, n_bg=2, tpn=1)
    tr_d, tr_l = docs[:n_train], labels[:n_train]
    te_d, te_l = docs[n_train:], labels[n_train:]
    bow = [GatedBOWDocument(*_bow(d), frontier=frozenset({int(y)}))
           for d, y in zip(tr_d, tr_l)]

    # Reference arm: the validated in-memory lr=1.0 variational EM.
    m_ref = DagGatedOnlineLDA(lay, V, alpha=0.1, eta=0.02, random_seed=0)
    gp_ref = fit_gated_svi_local(m_ref, bow, n_iter=n_iter, seed=0)
    ev_ref = placement_metrics(svi_node_profiles(m_ref, gp_ref, te_d, lay), te_l, lay)

    # Production arm: the same model, same budget, through VIRunner's full-batch
    # path (mini_batch_fraction=None).
    np.random.seed(0)
    m_prod = DagGatedOnlineLDA(lay, V, alpha=0.1, eta=0.02, random_seed=0)
    rdd = spark.sparkContext.parallelize(bow, 4).persist()
    rdd.count()                     # materialize for VIRunner's cache precondition
    res = VIRunner(m_prod, VIConfig(max_iterations=n_iter, convergence_tol=1e-12,
                                    mini_batch_fraction=None, random_seed=0)).fit(rdd)
    # THE GATE, asserted first so it is what reports on a regression: same ELBO. The
    # reference keeps no ELBO trace, so score its final parameters with the same
    # full-batch statistics the production arm used.
    elbo_ref = float(m_ref.compute_elbo(gp_ref, m_ref.local_update(bow, gp_ref)))
    assert res.elbo_trace[-1] >= elbo_ref - 20.0, (
        f"production ELBO {res.elbo_trace[-1]:.2f} vs reference {elbo_ref:.2f} "
        f"(gap {res.elbo_trace[-1] - elbo_ref:+.2f} nats); the damped-rho "
        f"regression shows up here as ~-206 nats")

    # Corollary of a correct step size: it finishes well inside the budget, because
    # batch VB with rho=1 reaches its fixed point fast (measured: iteration 6 of 40,
    # last relative ELBO change ~1e-13). The damped schedule cannot converge at all
    # here -- its step vanishes faster than it closes the residual.
    assert res.converged is True, (res.n_iterations, res.elbo_trace[-3:])

    ev_prod = placement_metrics(svi_node_profiles(m_prod, res.global_params, te_d, lay),
                       te_l, lay)

    # Secondary floor only -- see the docstring: these are NOT sensitive to the
    # regression this test exists for.
    assert ev_prod["mrr"] >= ev_ref["mrr"] - 0.05, (ev_prod["mrr"], ev_ref["mrr"])
    assert ev_prod["top2"] >= ev_ref["top2"] - 0.05, (ev_prod["top2"], ev_ref["top2"])
    for dep, auc_ref in ev_ref["auc_by_depth"].items():
        assert ev_prod["auc_by_depth"][dep] >= auc_ref - 0.05, (
            dep, ev_prod["auc_by_depth"][dep], auc_ref)
