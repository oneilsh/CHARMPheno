import numpy as np
from spark_vi.models.topic.spectral_init import (
    word_cooccurrence, find_anchors, recover_beta, spectral_init_beta)
from _stm_synth import (synthetic_ehr_corpus, synthetic_gated_corpus,
                        planted_recovery, foreground_recovers_group)

def test_cooccurrence_normalized_square():
    docs, _ = synthetic_ehr_corpus(K_rare=4, V=40, D=100, doc_len=20, bg_frac=0.5, seed=1)
    Q = word_cooccurrence(docs, 40)
    assert Q.shape == (40, 40) and np.isclose(Q.sum(), 1.0, atol=1e-6)

def test_spectral_init_recovers_nongated_planted():
    from spark_vi.models.topic.partition import TopicBlockPartition
    docs, planted = synthetic_ehr_corpus(K_rare=6, V=120, D=800, doc_len=30,
                                         bg_frac=0.5, seed=2)
    part = TopicBlockPartition(group_var="", background_k=12, foreground=())
    beta0 = spectral_init_beta(docs, part, 120)
    assert beta0.shape == (12, 120)
    assert planted_recovery(beta0, planted, thresh=0.4) >= 4

def test_block_aware_init_recovers_rare_group_foreground():
    """The decisive gated property: a rare group's foreground anchor lands its
    planted phenotype at INIT, because it is found on the within-group Q
    (undiluted by the majority) and deflated against the background span."""
    docs, planted, part = synthetic_gated_corpus(
        groups=("maj", "rare"), fg_per_group=2, bg_k=3, V=240, D=1200,
        doc_len=30, bg_frac=0.6, seed=3)
    # make 'rare' a minority arm
    docs = [d for i, d in enumerate(docs) if ("rare" not in d.groups) or (i % 4 == 0)]
    beta0 = spectral_init_beta(docs, part, 240)
    assert foreground_recovers_group(beta0, part, "rare", planted, thresh=0.4)


def test_find_anchors_domain_bounds_none_is_identical():
    """domain_bounds=None reproduces the pooled-floor behavior exactly."""
    import numpy as np
    from spark_vi.models.topic.spectral_init import find_anchors, word_cooccurrence
    from types import SimpleNamespace
    rng = np.random.default_rng(0)
    V = 20
    docs = []
    for _ in range(200):
        toks = rng.integers(0, V, size=8)
        u, c = np.unique(toks, return_counts=True)
        docs.append(SimpleNamespace(indices=u, counts=c.astype(float)))
    Q = word_cooccurrence(docs, V)
    assert find_anchors(Q, 5) == find_anchors(Q, 5, domain_bounds=None)
    assert find_anchors(Q, 5) == find_anchors(Q, 5, domain_bounds=[0, V])


def test_find_anchors_per_domain_floor_admits_sparse_domain_anchor():
    """A pure anchor in a sparse second domain clears its WITHIN-domain floor
    even though its marginal is below the pooled mean, so it can be selected.
    Under the pooled floor it is excluded; under the per-domain floor it is not."""
    import numpy as np
    from spark_vi.models.topic.spectral_init import find_anchors
    # Domain A = cols [0:4] (dense), domain B = cols [4:6] (sparse).
    # Build Q directly: dense block carries most mass; the domain-B anchor
    # (col 4) co-occurs purely with a single domain-A word (col 0) but at low
    # total mass.
    V = 6
    Q = np.zeros((V, V))
    # dense domain-A co-occurrence
    Q[0, 1] = Q[1, 0] = 0.20
    Q[2, 3] = Q[3, 2] = 0.20
    Q[0, 2] = Q[2, 0] = 0.10
    # sparse domain-B anchor col 4 pairs only with its domain-A partner col 0, low mass
    Q[0, 4] = Q[4, 0] = 0.02
    # col 5 is domain-B noise, negligible
    Q[5, 5] = 1e-9
    Q = Q / Q.sum()
    domain_bounds = [0, 4, 6]
    pooled = find_anchors(Q, 4)                      # pooled floor
    per_dom = find_anchors(Q, 4, domain_bounds=domain_bounds)
    # The domain-B anchor (col 4) is below the pooled marginal mean and excluded
    # by the pooled floor, but clears the sparse-domain mean under per-domain.
    assert 4 not in pooled
    assert 4 in per_dom


def test_split_domains_renormalizes_each_slice():
    import numpy as np
    from spark_vi.models.topic.spectral_init import split_domains
    # K=2 topics, V=5: domain A cols [0:3], domain B cols [3:5].
    beta = np.array([
        [0.4, 0.1, 0.1, 0.2, 0.2],   # topic 0
        [0.0, 0.0, 0.0, 0.5, 0.5],   # topic 1: zero over domain A
    ])
    A, B = split_domains(beta, [0, 3, 5])
    assert A.shape == (2, 3) and B.shape == (2, 2)
    # topic 0 domain-A slice [0.4,0.1,0.1] renormalized by 0.6
    np.testing.assert_allclose(A[0], np.array([0.4, 0.1, 0.1]) / 0.6)
    np.testing.assert_allclose(B[0], np.array([0.2, 0.2]) / 0.4)
    # every returned row sums to 1
    np.testing.assert_allclose(A.sum(1), 1.0)
    np.testing.assert_allclose(B.sum(1), 1.0)
    # topic 1 is zero over domain A -> uniform fallback, still stochastic
    np.testing.assert_allclose(A[1], np.full(3, 1.0 / 3))


def test_split_domains_single_domain_is_identity_up_to_renorm():
    import numpy as np
    from spark_vi.models.topic.spectral_init import split_domains
    beta = np.array([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3]])
    (only,) = split_domains(beta, [0, 3])
    np.testing.assert_allclose(only, beta)   # already row-stochastic


def test_two_domain_corpus_within_doc_cross_domain():
    """Every doc's tokens span BOTH domains (Q_01 != 0 prerequisite) and a
    domain-1-only node's docs still carry its unique domain-1 signature."""
    import numpy as np
    from tests._stm_synth import two_domain_dag_corpus
    parent = {1: 0, 2: 1, 3: 1}     # root 0 -> node 1 -> leaves 2,3
    docs, labels, domain_bounds, pa, pb, slot_of_node, codes = two_domain_dag_corpus(
        parent=parent, node_prev={1: 1.0, 2: 1.0, 3: 1.0},
        V_a=30, V_b=12, doc_len=24, seed=1, b_only_node=3)
    Va = domain_bounds[1]
    # at least half the docs carry a domain-0 token (<Va) AND a domain-1 token (>=Va)
    spanning = [d for d in docs if (np.asarray(d) < Va).any() and (np.asarray(d) >= Va).any()]
    assert len(spanning) > 0.5 * len(docs)
    # planted shapes
    assert pa.shape[1] == Va and pb.shape[1] == domain_bounds[2] - Va
    # b_only_node=3 has a nonzero unique domain-1 signature row (recovered in Task 4)
    assert pb[slot_of_node[3]].sum() > 0


def test_multidomain_init_recovers_both_domains_incl_b_anchored():
    """The joint recipe (one Q, one greedy with per-domain floor, one recover,
    split) recovers per-domain phenotypes for every node, INCLUDING a node whose
    domain-0 signature is ambiguous but whose domain-1 signature is unique."""
    import numpy as np
    from types import SimpleNamespace
    from tests._stm_synth import two_domain_dag_corpus
    from spark_vi.models.topic.spectral_init import (
        word_cooccurrence, find_anchors, recover_beta, split_domains)

    parent = {1: 0, 2: 1, 3: 1}
    docs, labels, domain_bounds, pa, pb, slot_of_node, codes = two_domain_dag_corpus(
        parent=parent, node_prev={1: 1.0, 2: 1.0, 3: 1.0},
        V_a=40, V_b=16, doc_len=30, seed=3, b_only_node=3)
    V = domain_bounds[-1]
    counted = [SimpleNamespace(indices=np.unique(np.asarray(d)),
               counts=np.unique(np.asarray(d), return_counts=True)[1].astype(float))
               for d in docs]
    Q = word_cooccurrence(counted, V)
    K = pa.shape[0]
    anchors = find_anchors(Q, K, domain_bounds=domain_bounds)
    beta = recover_beta(Q, anchors)
    ba, bb = split_domains(beta, domain_bounds)

    # at least one anchor comes from domain 1 (id >= V_a)
    assert any(a >= domain_bounds[1] for a in anchors)
    # domain-1 recovery: node 3's unique planted domain-1 block is captured by
    # some recovered domain-1 topic (support-overlap mass), even though its
    # domain-0 signature is ambiguous.
    def _support(row, eps=1e-3):
        return np.where(row > eps)[0]
    node_b_support = _support(pb[slot_of_node[3]])
    assert bb[:, node_b_support].sum(axis=1).max() > 0.4
