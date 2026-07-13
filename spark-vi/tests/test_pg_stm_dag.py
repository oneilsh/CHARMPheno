import numpy as np
import pytest
from spark_vi.models.topic.pg_stm_dag import DagGate, offset_penalty, dag_offset_ridge


def test_dag_closure_and_indicator_over_two_levels():
    # 0=root; 1,2 anchors under root; 3 = subtype under anchor 1
    dag = DagGate([(), (0,), (0,), (1,)])
    assert dag.n_nodes == 4
    assert dag.closure(frozenset({3})) == frozenset({3, 1, 0})   # subtype -> anchor -> root
    assert dag.closure(frozenset({2})) == frozenset({2, 0})
    z = dag.closure_indicator(frozenset({3}))
    assert z.dtype == np.float64
    assert list(z) == [1.0, 1.0, 0.0, 1.0]                       # nodes 0,1,3 on; 2 off


def test_dag_diamond_shared_ancestor_counted_once():
    # 3 has two parents 1 and 2, both under root 0 (a diamond)
    dag = DagGate([(), (0,), (0,), (1, 2)])
    assert dag.closure(frozenset({3})) == frozenset({3, 1, 2, 0})
    assert dag.closure_indicator(frozenset({3})).sum() == 4      # 0 once, not twice


def test_dag_depth_is_shortest_root_distance():
    dag = DagGate([(), (0,), (1,), (0, 2)])   # node 3 reachable via 0 (d=1) or via 2 (d=3)
    assert list(dag.depth) == [0, 1, 2, 1]    # shortest wins for node 3


def test_dag_rejects_parent_index_not_less_than_child():
    with pytest.raises(ValueError):
        DagGate([(), (2,), (0,)])             # node 1's parent 2 > 1 -> not topo-ordered


def test_dag_dump_lists_nodes_with_depth_and_parents():
    dag = DagGate([(), (0,), (1,)])
    d = dag.dump()
    assert d[2] == {"node": 2, "depth": 2, "parents": [1]}


def test_offset_penalty_is_depth_scaled_on_node_block_only():
    dag = DagGate([(), (0,), (1,)])            # depths 0,1,2
    pen = offset_penalty(P=2, dag=dag, gamma_ridge=1e-6, lam_base=2.0, gamma_depth=1.0)
    assert pen.shape == (2 + 3,)
    assert np.allclose(pen[:2], 1e-6)          # covariates lightly ridged
    assert np.allclose(pen[2:], [2.0 * 1, 2.0 * 2, 2.0 * 3])   # lam_base*(1+depth)


def test_dag_offset_ridge_recovers_well_posed_coefficients():
    rng = np.random.default_rng(0)
    n, d, k = 500, 4, 3
    W = rng.standard_normal((n, d))
    coeff = rng.standard_normal((d, k))
    M = W @ coeff
    got = dag_offset_ridge(W.T @ W, W.T @ M, penalty=np.full(d, 1e-8))
    assert np.allclose(got, coeff, atol=1e-4)


def test_dag_offset_ridge_shrinks_an_unconstrained_column_to_zero():
    # design column 3 is all-zero (a "never-active" node) -> its coeff row must go ~0
    rng = np.random.default_rng(1)
    n, k = 400, 2
    W = rng.standard_normal((n, 4)); W[:, 3] = 0.0
    M = W[:, :3] @ rng.standard_normal((3, k))
    pen = np.array([1e-8, 1e-8, 1e-8, 5.0])
    got = dag_offset_ridge(W.T @ W, W.T @ M, penalty=pen)
    assert np.allclose(got[3], 0.0, atol=1e-9)      # unconstrained + penalized -> exactly ~0


from spark_vi.models.topic.pg_stm import PGSTMVI
from spark_vi.models.topic.pg_stm_dag import PGSTMDag, root_only_dag
from tests._stm_synth import gated_ln_corpus_stick


def test_pgstmdag_root_only_matches_flat_pgstmvi():
    """PLANTED: a stick-native gated corpus. REAL: nothing. Synthetic -> MATH-CORRECTNESS
    only: with a root-only DAG (the offset is a single global intercept, collinear with
    the covariate intercept), PGSTMDag is a reparameterization of PGSTMVI and must return
    the SAME beta and Sigma. Proves the augmented-covariate machinery does not perturb the
    validated flat model. Does NOT prove anything about multi-level DAG behavior."""
    docs, part, _St, _b = gated_ln_corpus_stick(
        group_weights={"A": 0.5, "B": 0.5}, fg_per_group=2, bg_k=3, V=60, D=300,
        doc_len=40, seed=0)
    P = docs[0].x.shape[0]
    vi = PGSTMVI(K=part.K, V=60, partition=part, P=P, n_iter=40, seed=0).fit(docs)
    dag = root_only_dag()
    doc_nodes = [frozenset({0})] * len(docs)                 # every doc attests the root
    out = PGSTMDag(K=part.K, V=60, partition=part, dag=dag, P=P, n_iter=40,
                   gamma_ridge=1e-6, lam_base=1e-6, gamma_depth=1.0, seed=0).fit(docs, doc_nodes)
    assert np.allclose(out["beta"], vi["beta"], atol=2e-3)
    assert np.allclose(out["Sigma"], vi["Sigma"], atol=2e-3)
    assert out["B"].shape == (1, part.K - 1)                 # one node offset row


from spark_vi.models.topic.pg_stm import stick_layout
from tests._stm_synth import dag_offset_corpus, real_beta_from


def test_offset_recovery_through_two_level_closure():
    """PLANTED: node offsets on a root->anchor->subtype DAG + a planted Sigma. REAL: beta
    (realistic overlap, topic_overlap=0.6) and doc-length distribution. Realistic-overlap
    synthetic -> MATH-CORRECTNESS: given a known closure structure, PGSTMDag recovers the
    planted node offsets (subtype offset separated from its anchor's) through a two-level
    closure. Does NOT prove real-data offsets are recoverable, nor transfer."""
    from spark_vi.models.topic.partition import TopicBlockPartition
    part = TopicBlockPartition(group_var="g", background_k=6,
                               foreground=(("A", 4), ("B", 4)))
    K, V = part.K, 300
    beta = real_beta_from(K, V, seed=2)
    # DAG: 0 root; 1,2 anchors (= groups A,B); 3 = subtype under anchor 1
    dag = DagGate([(), (0,), (0,), (1,)])
    Ksm1 = K - 1
    rng = np.random.default_rng(3)
    node_offsets = {0: np.zeros(Ksm1), 1: rng.standard_normal(Ksm1),
                    2: rng.standard_normal(Ksm1), 3: rng.standard_normal(Ksm1)}
    sigma_true = 3.0 * np.eye(Ksm1)
    docs, doc_nodes = dag_offset_corpus(
        dag=dag, node_offsets=node_offsets, partition=part, beta=beta,
        node_of_group={"A": 1, "B": 2}, doc_nodes_plan={1: 400, 2: 400, 3: 400},
        sigma_true=sigma_true, doc_len=80, seed=4)
    out = PGSTMDag(K=K, V=V, partition=part, dag=dag, P=docs[0].x.shape[0],
                   n_iter=60, lam_base=1e-3, gamma_depth=1.0, seed=0).fit(docs, doc_nodes)
    B = out["B"]
    # recovery is up to the root/intercept reparam; compare on the ACTIVE sticks of each
    # node's own block via correlation of recovered vs planted subtype offset.
    lay = stick_layout(part)
    a1 = lay["groups"]["A"]["active"]
    r = np.corrcoef(B[3][a1], node_offsets[3][a1])[0, 1]
    assert r > 0.6, f"subtype offset not recovered through the 2-level closure (r={r:.2f})"
