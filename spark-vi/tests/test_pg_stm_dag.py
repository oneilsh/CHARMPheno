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
