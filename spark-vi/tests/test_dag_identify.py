import numpy as np
from spark_vi.models.topic.pg_stm_dag import DagGate
from spark_vi.models.topic.dag_identify import closure_gram


def test_closure_gram_matches_hand_computation():
    """Deterministic linear-algebra check; no empirical or transfer claim. On a 3-node
    chain DAG and a hand-built document set, the pooled closure Gram equals the
    hand-computed sum of outer products of the non-root closure indicators."""
    dag = DagGate([(), (0,), (1,)])           # root 0; node 1 child of root; node 2 child of 1
    # doc at node 1 -> closure {0,1} -> z=[1,0]; doc at node 2 -> closure {0,1,2} -> z=[1,1]
    doc_nodes = [frozenset({1}), frozenset({2}), frozenset({2})]
    G = closure_gram(dag, doc_nodes)
    # outer([1,0]) + 2*outer([1,1]) = [[1,0],[0,0]] + 2*[[1,1],[1,1]]
    assert G.shape == (2, 2)
    assert np.allclose(G, np.array([[3.0, 2.0], [2.0, 2.0]]))
