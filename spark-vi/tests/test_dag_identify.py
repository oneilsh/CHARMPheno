import numpy as np
from spark_vi.models.topic.pg_stm_dag import DagGate
from spark_vi.models.topic.partition import TopicBlockPartition
from spark_vi.models.topic.dag_identify import closure_gram, foreground_grams, identifiability_spectrum


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


def test_foreground_gram_exposes_anchor_level_vs_intercept_collinearity():
    """Deterministic linear-algebra check; no empirical or transfer claim. Within a group
    whose documents all attest its anchor, the intercept column equals that anchor's
    closure-indicator column, so the per-group foreground Gram is rank-deficient along the
    level-vs-intercept direction (a zero eigenvalue) -- the per-node absolute-level design
    wall of insight 0054. Proves the foreground Gram surfaces the wall; does NOT prove
    anything about recovery or real data."""
    part = TopicBlockPartition(group_var="g", background_k=2, foreground=(("A", 2),))
    dag = DagGate([(), (0,)])                  # root 0; node 1 = anchor A
    # every group-A doc attests the anchor (node 1) -> z = [1]; w = [intercept=1, z=1]
    doc_nodes = [frozenset({1}), frozenset({1}), frozenset({1})]
    doc_groups = ["A", "A", "A"]
    grams = foreground_grams(dag, doc_nodes, doc_groups, part)
    A = grams["A"]
    assert A.shape == (2, 2)                    # [intercept, node1]
    # both columns are all-ones over the 3 docs -> A = 3 * ones((2,2)) -> rank 1
    assert np.allclose(A, 3.0 * np.ones((2, 2)))
    evals = np.linalg.eigvalsh(A)
    assert np.isclose(evals.min(), 0.0)        # level-vs-intercept null direction


def test_spectrum_is_raw_and_ascending_and_flags_exact_confound_as_zero():
    """Deterministic linear-algebra check; no empirical or transfer claim. The spectrum is
    the raw ascending eigendecomposition with no threshold: a full-rank Gram has all
    positive eigenvalues, and a Gram with two identical columns has an exact zero
    eigenvalue whose eigenvector is the difference direction. Proves the kernel is
    threshold-free; asserts no tier or collapse."""
    G_full = np.array([[3.0, 2.0], [2.0, 2.0]])
    sp = identifiability_spectrum(G_full)
    assert np.all(np.diff(sp["eigenvalues"]) >= -1e-12)          # ascending
    assert sp["eigenvalues"].min() > 1e-9                        # full rank
    # two identical columns (z_a == z_b) -> exact null direction e_a - e_b
    G_conf = np.array([[4.0, 4.0], [4.0, 4.0]])
    sp2 = identifiability_spectrum(G_conf)
    assert np.isclose(sp2["eigenvalues"][0], 0.0)
    v = sp2["eigenvectors"][:, 0]
    assert np.isclose(abs(v[0]), abs(v[1]))                      # supported on {a,b} equally
