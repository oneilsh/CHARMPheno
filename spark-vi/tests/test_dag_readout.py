import numpy as np
from spark_vi.models.topic.dag_readout import assemble_readout, node_prevalence
from spark_vi.models.topic.pg_stm_dag import DagGate


class DummyDag:
    """Minimal stand-in for DagGate exposing only what assemble_readout needs:
    n_nodes and a parents list of parent-tuples (node 0 = root, no parents)."""

    def __init__(self, n_nodes):
        self.n_nodes = n_nodes
        self.parents = [()] + [(0,) for _ in range(n_nodes - 1)]


def test_readout_is_a_fixed_keyset_with_the_four_statuses():
    U, Ksm1, n_kept = 4, 5, 200                          # nodes 1..4
    node_map = np.array([0, 1, 2, 3, 2])                 # node 4 merged into node 2 (rep)
    rng = np.random.default_rng(0)
    draws = rng.standard_normal((n_kept, U, Ksm1)) * 0.1 + 1.0
    classification = {"gauge_nodes": frozenset({1, 2}),
                      "unresolved": {4: {"attest_node": 2, "docs_needed": 250}}}
    ro = assemble_readout(DummyDag(n_nodes=5), draws, node_map, classification, ci_level=0.90)

    assert ro["calibration"] == "absolute"
    assert set(ro["coordinates"]) == {1, 2, 3, 4}        # fixed key set = all non-root nodes
    assert ro["coordinates"][1]["status"] == "gauge" and "increment_mean" not in ro["coordinates"][1]
    assert ro["coordinates"][4]["status"] == "unresolved"
    assert ro["coordinates"][4]["recipe"]["docs_needed"] == 250 and "increment_mean" not in ro["coordinates"][4]
    assert ro["coordinates"][3]["status"] == "identified"
    c3 = ro["coordinates"][3]
    # increment_mean/ci_low/ci_high are per-stick vectors (length Ksm1); compare elementwise
    # since numpy disallows chained `<` on multi-element arrays (ambiguous truth value).
    assert (c3["ci_low"] < c3["increment_mean"]).all()
    assert (c3["increment_mean"] < c3["ci_high"]).all()


def test_prevalence_adds_soft_membership_to_labeled_mass():
    # Convention: labeled_mass counts HARD-attested docs only (a doc with a single
    # candidate). inferred_total = labeled_mass + the soft/partial-label membership
    # mass landing on candidates whose closure contains the node. recall_ratio =
    # labeled_mass / inferred_total <= 1.0: it measures how much the hard labels
    # UNDERCOUNT relative to the soft-inferred total (not the other way around).
    dag = DagGate([(), (0,), (0,), (1,), (1,)])          # A=1 with leaves A1=3, A2=4
    doc_nodes = [frozenset({3})] * 10                     # 10 hard A1 docs
    doc_candidates = [[(1.0, frozenset({3}))]] * 10
    memberships = [np.array([1.0])] * 10
    # add 4 partial-label docs under A, each 0.75 A1 / 0.25 A2
    doc_nodes += [frozenset({3})] * 4
    cand = [(0.75, frozenset({3})), (0.25, frozenset({4}))]
    doc_candidates += [cand] * 4
    memberships += [np.array([0.75, 0.25])] * 4

    prev = node_prevalence(dag, doc_nodes, doc_candidates, memberships)
    assert np.isclose(prev[3]["labeled_mass"], 10)                  # hard-attested A1 docs only
    assert np.isclose(prev[3]["inferred_total"], 10 + 4 * 0.75)     # + soft mass landing on A1
    assert prev[3]["recall_ratio"] < 1.0                            # labeled undercounts inferred
    assert np.isclose(prev[3]["recall_ratio"], 10 / 13.0)
