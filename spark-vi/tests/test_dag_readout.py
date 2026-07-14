import numpy as np
from spark_vi.models.topic.dag_readout import assemble_readout


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
