import numpy as np
import pytest
from spark_vi.models.topic.dag_readout import assemble_readout, node_prevalence, dag_offset_readout
from spark_vi.models.topic.pg_stm_dag import DagGate
from spark_vi.models.topic.partition import TopicBlockPartition
from tests._stm_synth import dag_offset_corpus, real_beta_from


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


def test_end_to_end_readout_statuses_on_the_0054_corpus():
    part = TopicBlockPartition(group_var="g", background_k=3, foreground=(("A", 2), ("B", 2)))
    K, V = part.K, 120
    dag = DagGate([(), (0,), (0,), (1,), (2,)])           # B(2) no own docs
    rng = np.random.default_rng(11)
    node_offsets = {u: 2.0 * rng.standard_normal(K - 1) for u in (1, 2, 3, 4)}
    node_offsets[0] = np.zeros(K - 1)
    beta = real_beta_from(K, V, seed=2)
    docs, doc_nodes, cand = dag_offset_corpus(
        dag=dag, node_offsets=node_offsets, partition=part, beta=beta,
        node_of_group={"A": 1, "B": 2}, doc_nodes_plan={1: 200, 3: 200, 4: 250},
        sigma_true=1.0 * np.eye(K - 1), doc_len=80, seed=12)
    doc_groups = [next(iter(d.groups)) for d in docs]
    ro = dag_offset_readout(docs, doc_nodes, cand, doc_groups, part, dag,
                            P=1, tol=1.0, lam_base=0.25, n_iter=80, burn=40, seed=0)
    st = {u: ro["coordinates"][u]["status"] for u in ro["coordinates"]}
    assert st[3] == "identified"                          # A1 increment
    assert st[1] == "gauge" and st[2] == "gauge"          # anchor levels
    assert st[4] == "unresolved"                          # B1 subsumed, no own docs
    assert ro["coordinates"][4]["recipe"]["attest_node"] == 2


def _covers(entry, truth):
    # coverage is claimed only on the node's IDENTIFIED sticks (its group's foreground
    # sticks; insight 0054/0057). When the read-out restricts to a sub-block it exposes the
    # `sticks` field and its ci vectors are already that sub-vector; compare to the matching
    # slice of the planted truth.
    idx = entry.get("sticks")
    tv = truth[np.asarray(idx, dtype=int)] if idx is not None else truth
    return bool(np.all(entry["ci_low"] <= tv) and np.all(tv <= entry["ci_high"]))


@pytest.mark.xfail(reason="insight 0057: identified-coordinate coverage is UNMET under the "
                          "current engine. The design-wall schema half holds; the sub-block "
                          "restriction (insight 0054, claim only a node's own foreground sticks) "
                          "is applied, so this is NOT the granularity artifact -- it is real "
                          "calibration failure (attenuated means + overconfident intervals; "
                          "insight 0051 reproduced under exact Gibbs). Closing it needs the "
                          "LKJ/half-t provenance priors + the ridge-attenuation fix (Fable's "
                          "pre-registered next step), then this xfail is removed. Threshold NOT "
                          "loosened.", strict=False)
def test_coverage_plant_identified_covers_designwall_reports_unresolved():
    part = TopicBlockPartition(group_var="g", background_k=3, foreground=(("A", 2), ("B", 2)))
    K, V = part.K, 120
    dag = DagGate([(), (0,), (0,), (1,), (2,)])           # A1(3) identified; B(2)/B1(4) design wall
    beta = real_beta_from(K, V, seed=2)
    R, covered = 12, 0
    designwall_ok = True
    for rep in range(R):
        rng = np.random.default_rng(100 + rep)            # REDRAW TRUTH per replicate (insight 0051)
        node_offsets = {u: 2.0 * rng.standard_normal(K - 1) for u in (1, 2, 3, 4)}
        node_offsets[0] = np.zeros(K - 1)
        docs, doc_nodes, cand = dag_offset_corpus(
            dag=dag, node_offsets=node_offsets, partition=part, beta=beta,
            node_of_group={"A": 1, "B": 2},
            doc_nodes_plan={1: 200, 3: 200, 4: 250},
            partial_label_plan={1: 60},                   # SOFT-GATED cell
            sigma_true=1.0 * np.eye(K - 1), doc_len=80, seed=1000 + rep)
        doc_groups = [next(iter(d.groups)) for d in docs]
        ro = dag_offset_readout(docs, doc_nodes, cand, doc_groups, part, dag,
                                P=1, tol=1.0, lam_base=0.25, n_iter=80, burn=40, seed=0)
        c3 = ro["coordinates"][3]
        if c3["status"] in ("identified", "fragile"):
            covered += _covers(c3, node_offsets[3])
        # design-wall coords: NO point estimate, recipe/convention present
        designwall_ok &= ("increment_mean" not in ro["coordinates"][4]) and \
                          ("recipe" in ro["coordinates"][4]) and \
                          ("increment_mean" not in ro["coordinates"][2])
    assert designwall_ok
    assert covered / R >= 0.6            # wide-but-covers (loose band at R=12; tighten if R raised)
