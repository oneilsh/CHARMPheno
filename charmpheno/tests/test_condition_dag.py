from charmpheno.omop.condition_dag import build_condition_dag

DIAMOND_EDGES = [(100, 101), (100, 102), (101, 103), (102, 103), (103, 104)]
DIAMOND_NODES = {100, 101, 102, 103, 104}
ANCHOR = 100


def test_build_multiparent_and_depth():
    dag = build_condition_dag(DIAMOND_EDGES, ANCHOR, DIAMOND_NODES)
    assert dag.nodes() == DIAMOND_NODES
    assert set(dag.parents[103]) == {101, 102}          # multi-parent
    assert dag.depth(104) == 3 and dag.depth(103) == 2 and dag.depth(100) == 0
    assert sorted(dag.children()[101]) == [103]


def test_build_orphan_attaches_to_anchor():
    # 202 has no in-set parent edge -> should attach to the anchor
    dag = build_condition_dag([(200, 201)], anchor=200, node_ids={200, 201, 202})
    assert dag.parents[202] == [200]
    assert dag.depth(202) == 1
    assert dag.orphans == {202}                          # orphan surfaced for observability
    assert 201 not in dag.orphans                        # a genuine edge child is not an orphan


def test_user_dag_is_the_estimators_contract():
    """`{node: [parents]}` over every node, int concept ids, the real anchor a
    top-level node; it parses as the estimator's DAG (spec 2026-10-30 R1)."""
    from spark_vi.dag.nodeset import ROOT, parse_dag
    dag = build_condition_dag(DIAMOND_EDGES, ANCHOR, DIAMOND_NODES)
    ud = dag.user_dag()
    assert ud == {100: [], 101: [100], 102: [100], 103: [101, 102], 104: [103]}
    parsed = parse_dag(ud)
    assert parsed.parents[100] == (ROOT,)
    assert parsed.parents[103] == (101, 102)


def test_user_dag_drops_a_synthetic_forest_root():
    dag = build_condition_dag([(-1, 100), (-1, 500), (100, 200)], anchor=-1,
                              node_ids={100, 200, 500})
    assert dag.user_dag(synthetic_root=-1) == {100: [], 200: [100], 500: []}
    assert -1 in dag.user_dag()                          # kept when it is a real anchor


def test_real_diabetes_subtree_structure():
    import csv
    from pathlib import Path
    from spark_vi.dag.nodeset import fit_node_set
    path = Path(__file__).parent / "data" / "diabetes_subtree_edges.csv"
    with open(path) as fh:
        rows = list(csv.reader(fh))[1:]                 # skip header
    edges = [(int(a), int(d)) for a, d in rows]
    ANCHOR_DM = 201820                                  # SNOMED "Diabetes mellitus"
    node_ids = {a for a, _ in edges} | {d for _, d in edges} | {ANCHOR_DM}
    dag = build_condition_dag(edges, ANCHOR_DM, node_ids)
    assert len(dag.nodes()) == 127                       # the real diabetes type/status taxonomy
    assert max(dag.depth(n) for n in dag.nodes()) == 4
    multiparent = sum(1 for c, ps in dag.parents.items() if len(ps) > 1)
    assert multiparent == 12                             # real type x status cross-axes
    # the estimator's node-set learning takes it as is: one document per node
    # keeps every node (minDocsPerNode=1), the anchor included (a top-level node)
    ud = dag.user_dag()
    ns = fit_node_set(ud, [[n] for n in ud], min_docs_per_node=1)
    assert len(ns.nodes) + len(ns.spliced) == 127
