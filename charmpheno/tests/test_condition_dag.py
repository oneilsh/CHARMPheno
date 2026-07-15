from charmpheno.omop.condition_dag import ConditionDag, build_condition_dag

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
