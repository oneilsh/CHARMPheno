from charmpheno.omop.condition_dag import ConditionDag, build_condition_dag, prune_by_attestation, pruning_ledger

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

def test_prune_drops_low_count_and_rewires():
    dag = build_condition_dag(DIAMOND_EDGES, ANCHOR, DIAMOND_NODES)
    counts = {101: 50, 102: 40, 103: 0, 104: 20}     # 103 is below threshold
    pruned = prune_by_attestation(dag, counts, min_n=5)
    assert 103 not in pruned.nodes()                  # dropped
    assert pruned.nodes() == {100, 101, 102, 104}
    assert set(pruned.parents[104]) == {101, 102}     # 104 rewired past dropped 103 to its parents

def test_prune_never_drops_anchor():
    dag = build_condition_dag(DIAMOND_EDGES, ANCHOR, DIAMOND_NODES)
    pruned = prune_by_attestation(dag, counts={}, min_n=999)   # everything below threshold
    assert pruned.nodes() == {ANCHOR}                 # only the anchor survives

def test_ledger_counts_and_coarsening():
    dag = build_condition_dag(DIAMOND_EDGES, ANCHOR, DIAMOND_NODES)
    counts = {101: 50, 102: 40, 103: 0, 104: 20}
    pruned = prune_by_attestation(dag, counts, min_n=5)
    # 2 of 4 patients had their most-specific node (103) pruned -> coarsened
    led = pruning_ledger(dag, pruned, counts,
                         cohort_frontiers=[{103}, {104}, {101}, {103}])
    assert led["K_nodes"] == 4 and led["dropped"] == 1
    assert led["dropped_by_depth"] == {2: 1}           # 103 was at depth 2
    assert abs(led["coarsening_rate"] - 0.5) < 1e-9    # 2 of 4 patients coarsened
    # dropped 103 (depth 2) rewires to its nearest surviving ancestors 101/102 (depth 1), so the
    # true depth drop is 1 -- NOT 2 (it does not fall all the way back to the anchor).
    assert abs(led["mean_depth_drop"] - 1.0) < 1e-9

def test_to_engine_maps_anchor_to_zero_and_loads_into_daglayout():
    from spark_vi.models.topic.dag_placement import DagLayout
    dag = build_condition_dag(DIAMOND_EDGES, ANCHOR, DIAMOND_NODES)
    parent_int, int2cid, cid2int = dag.to_engine()
    assert cid2int[ANCHOR] == 0                          # anchor -> root 0
    assert int2cid[cid2int[103]] == 103                  # round-trips
    lay = DagLayout(parent_int, n_bg=2, tpn=1)
    assert lay.K == 2 + 4                                # 4 non-root nodes (101,102,103,104) + 2 bg
    assert 0 in lay.closure(cid2int[104])                # every node's closure reaches the root

def test_real_diabetes_subtree_structure():
    import csv
    from pathlib import Path
    from spark_vi.models.topic.dag_placement import DagLayout
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
    parent_int, int2cid, _ = dag.to_engine()
    lay = DagLayout(parent_int, n_bg=2, tpn=1)
    assert lay.K == 2 + 126                              # 126 non-root nodes + 2 background
