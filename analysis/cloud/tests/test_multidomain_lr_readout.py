import numpy as np


def test_children_map_and_subtree_from_parent_int():
    from multidomain_lr_readout import children_map, subtree_nodes
    # forest: 100 -> {200 (->201), 300}; 200 has child 201.
    parent_int = {200: 100, 300: 100, 201: 200}
    cmap = children_map(parent_int)
    assert cmap[100] == {200, 300} and cmap[200] == {201}
    # subtree(100) = 100 + all descendants; subtree(200) = {200, 201}
    assert subtree_nodes(parent_int, 100) == {100, 200, 300, 201}
    assert subtree_nodes(parent_int, 200) == {200, 201}
    assert subtree_nodes(parent_int, 300) == {300}       # leaf


def test_build_domain_bows_shapes_and_frontier():
    from multidomain_lr_readout import build_domain_bows
    # two fake collected rows, 2 domains (V0=4, V1=3); features are (indices, values)
    class FakeVec:
        def __init__(self, size, idx, val):
            self.size, self.indices, self.values = size, np.array(idx), np.array(val, dtype=float)
    rows = [
        {"person_id": "a", "features_0": FakeVec(4, [0, 2], [1.0, 3.0]),
         "features_1": FakeVec(3, [1], [2.0]), "frontier": [5]},
        {"person_id": "b", "features_0": FakeVec(4, [], []),
         "features_1": FakeVec(3, [0, 2], [1.0, 1.0]), "frontier": []},
    ]
    bows, frontiers, pids = build_domain_bows(rows, ["features_0", "features_1"], [4, 3])
    assert set(bows) == {0, 1}
    assert bows[0].shape == (2, 4) and bows[1].shape == (2, 3)
    assert bows[0][0, 2] == 3.0 and bows[1][1, 0] == 1.0   # values landed
    assert frontiers == [[5], []] and pids == ["a", "b"]


def test_per_disease_auc_row_uses_max_over_subtree():
    # Node d=1 has child 3 (both scoreable). A doc positive for d if frontier hits
    # {1,3}. The per-disease score is max over subtree(d) columns. Give doc0 a high
    # score at node 3 (subtype) and frontier {3}: it must count as a positive for d
    # and the max-over-subtree must pick up node 3.
    from spark_vi.models.topic.dag_placement import DagLayout
    from multidomain_lr_readout import per_disease_auc_row
    lay = DagLayout({1: 0, 2: 0, 3: 1}, n_bg=1, tpn=1)     # nodes: 1,2,3 ; 3 child of 1
    parent_int = {1: 0, 2: 0, 3: 1}
    # scores [n_docs x n_nodes] aligned to lay.nodes; make node 3 high for doc0.
    n3 = lay.nodes.index(3)
    scores = np.zeros((4, len(lay.nodes)))
    scores[0, n3] = 10.0                                   # doc0 strong at subtype 3
    frontiers = [[3], [], [], []]                          # only doc0 has disease d=1 (via 3)
    auc, n_pos = per_disease_auc_row(scores, frontiers, anchor=1, lay=lay,
                                     parent_int=parent_int)
    assert n_pos == 1
    assert auc == 1.0                                      # doc0 ranks top -> perfect


def test_build_parser_defaults():
    from multidomain_lr_readout import build_parser
    args = build_parser().parse_args(["--run-dir", "/runs/0071-x"])
    assert args.run_dir == "/runs/0071-x"
    assert args.alpha_grid == "0,1,10,100,inf"             # default sweep
