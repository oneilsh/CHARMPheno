import numpy as np


def test_children_map_and_subtree_from_parent_int():
    from multidomain_lr_readout import children_map, subtree_nodes
    # forest: 100 -> {200, 300}; 201 is MULTI-PARENT (child of both 200 and 300)
    # -- parent_int is always list-valued in production (ConditionDag.to_engine
    # emits {child: [parents]} even for single-parent nodes).
    parent_int = {200: [100], 300: [100], 201: [200, 300]}
    cmap = children_map(parent_int)
    assert cmap[100] == {200, 300}
    assert cmap[200] == {201}
    assert cmap[300] == {201}
    # subtree(100) = 100 + all descendants; subtree(200)/(300) both reach 201
    # through their respective edge to the multi-parent node.
    assert subtree_nodes(parent_int, 100) == {100, 200, 300, 201}
    assert subtree_nodes(parent_int, 200) == {200, 201}
    assert subtree_nodes(parent_int, 300) == {300, 201}


def test_save_load_test_set_roundtrip(tmp_path):
    # Drive BOTH halves of the driver<->readout contract through the SHARED
    # save_test_set/load_test_set pair (the driver writes via save_test_set), so a
    # writer-side filename/key rename can't pass unnoticed. 2 domains (V0=4, V1=3),
    # 2 docs; aff_frontiers deliberately differ from frontiers to prove they stay
    # separate (the theta-baseline alignment guarantee).
    from scipy import sparse as sp
    from multidomain_lr_readout import save_test_set, load_test_set
    bows_in = {0: sp.csr_matrix(np.array([[1.0, 0, 2, 0], [0, 0, 0, 3]])),
               1: sp.csr_matrix(np.array([[0.0, 1, 0], [1, 0, 1]]))}
    save_test_set(tmp_path, bows_in, np.zeros((2, 5)),
                  frontiers=[[5], []], aff_frontiers=[[7], []])
    bows, frontiers, aff, aff_frontiers, n = load_test_set(tmp_path, 2)
    assert set(bows) == {0, 1}
    assert bows[0].shape == (2, 4) and bows[1].shape == (2, 3)
    assert bows[0][0, 2] == 2.0 and bows[1][1, 0] == 1.0   # values landed
    assert frontiers == [[5], []] and aff_frontiers == [[7], []]  # kept separate
    assert aff.shape == (2, 5) and n == 2


def test_load_test_set_missing_raises(tmp_path):
    import pytest
    from multidomain_lr_readout import load_test_set
    with pytest.raises(SystemExit):        # no test_meta.json -> clear SystemExit
        load_test_set(tmp_path, 2)


def test_per_disease_auc_row_uses_max_over_subtree():
    # Node d=1 has child 3 (both scoreable). A doc positive for d if frontier hits
    # {1,3}. The per-disease score is max over subtree(d) columns. Give doc0 a high
    # score at node 3 (subtype) and frontier {3}: it must count as a positive for d
    # and the max-over-subtree must pick up node 3.
    from spark_vi.models.topic.dag_placement import DagLayout
    from multidomain_lr_readout import per_disease_auc_row
    lay = DagLayout({1: 0, 2: 0, 3: 1}, n_bg=1, tpn=1)     # nodes: 1,2,3 ; 3 child of 1
    # list-valued, matching production (ConditionDag.to_engine always emits a
    # parents LIST, even for single-parent nodes).
    parent_int = {1: [0], 2: [0], 3: [1]}
    # scores [n_docs x n_nodes] aligned to lay.nodes; make node 3 high for doc0.
    n3 = lay.nodes.index(3)
    scores = np.zeros((4, len(lay.nodes)))
    scores[0, n3] = 10.0                                   # doc0 strong at subtype 3
    frontiers = [[3], [], [], []]                          # only doc0 has disease d=1 (via 3)
    auc, n_pos = per_disease_auc_row(scores, frontiers, anchor=1, lay=lay,
                                     parent_int=parent_int)
    assert n_pos == 1
    assert auc == 1.0                                      # doc0 ranks top -> perfect


def test_load_lambda_dict_reads_sidecars(tmp_path):
    # The multidomain driver clobbers save_result's manifest, so the readout must
    # load the lambda .npy sidecars directly, keyed by their integer domain suffix
    # (and ignore the alpha sidecar).
    from multidomain_lr_readout import load_lambda_dict
    params = tmp_path / "params"
    params.mkdir()
    np.save(params / "lambda_0.npy", np.ones((4, 6)))
    np.save(params / "lambda_1.npy", np.ones((4, 3)))
    np.save(params / "alpha.npy", np.ones(4))            # not lambda_<m> -> ignored
    lam = load_lambda_dict(tmp_path)
    assert set(lam) == {0, 1}
    assert lam[0].shape == (4, 6) and lam[1].shape == (4, 3)


def test_load_lambda_dict_missing_raises(tmp_path):
    import pytest
    from multidomain_lr_readout import load_lambda_dict
    (tmp_path / "params").mkdir()
    with pytest.raises(SystemExit):
        load_lambda_dict(tmp_path)


def test_build_parser_defaults():
    from multidomain_lr_readout import build_parser
    args = build_parser().parse_args(["--run-dir", "/runs/0071-x"])
    assert args.run_dir == "/runs/0071-x"
    assert args.alpha_grid == "0,1,10,100,inf"             # default sweep


def _pr_fixture():
    """A 1-node layout (node 1, no children) + scores/frontiers helper: node 1 is
    the anchor, so subtree(1) == {1} and the per-disease score is that column."""
    from spark_vi.models.topic.dag_placement import DagLayout
    lay = DagLayout({1: [0]}, n_bg=1, tpn=1)
    parent_int = {1: [0]}
    return lay, parent_int


def test_per_disease_pr_perfect_ranker():
    from multidomain_lr_readout import per_disease_pr
    lay, parent_int = _pr_fixture()
    col = lay.nodes.index(1)
    # 2 positives ranked strictly above 6 negatives -> perfect PR.
    scores = np.zeros((8, len(lay.nodes)))
    scores[:, col] = [9.0, 8.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    frontiers = [[1], [1], [], [], [], [], [], []]
    pr_auc, prec_at, n_pos = per_disease_pr(scores, frontiers, 1, lay, parent_int,
                                           recalls=(0.5, 0.8))
    assert n_pos == 2
    assert pr_auc == 1.0
    assert prec_at[0.5] == 1.0 and prec_at[0.8] == 1.0


def test_per_disease_pr_constant_scorer_is_exactly_prevalence_any_arrangement():
    # A constant (zero-information) scorer must yield AP == prevalence EXACTLY and
    # must not depend on WHICH rows are the positives (ties collapse to one
    # threshold). This is the regression guard for the row-order tie bug.
    from multidomain_lr_readout import per_disease_pr
    lay, parent_int = _pr_fixture()
    col = lay.nodes.index(1)
    n, k = 100, 20
    scores = np.zeros((n, len(lay.nodes)))
    scores[:, col] = 1.0                      # all tied
    for positives in ([i for i in range(n) if i % 5 == 0],   # spread
                      list(range(k)),                        # first rows
                      list(range(n - k, n))):                # last rows
        frontiers = [[1] if i in set(positives) else [] for i in range(n)]
        pr_auc, prec_at, n_pos = per_disease_pr(scores, frontiers, 1, lay, parent_int)
        assert n_pos == k
        assert abs(pr_auc - k / n) < 1e-9              # EXACTLY prevalence
        assert abs(prec_at[0.5] - k / n) < 1e-9
        assert abs(prec_at[0.8] - k / n) < 1e-9


def test_per_disease_pr_uses_max_over_subtree():
    from spark_vi.models.topic.dag_placement import DagLayout
    from multidomain_lr_readout import per_disease_pr
    lay = DagLayout({1: 0, 2: 0, 3: 1}, n_bg=1, tpn=1)   # node 3 is a child of 1
    parent_int = {1: [0], 2: [0], 3: [1]}
    n3 = lay.nodes.index(3)
    scores = np.zeros((4, len(lay.nodes)))
    scores[0, n3] = 10.0                                 # doc0 strong at SUBTYPE 3
    frontiers = [[3], [], [], []]                        # doc0 positive for anchor 1 via 3
    pr_auc, prec_at, n_pos = per_disease_pr(scores, frontiers, 1, lay, parent_int)
    assert n_pos == 1                                    # subtree positive counted
    assert pr_auc == 1.0                                 # max-over-subtree ranks it top
    assert prec_at[0.5] == 1.0


def test_per_disease_pr_reads_the_right_operating_point():
    from multidomain_lr_readout import per_disease_pr
    lay, parent_int = _pr_fixture()
    col = lay.nodes.index(1)
    # Ranked order: P N P N  (2 positives). Cumulative precision/recall:
    #   rank1 P: tp=1 -> prec 1/1=1.00, rec 0.5
    #   rank2 N: tp=1 -> prec 1/2=0.50, rec 0.5
    #   rank3 P: tp=2 -> prec 2/3=0.667, rec 1.0
    # precision@0.5 = 1.00 (first index reaching rec>=0.5)
    # precision@1.0 = 0.667 ; precision@1.5 unreachable -> nan
    scores = np.zeros((4, len(lay.nodes)))
    scores[:, col] = [4.0, 3.0, 2.0, 1.0]
    frontiers = [[1], [], [1], []]
    pr_auc, prec_at, n_pos = per_disease_pr(scores, frontiers, 1, lay, parent_int,
                                           recalls=(0.5, 1.0, 1.5))
    assert n_pos == 2
    assert prec_at[0.5] == 1.0
    assert abs(prec_at[1.0] - 2.0 / 3.0) < 1e-9
    assert np.isnan(prec_at[1.5])              # unreachable recall
    # AP = sum over positives of precision at that positive = (1.0 + 0.667)/2
    assert abs(pr_auc - (1.0 + 2.0 / 3.0) / 2.0) < 1e-9


def test_per_disease_pr_one_class_is_nan():
    from multidomain_lr_readout import per_disease_pr
    lay, parent_int = _pr_fixture()
    scores = np.zeros((4, len(lay.nodes)))
    pr_auc, prec_at, n_pos = per_disease_pr(scores, [[], [], [], []], 1, lay,
                                           parent_int)
    assert n_pos == 0
    assert np.isnan(pr_auc) and np.isnan(prec_at[0.5])
