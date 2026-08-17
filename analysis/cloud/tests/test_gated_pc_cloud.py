"""Tests for the gated_pc cloud driver: the pure pc_topics_lr scorer, the
DAG-closure-parents densifier, and the arg surface. The end-to-end BQ+fit run is
the cluster smoke (main() reads the CDR via the spark-bigquery connector)."""

import numpy as np


def test_pc_topics_lr_bundle_separable_theta_scores_high():
    from gated_pc_cloud import pc_topics_lr_bundle
    # C=2 nodes; theta dim 0 predicts node 0, dim 1 predicts node 1 (separable).
    Pi_tr = np.array([[1.0, 0.0], [0.9, 0.1], [0.1, 0.9], [0.0, 1.0],
                      [0.8, 0.2], [0.2, 0.8]])
    y_tr = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 1]], float)
    Pi_te = np.array([[0.95, 0.05], [0.05, 0.95], [0.85, 0.15], [0.15, 0.85]])
    y_te = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], float)
    m_tr = np.ones_like(y_tr)
    m_te = np.ones_like(y_te)
    b = pc_topics_lr_bundle(Pi_tr, y_tr, m_tr, Pi_te, y_te, m_te, C=2)
    assert b["macro"]["auc"] == 1.0            # perfectly separable
    assert b["macro"]["n_labels_scored"] == 2


def test_pc_topics_lr_bundle_masks_unobserved_cells():
    from gated_pc_cloud import pc_topics_lr_bundle
    # Node 1 is entirely UNOBSERVED in test (mask all zero) -> skipped from macro.
    Pi_tr = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.1, 0.9]])
    y_tr = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], float)
    Pi_te = np.array([[0.9, 0.1], [0.1, 0.9]])
    y_te = np.array([[1, 0], [0, 1]], float)
    m_tr = np.ones_like(y_tr)
    m_te = np.array([[1, 0], [1, 0]], float)   # node 1 unobserved in test
    b = pc_topics_lr_bundle(Pi_tr, y_tr, m_tr, Pi_te, y_te, m_te, C=2)
    assert b["macro"]["n_labels_scored"] == 1  # only node 0 had observed test cells
    assert b["per_label"][1]["skipped"] is not None


def test_dag_closure_parents_densifies_parent_int_over_range_C():
    from gated_pc_cloud import dag_closure_parents
    # engine ids: 0 root; 1,2 -> 0; 3 -> {1,2} (diamond). C=4.
    parent_int = {1: [0], 2: [0], 3: [1, 2]}
    cp = dag_closure_parents(parent_int, C=4)
    assert cp == [[], [0], [0], [1, 2]]        # root (0) and gaps -> []
    assert len(cp) == 4


def test_dag_children_and_depth():
    from gated_pc_cloud import _dag_children_and_depth
    # 0 root; 1,2 -> 0; 3,4 -> 1. C=5.
    parent_int = {1: [0], 2: [0], 3: [1], 4: [1]}
    children, depth = _dag_children_and_depth(parent_int, C=5)
    assert children[0] == [1, 2] and sorted(children[1]) == [3, 4]
    assert children[3] == [] and children[4] == []
    assert depth == {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}


def test_conditional_readout_sharpens_within_parent_cohort():
    from gated_pc_cloud import conditional_readout
    # 0 root; 1,2 -> 0; 3,4 -> 1. Among node-1's cohort, node 3 vs 4 is perfectly
    # separable, so the conditional AUC and multiclass top-1 must be 1.0.
    parent_int = {1: [0], 2: [0], 3: [1], 4: [1]}
    C = 5
    rng = np.random.default_rng(0)
    rows = []
    # closure labels: a doc at leaf L is positive at L, its parent 1, and root 0.
    for _ in range(20):   # at node 3
        rows.append(([1, 1, 0, 1, 0], [0.1, 0.5, 0.1, 0.9, 0.1]))
    for _ in range(20):   # at node 4
        rows.append(([1, 1, 0, 0, 1], [0.1, 0.5, 0.1, 0.1, 0.9]))
    for _ in range(20):   # at node 2 (sibling of 1, outside 1's cohort)
        rows.append(([1, 0, 1, 0, 0], [0.1, 0.1, 0.5, 0.1, 0.1]))
    y = np.array([r[0] for r in rows], float)
    proba = np.array([r[1] for r in rows], float)
    mask = np.ones_like(y)
    cond = conditional_readout(proba, y, mask, parent_int, C, min_count=5)

    # parent-1 cohort has both children scored, each perfectly separable.
    e13 = [e for e in cond["edges"] if e["parent"] == 1 and e["child"] == 3]
    e14 = [e for e in cond["edges"] if e["parent"] == 1 and e["child"] == 4]
    assert e13 and e14
    assert e13[0]["cond_auc"] == 1.0 and e14[0]["cond_auc"] == 1.0
    assert e13[0]["depth"] == 1                       # parent 1 is at depth 1
    p1 = [p for p in cond["parents"] if p["parent"] == 1]
    assert p1 and p1[0]["top1"] == 1.0               # argmax child always correct
    # honesty fields: majority baseline (20/20 -> 0.5), balanced accuracy, ECE.
    assert p1[0]["majority"] == 0.5                   # 20 vs 20 children -> 0.5 baseline
    assert p1[0]["bal_acc"] == 1.0                    # perfect per-child recall
    assert cond["ece"] is not None and cond["ece"] >= 0.0
    # per-node reliability: each scored edge carries its OWN ECE, and the summary
    # reports mean/max/worst over them so pooling can't hide a miscalibrated node.
    assert e13[0]["ece"] is not None and e13[0]["ece"] >= 0.0
    ne = cond["node_ece"]
    assert ne is not None and ne["n_nodes"] == len(cond["edges"])
    assert ne["max"] >= ne["mean"] >= 0.0
    assert ne["max"] == max(e["ece"] for e in cond["edges"])


def test_ece_perfectly_calibrated_is_zero_and_miscalibrated_is_positive():
    from gated_pc_cloud import _ece
    import numpy as np
    rng = np.random.default_rng(0)
    # p == empirical frequency in each bin -> ECE ~ 0.
    p = np.repeat([0.05, 0.25, 0.55, 0.85], 400)
    y = np.concatenate([(rng.random(400) < q).astype(float)
                        for q in (0.05, 0.25, 0.55, 0.85)])
    assert _ece(y, p) < 0.05
    # confident-but-wrong -> large ECE.
    assert _ece(np.zeros(100), np.full(100, 0.9)) > 0.5


def test_calibrate_per_node_fixes_ece_and_preserves_ranking():
    from gated_pc_cloud import calibrate_per_node, _ece
    import numpy as np
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    C = 1
    n = 4000
    # true P = sigmoid of a signal; miscalibrate by squaring the score (monotone,
    # so ranking is preserved but reliability is wrong).
    signal = rng.standard_normal(n)
    true_p = 1.0 / (1.0 + np.exp(-signal))
    y = (rng.random(n) < true_p).astype(float)
    bad = true_p ** 2                                    # miscalibrated, same ranking
    # split train/test
    tr = np.zeros(n, bool); tr[: n // 2] = True; te = ~tr
    proba_tr = bad[:, None]; proba_te = bad[:, None]
    yv = y[:, None]; mask = np.ones((n, 1))
    cal = calibrate_per_node(proba_tr[tr], yv[tr], mask[tr], proba_te[te], C)
    # calibration lowers ECE...
    assert _ece(yv[te, 0], cal[:, 0]) < _ece(yv[te, 0], proba_te[te, 0])
    # ...and approximately preserves the ranking (isotonic is monotone; only the
    # flat regions it induces create ties that move AUC slightly).
    assert abs(roc_auc_score(yv[te, 0], cal[:, 0])
               - roc_auc_score(yv[te, 0], proba_te[te, 0])) < 0.01


def test_precision_at_recall_and_recall_at_fdr():
    from gated_pc_cloud import precision_at_recall, recall_at_fdr
    # perfectly separable: at any recall, precision 1.0; at any FDR, recall 1.0.
    y = [0, 0, 1, 1]
    p = [0.1, 0.2, 0.8, 0.9]
    assert precision_at_recall(y, p, [0.5, 1.0]) == {0.5: 1.0, 1.0: 1.0}
    assert recall_at_fdr(y, p, [0.0, 0.5]) == {0.0: 1.0, 0.5: 1.0}
    # one hard positive ranked below a negative: recall 1.0 now costs some precision.
    p2 = [0.1, 0.85, 0.8, 0.9]        # neg[1]=0.85 outranks pos[2]=0.8
    par = precision_at_recall(y, p2, [1.0])
    assert par[1.0] < 1.0             # to catch both positives we admit the negative
    # degenerate single-class column -> NaN, not an exception.
    import math
    assert math.isnan(precision_at_recall([1, 1, 1], [0.2, 0.5, 0.9], [0.5])[0.5])


def test_score_arm_separable_and_detection_shape():
    import numpy as np
    from gated_pc_cloud import score_arm
    Pi_tr = np.array([[1., 0], [.9, .1], [.1, .9], [0, 1.], [.8, .2], [.2, .8]])
    y_tr = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 1]], float)
    Pi_te = np.array([[.95, .05], [.05, .95], [.85, .15], [.15, .85]])
    y_te = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], float)
    arm = score_arm(Pi_tr, y_tr, np.ones_like(y_tr), Pi_te, y_te, np.ones_like(y_te),
                    2, recall_targets=[0.5, 0.9], fdr_targets=[0.1, 0.5])
    assert arm["ranking"]["auc"] == 1.0
    assert set(arm["pr"]["par"]) == {0.5, 0.9}
    assert set(arm["pr"]["raf"]) == {0.1, 0.5}
    assert arm["pr"]["n_scored"] == 2
    # detection present with an AP + a precision@recall map (root=node0 label).
    assert "ap" in arm["detection"] and set(arm["detection"]["par"]) == {0.5, 0.9}


def test_parse_args_surface_defaults_and_arms():
    from gated_pc_cloud import parse_args
    a = parse_args([
        "--cdr", "p.d", "--billing", "bp", "--disease", "rare6",
        "--n-bg", "2", "--tpn", "1", "--weight-y", "80", "--out-dir", "/tmp/g",
        "--with-dag-head",
    ])
    assert a.disease == "rare6" and a.weight_y == 80.0
    assert a.head_optimizer == "newton"        # settled convergent head default
    assert a.head_l2 == 1e-3                    # absolute ridge (ADR 0041)
    assert a.label_mask_mode == "full"         # full-observation default
    assert a.with_dag_head is True
    assert a.skip_unsup_gated is False
    assert a.weight_y_warmup_iters == 10
    assert a.eval_every == 0                   # per-iter eval off by default


def test_head_l2_ridge_round_trips_through_estimator_and_engine():
    """--head-l2 flow: parse_args -> _build_pc_estimator (shim Params) ->
    OnlinePCLDA engine (head_l2 attribute). The ridge is the sole head regularizer
    (ADR 0043: Firth + Path B removed)."""
    from gated_pc_cloud import parse_args, _build_pc_estimator
    from spark_vi.mllib.topic.pc import _build_model_and_config
    a = parse_args([
        "--cdr", "p.d", "--billing", "bp", "--out-dir", "/tmp/g",
        "--head-optimizer", "newton", "--head-l2", "0.01", "--k", "4",
    ])
    assert a.head_l2 == 0.01 and a.head_optimizer == "newton"
    a._C = 1
    est = _build_pc_estimator(a, weight_y=50.0, gated=False)
    assert est.getOrDefault("headL2") == 0.01
    model, _ = _build_model_and_config(est, vocab_size=8)
    assert model.head_l2 == 0.01               # the shim built a ridge-only engine
    assert model.head_optimizer == "newton"
