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
