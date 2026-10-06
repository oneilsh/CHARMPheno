"""Stacked (closure-product) readout arm — spec 2026-10-06 Part A, off-Spark.

`gated_pc_cloud.closure_matrix` / `stacked_proba` / `stacked_readout`: the
product P_stack(c) = Π_{a ∈ closure(c)} σ(z_a) on a frozen (D, C) proba, its
DagClosureHead agreement, the monotone P(child) ≤ P(parent), and the block's
bookkeeping (root-only detection, paired per-node delta, marginal ECE by depth).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CLOUD = REPO_ROOT / "analysis" / "cloud"
for _p in (str(REPO_ROOT), str(CLOUD)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in (str(REPO_ROOT), str(CLOUD), os.environ.get("PYTHONPATH", "")) if p)

import gated_pc_cloud as gpc  # noqa: E402

# Root 0; 1, 2, 6 under the root; 3 under 1; 4 is a DIAMOND child of 1 and 2;
# 5 under 4 (closure(5) = {0, 1, 2, 4, 5}: the root counted once, not twice).
C = 7
PARENT = {1: [0], 2: [0], 3: [1], 4: [1, 2], 5: [4], 6: [0]}
RT, FT = [0.5, 0.9], [0.25]


def test_closure_matrix_counts_each_ancestor_once_and_includes_the_root():
    M = gpc.closure_matrix(PARENT, C)
    assert M.shape == (C, C)
    assert np.array_equal(np.flatnonzero(M[5]), [0, 1, 2, 4, 5])   # diamond: 0 once
    assert np.array_equal(np.flatnonzero(M[0]), [0])
    assert np.array_equal(np.flatnonzero(M[6]), [0, 6])
    assert (np.diag(M) == 1).all()


def test_closure_matrix_matches_the_dag_closure_head_convention():
    """One definition of 'closure' for the co-fit head and the readout arm."""
    from spark_vi.models.topic.pc import DagClosureHead
    head = DagClosureHead(gpc.dag_closure_parents(PARENT, C))
    assert np.array_equal(gpc.closure_matrix(PARENT, C), head._closure_matrix)


def _proba(seed=0, D=50):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.95, size=(D, C)).astype(np.float32)
    return p


def test_stacked_proba_is_the_closure_product_and_is_monotone_down_the_dag():
    p = _proba()
    M = gpc.closure_matrix(PARENT, C)
    P = gpc.stacked_proba(p, M)
    assert P.dtype == np.float32
    ref = np.prod(p[:, [0, 1, 2, 4, 5]].astype(np.float64), axis=1)
    assert np.allclose(P[:, 5], ref, rtol=1e-5)
    assert np.allclose(P[:, 0], p[:, 0], rtol=1e-6)
    for child, parents in PARENT.items():
        for par in parents:
            assert (P[:, child] <= P[:, par] + 1e-7).all(), (child, par)


def test_stacked_proba_chunking_is_exact_and_the_floor_saves_descendants():
    p = _proba(seed=3, D=37)
    M = gpc.closure_matrix(PARENT, C)
    assert np.array_equal(gpc.stacked_proba(p, M, chunk_rows=4),
                          gpc.stacked_proba(p, M, chunk_rows=1000))
    # A degenerate-negative ancestor (constant 0.0 fallback) must not zero every
    # descendant into a constant column: the floor keeps the product finite and
    # the descendants' own variation intact.
    q = p.copy()
    q[:, 1] = 0.0
    P = gpc.stacked_proba(q, M)
    assert np.isfinite(P).all()
    assert np.ptp(P[:, 3]) > 0 and np.ptp(P[:, 5]) > 0


def _labels(p, seed=1):
    rng = np.random.default_rng(seed)
    y = (rng.random(p.shape) < 0.5).astype(np.uint8)
    y[:, 0] = (rng.random(p.shape[0]) < 0.6).astype(np.uint8)   # cases vs background
    m = np.ones_like(y)
    return y, m


def test_stacked_readout_block_has_the_three_detection_reads_and_paired_bookkeeping():
    p = _proba(seed=5, D=120)
    y, m = _labels(p)
    prev = gpc.readout_from_proba(p, y, m, C, recall_targets=RT, fdr_targets=FT)
    blk = gpc.stacked_readout(p, y, m, PARENT, C, recall_targets=RT, fdr_targets=FT,
                              min_count=0, prevalent=prev, arm_label="arm")
    assert blk["naming"] == gpc.STACKED_NAMING
    assert blk["closure_terms"]["max"] == 5
    # Root-only detection IS detection_readout on the root column alone.
    root = gpc.detection_readout(np.stack([p[:, 0], p[:, 0]], 1), y, RT)
    assert blk["root_only_detection"]["auc"] == pytest.approx(root["auc"])
    assert blk["prevalent_detection"]["auc"] == pytest.approx(prev["detection"]["auc"])
    # The stacked detection is scored on the product (not the flat proba).
    M = gpc.closure_matrix(PARENT, C)
    own = gpc.readout_from_proba(gpc.stacked_proba(p, M), y, m, C,
                                 recall_targets=RT, fdr_targets=FT)
    assert blk["readout"]["detection"]["auc"] == pytest.approx(own["detection"]["auc"])
    pv = blk["paired_vs_prevalent"]
    a = pv["all"]
    assert a["n"] == len(set(prev["per_node"]) & set(blk["readout"]["per_node"]))
    assert a["wins"] + a["losses"] + a["ties"] == a["n"]
    assert sum(s["n"] for s in pv["by_depth"].values()) == a["n"]
    ece = blk["marginal_ece_by_depth"]
    assert "0" in ece and ece["0"]["n_nodes"] == 1          # the root head itself
    assert all(0.0 <= v[k] <= 1.0 for v in ece.values() for k in ("flat", "stacked"))
    # The de-novo read: per-node AUC over ALL docs, both arms, same scorer.
    from analysis.pc.evaluate import _bundle_masked
    mg = blk["marginal_ranking"]
    ref = _bundle_masked(gpc.stacked_proba(p, M), y, np.ones_like(y), C, 0)
    for c, r in mg["stacked"]["per_node"].items():
        assert r["auc"] == pytest.approx(ref["per_label"][c]["auc"])
    assert mg["stacked"]["macro"]["auc"] == pytest.approx(ref["macro"]["auc"])
    assert mg["paired"]["all"]["n"] == len(
        set(mg["flat"]["per_node"]) & set(mg["stacked"]["per_node"]))
    text = gpc.format_stacked_readout(blk)
    assert "root head alone" in text and "stacked: max_c P_stack(c)" in text
    assert "de-novo" in text


def test_a_constant_root_leaves_depth_one_ranking_tied_with_the_flat_arm():
    """The closure-mask situation the arm exists to fix: with the root head the
    degenerate constant 1.0, P_stack(c) == sigma(z_c) for every depth-1 node, so
    the paired delta there is exactly zero — the product adds nothing without a
    root head that saw the background."""
    p = _proba(seed=8, D=100)
    p[:, 0] = 1.0
    y, m = _labels(p)
    prev = gpc.readout_from_proba(p, y, m, C, recall_targets=RT, fdr_targets=FT)
    blk = gpc.stacked_readout(p, y, m, PARENT, C, recall_targets=RT, fdr_targets=FT,
                              prevalent=prev)
    d1 = blk["paired_vs_prevalent"]["by_depth"]["1"]
    assert d1["n"] == 3 and d1["ties"] == 3
    assert blk["root_only_detection"]["skipped"]           # a constant column
