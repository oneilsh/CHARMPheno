"""Tests for the per-label metric primitives in :mod:`analysis.pc.evaluate`:
`_score_label` (min-count masking, degenerate-label skips, top-k screening),
`_macro`, `_bundle_masked`, and the feature-scale invariance of the per-label
logistic baselines. sklearn is allowed here (eval layer).
"""
from __future__ import annotations

import numpy as np

from analysis.pc.evaluate import (
    _bundle_masked,
    _score_label,
)


def test_score_label_default_scores_small_columns():
    """min_count=0 (library default) scores a small-but-two-class column."""
    y = np.array([1, 0, 1, 0, 1])           # 3 pos / 2 neg — tiny but valid
    p = np.array([0.9, 0.1, 0.8, 0.2, 0.7])
    d = _score_label(y, p)                   # default min_count=0
    assert d["skipped"] is None
    assert d["auc"] is not None
    assert d["n_pos"] == 3 and d["n_neg"] == 2


def test_score_label_masks_below_min_count():
    """A column with a class below min_count is skipped (AUC not computed)."""
    y = np.concatenate([np.ones(5), np.zeros(50)])      # 5 pos / 50 neg
    p = np.linspace(0, 1, y.size)
    d = _score_label(y, p, min_count=20)
    assert d["skipped"] is not None and "small test column" in d["skipped"]
    assert d["auc"] is None and d["ap"] is None
    # counts are still recorded (the formatter suppresses them for display)
    assert d["n_pos"] == 5 and d["n_neg"] == 50


def test_score_label_keeps_both_sides_above_min_count():
    y = np.concatenate([np.ones(25), np.zeros(30)])     # 25 / 30 — both >= 20
    p = np.linspace(0, 1, y.size)
    d = _score_label(y, p, min_count=20)
    assert d["skipped"] is None and d["auc"] is not None


def test_score_label_degenerate_reason_distinct_from_small():
    """An all-one-class column is 'degenerate', not 'small' — even under min_count."""
    y = np.ones(3)
    p = np.array([0.2, 0.5, 0.9])
    d = _score_label(y, p, min_count=20)
    assert d["skipped"] is not None and "degenerate" in d["skipped"]


def test_score_label_topk_perfect_ranker():
    """A perfect ranker puts all positives on top: precision@top-k = 1 until the
    slice exceeds the positive count, and lift = 1/prevalence at that ceiling."""
    # 10 pos / 90 neg (prevalence 0.10); top-10% slice = 10 rows, all positive.
    y = np.concatenate([np.ones(10), np.zeros(90)])
    p = np.concatenate([np.linspace(1.0, 0.9, 10), np.linspace(0.5, 0.0, 90)])
    d = _score_label(y, p, topk_frac=0.10)
    assert d["topk_frac"] == 0.10
    assert d["prec_at_k"] == 1.0                 # top 10 are all cases
    assert d["recall_at_k"] == 1.0               # and that captures every case
    assert abs(d["lift_at_k"] - 10.0) < 1e-9     # 1.0 / prevalence(0.10)


def test_score_label_topk_random_ranker_lift_near_one():
    """A non-informative (constant-ish random) ranker gives lift ~ 1 — no better
    than flagging a random slice / predicting the majority."""
    rng = np.random.default_rng(0)
    y = np.concatenate([np.ones(50), np.zeros(4950)])   # prevalence 0.01
    p = rng.random(y.size)                               # unrelated to y
    d = _score_label(y, p, topk_frac=0.01)
    assert d["lift_at_k"] is not None
    assert 0.0 <= d["lift_at_k"] <= 3.0                  # scatters around 1, not >>1


def test_score_label_topk_absent_on_skips():
    """Skip records (degenerate / small) carry no top-k fields (readers .get)."""
    d = _score_label(np.ones(5), np.linspace(0, 1, 5))   # all-positive -> skip
    assert d["skipped"] is not None
    assert d.get("lift_at_k") is None and "lift_at_k" not in d


def test_macro_pools_topk_over_scored_labels():
    """_macro averages lift/prec/recall over the scored labels and echoes topk_frac."""
    from analysis.pc.evaluate import _macro
    y = np.concatenate([np.ones(10), np.zeros(90)])
    p_good = np.concatenate([np.ones(10), np.zeros(90)])       # lift 10
    p_flat = np.concatenate([np.zeros(10), np.ones(90)])       # positives ranked last
    per_label = {0: _score_label(y, p_good, topk_frac=0.10),
                 1: _score_label(y, p_flat, topk_frac=0.10),
                 2: _score_label(np.ones(5), np.linspace(0, 1, 5))}  # skipped
    m = _macro(per_label)
    assert m["topk_frac"] == 0.10
    assert m["n_labels_scored"] == 2                 # the skip drops out
    assert m["lift_at_k"] is not None                # pooled over the 2 scored
    assert 0.0 <= m["prec_at_k"] <= 1.0


def test_bundle_masked_threads_min_count_and_drops_from_macro():
    """_bundle_masked masks a small column and excludes it from the macro."""
    D, C = 80, 2
    rng = np.random.default_rng(0)
    proba = rng.random((D, C))
    y = np.zeros((D, C))
    mask = np.ones((D, C))
    # label 0: healthy (40/40); label 1: only 5 positives (small)
    y[:40, 0] = 1.0
    y[:5, 1] = 1.0
    bundle = _bundle_masked(proba, y, mask, C, min_count=20)
    assert bundle["per_label"][0]["skipped"] is None
    assert bundle["per_label"][1]["skipped"] is not None
    assert bundle["macro"]["n_labels_scored"] == 1
    assert bundle["macro"]["n_labels_skipped"] == 1


# --- two-stage skip / baseline_max_iter (--skip-two-stage / --baseline-max-iter) --

# --- LR baselines standardize features (converge on high-dim counts) ----------

def test_lr_masked_is_feature_scale_invariant():
    """Standardization makes the masked LR baseline invariant to feature scale —
    the proof the scaler is applied (unscaled LR would differ and, on high-dim
    counts, fail to converge)."""
    from analysis.pc.evaluate import _lr_proba_per_label_masked
    rng = np.random.default_rng(0)
    X_tr = rng.integers(0, 5, size=(60, 8)).astype(float)
    X_te = rng.integers(0, 5, size=(20, 8)).astype(float)
    y = np.zeros((60, 2))
    y[:30, 0] = 1.0
    y[::2, 1] = 1.0
    mask = np.ones((60, 2))
    p1 = _lr_proba_per_label_masked(X_tr, y, mask, X_te, 2)
    # Scale features by a large per-column constant -> standardization cancels it.
    scale = np.array([1.0, 1000.0, 10.0, 500.0, 1.0, 250.0, 5.0, 100.0])
    p2 = _lr_proba_per_label_masked(X_tr * scale, y, mask, X_te * scale, 2)
    assert np.allclose(p1, p2, atol=1e-6)


def test_lr_single_task_is_feature_scale_invariant():
    from analysis.pc.evaluate import _lr_proba_per_label
    rng = np.random.default_rng(1)
    X_fit = rng.integers(0, 5, size=(50, 5)).astype(float)
    X_te = rng.integers(0, 5, size=(15, 5)).astype(float)
    y = np.zeros((50, 1)); y[:25, 0] = 1.0
    scale = np.array([1.0, 1000.0, 10.0, 500.0, 2.0])
    p1 = _lr_proba_per_label(X_fit, y, X_te, 1)
    p2 = _lr_proba_per_label(X_fit * scale, y, X_te * scale, 1)
    assert np.allclose(p1, p2, atol=1e-6)

