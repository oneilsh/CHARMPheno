"""Tests for the reusable eval harness :mod:`analysis.pc.evaluate` (plan Task B2).

Three things are checked:

  1. **Well-formed + deterministic + PC wins/ties** on the vendored
     ``toy_bars_3x3`` oracle corpus: the harness returns the right keys/shapes,
     a macro summary is present, two runs with the same seed are bit-for-bit
     identical, and PC's macro-AUC is >= the two-stage baseline's (the Hughes
     headline — PC should win or at least tie). The formatted table is printed.
  2. **Degenerate-label handling** on a small synthetic multi-label set: a
     constant heldout label column is recorded as skipped (AUC undefined) and
     dropped from the macro-average, with no exception.
  3. **format_results_table** produces a readable string.

sklearn is allowed here (eval/baseline layer). Runtime is kept modest with small
K / capped ``max_iter`` / reduced ``pi_iters`` (fidelity is not what these test).
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import scipy.sparse as sp

from analysis.pc.evaluate import evaluate_pc_vs_baselines, format_results_table

_VENDORED = os.path.join(
    os.path.dirname(__file__), "data", "toy_bars_3x3"
)
_DATA_DIR = (
    os.path.join(os.environ["PC_REPO_DIR"], "datasets", "toy_bars_3x3")
    if os.environ.get("PC_REPO_DIR")
    else _VENDORED
)


def _load_split(split):
    """Rebuild a dense count matrix + label array for one toy_bars split."""
    z = np.load(os.path.join(_DATA_DIR, f"X_csr_{split}.npz"))
    X = sp.csr_matrix(
        (z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"])
    ).toarray().astype(np.float64)
    y = np.load(os.path.join(_DATA_DIR, f"Y_{split}.npy")).astype(np.float64)
    return X, y


def _assert_bundle_shape(bundle, C):
    """Every model bundle has per_label entries for each label and a macro block."""
    assert set(bundle.keys()) == {"per_label", "macro"}
    assert set(bundle["per_label"].keys()) == set(range(C))
    for c in range(C):
        d = bundle["per_label"][c]
        for key in ("auc", "ap", "n_pos", "n_neg", "skipped"):
            assert key in d, f"missing {key} in per-label record"
    m = bundle["macro"]
    for key in ("auc", "ap", "n_labels_scored", "n_labels_skipped"):
        assert key in m, f"missing {key} in macro record"


@pytest.mark.skipif(
    not os.path.isdir(_DATA_DIR),
    reason=f"toy_bars_3x3 reference dataset not found at {_DATA_DIR}",
)
def test_harness_wellformed_deterministic_and_pc_beats_two_stage(capsys):
    """Harness is well-formed + deterministic, and PC macro-AUC >= two-stage."""
    Xtr, Ytr = _load_split("train")
    Xte, Yte = _load_split("test")

    kw = dict(
        K=4, weight_y=10.0, alpha=1.1, tau=1.1,
        pi_iters=50, max_iter=120, seed=0,
    )
    res = evaluate_pc_vs_baselines(Xtr, Ytr, Xte, Yte, **kw)

    # (a) structure
    assert set(res.keys()) == {"PC", "two_stage", "lr_codes", "meta"}
    C = res["meta"]["C"]
    assert C == 1
    for key in ("PC", "two_stage", "lr_codes"):
        _assert_bundle_shape(res[key], C)
    assert res["meta"]["n_train"] == Xtr.shape[0]
    assert res["meta"]["n_test"] == Xte.shape[0]
    assert res["meta"]["n_labeled"] == Xtr.shape[0]

    # (b) determinism given the seed
    res2 = evaluate_pc_vs_baselines(Xtr, Ytr, Xte, Yte, **kw)
    for key in ("PC", "two_stage", "lr_codes"):
        assert res[key]["macro"]["auc"] == res2[key]["macro"]["auc"]
        assert res[key]["macro"]["ap"] == res2[key]["macro"]["ap"]

    # (c) the Hughes headline: PC wins or ties the unsupervised two-stage
    pc_auc = res["PC"]["macro"]["auc"]
    ts_auc = res["two_stage"]["macro"]["auc"]
    assert pc_auc is not None and ts_auc is not None
    assert pc_auc >= ts_auc, (
        f"PC macro-AUC {pc_auc:.4f} did not meet/exceed two-stage {ts_auc:.4f}"
    )

    with capsys.disabled():
        print()
        print(format_results_table(res))


def _synthetic_two_label(seed=0):
    """Small two-label synthetic corpus; label 1's TEST column is made constant.

    Label 0 carries a real, learnable signal (topic-A-heavy docs are positive);
    label 1 is a normal signal at train time but its heldout column is forced
    all-0 to exercise the degenerate-skip path.
    """
    rng = np.random.default_rng(seed)
    V = 6
    # Two "themes": words {0,1,2} vs {3,4,5}.
    theme_a = np.array([5.0, 5.0, 5.0, 0.5, 0.5, 0.5])
    theme_b = np.array([0.5, 0.5, 0.5, 5.0, 5.0, 5.0])

    def make(n):
        X = np.zeros((n, V))
        y = np.zeros((n, 2))
        for i in range(n):
            is_a = rng.random() < 0.5
            rate = theme_a if is_a else theme_b
            X[i] = rng.poisson(rate)
            y[i, 0] = 1.0 if is_a else 0.0          # label 0: theme-A indicator
            y[i, 1] = 1.0 if X[i].sum() > 12 else 0  # label 1: some other signal
        return X, y

    Xtr, Ytr = make(40)
    Xte, Yte = make(20)
    # Force label-0 test column to have both classes (so it stays scorable).
    Yte[0, 0], Yte[1, 0] = 1.0, 0.0
    # Degenerate: label-1 heldout column all-0 -> AUC undefined -> must be skipped.
    Yte[:, 1] = 0.0
    return Xtr, Ytr, Xte, Yte


def test_degenerate_label_is_skipped_not_crash(capsys):
    """A constant heldout label column is recorded as skipped, no exception."""
    Xtr, Ytr, Xte, Yte = _synthetic_two_label(seed=0)

    res = evaluate_pc_vs_baselines(
        Xtr, Ytr, Xte, Yte,
        K=2, weight_y=5.0, pi_iters=20, max_iter=40, seed=0,
    )

    for key in ("PC", "two_stage", "lr_codes"):
        block = res[key]
        _assert_bundle_shape(block, 2)
        # label 1 (constant heldout column) is skipped with a recorded reason
        lab1 = block["per_label"][1]
        assert lab1["skipped"] is not None
        assert lab1["auc"] is None and lab1["ap"] is None
        # label 0 is scorable
        lab0 = block["per_label"][0]
        assert lab0["skipped"] is None
        assert lab0["auc"] is not None
        # macro is over the one scored label only
        assert block["macro"]["n_labels_scored"] == 1
        assert block["macro"]["n_labels_skipped"] == 1
        assert block["macro"]["auc"] == lab0["auc"]

    table = format_results_table(res)
    assert "MACRO" in table
    assert "skipped" in table
    with capsys.disabled():
        print()
        print(table)
