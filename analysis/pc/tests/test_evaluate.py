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

from analysis.pc.evaluate import (
    _bundle_masked,
    _score_label,
    evaluate_pc_multitask,
    evaluate_pc_vs_baselines,
    format_results_table,
)

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


# --- multi-task / per-cell missing-label path (evaluate_pc_multitask) -------

def _mt_corpus(seed, D, C=2, V=12, n_tok=80):
    """Small, easily-learnable multi-outcome corpus: ``C`` disjoint topic blocks,
    each outcome driven by its own topic's per-doc weight. Fast to fit; both PC
    and the two-stage baseline learn it well (so PC macro-AUC >= two-stage holds).
    """
    rng = np.random.default_rng(seed)
    block = V // C
    topics = np.full((C, V), 0.03)
    for k in range(C):
        topics[k, k * block:(k + 1) * block] += 1.0
    topics /= topics.sum(axis=1, keepdims=True)
    theta = rng.dirichlet(np.full(C, 0.5), size=D)
    X = np.zeros((D, V))
    for d in range(D):
        X[d] = rng.multinomial(n_tok, theta[d] @ topics)
    y = np.zeros((D, C))
    for c in range(C):
        y[:, c] = (theta[:, c] > np.median(theta[:, c])).astype(float)
    return X, y


def _index_drug_mask(D, C, seed):
    """A ``(D, C)`` mask with exactly one observed cell per row (index-drug)."""
    rng = np.random.default_rng(seed)
    obs = rng.integers(0, C, size=D)
    m = np.zeros((D, C))
    m[np.arange(D), obs] = 1.0
    return m


def test_multitask_harness_wellformed_deterministic_and_pc_ge_two_stage(capsys):
    """``evaluate_pc_multitask`` under an index-drug mask: same result-dict shape
    as the single-task harness (so ``format_results_table`` works), per-column
    observed-N reported, deterministic, and PC macro-AUC >= two-stage.

    One-observed-cell-per-row (index-drug) is the primary use, but any per-cell
    mask is accepted; the general-mask + degenerate path is covered below.
    """
    C = 2
    Xtr, Ytr = _mt_corpus(0, 120, C=C)
    Xte, Yte = _mt_corpus(1, 80, C=C)
    mask_tr = _index_drug_mask(120, C, seed=5)      # one observed outcome / row
    mask_te = np.ones((80, C))                       # score every test cell
    assert np.array_equal(mask_tr.sum(axis=1), np.ones(120))  # index-drug invariant

    kw = dict(K=2, weight_y=8.0, pi_iters=25, max_iter=50, seed=0)
    res = evaluate_pc_multitask(Xtr, Ytr, mask_tr, Xte, Yte, mask_te, **kw)

    # (a) same structure as evaluate_pc_vs_baselines
    assert set(res.keys()) == {"PC", "two_stage", "lr_codes", "meta"}
    assert res["meta"]["C"] == C
    for key in ("PC", "two_stage", "lr_codes"):
        _assert_bundle_shape(res[key], C)
    # (b) per-column observed-N (train/test) in meta
    meta = res["meta"]
    assert meta["n_obs_train"] == [int(mask_tr[:, c].sum()) for c in range(C)]
    assert meta["n_obs_test"] == [80, 80]
    assert meta["n_labeled"] == int(mask_tr.sum())

    # (c) determinism given the seed
    res2 = evaluate_pc_multitask(Xtr, Ytr, mask_tr, Xte, Yte, mask_te, **kw)
    for key in ("PC", "two_stage", "lr_codes"):
        assert res[key]["macro"]["auc"] == res2[key]["macro"]["auc"]

    # (d) PC macro-AUC meets or beats the unsupervised two-stage
    pc = res["PC"]["macro"]["auc"]
    ts = res["two_stage"]["macro"]["auc"]
    assert pc is not None and ts is not None
    assert pc >= ts, f"PC macro-AUC {pc:.4f} < two-stage {ts:.4f}"

    # (e) the formatted table renders, including the observed-cells footer
    table = format_results_table(res)
    assert "MACRO" in table and "observed cells per label" in table
    with capsys.disabled():
        print()
        print(table)


def test_multitask_general_mask_degenerate_column_skipped(capsys):
    """A GENERAL per-cell mask (not one-per-row) with a degenerate observed test
    column: the outcome whose observed heldout cells are single-class is skipped
    (AUC undefined) and dropped from the macro, no exception."""
    C = 2
    Xtr, Ytr = _mt_corpus(0, 100, C=C)
    Xte, Yte = _mt_corpus(1, 60, C=C)

    rng = np.random.default_rng(3)
    mask_tr = (rng.random((100, C)) >= 0.3).astype(float)   # general per-cell mask
    mask_te = (rng.random((60, C)) >= 0.3).astype(float)
    for c in range(C):                                       # keep every column non-empty
        if mask_tr[:, c].sum() == 0:
            mask_tr[0, c] = 1.0
    # Force outcome 1's OBSERVED heldout cells to be all-negative -> degenerate.
    obs1 = mask_te[:, 1].astype(bool)
    assert obs1.any()
    Yte[obs1, 1] = 0.0

    res = evaluate_pc_multitask(
        Xtr, Ytr, mask_tr, Xte, Yte, mask_te,
        K=2, weight_y=5.0, pi_iters=20, max_iter=40, seed=0,
    )

    for key in ("PC", "two_stage", "lr_codes"):
        block = res[key]
        _assert_bundle_shape(block, C)
        lab1 = block["per_label"][1]
        assert lab1["skipped"] is not None
        assert lab1["auc"] is None and lab1["ap"] is None
        assert block["macro"]["n_labels_skipped"] == 1
        assert block["macro"]["n_labels_scored"] == 1

    table = format_results_table(res)
    assert "skipped" in table
    with capsys.disabled():
        print()
        print(table)


# --- small-count label masking (min_label_count / --min-label-count) ----------

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


def test_format_results_table_suppresses_small_counts():
    """The printed table shows '<20' for masked-column counts, not the raw N."""
    per_label = {
        0: {"auc": 0.66, "ap": 0.6, "n_pos": 404, "n_neg": 457, "skipped": None},
        1: {"auc": None, "ap": None, "n_pos": 5, "n_neg": 12,
            "skipped": "small test column (min class < 20); ... masked"},
    }

    def _macro_of(pl):
        aucs = [d["auc"] for d in pl.values() if d["skipped"] is None]
        return {"auc": float(np.mean(aucs)), "ap": 0.6,
                "n_labels_scored": len(aucs),
                "n_labels_skipped": len(pl) - len(aucs)}

    block = {"per_label": per_label, "macro": _macro_of(per_label)}
    results = {
        "PC": block, "two_stage": block, "lr_codes": block,
        "meta": {
            "C": 2, "K": 25, "weight_y": 100.0, "n_train": 100, "n_test": 50,
            "n_labeled": 100, "min_label_count": 20,
            "model_names": {"PC": "PC", "two_stage": "TS", "lr_codes": "LR"},
        },
    }
    table = format_results_table(results)
    # the healthy label's real counts appear; the small label's are suppressed
    assert "404/457" in table
    assert "5/12" not in table
    assert "<20/<20" in table
    assert "small" in table                  # skipped footer mentions the reason


def test_format_results_table_no_suppression_when_threshold_zero():
    """min_label_count=0 (or absent) prints raw counts (library default)."""
    per_label = {
        0: {"auc": 0.6, "ap": 0.5, "n_pos": 3, "n_neg": 4, "skipped": None},
    }
    block = {"per_label": per_label,
             "macro": {"auc": 0.6, "ap": 0.5,
                       "n_labels_scored": 1, "n_labels_skipped": 0}}
    results = {
        "PC": block, "two_stage": block, "lr_codes": block,
        "meta": {"C": 1, "K": 4, "weight_y": 10.0, "n_train": 10, "n_test": 7,
                 "n_labeled": 10,  # no min_label_count key -> threshold 0
                 "model_names": {"PC": "PC", "two_stage": "TS", "lr_codes": "LR"}},
    }
    table = format_results_table(results)
    assert "3/4" in table and "<" not in table


# --- two-stage skip / baseline_max_iter (--skip-two-stage / --baseline-max-iter) --

def test_multitask_baseline_probas_skip_returns_none_two_stage():
    from analysis.pc.evaluate import multitask_baseline_probas
    rng = np.random.default_rng(0)
    X_tr = rng.integers(0, 3, size=(30, 6)).astype(float)
    X_te = rng.integers(0, 3, size=(12, 6)).astype(float)
    y_tr = np.zeros((30, 2)); y_tr[:15, 0] = 1.0; y_tr[::2, 1] = 1.0
    mask = np.ones((30, 2))
    shared = dict(K=3, C=2, alpha=1.1, tau=1.1, pi_iters=10, max_iter=20,
                  doc_batch_size=64, seed=0)
    ts, lrc = multitask_baseline_probas(
        X_tr, y_tr, mask, X_te, 2, shared, skip_two_stage=True)
    assert ts is None                      # two-stage skipped -> no probas
    assert lrc.shape == (12, 2)            # LR-on-codes still computed


def test_evaluate_pc_multitask_skip_two_stage_omits_key(capsys):
    Xtr, Ytr = _mt_corpus(0, D=48, C=2)
    Xte, Yte = _mt_corpus(1, D=24, C=2)
    mtr = np.ones((48, 2)); mte = np.ones((24, 2))
    res = evaluate_pc_multitask(
        Xtr, Ytr, mtr, Xte, Yte, mte,
        K=2, weight_y=5.0, pi_iters=15, max_iter=30, seed=0,
        skip_two_stage=True,
    )
    assert "two_stage" not in res                     # dropped entirely
    assert "PC" in res and "lr_codes" in res
    assert res["meta"]["two_stage_skipped"] is True
    table = format_results_table(res)                 # must not KeyError
    assert "two-stage baseline skipped" in table
    assert "two-stage (unsup+LR)" not in table        # no rows for it


def test_evaluate_pc_multitask_baseline_max_iter_runs(capsys):
    # A tiny baseline_max_iter still produces a two_stage bundle (just cheaper).
    Xtr, Ytr = _mt_corpus(0, D=48, C=2)
    Xte, Yte = _mt_corpus(1, D=24, C=2)
    mtr = np.ones((48, 2)); mte = np.ones((24, 2))
    res = evaluate_pc_multitask(
        Xtr, Ytr, mtr, Xte, Yte, mte,
        K=2, weight_y=5.0, pi_iters=15, max_iter=30, seed=0,
        baseline_max_iter=5,
    )
    assert "two_stage" in res and res["meta"]["two_stage_skipped"] is False
