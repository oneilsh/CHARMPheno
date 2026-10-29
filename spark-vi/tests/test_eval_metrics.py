"""Exact metrics from score tallies (`spark_vi.eval.metrics`, ADR 0051 §3).

scikit-learn is the oracle for AUC, AP and the PR curve; the tallies must give the
same numbers on any column (ties included) and merge to the same answer in any split,
which is what the Spark backend relies on. One slow test checks Spark against local.
"""
import math

import numpy as np
import pytest

from spark_vi.eval.metrics import core, local, report

RT, FQ = (0.25, 0.5, 0.9, 1.0), (0.0, 0.1, 0.5)


def _col(seed, n=400, ties=False, prev=0.2):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prev).astype(np.uint8)
    s = rng.normal(size=n) + 1.2 * y
    if ties:
        s = np.round(s, 1)                       # many tied levels
    return s.astype(np.float32), y


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_sklearn(seed, ties):
    sk = pytest.importorskip("sklearn.metrics")
    s, y = _col(seed, ties=ties)
    m = core.node_metrics(core.tally(s, y), recall_targets=RT, fdr_targets=FQ)
    assert m["auc"] == pytest.approx(sk.roc_auc_score(y, s), abs=1e-12)
    assert m["ap"] == pytest.approx(sk.average_precision_score(y, s), abs=1e-12)
    prec, rec, _ = sk.precision_recall_curve(y, s)
    for r in RT:
        assert m["par"][r] == pytest.approx(float(np.max(prec[rec >= r])), abs=1e-12)
    for q in FQ:
        ok = prec >= 1 - q
        want = float(np.max(rec[ok])) if ok.any() else 0.0
        assert m["raf"][q] == pytest.approx(want, abs=1e-12)
    assert m["n_pos"] == int(y.sum()) and m["n_neg"] == int((1 - y).sum())


@pytest.mark.parametrize("ties", [False, True])
def test_merge_in_any_split_gives_the_same_metrics(ties):
    s, y = _col(5, n=1000, ties=ties)
    whole = core.node_metrics(core.tally(s, y), recall_targets=RT, fdr_targets=FQ)
    rng = np.random.default_rng(0)
    cut = np.sort(rng.choice(np.arange(1, 1000), 6, replace=False))
    t = core.EMPTY
    for part in np.split(rng.permutation(1000), cut):
        t = core.merge(t, core.tally(s[part], y[part]))
    merged = core.node_metrics(t, recall_targets=RT, fdr_targets=FQ)
    for k in ("auc", "ap", "prec_at_k", "recall_at_k", "lift_at_k"):
        assert merged[k] == pytest.approx(whole[k], abs=1e-12)
    assert merged["par"] == whole["par"] and merged["raf"] == whole["raf"]


def test_topk_without_ties_equals_the_row_cut():
    s, y = _col(3, n=500)
    for frac in (0.01, 0.1, 0.33):
        m = core.node_metrics(core.tally(s, y), topk_frac=frac)
        k = max(1, math.ceil(frac * 500))
        tp = int(y[np.argsort(-s, kind="stable")[:k]].sum())
        assert m["prec_at_k"] == pytest.approx(tp / k)
        assert m["recall_at_k"] == pytest.approx(tp / y.sum())
        assert m["lift_at_k"] == pytest.approx((tp / k) / y.mean())


def test_topk_inside_a_tie_counts_the_tie_at_its_mean_rate():
    # top 3 of: 0.9 (pos) then a 4-way tie at 0.5 with 2 positives
    s = np.array([0.9, 0.5, 0.5, 0.5, 0.5, 0.1])
    y = np.array([1, 1, 0, 1, 0, 0])
    m = core.node_metrics(core.tally(s, y), topk_frac=0.5)          # k = 3
    assert m["prec_at_k"] == pytest.approx((1 + 2 * 0.5) / 3)


def test_single_class_and_constant_columns():
    m = core.node_metrics(core.tally([0.1, 0.2], [1, 1]), recall_targets=(0.5,))
    assert m["auc"] is None and math.isnan(m["par"][0.5])
    assert report.score_record(m)["skip_code"] == report.SKIP_DEGENERATE
    c = core.node_metrics(core.tally(np.full(50, 0.3), np.arange(50) % 2))
    assert c["auc"] == 0.5 and c["n_distinct"] == 1
    assert report.score_record(c)["skipped"] is None
    assert report.score_record(c, skip_constant=True)["skip_code"] == report.SKIP_CONSTANT
    assert report.score_record(c, min_count=30)["skip_code"] == report.SKIP_SMALL


def test_curve_ends_and_thinning():
    s, y = _col(4, n=2000)
    cv = core.curve(core.tally(s, y), max_points=50)
    assert len(cv["fpr"]) <= 50
    assert cv["tpr"][-1] == 1.0 and cv["fpr"][-1] == 1.0
    assert cv["threshold"][0] == pytest.approx(float(s.max()))
    assert all(a >= b for a, b in zip(cv["threshold"], cv["threshold"][1:]))


def test_calibration_bins_match_the_direct_ece():
    rng = np.random.default_rng(0)
    p = rng.random(1000)
    p[:5] = 1.0
    y = (rng.random(1000) < p).astype(float)
    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, edges) - 1, 0, 9)
    want = sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
               for b in range(10) if (idx == b).any())
    halves = [core.calibration_bins(p[:400], y[:400]), core.calibration_bins(p[400:], y[400:])]
    merged = tuple(a + b for a, b in zip(*halves))
    assert core.ece_from_bins(merged) == pytest.approx(want, abs=1e-12)
    assert core.ece(p, y) == pytest.approx(want, abs=1e-12)
    assert local.per_key_calibration([("a", p[:400], y[:400]), ("a", p[400:], y[400:])])[
        "a"] == pytest.approx(want, abs=1e-12)


def test_macro_and_pr_records():
    recs = {0: report.score_record(core.node_metrics(core.tally(*_col(0)))),
            1: report.score_record(core.node_metrics(core.tally([.1, .2], [1, 1])))}
    mac = report.macro(recs)
    assert mac["n_labels_scored"] == 1 and mac["auc"] == recs[0]["auc"]
    assert mac["skipped_by_reason"][report.SKIP_DEGENERATE] == 1
    m = core.node_metrics(core.tally(*_col(1)), recall_targets=RT, fdr_targets=FQ)
    pr = {0: report.pr_record(m), 1: report.pr_record(m, min_count=10_000)}
    assert pr[1]["skipped"] and report.pr_macro(pr, RT, FQ)["n_scored"] == 1


def _pieces(D=600, C=5, n_parts=4, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random((D, C)) < 0.15).astype(np.uint8)
    s = np.round(rng.normal(size=(D, C)) + y, 2).astype(np.float32)
    keep = rng.random((D, C)) < 0.8
    parts = np.array_split(np.arange(D), n_parts)
    return s, y, keep, parts


def test_local_pieces_match_whole_columns():
    s, y, keep, parts = _pieces()
    pieces = [pc for rows in parts
              for pc in local.column_pieces(s[rows], y[rows], keep[rows])]
    got = local.per_key_metrics(pieces, recall_targets=RT, curve_keys=(2,))
    for c in range(s.shape[1]):
        r = keep[:, c]
        want = core.node_metrics(core.tally(s[r, c], y[r, c]), recall_targets=RT)
        assert got[c]["auc"] == pytest.approx(want["auc"], abs=1e-12)
    assert got[2]["curve"] is not None and "curve" not in got[1]


@pytest.mark.slow
def test_spark_matches_local(spark):
    from spark_vi.eval.metrics import spark as sm

    s, y, keep, parts = _pieces(D=3000, C=8)
    blocks = [(s[r], y[r], keep[r]) for r in parts]
    rdd = spark.sparkContext.parallelize(blocks, len(blocks)).flatMap(
        lambda b: local.column_pieces(*b, key=lambda c: ("v", c)))
    got = sm.per_key_metrics(rdd, recall_targets=RT, fdr_targets=FQ,
                             curve_keys={("v", 3)}, max_points=40)
    want = local.per_key_metrics(
        [pc for b in blocks for pc in local.column_pieces(*b, key=lambda c: ("v", c))],
        recall_targets=RT, fdr_targets=FQ, curve_keys={("v", 3)}, max_points=40)
    assert set(got) == set(want)
    for k in want:
        for f in ("auc", "ap", "n_pos", "n_neg", "n_distinct", "lift_at_k"):
            assert got[k][f] == pytest.approx(want[k][f], abs=1e-12)
        assert got[k]["par"] == want[k]["par"] and got[k]["raf"] == want[k]["raf"]
    assert got[("v", 3)]["curve"] == want[("v", 3)]["curve"]
    cal = sm.per_key_calibration(spark.sparkContext.parallelize(
        [("a", s[r, 0].clip(0, 1), y[r, 0]) for r in parts], 3))
    assert cal["a"] == pytest.approx(local.per_key_calibration(
        [("a", s[r, 0].clip(0, 1), y[r, 0]) for r in parts])["a"], abs=1e-12)


def test_ece_perfectly_calibrated_is_zero_and_miscalibrated_is_positive():
    rng = np.random.default_rng(0)
    p = np.repeat([0.05, 0.25, 0.55, 0.85], 400)
    y = np.concatenate([(rng.random(400) < q).astype(float)
                        for q in (0.05, 0.25, 0.55, 0.85)])
    assert core.ece(p, y) < 0.05
    assert core.ece(np.full(100, 0.9), np.zeros(100)) > 0.5
    assert core.ece([], []) is None
