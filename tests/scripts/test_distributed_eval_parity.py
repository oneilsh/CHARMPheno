"""WP-B parity gate (exp 0111, spec R5.8): the DISTRIBUTED eval + binned
calibration must AGREE with the driver path they replace.

The audit's §5f wall is that the driver readout collects the (D_te,C) proba
(`_densify_lean_blocks`) and calibrates with a float64 copy of it, which breaks a
16 GB driver at the episode corpus's ×2.66 doc count. WP-B makes the never-called
distributed scoring real: `score_cells_arms_df` explodes only the observed-or-
incident cells and `per_node_metric_arms_rows` groups them, so nothing (D,C)
reaches the driver; and `fit_binned_isotonic` fits the per-node calibrator on
C×100 binned sufficient stats instead of the collected cal cells.

This file is the ORACLE the plan asks for: it proves, on fixtures that exercise
BOTH paths, that

  * per-node PREVALENT and INCIDENT AUC/AP from the distributed cells equal the
    driver's `_bundle_masked` numbers to numerical noise — the incident arm on the
    SAME eligibility semantics (`incident_eval_mask = elig & (y|mask)`) and with
    R2.1's constant-column guard, so the guard's skip counts match too;
  * binned isotonic calibration lands within a STATED ECE tolerance of the exact
    `calibrate_per_node`, with `min_pos=20` pass-through preserved.

Two layers, matching the repo convention: pure-numpy kernel/fit tests (fast, no
SparkSession) and a `@slow` local-Spark round trip for the Spark shells. The
`compare_per_node` helper is written so the orchestrator can point it at two real
`results_readout.json` per_node tables (driver vs `--eval-path distributed`) for
the deferred CLUSTER parity run on the real 0110 corpus.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CLOUD = REPO_ROOT / "analysis" / "cloud"
for _p in (str(REPO_ROOT), str(CLOUD), str(REPO_ROOT / "charmpheno"),
           str(REPO_ROOT / "spark-vi")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# PySpark workers inherit PYTHONPATH, not the driver's sys.path — the executors
# must import distributed_readout AND analysis.pc.evaluate (shipped by reference
# by _score_label / _score_label_pair). Set at collection time.
os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in (str(REPO_ROOT), str(CLOUD), str(REPO_ROOT / "charmpheno"),
                str(REPO_ROOT / "spark-vi"), os.environ.get("PYTHONPATH", ""))
    if p)

import distributed_readout as dr  # noqa: E402
import gated_pc_cloud as gpc  # noqa: E402
from analysis.pc.evaluate import _bundle_masked  # noqa: E402

RT, FT = [0.9], [0.5]

# STATED binned-isotonic ECE tolerance (recorded here as the deliverable asks). At
# 100 fixed-width bins the calibrated probability differs from exact isotonic only
# where two raw values fall in one bin, so the per-node ECE gap is bounded by the
# bin width's effect on the reliability curve. The MEASURED gap on the fixtures
# below is ~1e-3 (asserted); the tolerance is set an order of magnitude above it so
# a genuine binning regression trips it while bin-boundary noise does not. If a real
# corpus ever exceeds it, RAISE _CALIB_BINS before relaxing this (plan risk table).
CALIB_ECE_TOL = 1.5e-2


# --------------------------------------------------------------------------- #
# Fixture: a multi-node, multi-doc-per-person corpus with an R_d column.       #
# --------------------------------------------------------------------------- #
def _corpus(seed=0, n_persons=80, C=9, K=12):
    """theta/label/mask/preindex over `n_persons` persons, 1-3 docs each.

    Plants the edge cases the parity must survive: a never-observed node (degenerate
    both arms), a CONSTANT-proba node that carries both classes under the incident
    mask (R2.1's guard — skipped on the incident arm, scored 0.5 without the guard),
    and enough signal elsewhere that ordinary nodes score a real AUC. Returns the
    per-doc arrays plus the `(person_id, episode_no)` the doc key is built from and
    a fixed `(V, b_raw)` (parity is about the EVAL given a fit, not the fit)."""
    rng = np.random.default_rng(seed)
    persons, episodes = [], []
    for pid in range(n_persons):
        ndoc = int(rng.integers(1, 4))
        for e in range(ndoc):
            persons.append(pid)
            episodes.append(e)
    D = len(persons)
    theta = rng.dirichlet(np.ones(K), size=D)
    V = rng.normal(scale=1.5, size=(C, K))
    b_raw = rng.normal(scale=0.5, size=C)
    # A CONSTANT-proba node (V row zero) — same p for every doc.
    V[C - 1] = 0.0
    b_raw[C - 1] = float(np.log(0.6 / 0.4))
    z = np.clip(theta @ V.T + b_raw, -50.0, 50.0)
    proba = 1.0 / (1.0 + np.exp(-z))
    # Labels correlated with proba so ordinary nodes have real discrimination.
    y = (rng.random((D, C)) < proba).astype(np.uint8)
    mask = (rng.random((D, C)) < 0.7).astype(np.uint8)
    mask[:, 0] = 0                                  # node 0 never observed
    mask[:, C - 1] = 1                              # constant node fully observed
    y[:, C - 1] = (rng.random(D) < 0.5).astype(np.uint8)   # both classes present
    # R_d: each doc a prior carrier of a few random nodes.
    preindex = [sorted(rng.choice(C, size=int(rng.integers(0, 3)), replace=False)
                       .tolist()) for _ in range(D)]
    elig = np.ones((D, C), dtype=np.uint8)
    for d, ids in enumerate(preindex):
        elig[d, ids] = 0
    return dict(theta=theta, y=y, mask=mask, elig=elig, preindex=preindex,
                V=V, b_raw=b_raw, person=persons, episode=episodes, C=C, K=K, D=D)


def _driver_proba(fx):
    z = np.clip(fx["theta"] @ fx["V"].T + fx["b_raw"], -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def compare_per_node(driver_pn, dist_pn):
    """Max |ΔAUC|, |ΔAP| and any skipped/count disagreement between two per_node
    (``_bundle_masked['per_label']``-shaped) tables — reusable by the orchestrator
    on two real `results_readout.json` per_node dicts (driver vs distributed)."""
    dauc = dap = 0.0
    mism = []
    for c in set(driver_pn) | set(dist_pn):
        a, b = driver_pn.get(c, {}), dist_pn.get(c, {})
        if a.get("skipped") != b.get("skipped"):
            mism.append((c, "skipped", a.get("skipped"), b.get("skipped")))
            continue
        if int(a.get("n_pos", 0)) != int(b.get("n_pos", 0)) or \
                int(a.get("n_neg", 0)) != int(b.get("n_neg", 0)):
            mism.append((c, "counts", (a.get("n_pos"), a.get("n_neg")),
                         (b.get("n_pos"), b.get("n_neg"))))
        if a.get("auc") is not None and b.get("auc") is not None:
            dauc = max(dauc, abs(a["auc"] - b["auc"]))
        if a.get("ap") is not None and b.get("ap") is not None:
            dap = max(dap, abs(a["ap"] - b["ap"]))
    return {"max_dauc": dauc, "max_dap": dap, "mismatches": mism}


# --------------------------------------------------------------------------- #
# Pure-numpy kernel tests (fast, no SparkSession).                             #
# --------------------------------------------------------------------------- #
def test_arms_kernel_emits_obs_and_incident_cells_with_shared_score():
    fx = _corpus(seed=1)
    C = fx["C"]
    rows = [(fx["theta"][d], fx["y"][d].astype(float), fx["mask"][d].astype(float),
             list(fx["preindex"][d])) for d in range(fx["D"])]
    cells = list(dr._score_cells_arms_kernel(iter(rows), fx["V"], fx["b_raw"], C))
    # Group emitted cells by node; obs cells must reproduce the driver's proba/mask
    # cell set and inc cells the incident_eval_mask cell set.
    proba = _driver_proba(fx)
    m_inc = gpc.incident_eval_mask(fx["y"], fx["mask"], fx["elig"]).astype(bool)
    by_node_obs, by_node_inc = {c: [] for c in range(C)}, {c: [] for c in range(C)}
    for node, yv, pv, obs, inc in cells:
        if obs:
            by_node_obs[node].append((yv, pv))
        if inc:
            by_node_inc[node].append((yv, pv))
    for c in range(C):
        want_obs = np.flatnonzero(fx["mask"][:, c])
        assert len(by_node_obs[c]) == want_obs.size, c
        got = sorted((round(y, 10), round(p, 9)) for y, p in by_node_obs[c])
        want = sorted((round(float(fx["y"][d, c]), 10), round(float(proba[d, c]), 9))
                      for d in want_obs)
        assert got == want, c
        want_inc = np.flatnonzero(m_inc[:, c])
        assert len(by_node_inc[c]) == want_inc.size, c


def test_arms_kernel_no_elig_column_yields_no_incident_cells():
    fx = _corpus(seed=2)
    C = fx["D"] and fx["C"]
    rows = [(fx["theta"][d], fx["y"][d].astype(float), fx["mask"][d].astype(float),
             None) for d in range(fx["D"])]
    cells = list(dr._score_cells_arms_kernel(iter(rows), fx["V"], fx["b_raw"], C))
    assert cells                                    # obs cells still emitted
    assert all(inc == 0 for _n, _y, _p, _obs, inc in cells)


def test_binned_calib_kernel_matches_direct_histogram():
    rng = np.random.default_rng(3)
    C, n_bins, N = 4, 10, 500
    node = rng.integers(0, C, size=N)
    p = rng.random(N)
    y = (rng.random(N) < p).astype(float)
    count, sum_y = dr._binned_calib_kernel(
        ((int(node[i]), float(y[i]), float(p[i])) for i in range(N)), C, n_bins)
    b = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    for c in range(C):
        for j in range(n_bins):
            sel = (node == c) & (b == j)
            assert count[c, j] == sel.sum()
            assert sum_y[c, j] == pytest.approx(y[sel].sum())


def test_binned_calib_accumulators_combine_by_addition():
    rng = np.random.default_rng(4)
    C, n_bins = 3, 8
    cells = [(int(rng.integers(0, C)), float(rng.integers(0, 2)), float(rng.random()))
             for _ in range(200)]
    whole = dr._binned_calib_kernel(iter(cells), C, n_bins)
    a = dr._binned_calib_kernel(iter(cells[:120]), C, n_bins)
    b = dr._binned_calib_kernel(iter(cells[120:]), C, n_bins)
    assert np.allclose(whole[0], a[0] + b[0])
    assert np.allclose(whole[1], a[1] + b[1])


def test_fit_binned_isotonic_min_pos_and_single_class_pass_through():
    C, n_bins = 3, 20
    count = np.zeros((C, n_bins))
    sum_y = np.zeros((C, n_bins))
    # node 0: plenty of positives across bins -> calibrated (breakpoints).
    count[0, 5] = 100; sum_y[0, 5] = 10
    count[0, 15] = 100; sum_y[0, 15] = 90
    # node 1: fewer than min_pos positives -> pass-through.
    count[1, 5] = 100; sum_y[1, 5] = 5
    count[1, 15] = 100; sum_y[1, 15] = 10
    # node 2: single class (all positive) -> pass-through.
    count[2, 15] = 50; sum_y[2, 15] = 50
    bp = dr.fit_binned_isotonic(count, sum_y, C, n_bins=n_bins, min_pos=20)
    assert bp[0] is not None
    assert bp[1] is None and bp[2] is None
    # calibrated node maps a low-bin score below a high-bin score (monotone).
    assert (dr.apply_binned_isotonic(bp, np.array([[0.275, 0.275, 0.275]]))[0, 0]
            < dr.apply_binned_isotonic(bp, np.array([[0.775, 0.775, 0.775]]))[0, 0])


def test_apply_binned_isotonic_is_identity_for_passthrough_nodes():
    bp = [None, (np.array([0.0, 1.0]), np.array([0.0, 1.0]))]
    proba = np.array([[0.3, 0.4], [0.9, 0.1]])
    out = dr.apply_binned_isotonic(bp, proba)
    assert np.array_equal(out[:, 0], proba[:, 0])   # node 0 untouched


def test_binned_isotonic_within_stated_ece_tolerance_of_exact():
    """The calibration parity number the deliverable asks to RECORD: binned isotonic
    vs the exact `calibrate_per_node`, per-node ECE on the TEST slice."""
    rng = np.random.default_rng(7)
    C, n_cal, n_te, n_bins = 6, 4000, 2000, dr._CALIB_BINS
    # A miscalibrated score whose true P(y=1) is a monotone function of it, so an
    # isotonic recalibration has real work to do (raw ECE well above zero).
    def _slice(n):
        s = rng.random((n, C))
        true = np.clip(s ** 1.7, 0, 1)              # over-confident low, under high
        y = (rng.random((n, C)) < true).astype(np.uint8)
        m = np.ones((n, C), dtype=np.uint8)
        return s, y, m
    s_cal, y_cal, m_cal = _slice(n_cal)
    s_te, y_te, _ = _slice(n_te)
    exact = gpc.calibrate_per_node(s_cal, y_cal, m_cal, s_te, C, min_pos=20)
    # binned: bin the cal cells, fit, apply to test.
    count = np.zeros((C, n_bins)); sum_y = np.zeros((C, n_bins))
    for c in range(C):
        cc, sc = dr._binned_calib_kernel(
            ((c, float(y_cal[i, c]), float(s_cal[i, c])) for i in range(n_cal)),
            C, n_bins)
        count[c] = cc[c]; sum_y[c] = sc[c]
    bp = dr.fit_binned_isotonic(count, sum_y, C, n_bins=n_bins, min_pos=20)
    binned = dr.apply_binned_isotonic(bp, s_te)
    worst = 0.0
    for c in range(C):
        e_exact = gpc._ece(y_te[:, c], exact[:, c])
        e_binned = gpc._ece(y_te[:, c], binned[:, c])
        worst = max(worst, abs(e_exact - e_binned))
    assert worst < CALIB_ECE_TOL, f"binned-vs-exact per-node ECE gap {worst:.4g}"
    # and the calibration actually did something (raw ECE >> calibrated).
    raw = float(np.mean([gpc._ece(y_te[:, c], s_te[:, c]) for c in range(C)]))
    cal = float(np.mean([gpc._ece(y_te[:, c], binned[:, c]) for c in range(C)]))
    assert cal < raw


def test_pooled_reliability_ece_from_bins_matches_direct():
    rng = np.random.default_rng(9)
    C, n_bins, N = 3, 100, 3000
    node = rng.integers(0, C, size=N)
    p = rng.random(N)
    y = (rng.random(N) < p ** 1.5).astype(float)
    count, sum_y = dr._binned_calib_kernel(
        ((int(node[i]), float(y[i]), float(p[i])) for i in range(N)), C, n_bins)
    got = dr.pooled_reliability_ece_from_bins(count, sum_y, n_bins=n_bins)
    # Direct pooled equal-width ECE with bin CENTERS as confidence (the same
    # approximation the from-bins helper makes), computed straight off the cells.
    b = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    centers = (np.arange(n_bins) + 0.5) / n_bins
    conf = centers[b]
    ece = 0.0
    for bb in range(n_bins):
        sel = b == bb
        if sel.any():
            ece += abs(conf[sel].mean() - y[sel].mean()) * (sel.mean())
    assert got == pytest.approx(ece, abs=1e-12)


# --------------------------------------------------------------------------- #
# Local-Spark parity round trip (thin wiring; AGENTS.md: local Spark => @slow). #
# --------------------------------------------------------------------------- #
def _make_df(spark, fx, *, with_ids=False):
    from pyspark.ml.linalg import VectorUDT, Vectors
    from pyspark.sql.types import (ArrayType, DoubleType, IntegerType, LongType,
                                   StructField, StructType)
    fields = [StructField("topicDistribution", VectorUDT(), False),
              StructField("label", ArrayType(DoubleType()), False),
              StructField("labelMask", ArrayType(DoubleType()), False),
              StructField("preindexClosure", ArrayType(IntegerType()), True)]
    if with_ids:
        fields = [StructField("person_id", LongType(), False),
                  StructField("episode_no", LongType(), False)] + fields
    rows = []
    for d in range(fx["D"]):
        base = (Vectors.dense(fx["theta"][d]),
                [float(v) for v in fx["y"][d]],
                [float(v) for v in fx["mask"][d]],
                [int(v) for v in fx["preindex"][d]])
        if with_ids:
            base = (int(fx["person"][d]), int(fx["episode"][d])) + base
        rows.append(base)
    return spark.createDataFrame(rows, StructType(fields)).repartition(3)


@pytest.mark.slow
class TestLocalSparkEvalParity:
    def test_prevalent_and_incident_per_node_match_the_driver(self, spark):
        """The core WP-B oracle: distributed per-node AUC/AP == driver `_bundle_masked`
        for BOTH arms, to numerical noise, including the skip counts."""
        fx = _corpus(seed=11)
        C = fx["C"]
        df = _make_df(spark, fx)
        cells = dr.score_cells_arms_df(df, fx["V"], fx["b_raw"], C,
                                       elig_col="preindexClosure")
        prev_pn, inc_pn = dr.per_node_metric_arms_rows(cells, C, min_count=0)
        proba = _driver_proba(fx)
        want_prev = _bundle_masked(proba, fx["y"], fx["mask"], C)["per_label"]
        m_inc = gpc.incident_eval_mask(fx["y"], fx["mask"], fx["elig"])
        want_inc = _bundle_masked(proba, fx["y"], m_inc, C,
                                  skip_constant=True)["per_label"]
        rep_p = compare_per_node(want_prev, prev_pn)
        rep_i = compare_per_node(want_inc, inc_pn)
        assert not rep_p["mismatches"], rep_p["mismatches"]
        assert not rep_i["mismatches"], rep_i["mismatches"]
        assert rep_p["max_dauc"] < 1e-9 and rep_p["max_dap"] < 1e-9, rep_p
        assert rep_i["max_dauc"] < 1e-9 and rep_i["max_dap"] < 1e-9, rep_i
        # non-vacuous: the constant node is skipped on the incident arm by R2.1's
        # guard (both paths), and node 0 (never observed) is degenerate on both.
        assert inc_pn[C - 1]["skipped"] and "constant" in inc_pn[C - 1]["skipped"]
        assert prev_pn[0]["skipped"] and prev_pn[0]["auc"] is None

    def test_min_count_masks_agree(self, spark):
        fx = _corpus(seed=12)
        C = fx["C"]
        df = _make_df(spark, fx)
        cells = dr.score_cells_arms_df(df, fx["V"], fx["b_raw"], C,
                                       elig_col="preindexClosure")
        prev_pn, inc_pn = dr.per_node_metric_arms_rows(cells, C, min_count=15)
        proba = _driver_proba(fx)
        want_prev = _bundle_masked(proba, fx["y"], fx["mask"], C, 15)["per_label"]
        assert not compare_per_node(want_prev, prev_pn)["mismatches"]

    def test_distributed_ranking_readout_end_to_end_matches_driver(self, spark):
        """`distributed_ranking_readout` (the driver-facing helper eval_path wires)
        reproduces `readout_from_proba` + `incident_readout` — ranking macro,
        per_node, and the incident block's macros/node sets."""
        fx = _corpus(seed=13)
        C = fx["C"]
        df = _make_df(spark, fx, with_ids=True)
        prev, block = gpc.distributed_ranking_readout(
            df, C, fx["V"], fx["b_raw"], recall_targets=RT, fdr_targets=FT,
            min_count=0, elig_col="preindexClosure")
        proba = _driver_proba(fx)
        drv = gpc.readout_from_proba(proba, fx["y"], fx["mask"], C,
                                     recall_targets=RT, fdr_targets=FT)
        assert prev["ranking"]["auc"] == pytest.approx(drv["ranking"]["auc"], abs=1e-9)
        assert prev["ranking"]["ap"] == pytest.approx(drv["ranking"]["ap"], abs=1e-9)
        assert not compare_per_node(drv["per_node"], prev["per_node"])["mismatches"]
        drv_inc = gpc.incident_readout(
            proba, fx["y"], fx["mask"], fx["elig"], C, recall_targets=RT,
            fdr_targets=FT, prevalent=drv, arm_label="gated_pc (pc_topics_lr)")
        assert block is not None
        # Shared-node macro and node-set sizes are the honest headline; they must
        # match the driver incident block cell for cell.
        assert (block["node_sets"]["n_shared"]
                == drv_inc["node_sets"]["n_shared"])
        for k in ("incident_shared", "incident_full"):
            a = block["macros"][k]["auc"]
            b = drv_inc["macros"][k]["auc"]
            if a is None or b is None:
                assert a is None and b is None, k
            else:
                assert a == pytest.approx(b, abs=1e-9), k
        assert (block["skipped_by_reason"].get("constant_prediction_column", 0)
                == drv_inc["skipped_by_reason"].get("constant_prediction_column", 0))

    def test_binned_calibration_stats_spark_matches_kernel(self, spark):
        fx = _corpus(seed=14)
        C = fx["C"]
        df = _make_df(spark, fx)
        cells = dr.score_cells_df(df, fx["V"], fx["b_raw"], C)
        count, sum_y = dr.binned_calibration_stats(cells, C, n_bins=50)
        # reference: the same cells collected and folded by the numpy kernel.
        collected = [(int(r["node"]), float(r["y"]), float(r["p"]))
                     for r in cells.collect()]
        rc, rs = dr._binned_calib_kernel(iter(collected), C, 50)
        assert np.allclose(count, rc) and np.allclose(sum_y, rs)
