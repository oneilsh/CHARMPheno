"""End-to-end gate for the DISTRIBUTED readout wired into the gated_pc driver
(Package C of `docs/superpowers/plans/2026-08-20-distributed-readout-plan.md`).

`tests/scripts/test_distributed_readout.py` covers the Spark seams in isolation and
`analysis/pc/tests/test_batched_lr.py` covers the solver against the sklearn oracle.
What is left — and what this file pins — is the JOIN: `gated_pc_cloud`'s
`distributed_score_arm` must produce the same readout as `score_arm` (the driver
path) when both are handed the SAME frozen θ, including the degenerate-node
fallback, which is the plan's open question ("must yield the same constant-
prediction fallback as `_lr_proba_per_label_masked`, so macro means stay
comparable"). The cluster A/B at C=444 is the full-scale version of exactly this
comparison; this is its miniature, so a formulation regression is caught in 20s
instead of a Dataproc run.

The same JOIN is pinned for the STANDALONE re-readout tool (`gated_pc_readout`,
the recovery path when a finished fit's readout output is lost): its `run_readout`
must route both arms through either path and agree with the driver readout on the
same frozen θ, and it must SKIP the co-fit head arm on the unsupervised mainline
(weightY=0 → the transform appends no `probability` column) instead of dying on the
missing column after the expensive arm has already been computed.

Tolerances (per-node AUC 2e-3, macro 1e-3) are set by the ORACLE, not by us:
sklearn's default `tol=1e-4` stops its own solver ~5e-4 from the optimum in
predicted probability, and it is the less-converged of the two parties. Asserting
tighter would be asserting on sklearn's stopping rule.
"""
import json
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
# PySpark workers are separate processes and inherit PYTHONPATH, not the driver's
# sys.path. BOTH entries are needed: REPO_ROOT for `analysis.pc.*`, and
# analysis/cloud for the TOP-LEVEL `distributed_readout` — which is the name
# gated_pc_cloud imports it under, and therefore the name cloudpickle stamps into
# every mapPartitions closure (on the cluster that name comes from --py-files; see
# scripts/run_experiment.py:build_spark_submit_cmd). Set at import time so it is in
# place before the session-scoped `spark` fixture builds the context.
os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in (str(REPO_ROOT), str(CLOUD), os.environ.get("PYTHONPATH", "")) if p)

import gated_pc_cloud as gpc  # noqa: E402
import gated_pc_readout as gpr  # noqa: E402

C, K, D_TR, D_TE = 6, 10, 200, 120
RECALL_TARGETS = [0.5, 0.9]
FDR_TARGETS = [0.25]


def _make_arrays(seed=0):
    """Frozen-θ readout problem with the node roles the real corpus has.

    Node 0 is the root (observed and positive for everyone — degenerate in TRAIN
    the way the real root is), node 1 is DEGENERATE-negative (never positive among
    its observed train rows, so the oracle refuses to fit and predicts 0.0), node 2
    is RARE (~15 train positives — the Q1 tail `readout_sample_frac` destroys), and
    nodes 3-5 are ordinary. Masks are sparse (~60%) so the per-node fits really do
    read different row sets, and node 5 is never observed in TEST (a skipped
    column, which must be skipped identically by both paths).
    """
    rng = np.random.default_rng(seed)
    Pi_tr = rng.dirichlet(np.full(K, 0.4), size=D_TR)
    Pi_te = rng.dirichlet(np.full(K, 0.4), size=D_TE)
    W = rng.standard_normal((C, K)) * 3.0
    b = rng.standard_normal(C) * 0.5

    def draw(P):
        z = P @ W.T + b
        return (rng.random(z.shape) < 1.0 / (1.0 + np.exp(-z))).astype(np.float64)

    y_tr, y_te = draw(Pi_tr), draw(Pi_te)
    m_tr = (rng.random((D_TR, C)) < 0.6).astype(np.float64)
    m_te = (rng.random((D_TE, C)) < 0.6).astype(np.float64)

    y_tr[:, 0] = 1.0                              # root: all-positive (degenerate)
    m_tr[:, 0] = 1.0
    y_te[:, 0] = (rng.random(D_TE) < 0.7).astype(np.float64)   # scoreable in test
    m_te[:, 0] = 1.0

    m_tr[:, 1] = 1.0                              # degenerate-negative node
    y_tr[:, 1] = 0.0

    # rare node: keep 15 train positives and whatever negatives exist (the true
    # logistic makes node 2 mostly positive, so the negatives are the scarce side).
    pos = np.flatnonzero(y_tr[:, 2] == 1.0)
    neg = np.flatnonzero(y_tr[:, 2] == 0.0)
    m_tr[:, 2] = 0.0
    m_tr[rng.choice(pos, size=15, replace=False), 2] = 1.0
    m_tr[rng.choice(neg, size=min(80, neg.size), replace=False), 2] = 1.0

    m_te[:, C - 1] = 0.0                          # never observed in test
    return Pi_tr, y_tr, m_tr, Pi_te, y_te, m_te


def _make_df(spark, Pi, y, mask, offset=0, parts=3, proba=None):
    """The frame a supervised `OnlinePCLDAModel.transform` produces. `proba` appends
    the co-fit head's `probability` column; leaving it None is the weightY=0 shape,
    where that column genuinely does not exist."""
    from pyspark.ml.linalg import VectorUDT, Vectors
    from pyspark.sql.types import (ArrayType, DoubleType, LongType, StructField,
                                   StructType)
    fields = [
        StructField("person_id", LongType(), False),
        StructField("topicDistribution", VectorUDT(), False),
        StructField("label", ArrayType(DoubleType()), False),
        StructField("labelMask", ArrayType(DoubleType()), False),
    ]
    if proba is not None:
        fields.append(StructField("probability", VectorUDT(), False))
    rows = [tuple([int(offset + d), Vectors.dense(Pi[d]),
                   [float(v) for v in y[d]], [float(v) for v in mask[d]]]
                  + ([Vectors.dense(proba[d])] if proba is not None else []))
            for d in range(Pi.shape[0])]
    return spark.createDataFrame(rows, StructType(fields)).repartition(parts)


@pytest.mark.slow
class TestDistributedArmMatchesDriverArm:
    """`distributed_score_arm` vs `score_arm` on identical frozen θ."""

    @pytest.fixture(scope="class")
    def both(self, spark):
        arrays = _make_arrays()
        Pi_tr, y_tr, m_tr, Pi_te, y_te, m_te = arrays
        train_df = _make_df(spark, Pi_tr, y_tr, m_tr)
        test_df = _make_df(spark, Pi_te, y_te, m_te, offset=10_000)
        dist = gpc.distributed_score_arm(
            train_df, test_df, C, K, recall_targets=RECALL_TARGETS,
            fdr_targets=FDR_TARGETS, min_count=0, label="test-arm")
        drv = gpc.score_arm(Pi_tr, y_tr, m_tr, Pi_te, y_te, m_te, C,
                            recall_targets=RECALL_TARGETS, fdr_targets=FDR_TARGETS)
        return arrays, dist, drv

    def test_lean_bundle_dtypes_and_labels(self, both):
        """The lean collect is the memory claim: float32 proba + uint8 y/mask, and
        it must reproduce the label/mask arrays the driver collect returns."""
        (_, y_tr, _, _, y_te, m_te), dist, _ = both
        _, proba, y_got, m_got, persons = dist
        assert proba.dtype == np.float32
        assert y_got.dtype == np.uint8 and m_got.dtype == np.uint8
        assert proba.shape == (D_TE, C)
        # Rows come back in partition order, so align by person_id before comparing.
        order = np.argsort(np.asarray(persons))
        assert np.array_equal(y_got[order], y_te.astype(np.uint8))
        assert np.array_equal(m_got[order], m_te.astype(np.uint8))

    def test_degenerate_nodes_get_the_oracle_constant(self, both):
        """Nodes 0 (all-positive) and 1 (all-negative) among their observed TRAIN
        rows are exactly the case `_lr_proba_per_label_masked` refuses to fit; both
        paths must emit the lone class value, bit for bit."""
        _, dist, _ = both
        _, proba, _, _, _ = dist
        assert np.all(proba[:, 0] == np.float32(1.0))
        assert np.all(proba[:, 1] == np.float32(0.0))

    def test_per_node_auc_matches_driver_readout(self, both):
        _, dist, drv = both
        r_dist = dist[0]
        pd_, pv = r_dist["per_node"], drv["per_node"]
        assert set(pd_) == set(pv), "the two paths scored different node sets"
        assert len(pd_) >= 3
        for c in pd_:
            assert abs(pd_[c]["auc"] - pv[c]["auc"]) < 2e-3, c
            assert abs(pd_[c]["ap"] - pv[c]["ap"]) < 5e-3, c
            assert pd_[c]["n_pos"] == pv[c]["n_pos"], c

    def test_macro_and_skip_bookkeeping_match(self, both):
        _, dist, drv = both
        a, b = dist[0]["ranking"], drv["ranking"]
        assert a["n_labels_scored"] == b["n_labels_scored"]
        assert a["n_labels_skipped"] == b["n_labels_skipped"]
        assert abs(a["auc"] - b["auc"]) < 1e-3
        assert abs(a["ap"] - b["ap"]) < 1e-3
        # The rest of the readout stack rides on the same proba, so check one
        # operating-point and the detection head too (both consume the float32
        # matrix directly — this is the dtype-agnostic claim, exercised).
        assert abs(dist[0]["pr"]["par"][0.9] - drv["pr"]["par"][0.9]) < 5e-3
        assert abs(dist[0]["detection"]["auc"] - drv["detection"]["auc"]) < 2e-3

    def test_degenerate_node_scored_identically_downstream(self, both):
        """A constant column yields AUC exactly 0.5; the point is that BOTH paths
        put node 0 in the macro with the same value rather than one skipping it."""
        _, dist, drv = both
        assert dist[0]["per_node"][0]["auc"] == pytest.approx(0.5, abs=1e-12)
        assert drv["per_node"][0]["auc"] == pytest.approx(0.5, abs=1e-12)


@pytest.mark.slow
def test_ab_harness_runs_both_paths_and_reports(spark, capsys):
    """The cardiovascular gate's machinery, in miniature: it must run BOTH paths and
    print the deltas without asserting (a REPORT is the deliverable — the run is
    diagnostic, and a hard assert on sklearn's stopping rule would be a false gate)."""
    Pi_tr, y_tr, m_tr, Pi_te, y_te, m_te = _make_arrays(seed=3)
    train_df = _make_df(spark, Pi_tr, y_tr, m_tr)
    test_df = _make_df(spark, Pi_te, y_te, m_te, offset=10_000)
    out = gpc.readout_ab_report(
        train_df, test_df, C, K, recall_targets=RECALL_TARGETS,
        fdr_targets=FDR_TARGETS, min_count=0, label="ab-test", seed=7)
    txt = capsys.readouterr().out
    assert "A/B readout equality gate" in txt
    assert "per-node |ΔAUC|" in txt and "max |Δp|" in txt
    assert abs(out["distributed"]["ranking"]["auc"]
               - out["driver"]["ranking"]["auc"]) < 1e-3


@pytest.mark.slow
def test_lean_head_proba_collect_matches_driver_collect(spark):
    """The co-fit head's `probability` column needs the same lean treatment (no
    L-BFGS involved) — `_collect_lean_proba(score_col="probability")` must equal
    `_collect_head_proba` cell for cell, up to float32 rounding."""
    from pyspark.ml.linalg import VectorUDT, Vectors
    from pyspark.sql.types import (ArrayType, DoubleType, LongType, StructField,
                                   StructType)
    rng = np.random.default_rng(5)
    p = rng.random((40, C))
    y = (rng.random((40, C)) < 0.4).astype(np.float64)
    m = (rng.random((40, C)) < 0.7).astype(np.float64)
    schema = StructType([
        StructField("person_id", LongType(), False),
        StructField("probability", VectorUDT(), False),
        StructField("label", ArrayType(DoubleType()), False),
        StructField("labelMask", ArrayType(DoubleType()), False),
    ])
    df = spark.createDataFrame(
        [(int(d), Vectors.dense(p[d]), [float(v) for v in y[d]],
          [float(v) for v in m[d]]) for d in range(40)], schema).repartition(3)

    lean_p, lean_y, lean_m, persons = gpc._collect_lean_proba(
        df, C, score_col="probability")
    order = np.argsort(np.asarray(persons))
    assert lean_p.dtype == np.float32
    assert np.allclose(lean_p[order], p, atol=1e-6, rtol=0)
    assert np.array_equal(lean_y[order], y.astype(np.uint8))
    assert np.array_equal(lean_m[order], m.astype(np.uint8))


@pytest.mark.slow
class TestReReadoutBothModes:
    """`gated_pc_readout.run_readout` — the re-readout of a FINISHED fit — in both
    modes, against the driver `score_arm` oracle on the same frozen θ."""

    @pytest.fixture(scope="class")
    def scored(self, spark):
        """What a finished fit's supervised transform leaves behind: θ + labels +
        the co-fit head's own `probability` column (stand-in values — the head arm
        reads that column, it never refits it)."""
        arrays = _make_arrays(seed=1)
        Pi_tr, y_tr, m_tr, Pi_te, y_te, m_te = arrays
        rng = np.random.default_rng(11)
        train_df = _make_df(spark, Pi_tr, y_tr, m_tr, proba=rng.random((D_TR, C)))
        test_df = _make_df(spark, Pi_te, y_te, m_te, offset=10_000,
                           proba=rng.random((D_TE, C)))
        return arrays, train_df, test_df

    @pytest.fixture(scope="class")
    def both(self, scored, tmp_path_factory):
        _, train_df, test_df = scored
        out = tmp_path_factory.mktemp("run_dir")
        manifest = {"C": C, "K": K, "weight_y": 1.0}
        dist = gpr.run_readout(
            train_df, test_df, manifest, recall_targets=RECALL_TARGETS,
            fdr_targets=FDR_TARGETS, min_count=0, readout_mode="distributed",
            out_dir=out)
        drv = gpr.run_readout(
            train_df, test_df, manifest, recall_targets=RECALL_TARGETS,
            fdr_targets=FDR_TARGETS, min_count=0, readout_mode="driver")
        return dist, drv, out

    def test_distributed_arm_matches_driver_arm(self, both, scored):
        """The recovery path's whole promise: re-reading distributed gives the same
        per-node numbers the driver path would have printed."""
        dist, drv, _ = both
        Pi_tr, y_tr, m_tr, Pi_te, y_te, m_te = scored[0]
        oracle = gpc.score_arm(Pi_tr, y_tr, m_tr, Pi_te, y_te, m_te, C,
                               recall_targets=RECALL_TARGETS,
                               fdr_targets=FDR_TARGETS)
        pd_, pv = dist["gated_pc"]["per_node"], drv["gated_pc"]["per_node"]
        assert set(pd_) == set(pv) == set(oracle["per_node"])
        assert len(pd_) >= 3
        for c in pd_:
            assert pv[c]["auc"] == oracle["per_node"][c]["auc"], c   # same call
            assert abs(pd_[c]["auc"] - oracle["per_node"][c]["auc"]) < 2e-3, c
        assert abs(dist["gated_pc"]["ranking"]["auc"]
                   - oracle["ranking"]["auc"]) < 1e-3

    def test_head_arm_present_and_mode_agnostic(self, both):
        """The co-fit head arm reads the `probability` column as-is (no LR), so the
        lean collector and the driver collector must give the identical readout."""
        dist, drv, _ = both
        assert dist["gated_pc_head"]["ranking"]["auc"] == pytest.approx(
            drv["gated_pc_head"]["ranking"]["auc"], abs=1e-6)

    def test_results_written_without_clobbering_the_fits_record(self, both):
        """Durability is the point of the re-run (exp 0103 lost its readout to a
        terminal): every arm lands in results_readout.json — and NOT in the fit's
        own results_partial.json."""
        dist, _, out = both
        got = json.loads((out / "results_readout.json").read_text())
        assert set(got) == {"gated_pc", "gated_pc_head"} == set(dist)
        assert got["gated_pc"]["ranking"]["auc"] == pytest.approx(
            dist["gated_pc"]["ranking"]["auc"])
        assert not (out / "results_partial.json").exists()


@pytest.mark.slow
@pytest.mark.parametrize("weight_y,with_proba", [(0.0, True), (1.0, False)])
def test_head_arm_skipped_when_there_is_no_co_fit_head(spark, capsys, weight_y,
                                                       with_proba):
    """The scaled-back mainline is an UNSUPERVISED gate + post-hoc readout: weightY=0
    means `transform` appends no `probability` column at all. EITHER witness alone
    must skip the head arm with a log line — the manifest's weight_y=0 (here with a
    column present, the belt-and-braces case) and the missing column (which covers a
    manifest too old to record weight_y). Before this the re-readout raised an
    AnalysisException on the `probability` select, AFTER paying for the pc_topics_lr
    arm."""
    Pi_tr, y_tr, m_tr, Pi_te, y_te, m_te = _make_arrays(seed=2)
    rng = np.random.default_rng(4)

    def _p(n):
        return rng.random((n, C)) if with_proba else None

    train_df = _make_df(spark, Pi_tr, y_tr, m_tr, proba=_p(D_TR))
    test_df = _make_df(spark, Pi_te, y_te, m_te, offset=10_000, proba=_p(D_TE))
    got = gpr.run_readout(
        train_df, test_df, {"C": C, "K": K, "weight_y": weight_y},
        recall_targets=RECALL_TARGETS, fdr_targets=FDR_TARGETS, min_count=0,
        readout_mode="driver")
    assert set(got) == {"gated_pc"}, "the head arm must not be scored"
    assert "co-fit head arm SKIPPED" in capsys.readouterr().out


@pytest.mark.slow
def test_re_readout_ab_check_runs_the_gate(spark, capsys):
    """`--readout-ab-check` on the re-readout is the same report the fit driver
    prints, on re-transformed frames: it must run BOTH paths and print the deltas
    (the recovery run is often the only place the gate can still be run — the fit is
    hours, the re-readout is minutes).

    The manifest deliberately omits K, which is also the K-fallback path: a run old
    enough not to record it must still route distributed, off the θ width."""
    Pi_tr, y_tr, m_tr, Pi_te, y_te, m_te = _make_arrays(seed=5)
    train_df = _make_df(spark, Pi_tr, y_tr, m_tr)
    test_df = _make_df(spark, Pi_te, y_te, m_te, offset=10_000)
    gpr.run_readout(train_df, test_df, {"C": C, "weight_y": 0.0},
                    recall_targets=RECALL_TARGETS, fdr_targets=FDR_TARGETS,
                    min_count=0, readout_mode="distributed", ab_check=True)
    txt = capsys.readouterr().out
    assert "A/B readout equality gate" in txt and "per-node |ΔAUC|" in txt


def test_readout_tool_parses_the_mode_flags():
    """Same flag names/defaults as the fit driver, so a re-run of a lost readout is
    the fit's command with the same knobs."""
    a = gpr.build_parser().parse_args(["--run-dir", "/tmp/r"])
    assert a.readout_mode == "auto" and a.readout_ab_check is False
    b = gpr.build_parser().parse_args(
        ["--run-dir", "/tmp/r", "--readout-mode", "distributed",
         "--readout-ab-check"])
    assert b.readout_mode == "distributed" and b.readout_ab_check is True


def test_resolve_readout_mode_auto_threshold():
    """`auto` keeps every cardiovascular-scale run (C=444) on the byte-identical
    driver path and routes whole-Mondo (C~3,300) to the distributed fit."""
    assert gpc.resolve_readout_mode("auto", 444) == "driver"
    assert gpc.resolve_readout_mode("auto", 500) == "driver"
    assert gpc.resolve_readout_mode("auto", 501) == "distributed"
    assert gpc.resolve_readout_mode("auto", 3300) == "distributed"
    assert gpc.resolve_readout_mode("driver", 3300) == "driver"
    assert gpc.resolve_readout_mode("distributed", 10) == "distributed"
    with pytest.raises(ValueError):
        gpc.resolve_readout_mode("nope", 10)


def test_parse_args_readout_flags():
    a = gpc.parse_args(["--cdr", "x", "--billing", "y", "--out-dir", "/tmp/o"])
    assert a.readout_mode == "auto" and a.readout_ab_check is False
    b = gpc.parse_args(["--cdr", "x", "--billing", "y", "--out-dir", "/tmp/o",
                        "--readout-mode", "distributed", "--readout-ab-check"])
    assert b.readout_mode == "distributed" and b.readout_ab_check is True


def test_densify_lean_blocks_handles_dense_mask_marker():
    """`m_idx is None` is the all-ones-mask marker (`--label-mask-mode full`), and
    the densifier must expand it rather than leaving the mask empty."""
    P = np.arange(6, dtype=np.float32).reshape(2, 3)
    blocks = [(np.array([7, 8], dtype=np.int64), P,
               np.array([0, 2], dtype=np.int32), np.array([0, 1, 2], dtype=np.int64),
               None, None)]
    proba, y, mask, ids = gpc._densify_lean_blocks(blocks, 3)
    assert np.array_equal(proba, P)
    assert np.array_equal(y, np.array([[1, 0, 0], [0, 0, 1]], dtype=np.uint8))
    assert np.array_equal(mask, np.ones((2, 3), dtype=np.uint8))
    assert ids == [7, 8]


def test_stdout_tee_appends_sanitized_lines_and_survives_truncation(tmp_path, capsys):
    """The durable driver log exists because exp 0103 lost 4h of results to a
    summary.md that came back empty — appends riding a long-lived handle on a
    replaced/truncated file land on an unlinked inode. The tee must (a) commit
    complete lines only, (b) drop the wrapper's patient/noise patterns, and
    (c) keep appending to a file that was truncated mid-run, which is the
    per-line-reopen property the whole design hangs on."""
    import io
    from _driver_common import _StdoutTee

    log = tmp_path / "driver_log.md"
    real = io.StringIO()
    tee = _StdoutTee(log, real)
    tee.write("[driver] part")           # incomplete: nothing committed yet
    assert not log.exists() or log.read_text() == ""
    tee.write("ial line\n")
    tee.write("person_id=12345 secret\n")     # patient pattern: dropped
    tee.write("26/08/21 01:02:03 INFO Noise: x\n")  # log4j noise: dropped
    log.write_text("")                   # simulate mid-run truncation
    tee.write("[driver] after truncation\n")
    assert log.read_text() == "[driver] after truncation\n"
    assert "[driver] partial line" in real.getvalue()
    assert "secret" in real.getvalue()   # terminal still sees everything
