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

The WARM-START flow is pinned here for the same reason: the calibration split's
second batched solve starts from the arm's main fit (and the A/B gate's sampled
refit from the full-data one), which must change the wall-clock and nothing else —
same per-node readout, same degenerate no-ops.

The FIT-ONLY SAVE is pinned here for a related reason: the readout is where these
runs die, and until it landed a readout death also destroyed the fit that preceded
it. `_save_fit` is unit-tested (both writes, the multi-domain λ branch) and its
ORDERING inside `main` — after the fit, before any readout work — is pinned
structurally, the same way the calibration gate is, because `main` cannot be
called without a BigQuery corpus.

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
import run_experiment as rex  # noqa: E402

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


def _make_df(spark, Pi, y, mask, offset=0, parts=3, proba=None, preindex=None):
    """The frame a supervised `OnlinePCLDAModel.transform` produces. `proba` appends
    the co-fit head's `probability` column; leaving it None is the weightY=0 shape,
    where that column genuinely does not exist. `preindex` appends E1's sparse
    `preindexClosure` (`array<int>` of the engine ids the doc already carried before
    its index), which is what the incident arm's fourth CSR run reads."""
    from pyspark.ml.linalg import VectorUDT, Vectors
    from pyspark.sql.types import (ArrayType, DoubleType, IntegerType, LongType,
                                   StructField, StructType)
    fields = [
        StructField("person_id", LongType(), False),
        StructField("topicDistribution", VectorUDT(), False),
        StructField("label", ArrayType(DoubleType()), False),
        StructField("labelMask", ArrayType(DoubleType()), False),
    ]
    if proba is not None:
        fields.append(StructField("probability", VectorUDT(), False))
    if preindex is not None:
        fields.append(StructField("preindexClosure", ArrayType(IntegerType()),
                                  False))
    rows = [tuple([int(offset + d), Vectors.dense(Pi[d]),
                   [float(v) for v in y[d]], [float(v) for v in mask[d]]]
                  + ([Vectors.dense(proba[d])] if proba is not None else [])
                  + ([[int(v) for v in preindex[d]]] if preindex is not None
                     else []))
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
        _, proba, y_got, m_got, persons = dist[:5]
        V, b_raw = dist[5]                        # the fit params, for warm starts
        assert V.shape == (C, K) and b_raw.shape == (C,)
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
        _, proba, _, _, _ = dist[:5]
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
def test_warm_started_calibration_fit_matches_the_cold_one(spark, capsys):
    """The driver's calibration flow, warm vs cold, on the same 75% hash split.

    The isotonic calibrator needs an OUT-OF-SAMPLE fit, so a supervised arm pays a
    SECOND batched solve on 75% of the rows the main fit already converged on.
    Warm-starting it from the main fit's raw-θ params is a wall-clock device and
    must be nothing else, which is what this pins end to end through Spark:

      - same readout. Both solves run to `gtol` on the same convex problems, so
        per-node AUC must agree to the tolerance the whole file uses (2e-3, set by
        sklearn's own stopping rule);
      - same degenerate handling. A node can be single-class in the 75% split
        without being single-class overall, and the warm start must not resurrect
        one: the fit masks it, so its `x0` row is zeroed and it freezes at
        iteration 0 with all-zero params and the oracle's constant;
      - and it must pay under a CAP, which is the regime it exists for — at
        `max_iter=3` the warm fit is several times closer to the converged answer
        in predicted probability than the cold one.

    Deliberately NOT asserted: that the warm solve reaches `gtol` in fewer
    iterations. Measured on this fixture it does not (16 vs 13 iterations, 23 vs
    14 passes) — L-BFGS's endgame is governed by the curvature history it built,
    not by where it started, so the payoff is the capped-budget iterate. See the
    "what it does NOT buy" section of `solve_batched_lr`'s docstring.
    """
    from pyspark.sql import functions as F

    Pi_tr, y_tr, m_tr, Pi_te, y_te, m_te = _make_arrays(seed=4)
    train_df = _make_df(spark, Pi_tr, y_tr, m_tr).cache()
    test_df = _make_df(spark, Pi_te, y_te, m_te, offset=10_000).cache()
    # the driver's own split: a hash of person_id, deterministic and complementary
    h = F.pmod(F.hash(F.col("person_id"), F.lit(0)), F.lit(4))
    fit_df = train_df.filter(h != 0).cache()

    V, b_raw, _, _, _ = gpc._fit_readout_heads(train_df, C, K, label="main-fit")
    cold = gpc._fit_readout_heads(fit_df, C, K, label="calibration-fit cold")
    warm = gpc._fit_readout_heads(fit_df, C, K, label="calibration-fit warm",
                                  warm_start=(V, b_raw))
    assert "(warm start)" in capsys.readouterr().out

    assert cold[4]["warm_started"] is False and warm[4]["warm_started"] is True
    assert np.array_equal(cold[3], warm[3]), "same rows must give the same mask"

    def _proba(fit, max_iter=None):
        p, y, m, _, _ = gpc._collect_lean_proba(
            test_df, C, fit[0], fit[1], degenerate=fit[3], const=fit[2])
        return p, gpc.readout_from_proba(p, y, m, C, recall_targets=RECALL_TARGETS,
                                         fdr_targets=FDR_TARGETS, min_count=0)

    p_cold, r_cold = _proba(cold)
    p_warm, r_warm = _proba(warm)
    assert set(r_cold["per_node"]) == set(r_warm["per_node"])
    assert len(r_cold["per_node"]) >= 3
    for c in r_cold["per_node"]:
        assert abs(r_cold["per_node"][c]["auc"]
                   - r_warm["per_node"][c]["auc"]) < 2e-3, c

    # the degenerate nodes of THIS fit are untouched by the warm start
    deg = np.asarray(warm[3], dtype=bool)
    assert deg.any(), "fixture must keep at least one degenerate node"
    assert np.all(warm[0][deg] == 0.0) and np.all(warm[1][deg] == 0.0)
    assert (warm[4]["n_iter"][deg] == 0).all()
    assert np.array_equal(p_warm[:, deg], p_cold[:, deg])

    # ...and the cap is where it pays: 3 iterations, same budget, both paths.
    cap_cold = gpc._fit_readout_heads(fit_df, C, K, max_iter=3,
                                      label="calibration-fit cold@3")
    cap_warm = gpc._fit_readout_heads(fit_df, C, K, max_iter=3, warm_start=(V, b_raw),
                                      label="calibration-fit warm@3")
    d_cold = np.abs(_proba(cap_cold)[0].astype(np.float64) - p_cold).max()
    d_warm = np.abs(_proba(cap_warm)[0].astype(np.float64) - p_cold).max()
    assert d_warm < d_cold, f"capped warm={d_warm:.3e} not better than cold={d_cold:.3e}"
    for df in (train_df, test_df, fit_df):
        df.unpersist()


@pytest.mark.slow
def test_ab_gate_warm_starts_its_sampled_fit_from_the_full_fit(spark, capsys):
    """At `sample_frac<1` the passed-in full-data result is dropped — but not its fit.

    The gate has to compare two solvers on ONE dataset, so a result fit on all the
    rows cannot serve as its distributed side and the report refits on the sample.
    That refit is the same C convex problems on a subset of their rows, so the
    full-data `(V, b_raw)` is exactly what it should start from; the plumbing has
    to lift the params out BEFORE it drops the result, which is what this pins.
    """
    Pi_tr, y_tr, m_tr, Pi_te, y_te, m_te = _make_arrays(seed=6)
    train_df = _make_df(spark, Pi_tr, y_tr, m_tr).cache()
    test_df = _make_df(spark, Pi_te, y_te, m_te, offset=10_000).cache()
    dist = gpc.distributed_score_arm(
        train_df, test_df, C, K, recall_targets=RECALL_TARGETS,
        fdr_targets=FDR_TARGETS, min_count=0, label="ab-warm")
    capsys.readouterr()
    gpc.readout_ab_report(
        train_df, test_df, C, K, recall_targets=RECALL_TARGETS,
        fdr_targets=FDR_TARGETS, min_count=0, label="ab-warm", seed=11,
        sample_frac=0.7, distributed=dist)
    txt = capsys.readouterr().out
    # WP-A2 (R5.5): the sample is doc-key hashed, not `DataFrame.sample()`'s
    # row-position draw — see gated_pc_cloud.readout_ab_report's docstring.
    assert "restricted to the SAME 0.7 doc-key sample" in txt
    assert "(warm start)" in txt, "the sampled refit must start from the full fit"
    assert "A/B readout equality gate" in txt
    train_df.unpersist(); test_df.unpersist()


@pytest.mark.slow
def test_theta_topm_equal_to_K_is_the_identity_truncation(spark):
    """`readout_theta_topm=K` keeps every entry, so the sparse path must reproduce
    the dense path EXACTLY — same fit, same probabilities, same readout.

    This is the null A/B for the whole lever: it isolates the sparse machinery
    (packing, chunking, by-node regrouping, bincount gradients) from the truncation
    itself, so a failure here is a plumbing bug and never a modelling question.
    """
    Pi_tr, y_tr, m_tr, Pi_te, y_te, m_te = _make_arrays(seed=8)
    train_df = _make_df(spark, Pi_tr, y_tr, m_tr).cache()
    test_df = _make_df(spark, Pi_te, y_te, m_te, offset=10_000).cache()
    kw = dict(recall_targets=RECALL_TARGETS, fdr_targets=FDR_TARGETS, min_count=0)
    full = gpc.distributed_score_arm(train_df, test_df, C, K, label="full", **kw)
    trunc = gpc.distributed_score_arm(train_df, test_df, C, K, label="topm=K",
                                      theta_topm=K, **kw)
    assert np.array_equal(trunc[1], full[1]), "identical proba, bit for bit"
    assert np.allclose(trunc[5][0], full[5][0], atol=1e-12, rtol=0)
    assert np.allclose(trunc[5][1], full[5][1], atol=1e-12, rtol=0)
    for c in full[0]["per_node"]:
        assert trunc[0]["per_node"][c]["auc"] == full[0]["per_node"][c]["auc"], c
    assert trunc[0]["ranking"]["auc"] == full[0]["ranking"]["auc"]
    train_df.unpersist(); test_df.unpersist()


@pytest.mark.slow
def test_theta_topm_on_concentrated_theta_keeps_the_readout(spark, capsys):
    """Top-m on CONCENTRATED θ: the readout survives, and how well is a property of
    the DATA, not of the code.

    The fixture draws Dirichlet(0.05) rows — the sparse regime the whole-Mondo
    Dirichlet(0.5)-over-3,827-topics posterior mean is claimed to be in — so a
    truncation that keeps most of the mass should move per-node AUC by little. The
    tolerance here is DELIBERATELY loose and is not a correctness gate: exactness is
    pinned against the dense-on-truncated oracle in
    `test_distributed_readout.py`, and how much AUC a given m costs on a given corpus
    is exactly what `theta_topm_coverage` exists to report before anyone sets the
    flag. Assert tightly here and the test would be asserting on a Dirichlet draw.
    """
    rng = np.random.default_rng(21)
    K_wide, topm = 40, 6
    Pi_tr = rng.dirichlet(np.full(K_wide, 0.05), size=D_TR)
    Pi_te = rng.dirichlet(np.full(K_wide, 0.05), size=D_TE)
    W = rng.standard_normal((C, K_wide)) * 3.0
    b = rng.standard_normal(C) * 0.5

    def draw(P):
        z = P @ W.T + b
        return (rng.random(z.shape) < 1.0 / (1.0 + np.exp(-z))).astype(np.float64)

    y_tr, y_te = draw(Pi_tr), draw(Pi_te)
    m_tr = np.ones((D_TR, C))
    m_te = np.ones((D_TE, C))
    train_df = _make_df(spark, Pi_tr, y_tr, m_tr).cache()
    test_df = _make_df(spark, Pi_te, y_te, m_te, offset=10_000).cache()
    kw = dict(recall_targets=RECALL_TARGETS, fdr_targets=FDR_TARGETS, min_count=0)

    # the measurement that justifies the setting, on the same frame
    cov = _dr_module().theta_topm_coverage(train_df, K_wide, ms=(topm, K_wide))
    assert cov[K_wide][0] == pytest.approx(1.0)
    assert cov[topm][0] > 0.5, "fixture must actually be concentrated"

    full = gpc.distributed_score_arm(train_df, test_df, C, K_wide, label="full", **kw)
    trunc = gpc.distributed_score_arm(train_df, test_df, C, K_wide,
                                      label="topm", theta_topm=topm, **kw)
    assert "(theta top-m=6)" in capsys.readouterr().out
    shared = sorted(set(full[0]["per_node"]) & set(trunc[0]["per_node"]))
    assert len(shared) >= 3
    d_auc = np.array([abs(trunc[0]["per_node"][c]["auc"]
                          - full[0]["per_node"][c]["auc"]) for c in shared])
    assert d_auc.max() < 0.15, f"top-{topm} cost more AUC than the mass it dropped"
    train_df.unpersist(); test_df.unpersist()


@pytest.mark.slow
def test_coverage_line_is_logged_only_for_large_fits(spark, capsys, monkeypatch):
    """The measurement is always on WHERE IT MATTERS and nowhere else.

    Small fits (every cardiovascular-scale run) must not pay an extra data pass for a
    diagnostic about a lever nobody is pulling; whole-Mondo-scale fits must print the
    coverage whether or not `theta_topm` is set, so the run of record carries the
    evidence for — or against — the setting it used.
    """
    Pi_tr, y_tr, m_tr, _, _, _ = _make_arrays(seed=9)
    train_df = _make_df(spark, Pi_tr, y_tr, m_tr).cache()
    gpc._fit_readout_heads(train_df, C, K, label="small", max_iter=2)
    assert "theta top-m mass" not in capsys.readouterr().out

    monkeypatch.setattr(gpc, "_COVERAGE_MIN_FIT_BYTES", 0)
    gpc._fit_readout_heads(train_df, C, K, label="big", max_iter=2)
    txt = capsys.readouterr().out
    assert "big: theta top-m mass:" in txt
    line = next(ln for ln in txt.splitlines() if "theta top-m mass" in ln)
    assert all(f"m={m}:" in line for m in (64, 128, 256, 512))
    assert line.rstrip().endswith("(mean/p10)")
    # K=10 here, so every m keeps all the mass — the identity end of the scale
    assert "m=64:1.000/0.999" in line
    train_df.unpersist()


def _dr_module():
    """The partition-kernel module under the TOP-LEVEL name the driver imports it
    as (see this file's PYTHONPATH note)."""
    import distributed_readout

    return distributed_readout


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

    lean_p, lean_y, lean_m, persons, _ = gpc._collect_lean_proba(
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
        # FLAG-OFF (spec E2 acceptance): a corpus without E1's column produces
        # exactly the two arms it always did — no incident block, no changed
        # numbers. This is the regression that protects the 0104/0109 controls.
        assert set(got) == {"gated_pc", "gated_pc_head"} == set(dist)
        assert got["gated_pc"]["ranking"]["auc"] == pytest.approx(
            dist["gated_pc"]["ranking"]["auc"])
        assert not (out / "results_partial.json").exists()


@pytest.mark.slow
def test_re_readout_emits_the_incident_block_from_a_saved_fit(spark, tmp_path,
                                                              capsys):
    """WP4's actual delivery route: `gated_pc_readout` over a corpus carrying E1's
    column produces `gated_pc_incident` beside the prevalent arms, with no re-fit.

    Also the flag-off half of the acceptance, on the SAME frames: dropping
    `elig_col` leaves the prevalent numbers byte-identical, so the incident arm is
    provably an addition rather than a perturbation.
    """
    Pi_tr, y_tr, m_tr, Pi_te, y_te, m_te = _make_arrays(seed=5)
    rng = np.random.default_rng(6)
    # Prior carriers: a third of the docs already carried node 3 pre-index, so the
    # incident arm scores a strictly smaller (and different) cohort there.
    pre_tr = [[3] if d % 3 == 0 else [] for d in range(D_TR)]
    pre_te = [[3] if d % 3 == 0 else [] for d in range(D_TE)]
    train_df = _make_df(spark, Pi_tr, y_tr, m_tr, proba=rng.random((D_TR, C)),
                        preindex=pre_tr)
    test_df = _make_df(spark, Pi_te, y_te, m_te, offset=10_000,
                       proba=rng.random((D_TE, C)), preindex=pre_te)
    manifest = {"C": C, "K": K, "weight_y": 1.0}
    out = tmp_path / "run"
    out.mkdir()
    on = gpr.run_readout(train_df, test_df, manifest,
                         recall_targets=RECALL_TARGETS, fdr_targets=FDR_TARGETS,
                         min_count=0, readout_mode="distributed", out_dir=out,
                         elig_col="preindexClosure")
    off = gpr.run_readout(train_df, test_df, manifest,
                          recall_targets=RECALL_TARGETS, fdr_targets=FDR_TARGETS,
                          min_count=0, readout_mode="distributed")
    block = on["gated_pc_incident"]
    assert "gated_pc_incident" not in off
    # the prevalent arm is untouched by the extra CSR run
    assert on["gated_pc"]["ranking"]["auc"] == pytest.approx(
        off["gated_pc"]["ranking"]["auc"], abs=2e-3)
    # ...and it landed on disk, where the recovery path needs it
    got = json.loads((out / "results_readout.json").read_text())
    assert set(got) == {"gated_pc", "gated_pc_head", "gated_pc_incident"}
    assert "PREVALENT-FIT" in got["gated_pc_incident"]["naming"]
    assert set(block["macros"]) == {"prevalent_full", "prevalent_shared",
                                    "incident_full", "incident_shared"}
    # eligibility really did bite: node 3's prior carriers left both classes
    assert block["eligibility"]["n_eligible_cells"] < D_TE * C
    assert "constant_prediction_column" in block["skipped_by_reason"]
    txt = capsys.readouterr().out
    assert "INCIDENT" in txt or "incident readout" in txt


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


def test_parse_args_theta_topm_and_calibration_defaults_are_the_old_behaviour():
    """Both new knobs default to what every existing run already did: full-K theta
    and calibration ON. A default change here would silently reinterpret every
    committed exp doc."""
    a = gpc.parse_args(["--cdr", "x", "--billing", "y", "--out-dir", "/tmp/o"])
    assert a.readout_theta_topm == 0
    assert a.readout_calibration == "on"
    b = gpc.parse_args(["--cdr", "x", "--billing", "y", "--out-dir", "/tmp/o",
                        "--readout-theta-topm", "256",
                        "--readout-calibration", "off"])
    assert b.readout_theta_topm == 256 and b.readout_calibration == "off"


def test_resolve_readout_calibration_gates_the_block():
    """The one gate `main` consults, so the manifest, the log line and the dev
    profile cannot disagree about what the string meant."""
    assert gpc.resolve_readout_calibration("on") is True
    assert gpc.resolve_readout_calibration("off") is False
    assert gpc.resolve_readout_calibration(None) is True     # pre-flag namespace
    with pytest.raises(ValueError):
        gpc.resolve_readout_calibration("maybe")


def test_calibration_block_is_entirely_inside_the_gate():
    """`--readout-calibration off` must skip the WORK, not just the print.

    `main` cannot be called without a BigQuery corpus, so the "not executed" claim is
    pinned structurally instead: every piece of the calibration block — its second
    batched solve, its two lean collects, the isotonic fit and the result key — has
    to live inside the `if run_calibration:` suite, at a deeper indent. A future edit
    that hoists any of them out (the easy mistake: computing `proba_te_fit`
    unconditionally because something below "might" want it) fails here, which is the
    only place it would be caught before a cluster run pays for it.
    """
    import inspect
    import textwrap

    src = textwrap.dedent(inspect.getsource(gpc.main)).splitlines()
    starts = [i for i, ln in enumerate(src) if ln.strip() == "if run_calibration:"]
    assert len(starts) == 1, "one gate, one block"
    head = starts[0]
    indent = len(src[head]) - len(src[head].lstrip())
    end = next((i for i in range(head + 1, len(src))
                if src[i].strip() and (len(src[i]) - len(src[i].lstrip())) <= indent),
               len(src))
    block = "\n".join(src[head:end])
    outside = "\n".join(src[:head] + src[end:])
    for marker in ('label="gated_pc calibration-fit"', "calibrate_per_node(",
                   '"gated_pc_conditional_cal"', "conditional ECE (VOI readiness"):
        assert marker in block, marker
        assert marker not in outside, f"{marker} escaped the gate"


def test_save_fit_writes_the_model_then_overwrites_it_with_the_full_record(tmp_path):
    """One writer, two calls: a fit-only floor and the authoritative final record.

    The npz is the FIT — hours of CAVI that no readout failure should be allowed to
    throw away — so it lands as soon as the fit exists, next to a manifest that says
    so (`partial="fit-only"`, `results=None`). The final call writes the SAME two
    paths with the arms filled in and the marker cleared, so a finished run's
    manifest is what it always was and a died-in-readout run is still re-scoreable
    by `gated_pc_readout` off the npz.
    """
    gp = {"lambda": np.arange(6.0).reshape(2, 3), "alpha": np.ones(2),
          "w_CK": np.full((4, 2), 0.5)}
    fields = {"model_class": "gated_pc", "C": 4, "K": 2, "readout_mode": "distributed"}

    early = gpc._save_fit(tmp_path, gp, 4, fields, partial="fit-only")
    assert early["partial"] == "fit-only" and early["results"] is None
    assert early["per_node_domain_mass"] is None
    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert on_disk == early
    with np.load(tmp_path / "gated_pc_result.npz") as z:
        assert np.array_equal(z["lambda"], gp["lambda"])
        assert np.array_equal(z["b_CK"], np.zeros(4))    # absent b_CK -> zeros
    # the caller's field dict is a template, not state the writer mutates
    assert "results" not in fields and "partial" not in fields

    results = {"gated_pc": {"ranking": {"auc": 0.9}}}
    final = gpc._save_fit(tmp_path, gp, 4, fields, results=results,
                          domain_mass={0: [1.0]})
    assert final["partial"] is None and final["results"] == results
    assert final["per_node_domain_mass"] == {"0": [1.0]}
    assert json.loads((tmp_path / "manifest.json").read_text()) == final
    assert final["model_class"] == early["model_class"] == "gated_pc"


def test_save_fit_splits_a_multi_domain_lambda_dict(tmp_path):
    """`np.savez` cannot store a dict, so a multi-domain λ goes out as
    `lambda_0, lambda_1, ...` — the shape `reconstruct_model` reads back. Pinned
    because the save block moved into a helper and this is its one branch."""
    gp = {"lambda": {1: np.ones((2, 2)), 0: np.zeros((2, 2))},
          "alpha": np.ones(2), "w_CK": np.zeros((3, 2)), "b_CK": np.full(3, 0.25)}
    gpc._save_fit(tmp_path, gp, 3, {"C": 3}, partial="fit-only")
    with np.load(tmp_path / "gated_pc_result.npz") as z:
        assert set(z.files) == {"lambda_0", "lambda_1", "alpha", "w_CK", "b_CK"}
        assert np.array_equal(z["lambda_1"], np.ones((2, 2)))
        assert np.array_equal(z["b_CK"], np.full(3, 0.25))


def test_the_fit_only_save_precedes_every_readout_in_main():
    """The early save must run BEFORE any readout work, or it saves nothing new.

    `main` needs a BigQuery corpus, so — as with the calibration gate above — the
    ordering claim is pinned structurally: the `partial="fit-only"` write has to
    appear after the estimator's `.fit(` and before the first line that spends
    cluster time on a readout (the θ transforms, either scoring path, the A/B
    gate). The failure this catches is the natural drift: someone moves the save
    "next to the other save" at the end of the run, and the next readout death
    costs a whole fit again.
    """
    import inspect
    import textwrap

    src = textwrap.dedent(inspect.getsource(gpc.main))
    lines = src.splitlines()

    def _line(pred, what):
        hits = [i for i, ln in enumerate(lines) if pred(ln)]
        assert hits, f"no line matching {what}"
        return hits[0]

    fit = _line(lambda ln: "pc_model = pc_est.fit(" in ln, "the gated_pc fit")
    early = _line(lambda ln: 'partial="fit-only"' in ln, "the fit-only save")
    readout = min(_line(lambda ln: "pc_model.transform(" in ln, "the θ transform"),
                  _line(lambda ln: "distributed_score_arm(" in ln, "the dist arm"),
                  _line(lambda ln: "_collect_theta_labels(" in ln, "the θ collect"))
    final = _line(lambda ln: "results=results" in ln, "the final save")
    assert fit < early < readout, (fit, early, readout)
    assert readout < final
    # ...and both writes go through the one helper, so they cannot drift apart
    assert len([ln for ln in lines if "_save_fit(" in ln]) == 2


def test_re_readout_never_indexes_a_possibly_null_manifest_results():
    """A fit-only manifest has `results: None`, and the re-readout tool is exactly
    the thing you run against one — so its echo of the other arms must treat that
    as "nothing stored yet" and say so, never index into it."""
    import inspect

    src = inspect.getsource(gpr.main)
    assert 'manifest.get("results") or {}' in src
    assert 'manifest["results"]' not in src
    assert 'manifest.get("partial")' in src


def test_dev_profile_skips_the_calibration_solve(monkeypatch, capsys):
    """CHARM_DEV drops the calibration block outright. The isotonic ECE is a
    reliability diagnostic, never a ranking signal, so the dev loop's comparisons are
    unaffected while the supervised arm loses a whole second batched solve."""
    monkeypatch.setenv("CHARM_DEV", "1")
    base = {"model_class": "gated_pc", "max_iter": 100}
    dev = rex._apply_dev_profile(dict(base))
    assert dev["readout_calibration"] == "off"
    assert "readout_calibration=off" in capsys.readouterr().out
    monkeypatch.delenv("CHARM_DEV")
    assert "readout_calibration" not in rex._apply_dev_profile(dict(base))


def test_gated_pc_args_pass_the_new_front_matter_keys_through(monkeypatch):
    """Front-matter -> argv, following the `readout_max_iter` pattern: absent keys
    emit NOTHING, so an exp doc that predates the flags produces byte-identical
    argv."""
    # the CDR/billing pair is workspace env, not config (see _require_workspace_env)
    monkeypatch.setenv("WORKSPACE_CDR", "cdr")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    base = {"source_table": "t", "person_mod": 1, "vocab_size": 5000, "min_df": 20,
            "min_patient_count": 20, "doc_min_length": 10, "max_iter": 100,
            "min_n": 0, "n_bg": 8, "tpn": 1, "seed": 0,
            "readout_mode": "distributed"}
    argv = rex.build_gated_pc_args(
        dict(base, readout_theta_topm=256, readout_calibration="off"), "/tmp/out")
    assert argv[argv.index("--readout-theta-topm") + 1] == "256"
    assert argv[argv.index("--readout-calibration") + 1] == "off"
    plain = rex.build_gated_pc_args(dict(base), "/tmp/out")
    assert "--readout-theta-topm" not in plain
    assert "--readout-calibration" not in plain


def test_densify_lean_blocks_handles_dense_mask_marker():
    """`m_idx is None` is the all-ones-mask marker (`--label-mask-mode full`), and
    the densifier must expand it rather than leaving the mask empty."""
    P = np.arange(6, dtype=np.float32).reshape(2, 3)
    blocks = [(np.array([7, 8], dtype=np.int64), P,
               np.array([0, 2], dtype=np.int32), np.array([0, 1, 2], dtype=np.int64),
               None, None)]
    proba, y, mask, ids, _elig = gpc._densify_lean_blocks(blocks, 3)
    assert np.array_equal(proba, P)
    assert np.array_equal(y, np.array([[1, 0, 0], [0, 0, 1]], dtype=np.uint8))
    assert np.array_equal(mask, np.ones((2, 3), dtype=np.uint8))
    assert ids == [7, 8]


def test_stdout_tee_appends_sanitized_lines_and_survives_truncation(tmp_path, capsys):
    """The durable driver log exists because the runs dir is a gcsfuse mount:
    a long-lived handle uploads only on close (the original empty-summary.md),
    and per-LINE closes mutation-storm the GCS object until ENOSPC (the 0104
    smoke deaths) — so the tee batches lines and closes per flush interval.
    It must (a) commit complete lines only, (b) drop the wrapper's
    patient/noise patterns, and (c) keep appending to a file truncated
    mid-run (each flush reopens, so it never writes to a stale handle).
    flush_every_s=0 makes every write flush, so the batching is exercised
    at its immediate-mode boundary."""
    import io
    from _driver_common import _StdoutTee

    log = tmp_path / "driver_log.md"
    real = io.StringIO()
    tee = _StdoutTee(log, real, flush_every_s=0.0)
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


def test_stdout_tee_batches_between_flush_intervals(tmp_path):
    """With a long interval, lines accumulate in memory and reach disk only on
    an explicit flush — one object mutation per batch is the property that
    keeps gcsfuse alive at whole-Mondo log volume."""
    import io
    from _driver_common import _StdoutTee

    log = tmp_path / "driver_log.md"
    tee = _StdoutTee(log, io.StringIO(), flush_every_s=3600.0)
    tee.write("[driver] one\n[driver] two\n")
    assert not log.exists() or log.read_text() == ""   # still buffered
    tee._flush_pending()
    assert log.read_text() == "[driver] one\n[driver] two\n"
    tee.write("[driver] three\n")
    tee._flush_pending()
    assert log.read_text().endswith("[driver] three\n")
