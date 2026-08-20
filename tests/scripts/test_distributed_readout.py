"""Tests for `analysis/cloud/distributed_readout.py` (Package B of
`docs/superpowers/plans/2026-08-20-distributed-readout-plan.md`).

Two layers, matching the repo convention stated in `gated_pc_cloud.py` (Spark
wiring is cluster-covered; pure numpy partition kernels are unit-tested):

  * kernel tests — `_moments_kernel` / `_stats_kernel` against direct dense numpy,
    both mask-density code paths, and the gradient against finite differences;
  * a tiny local-Spark round trip (marked `slow`, per AGENTS.md) asserting the
    Spark shells reproduce the in-memory kernel results and that
    `per_node_metric_rows` equals `analysis.pc.evaluate._bundle_masked` — the
    plan's "What must NOT change" equality, in miniature.

The fold formulas (standardized <-> raw) are written out INLINE here rather than
imported from `analysis/pc/batched_lr.py` (Package A): the point of the injection
seam is that this module is testable without the solver, so the tests must not
create the dependency they exist to rule out.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# PySpark workers are separate processes and inherit PYTHONPATH, not the driver's
# sys.path — without this the executors cannot import either the module under test
# or `analysis.pc.evaluate` (which `per_node_metric_rows` ships by reference). Set
# at import time, i.e. during collection, so it is in place before the session-
# scoped `spark` fixture builds the context.
os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in (str(REPO_ROOT), os.environ.get("PYTHONPATH", "")) if p)

from analysis.cloud import distributed_readout as dr  # noqa: E402

C, K, D = 8, 12, 60


# --------------------------------------------------------------------------- #
# Local copies of Package A's fold formulas (see module docstring).            #
# --------------------------------------------------------------------------- #
def _fold_standardization(W_std, b_std, mu, sd):
    """z = W_std . (theta - mu)/sd + b_std  ==  (W_std/sd) . theta + b_raw."""
    V = W_std / sd
    b_raw = b_std - (W_std * mu / sd).sum(axis=1)
    return V, b_raw


def _standardized_grad_from_raw(g_raw, s, mu, sd):
    """dz/dW_std = (theta - mu)/sd, so g_std = (sum (p-y) theta - mu sum (p-y))/sd."""
    return (g_raw - mu * s[:, None]) / sd


# --------------------------------------------------------------------------- #
# Fixtures / references.                                                       #
# --------------------------------------------------------------------------- #
def _make_data(seed=0):
    """Random theta/label/mask with all-ones, sparse and all-zero mask rows."""
    rng = np.random.default_rng(seed)
    Pi = rng.dirichlet(np.ones(K), size=D)
    mask = (rng.random((D, C)) < 0.35).astype(np.float64)
    mask[0] = 1.0                                   # dense-path row
    mask[1] = 0.0                                   # no observed cell at all
    mask[2, :] = 0.0
    mask[2, 3] = 1.0                                # maximally sparse row
    mask[:, 0] = 1.0                                # a root-like always-observed node
    mask[:, C - 1] = 0.0                            # a never-observed node
    y = (rng.random((D, C)) < 0.4).astype(np.float64)
    y[:, 1] = 1.0                                   # degenerate (all-positive) node
    return Pi, y, mask


def _rows(Pi, y, mask):
    return [(Pi[d], y[d], mask[d]) for d in range(Pi.shape[0])]


def _ref_moments(Pi, y, mask):
    sum_theta = np.zeros((C, K))
    sum_sq = np.zeros((C, K))
    n_obs = np.zeros(C)
    n_pos = np.zeros(C)
    for c in range(C):
        rows = np.flatnonzero(mask[:, c])
        sum_theta[c] = Pi[rows].sum(axis=0)
        sum_sq[c] = (Pi[rows] ** 2).sum(axis=0)
        n_obs[c] = rows.size
        n_pos[c] = y[rows, c].sum()
    return sum_theta, sum_sq, n_obs, n_pos


def _ref_stats(Pi, y, mask, V, b_raw):
    """Dense per-node reference: full (D,C) score matrix, masked reductions."""
    Z = np.clip(Pi @ V.T + b_raw[None, :], -50.0, 50.0)
    P = 1.0 / (1.0 + np.exp(-Z))
    loss = np.zeros(C)
    g_raw = np.zeros((C, K))
    s = np.zeros(C)
    for c in range(C):
        rows = np.flatnonzero(mask[:, c])
        zc, yc = Z[rows, c], y[rows, c]
        r = P[rows, c] - yc
        loss[c] = np.sum(np.logaddexp(0.0, zc) - yc * zc)
        g_raw[c] = r @ Pi[rows]
        s[c] = r.sum()
    return loss, g_raw, s, P


def _params(seed=1):
    rng = np.random.default_rng(seed)
    return rng.normal(scale=0.7, size=(C, K)), rng.normal(scale=0.3, size=C)


# --------------------------------------------------------------------------- #
# Pure kernel tests.                                                           #
# --------------------------------------------------------------------------- #
def test_moments_kernel_matches_dense_reference():
    Pi, y, mask = _make_data()
    got = dr._moments_kernel(_rows(Pi, y, mask), C, K)
    for a, b in zip(got, _ref_moments(Pi, y, mask)):
        assert np.allclose(a, b, atol=1e-10, rtol=0)


def test_moments_to_mu_sd_matches_population_std():
    Pi, y, mask = _make_data()
    sum_theta, sum_sq, n_obs, _ = dr._moments_kernel(_rows(Pi, y, mask), C, K)
    mu, sd = dr.moments_to_mu_sd(sum_theta, sum_sq, n_obs)
    for c in range(C):
        rows = np.flatnonzero(mask[:, c])
        if rows.size == 0:                              # never-observed node
            assert np.allclose(mu[c], 0.0)
            assert np.allclose(sd[c], 1.0)              # identity moments
            continue
        assert np.allclose(mu[c], Pi[rows].mean(axis=0), atol=1e-10, rtol=0)
        assert np.allclose(sd[c], Pi[rows].std(axis=0, ddof=0), atol=1e-10, rtol=0)


def test_moments_to_mu_sd_zero_variance_column_gets_unit_scale():
    """Constant-on-observed-rows features must standardize to inert (sd=1), NOT to
    the eps floor — Package A's `standardization_moments` argues the floor
    manufactures a ~1e-4*n spurious gradient out of the (g_raw - s*mu)/sd
    cancellation residual, so the two twins must agree on `1.0`."""
    Pi = np.tile(np.array([0.5, 0.25, 0.25]), (5, 1))   # every column constant
    mask = np.ones((5, 2))
    y = np.zeros((5, 2))
    sum_theta, sum_sq, n_obs, _ = dr._moments_kernel(_rows(Pi, y, mask), 2, 3)
    mu, sd = dr.moments_to_mu_sd(sum_theta, sum_sq, n_obs)
    assert np.allclose(mu, Pi[0])
    assert np.allclose(sd, 1.0)


def test_stats_kernel_matches_dense_reference():
    Pi, y, mask = _make_data()
    W, b = _params()
    mu, sd = np.zeros((C, K)), np.ones((C, K))          # raw space: fold is identity
    V, b_raw = _fold_standardization(W, b, mu, sd)
    loss, g_raw, s = dr._stats_kernel(_rows(Pi, y, mask), V, b_raw, C, K)
    r_loss, r_g, r_s, _ = _ref_stats(Pi, y, mask, V, b_raw)
    assert np.allclose(loss, r_loss, atol=1e-10, rtol=0)
    assert np.allclose(g_raw, r_g, atol=1e-10, rtol=0)
    assert np.allclose(s, r_s, atol=1e-10, rtol=0)


def test_stats_kernel_sparse_and_dense_score_paths_agree(monkeypatch):
    """The |obs| >= frac*C fast path must be a pure cost choice, not a semantic one."""
    Pi, y, mask = _make_data(seed=3)
    V, b_raw = _params(seed=4)
    monkeypatch.setattr(dr, "_DENSE_MASK_FRACTION", 0.0)      # always full matvec
    dense = dr._stats_kernel(_rows(Pi, y, mask), V, b_raw, C, K)
    monkeypatch.setattr(dr, "_DENSE_MASK_FRACTION", 2.0)      # always gather V[obs]
    sparse = dr._stats_kernel(_rows(Pi, y, mask), V, b_raw, C, K)
    for a, b in zip(dense, sparse):
        assert np.allclose(a, b, atol=1e-10, rtol=0)


def test_score_cells_kernel_sparse_and_dense_paths_agree(monkeypatch):
    Pi, y, mask = _make_data(seed=5)
    V, b_raw = _params(seed=6)
    monkeypatch.setattr(dr, "_DENSE_MASK_FRACTION", 0.0)
    dense = list(dr._score_cells_kernel(_rows(Pi, y, mask), V, b_raw, C))
    monkeypatch.setattr(dr, "_DENSE_MASK_FRACTION", 2.0)
    sparse = list(dr._score_cells_kernel(_rows(Pi, y, mask), V, b_raw, C))
    assert len(dense) == int(mask.sum())
    assert len(dense) == len(sparse)
    # Same cells in the same order; p differs only in the last ulp (BLAS matvec vs
    # gathered dot re-associates the K-term sum).
    assert [(n, yv) for n, yv, _ in dense] == [(n, yv) for n, yv, _ in sparse]
    assert np.allclose([p for _, _, p in dense], [p for _, _, p in sparse],
                       atol=1e-14, rtol=0)


def test_score_cells_kernel_matches_dense_reference():
    Pi, y, mask = _make_data(seed=7)
    V, b_raw = _params(seed=8)
    _, _, _, P = _ref_stats(Pi, y, mask, V, b_raw)
    cells = list(dr._score_cells_kernel(_rows(Pi, y, mask), V, b_raw, C))
    assert len(cells) == int(mask.sum())
    seen = np.zeros((D, C), dtype=int)
    it = iter(cells)
    for d in range(D):
        for c in np.flatnonzero(mask[d]):
            node, yv, pv = next(it)
            assert node == int(c)
            assert yv == y[d, c]
            assert abs(pv - P[d, c]) < 1e-10
            seen[d, c] = 1
    assert np.array_equal(seen, mask.astype(int))


def test_stats_kernel_gradient_matches_finite_differences():
    """(loss, g_raw, s) folded to standardized space == d/d(W_std, b_std) of the
    summed masked log-loss. This is the contract Package A's solver consumes."""
    Pi, y, mask = _make_data(seed=11)
    sum_theta, sum_sq, n_obs, _ = dr._moments_kernel(_rows(Pi, y, mask), C, K)
    mu, sd = dr.moments_to_mu_sd(sum_theta, sum_sq, n_obs)
    # The never-observed node has sd == eps; keep it out of the FD probe (its
    # objective is identically zero, so its gradient is trivially zero anyway).
    W, b = _params(seed=12)

    def _total_loss(W_std, b_std):
        V, b_raw = _fold_standardization(W_std, b_std, mu, sd)
        loss, _, _ = dr._stats_kernel(_rows(Pi, y, mask), V, b_raw, C, K)
        return loss

    V, b_raw = _fold_standardization(W, b, mu, sd)
    loss, g_raw, s = dr._stats_kernel(_rows(Pi, y, mask), V, b_raw, C, K)
    gW_std = _standardized_grad_from_raw(g_raw, s, mu, sd)
    assert np.allclose(loss, _total_loss(W, b), atol=1e-12, rtol=0)

    h = 1e-6
    rng = np.random.default_rng(13)
    probes = [(int(c), int(k)) for c in range(C - 1)                # skip unobserved
              for k in rng.choice(K, size=3, replace=False)]
    for c, k in probes:
        Wp, Wm = W.copy(), W.copy()
        Wp[c, k] += h
        Wm[c, k] -= h
        fd = (_total_loss(Wp, b)[c] - _total_loss(Wm, b)[c]) / (2 * h)
        assert abs(fd - gW_std[c, k]) <= 1e-6 * max(1.0, abs(gW_std[c, k])), (c, k)
    for c in range(C - 1):
        bp, bm = b.copy(), b.copy()
        bp[c] += h
        bm[c] -= h
        fd = (_total_loss(W, bp)[c] - _total_loss(W, bm)[c]) / (2 * h)
        assert abs(fd - s[c]) <= 1e-6 * max(1.0, abs(s[c])), c


def test_dense_triples_roundtrips_packed_rows():
    """The recycled-buffer rehydration must be indistinguishable from dense rows."""
    Pi, y, mask = _make_data(seed=17)
    packed = [(Pi[d], np.flatnonzero(mask[d]).astype(np.int32),
               y[d][np.flatnonzero(mask[d])]) for d in range(D)]
    got = dr._moments_kernel(dr._dense_triples(packed, C), C, K)
    for a, b in zip(got, _ref_moments(Pi, y, mask)):
        assert np.allclose(a, b, atol=1e-10, rtol=0)


# --------------------------------------------------------------------------- #
# Local-Spark round trip (thin wiring; AGENTS.md: local Spark => @slow).       #
# --------------------------------------------------------------------------- #
def _make_df(spark, Pi, y, mask):
    from pyspark.ml.linalg import VectorUDT, Vectors
    from pyspark.sql.types import (ArrayType, DoubleType, LongType, StructField,
                                   StructType)
    schema = StructType([
        StructField("person_id", LongType(), False),
        StructField("topicDistribution", VectorUDT(), False),
        StructField("label", ArrayType(DoubleType()), False),
        StructField("labelMask", ArrayType(DoubleType()), False),
    ])
    rows = [(int(d), Vectors.dense(Pi[d]), [float(v) for v in y[d]],
             [float(v) for v in mask[d]]) for d in range(Pi.shape[0])]
    return spark.createDataFrame(rows, schema).repartition(3)


@pytest.mark.slow
class TestLocalSparkRoundTrip:
    def test_masked_moments_matches_numpy(self, spark):
        Pi, y, mask = _make_data(seed=21)
        df = _make_df(spark, Pi, y, mask)
        mu, sd, n_obs, n_pos = dr.masked_moments(df, C, K)
        r_sum, r_sq, r_n, r_pos = _ref_moments(Pi, y, mask)
        r_mu, r_sd = dr.moments_to_mu_sd(r_sum, r_sq, r_n)
        assert np.allclose(mu, r_mu, atol=1e-10, rtol=0)
        assert np.allclose(sd, r_sd, atol=1e-10, rtol=0)
        assert np.allclose(n_obs, r_n)
        assert np.allclose(n_pos, r_pos)

    def test_spark_stats_fn_matches_in_memory_kernel(self, spark):
        Pi, y, mask = _make_data(seed=22)
        df = _make_df(spark, Pi, y, mask)
        mu, sd, _, _ = dr.masked_moments(df, C, K)
        W, b = _params(seed=23)
        with dr.make_spark_stats_fn(
            df, C, K, mu, sd,
            fold_standardization=_fold_standardization,
            standardized_grad_from_raw=_standardized_grad_from_raw,
        ) as stats_fn:
            loss, gW_std, gb = stats_fn(W, b)
            loss2, _, _ = stats_fn(W, b)        # repeat pass off the cached RDD
        V, b_raw = _fold_standardization(W, b, mu, sd)
        r_loss, r_g, r_s = dr._stats_kernel(_rows(Pi, y, mask), V, b_raw, C, K)
        assert np.allclose(loss, r_loss, atol=1e-10, rtol=0)
        assert np.allclose(loss2, r_loss, atol=1e-10, rtol=0)
        assert np.allclose(gb, r_s, atol=1e-10, rtol=0)
        assert np.allclose(gW_std, _standardized_grad_from_raw(r_g, r_s, mu, sd),
                           atol=1e-10, rtol=0)

    def test_score_cells_df_matches_numpy(self, spark):
        Pi, y, mask = _make_data(seed=24)
        df = _make_df(spark, Pi, y, mask)
        V, b_raw = _params(seed=25)
        cells = dr.score_cells_df(df, V, b_raw, C)
        collected = cells.collect()
        assert len(collected) == int(mask.sum())
        _, _, _, P = _ref_stats(Pi, y, mask, V, b_raw)
        # Cells carry no doc id (plan §3: 16 bytes/cell), so compare the per-node
        # MULTISET of (y, p) values against the numpy reference.
        for c in range(C):
            got = sorted((round(r["y"], 12), round(r["p"], 12))
                         for r in collected if r["node"] == c)
            rows = np.flatnonzero(mask[:, c])
            want = sorted((round(float(y[d, c]), 12), round(float(P[d, c]), 12))
                          for d in rows)
            assert len(got) == len(want)
            for (gy, gp), (wy, wp) in zip(got, want):
                assert gy == wy
                assert abs(gp - wp) < 1e-9

    @pytest.mark.parametrize("engine", ["rdd", "pandas"])
    def test_per_node_metric_rows_matches_bundle_masked(self, spark, engine):
        from analysis.pc.evaluate import _bundle_masked

        Pi, y, mask = _make_data(seed=26)
        df = _make_df(spark, Pi, y, mask)
        V, b_raw = _params(seed=27)
        _, _, _, P = _ref_stats(Pi, y, mask, V, b_raw)
        cells = dr.score_cells_df(df, V, b_raw, C)
        try:
            got = dr.per_node_metric_rows(cells, C, engine=engine)
        except Exception as exc:                     # pragma: no cover - env guard
            if engine == "pandas" and "Unsafe" in str(exc):
                pytest.skip("Spark 3.5's bundled Arrow cannot allocate direct "
                            "buffers on this JDK; the 'pandas' engine is "
                            "cluster-covered only")
            raise
        want = _bundle_masked(P, y, mask, C)["per_label"]
        assert set(got) == set(want)
        n_skipped = 0
        for c in range(C):
            g, w = got[c], want[c]
            assert g["skipped"] == w["skipped"], c
            assert g["n_pos"] == w["n_pos"] and g["n_neg"] == w["n_neg"], c
            if w["skipped"] is not None:
                n_skipped += 1
                assert g["auc"] is None and g["ap"] is None
                continue
            assert abs(g["auc"] - w["auc"]) < 1e-9, c
            assert abs(g["ap"] - w["ap"]) < 1e-9, c
        # The fixture plants an all-positive node and a never-observed node, so the
        # skip path is genuinely exercised (not vacuously equal).
        assert n_skipped >= 2

    def test_per_node_metric_rows_min_count_skips_small_nodes(self, spark):
        from analysis.pc.evaluate import _bundle_masked

        Pi, y, mask = _make_data(seed=28)
        df = _make_df(spark, Pi, y, mask)
        V, b_raw = _params(seed=29)
        _, _, _, P = _ref_stats(Pi, y, mask, V, b_raw)
        got = dr.per_node_metric_rows(dr.score_cells_df(df, V, b_raw, C),
                                      C, min_count=10)
        want = _bundle_masked(P, y, mask, C, min_count=10)["per_label"]
        for c in range(C):
            assert got[c]["skipped"] == want[c]["skipped"], c
        assert any(w["skipped"] is not None and "small test column" in w["skipped"]
                   for w in want.values())
