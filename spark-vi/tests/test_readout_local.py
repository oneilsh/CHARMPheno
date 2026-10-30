"""The readout's local backend (`spark_vi.readout.local`) and the backend-agnostic
`fit_heads` (ADR 0051 §2).

The local backend calls the same kernels as the Spark one over rows held in memory,
so its contract is equality with those kernels; one slow test checks the whole fit
against the Spark backend on a local-Spark frame.
"""
import numpy as np
import pytest

from spark_vi.readout import LocalReadoutData, fit_heads, read_heads, write_heads
from spark_vi.readout import kernels as kern
from spark_vi.readout.solver import fold_standardization, standardized_grad_from_raw

C, K, D = 6, 10, 300


def _data(seed=0, all_pos_node=None):
    rng = np.random.default_rng(seed)
    theta = rng.dirichlet(np.full(K, 0.3), size=D)
    W = rng.normal(size=(C, K)) * 4
    y = (rng.random((D, C)) < 1 / (1 + np.exp(-(theta @ W.T - 1)))).astype(float)
    mask = (rng.random((D, C)) < 0.7).astype(float)
    mask[:, 0] = 1.0
    if all_pos_node is not None:
        y[:, all_pos_node] = mask[:, all_pos_node]
    return theta, y, mask


def _truncate(theta, m):
    out = np.zeros_like(theta)
    for d in range(theta.shape[0]):
        idx, val = kern.topm_sparse(theta[d], m)
        out[d, idx] = val
    return out


@pytest.mark.parametrize("topm", [0, 4])
def test_moments_equal_the_dense_kernel_on_the_same_design(topm):
    theta, y, mask = _data(1)
    got = LocalReadoutData.from_arrays(theta, y, mask, topm=topm).moments()
    design = _truncate(theta, topm) if topm else theta
    s1, s2, n, npos = kern.moments_kernel(zip(design, y, mask), C, K)
    mu, sd = kern.moments_to_mu_sd(s1, s2, n)
    np.testing.assert_allclose(got[0], mu, atol=1e-12)
    np.testing.assert_allclose(got[1], sd, atol=1e-12)
    np.testing.assert_array_equal(got[2], n)
    np.testing.assert_array_equal(got[3], npos)


@pytest.mark.parametrize("topm", [0, 4])
def test_stats_fn_equals_the_dense_kernel_and_honours_node_mask(topm):
    theta, y, mask = _data(2)
    data = LocalReadoutData.from_arrays(theta, y, mask, topm=topm)
    mu, sd, _, _ = data.moments()
    rng = np.random.default_rng(3)
    W, b = rng.normal(size=(C, K)), rng.normal(size=C)
    with data.stats_fn(mu, sd, fold_standardization=fold_standardization,
                       standardized_grad_from_raw=standardized_grad_from_raw) as fn:
        loss, gW, gb = fn(W, b)
        keep = np.array([True, False, True, False, False, True])
        mloss, mgW, mgb = fn(W, b, node_mask=keep)
    V, b_raw = fold_standardization(W, b, mu, sd)
    design = _truncate(theta, topm) if topm else theta
    r_loss, r_g, r_s = kern.stats_kernel(zip(design, y, mask), V, b_raw, C, K)
    np.testing.assert_allclose(loss, r_loss, atol=1e-9)
    np.testing.assert_allclose(gW, standardized_grad_from_raw(r_g, r_s, mu, sd),
                               atol=1e-9)
    np.testing.assert_allclose(gb, r_s, atol=1e-9)
    np.testing.assert_allclose(mloss[keep], loss[keep], atol=1e-12)
    assert not mloss[~keep].any() and not mgW[~keep].any() and not mgb[~keep].any()


def test_fit_heads_scores_and_applies_the_degenerate_fallback(tmp_path):
    theta, y, mask = _data(4, all_pos_node=3)
    data = LocalReadoutData.from_arrays(theta, y, mask)
    V, b_raw, const, deg, info = fit_heads(data, max_iter=100)
    assert deg[3] and const[3] == 1.0 and not deg[[0, 1, 2, 4, 5]].any()
    proba, yy, mm, ids = data.lean_proba(V, b_raw, degenerate=deg, const=const)
    assert proba.shape == (D, C) and ids == list(range(D))
    assert (proba[:, 3] == 1.0).all()
    np.testing.assert_array_equal(yy, y.astype(np.uint8))
    np.testing.assert_array_equal(mm, mask.astype(np.uint8))
    # The heads npz round-trips what a scoring-only pass needs.
    assert write_heads(tmp_path, "arm", V, b_raw, const, deg, C, K, 0)
    got = read_heads(tmp_path, "arm", C=C, K=K, theta_topm=0)
    np.testing.assert_array_equal(got[0], V)
    np.testing.assert_array_equal(got[3], deg)


def _frame(spark, theta, y, mask):
    from pyspark.ml.linalg import VectorUDT, Vectors
    from pyspark.sql.types import (ArrayType, DoubleType, LongType, StructField,
                                   StructType)

    schema = StructType([
        StructField("doc_key", LongType(), False),
        StructField("topicDistribution", VectorUDT(), False),
        StructField("label", ArrayType(DoubleType()), False),
        StructField("labelMask", ArrayType(DoubleType()), False)])
    rows = [(int(d), Vectors.dense(theta[d]), [float(v) for v in y[d]],
             [float(v) for v in mask[d]]) for d in range(theta.shape[0])]
    return spark.createDataFrame(rows, schema).repartition(3)


@pytest.mark.slow
@pytest.mark.parametrize("topm", [0, 4])
def test_local_and_spark_backends_fit_the_same_heads(spark, topm):
    from spark_vi.readout import SparkReadoutData

    theta, y, mask = _data(6, all_pos_node=2)
    df = _frame(spark, theta, y, mask)
    sp = SparkReadoutData(df, C, K, topm=topm)
    lo = LocalReadoutData.from_spark(df, C, K, topm=topm)
    Vs, bs, cs, ds, _ = fit_heads(sp, max_iter=60)
    Vl, bl, cl, dl, _ = fit_heads(lo, max_iter=60)
    np.testing.assert_array_equal(ds, dl)
    np.testing.assert_allclose(Vl, Vs, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(bl, bs, rtol=1e-6, atol=1e-8)
    ps, *_s, ks = sp.lean_proba(Vs, bs, degenerate=ds, const=cs)
    pl, *_l, kl = lo.lean_proba(Vl, bl, degenerate=dl, const=cl)
    order_s, order_l = np.argsort(ks), np.argsort(kl)
    np.testing.assert_allclose(pl[order_l], ps[order_s], atol=1e-5)


def test_score_blocks_carries_extra_columns_like_the_spark_backend():
    theta, y, mask = _data(5)
    data = LocalReadoutData.from_arrays(theta, y, mask)
    C, K = y.shape[1], theta.shape[1]
    V, b = np.zeros((C, K)), np.zeros(C)
    names = [f"row{i}" for i in range(len(theta))]
    blk = data.score_blocks(V, b, extra={"name": names})
    assert blk.blocks[0]["name"] == names
    with pytest.raises(ValueError):
        data.score_blocks(V, b, extra={"name": names[:-1]})
