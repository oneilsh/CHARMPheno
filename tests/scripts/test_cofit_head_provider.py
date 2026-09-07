"""The DISTRIBUTED lbfgs co-fit head re-scoring provider (plan 2026-09-07 WP3a).

`analysis/cloud/distributed_readout.CofitHeadStatsProvider` is the cluster twin of
`spark_vi.models.topic.pc.OnlinePCLDA.make_head_stats_provider_inmemory`: on each
outer SVI iter it re-scores the CURRENT θ over the train split with a distributed
CAVI pass and hands the engine the identical `(stats_fn, mu, sd, n_obs, n_pos)`
tuple, only via `treeAggregate` instead of a driver doc-loop.

The correctness gate the plan names: on the SAME docs and the SAME global params
the distributed provider's per-node loss/gradient (and mu/sd/n_obs/n_pos) MATCH the
in-memory provider's to numerical tolerance — the proof the distributed twin is
faithful before any cluster minute (exp 0120 is the scale test). Both the
head_standardize=True (real per-node moments) and =False (identity) branches are
covered, with and without the solver's `node_mask` line-search restriction.

Two layers per the repo convention (Spark wiring cluster-covered; numpy kernels
unit-tested): a pure `_cofit_theta_kernel` == engine `_plain_cavi_theta` unit
check, and the slow local-Spark end-to-end provider equality.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (str(REPO_ROOT), str(REPO_ROOT / "spark-vi")):
    if p not in sys.path:
        sys.path.insert(0, p)
# PySpark workers inherit PYTHONPATH, not the driver's sys.path — the θ-scoring UDF
# ships `distributed_readout` (and spark_vi's feature helpers) by module reference,
# so the executors must be able to import them. Set at collection time, before the
# session-scoped `spark` fixture builds the context.
os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in (str(REPO_ROOT), str(REPO_ROOT / "spark-vi"),
                os.environ.get("PYTHONPATH", "")) if p)

from analysis.cloud import distributed_readout as dr  # noqa: E402
from spark_vi.models.topic.batched_lr import (  # noqa: E402
    fold_standardization, standardized_grad_from_raw)
from spark_vi.models.topic.pc import OnlinePCLDA  # noqa: E402
from spark_vi.models.topic.types import PCDocument  # noqa: E402


def _corpus(seed=0, D=40, V=24, K=6, C=3):
    """A small planted-head PC corpus (node c reads topics {2c, 2c+1})."""
    rng = np.random.default_rng(seed)
    w = V // K
    beta = np.full((K, V), 0.02)
    for k in range(K):
        beta[k, k * w:(k + 1) * w] += 1.0
    beta /= beta.sum(axis=1, keepdims=True)
    cum = np.cumsum(beta, axis=1)
    sig = [np.array([2 * c, 2 * c + 1]) for c in range(C)]
    docs = []
    for _ in range(D):
        theta = rng.dirichlet(np.full(K, 0.3))
        length = int(rng.integers(30, 50))
        z = rng.choice(K, size=length, p=theta)
        u = rng.random(length)
        words = np.array([int(np.searchsorted(cum[zi], ui))
                          for zi, ui in zip(z, u)])
        idx, cnt = np.unique(words, return_counts=True)
        y = np.array([float(rng.random() < 1.0 / (1.0 + np.exp(
            -8.0 * (theta[sig[c]].sum() - 0.34)))) for c in range(C)])
        # A partly-observed mask so the masking paths (moments + stats) are exercised.
        mask = (rng.random(C) < 0.85).astype(float)
        mask[0] = 1.0
        docs.append(PCDocument(indices=idx.astype(np.int32),
                               counts=cnt.astype(np.float64), length=length,
                               y=y, label_mask=mask))
    return docs, V, K, C


# --------------------------------------------------------------------------- #
# 1. the θ kernel reproduces the engine's _plain_cavi_theta byte-for-byte      #
# --------------------------------------------------------------------------- #
def test_cofit_theta_kernel_matches_plain_cavi_theta():
    docs, V, K, C = _corpus(seed=1, D=12)
    m = OnlinePCLDA(K=K, vocab_size=V, C=C, weight_y=1.0, grad_cavi_iters=8,
                    head_optimizer="lbfgs", random_seed=0)
    gp = m.initialize_global(None)
    expElogbeta = m._expElogbeta_from_lambda(gp["lambda"])
    alpha_vec = np.asarray(gp["alpha"], dtype=np.float64)
    for d in docs:
        eb_d = expElogbeta[:, np.asarray(d.indices)]
        ref = m._plain_cavi_theta(
            eb_d, np.asarray(d.counts, dtype=np.float64), alpha_vec,
            m.grad_cavi_iters)
        got = dr._cofit_theta_kernel(
            d.indices, d.counts, expElogbeta, alpha_vec, m.grad_cavi_iters)
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12)


# --------------------------------------------------------------------------- #
# THE CORRECTNESS GATE: distributed provider == in-memory provider             #
# --------------------------------------------------------------------------- #
def _rows_to_df(spark, docs, V, C):
    from pyspark.ml.linalg import DenseVector, SparseVector
    rows = []
    for d in docs:
        idx = [int(i) for i in np.asarray(d.indices)]
        val = [float(c) for c in np.asarray(d.counts, dtype=np.float64)]
        rows.append((SparseVector(V, idx, val),
                     DenseVector([float(x) for x in d.y]),
                     DenseVector([float(x) for x in d.label_mask])))
    return spark.createDataFrame(rows, ["features", "label", "labelMask"])


@pytest.mark.slow
@pytest.mark.parametrize("standardize", [True, False])
def test_distributed_provider_matches_inmemory(spark, standardize):
    docs, V, K, C = _corpus(seed=0, D=40, V=24, K=6, C=3)
    m = OnlinePCLDA(K=K, vocab_size=V, C=C, weight_y=5.0, grad_cavi_iters=8,
                    head_optimizer="lbfgs", head_intercept=standardize,
                    head_standardize=standardize, head_std_floor=1e-6,
                    random_seed=0)
    # A non-trivial θ: take one update step so λ (and the head) have moved off the
    # flat seed, exactly the moving-θ regime the co-fit provider re-scores.
    gp = m.initialize_global(None)
    gp = m.update_global(gp, m.local_update(docs, gp), learning_rate=0.4)
    p_mem = m.make_head_stats_provider_inmemory(docs)

    df = _rows_to_df(spark, docs, V, C)
    prov = dr.make_cofit_head_stats_provider(
        df, C, K, grad_cavi_iters=m.grad_cavi_iters,
        head_standardize=m.head_standardize, head_std_floor=m.head_std_floor,
        expElogbeta_fn=m._expElogbeta_from_lambda,
        fold_standardization=fold_standardization,
        standardized_grad_from_raw=standardized_grad_from_raw,
        features_col="features", label_col="label", mask_col="labelMask")
    try:
        sf_m, mu_m, sd_m, nobs_m, npos_m = p_mem(gp)
        sf_d, mu_d, sd_d, nobs_d, npos_d = prov(gp)

        np.testing.assert_allclose(mu_d, mu_m, rtol=0, atol=1e-8)
        np.testing.assert_allclose(sd_d, sd_m, rtol=0, atol=1e-8)
        np.testing.assert_array_equal(nobs_d, nobs_m)
        np.testing.assert_array_equal(npos_d, npos_m)

        rng = np.random.default_rng(7)
        W = rng.normal(scale=0.5, size=(C, K))
        b = rng.normal(scale=0.3, size=C)
        node_mask = np.array([True, False, True])
        for mask in (None, node_mask):
            lm, gWm, gbm = sf_m(W, b, node_mask=mask)
            ld, gWd, gbd = sf_d(W, b, node_mask=mask)
            np.testing.assert_allclose(ld, lm, rtol=1e-9, atol=1e-8)
            np.testing.assert_allclose(gWd, gWm, rtol=1e-9, atol=1e-8)
            np.testing.assert_allclose(gbd, gbm, rtol=1e-9, atol=1e-8)
    finally:
        prov.close()


@pytest.mark.slow
def test_provider_close_releases_and_is_idempotent(spark):
    docs, V, K, C = _corpus(seed=2, D=20, V=24, K=6, C=3)
    m = OnlinePCLDA(K=K, vocab_size=V, C=C, weight_y=1.0, grad_cavi_iters=6,
                    head_optimizer="lbfgs", random_seed=0)
    gp = m.initialize_global(None)
    df = _rows_to_df(spark, docs, V, C)
    prov = dr.make_cofit_head_stats_provider(
        df, C, K, grad_cavi_iters=m.grad_cavi_iters,
        head_standardize=False, head_std_floor=0.0,
        expElogbeta_fn=m._expElogbeta_from_lambda,
        fold_standardization=fold_standardization,
        standardized_grad_from_raw=standardized_grad_from_raw,
        features_col="features", label_col="label", mask_col="labelMask")
    # Two outer-iter calls: the second must release the first's cached scored df /
    # projection (no leak between iters), and close() after must be a safe no-op.
    prov(gp)
    prov(gp)
    assert prov.n_calls == 2
    prov.close()
    prov.close()


# --------------------------------------------------------------------------- #
# THE MULTI-DOMAIN GATE: exp 0120's ACTUAL path (per-domain features_cols)      #
# --------------------------------------------------------------------------- #
# exp 0120 == 0118's config, which is MULTI-DOMAIN (extra_domains -> per-domain
# feature columns), so the cluster fit exercises `score_cofit_theta_df`'s
# `_concat_domain_features` branch, NOT the single fused `features_col` above. This
# proves that branch scores the SAME θ the engine's own multi-domain head reads: the
# distributed provider must match the in-memory provider on multi-domain docs, where
# λ is a per-domain dict `{m: (K, V_m)}` and a doc's global token ids are the
# per-domain local ids concatenated with each domain's cumulative offset.
def _multidomain_engine_and_docs(seed=0, D=40, V0=18, V1=12, C=3, standardize=True):
    """A 2-domain Gated-PC OnlinePCLDA (lbfgs head) + its GatedPCDocuments.

    Mirrors `test_pc_multidomain_correction._two_domain_gated_pc`: an OnlinePCLDA
    wrapping a 2-domain GatedOnlineLDA over a flat 2-node DAG. Docs carry tokens in
    BOTH domains (global ids [0,V0) domain 0, [V0,V0+V1) domain 1) so the per-domain
    concat is genuinely exercised, plus a frontier (gates λ training) and partly-
    observed labels (exercises the masked moments/stats paths)."""
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    from spark_vi.models.topic.types import GatedPCDocument

    parent = {1: 0, 2: 0}                                  # flat DAG: 2 disease nodes
    lay = DagLayout(parent, n_bg=2, tpn=1)
    V = V0 + V1
    engine = GatedOnlineLDA(lay, vocab_size=V, domains=[V0, V1], random_seed=0)
    m = OnlinePCLDA(K=engine.K, vocab_size=V, C=C, weight_y=5.0, grad_cavi_iters=8,
                    head_optimizer="lbfgs", head_intercept=standardize,
                    head_standardize=standardize, head_std_floor=1e-6,
                    random_seed=0, topic_engine=engine)
    rng = np.random.default_rng(seed)
    fronts = [{1}, {2}, set(), {1, 2}]
    docs = []
    for d in range(D):
        n0 = int(rng.integers(2, 6))
        n1 = int(rng.integers(2, 5))
        i0 = np.sort(rng.choice(np.arange(0, V0), size=n0, replace=False))
        i1 = np.sort(rng.choice(np.arange(V0, V), size=n1, replace=False))
        idx = np.concatenate([i0, i1]).astype(np.int32)   # globally sorted
        cnt = rng.integers(1, 5, size=idx.size).astype(np.float64)
        y = rng.integers(0, 2, size=C).astype(np.float64)
        mask = (rng.random(C) < 0.85).astype(np.float64)
        mask[0] = 1.0
        docs.append(GatedPCDocument(
            indices=idx, counts=cnt, length=int(cnt.sum()), y=y,
            label_mask=mask, frontier=frozenset(fronts[d % len(fronts)])))
    return m, engine, docs, V0, V1, C


def _md_rows_to_df(spark, docs, V0, V1, C):
    """Split each doc's GLOBAL token ids back into per-domain sparse feature columns.

    The inverse of the fit's `_concat_domain_features`: a global id g < V0 is domain
    0's local id g; g >= V0 is domain 1's local id g - V0. The provider's UDF then
    re-concatenates these exactly, so the θ it scores must equal the in-memory
    provider's over the same global ids."""
    from pyspark.ml.linalg import DenseVector, SparseVector

    rows = []
    for d in docs:
        idx = np.asarray(d.indices)
        cnt = np.asarray(d.counts, dtype=np.float64)
        d0 = idx < V0
        sv0 = SparseVector(V0, [int(i) for i in idx[d0]],
                           [float(c) for c in cnt[d0]])
        sv1 = SparseVector(V1, [int(i - V0) for i in idx[~d0]],
                           [float(c) for c in cnt[~d0]])
        rows.append((sv0, sv1,
                     DenseVector([float(x) for x in d.y]),
                     DenseVector([float(x) for x in d.label_mask])))
    return spark.createDataFrame(
        rows, ["features_0", "features_1", "label", "labelMask"])


@pytest.mark.slow
@pytest.mark.parametrize("standardize", [True, False])
def test_distributed_provider_matches_inmemory_multidomain(spark, standardize):
    m, engine, docs, V0, V1, C = _multidomain_engine_and_docs(
        seed=0, D=40, V0=18, V1=12, C=3, standardize=standardize)
    K, V = engine.K, V0 + V1
    # Move λ off the flat seed (λ is a per-domain dict) so θ is non-trivial.
    gp = m.initialize_global(None)
    assert isinstance(gp["lambda"], dict) and set(gp["lambda"]) == {0, 1}
    gp = m.update_global(gp, m.local_update(docs, gp), learning_rate=0.4)
    p_mem = m.make_head_stats_provider_inmemory(docs)

    df = _md_rows_to_df(spark, docs, V0, V1, C)
    prov = dr.make_cofit_head_stats_provider(
        df, C, K, grad_cavi_iters=m.grad_cavi_iters,
        head_standardize=m.head_standardize, head_std_floor=m.head_std_floor,
        expElogbeta_fn=m._expElogbeta_from_lambda,
        fold_standardization=fold_standardization,
        standardized_grad_from_raw=standardized_grad_from_raw,
        features_cols=["features_0", "features_1"], domain_sizes=[V0, V1],
        label_col="label", mask_col="labelMask")
    try:
        sf_m, mu_m, sd_m, nobs_m, npos_m = p_mem(gp)
        sf_d, mu_d, sd_d, nobs_d, npos_d = prov(gp)

        np.testing.assert_allclose(mu_d, mu_m, rtol=0, atol=1e-8)
        np.testing.assert_allclose(sd_d, sd_m, rtol=0, atol=1e-8)
        np.testing.assert_array_equal(nobs_d, nobs_m)
        np.testing.assert_array_equal(npos_d, npos_m)

        rng = np.random.default_rng(11)
        W = rng.normal(scale=0.5, size=(C, K))
        b = rng.normal(scale=0.3, size=C)
        node_mask = np.array([True, False, True])
        for mask in (None, node_mask):
            lm, gWm, gbm = sf_m(W, b, node_mask=mask)
            ld, gWd, gbd = sf_d(W, b, node_mask=mask)
            np.testing.assert_allclose(ld, lm, rtol=1e-9, atol=1e-8)
            np.testing.assert_allclose(gWd, gWm, rtol=1e-9, atol=1e-8)
            np.testing.assert_allclose(gbd, gbm, rtol=1e-9, atol=1e-8)
    finally:
        prov.close()
