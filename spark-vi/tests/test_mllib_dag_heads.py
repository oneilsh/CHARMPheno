"""DagHeadsEstimator / DagHeadsModel (ADR 0051 §4): the readout heads as an MLlib
stage. Fit equals `readout.fit_heads` on the same frame, transform equals the lean
scoring kernel and the closure product, persistence round-trips, and
`Pipeline([DagGatedLDAEstimator, DagHeadsEstimator])` runs end to end."""
from __future__ import annotations

import numpy as np
import pytest

pyspark = pytest.importorskip("pyspark")
from pyspark.ml.linalg import Vectors  # noqa: E402

PARENT = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
C = 7


def _closure_labels(leaf):
    lab = {leaf}
    while lab and min(lab) in PARENT or any(n in PARENT for n in lab):
        new = {PARENT[n] for n in lab if n in PARENT} - lab
        if not new:
            break
        lab |= new
    return [1.0 if c in lab else 0.0 for c in range(C)]


def _theta_df(spark, n=300, K=6, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(C, K)) * 3
    rows = []
    for _ in range(n):
        th = rng.dirichlet(np.full(K, 0.5))
        y = (rng.random(C) < 1 / (1 + np.exp(-(W @ th - 0.5)))).astype(float)
        y[0] = 1.0                                     # root: all-positive -> constant
        rows.append((Vectors.dense(th), y.tolist(), [1.0] * C))
    return spark.createDataFrame(rows, ["topicDistribution", "label", "labelMask"])


def test_defaults_and_dag_param():
    from spark_vi.mllib import DagHeadsEstimator
    est = DagHeadsEstimator()
    assert est.getOrDefault("featuresCol") == "topicDistribution"
    assert est.getDagParent() is None
    est.setDagParent({3: 1, 4: [1, 2]})
    assert est.getDagParent() == {3: [1], 4: [1, 2]}


@pytest.mark.slow
@pytest.mark.parametrize("engine", ["spark", "local"])
def test_fit_and_transform_equal_the_readout(spark, engine):
    from spark_vi.dag.ops import closure_matrix
    from spark_vi.mllib import DagHeadsEstimator
    from spark_vi.readout import SparkReadoutData, fit_heads
    from spark_vi.readout.stacked import stacked_proba

    df = _theta_df(spark).cache()
    est = DagHeadsEstimator(maxIter=200, engine=engine,
                            stackedProbabilityCol="stackedProbability")
    est.setDagParent(PARENT)
    model = est.fit(df)
    V, b, const, deg, _ = fit_heads(SparkReadoutData(df, C, 6), max_iter=200)
    np.testing.assert_allclose(model.V, V, rtol=1e-6, atol=1e-8)
    assert model.degenerate[0] and model.const[0] == 1.0
    rows = model.transform(df).select("topicDistribution", "nodeProbability",
                                      "stackedProbability").collect()
    th = np.stack([r[0].toArray() for r in rows])
    p = np.stack([r[1].toArray() for r in rows])
    want = 1 / (1 + np.exp(-(th @ model.V.T + model.b_raw)))
    want[:, 0] = 1.0
    np.testing.assert_allclose(p, want, atol=1e-6)
    np.testing.assert_allclose(np.stack([r[2].toArray() for r in rows]),
                               stacked_proba(p, closure_matrix({c: [q] for c, q in PARENT.items()}, C)),
                               atol=1e-6)


@pytest.mark.slow
def test_save_load_round_trip(spark, tmp_path):
    from spark_vi.mllib import DagHeadsEstimator, DagHeadsModel

    df = _theta_df(spark, n=120)
    model = DagHeadsEstimator(maxIter=50, thetaTopM=3).fit(df)
    model.save(tmp_path / "heads")
    back = DagHeadsModel.load(tmp_path / "heads")
    np.testing.assert_array_equal(back.V, model.V)
    assert back.theta_topm == 3 and back.getOrDefault("thetaTopM") == 3
    a = model.transform(df).select("nodeProbability").collect()
    b = back.transform(df).select("nodeProbability").collect()
    assert [r[0] for r in a] == [r[0] for r in b]


@pytest.mark.slow
def test_pipeline_gated_lda_then_heads(spark):
    from pyspark.ml import Pipeline
    from spark_vi.mllib import DagGatedLDAEstimator, DagHeadsEstimator

    rng = np.random.default_rng(0)
    rows = []
    for _ in range(60):
        leaf = int(rng.choice([3, 4, 5, 6]))
        idx = sorted(rng.choice(30, size=6, replace=False).tolist())
        rows.append((Vectors.sparse(30, idx, [1.0] * len(idx)), [leaf],
                     _closure_labels(leaf), [1.0] * C))
    df = spark.createDataFrame(rows, ["features", "frontier", "label", "labelMask"])
    lda = DagGatedLDAEstimator(gateNBg=2, gateTpn=1, maxIter=3, seed=0,
                               subsamplingRate=1.0).setGateParent(PARENT)
    heads = DagHeadsEstimator(maxIter=30, stackedProbabilityCol="stacked")
    heads.setDagParent(PARENT)
    out = Pipeline(stages=[lda, heads]).fit(df).transform(df)
    r = out.select("nodeProbability", "stacked").head()
    assert r[0].size == C and r[1].size == C
    assert np.all(r[1].toArray() <= r[0].toArray() + 1e-6)   # product <= own factor
