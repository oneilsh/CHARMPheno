"""Increment-1 shim smoke tests for PCEstimator / PCModel (weightY == 0).

Mirrors test_gated_lda_shim.py: a tiny synthetic Spark DataFrame fits
end-to-end and transforms, appending the label-free topic-distribution column,
deterministically given a seed. The label columns are tolerated-absent on the
weightY == 0 path (they carry no information there), and are also exercised
present to pin the STM-style threading.
"""
import numpy as np
import pytest

pyspark = pytest.importorskip("pyspark")
from pyspark.ml.linalg import Vectors

# `spark` is the session-scoped fixture from tests/conftest.py.


def _block_rows(n_per_topic=12, K=3, block=4, seed=0, with_labels=False):
    """Well-clustered rows: each doc favors a contiguous V/K vocab block."""
    rng = np.random.default_rng(seed)
    V = K * block
    rows = []
    for t in range(K):
        favored = list(range(t * block, (t + 1) * block))
        for _ in range(n_per_topic):
            counts = np.zeros(V)
            for w in rng.choice(favored, size=10, replace=True):
                counts[w] += 1.0
            idx = sorted(np.nonzero(counts)[0].tolist())
            fv = Vectors.sparse(V, idx, [float(counts[i]) for i in idx])
            rows.append((fv, [float(t % 2)]) if with_labels else (fv,))
    cols = ["features", "label"] if with_labels else ["features"]
    return rows, cols, V


def test_pc_shim_fit_transform_smoke(spark):
    from spark_vi.mllib.topic.pc import PCEstimator
    rows, cols, V = _block_rows(seed=0)
    df = spark.createDataFrame(rows, cols)

    est = PCEstimator(k=3, maxIter=4, seed=0, subsamplingRate=1.0,
                      topicDistributionCol="topicDistribution")
    model = est.fit(df)
    out = model.transform(df)
    assert "topicDistribution" in out.columns
    td = out.select("topicDistribution").head()[0]
    assert len(td) == 3                        # one weight per topic
    assert abs(float(sum(td)) - 1.0) < 1e-6    # simplex
    # Head seeded and left at init on the unsupervised path.
    assert np.allclose(model.headWeights(), 0.0)
    assert model.headWeights().shape == (1, 3)


def test_pc_shim_transform_is_deterministic(spark):
    from spark_vi.mllib.topic.pc import PCEstimator
    rows, cols, V = _block_rows(seed=1)
    df = spark.createDataFrame(rows, cols)
    model = PCEstimator(k=3, maxIter=4, seed=0, subsamplingRate=1.0).fit(df)

    a1 = [r[0] for r in model.transform(df).select("topicDistribution").collect()]
    a2 = [r[0] for r in model.transform(df).select("topicDistribution").collect()]
    assert len(a1) == len(a2) == len(rows)
    for v1, v2 in zip(a1, a2):
        assert np.array_equal(v1.toArray(), v2.toArray())


def test_pc_shim_threads_label_columns_when_present(spark):
    # labelCol/labelMaskCol are threaded into every PCDocument (STM-style) and
    # tolerated at weightY == 0 — the fit succeeds and is unaffected by them.
    from spark_vi.mllib.topic.pc import PCEstimator
    rows, cols, V = _block_rows(seed=2, with_labels=True)
    df = spark.createDataFrame(rows, cols)
    model = PCEstimator(k=3, maxIter=4, seed=0, subsamplingRate=1.0,
                        numLabels=1, labelCol="label").fit(df)
    out = model.transform(df)
    assert out.select("topicDistribution").head()[0] is not None


def test_pc_shim_weight_y_default_zero_and_settable():
    from spark_vi.mllib.topic.pc import PCEstimator
    est = PCEstimator(k=3)
    assert est.getOrDefault("weightY") == 0.0        # increment-1 default
    assert est.getOrDefault("numLabels") == 1
    est2 = PCEstimator(k=3, weightY=5.0, numLabels=2)
    assert est2.getOrDefault("weightY") == 5.0
    assert est2.getOrDefault("numLabels") == 2


def test_pc_shim_weight_y_positive_requires_label_col(spark):
    # Supervised fit without labels is a user error (the head would see no
    # signal); fail fast rather than silently run an unsupervised fit under a
    # supervised name.
    from spark_vi.mllib.topic.pc import PCEstimator
    rows, cols, V = _block_rows(seed=3)
    df = spark.createDataFrame(rows, cols)
    with pytest.raises(ValueError, match="requires labelCol"):
        PCEstimator(k=3, maxIter=2, weightY=1.0).fit(df)


def test_pc_shim_supervised_fit_moves_head_and_emits_probability(spark):
    # weightY > 0 with a labelCol: the head moves off its zero seed and transform
    # appends the head-derived probabilityCol alongside topicDistribution.
    from spark_vi.mllib.topic.pc import PCEstimator
    rows, cols, V = _block_rows(seed=4, with_labels=True)
    df = spark.createDataFrame(rows, cols)
    model = PCEstimator(
        k=3, maxIter=6, seed=0, subsamplingRate=1.0,
        numLabels=1, labelCol="label", weightY=50.0, gradCaviIters=10,
    ).fit(df)

    # Head trained off zero (supervised correction actually ran).
    assert not np.allclose(model.headWeights(), 0.0)
    assert model.headWeights().shape == (1, 3)

    out = model.transform(df)
    assert "topicDistribution" in out.columns
    assert "probability" in out.columns
    row = out.select("topicDistribution", "probability").head()
    td, prob = row[0], row[1]
    assert len(td) == 3 and abs(float(sum(td)) - 1.0) < 1e-6   # simplex theta
    assert len(prob) == 1                                       # one P(y=1) per label
    assert 0.0 <= float(prob[0]) <= 1.0                         # a probability
