"""Increment-1 shim smoke tests for OnlinePCLDAEstimator / OnlinePCLDAModel (weightY == 0).

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
    from spark_vi.mllib.topic.pc import OnlinePCLDAEstimator
    rows, cols, V = _block_rows(seed=0)
    df = spark.createDataFrame(rows, cols)

    est = OnlinePCLDAEstimator(k=3, maxIter=4, seed=0, subsamplingRate=1.0,
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


def test_pc_shim_readouts_parity_with_lda(spark):
    """OnlinePCLDAModel exposes the LDA-parity topic readouts: describeTopics,
    trainedAlpha (length K), trainedTopicConcentration (float). logLikelihood /
    logPerplexity are honest v1 NotImplementedError stubs, as in the LDA shim."""
    from spark_vi.mllib.topic.pc import OnlinePCLDAEstimator
    rows, cols, V = _block_rows(seed=1)
    df = spark.createDataFrame(rows, cols)
    model = OnlinePCLDAEstimator(k=3, maxIter=4, seed=0, subsamplingRate=1.0).fit(df)

    # trainedAlpha -> length-K vector; trainedTopicConcentration -> float.
    assert model.trainedAlpha().shape == (3,)
    assert isinstance(model.trainedTopicConcentration(), float)

    # describeTopics -> (topic, termIndices, termWeights), K rows, m terms each,
    # weights descending in (0, 1] (top of a row-stochastic beta).
    dt = model.describeTopics(maxTermsPerTopic=4).collect()
    assert len(dt) == 3
    assert set(dt[0].asDict()) == {"topic", "termIndices", "termWeights"}
    assert len(dt[0]["termIndices"]) == 4 and len(dt[0]["termWeights"]) == 4
    w = dt[0]["termWeights"]
    assert all(w[i] >= w[i + 1] for i in range(len(w) - 1))
    assert 0.0 < w[0] <= 1.0
    with pytest.raises(ValueError):
        model.describeTopics(maxTermsPerTopic=0)

    # v1 stubs, parity with OnlineLDAModel.
    for fn in (model.logLikelihood, model.logPerplexity):
        with pytest.raises(NotImplementedError):
            fn(df)


def test_pc_shim_transform_is_deterministic(spark):
    from spark_vi.mllib.topic.pc import OnlinePCLDAEstimator
    rows, cols, V = _block_rows(seed=1)
    df = spark.createDataFrame(rows, cols)
    model = OnlinePCLDAEstimator(k=3, maxIter=4, seed=0, subsamplingRate=1.0).fit(df)

    a1 = [r[0] for r in model.transform(df).select("topicDistribution").collect()]
    a2 = [r[0] for r in model.transform(df).select("topicDistribution").collect()]
    assert len(a1) == len(a2) == len(rows)
    for v1, v2 in zip(a1, a2):
        assert np.array_equal(v1.toArray(), v2.toArray())


def test_pc_shim_threads_label_columns_when_present(spark):
    # labelCol/labelMaskCol are threaded into every PCDocument (STM-style) and
    # tolerated at weightY == 0 — the fit succeeds and is unaffected by them.
    from spark_vi.mllib.topic.pc import OnlinePCLDAEstimator
    rows, cols, V = _block_rows(seed=2, with_labels=True)
    df = spark.createDataFrame(rows, cols)
    model = OnlinePCLDAEstimator(k=3, maxIter=4, seed=0, subsamplingRate=1.0,
                        numLabels=1, labelCol="label").fit(df)
    out = model.transform(df)
    assert out.select("topicDistribution").head()[0] is not None


def test_pc_shim_weight_y_default_zero_and_settable():
    from spark_vi.mllib.topic.pc import OnlinePCLDAEstimator
    est = OnlinePCLDAEstimator(k=3)
    assert est.getOrDefault("weightY") == 0.0        # increment-1 default
    assert est.getOrDefault("numLabels") == 1
    est2 = OnlinePCLDAEstimator(k=3, weightY=5.0, numLabels=2)
    assert est2.getOrDefault("weightY") == 5.0
    assert est2.getOrDefault("numLabels") == 2


def test_pc_shim_weight_y_positive_requires_label_col(spark):
    # Supervised fit without labels is a user error (the head would see no
    # signal); fail fast rather than silently run an unsupervised fit under a
    # supervised name.
    from spark_vi.mllib.topic.pc import OnlinePCLDAEstimator
    rows, cols, V = _block_rows(seed=3)
    df = spark.createDataFrame(rows, cols)
    with pytest.raises(ValueError, match="requires labelCol"):
        OnlinePCLDAEstimator(k=3, maxIter=2, weightY=1.0).fit(df)


def test_pc_shim_dag_closure_head_end_to_end(spark):
    # closureParents selects the DAG-closure head: supervised fit trains it (Newton),
    # and transform's probabilityCol is the closure PRODUCT P(node_l) — monotone
    # P(child) <= P(parent) — NOT the flat sigmoid.
    import json
    from spark_vi.mllib.topic.pc import OnlinePCLDAEstimator
    from pyspark.ml.linalg import Vectors
    parents = [[], [0], [0], [1, 2]]                 # 0 root; 1,2 under 0; 3 diamond under 1&2
    rng = np.random.default_rng(0)
    V = 12
    rows = []
    for _ in range(24):
        counts = np.zeros(V)
        for w in rng.integers(0, V, size=8):
            counts[w] += 1.0
        idx = sorted(np.nonzero(counts)[0].tolist())
        fv = Vectors.sparse(V, idx, [float(counts[j]) for j in idx])
        y = [float(x) for x in rng.integers(0, 2, size=4)]
        rows.append((fv, y))
    df = spark.createDataFrame(rows, ["features", "label"])

    model = OnlinePCLDAEstimator(
        k=4, maxIter=6, seed=0, subsamplingRate=1.0,
        numLabels=4, labelCol="label", weightY=50.0, gradCaviIters=10,
        headOptimizer="newton", closureParents=json.dumps(parents),
    ).fit(df)
    assert not np.allclose(model.headWeights(), 0.0)     # DAG head trained (Newton)

    P = np.asarray(model.transform(df).select("probability").head()[0].toArray())
    assert P.shape == (4,)
    assert P[1] <= P[0] + 1e-9 and P[2] <= P[0] + 1e-9   # closure-product monotonicity
    assert P[3] <= P[1] + 1e-9 and P[3] <= P[2] + 1e-9


def test_pc_shim_gated_pc_end_to_end(spark):
    # gateParent injects the GATED topic engine (Gated-PC, ADR 0042): K is derived from
    # the layout (overriding k), each row's frontierCol gates its topic training, and the
    # (DAG-closure) head rides on the ungated theta. Fit + transform end-to-end; the
    # probability column is the closure product over the gate-derived topics.
    import json
    from spark_vi.mllib.topic.pc import OnlinePCLDAEstimator
    from pyspark.ml.linalg import Vectors
    parent = {1: 0, 2: 0, 3: 1, 4: 1}               # nodes 1..4; K = nBg(3) + 4 = 7
    closure = [[], [0], [0], [1], [1]]              # C = 5 label heads (ids 0..4)
    rng = np.random.default_rng(0)
    V = 21
    rows = []
    for _ in range(40):
        leaf = int(rng.choice([3, 4]))
        ids = {leaf, 1, 0}                          # closure of a leaf under node 1
        counts = np.zeros(V)
        for w in rng.integers(0, V, size=10):
            counts[w] += 1.0
        idx = sorted(np.nonzero(counts)[0].tolist())
        fv = Vectors.sparse(V, idx, [float(counts[j]) for j in idx])
        y = [1.0 if c in ids else 0.0 for c in range(5)]
        rows.append((fv, y, [leaf]))
    df = spark.createDataFrame(rows, ["features", "label", "frontier"])

    est = OnlinePCLDAEstimator(
        k=99, maxIter=6, seed=0, subsamplingRate=1.0,
        numLabels=5, labelCol="label", weightY=30.0, gradCaviIters=10,
        headOptimizer="newton", headLr=0.7, closureParents=json.dumps(closure),
        frontierCol="frontier",
    ).setGateParent(parent)
    est._set(gateNBg=3, gateTpn=1)
    model = est.fit(df)

    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    # K comes from the layout (3 + 4*1 = 7), NOT the k=99 param.
    assert model._result.global_params["lambda"].shape[0] == 7
    assert not np.allclose(model.headWeights(), 0.0)          # gated head trained
    P = np.asarray(model.transform(df).select("probability").head()[0].toArray())
    assert P.shape == (5,)
    assert P[1] <= P[0] + 1e-9 and P[3] <= P[1] + 1e-9        # closure monotonicity holds


def test_pc_shim_gated_requires_frontier_col(spark):
    # gateParent set but frontierCol missing from the input -> fail fast.
    from spark_vi.mllib.topic.pc import OnlinePCLDAEstimator
    rows, cols, V = _block_rows(seed=1, with_labels=True)
    df = spark.createDataFrame(rows, cols)                    # no 'frontier' column
    est = OnlinePCLDAEstimator(
        k=5, maxIter=2, numLabels=1, labelCol="label", weightY=1.0,
    ).setGateParent({1: 0})
    with pytest.raises(ValueError, match="frontierCol"):
        est.fit(df)


def test_pc_shim_rejects_closure_parents_count_mismatch(spark):
    # closureParents length must equal numLabels — fail fast with a clear message.
    import json
    from spark_vi.mllib.topic.pc import OnlinePCLDAEstimator
    rows, cols, V = _block_rows(seed=5, with_labels=True)
    df = spark.createDataFrame(rows, cols)
    with pytest.raises(ValueError, match="one node per label head"):
        OnlinePCLDAEstimator(
            k=3, maxIter=2, numLabels=1, labelCol="label", weightY=1.0,
            closureParents=json.dumps([[], [0], [0]]),   # 3 nodes vs numLabels=1
        ).fit(df)


def test_pc_shim_supervised_fit_moves_head_and_emits_probability(spark):
    # weightY > 0 with a labelCol: the head moves off its zero seed and transform
    # appends the head-derived probabilityCol alongside topicDistribution.
    from spark_vi.mllib.topic.pc import OnlinePCLDAEstimator
    rows, cols, V = _block_rows(seed=4, with_labels=True)
    df = spark.createDataFrame(rows, cols)
    model = OnlinePCLDAEstimator(
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


def test_pc_shim_multidomain_gated_pc_end_to_end(spark):
    """Multi-domain Gated-PC through the shim: two per-domain feature columns
    (featuresCols) inject the per-domain-lambda GatedOnlineLDA, a supervised
    weightY>0 fit runs the DOMAIN-AWARE topic correction (dict-lambda), and both
    fit (dict-lambda blocks) and transform (per-domain feature concatenation)
    round-trip. This is the 30c shim gate for the multi-domain PC path."""
    from spark_vi.mllib.topic.pc import OnlinePCLDAEstimator
    from pyspark.ml.linalg import Vectors
    parent = {1: 0, 2: 0}                          # 2 disease nodes; K = nBg(2)+2 = 4
    V0, V1 = 16, 9                                  # domain 0 conditions, 1 measurement
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(48):
        node = int(rng.choice([1, 2]))
        # domain 0: a node-correlated word band; domain 1: a node-correlated band.
        c0 = np.zeros(V0)
        for w in rng.integers(node * 5, node * 5 + 5, size=6):
            c0[w % V0] += 1.0
        c1 = np.zeros(V1)
        for w in rng.integers(node * 3, node * 3 + 3, size=4):
            c1[w % V1] += 1.0
        i0 = sorted(np.nonzero(c0)[0].tolist())
        i1 = sorted(np.nonzero(c1)[0].tolist())
        f0 = Vectors.sparse(V0, i0, [float(c0[j]) for j in i0])
        f1 = Vectors.sparse(V1, i1, [float(c1[j]) for j in i1])
        y = [1.0, 1.0 if node == 1 else 0.0, 1.0 if node == 2 else 0.0]   # C=3: root+2
        rows.append((f0, f1, y, [node]))
    df = spark.createDataFrame(rows, ["features_0", "features_1", "label", "frontier"])

    est = OnlinePCLDAEstimator(
        k=99, maxIter=6, seed=0, subsamplingRate=1.0,
        featuresCols=["features_0", "features_1"],
        numLabels=3, labelCol="label", weightY=30.0, gradCaviIters=10,
        headOptimizer="newton", headLr=0.7, frontierCol="frontier",
    ).setGateParent(parent)
    est._set(gateNBg=2, gateTpn=1)
    model = est.fit(df)

    lam = model._result.global_params["lambda"]
    assert isinstance(lam, dict) and set(lam) == {0, 1}        # per-domain dict λ
    assert lam[0].shape == (4, V0) and lam[1].shape == (4, V1)  # K from layout, per-domain V
    assert not np.allclose(model.headWeights(), 0.0)           # supervised correction ran
    assert model.headWeights().shape == (3, 4)

    out = model.transform(df)
    td, prob = out.select("topicDistribution", "probability").head()
    assert len(td) == 4 and abs(float(sum(td)) - 1.0) < 1e-6   # simplex θ over K=4
    assert len(prob) == 3 and all(0.0 <= float(p) <= 1.0 for p in prob)


def test_pc_shim_multidomain_requires_gate(spark):
    """featuresCols without gateParent fails fast: the per-node gate is the shared
    structure the per-domain blocks specialize under."""
    from spark_vi.mllib.topic.pc import OnlinePCLDAEstimator
    from pyspark.ml.linalg import Vectors
    rows = [(Vectors.sparse(4, [0], [1.0]), Vectors.sparse(3, [1], [1.0]), [1.0])
            for _ in range(4)]
    df = spark.createDataFrame(rows, ["features_0", "features_1", "label"])
    with pytest.raises(ValueError, match="require gateParent"):
        OnlinePCLDAEstimator(
            k=3, maxIter=2, numLabels=1, labelCol="label", weightY=1.0,
            featuresCols=["features_0", "features_1"],
        ).fit(df)
