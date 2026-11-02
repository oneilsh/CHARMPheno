"""DagHeadsEstimator / DagHeadsModel (ADR 0051 §4; spec 2026-10-30 R6, R8, R13): the
readout heads as an MLlib stage on the DAG-gated estimator's inputs.

The heads take a labels column in user node ids plus a DAG (node set learned at fit)
or a fitted DagGatedLDAModel (its node set reused), build the closure labels and
masks themselves, and answer by node. Checked against `readout.fit_heads` on dense
targets written out by hand (the pre-redesign input contract), so the heads are the
same numbers as before for the same labels."""
from __future__ import annotations

import logging

import numpy as np
import pytest

pyspark = pytest.importorskip("pyspark")
from pyspark.ml.linalg import Vectors  # noqa: E402

PARENT = {1: None, 2: None, 3: 1, 4: 1, 5: 2, 6: 2}
UP = {3: 1, 4: 1, 5: 2, 6: 2}                 # leaf -> top-level node
C = 7                                         # root + 6 nodes; index n = node n here
K = 6


def _closure(labels):
    """Hand-written closure over PARENT in output indices (root 0)."""
    act = set()
    for n in labels or []:
        act |= {0, n} | ({UP[n]} if n in UP else set())
    return act


def _dense(labels, mode="full"):
    act = _closure(labels)
    y = [1.0 if c in act else 0.0 for c in range(C)]
    if mode == "full":
        return y, [1.0] * C
    obs = set(act)
    for c in act:
        if c in (1, 2):
            obs |= {1, 2}
        elif c in UP:
            obs |= {n for n, p in UP.items() if p == UP[c]}
    return y, [1.0 if c in obs else 0.0 for c in range(C)]


def _rows(n=300, seed=0, background=0.15):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(4, K)) * 3
    rows = []
    for _ in range(n):
        th = rng.dirichlet(np.full(K, 0.5))
        if rng.random() < background:
            labels = [] if rng.random() < 0.5 else None
        else:
            leaf = 3 + int(np.argmax(W @ th + rng.gumbel(size=4)))
            labels = [leaf] + ([UP[leaf]] if rng.random() < 0.3 else [])
        rows.append((th, labels))
    return rows


def _schema(**cols):
    """A StructType from `name=type` ('vector', 'array<int>', 'array<double>')."""
    from pyspark.ml.linalg import VectorUDT
    from pyspark.sql.types import (ArrayType, DoubleType, IntegerType, StructField,
                                   StructType)
    t = {"vector": VectorUDT(), "array<int>": ArrayType(IntegerType()),
         "array<double>": ArrayType(DoubleType())}
    return StructType([StructField(n, t[v]) for n, v in cols.items()])


def _df(spark, rows, mode="full"):
    out = []
    for th, labels in rows:
        y, m = _dense(labels, mode)
        out.append((Vectors.dense(th), labels, y, m))
    return spark.createDataFrame(out, _schema(
        topicDistribution="vector", labels="array<int>", label="array<double>",
        labelMask="array<double>"))


def _reference(df, C=C, K=K, **kw):
    from spark_vi.readout import SparkReadoutData, fit_heads
    return fit_heads(SparkReadoutData(df, C, K), **kw)


def test_defaults_and_dag_param():
    from spark_vi.mllib import DagHeadsEstimator
    est = DagHeadsEstimator()
    assert est.getOrDefault("featuresCol") == "topicDistribution"
    assert est.getOrDefault("labelsCol") == "labels"
    assert est.getOrDefault("labelMaskMode") == "full"
    assert est.getDag() == {}
    est.setDag({3: 1, 4: [1, "x"]})
    assert est.getDag() == {1: [], "x": [], 3: [1], 4: [1, "x"]}
    assert DagHeadsEstimator(dag=PARENT).getDag()[3] == [1]
    for gone in ("labelCol", "labelMaskCol", "dagParent"):
        assert not est.hasParam(gone)


def test_set_topic_model_refuses_other_objects():
    from spark_vi.mllib import DagHeadsEstimator
    with pytest.raises(TypeError, match="DagGatedLDAModel"):
        DagHeadsEstimator().setTopicModel(object())


@pytest.mark.slow
@pytest.mark.parametrize("engine", ["spark", "local"])
def test_from_dag_equals_the_readout_on_dense_labels(spark, engine):
    from spark_vi.dag.ops import closure_matrix
    from spark_vi.mllib import DagHeadsEstimator
    from spark_vi.readout.stacked import stacked_proba

    df = _df(spark, _rows()).cache()
    model = DagHeadsEstimator(maxIter=200, engine=engine, dag=PARENT,
                              stackedProbabilityCol="stacked").fit(
        df.select("topicDistribution", "labels"))
    V, b, const, deg, _ = _reference(df, max_iter=200)
    np.testing.assert_allclose(model.V, V, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(model.b_raw, b, rtol=1e-6, atol=1e-8)
    assert model.nodes() == [1, 2, 3, 4, 5, 6]
    assert [model.nodeIndex(n) for n in model.nodes()] == list(range(1, C))
    assert model.nodeIndex(None) == 0 and model.degenerateNodes() == {}
    rows = model.transform(df).select("topicDistribution", "nodeProbability",
                                      "stacked").collect()
    th = np.stack([r[0].toArray() for r in rows])
    p = np.stack([r[1].toArray() for r in rows])
    np.testing.assert_allclose(p, 1 / (1 + np.exp(-(th @ model.V.T + model.b_raw))),
                               atol=1e-6)
    M = closure_matrix({c: [UP.get(c, 0)] for c in range(1, C)}, C)
    np.testing.assert_allclose(np.stack([r[2].toArray() for r in rows]),
                               stacked_proba(p, M), atol=1e-6)


@pytest.mark.slow
def test_closure_mask_mode_equals_the_readout(spark):
    from spark_vi.mllib import DagHeadsEstimator

    df = _df(spark, _rows(seed=1), mode="closure").cache()
    model = DagHeadsEstimator(maxIter=200, dag=PARENT, labelMaskMode="closure").fit(df)
    V, b, const, deg, _ = _reference(df, max_iter=200)
    np.testing.assert_allclose(model.V, V, rtol=1e-6, atol=1e-8)
    np.testing.assert_array_equal(model.degenerate, deg)
    lab = model.attachLabels(df, labelCol="y", labelMaskCol="m").select(
        "label", "labelMask", "y", "m").collect()
    for r in lab:
        np.testing.assert_array_equal(r["y"].toArray(), r["label"])
        np.testing.assert_array_equal(r["m"].toArray(), r["labelMask"])


@pytest.mark.slow
def test_degenerate_nodes_reported_and_logged(spark, capsys):
    from spark_vi.mllib import DagHeadsEstimator

    df = _df(spark, _rows(n=120, background=0.0))      # root: every row positive
    model = DagHeadsEstimator(maxIter=30, dag=PARENT).fit(df)
    assert model.degenerateNodes() == {None: 1.0}
    assert "1 degenerate node(s)" in capsys.readouterr().out
    p = model.transform(df).select("nodeProbability").head()[0].toArray()
    assert p[0] == 1.0


@pytest.mark.slow
def test_node_set_learning_prunes_and_lands(spark):
    from spark_vi.mllib import DagHeadsEstimator

    rows = [r for r in _rows(n=200, seed=2) if r[1] != [6] and r[1] != [6, 2]]
    rows.append((np.full(K, 1.0 / K), [6]))           # node 6: support 1
    df = _df(spark, rows).cache()
    model = DagHeadsEstimator(maxIter=100, dag=PARENT, minDocsPerNode=2).fit(df)
    assert model.prunedNodes() == {6: [2]}
    assert 6 not in model.nodes()
    with pytest.raises(KeyError, match="pruned"):
        model.nodeIndex(6)
    # The [6] document is a positive for 2 (its landing), as dense labels say.
    two = model.nodeIndex(2)
    lab = model.attachLabels(df.where("array_contains(labels, 6)")).head()
    assert lab["label"][two] == 1.0 and lab["label"].toArray().sum() == 2.0
    # Same heads as fit_heads on the hand-built targets with column 6 dropped (the
    # [6] document's closure already holds 2, its landing).
    from pyspark.sql import functions as F
    assert [model.nodeIndex(n) for n in model.nodes()] == model.nodes()
    keep = [0] + model.nodes()
    sub = df.select("topicDistribution",
                    F.array(*[F.col("label")[c] for c in keep]).alias("label"),
                    F.array(*[F.col("labelMask")[c] for c in keep]).alias("labelMask"))
    V, b, _, _, _ = _reference(sub, C=len(keep), max_iter=100)
    np.testing.assert_allclose(model.V, V, rtol=1e-6, atol=1e-8)


@pytest.mark.slow
def test_unknown_labels_error_or_ignore(spark):
    from spark_vi.mllib import DagHeadsEstimator

    rows = _rows(n=80, seed=3) + [(np.full(K, 1.0 / K), [3, 99])]
    df = spark.createDataFrame([(Vectors.dense(t), l) for t, l in rows],
                               _schema(topicDistribution="vector", labels="array<int>"))
    with pytest.raises(ValueError, match="99"):
        DagHeadsEstimator(maxIter=5, dag=PARENT).fit(df)
    m = DagHeadsEstimator(maxIter=5, dag=PARENT, unknownLabels="ignore").fit(df)
    assert m.nodes() == [1, 2, 3, 4, 5, 6]
    with pytest.raises(ValueError, match="every DAG node id is a str"):
        DagHeadsEstimator(dag={"a": None}).fit(df)


def _topic_df(spark, n=80, seed=0):
    """Bag-of-words docs with gate labels (`labels`) and a different readout
    attestation (`readout`, R8): the readout labels never mention node 6."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        leaf = [3, 4, 5, 6][i % 4]
        idx = sorted(rng.choice(30, size=6, replace=False).tolist())
        readout = [] if i % 5 == 0 else [[3, 4, 5, 5][i % 4]]
        rows.append((Vectors.sparse(30, idx, [1.0] * 6), [leaf], readout))
    rows.append((Vectors.sparse(30, [0], [1.0]), [6, 7], [3]))   # 7: rare, pruned
    return spark.createDataFrame(rows, _schema(features="vector", labels="array<int>",
                                               readout="array<int>"))


@pytest.mark.slow
def test_from_topic_model_reuses_its_node_set(spark):
    from pyspark.sql import functions as F

    from spark_vi.mllib import DagGatedLDAEstimator, DagHeadsEstimator

    dag = {**PARENT, 7: 6}
    df = _topic_df(spark)
    lda = DagGatedLDAEstimator(gateNBg=2, gateTpn=1, maxIter=3, seed=0,
                               subsamplingRate=1.0, minDocsPerNode=2, dag=dag).fit(df)
    assert lda.prunedNodes() == {7: [6]}
    scored = lda.transform(df).cache()
    heads = DagHeadsEstimator(maxIter=100, labelsCol="readout").setTopicModel(lda)
    model = heads.fit(scored)
    kept = list(lda._ns().nodes)
    assert model.nodes() == kept
    assert model.dag() == lda.dag() and model.prunedNodes() == lda.prunedNodes()
    # Node 6 is in the topic model's node set but never in the readout labels: an
    # all-negative head, a constant 0.
    assert model.degenerateNodes() == {6: 0.0}
    # Reference: dense targets over the topic model's engine ids.
    eng = lda._ns().engine_id
    par = {e: ps for e, ps in lda._ns().engine_parents.items()}

    def clo(e):
        out = {e}
        for p in par.get(e, []):
            out |= clo(p)
        return out
    Cm = len(kept) + 1

    def dense(labels):
        act = set()
        for n in labels or []:
            act |= clo(eng[n]) | {0}
        return [1.0 if c in act else 0.0 for c in range(Cm)], [1.0] * Cm
    ref = spark.createDataFrame(
        [(r["topicDistribution"],) + tuple(dense(r["readout"]))
         for r in scored.select("topicDistribution", "readout").collect()],
        _schema(topicDistribution="vector", label="array<double>",
                labelMask="array<double>"))
    V, b, _, deg, _ = _reference(ref, C=Cm, K=ref.head()[0].size, max_iter=100)
    np.testing.assert_allclose(model.V, V, rtol=1e-6, atol=1e-8)
    # Labels a pruned gate node carries land as the topic model landed them.
    lab = model.attachLabels(scored.select("readout").limit(1).withColumn(
        "readout", F.array(F.lit(7)))).head()
    assert lab["label"][model.nodeIndex(6)] == 1.0
    # The DAG and minDocsPerNode belong to the DAG path.
    with pytest.raises(ValueError, match="not both"):
        DagHeadsEstimator(labelsCol="readout", dag=dag).setTopicModel(lda).fit(scored)
    with pytest.raises(ValueError, match="minDocsPerNode"):
        DagHeadsEstimator(labelsCol="readout", minDocsPerNode=3).setTopicModel(
            lda).fit(scored)
    bad = scored.withColumn("readout", F.array(F.lit(42)))
    with pytest.raises(ValueError, match="42"):
        heads.fit(bad)
    with pytest.raises(ValueError, match="needs a label DAG"):
        DagHeadsEstimator().fit(scored)


@pytest.mark.slow
def test_save_load_round_trip_keeps_the_node_mapping(spark, tmp_path):
    from spark_vi.mllib import DagHeadsEstimator, DagHeadsModel

    df = _df(spark, _rows(n=120, background=0.0))
    model = DagHeadsEstimator(maxIter=50, thetaTopM=3, dag=PARENT).fit(df)
    uri = f"file://{tmp_path}/heads"
    model.save(uri)
    back = DagHeadsModel.load(uri)
    np.testing.assert_array_equal(back.V, model.V)
    assert back.theta_topm == 3 and back.getOrDefault("thetaTopM") == 3
    assert back.nodes() == model.nodes() and back.dag() == model.dag()
    assert back.degenerateNodes() == model.degenerateNodes() == {None: 1.0}
    a = model.transform(df).select("nodeProbability").collect()
    b = back.transform(df).select("nodeProbability").collect()
    assert [r[0] for r in a] == [r[0] for r in b]


@pytest.mark.slow
def test_save_dir_checkpoint_and_resume(spark, tmp_path, monkeypatch, caplog):
    import spark_vi.readout.heads as heads_mod
    from spark_vi.io import fs
    from spark_vi.mllib import DagHeadsEstimator, DagHeadsModel

    df = _df(spark, _rows(n=200, seed=4)).cache()
    base = f"file://{tmp_path}"
    # A finished fit writes the model to saveDir and removes its checkpoint.
    full = DagHeadsEstimator(maxIter=200, dag=PARENT, saveDir=f"{base}/full",
                             saveInterval=2).fit(df)
    assert not fs.exists(f"{base}/full/heads_ckpt.npz")
    np.testing.assert_array_equal(DagHeadsModel.load(f"{base}/full").V, full.V)

    # A "crashed" fit: the checkpoint survives (its removal is suppressed).
    monkeypatch.setattr(heads_mod.fs, "remove", lambda uri: None)
    DagHeadsEstimator(maxIter=3, dag=PARENT, saveDir=f"{base}/crash",
                      saveInterval=1).fit(df)
    monkeypatch.undo()
    assert fs.exists(f"{base}/crash/heads_ckpt.npz")
    with caplog.at_level(logging.INFO, logger="spark_vi.readout.heads"):
        resumed = DagHeadsEstimator(maxIter=200, dag=PARENT,
                                    resumeFrom=f"{base}/crash").fit(df)
    assert "resuming batched solve from checkpoint" in caplog.text
    np.testing.assert_allclose(resumed.V, full.V, rtol=1e-3, atol=1e-3)

    # Resume from a finished model: a warm start at its optimum stays there.
    warm = DagHeadsEstimator(maxIter=1, dag=PARENT, resumeFrom=f"{base}/full").fit(df)
    np.testing.assert_allclose(warm.V, full.V, rtol=1e-3, atol=1e-3)
    cold = DagHeadsEstimator(maxIter=1, dag=PARENT).fit(df)
    assert np.abs(cold.V - full.V).max() > 1e-2

    # Without resumeFrom a saveDir holding a checkpoint is NOT read.
    monkeypatch.setattr(heads_mod.fs, "remove", lambda uri: None)
    DagHeadsEstimator(maxIter=3, dag=PARENT, saveDir=f"{base}/crash2",
                      saveInterval=1).fit(df)
    monkeypatch.undo()
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="spark_vi.readout.heads"):
        DagHeadsEstimator(maxIter=2, dag=PARENT, saveDir=f"{base}/crash2",
                          saveInterval=1).fit(df)
    assert "resuming" not in caplog.text

    with pytest.raises(ValueError, match="saveInterval=0"):
        DagHeadsEstimator(dag=PARENT, saveInterval=0).fit(df)
    with pytest.raises(ValueError, match="requires saveDir"):
        DagHeadsEstimator(dag=PARENT, saveInterval=5).fit(df)
    with pytest.raises(FileNotFoundError):
        DagHeadsEstimator(dag=PARENT, resumeFrom=f"{base}/nothing").fit(df)
    from pyspark.sql import functions as F
    eight = df.withColumn("labels", F.when(F.rand(0) < 0.1, F.array(F.lit(8)))
                          .otherwise(F.col("labels")))
    with pytest.raises(ValueError, match="other nodes"):
        DagHeadsEstimator(maxIter=2, dag={**PARENT, 8: 6},
                          resumeFrom=f"{base}/full").fit(eight)


@pytest.mark.slow
def test_local_from_labels_equals_the_estimator(spark):
    from spark_vi.mllib import DagHeadsEstimator
    from spark_vi.readout import LocalReadoutData, fit_heads

    rows = _rows(n=150, seed=5)
    df = _df(spark, rows)
    model = DagHeadsEstimator(maxIter=100, dag=PARENT, labelMaskMode="closure",
                              engine="local").fit(df)
    data = LocalReadoutData.from_labels(np.stack([r[0] for r in rows]),
                                        [r[1] for r in rows], model._labeler())
    V, b, _, _, _ = fit_heads(data, max_iter=100)
    np.testing.assert_allclose(model.V, V, rtol=1e-6, atol=1e-8)


@pytest.mark.slow
def test_pipeline_gated_lda_then_heads(spark):
    from pyspark.ml import Pipeline

    from spark_vi.mllib import DagGatedLDAEstimator, DagHeadsEstimator

    df = _topic_df(spark).drop("readout").where("NOT array_contains(labels, 7)")
    lda = DagGatedLDAEstimator(gateNBg=2, gateTpn=1, maxIter=3, seed=0,
                               subsamplingRate=1.0, dag=PARENT)
    heads = DagHeadsEstimator(maxIter=30, stackedProbabilityCol="stacked", dag=PARENT)
    fitted = Pipeline(stages=[lda, heads]).fit(df)
    lda_m, heads_m = fitted.stages
    assert heads_m.nodes() == list(lda_m._ns().nodes)   # same labels, same node set
    r = fitted.transform(df).select("nodeProbability", "stacked").head()
    assert r[0].size == C and r[1].size == C
    assert np.all(r[1].toArray() <= r[0].toArray() + 1e-6)   # product <= own factor
