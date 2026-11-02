"""MLlib persistence of the DAG-gated stages (spec 2026-10-30 R13 Level 2).

DagGatedLDAEstimator / DagGatedLDAModel and DagHeadsEstimator / DagHeadsModel are
``MLWritable`` / ``MLReadable``: MLlib's Params metadata plus our data files (the
VIResult, the heads npz, the node set), so each saves and loads on its own and inside
a ``Pipeline`` / ``PipelineModel``, ``write().overwrite()`` replaces a save, and the
JSON-valued Params (the DAG, vocabularies, nodeTerms, gateSeedProfile) keep their
str / int ids (1 and "1" stay different)."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

pyspark = pytest.importorskip("pyspark")
from pyspark.ml import Pipeline, PipelineModel  # noqa: E402
from pyspark.ml.linalg import Vectors  # noqa: E402

from test_mllib_dag_gated_lda import (  # noqa: E402
    MIXED,
    PARENT,
    _df,
    _est,
    _str_dag_df,
    _str_est,
)

# A DAG where 1 and "1" are different nodes.
TWIN = {1: None, "1": 1, 2: "1", "x": [1, 2]}


def _theta(df, col="topicDistribution"):
    return np.array([r[0].toArray() for r in df.select(col).collect()])


def _uri(tmp_path, name):
    return f"file://{tmp_path}/{name}"


# --------------------------------------------------------------------------- #
# DagGatedLDAEstimator: Params only                                            #
# --------------------------------------------------------------------------- #

def test_estimator_round_trip_keeps_mixed_id_params(spark, tmp_path):
    from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAEstimator
    est = DagGatedLDAEstimator.recommended(
        dag=TWIN, minDocsPerNode=7, seed=3, featuresCols=["f0", "f1"],
        vocabularies=[["t0", "t1"], [1, "1", "t0"]],
        nodeTerms={1: ["t1"], "1": [("f1", 1)], 2: [("f1", "1"), ("f0", "t0")]},
        gateSeedProfile={"1": {("f1", "t0"): 2.0}, 1: {"t1": 0.5}})
    est.write().save(_uri(tmp_path, "est"))
    back = DagGatedLDAEstimator.load(_uri(tmp_path, "est"))
    assert back.uid == est.uid
    assert back.getDag() == est.getDag() == {1: [], "1": [1], 2: ["1"], "x": [1, 2]}
    assert back.getVocabularies() == [["t0", "t1"], [1, "1", "t0"]]
    assert back.getNodeTerms() == est.getNodeTerms()
    assert set(back.getNodeTerms()) == {1, "1", 2}
    assert back.getNodeTerms()["1"] == [("f1", 1)]
    assert back.getGateSeedProfile() == {"1": {("f1", "t0"): 2.0}, 1: {"t1": 0.5}}
    for p in est.params:
        assert back.isSet(p.name) == est.isSet(p), p.name
        if est.isDefined(p):
            assert back.getOrDefault(p.name) == est.getOrDefault(p), p.name
    assert back.getDocConcentration() == [0.5] and not back.isSet("topicConcentration")


# --------------------------------------------------------------------------- #
# DagGatedLDAModel: Params + the VIResult                                      #
# --------------------------------------------------------------------------- #

def test_model_round_trip_multi_domain(spark, tmp_path):
    """The str-id, two-domain fit: node set, Params (vocabularies -> describeNodes
    terms), the VIResult, and transform over featuresCols, identical after load."""
    from spark_vi.io.export import load_result
    from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAModel
    df = _str_dag_df(spark)
    model = _str_est().fit(df)
    path = str(tmp_path / "m")                     # a plain path (qualified to file:)
    model.save(path)
    assert sorted(os.listdir(path)) == ["metadata", "result"]
    # the data is the fit checkpoints' format: readable without the stage
    raw = load_result(os.path.join(path, "result"))
    assert raw.metadata["dag_node_set"] == model.result.metadata["dag_node_set"]

    back = DagGatedLDAModel.load(path)
    assert back.uid == model.uid
    for m, arr in model.result.global_params["lambda"].items():
        np.testing.assert_array_equal(back.result.global_params["lambda"][m], arr)
    assert back.result.elbo_trace == model.result.elbo_trace
    assert back.result.n_iterations == model.result.n_iterations
    assert back.dag() == model.dag() and back.prunedNodes() == model.prunedNodes()
    assert back.splicedNodes() == model.splicedNodes()
    assert back.nodeSupport() == model.nodeSupport()
    assert all(back.nodeTopics(n) == model.nodeTopics(n) for n in model.dag())
    assert back._frontier_fn() == model._frontier_fn()
    assert back.getFeaturesCols() == ["f0", "f1"]
    assert back.getVocabularies() == model.getVocabularies()
    assert back.getOrDefault("minDocsPerNode") == 2 and back.isSet("minDocsPerNode")
    a = model.describeNodes(3).orderBy("topic").collect()
    b = back.describeNodes(3).orderBy("topic").collect()
    assert a == b and a[-1].terms is not None
    np.testing.assert_array_equal(_theta(back.transform(df)), _theta(model.transform(df)))


def test_model_round_trip_mixed_int_ids(spark, tmp_path):
    from pyspark.ml.linalg import VectorUDT
    from pyspark.sql.types import ArrayType, IntegerType, StructField, StructType

    from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAModel
    rng = np.random.default_rng(0)
    rows = []
    for i in range(30):
        idx = sorted(rng.choice(10, size=3, replace=False).tolist())
        rows.append((Vectors.sparse(10, idx, [1.0] * 3), [[3], [2], [1], []][i % 4]))
    df = spark.createDataFrame(rows, StructType([
        StructField("features", VectorUDT()),
        StructField("labels", ArrayType(IntegerType()))]))
    model = _est(dag=MIXED, maxIter=2).fit(df)
    model.save(_uri(tmp_path, "m"))
    back = DagGatedLDAModel.load(_uri(tmp_path, "m"))
    assert back.dag() == model.dag() and 1 in back.dag() and "1" not in back.dag()
    assert back.getDag() == model.getDag()
    assert back.nodeTopics(1) == model.nodeTopics(1)
    np.testing.assert_array_equal(_theta(back.transform(df)), _theta(model.transform(df)))


def test_model_save_refuses_an_existing_path_then_overwrites(spark, tmp_path):
    from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAModel
    df = _df(spark)
    m1 = _est(maxIter=1).fit(df)
    m2 = _est(maxIter=2, seed=5).fit(df)
    uri = _uri(tmp_path, "m")
    m1.save(uri)
    with pytest.raises(FileExistsError, match="overwrite"):
        m2.save(uri)
    m2.write().overwrite().save(uri)
    back = DagGatedLDAModel.load(uri)
    assert back.uid == m2.uid and back.result.n_iterations == 2
    np.testing.assert_array_equal(back.result.global_params["lambda"],
                                  m2.result.global_params["lambda"])


def test_model_load_refuses_another_class_and_another_engine(spark, tmp_path):
    from spark_vi.mllib import DagHeadsModel
    from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAModel
    df = _df(spark)
    model = _est(maxIter=1).fit(df)
    path = str(tmp_path / "m")
    model.save(path)
    with pytest.raises(ValueError, match="not a spark_vi.mllib.readout.DagHeadsModel"):
        DagHeadsModel.load(path)
    man = os.path.join(path, "result", "manifest.json")
    doc = json.loads(open(man).read())
    doc["metadata"]["model_class"] = "OnlineLDA"
    open(man, "w").write(json.dumps(doc))
    with pytest.raises(ValueError, match="DagGatedOnlineLDA"):
        DagGatedLDAModel.load(path)


# --------------------------------------------------------------------------- #
# DagHeads                                                                     #
# --------------------------------------------------------------------------- #

def _heads_frame(spark, n=120, seed=0):
    from test_mllib_dag_heads import _df as heads_df
    from test_mllib_dag_heads import _rows
    return heads_df(spark, _rows(n=n, seed=seed))


def test_heads_model_round_trip(spark, tmp_path):
    from spark_vi.mllib import DagHeadsEstimator, DagHeadsModel
    df = _heads_frame(spark)
    model = DagHeadsEstimator(maxIter=40, thetaTopM=3, labelMaskMode="closure",
                              stackedProbabilityCol="stacked", dag=PARENT).fit(df)
    uri = _uri(tmp_path, "heads")
    model.save(uri)
    assert sorted(os.listdir(tmp_path / "heads")) == [
        "metadata", "node_set.json", "readout_heads_model.npz"]
    back = DagHeadsModel.load(uri)
    assert back.uid == model.uid
    for a in ("V", "b_raw", "const", "degenerate"):
        np.testing.assert_array_equal(getattr(back, a), getattr(model, a))
    assert back.nodes() == model.nodes() and back.dag() == model.dag()
    assert back.degenerateNodes() == model.degenerateNodes()
    assert back.getOrDefault("labelMaskMode") == "closure"
    assert back.getOrDefault("stackedProbabilityCol") == "stacked"
    a = model.transform(df).select("nodeProbability", "stacked").collect()
    b = back.transform(df).select("nodeProbability", "stacked").collect()
    assert a == b
    with pytest.raises(FileExistsError):
        model.save(uri)
    model.write().overwrite().save(uri)


def test_heads_model_mixed_ids(spark, tmp_path):
    """Node ids 1 and "1" stay apart through node_set.json."""
    from pyspark.ml.linalg import VectorUDT
    from pyspark.sql.types import ArrayType, StringType, StructField, StructType

    from spark_vi.mllib import DagHeadsEstimator, DagHeadsModel
    dag = {"a": None, "1": "a", "b": None}
    rng = np.random.default_rng(1)
    rows = [(Vectors.dense(rng.dirichlet(np.ones(4))), [["1"], ["b"], ["a"], []][i % 4])
            for i in range(60)]
    df = spark.createDataFrame(rows, StructType([
        StructField("topicDistribution", VectorUDT()),
        StructField("labels", ArrayType(StringType()))]))
    model = DagHeadsEstimator(maxIter=20, dag=dag).fit(df)
    model.save(_uri(tmp_path, "h"))
    back = DagHeadsModel.load(_uri(tmp_path, "h"))
    assert back.nodes() == model.nodes() and "1" in back.nodes() and 1 not in back.nodes()
    assert back.nodeIndex("1") == model.nodeIndex("1")
    assert back.getDag() == {"a": [], "1": ["a"], "b": []}


def test_heads_estimator_round_trip_with_and_without_topic_model(spark, tmp_path):
    from spark_vi.mllib import DagHeadsEstimator
    from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAModel
    est = DagHeadsEstimator(maxIter=7, regParam=0.5, labelsCol="readout", dag=TWIN)
    est.save(_uri(tmp_path, "e1"))
    back = DagHeadsEstimator.load(_uri(tmp_path, "e1"))
    assert back.getDag() == est.getDag() and back.getTopicModel() is None
    assert back.getOrDefault("maxIter") == 7 and back.getOrDefault("labelsCol") == "readout"

    df = _df(spark)
    lda = _est(maxIter=1).fit(df)
    held = DagHeadsEstimator(maxIter=7).setTopicModel(lda)
    held.save(_uri(tmp_path, "e2"))
    assert "topicModel" in os.listdir(tmp_path / "e2")
    back2 = DagHeadsEstimator.load(_uri(tmp_path, "e2"))
    tm = back2.getTopicModel()
    assert isinstance(tm, DagGatedLDAModel) and tm.uid == lda.uid
    assert tm.dag() == lda.dag()
    scored = lda.transform(df)
    a = held.fit(scored)
    b = back2.fit(scored)
    np.testing.assert_allclose(b.V, a.V, rtol=1e-6, atol=1e-8)
    assert b.nodes() == a.nodes()


def test_heads_save_dir_keeps_other_files(spark, tmp_path):
    """saveDir is written in place: other files there survive, a re-fit replaces the
    earlier model."""
    from spark_vi.mllib import DagHeadsEstimator, DagHeadsModel
    df = _heads_frame(spark)
    run = tmp_path / "run"
    run.mkdir()
    (run / "notes.md").write_text("keep me")
    DagHeadsEstimator(maxIter=5, dag=PARENT, saveDir=f"file://{run}").fit(df)
    second = DagHeadsEstimator(maxIter=40, dag=PARENT, saveDir=f"file://{run}").fit(df)
    assert (run / "notes.md").read_text() == "keep me"
    np.testing.assert_array_equal(DagHeadsModel.load(str(run)).V, second.V)


# --------------------------------------------------------------------------- #
# Pipeline                                                                     #
# --------------------------------------------------------------------------- #

def _docs(spark):
    """Word lists + labels for featurizer -> topic model -> heads."""
    rng = np.random.default_rng(2)
    words = {3: ["a", "b", "c"], 4: ["d", "e", "f"], 5: ["g", "h", "i"], 6: ["j", "k"]}
    rows = []
    for i in range(80):
        leaf = [3, 4, 5, 6][i % 4]
        w = list(rng.choice(words[leaf], size=4)) + list(rng.choice(list("xyz"), size=2))
        codes = [int(c) for c in rng.choice([10, 11, 12], size=2)]
        rows.append((i, [str(t) for t in w], codes, [leaf]))
    return spark.createDataFrame(
        rows, "doc_id long, words array<string>, codes array<int>, labels array<int>")


def _pipeline():
    from spark_vi.mllib import DagGatedFeaturizer, DagHeadsEstimator
    from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAEstimator
    feat = (DagGatedFeaturizer().addArrayDomain("words", "words")
            .addArrayDomain("codes", "codes"))
    lda = DagGatedLDAEstimator(dag=PARENT, gateNBg=2, maxIter=2, seed=0,
                               subsamplingRate=1.0,
                               featuresCols=["words_features", "codes_features"],
                               labelsCol="labels")
    heads = DagHeadsEstimator(dag=PARENT, maxIter=30)
    return Pipeline(stages=[feat, lda, heads])


def test_pipeline_and_pipeline_model_round_trip(spark, tmp_path):
    docs = _docs(spark)
    pipe = _pipeline()
    pipe.save(_uri(tmp_path, "pipe"))
    pipe2 = Pipeline.load(_uri(tmp_path, "pipe"))
    assert [type(s).__name__ for s in pipe2.getStages()] == [
        "DagGatedFeaturizer", "DagGatedLDAEstimator", "DagHeadsEstimator"]
    assert pipe2.getStages()[1].getDag() == PARENT_AS_GIVEN

    pm = pipe.fit(docs)
    uri = _uri(tmp_path, "pm")
    pm.save(uri)
    pm2 = PipelineModel.load(uri)
    feat, lda, heads = pm2.stages
    assert lda.dag() == pm.stages[1].dag() and heads.nodes() == pm.stages[2].nodes()
    a = pm.transform(docs).orderBy("doc_id")
    b = pm2.transform(docs).orderBy("doc_id")
    np.testing.assert_array_equal(_theta(b), _theta(a))
    np.testing.assert_array_equal(_theta(b, "nodeProbability"),
                                  _theta(a, "nodeProbability"))
    # overwrite replaces the whole PipelineModel
    with pytest.raises(Exception):
        pm.save(uri)
    pm.write().overwrite().save(uri)
    assert PipelineModel.load(uri).stages[2].uid == pm.stages[2].uid


PARENT_AS_GIVEN = {1: [], 2: [], 3: [1], 4: [1], 5: [2], 6: [2]}
