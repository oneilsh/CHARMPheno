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

# Fixtures copied from spark-vi's tests/test_mllib_dag_gated_lda.py (ADR 0054: the
# heads' tests moved here; charmpheno tests do not import spark-vi's test modules).
PARENT = {"1": None, "2": None, "3": "1", "4": "1", "5": "2", "6": "2"}


MIXED = {"A": None, "B": None, "1": "A", "c": "1", "2": ["A", "B"], "3": "2",
         "rare": "3"}


V0_TERMS = [f"t{i}" for i in range(8)]


V1_TERMS = ["100", "101", "102", "103", "t0"]   # "t0" is in both vocabularies


def _df(spark, n=40, V=30, seed=0, col="labels"):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        leaf = str(int(rng.choice([3, 4, 5, 6])))
        idx = sorted(rng.choice(V, size=6, replace=False).tolist())
        rows.append((Vectors.sparse(V, idx, [1.0] * len(idx)), [leaf]))
    return spark.createDataFrame(rows, ["features", col])


def _est(**kw):
    from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAEstimator
    kw.setdefault("maxIter", 3)
    kw.setdefault("seed", 0)
    kw.setdefault("subsamplingRate", 1.0)
    kw.setdefault("optimizeDocConcentration", False)   # fixed alpha unless asked
    kw.setdefault("dag", PARENT)
    return DagGatedLDAEstimator(gateNBg=2, gateTpn=1, **kw)


def _mixed_rows(seed=0, n=48, specific_only=False):
    """Rows for the MIXED DAG over two domains. Labels carry ancestors and
    duplicates unless `specific_only` (then only the most-specific labels: the same
    gate)."""
    rng = np.random.default_rng(seed)
    full = {  # label list as written -> its most-specific labels
        "c": (["c", "1", "A", "c"], ["c"]),
        "3": (["3", "2", "B", "3"], ["3"]),
        "2": (["2"], ["2"]),
        "B": (["B", "B"], ["B"]),
        "A+3": (["A", "3", "c"], ["3", "c"]),
    }
    keys = list(full)
    rows = []
    for i in range(n):
        if i % 8 == 7:
            lab = [] if i % 16 == 7 else None                 # background docs
        else:
            k = keys[i % len(keys)]
            lab = full[k][1] if specific_only else full[k][0]
        i0 = sorted(rng.choice(8, size=3, replace=False).tolist())
        i1 = sorted(rng.choice(5, size=2, replace=False).tolist())
        rows.append((Vectors.sparse(8, i0, [1.0] * 3), Vectors.sparse(5, i1, [2.0] * 2),
                     lab))
    rare = ["3", "rare"] if not specific_only else ["rare"]
    rows.append((Vectors.sparse(8, [0], [1.0]), Vectors.sparse(5, [1], [1.0]), rare))
    return rows


def _str_dag_df(spark, specific_only=False, seed=0):
    """The MIXED DAG's rows as a DataFrame, labels as array<string>."""
    from pyspark.sql.types import ArrayType, StringType, StructField, StructType
    from pyspark.ml.linalg import VectorUDT
    schema = StructType([StructField("f0", VectorUDT()), StructField("f1", VectorUDT()),
                         StructField("labels", ArrayType(StringType()))])
    rows = [(a, b, None if lab is None else [str(x) for x in lab])
            for a, b, lab in _mixed_rows(seed=seed, specific_only=specific_only)]
    return spark.createDataFrame(rows, schema)


STR_DAG = MIXED


def _str_est(**kw):
    from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAEstimator
    kw.setdefault("maxIter", 3)
    kw.setdefault("seed", 0)
    kw.setdefault("subsamplingRate", 1.0)
    kw.setdefault("optimizeDocConcentration", False)   # fixed alpha unless asked
    kw.setdefault("minDocsPerNode", 2)
    kw.setdefault("dag", STR_DAG)
    # String term ids throughout, as a vectorizer's vocabularies are.
    kw.setdefault("vocabularies", [V0_TERMS, V1_TERMS])
    return DagGatedLDAEstimator(gateNBg=2, gateTpn=1, featuresCols=["f0", "f1"], **kw)



# A DAG with a multi-parent node.
TWIN = {"a": None, "b": "a", "c": "b", "x": ["a", "c"]}


def _theta(df, col="topicDistribution"):
    return np.array([r[0].toArray() for r in df.select(col).collect()])


def _uri(tmp_path, name):
    return f"file://{tmp_path}/{name}"


# --------------------------------------------------------------------------- #
# DagGatedLDAEstimator: Params only                                            #
# --------------------------------------------------------------------------- #

def test_model_load_refuses_another_class_and_another_engine(spark, tmp_path):
    from charmpheno.readout.mllib import DagHeadsModel
    from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAModel
    df = _df(spark)
    model = _est(maxIter=1).fit(df)
    path = str(tmp_path / "m")
    model.save(path)
    with pytest.raises(ValueError, match="not a charmpheno.readout.mllib.DagHeadsModel"):
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
    from charmpheno.readout.mllib import DagHeadsEstimator, DagHeadsModel
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

    from charmpheno.readout.mllib import DagHeadsEstimator, DagHeadsModel
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
    from charmpheno.readout.mllib import DagHeadsEstimator
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
    from charmpheno.readout.mllib import DagHeadsEstimator, DagHeadsModel
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
        rows.append((i, [str(t) for t in w], codes, [str(leaf)]))
    return spark.createDataFrame(
        rows, "doc_id long, words array<string>, codes array<int>, labels array<string>")


def _pipeline():
    from spark_vi.mllib import DagGatedFeaturizer
    from charmpheno.readout.mllib import DagHeadsEstimator
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


PARENT_AS_GIVEN = {"1": [], "2": [], "3": ["1"], "4": ["1"], "5": ["2"], "6": ["2"]}


# --------------------------------------------------------------------------- #
# Local paths are written and read on the driver: no Spark job                 #
# --------------------------------------------------------------------------- #

def _jobs_in(sc, group, fn):
    """Run fn() under a job group; return how many Spark jobs it started."""
    sc.setJobGroup(group, group)
    try:
        fn()
    finally:
        sc.setLocalProperty("spark.jobGroup.id", None)
    return len(sc.statusTracker().getJobIdsForGroup(group))


