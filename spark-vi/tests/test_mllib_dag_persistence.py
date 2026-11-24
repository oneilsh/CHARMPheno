"""MLlib persistence of the DAG-gated stages (spec 2026-10-30 R13 Level 2).

DagGatedLDAEstimator / DagGatedLDAModel are
``MLWritable`` / ``MLReadable``: MLlib's Params metadata plus our data files (the
VIResult, the node set), so each saves and loads on its own and inside
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
    PARENT,
    _df,
    _est,
    _str_dag_df,
    _str_est,
)

# A DAG with a multi-parent node.
TWIN = {"a": None, "b": "a", "c": "b", "x": ["a", "c"]}


def _theta(df, col="topicDistribution"):
    return np.array([r[0].toArray() for r in df.select(col).collect()])


def _uri(tmp_path, name):
    return f"file://{tmp_path}/{name}"


# --------------------------------------------------------------------------- #
# DagGatedLDAEstimator: Params only                                            #
# --------------------------------------------------------------------------- #

def test_estimator_round_trip_keeps_structured_params(spark, tmp_path):
    """The structured Params survive save / load (node and term ids are strings,
    ADR 0053)."""
    from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAEstimator
    est = DagGatedLDAEstimator.recommended(
        dag=TWIN, minDocsPerNode=7, seed=3, featuresCols=["f0", "f1"],
        vocabularies=[["t0", "t1"], ["1", "2", "t0"]],
        nodeTerms={"a": ["t1"], "b": [("f1", "2")], "c": [("f1", "1"), ("f0", "t0")]},
        gateSeedProfile={"b": {("f1", "t0"): 2.0}, "a": {"t1": 0.5}})
    est.write().save(_uri(tmp_path, "est"))
    back = DagGatedLDAEstimator.load(_uri(tmp_path, "est"))
    assert back.uid == est.uid
    assert back.getDag() == est.getDag() == {"a": [], "b": ["a"], "c": ["b"],
                                             "x": ["a", "c"]}
    assert back.getVocabularies() == [["t0", "t1"], ["1", "2", "t0"]]
    assert back.getNodeTerms() == est.getNodeTerms()
    assert set(back.getNodeTerms()) == {"a", "b", "c"}
    assert back.getNodeTerms()["b"] == [("f1", "2")]
    assert back.getGateSeedProfile() == {"b": {("f1", "t0"): 2.0}, "a": {"t1": 0.5}}
    for p in est.params:
        assert back.isSet(p.name) == est.isSet(p), p.name
        if est.isDefined(p):
            assert back.getOrDefault(p.name) == est.getOrDefault(p), p.name
    assert back.getDocConcentration() == [0.5] and not back.isSet("topicConcentration")


def test_estimator_save_refuses_an_existing_path_unless_overwrite(spark, tmp_path):
    """Code review C1: the driver-side Params writer follows MLlib's rule."""
    from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAEstimator
    uri = _uri(tmp_path, "est")
    DagGatedLDAEstimator(dag=TWIN, minDocsPerNode=7).write().save(uri)
    est = DagGatedLDAEstimator(dag=TWIN, minDocsPerNode=9)
    with pytest.raises(FileExistsError, match="overwrite"):
        est.write().save(uri)
    assert DagGatedLDAEstimator.load(uri).getMinDocsPerNode() == 7
    est.write().overwrite().save(uri)
    assert DagGatedLDAEstimator.load(uri).getMinDocsPerNode() == 9


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


def test_local_save_and_load_run_no_spark_job(spark, tmp_path):
    """A `file://` path exists on the driver's machine only (a run dir on the master's
    workspace mount); MLlib's DefaultParamsWriter writes its metadata through a
    one-task Spark job, i.e. on an executor, which fails on a cluster. Every stage
    here must save and load such a path without a Spark job, and leave MLlib's own
    metadata layout (metadata/part-00000 + _SUCCESS)."""
    from spark_vi.mllib.topic.dag_gated_lda import (DagGatedLDAEstimator,
                                                    DagGatedLDAModel)
    from spark_vi.mllib.topic.featurizer import (DagGatedFeaturizer,
                                                 DagGatedFeaturizerModel)
    sc = spark.sparkContext
    df = _df(spark)
    est = _est(maxIter=2)
    model = est.fit(df)
    fz = (DagGatedFeaturizer(docIdCol="id")
          .addArrayDomain("a", "terms").setLabelsInputCol("labs"))
    fdf = spark.createDataFrame([(1, ["x", "y"], ["1"]), (2, ["y"], [])],
                                "id int, terms array<string>, labs array<string>")
    fmodel = fz.fit(fdf)
    for name, stage, cls in [("est", est, DagGatedLDAEstimator),
                             ("model", model, DagGatedLDAModel),
                             ("fz", fz, DagGatedFeaturizer),
                             ("fzm", fmodel, DagGatedFeaturizerModel)]:
        uri = _uri(tmp_path, name)
        assert _jobs_in(sc, f"save-{name}", lambda: stage.write().overwrite().save(uri)) == 0
        md = os.listdir(tmp_path / name / "metadata")
        assert "part-00000" in md and "_SUCCESS" in md, (name, md)
        back = {}
        assert _jobs_in(sc, f"load-{name}", lambda: back.setdefault("s", cls.load(uri))) == 0
        assert back["s"].uid == stage.uid
    assert fmodel.vocabularies() == DagGatedFeaturizerModel.load(_uri(tmp_path, "fzm")).vocabularies()


def test_metadata_written_by_mllib_still_loads(spark, tmp_path):
    """The driver-side reader reads MLlib's own writer's files too."""
    from pyspark.ml.util import DefaultParamsWriter
    from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAEstimator
    est = _est(maxIter=3)
    DefaultParamsWriter.saveMetadata(est, str(tmp_path / "e"), spark.sparkContext)
    back = DagGatedLDAEstimator.load(_uri(tmp_path, "e"))
    assert back.getOrDefault("maxIter") == 3 and back.getDag() == est.getDag()
