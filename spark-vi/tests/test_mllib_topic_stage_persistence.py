"""MLlib persistence of OnlineLDAModel / OnlineHDPModel (ADR 0055).

Both moved from the bare-VIResult `save` / `load` to `_ResultStagePersistence`:
`<path>/metadata/` (Params) + `<path>/result/` (the VIResult), Pipeline-capable,
MLlib's refuse-unless-overwrite rule. A bare VIResult directory (the old layout, and
every saveDir checkpoint) still loads, with Params at their defaults."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def corpus(spark):
    from pyspark.ml.linalg import Vectors
    rng = np.random.default_rng(0)
    favored = {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8]}
    rows = []
    for d in range(15):
        counts = np.zeros(9)
        for w in rng.choice(favored[d % 3], size=15, replace=True):
            counts[w] += 1.0
        rows.append((Vectors.dense(counts.tolist()),))
    return spark.createDataFrame(rows, schema=["features"])


def _lda():
    from spark_vi.mllib.topic.lda import OnlineLDAEstimator, OnlineLDAModel
    return (OnlineLDAEstimator(k=3, maxIter=2, seed=0, subsamplingRate=1.0,
                               topicDistributionCol="theta"), OnlineLDAModel)


def _hdp():
    from spark_vi.mllib.topic.hdp import OnlineHDPEstimator, OnlineHDPModel
    return (OnlineHDPEstimator(k=4, docTruncation=2, maxIter=2, seed=0,
                               subsamplingRate=1.0, topicDistributionCol="theta"),
            OnlineHDPModel)


KINDS = [pytest.param(_lda, id="lda"), pytest.param(_hdp, id="hdp")]


@pytest.fixture(scope="module", params=KINDS)
def fitted(request, corpus):
    est, model_cls = request.param()
    return est.fit(corpus), model_cls


def test_save_writes_metadata_and_result_and_round_trips_params(fitted, tmp_path):
    model, cls = fitted
    path = tmp_path / "m"
    model.save(str(path))
    assert (path / "metadata" / "part-00000").exists()
    assert (path / "result" / "manifest.json").exists()
    loaded = cls.load(str(path))
    assert loaded.uid == model.uid
    assert loaded.getOrDefault("topicDistributionCol") == "theta"
    assert loaded.getOrDefault("k") == model.getOrDefault("k")
    for name, arr in model.result.global_params.items():
        np.testing.assert_array_equal(loaded.result.global_params[name], arr)


def test_save_refuses_an_existing_path_unless_overwrite(fitted, tmp_path):
    model, cls = fitted
    path = tmp_path / "m"
    model.save(str(path))
    with pytest.raises(FileExistsError):
        model.save(str(path))
    model.write().overwrite().save(str(path))
    assert cls.load(str(path)).uid == model.uid


def test_old_bare_result_layout_still_loads_with_default_params(fitted, tmp_path):
    from spark_vi.io.export import save_result
    model, cls = fitted
    path = tmp_path / "old"
    save_result(model.result, str(path))          # the pre-ADR-0055 Model.save layout
    loaded = cls.load(str(path))
    assert loaded.getOrDefault("topicDistributionCol") == "topicDistribution"
    for name, arr in model.result.global_params.items():
        np.testing.assert_array_equal(loaded.result.global_params[name], arr)


def test_pipeline_model_round_trip(fitted, corpus, tmp_path):
    from pyspark.ml import PipelineModel
    model, _ = fitted
    pm = PipelineModel(stages=[model])
    path = tmp_path / "pipe"
    pm.save(str(path))
    loaded = PipelineModel.load(str(path))
    a = [r["theta"].toArray() for r in pm.transform(corpus).select("theta").collect()]
    b = [r["theta"].toArray() for r in loaded.transform(corpus).select("theta").collect()]
    # transform folds each document in by CAVI to caviTol (1e-3), from a random start
    np.testing.assert_allclose(np.array(a), np.array(b), atol=1e-3)


def test_load_refuses_another_model_class(tmp_path, corpus):
    lda_est, LDA = _lda()
    _, HDP = _hdp()
    path = tmp_path / "m"
    lda_est.fit(corpus).save(str(path))
    with pytest.raises(ValueError):
        HDP.load(str(path))
