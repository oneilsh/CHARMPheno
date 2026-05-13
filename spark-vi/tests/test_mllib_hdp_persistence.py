"""Persistence Params + Model.save/load tests for the OnlineHDP MLlib shim.

Covers the three new Estimator Params (saveInterval, saveDir, resumeFrom),
their fit-time validation, and the save/load round-trip on the Model — plus
an HDP-specific check that T/K/V are recovered from result.metadata so the
simplified OnlineHDPModel(result) constructor reconstructs shape correctly.
"""
from __future__ import annotations

import json

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Estimator: Param surface for save/resume
# ---------------------------------------------------------------------------

def test_estimator_default_params_for_save_interval_dir_resume_from():
    from spark_vi.mllib.topic.hdp import OnlineHDPEstimator

    e = OnlineHDPEstimator()
    assert e.getOrDefault("saveInterval") == -1
    assert e.getOrDefault("saveDir") == ""
    assert e.getOrDefault("resumeFrom") == ""


def test_estimator_setters_round_trip(tmp_path):
    from spark_vi.mllib.topic.hdp import OnlineHDPEstimator

    e = OnlineHDPEstimator()
    e.setSaveInterval(7)
    e.setSaveDir(str(tmp_path / "saves"))
    e.setResumeFrom(str(tmp_path / "resume"))

    assert e.getSaveInterval() == 7
    assert e.getSaveDir() == str(tmp_path / "saves")
    assert e.getResumeFrom() == str(tmp_path / "resume")


def test_constructor_accepts_persistence_kwargs(tmp_path):
    """Constructor kwargs must reach saveInterval / saveDir / resumeFrom.

    The cloud driver constructs the Estimator with kwargs (matching the
    style used for `learningOffset` etc.); a missing kwarg in the
    explicit @keyword_only signature would crash with TypeError. This
    test pins the contract.

    If this test fails, you almost certainly added a Param on
    `_PersistenceParams` without enumerating it in
    `OnlineHDPEstimator.__init__`'s signature. See the
    `_PersistenceParams` docstring for the rule.
    """
    from spark_vi.mllib.topic.hdp import OnlineHDPEstimator

    e = OnlineHDPEstimator(
        saveInterval=10,
        saveDir=str(tmp_path / "saves"),
        resumeFrom=str(tmp_path / "resume"),
    )
    assert e.getSaveInterval() == 10
    assert e.getSaveDir() == str(tmp_path / "saves")
    assert e.getResumeFrom() == str(tmp_path / "resume")


# ---------------------------------------------------------------------------
# Estimator: fit-time validation
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def _tiny_hdp_df(spark):
    """Minimal DF for fit-time validation tests."""
    from pyspark.ml.linalg import Vectors
    rows = [(Vectors.dense([1.0, 0.0, 1.0]),) for _ in range(4)]
    return spark.createDataFrame(rows, schema=["features"])


def test_estimator_rejects_save_interval_zero(_tiny_hdp_df):
    from spark_vi.mllib.topic.hdp import OnlineHDPEstimator

    e = OnlineHDPEstimator(k=4, docTruncation=2, maxIter=1, subsamplingRate=1.0)
    e.setSaveInterval(0)
    with pytest.raises(ValueError, match="saveInterval=0"):
        e.fit(_tiny_hdp_df)


def test_estimator_rejects_save_interval_positive_without_dir(_tiny_hdp_df):
    from spark_vi.mllib.topic.hdp import OnlineHDPEstimator

    e = OnlineHDPEstimator(k=4, docTruncation=2, maxIter=1, subsamplingRate=1.0)
    e.setSaveInterval(5)
    with pytest.raises(ValueError, match="saveDir"):
        e.fit(_tiny_hdp_df)


def test_estimator_rejects_resume_from_when_path_has_no_manifest(
    tmp_path, _tiny_hdp_df,
):
    from spark_vi.mllib.topic.hdp import OnlineHDPEstimator

    e = OnlineHDPEstimator(k=4, docTruncation=2, maxIter=1, subsamplingRate=1.0)
    empty = tmp_path / "no_manifest_here"
    empty.mkdir()
    e.setResumeFrom(str(empty))
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        e.fit(_tiny_hdp_df)


# ---------------------------------------------------------------------------
# Model: save / load round-trip
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def _persistence_hdp_corpus_df(spark):
    """Small but real-shaped corpus so a few iters of fit do something."""
    from pyspark.ml.linalg import Vectors

    rng = np.random.default_rng(0)
    rows = []
    favored = {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8]}
    for doc_id in range(15):
        cluster = doc_id % 3
        counts = np.zeros(9, dtype=np.float64)
        for w in rng.choice(favored[cluster], size=15, replace=True):
            counts[w] += 1.0
        rows.append((Vectors.dense(counts.tolist()),))
    return spark.createDataFrame(rows, schema=["features"])


def test_model_save_then_load_round_trips_global_params(
    tmp_path, _persistence_hdp_corpus_df,
):
    from spark_vi.mllib.topic.hdp import OnlineHDPEstimator, OnlineHDPModel

    estimator = OnlineHDPEstimator(
        k=8, docTruncation=4, maxIter=3, seed=0, subsamplingRate=1.0,
    )
    model = estimator.fit(_persistence_hdp_corpus_df)

    save_path = tmp_path / "hdp_save"
    model.save(str(save_path))

    loaded = OnlineHDPModel.load(str(save_path))

    for name, arr in model.result.global_params.items():
        np.testing.assert_array_equal(loaded.result.global_params[name], arr)


def test_model_load_rejects_wrong_model_class(tmp_path):
    """Loading an LDA-stamped manifest with OnlineHDPModel.load must raise."""
    from spark_vi.mllib.topic.hdp import OnlineHDPModel

    save_dir = tmp_path / "lda_marked"
    save_dir.mkdir()
    (save_dir / "params").mkdir()
    np.save(save_dir / "params" / "lambda.npy", np.zeros((2, 3)))
    manifest = {
        "format_version": 1,
        "elbo_trace": [],
        "n_iterations": 0,
        "converged": False,
        "metadata": {"model_class": "OnlineLDA", "K": 2, "V": 3},
        "param_names": ["lambda"],
        "diagnostic_traces": {},
    }
    (save_dir / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="OnlineHDP"):
        OnlineHDPModel.load(str(save_dir))


def test_model_save_load_then_transform_works(
    tmp_path, _persistence_hdp_corpus_df,
):
    from spark_vi.mllib.topic.hdp import OnlineHDPEstimator, OnlineHDPModel

    estimator = OnlineHDPEstimator(
        k=8, docTruncation=4, maxIter=3, seed=0, subsamplingRate=1.0,
    )
    model = estimator.fit(_persistence_hdp_corpus_df)

    save_path = tmp_path / "hdp_save_transform"
    model.save(str(save_path))
    loaded = OnlineHDPModel.load(str(save_path))

    out = loaded.transform(_persistence_hdp_corpus_df)
    assert "topicDistribution" in out.columns
    rows = out.select("topicDistribution").collect()
    for r in rows:
        arr = np.asarray(r["topicDistribution"].toArray())
        assert arr.shape == (8,)
        np.testing.assert_allclose(arr.sum(), 1.0, atol=1e-6)


def test_model_save_load_round_trips_T_K_V_via_metadata(
    tmp_path, _persistence_hdp_corpus_df,
):
    """The simplified OnlineHDPModel(result) constructor must recover T, K, V
    from result.metadata. Verify the loaded model's _T / _K / vocabSize match
    the values the Estimator was configured with.
    """
    from spark_vi.mllib.topic.hdp import OnlineHDPEstimator, OnlineHDPModel

    T_set, K_set = 8, 4
    estimator = OnlineHDPEstimator(
        k=T_set, docTruncation=K_set, maxIter=3, seed=0, subsamplingRate=1.0,
    )
    model = estimator.fit(_persistence_hdp_corpus_df)
    assert model._T == T_set
    assert model._K == K_set
    V = model.vocabSize()

    save_path = tmp_path / "hdp_metadata_roundtrip"
    model.save(str(save_path))
    loaded = OnlineHDPModel.load(str(save_path))

    assert loaded._T == T_set
    assert loaded._K == K_set
    assert loaded.vocabSize() == V


def test_estimator_fit_with_savedir_only_writes_final(
    tmp_path, _persistence_hdp_corpus_df,
):
    """saveDir set, saveInterval=-1 → exactly one save (the end-of-fit one)."""
    from spark_vi.mllib.topic.hdp import OnlineHDPEstimator, OnlineHDPModel

    save_dir = tmp_path / "auto_save_final_only_hdp"
    estimator = OnlineHDPEstimator(
        k=8, docTruncation=4, maxIter=3, seed=0, subsamplingRate=1.0,
    )
    estimator.setSaveDir(str(save_dir))
    model = estimator.fit(_persistence_hdp_corpus_df)

    assert (save_dir / "manifest.json").exists()
    loaded = OnlineHDPModel.load(str(save_dir))
    for name, arr in model.result.global_params.items():
        np.testing.assert_array_equal(loaded.result.global_params[name], arr)
