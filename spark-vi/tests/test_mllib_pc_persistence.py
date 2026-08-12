"""Persistence Params + Model.save/load + resume tests for the PC MLlib shim.

Mirrors test_mllib_lda_persistence.py (and the LDA resume round-trip in
test_lda_integration.py): the three persistence Params (saveInterval, saveDir,
resumeFrom), their fit-time validation, the Model save/load round-trip, the
cross-class load rejection (a PC checkpoint must not load as LDA and vice-versa),
and the resume-continues-the-iteration-counter (N + M iters) invariant.

PC's fit goes through the SAME VIRunner.fit as LDA (both weightY == 0 and > 0),
so the persistence semantics are identical; these tests exercise the weightY == 0
(unsupervised) path since it needs no label columns and is the cheaper one to run.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from pyspark.ml.linalg import Vectors


# ---------------------------------------------------------------------------
# Estimator: Param surface for save/resume
# ---------------------------------------------------------------------------

def test_estimator_default_params_for_save_interval_dir_resume_from():
    from spark_vi.mllib.topic.pc import OnlinePCLDAEstimator

    e = OnlinePCLDAEstimator()
    assert e.getOrDefault("saveInterval") == -1
    assert e.getOrDefault("saveDir") == ""
    assert e.getOrDefault("resumeFrom") == ""


def test_estimator_setters_round_trip(tmp_path):
    from spark_vi.mllib.topic.pc import OnlinePCLDAEstimator

    e = OnlinePCLDAEstimator()
    e.setSaveInterval(5)
    e.setSaveDir(str(tmp_path / "saves"))
    e.setResumeFrom(str(tmp_path / "resume"))

    assert e.getSaveInterval() == 5
    assert e.getSaveDir() == str(tmp_path / "saves")
    assert e.getResumeFrom() == str(tmp_path / "resume")


def test_constructor_accepts_persistence_kwargs(tmp_path):
    """Constructor kwargs must reach saveInterval / saveDir / resumeFrom.

    The cloud driver constructs the Estimator with kwargs; a missing kwarg in
    the explicit @keyword_only signature would crash with TypeError. This test
    pins the contract (see the _PersistenceParams docstring for the rule).
    """
    from spark_vi.mllib.topic.pc import OnlinePCLDAEstimator

    e = OnlinePCLDAEstimator(
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
def _tiny_df(spark):
    """Minimal DF for fit-time validation tests (won't actually fit far)."""
    rows = [(Vectors.dense([1.0, 0.0, 1.0]),) for _ in range(4)]
    return spark.createDataFrame(rows, schema=["features"])


def test_estimator_rejects_save_interval_zero(_tiny_df):
    from spark_vi.mllib.topic.pc import OnlinePCLDAEstimator

    e = OnlinePCLDAEstimator(k=2, maxIter=1, subsamplingRate=1.0)
    e.setSaveInterval(0)
    with pytest.raises(ValueError, match="saveInterval=0"):
        e.fit(_tiny_df)


def test_estimator_rejects_save_interval_positive_without_dir(_tiny_df):
    from spark_vi.mllib.topic.pc import OnlinePCLDAEstimator

    e = OnlinePCLDAEstimator(k=2, maxIter=1, subsamplingRate=1.0)
    e.setSaveInterval(5)  # saveDir stays ""
    with pytest.raises(ValueError, match="saveDir"):
        e.fit(_tiny_df)


def test_estimator_rejects_resume_from_when_path_has_no_manifest(tmp_path, _tiny_df):
    from spark_vi.mllib.topic.pc import OnlinePCLDAEstimator

    e = OnlinePCLDAEstimator(k=2, maxIter=1, subsamplingRate=1.0)
    empty = tmp_path / "no_manifest_here"
    empty.mkdir()
    e.setResumeFrom(str(empty))
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        e.fit(_tiny_df)


# ---------------------------------------------------------------------------
# Model: save / load round-trip
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def _persistence_corpus_df(spark):
    """Small but real-shaped corpus so a few iters of fit do something."""
    rng = np.random.default_rng(0)
    rows = []
    favored = {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8]}
    for doc_id in range(15):
        topic = doc_id % 3
        counts = np.zeros(9, dtype=np.float64)
        for w in rng.choice(favored[topic], size=15, replace=True):
            counts[w] += 1.0
        rows.append((Vectors.dense(counts.tolist()),))
    return spark.createDataFrame(rows, schema=["features"])


def test_model_save_then_load_round_trips_global_params(tmp_path, _persistence_corpus_df):
    from spark_vi.mllib.topic.pc import OnlinePCLDAEstimator, OnlinePCLDAModel

    estimator = OnlinePCLDAEstimator(k=3, maxIter=3, seed=0, subsamplingRate=1.0)
    model = estimator.fit(_persistence_corpus_df)

    save_path = tmp_path / "pc_save"
    model.save(str(save_path))

    loaded = OnlinePCLDAModel.load(str(save_path))

    # Every entry in global_params must round-trip exactly.
    for name, arr in model.result.global_params.items():
        np.testing.assert_array_equal(loaded.result.global_params[name], arr)


def test_model_load_rejects_wrong_model_class(tmp_path):
    """Loading an LDA-stamped manifest with OnlinePCLDAModel.load must raise."""
    from spark_vi.mllib.topic.pc import OnlinePCLDAModel

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

    with pytest.raises(ValueError, match="OnlinePCLDA"):
        OnlinePCLDAModel.load(str(save_dir))


def test_pc_checkpoint_does_not_load_as_lda(tmp_path, _persistence_corpus_df):
    """A real PC checkpoint must be REJECTED by OnlineLDAModel.load (and vice-versa,
    covered by test_model_load_rejects_wrong_model_class): the model-class tag
    keeps the two checkpoint families from cross-loading."""
    from spark_vi.mllib.topic.lda import OnlineLDAModel
    from spark_vi.mllib.topic.pc import OnlinePCLDAEstimator

    model = OnlinePCLDAEstimator(k=3, maxIter=2, seed=0, subsamplingRate=1.0).fit(
        _persistence_corpus_df
    )
    save_path = tmp_path / "pc_ckpt"
    model.save(str(save_path))

    # The PC checkpoint is stamped model_class="OnlinePCLDA"; loading it as LDA
    # must raise (the LDA loader expects "OnlineLDA").
    with pytest.raises(ValueError, match="OnlineLDA"):
        OnlineLDAModel.load(str(save_path))


def test_model_save_load_then_transform_works(tmp_path, _persistence_corpus_df):
    from spark_vi.mllib.topic.pc import OnlinePCLDAEstimator, OnlinePCLDAModel

    estimator = OnlinePCLDAEstimator(k=3, maxIter=3, seed=0, subsamplingRate=1.0)
    model = estimator.fit(_persistence_corpus_df)

    save_path = tmp_path / "pc_save_transform"
    model.save(str(save_path))
    loaded = OnlinePCLDAModel.load(str(save_path))

    out = loaded.transform(_persistence_corpus_df)
    assert "topicDistribution" in out.columns
    rows = out.select("topicDistribution").collect()
    for r in rows:
        arr = np.asarray(r["topicDistribution"].toArray())
        assert arr.shape == (3,)
        np.testing.assert_allclose(arr.sum(), 1.0, atol=1e-6)


def test_estimator_fit_with_savedir_only_writes_final(tmp_path, _persistence_corpus_df):
    """saveDir set, saveInterval=-1 → exactly one save (the end-of-fit one)."""
    from spark_vi.mllib.topic.pc import OnlinePCLDAEstimator, OnlinePCLDAModel

    save_dir = tmp_path / "auto_save_final_only_pc"
    estimator = OnlinePCLDAEstimator(k=3, maxIter=3, seed=0, subsamplingRate=1.0)
    estimator.setSaveDir(str(save_dir))
    model = estimator.fit(_persistence_corpus_df)

    assert (save_dir / "manifest.json").exists()
    loaded = OnlinePCLDAModel.load(str(save_dir))
    for name, arr in model.result.global_params.items():
        np.testing.assert_array_equal(loaded.result.global_params[name], arr)


# ---------------------------------------------------------------------------
# Resume: iteration counter + ELBO trace continue (N + M), fit advances
# ---------------------------------------------------------------------------

def test_pc_resume_from_continues_iteration_count_and_elbo_trace(
    tmp_path, _persistence_corpus_df,
):
    """Resume a saved PC fit: the iteration counter and ELBO trace continue from
    where the prior run left off (N + M), not restart at zero, and training
    actually advanced (global params moved from the checkpoint).

    Step 1: fit N=3 iters with saveDir → run1 directory (n_iterations == 3).
    Step 2: fit a fresh Estimator for M=3 more iters with saveDir + resumeFrom
            both pointing at run1 → n_iterations == 6, elbo_trace length 6 with
            the first 3 entries identical to run1 (history preserved).
    """
    from spark_vi.mllib.topic.pc import OnlinePCLDAEstimator

    N, M = 3, 3
    save_dir = tmp_path / "pc_run1"

    estimator_a = OnlinePCLDAEstimator(k=3, maxIter=N, seed=2, subsamplingRate=1.0)
    estimator_a.setSaveDir(str(save_dir))
    model_a = estimator_a.fit(_persistence_corpus_df)
    result_a = model_a.result

    assert result_a.n_iterations == N
    assert len(result_a.elbo_trace) == N
    lambda_a = np.array(result_a.global_params["lambda"], copy=True)

    estimator_b = OnlinePCLDAEstimator(k=3, maxIter=M, seed=2, subsamplingRate=1.0)
    estimator_b.setSaveDir(str(save_dir))
    estimator_b.setResumeFrom(str(save_dir))
    model_b = estimator_b.fit(_persistence_corpus_df)
    result_b = model_b.result

    assert result_b.n_iterations == N + M, (
        f"resume should continue iteration counter; got {result_b.n_iterations}"
    )
    assert len(result_b.elbo_trace) == N + M, (
        f"resumed elbo_trace should be length {N + M}; got {len(result_b.elbo_trace)}"
    )
    # First N entries match the pre-resume trace exactly (resume preserves history).
    for i, (a, b) in enumerate(zip(result_a.elbo_trace, result_b.elbo_trace[:N])):
        assert a == b, f"resume mutated history at index {i}: pre={a}, post={b}"
    # The fit CONTINUED (did not restart): lambda advanced past the checkpoint.
    assert not np.allclose(result_b.global_params["lambda"], lambda_a), (
        "resumed fit should keep training (lambda should move past the checkpoint)"
    )
