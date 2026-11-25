"""Save / load round trips of the fitted OnlineSTMModel (MLlib stage persistence,
ADR 0055 / 0056): ``<path>/metadata/`` + ``<path>/result/`` (+ the formula's
``covariate_spec.json``)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyspark")


def _result(gp, metadata=None, **kw):
    from spark_vi.core.result import VIResult
    md = {"model_class": "OnlineSTM"}
    md.update(metadata or {})
    kw.setdefault("elbo_trace", [])
    kw.setdefault("n_iterations", 0)
    kw.setdefault("converged", False)
    return VIResult(global_params=gp, metadata=md, **kw)


def _gp(K=3, V=10, P=2):
    from spark_vi.models.topic.stm import OnlineSTM
    return OnlineSTM(K=K, vocab_size=V, P=P, random_seed=0).initialize_global(None)


class TestOnlineSTMModelPersistence:
    def test_save_and_load_roundtrips_result_and_params(self, spark, tmp_path: Path):
        from spark_vi.mllib.topic.stm import OnlineSTMModel
        gp = _gp()
        model = OnlineSTMModel(_result(gp, {"K": 3, "V": 10, "P": 2}))
        model._set(covariateNames=["intercept", "cohort_b"], k=3, lbfgsTol=1e-5)
        out_dir = tmp_path / "stm_model"
        model.save(str(out_dir))
        assert (out_dir / "metadata" / "part-00000").exists()
        assert (out_dir / "result" / "manifest.json").exists()
        assert not list(out_dir.rglob("*.pkl"))          # no pickle anywhere

        loaded = OnlineSTMModel.load(str(out_dir))
        for key in ("Gamma", "Sigma", "lambda", "eta"):
            np.testing.assert_array_equal(loaded.result.global_params[key], gp[key])
        assert loaded.result.metadata["K"] == 3
        assert loaded.getOrDefault("covariateNames") == ["intercept", "cohort_b"]
        assert loaded.getOrDefault("k") == 3
        assert loaded.getOrDefault("lbfgsTol") == 1e-5
        assert loaded.uid == model.uid
        assert loaded.covariateSpec is None

    def test_save_load_roundtrips_gamma_sigma_diagnostic_traces(self, spark, tmp_path):
        # The per-iter 2-D Gamma + 1-D Sigma + topic-block-label snapshots round-trip.
        from spark_vi.mllib.topic.stm import OnlineSTMModel
        traces = {
            "Gamma": [np.full((2, 3), 0.1), np.full((2, 3), 0.2)],   # 2-D
            "Sigma": [np.ones(3), np.full(3, 0.9)],                   # 1-D
            "topic_block_labels": [["background", "background", "rare"]] * 2,
        }
        model = OnlineSTMModel(_result(_gp(), {"K": 3}, diagnostic_traces=traces))
        out_dir = tmp_path / "stm_traces"
        model.save(str(out_dir))
        lg = OnlineSTMModel.load(str(out_dir)).result.diagnostic_traces
        assert set(lg) == {"Gamma", "Sigma", "topic_block_labels"}
        for got, want in zip(lg["Gamma"], traces["Gamma"]):
            np.testing.assert_array_equal(got, want)        # 2-D preserved
        for got, want in zip(lg["Sigma"], traces["Sigma"]):
            np.testing.assert_array_equal(got, want)
        assert lg["topic_block_labels"] == traces["topic_block_labels"]

    def test_save_load_roundtrips_resume_state(self, spark, tmp_path):
        """n_iterations / elbo_trace / converged survive save->load, so a resumed
        fit continues the Robbins-Monro step counter instead of restarting at t=0."""
        from spark_vi.mllib.topic.stm import OnlineSTMModel
        model = OnlineSTMModel(_result(
            _gp(), {"K": 3}, n_iterations=100, elbo_trace=[-100.0, -90.0, -85.0],
            converged=True))
        out_dir = tmp_path / "stm_resume"
        model.save(str(out_dir))
        loaded = OnlineSTMModel.load(str(out_dir)).result
        assert loaded.n_iterations == 100
        assert loaded.elbo_trace == [-100.0, -90.0, -85.0]
        assert loaded.converged is True

    def test_stm_hardening_metadata_roundtrips(self, spark, tmp_path):
        from spark_vi.mllib.topic.stm import OnlineSTMModel
        model = OnlineSTMModel(_result(_gp(), {"stm_hardening": {
            "reference_topic": True, "min_pair_support": 10}}))
        out_dir = tmp_path / "stm_model_hardening"
        model.save(str(out_dir))
        loaded = OnlineSTMModel.load(str(out_dir))
        assert loaded.result.metadata["stm_hardening"] == {
            "reference_topic": True, "min_pair_support": 10}

    def test_roundtrips_topic_blocks(self, spark, tmp_path):
        from spark_vi.mllib.topic.stm import OnlineSTMModel
        from spark_vi.models.topic.partition import TopicBlockPartition
        part = TopicBlockPartition("source_cohort", background_k=2,
                                   foreground=(("cancer", 1),))
        gp = {"lambda": np.ones((3, 4)), "eta": np.array(0.3),
              "Gamma": np.zeros((2, 3)), "Sigma": np.eye(3)}
        model = OnlineSTMModel(_result(gp, {"topic_block_spec": part.to_dict()}))
        assert model.topicBlockPartition() == part       # from the result's provenance
        model.save(str(tmp_path / "m"))
        assert OnlineSTMModel.load(str(tmp_path / "m")).topicBlockPartition() == part

    def test_n_pairs_round_trips_through_save_load(self, spark, tmp_path):
        from spark_vi.mllib.topic.stm import OnlineSTMModel
        gp = {"lambda": np.ones((3, 5)), "eta": np.array(0.01),
              "Gamma": np.zeros((1, 3)), "Sigma": np.eye(3),
              "n_pairs": np.array([[9, 4, 0], [4, 9, 2], [0, 2, 9]], dtype=float)}
        OnlineSTMModel(_result(gp, n_iterations=1)).save(str(tmp_path / "m"))
        loaded = OnlineSTMModel.load(str(tmp_path / "m"))
        assert np.array_equal(loaded.result.global_params["n_pairs"], gp["n_pairs"])

    def test_save_refuses_existing_path_unless_overwrite(self, spark, tmp_path):
        from spark_vi.mllib.topic.stm import OnlineSTMModel
        model = OnlineSTMModel(_result(_gp()))
        out = str(tmp_path / "m")
        model.save(out)
        with pytest.raises(Exception):
            model.save(out)
        model.write().overwrite().save(out)
        assert OnlineSTMModel.load(out).result.global_params["lambda"].shape == (3, 10)

    def test_bare_result_directory_does_not_load(self, spark, tmp_path):
        """A saveDir checkpoint / save_result output is not a saved stage (no
        backward compatibility, ADR 0055); wrap it with load_result instead."""
        from spark_vi.io.export import load_result, save_result
        from spark_vi.mllib.topic.stm import OnlineSTMModel
        save_result(_result(_gp()), tmp_path / "bare")
        with pytest.raises(FileNotFoundError):
            OnlineSTMModel.load(str(tmp_path / "bare"))
        wrapped = OnlineSTMModel(load_result(tmp_path / "bare"))
        assert wrapped.result.global_params["Gamma"].shape == (2, 3)

    def test_refuses_another_model_class(self, spark, tmp_path):
        from spark_vi.mllib.topic.stm import OnlineSTMModel
        model = OnlineSTMModel(_result(_gp(), {"model_class": "OnlineLDA"}))
        model.save(str(tmp_path / "m"))
        with pytest.raises(ValueError, match="OnlineLDA"):
            OnlineSTMModel.load(str(tmp_path / "m"))

    def test_pipeline_model_round_trip(self, spark, tmp_path):
        from pyspark.ml import PipelineModel
        from spark_vi.mllib.topic.stm import OnlineSTMModel
        model = OnlineSTMModel(_result(_gp()))
        model._set(covariateNames=["a", "b"])
        PipelineModel(stages=[model]).save(str(tmp_path / "pipe"))
        stage = PipelineModel.load(str(tmp_path / "pipe")).stages[0]
        assert isinstance(stage, OnlineSTMModel)
        assert stage.getOrDefault("covariateNames") == ["a", "b"]
        np.testing.assert_array_equal(stage.result.global_params["Sigma"],
                                      model.result.global_params["Sigma"])
