"""Save/load round-trip tests for StreamingSTM's fitted STMModel."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# A toy "ModelSpec" stub — anything pickle-able works for the persistence
# layer test. Must be at module level so pickle can locate it by name.
class _FakeSpec:
    def __init__(self):
        self.factor_levels = {"cohort": ["a", "b"]}


class TestSTMModelPersistence:
    def test_save_and_load_roundtrips_VIResult_and_ModelSpec(self, tmp_path: Path):
        from spark_vi.mllib.topic.stm import STMModel
        from spark_vi.models.topic.stm import OnlineSTM

        model = OnlineSTM(K=3, vocab_size=10, P=2, random_seed=0)
        gp = model.initialize_global(None)

        spec = _FakeSpec()  # module-level class — pickle-able
        stm_model = STMModel(
            global_params=gp,
            metadata={"K": 3, "V": 10, "P": 2},
            model_spec=spec,
            covariate_names=["intercept", "cohort_b"],
        )

        out_dir = tmp_path / "stm_model"
        stm_model.save(out_dir)
        loaded = STMModel.load(out_dir)

        # Global params round-trip.
        np.testing.assert_array_equal(loaded.global_params["Gamma"], gp["Gamma"])
        np.testing.assert_array_equal(loaded.global_params["Sigma"], gp["Sigma"])
        np.testing.assert_array_equal(loaded.global_params["lambda"], gp["lambda"])
        np.testing.assert_array_equal(loaded.global_params["eta"], gp["eta"])
        assert loaded.metadata == {"K": 3, "V": 10, "P": 2}
        # ModelSpec round-trips.
        assert loaded.model_spec.factor_levels == {"cohort": ["a", "b"]}
        assert loaded.covariate_names == ["intercept", "cohort_b"]

    def test_save_load_roundtrips_gamma_sigma_diagnostic_traces(self, tmp_path: Path):
        # STMModel.save no longer drops diagnostic_traces: the per-iter 2-D
        # Gamma + 1-D Sigma + topic-block-label snapshots round-trip.
        from spark_vi.mllib.topic.stm import STMModel
        from spark_vi.models.topic.stm import OnlineSTM

        model = OnlineSTM(K=3, vocab_size=10, P=2, random_seed=0)
        gp = model.initialize_global(None)
        traces = {
            "Gamma": [np.full((2, 3), 0.1), np.full((2, 3), 0.2)],   # 2-D
            "Sigma": [np.ones(3), np.full(3, 0.9)],                   # 1-D
            "topic_block_labels": [["background", "background", "rare"]] * 2,
        }
        stm_model = STMModel(
            global_params=gp, metadata={"K": 3, "V": 10, "P": 2},
            model_spec=_FakeSpec(), covariate_names=["intercept", "cohort_b"],
            diagnostic_traces=traces,
        )
        out_dir = tmp_path / "stm_traces"
        stm_model.save(out_dir)
        loaded = STMModel.load(out_dir)

        lg = loaded.diagnostic_traces
        assert set(lg) == {"Gamma", "Sigma", "topic_block_labels"}
        for got, want in zip(lg["Gamma"], traces["Gamma"]):
            np.testing.assert_array_equal(got, want)        # 2-D preserved
        for got, want in zip(lg["Sigma"], traces["Sigma"]):
            np.testing.assert_array_equal(got, want)
        assert lg["topic_block_labels"] == traces["topic_block_labels"]

    def test_save_load_roundtrips_resume_state(self, tmp_path: Path):
        """n_iterations / elbo_trace / converged must survive save->load so a
        resumed fit continues the Robbins-Monro step counter (rho_t depends on
        the loaded iteration count) instead of restarting from t=0."""
        from spark_vi.mllib.topic.stm import STMModel
        from spark_vi.models.topic.stm import OnlineSTM

        gp = OnlineSTM(K=3, vocab_size=10, P=2, random_seed=0).initialize_global(None)
        stm_model = STMModel(
            global_params=gp,
            metadata={"K": 3},
            model_spec=_FakeSpec(),
            covariate_names=["intercept", "cohort_b"],
            n_iterations=100,
            elbo_trace=[-100.0, -90.0, -85.0],
            converged=True,
        )

        out_dir = tmp_path / "stm_resume"
        stm_model.save(out_dir)
        loaded = STMModel.load(out_dir)

        assert loaded.n_iterations == 100
        assert loaded.elbo_trace == [-100.0, -90.0, -85.0]
        assert loaded.converged is True

    def test_default_resume_state_is_fresh(self, tmp_path: Path):
        """A model constructed without resume state defaults to a fresh fit:
        n_iterations=0, empty trace, not converged."""
        from spark_vi.mllib.topic.stm import STMModel
        from spark_vi.models.topic.stm import OnlineSTM

        gp = OnlineSTM(K=2, vocab_size=5, P=1, random_seed=0).initialize_global(None)
        m = STMModel(global_params=gp, metadata={}, model_spec=_FakeSpec(),
                     covariate_names=["intercept"])
        assert m.n_iterations == 0
        assert m.elbo_trace == []
        assert m.converged is False

    def test_stm_hardening_metadata_roundtrips(self, tmp_path: Path):
        from spark_vi.mllib.topic.stm import STMModel
        from spark_vi.models.topic.stm import OnlineSTM

        model = OnlineSTM(K=3, vocab_size=10, P=2, random_seed=0)
        gp = model.initialize_global(None)
        stm_model = STMModel(
            global_params=gp,
            metadata={"K": 3, "V": 10, "P": 2, "stm_hardening": {
                "reference_topic": True,
                "sigma_prior_scale": 2.0,
                "sigma_prior_count": 500.0,
            }},
            model_spec=_FakeSpec(),
            covariate_names=["intercept", "cohort_b"],
        )
        out_dir = tmp_path / "stm_model_hardening"
        stm_model.save(out_dir)
        loaded = STMModel.load(out_dir)
        assert loaded.metadata["stm_hardening"] == {
            "reference_topic": True,
            "sigma_prior_scale": 2.0,
            "sigma_prior_count": 500.0,
        }


def test_stmmodel_roundtrips_topic_block_spec(tmp_path):
    import numpy as np
    from spark_vi.mllib.topic.stm import STMModel
    from spark_vi.models.topic.partition import TopicBlockPartition
    part = TopicBlockPartition("source_cohort", background_k=2, foreground=(("cancer", 1),))
    model = STMModel(
        global_params={"lambda": np.ones((3, 4)), "eta": np.array(0.3),
                       "Gamma": np.zeros((2, 3)), "Sigma": np.ones(3)},
        metadata={"topic_block_spec": part.to_dict()},
        model_spec=None, covariate_names=["Intercept", "age"],
        topic_blocks=part)
    model.save(tmp_path)
    loaded = STMModel.load(tmp_path)
    assert loaded.topic_blocks == part
