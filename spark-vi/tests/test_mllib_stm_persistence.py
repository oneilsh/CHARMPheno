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
