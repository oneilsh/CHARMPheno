"""Tests for OnlineSTM's VIModel contract conformance."""
from __future__ import annotations

import numpy as np
import pytest

from spark_vi.models.topic.stm import OnlineSTM


class TestConstructor:
    def test_constructs_with_minimal_args(self):
        m = OnlineSTM(K=5, vocab_size=100, P=3)
        assert m.K == 5
        assert m.V == 100
        assert m.P == 3

    def test_rejects_invalid_K(self):
        with pytest.raises(ValueError, match="K must be >= 1"):
            OnlineSTM(K=0, vocab_size=100, P=3)

    def test_rejects_invalid_vocab_size(self):
        with pytest.raises(ValueError, match="vocab_size must be >= 1"):
            OnlineSTM(K=5, vocab_size=0, P=3)

    def test_rejects_invalid_P(self):
        with pytest.raises(ValueError, match="P must be >= 1"):
            OnlineSTM(K=5, vocab_size=100, P=0)

    def test_rejects_invalid_sigma_ridge(self):
        with pytest.raises(ValueError, match="sigma_ridge must be >= 0"):
            OnlineSTM(K=5, vocab_size=100, P=3, sigma_ridge=-1.0)


class TestInitializeGlobal:
    def test_returns_lambda_eta_gamma_sigma_shapes(self):
        m = OnlineSTM(K=4, vocab_size=20, P=2, random_seed=42)
        gp = m.initialize_global(data_summary=None)
        assert gp["lambda"].shape == (4, 20)
        assert gp["eta"].shape == ()
        assert gp["Gamma"].shape == (2, 4)
        assert gp["Sigma"].shape == (4,)
        # Sigma starts at sigma_init.
        np.testing.assert_allclose(gp["Sigma"], np.full(4, 1.0))
        # Gamma starts at zeros (covariates have no effect at init).
        np.testing.assert_allclose(gp["Gamma"], np.zeros((2, 4)))

    def test_seeded_init_is_deterministic(self):
        gp1 = OnlineSTM(K=3, vocab_size=10, P=2, random_seed=7).initialize_global(None)
        gp2 = OnlineSTM(K=3, vocab_size=10, P=2, random_seed=7).initialize_global(None)
        np.testing.assert_array_equal(gp1["lambda"], gp2["lambda"])


class TestGetMetadata:
    def test_returns_K_V_P(self):
        m = OnlineSTM(K=5, vocab_size=100, P=3)
        md = m.get_metadata()
        assert md == {"K": 5, "V": 100, "P": 3}
