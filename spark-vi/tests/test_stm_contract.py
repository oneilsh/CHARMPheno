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


class TestLocalUpdate:
    def test_returns_expected_keys_and_shapes(self):
        m = OnlineSTM(K=3, vocab_size=10, P=2, random_seed=0)
        gp = m.initialize_global(None)
        # Inject a non-degenerate Σ so Laplace doesn't collapse.
        gp["Sigma"] = np.full(3, 1.0)
        from spark_vi.models.topic.types import STMDocument
        docs = [
            STMDocument(
                indices=np.array([0, 3, 7], dtype=np.int32),
                counts=np.array([2.0, 1.0, 1.0]),
                length=4,
                x=np.array([1.0, 0.5]),
            ),
            STMDocument(
                indices=np.array([1, 4, 8], dtype=np.int32),
                counts=np.array([1.0, 3.0, 1.0]),
                length=5,
                x=np.array([-0.5, 1.0]),
            ),
        ]
        ss = m.local_update(docs, gp)
        assert ss["lambda_stats"].shape == (3, 10)
        assert ss["XtX"].shape == (2, 2)
        assert ss["XtMu"].shape == (2, 3)
        assert ss["residual_diag_stat"].shape == (3,)
        assert ss["n_docs"].shape == ()
        assert float(ss["n_docs"]) == 2.0
        # ELBO suff stats.
        assert ss["doc_loglik_sum"].shape == ()
        assert ss["doc_eta_kl_sum"].shape == ()

    def test_lambda_stats_only_touches_seen_columns(self):
        m = OnlineSTM(K=3, vocab_size=10, P=2, random_seed=0)
        gp = m.initialize_global(None)
        from spark_vi.models.topic.types import STMDocument
        doc = STMDocument(
            indices=np.array([2, 5], dtype=np.int32),
            counts=np.array([1.0, 1.0]),
            length=2,
            x=np.array([0.0, 0.0]),
        )
        ss = m.local_update([doc], gp)
        touched = set(np.flatnonzero(ss["lambda_stats"].sum(axis=0)).tolist())
        assert touched == {2, 5}

    def test_empty_partition_returns_zero_stats(self):
        m = OnlineSTM(K=3, vocab_size=10, P=2, random_seed=0)
        gp = m.initialize_global(None)
        ss = m.local_update([], gp)
        assert float(ss["n_docs"]) == 0.0
        np.testing.assert_array_equal(ss["lambda_stats"], np.zeros((3, 10)))
        np.testing.assert_array_equal(ss["XtX"], np.zeros((2, 2)))
        np.testing.assert_array_equal(ss["XtMu"], np.zeros((2, 3)))


class TestUpdateGlobal:
    def _make_state_with_stats(self):
        m = OnlineSTM(K=3, vocab_size=10, P=2, random_seed=0)
        gp = m.initialize_global(None)
        target_stats = {
            "lambda_stats": np.ones((3, 10)) * 0.5,
            "XtX": np.eye(2) * 100.0,
            "XtMu": np.array([[1.0, -1.0, 0.5], [0.5, 0.0, -0.5]]),
            "residual_diag_stat": np.array([5.0, 3.0, 2.0]),
            "doc_loglik_sum": np.array(-100.0),
            "doc_eta_kl_sum": np.array(5.0),
            "n_docs": np.array(50.0),
        }
        return m, gp, target_stats

    def test_lambda_natural_gradient_step(self):
        m, gp, target = self._make_state_with_stats()
        lam_before = gp["lambda"].copy()
        gp_new = m.update_global(gp, target, learning_rate=1.0)
        # At ρ=1.0 the update fully replaces λ with η + lambda_stats.
        expected = float(gp["eta"]) + target["lambda_stats"]
        np.testing.assert_allclose(gp_new["lambda"], expected)

    def test_lambda_partial_step(self):
        m, gp, target = self._make_state_with_stats()
        lam_before = gp["lambda"].copy()
        gp_new = m.update_global(gp, target, learning_rate=0.3)
        # (1-ρ)·old + ρ·target.
        expected = 0.7 * lam_before + 0.3 * (float(gp["eta"]) + target["lambda_stats"])
        np.testing.assert_allclose(gp_new["lambda"], expected)

    def test_gamma_solves_ridge_regression(self):
        m, gp, target = self._make_state_with_stats()
        gp_new = m.update_global(gp, target, learning_rate=1.0)
        # Γ̂ = (XᵀX + ridge·I)⁻¹ Xᵀμ.
        ridge = m.sigma_ridge
        expected_Gamma = np.linalg.solve(
            target["XtX"] + ridge * np.eye(2), target["XtMu"]
        )
        np.testing.assert_allclose(gp_new["Gamma"], expected_Gamma)

    def test_sigma_sample_covariance(self):
        m, gp, target = self._make_state_with_stats()
        gp_new = m.update_global(gp, target, learning_rate=1.0)
        expected_Sigma = target["residual_diag_stat"] / float(target["n_docs"])
        np.testing.assert_allclose(gp_new["Sigma"], expected_Sigma)

    def test_sigma_minimum_floor(self):
        """Σ should never go below a small floor to keep Laplace well-defined."""
        m, gp, target = self._make_state_with_stats()
        target["residual_diag_stat"] = np.array([0.0, 0.0, 0.0])
        gp_new = m.update_global(gp, target, learning_rate=1.0)
        assert np.all(gp_new["Sigma"] > 0)


class TestComputeELBO:
    def test_returns_finite_float(self):
        m = OnlineSTM(K=3, vocab_size=10, P=2, random_seed=0)
        gp = m.initialize_global(None)
        aggregated = {
            "doc_loglik_sum": np.array(-50.0),
            "doc_eta_kl_sum": np.array(3.0),
            "n_docs": np.array(10.0),
        }
        elbo = m.compute_elbo(gp, aggregated)
        assert np.isfinite(elbo)

    def test_includes_negative_global_beta_kl(self):
        """ELBO = doc_loglik - doc_eta_kl - global_beta_kl. Increasing the
        beta KL should decrease the ELBO."""
        m = OnlineSTM(K=3, vocab_size=10, P=2, random_seed=0)
        gp_low_kl = m.initialize_global(None)
        # Concentrate λ on one column → high KL vs uniform prior.
        gp_high_kl = {**gp_low_kl, "lambda": gp_low_kl["lambda"].copy()}
        gp_high_kl["lambda"][:, 0] *= 100.0
        agg = {
            "doc_loglik_sum": np.array(-50.0),
            "doc_eta_kl_sum": np.array(3.0),
            "n_docs": np.array(10.0),
        }
        elbo_low_kl = m.compute_elbo(gp_low_kl, agg)
        elbo_high_kl = m.compute_elbo(gp_high_kl, agg)
        assert elbo_high_kl < elbo_low_kl


class TestInferLocal:
    def test_returns_eta_theta_for_single_doc(self):
        m = OnlineSTM(K=3, vocab_size=10, P=2, random_seed=0)
        gp = m.initialize_global(None)
        from spark_vi.models.topic.types import STMDocument
        doc = STMDocument(
            indices=np.array([0, 3], dtype=np.int32),
            counts=np.array([2.0, 1.0]),
            length=3,
            x=np.array([0.5, -0.3]),
        )
        out = m.infer_local(doc, gp)
        assert out["eta"].shape == (3,)
        assert out["theta"].shape == (3,)
        np.testing.assert_allclose(out["theta"].sum(), 1.0, atol=1e-10)
        assert np.all(out["theta"] > 0)


class TestIterationSummary:
    def test_returns_compact_string(self):
        m = OnlineSTM(K=3, vocab_size=10, P=2, random_seed=0)
        gp = m.initialize_global(None)
        s = m.iteration_summary(gp)
        assert isinstance(s, str)
        assert "Γ" in s or "Gamma" in s
        assert "Σ" in s or "Sigma" in s


class TestIterationDiagnostics:
    def test_returns_traceable_arrays(self):
        m = OnlineSTM(K=3, vocab_size=10, P=2, random_seed=0)
        gp = m.initialize_global(None)
        d = m.iteration_diagnostics(gp)
        assert "Gamma" in d
        assert "Sigma" in d
        assert d["Gamma"].shape == (2, 3)
        assert d["Sigma"].shape == (3,)
