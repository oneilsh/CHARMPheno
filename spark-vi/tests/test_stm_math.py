"""Tests for STM per-doc inference math: STMDocument, gradient, Hessian, MAP."""
from __future__ import annotations

import numpy as np
import pytest

from spark_vi.models.topic.types import STMDocument


class TestSTMDocument:
    def test_constructs_with_indices_counts_length_x(self):
        doc = STMDocument(
            indices=np.array([0, 3, 5], dtype=np.int32),
            counts=np.array([2.0, 1.0, 3.0], dtype=np.float64),
            length=6,
            x=np.array([1.0, 0.5, -1.2], dtype=np.float64),
        )
        assert doc.length == 6
        assert doc.x.shape == (3,)
        assert doc.x.dtype == np.float64

    def test_is_frozen(self):
        doc = STMDocument(
            indices=np.array([0], dtype=np.int32),
            counts=np.array([1.0]),
            length=1,
            x=np.array([0.0]),
        )
        with pytest.raises((AttributeError, TypeError)):
            doc.length = 99


from scipy.special import softmax

from spark_vi.models.topic.stm import _stm_neg_log_joint, _stm_neg_log_joint_grad


def _make_small_doc_state(seed=0):
    """Construct a small (K, V, P) state for gradient / Hessian tests."""
    rng = np.random.default_rng(seed)
    K, V, P = 3, 5, 2
    # ExpElogbeta-style nonnegative K x V matrix, columns summing roughly to 1.
    expElogbeta = rng.gamma(shape=2.0, scale=1.0, size=(K, V))
    expElogbeta = expElogbeta / expElogbeta.sum(axis=0, keepdims=True)
    Gamma = rng.normal(size=(P, K))
    Sigma_diag = rng.gamma(shape=2.0, scale=0.5, size=K)
    x = rng.normal(size=P)
    indices = np.array([0, 2, 4], dtype=np.int32)
    counts = np.array([2.0, 1.0, 3.0], dtype=np.float64)
    return dict(
        K=K, V=V, P=P, expElogbeta=expElogbeta, Gamma=Gamma,
        Sigma_diag=Sigma_diag, x=x, indices=indices, counts=counts,
    )


class TestSTMGradient:
    def test_gradient_matches_finite_difference(self):
        st = _make_small_doc_state(seed=42)
        rng = np.random.default_rng(0)
        eta = rng.normal(size=st["K"]) * 0.3

        analytic = _stm_neg_log_joint_grad(
            eta,
            indices=st["indices"], counts=st["counts"],
            expElogbeta=st["expElogbeta"],
            Gamma=st["Gamma"], Sigma_diag=st["Sigma_diag"], x=st["x"],
        )

        eps = 1e-6
        numeric = np.zeros_like(eta)
        for k in range(st["K"]):
            eta_p = eta.copy(); eta_p[k] += eps
            eta_m = eta.copy(); eta_m[k] -= eps
            f_p = _stm_neg_log_joint(
                eta_p, indices=st["indices"], counts=st["counts"],
                expElogbeta=st["expElogbeta"],
                Gamma=st["Gamma"], Sigma_diag=st["Sigma_diag"], x=st["x"],
            )
            f_m = _stm_neg_log_joint(
                eta_m, indices=st["indices"], counts=st["counts"],
                expElogbeta=st["expElogbeta"],
                Gamma=st["Gamma"], Sigma_diag=st["Sigma_diag"], x=st["x"],
            )
            numeric[k] = (f_p - f_m) / (2 * eps)

        np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-6)


from spark_vi.models.topic.stm import _stm_neg_log_joint_hessian


class TestSTMHessian:
    def test_hessian_matches_finite_difference_of_grad(self):
        st = _make_small_doc_state(seed=7)
        rng = np.random.default_rng(1)
        eta = rng.normal(size=st["K"]) * 0.3

        analytic = _stm_neg_log_joint_hessian(
            eta, indices=st["indices"], counts=st["counts"],
            expElogbeta=st["expElogbeta"],
            Gamma=st["Gamma"], Sigma_diag=st["Sigma_diag"], x=st["x"],
        )

        eps = 1e-5
        numeric = np.zeros((st["K"], st["K"]))
        for j in range(st["K"]):
            eta_p = eta.copy(); eta_p[j] += eps
            eta_m = eta.copy(); eta_m[j] -= eps
            g_p = _stm_neg_log_joint_grad(
                eta_p, indices=st["indices"], counts=st["counts"],
                expElogbeta=st["expElogbeta"],
                Gamma=st["Gamma"], Sigma_diag=st["Sigma_diag"], x=st["x"],
            )
            g_m = _stm_neg_log_joint_grad(
                eta_m, indices=st["indices"], counts=st["counts"],
                expElogbeta=st["expElogbeta"],
                Gamma=st["Gamma"], Sigma_diag=st["Sigma_diag"], x=st["x"],
            )
            numeric[:, j] = (g_p - g_m) / (2 * eps)

        np.testing.assert_allclose(analytic, numeric, rtol=1e-3, atol=1e-5)

    def test_hessian_is_symmetric(self):
        st = _make_small_doc_state(seed=11)
        rng = np.random.default_rng(2)
        eta = rng.normal(size=st["K"]) * 0.3
        H = _stm_neg_log_joint_hessian(
            eta, indices=st["indices"], counts=st["counts"],
            expElogbeta=st["expElogbeta"],
            Gamma=st["Gamma"], Sigma_diag=st["Sigma_diag"], x=st["x"],
        )
        np.testing.assert_allclose(H, H.T, rtol=1e-12, atol=1e-12)

    def test_hessian_positive_definite_at_typical_point(self):
        st = _make_small_doc_state(seed=13)
        rng = np.random.default_rng(3)
        eta = rng.normal(size=st["K"]) * 0.3
        H = _stm_neg_log_joint_hessian(
            eta, indices=st["indices"], counts=st["counts"],
            expElogbeta=st["expElogbeta"],
            Gamma=st["Gamma"], Sigma_diag=st["Sigma_diag"], x=st["x"],
        )
        # Negative log joint is convex in η for prevalence-only STM with
        # diagonal Σ + nonnegative β; H should be PD.
        eigs = np.linalg.eigvalsh(H)
        assert np.all(eigs > 0), f"Hessian not PD: eigs={eigs}"


from spark_vi.models.topic.stm import _stm_doc_inference


class TestSTMDocInference:
    def test_converges_to_stationary_point(self):
        st = _make_small_doc_state(seed=99)
        eta_hat, nu_d, n_iter = _stm_doc_inference(
            indices=st["indices"], counts=st["counts"],
            expElogbeta=st["expElogbeta"],
            Gamma=st["Gamma"], Sigma_diag=st["Sigma_diag"], x=st["x"],
            max_iter=200, tol=1e-6,
        )
        # Gradient at η̂ should be ~zero.
        g = _stm_neg_log_joint_grad(
            eta_hat, indices=st["indices"], counts=st["counts"],
            expElogbeta=st["expElogbeta"],
            Gamma=st["Gamma"], Sigma_diag=st["Sigma_diag"], x=st["x"],
        )
        assert np.linalg.norm(g) < 1e-4, f"|g|={np.linalg.norm(g)} not converged"
        assert nu_d.shape == (st["K"], st["K"])
        # ν_d is symmetric positive definite.
        np.testing.assert_allclose(nu_d, nu_d.T, atol=1e-10)
        eigs = np.linalg.eigvalsh(nu_d)
        assert np.all(eigs > 0)

    def test_strong_prior_pulls_eta_toward_prior_mean(self):
        st = _make_small_doc_state(seed=1)
        # Override Σ to be very tight: posterior should ~= prior mean Γᵀx.
        Sigma_tight = np.full(st["K"], 1e-6)
        prior_mean = st["Gamma"].T @ st["x"]
        eta_hat, _, _ = _stm_doc_inference(
            indices=st["indices"], counts=st["counts"],
            expElogbeta=st["expElogbeta"],
            Gamma=st["Gamma"], Sigma_diag=Sigma_tight, x=st["x"],
            max_iter=200, tol=1e-8,
        )
        np.testing.assert_allclose(eta_hat, prior_mean, atol=1e-3)


from spark_vi.models.topic.stm import prior_topic_proportions


class TestPriorTopicProportions:
    """Covariate-implied (prior) topic proportions: softmax(Γᵀ x) for one doc.

    This is the per-document primitive the dashboard's corpus-mean α-equivalent
    averages over. It is the prior mean of η_d pushed through softmax — the
    topic mix a document is expected to have from its covariates alone, before
    any token evidence.
    """

    def test_matches_softmax_of_gamma_transpose_x(self):
        rng = np.random.default_rng(7)
        P, K = 2, 3
        Gamma = rng.normal(size=(P, K))
        x = rng.normal(size=P)

        result = prior_topic_proportions(Gamma, x)

        np.testing.assert_allclose(result, softmax(Gamma.T @ x))

    def test_is_a_probability_vector(self):
        rng = np.random.default_rng(8)
        Gamma = rng.normal(size=(4, 5))
        x = rng.normal(size=4)

        result = prior_topic_proportions(Gamma, x)

        assert result.shape == (5,)
        assert np.all(result >= 0.0)
        np.testing.assert_allclose(result.sum(), 1.0)

    def test_intercept_only_row_reproduces_gamma_intercept_softmax(self):
        # When x selects only the intercept (one-hot at the intercept row),
        # Γᵀx is exactly the intercept row, so the proportions equal
        # softmax(Γ[intercept]) — the v1 stand-in adapt_stm currently uses.
        rng = np.random.default_rng(9)
        P, K = 3, 4
        Gamma = rng.normal(size=(P, K))
        intercept_idx = 0
        x = np.zeros(P)
        x[intercept_idx] = 1.0

        result = prior_topic_proportions(Gamma, x)

        np.testing.assert_allclose(result, softmax(Gamma[intercept_idx]))


from spark_vi.models.topic.stm import corpus_mean_topic_proportions


class TestCorpusMeanTopicProportions:
    """The α-equivalent: (1/D) Σ_d softmax(Γᵀ x_d) over the design matrix X."""

    def test_equals_row_mean_of_per_row_proportions(self):
        rng = np.random.default_rng(11)
        D, P, K = 5, 2, 3
        Gamma = rng.normal(size=(P, K))
        X = rng.normal(size=(D, P))

        result = corpus_mean_topic_proportions(Gamma, X)

        expected = np.mean(
            [prior_topic_proportions(Gamma, X[d]) for d in range(D)], axis=0
        )
        np.testing.assert_allclose(result, expected)

    def test_is_a_probability_vector(self):
        rng = np.random.default_rng(12)
        Gamma = rng.normal(size=(3, 4))
        X = rng.normal(size=(7, 3))

        result = corpus_mean_topic_proportions(Gamma, X)

        assert result.shape == (4,)
        assert np.all(result >= 0.0)
        np.testing.assert_allclose(result.sum(), 1.0)

    def test_constant_covariates_reduce_to_single_row_softmax(self):
        # If every document shares the same covariate row, the corpus mean is
        # just that row's proportions — the nonlinearity has nothing to average.
        rng = np.random.default_rng(13)
        P, K = 3, 4
        Gamma = rng.normal(size=(P, K))
        x = rng.normal(size=P)
        X = np.tile(x, (6, 1))

        result = corpus_mean_topic_proportions(Gamma, X)

        np.testing.assert_allclose(result, prior_topic_proportions(Gamma, x))
