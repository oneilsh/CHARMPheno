import numpy as np
from spark_vi.models.topic.pg_stm import (
    sigma_iw_posterior_mean, gamma_ridge, beta_dirichlet_mean)


def test_iw_posterior_mean_recovers_planted_cov_large_n():
    # Draw e_d ~ N(0, Sigma_true); the IW posterior mean -> Sigma_true as D grows.
    rng = np.random.default_rng(0)
    dim = 4
    A = rng.normal(size=(dim, dim)); Sigma_true = A @ A.T + np.eye(dim)
    D = 20000
    E = rng.multivariate_normal(np.zeros(dim), Sigma_true, size=D)
    scatter = E.T @ E                                   # sum e_d e_d^T (V_d = 0 here)
    Psi0 = np.eye(dim); nu0 = dim + 2
    est = sigma_iw_posterior_mean(scatter, D, Psi0=Psi0, nu0=nu0, dim=dim)
    assert np.allclose(est, Sigma_true, atol=0.15)


def test_iw_posterior_mean_is_finite_and_pd_with_zero_data():
    # The whole point: even with NO informative data, the proper prior gives a finite PD mean.
    dim = 3
    est = sigma_iw_posterior_mean(np.zeros((dim, dim)), 0, Psi0=2.0*np.eye(dim), nu0=dim+2, dim=dim)
    assert np.all(np.isfinite(est))
    assert np.all(np.linalg.eigvalsh(est) > 0)


def test_gamma_ridge_recovers_planted():
    rng = np.random.default_rng(1)
    D, P, Km1 = 5000, 3, 4
    X = rng.normal(size=(D, P)); Gamma_true = rng.normal(size=(P, Km1))
    M = X @ Gamma_true + 0.01 * rng.normal(size=(D, Km1))
    est = gamma_ridge(M, X, ridge=1e-6)
    assert np.allclose(est, Gamma_true, atol=0.05)


def test_beta_dirichlet_mean_normalizes():
    stats = np.array([[10.0, 0.0, 2.0], [0.0, 5.0, 5.0]])
    beta = beta_dirichlet_mean(stats, eta=0.1)
    assert beta.shape == (2, 3)
    assert np.allclose(beta.sum(axis=1), 1.0)
    assert beta[0, 0] > beta[0, 1]                      # more mass where more counts
