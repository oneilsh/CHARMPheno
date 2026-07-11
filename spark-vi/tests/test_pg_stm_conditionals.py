import numpy as np
from scipy.integrate import quad
from spark_vi.models.topic.pg_stm import omega_expectation, omega_sample, psi_posterior


def test_omega_expectation_matches_pg_mean():
    # E[PG(b,c)] = (b/2c) tanh(c/2); check vs a large-sample MC of random_polyagamma
    from polyagamma import random_polyagamma
    rng = np.random.default_rng(0)
    b = np.array([2.0, 5.0]); c = np.array([0.7, 1.3])
    mc = random_polyagamma(h=np.repeat(b, 200000).reshape(2, -1).T,
                           z=np.repeat(c, 200000).reshape(2, -1).T,
                           random_state=rng).mean(axis=0)
    assert np.allclose(omega_expectation(b, c), mc, rtol=2e-2)


def test_omega_expectation_zero_limit():
    b = np.array([4.0])
    val = omega_expectation(b, np.array([1e-9]))
    assert abs(val[0] - 1.0) < 1e-6           # b/4 = 1.0


def test_psi_posterior_matches_bruteforce_1stick():
    # K=2 (one stick). Posterior over psi given a Binomial(N, sigma(psi)) likelihood and
    # Gaussian prior N(mu, s2) is, under PG with a FIXED omega, exactly N(m, V). Verify the
    # returned (m, V) equals the closed-form Gaussian obtained by completing the square.
    n = np.array([7.0, 3.0]); b = np.array([10.0])       # a=7, b=10, kappa=2
    mu = np.array([0.4]); s2 = 1.7
    Sigma_inv = np.array([[1.0 / s2]])
    omega = np.array([0.9])
    m, V = psi_posterior(n, b, mu, Sigma_inv, omega)
    kappa = n[:1] - b / 2
    V_ref = 1.0 / (1.0 / s2 + omega[0])
    m_ref = V_ref * (mu[0] / s2 + kappa[0])
    assert np.allclose(V, [[V_ref]], atol=1e-12)
    assert np.allclose(m, [m_ref], atol=1e-12)


def test_psi_posterior_two_sticks_coupled_prior():
    # With a correlated 2x2 prior, V must be (Sigma_inv + diag(omega))^-1 exactly.
    n = np.array([4.0, 2.0, 1.0]); b = np.array([7.0, 3.0])
    mu = np.array([0.1, -0.2])
    Sigma = np.array([[1.5, 0.6], [0.6, 1.2]]); Sigma_inv = np.linalg.inv(Sigma)
    omega = np.array([0.5, 0.8])
    m, V = psi_posterior(n, b, mu, Sigma_inv, omega)
    V_ref = np.linalg.inv(Sigma_inv + np.diag(omega))
    kappa = n[:2] - b / 2
    m_ref = V_ref @ (Sigma_inv @ mu + kappa)
    assert np.allclose(V, V_ref, atol=1e-12)
    assert np.allclose(m, m_ref, atol=1e-12)
