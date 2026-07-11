import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import expit
from spark_vi.models.topic.pg_stm import (
    expected_log_theta, token_responsibilities, stick_to_simplex)


def _elog_theta_quadrature(m, v, n_nodes=64):
    """Deterministic (seed-free) reference for E[log theta] under q(psi)=N(m, diag(v)).
    Because q factorizes over sticks, each E[log sigma(psi_j)] and E[log(1-sigma(psi_j))] is a
    1-D Gaussian integral, evaluated exactly by Gauss-Hermite quadrature (probabilists' weight
    e^{-x^2/2}). The sticks are then combined exactly the way expected_log_theta does. 64 nodes
    reproduce the 200-node value to machine precision here, so this is exact to ~1e-15."""
    m = np.asarray(m, dtype=np.float64); v = np.asarray(v, dtype=np.float64)
    nodes, weights = hermegauss(n_nodes)
    weights = weights / np.sqrt(2.0 * np.pi)          # integrate against N(0,1)
    ls_plus = np.empty_like(m); ls_minus = np.empty_like(m)
    for j in range(m.shape[0]):
        x = m[j] + np.sqrt(v[j]) * nodes              # psi_j ~ N(m_j, v_j)
        sig = expit(x)
        ls_plus[j] = np.sum(weights * np.log(sig))
        ls_minus[j] = np.sum(weights * np.log1p(-sig))
    K = m.shape[0] + 1
    out = np.empty(K, dtype=np.float64)
    cum = np.concatenate([[0.0], np.cumsum(ls_minus)])
    out[:K - 1] = ls_plus + cum[:K - 1]
    out[K - 1] = cum[K - 1]
    return out


def test_expected_log_theta_matches_quadrature():
    # Deterministic reference: no RNG, no seed, exact to ~machine precision. atol=2e-3 clears the
    # measured 4th-order delta-method residual (~1.1e-3 at the first, largest-v test point) with
    # headroom. The residual grows ~v^3, so the delta method degrades at large v; that high-v
    # regime is guarded by the VI-approx-equals-Gibbs cross-check in Task 6, not by this test.
    for m, v in [
        (np.array([0.3, -0.5, 0.1]), np.array([0.4, 0.2, 0.6])),     # brief's point, K-1=3
        (np.array([0.0, 1.0, -1.0]), np.array([0.3, 0.5, 0.1])),
        (np.array([0.5, 0.5, 0.5, 0.5]), np.array([0.2, 0.4, 0.6, 0.1])),  # K-1=4
    ]:
        approx = expected_log_theta(m, v)
        exact = _elog_theta_quadrature(m, v)
        assert np.allclose(approx, exact, atol=2e-3), (m, v, approx - exact)


def test_expected_log_theta_zero_var_is_exact_logtheta():
    m = np.array([0.3, -0.5, 0.1]); v = np.zeros(3)
    assert np.allclose(expected_log_theta(m, v), np.log(stick_to_simplex(m)), atol=1e-12)


def test_token_responsibilities_normalize_and_respect_gating():
    elog_theta = np.log(np.array([0.4, 0.3, 0.2, 0.1]))
    elog_beta = np.log(np.array([                    # (K=4, V=3)
        [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.34, 0.33, 0.33]]))
    idx = np.array([0, 1]); allowed = np.array([0, 1, 2])   # topic 3 masked out
    phi, n = token_responsibilities(idx, elog_theta, elog_beta, allowed, counts=np.array([2.0, 1.0]))
    assert np.allclose(phi.sum(axis=1), 1.0)
    assert np.allclose(phi[:, 3], 0.0)              # gated-out topic gets zero
    assert abs(n.sum() - 3.0) < 1e-9                # total mass = total tokens
