import numpy as np
from spark_vi.models.topic.pg_stm import stick_to_simplex, simplex_to_stick, stick_trials


def test_stick_to_simplex_sums_to_one():
    rng = np.random.default_rng(0)
    for _ in range(20):
        psi = rng.normal(size=5)          # K-1 = 5 -> K = 6
        theta = stick_to_simplex(psi)
        assert theta.shape == (6,)
        assert np.all(theta > 0)
        assert abs(theta.sum() - 1.0) < 1e-12


def test_roundtrip_psi_theta_psi():
    rng = np.random.default_rng(1)
    for _ in range(20):
        psi = rng.normal(size=7)
        theta = stick_to_simplex(psi)
        psi2 = simplex_to_stick(theta)
        assert np.allclose(psi, psi2, atol=1e-9)


def test_stick_trials_at_risk():
    n = np.array([3.0, 0.0, 5.0, 2.0])      # K=4, N=10
    b = stick_trials(n)                      # K-1 = 3
    assert np.allclose(b, [10.0, 7.0, 7.0])  # N, N-3, N-3-0
