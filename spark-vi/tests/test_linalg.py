import numpy as np
from spark_vi.models.topic._linalg import safe_inverse, nearest_spd, topic_correlation

def test_safe_inverse_matches_inv_for_spd():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(5, 5)); M = A @ A.T + np.eye(5)
    assert np.allclose(safe_inverse(M), np.linalg.inv(M))

def test_safe_inverse_repairs_indefinite():
    M = np.diag([1.0, -2.0, 3.0])  # indefinite
    inv = safe_inverse(M)
    w = np.linalg.eigvalsh(inv)
    assert np.all(w > 0)  # SPD result despite indefinite input

def test_nearest_spd_floors_eigenvalues_and_symmetrizes():
    M = np.array([[1.0, 0.9, 0.9],
                  [0.9, 1.0, -0.9],
                  [0.9, -0.9, 1.0]])  # symmetric but indefinite
    S = nearest_spd(M, floor=1e-6)
    assert np.allclose(S, S.T)
    assert np.min(np.linalg.eigvalsh(S)) >= 1e-6 - 1e-12

def test_nearest_spd_is_identity_on_spd():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(4, 4)); M = A @ A.T + np.eye(4)
    assert np.allclose(nearest_spd(M), M)

def test_topic_correlation_unit_diagonal_and_values():
    Sigma = np.array([[4.0, 2.0], [2.0, 9.0]])
    R = topic_correlation(Sigma)
    assert np.allclose(np.diag(R), 1.0)
    assert np.isclose(R[0, 1], 2.0 / np.sqrt(4.0 * 9.0))
