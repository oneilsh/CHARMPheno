import numpy as np
import pytest
from spark_vi.models.topic._linalg import pd_complete


def _is_pd(M):
    try:
        np.linalg.cholesky(M)
        return True
    except np.linalg.LinAlgError:
        return False


def test_identity_when_all_observed():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 5))
    Sigma = A @ A.T + np.eye(5)
    out = pd_complete(Sigma, np.ones((5, 5), bool))
    np.testing.assert_allclose(out, Sigma, atol=1e-9)


def test_decomposable_closed_form_oracle():
    # topics: 0 = separator (background), 1 and 2 cond. independent given 0.
    target = np.array([[2.0, 1.0, 0.8],
                       [1.0, 1.5, 0.0],   # (1,2) is free
                       [0.8, 0.0, 1.2]])
    mask = np.array([[1, 1, 1],
                     [1, 1, 0],
                     [1, 0, 1]], bool)
    out = pd_complete(target, mask)
    # Grone/Lauritzen closed form for the free cross entry:
    expected_12 = target[1, 0] * (1.0 / target[0, 0]) * target[0, 2]   # = 0.4
    assert abs(out[1, 2] - expected_12) < 1e-8
    assert abs(out[2, 1] - expected_12) < 1e-8
    # observed entries preserved exactly
    for i, j in [(0, 1), (0, 2), (1, 1), (2, 2), (0, 0)]:
        assert abs(out[i, j] - target[i, j]) < 1e-8
    # zero precision on the free entry (conditional independence)
    prec = np.linalg.inv(out)
    assert abs(prec[1, 2]) < 1e-7
    assert _is_pd(out)


def test_general_nonchordal_pd_and_zero_precision_on_free():
    # 4 topics in a 4-cycle of observed edges: 0-1,1-2,2-3,3-0 observed;
    # 0-2 and 1-3 free (non-decomposable pattern -> IPS must iterate).
    rng = np.random.default_rng(1)
    A = rng.standard_normal((4, 4))
    target = A @ A.T + 2 * np.eye(4)
    mask = np.array([[1, 1, 0, 1],
                     [1, 1, 1, 0],
                     [0, 1, 1, 1],
                     [1, 0, 1, 1]], bool)
    out = pd_complete(target, mask)
    assert _is_pd(out)
    prec = np.linalg.inv(out)
    assert abs(prec[0, 2]) < 1e-6 and abs(prec[1, 3]) < 1e-6   # free -> zero precision
    for i in range(4):                                          # observed preserved
        for j in range(4):
            if mask[i, j]:
                assert abs(out[i, j] - target[i, j]) < 1e-6


def test_non_pd_completable_observed_returns_pd_without_raising():
    # Observed entries that admit no PD completion: a near-rank-deficient block.
    target = np.array([[1.0, 0.99, 0.99],
                       [0.99, 1.0, -0.99],   # inconsistent with the above two
                       [0.99, -0.99, 1.0]])
    mask = np.ones((3, 3), bool)             # all "observed" but not PD
    out = pd_complete(target, mask)          # must not raise
    assert _is_pd(out)
    np.testing.assert_allclose(out, out.T, atol=1e-10)


def test_output_symmetric():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((6, 6))
    target = A @ A.T + np.eye(6)
    mask = np.ones((6, 6), bool)
    mask[0, 4] = mask[4, 0] = False
    out = pd_complete(target, mask)
    np.testing.assert_allclose(out, out.T, atol=1e-10)
