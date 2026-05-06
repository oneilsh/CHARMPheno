"""Pure-numpy tests for OnlineHDP module-level math helpers and CAVI.

No Spark — these test the math in isolation. Single document, hand-checked
shapes and values where possible.
"""
import numpy as np
from scipy.special import digamma


def test_log_normalize_rows_simplex_invariant():
    """exp(out) rows sum to 1; out and input differ by a constant per row."""
    from spark_vi.models.online_hdp import _log_normalize_rows

    rng = np.random.default_rng(0)
    M = rng.normal(size=(5, 7))
    out = _log_normalize_rows(M)

    assert out.shape == M.shape
    assert np.allclose(np.exp(out).sum(axis=1), 1.0)

    # Each row of (M - out) should be a single repeated constant (the log-norm).
    diff = M - out
    assert np.allclose(diff, diff[:, [0]])


def test_log_normalize_rows_handles_large_magnitudes():
    """Numerical stability under large positive entries (no inf/nan)."""
    from spark_vi.models.online_hdp import _log_normalize_rows

    M = np.array([[1000.0, 1001.0, 999.0],
                  [-1000.0, -1001.0, -999.0]])
    out = _log_normalize_rows(M)

    assert np.all(np.isfinite(out))
    assert np.allclose(np.exp(out).sum(axis=1), 1.0)


def test_expect_log_sticks_uniform_prior_known_values():
    """For Beta(1, 1) sticks, the math reduces to a closed form we can hand-check.

    With a = b = 1 (uniform Beta), digamma(1) = -gamma_euler ≈ -0.5772 and
    digamma(2) = 1 - gamma_euler ≈ 0.4228. So:
      Elog_W   = digamma(1) - digamma(2) = -1.0
      Elog_1mW = digamma(1) - digamma(2) = -1.0
    For T=3 (a, b each length 2):
      out[0] = Elog_W[0]                                = -1.0
      out[1] = Elog_W[1] + Elog_1mW[0]                  = -2.0
      out[2] =             Elog_1mW[0] + Elog_1mW[1]    = -2.0
    """
    from spark_vi.models.online_hdp import _expect_log_sticks

    a = np.array([1.0, 1.0])
    b = np.array([1.0, 1.0])
    out = _expect_log_sticks(a, b)

    assert out.shape == (3,)
    assert np.allclose(out, [-1.0, -2.0, -2.0])


def test_expect_log_sticks_truncation_handles_last_atom():
    """For T atoms with (T-1) sticks, the trailing entry receives only the
    cumulative E[log(1-W)] sum (q(W_T = 1) = 1 ⇒ E[log W_T] = 0)."""
    from spark_vi.models.online_hdp import _expect_log_sticks

    rng = np.random.default_rng(42)
    T_minus_1 = 5
    a = 1.0 + rng.gamma(2.0, 1.0, T_minus_1)
    b = 1.0 + rng.gamma(2.0, 1.0, T_minus_1)
    out = _expect_log_sticks(a, b)

    dig_sum = digamma(a + b)
    Elog_1mW = digamma(b) - dig_sum
    expected_last = Elog_1mW.sum()

    assert out.shape == (T_minus_1 + 1,)
    assert np.isclose(out[-1], expected_last)

    # Spot-check intermediate positions to catch cumsum off-by-one.
    assert np.isclose(out[0], digamma(a[0]) - dig_sum[0])
    expected_2 = (digamma(a[2]) - dig_sum[2]) + (
        (digamma(b[0]) - dig_sum[0]) + (digamma(b[1]) - dig_sum[1])
    )
    assert np.isclose(out[2], expected_2)
