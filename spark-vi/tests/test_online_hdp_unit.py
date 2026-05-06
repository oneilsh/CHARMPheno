"""Pure-numpy tests for OnlineHDP module-level math helpers and CAVI.

No Spark — these test the math in isolation. Single document, hand-checked
shapes and values where possible.
"""
import numpy as np


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
