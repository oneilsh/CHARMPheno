"""Correctness check for the pure-numpy Pólya-Gamma sampler (spark_vi.models.topic._pg)
against the reference ``polyagamma`` package (dev-only dependency) and the analytic
moments. The pure-numpy sampler exists so the engine needs no native polyagamma on the
Spark executors; this test is the "use the local polyagamma as a correctness check" gate.
"""
import numpy as np
import pytest

from spark_vi.models.topic._pg import random_polyagamma as pg_ours, _pg1_mean, _pg1_var

pgpkg = pytest.importorskip("polyagamma")


def _analytic_mean(h, z):
    z = abs(z)
    return h * (0.25 if z < 1e-8 else np.tanh(z / 2.0) / (2.0 * z))


@pytest.mark.parametrize("h,z", [(1, 0.0), (1, 1.5), (1, -3.0),
                                 (5, 0.7), (10, 2.0), (3, 0.01)])
def test_exact_pg_matches_analytic_and_reference_moments(h, z):
    """Empirical mean/variance of the pure-numpy exact sampler match the analytic PG
    moments AND the reference polyagamma package (small h = exact sum path)."""
    N = 40000
    rng = np.random.default_rng(0)
    ours = np.array([pg_ours(np.array([h]), np.array([z]), random_state=rng)[0]
                     for _ in range(N)])
    ref = pgpkg.random_polyagamma(h=h, z=z, size=N,
                                  random_state=np.random.default_rng(1))
    m_analytic = _analytic_mean(h, z)
    # means agree with the analytic value and each other (SEM ~ std/sqrt(N)).
    assert abs(ours.mean() - m_analytic) < 0.02 * max(1.0, abs(m_analytic)) + 0.005
    assert abs(ours.mean() - ref.mean()) < 0.01 + 0.02 * abs(ref.mean())
    # variances agree (looser — 4th-moment noise).
    assert abs(ours.var() - ref.var()) < 0.15 * ref.var() + 1e-3
    assert np.all(ours > 0)


def test_large_h_gaussian_approx_matches_reference():
    """For large h the CLT-normal approximation matches polyagamma's mean/variance."""
    h, z = 200, 1.3
    N = 20000
    rng = np.random.default_rng(0)
    ours = np.array([pg_ours(np.array([h]), np.array([z]), random_state=rng)[0]
                     for _ in range(N)])
    ref = pgpkg.random_polyagamma(h=h, z=z, size=N,
                                  random_state=np.random.default_rng(1))
    assert abs(ours.mean() - ref.mean()) < 0.01 * ref.mean()
    assert abs(ours.var() - ref.var()) < 0.10 * ref.var()


def test_zero_h_is_degenerate_zero():
    rng = np.random.default_rng(0)
    out = pg_ours(np.array([0, 2, 0]), np.array([1.0, 0.5, -2.0]), random_state=rng)
    assert out[0] == 0.0 and out[2] == 0.0 and out[1] > 0.0


def test_pg1_moment_helpers_match_reference():
    """The mean/var helpers used by the large-h approx match polyagamma at h=1."""
    for z in (0.0, 0.5, 2.0, 5.0):
        ref = pgpkg.random_polyagamma(h=1, z=z, size=200000,
                                      random_state=np.random.default_rng(2))
        assert abs(_pg1_mean(z) - ref.mean()) < 0.01
        assert abs(_pg1_var(z) - ref.var()) < 0.05 * ref.var() + 1e-4
