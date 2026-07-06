"""Tests for smooth_scale_log_quadratic (spark_vi.mllib.topic.stm).

The held-out-LL-vs-scale sweep (corpus_heldout_scale_sweep_gated{,_rdd}) scores
a small grid of scales c and previously reduced it via a raw argmax. The
held-out LL curve is a broad, flat SHELF (differences ~0.001-0.01 nats across
c in [2,12], within resampling noise), so argmax over a coarse, roughly-LINEAR
grid on a flat curve is a quantized, jittery point estimate. This module pins
the sign conventions of the replacement reducer: a local quadratic fit in log
c, which (a) recovers an off-grid vertex when the curve genuinely peaks, (b)
stays honest (large/undefined SE, near-zero curvature) when the curve is
flat, and (c) falls back to the raw grid argmax at a monotone boundary
instead of extrapolating a spurious interior peak.
"""
from __future__ import annotations

import numpy as np
import pytest

from spark_vi.mllib.topic.stm import smooth_scale_log_quadratic


def _geomspace_grid(lo=0.5, hi=32.0, num=13):
    return [round(x, 4) for x in np.geomspace(lo, hi, num=num)]


def test_recovers_known_vertex():
    # Exact downward parabola in log c: y = -(u - u0)^2, u0 = ln(6.3).
    # Grid straddles the vertex but need not land on it.
    grid = _geomspace_grid()
    u0 = np.log(6.3)
    lls = {c: -(np.log(c) - u0) ** 2 for c in grid}

    out = smooth_scale_log_quadratic(lls)

    assert out["interior"] is True
    assert out["curvature_q"] < 0
    assert out["c_star"] == pytest.approx(6.3, rel=0.05)


def test_flat_shelf_is_honest():
    # A near-constant LL "shelf" with a tiny resampling-noise-scale wobble:
    # two neighboring grid points are tied for the highest LL (a shallow
    # dip sits between them), which is exactly the kind of flat-shelf shape
    # the held-out sweep produces in practice. Handcrafted (not rng-based)
    # so the test is deterministic: two ties flanking a slightly lower
    # center point make the local window look locally CONVEX (a shallow
    # valley), not concave, so the estimator should decline to claim an
    # interior peak here at all.
    grid = _geomspace_grid()
    y = [-1.0] * len(grid)
    y[5] = -0.999   # tied-for-max point (left)
    y[7] = -0.999   # tied-for-max point (right)
    y[6] = -1.0002  # shallow dip between them -> locally convex window
    lls = dict(zip(grid, y))

    out = smooth_scale_log_quadratic(lls)

    # Curvature must be tiny (flat shelf), and the estimator must not
    # fabricate a sharp interior peak: either the p2>=0 fallback fires
    # (se_c is None, interior False) or, if it does report an interior
    # vertex, its SE must be large relative to c_star.
    assert abs(out["curvature_q"]) < 1e-2
    assert out["se_c"] is None or out["se_c"] > 0.5 * out["c_star"]
    assert out["interior"] is False


def test_monotone_rising_falls_back_to_top_boundary():
    # Strictly increasing LL: the true max is at the top grid point, and the
    # window there is convex/flat looking backward -- no spurious interior
    # vertex should be invented.
    grid = _geomspace_grid()
    lls = {c: np.log(c) for c in grid}

    out = smooth_scale_log_quadratic(lls)

    assert out["interior"] is False
    assert out["c_star"] == out["grid_argmax_c"]
    assert out["c_star"] == max(grid)


def test_monotone_falling_falls_back_to_bottom_boundary():
    # Symmetric case: strictly decreasing LL -> max at the bottom grid point.
    grid = _geomspace_grid()
    lls = {c: -np.log(c) for c in grid}

    out = smooth_scale_log_quadratic(lls)

    assert out["interior"] is False
    assert out["c_star"] == out["grid_argmax_c"]
    assert out["c_star"] == min(grid)


def test_degenerate_fewer_than_three_points():
    lls = {2.0: -1.0, 5.0: -0.5}

    out = smooth_scale_log_quadratic(lls)

    assert out["interior"] is False
    assert out["c_star"] == out["grid_argmax_c"] == 5.0
    assert out["curvature_q"] == 0.0
    assert out["se_log_c"] is None
    assert out["se_c"] is None
