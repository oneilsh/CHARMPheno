import numpy as np
from spark_vi.models.topic._linalg import (
    pd_complete,
    nearest_spd,
    min_frobenius_psd_completion,
)


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


# --- Dykstra min-Frobenius PSD fallback (non-PD-completable observed) ---

# Reviewer's canonical inconsistent target: 1~2 and 1~3 both strongly positive
# (0.99) yet 2~3 strongly negative (-0.99) — no PSD matrix matches all three, so
# the observed block is not PD-completable.
_INCONSISTENT = np.array([[1.0, 0.99, 0.99],
                          [0.99, 1.0, -0.99],
                          [0.99, -0.99, 1.0]])


def _observed_dev(out, target, mask):
    """Frobenius norm of the deviation on observed entries only."""
    diff = (out - target)[mask]
    return float(np.sqrt(np.sum(diff * diff)))


def test_min_frobenius_optimal_on_all_observed_inconsistent():
    # Contract #2, all-observed half: when EVERY entry is observed the affine
    # "match-observed" set is the single point {target}, so the closest PSD matrix
    # to it is exactly the global nearest-PSD projection nearest_spd(target) — there
    # is no free entry to exploit, and the min-Frobenius optimum provably EQUALS the
    # single floor here. The non-vacuous check is therefore that Dykstra reaches that
    # optimum (does not drift PAST it, as the old in-loop floor-and-continue could) and
    # returns a strictly-PD, symmetric result. The strict min-Frobenius IMPROVEMENT
    # over a single floor is exercised in the free-entry test below, where the routine
    # has degrees of freedom to use.
    target = _INCONSISTENT
    mask = np.ones((3, 3), bool)

    dyk = min_frobenius_psd_completion(target, mask)
    floored = nearest_spd(target)

    assert _is_pd(dyk)                                   # strictly PD
    np.testing.assert_allclose(dyk, dyk.T, atol=1e-10)   # symmetric
    # Reaches the optimum (the single floor IS the min-Frobenius point here);
    # never worse than it.
    assert _observed_dev(dyk, target, mask) <= _observed_dev(floored, target, mask) + 1e-9
    np.testing.assert_allclose(dyk, floored, atol=1e-6)


def test_pd_complete_inconsistent_all_observed_does_not_raise_and_is_pd():
    # pd_complete on the all-observed inconsistent target routes through the
    # min-Frobenius fallback (the observed-only init is indefinite) and returns a
    # strictly-PD, symmetric matrix.
    target = _INCONSISTENT
    mask = np.ones((3, 3), bool)
    out = pd_complete(target, mask)
    assert _is_pd(out)
    np.testing.assert_allclose(out, out.T, atol=1e-10)


def test_min_frobenius_strictly_beats_single_floor_with_free_entry():
    # Contract #2 (the crux improvement) AND #3: an inconsistent observed clique
    # (topics 0,1,2 all observed) PLUS a free entry (topic 3 tied to topic 0 only,
    # with 3~1 and 3~2 free). The free degrees of freedom let the Dykstra
    # min-Frobenius routine achieve a STRICTLY SMALLER observed-entry deviation than a
    # single nearest_spd floor of the same target — the whole point, not merely "is
    # PD". This is also the rare-correlated-clique path with zero prior coverage.
    target = np.array([[1.0, 0.99, 0.99, 0.5],
                       [0.99, 1.0, -0.99, 0.0],
                       [0.99, -0.99, 1.0, 0.0],
                       [0.5, 0.0, 0.0, 1.0]])
    mask = np.array([[1, 1, 1, 1],
                     [1, 1, 1, 0],
                     [1, 1, 1, 0],
                     [1, 0, 0, 1]], bool)
    assert not mask.all()                       # there IS a free entry

    dyk = min_frobenius_psd_completion(target, mask)
    floored = nearest_spd(target)

    assert _is_pd(dyk)                                   # strictly PD
    np.testing.assert_allclose(dyk, dyk.T, atol=1e-10)   # symmetric
    # Strictly smaller observed deviation — min-Frobenius beats the single floor.
    assert _observed_dev(dyk, target, mask) < _observed_dev(floored, target, mask)


def test_non_pd_completable_with_free_entry_routes_to_fallback():
    # Contract #3: the same inconsistent-clique-plus-free pattern through the full
    # pd_complete entry point. free_pairs is non-empty, so the fallback engages on the
    # rare-correlated-clique path (the up-front non-completability detector routes it;
    # the in-loop check is a further numerical safety net). Result must be strictly PD,
    # symmetric, and not raise.
    target = np.array([[1.0, 0.99, 0.99, 0.5],
                       [0.99, 1.0, -0.99, 0.0],
                       [0.99, -0.99, 1.0, 0.0],
                       [0.5, 0.0, 0.0, 1.0]])
    mask = np.array([[1, 1, 1, 1],
                     [1, 1, 1, 0],
                     [1, 1, 1, 0],
                     [1, 0, 0, 1]], bool)
    assert not mask.all()                       # there IS a free entry
    out = pd_complete(target, mask)             # must not raise
    assert _is_pd(out)                          # strictly PD
    np.testing.assert_allclose(out, out.T, atol=1e-10)


def test_min_frobenius_returns_observed_exactly_when_completable():
    # When the observed block already admits a PSD completion, the fallback must
    # converge into the intersection: observed entries reproduced (within tol) and
    # the result PD.
    rng = np.random.default_rng(4)
    A = rng.standard_normal((4, 4))
    target = A @ A.T + np.eye(4)                 # SPD -> trivially completable
    mask = np.ones((4, 4), bool)
    mask[0, 3] = mask[3, 0] = False
    out = min_frobenius_psd_completion(target, mask)
    assert _is_pd(out)
    np.testing.assert_allclose(out[mask], target[mask], atol=1e-6)
