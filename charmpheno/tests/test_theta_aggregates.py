from __future__ import annotations

import numpy as np
import pytest

from charmpheno.export.theta_aggregates import compute_theta_aggregates


def _uniform_gamma(n: int, k: int, seed: int = 42) -> np.ndarray:
    """Return a random γ matrix (all positive) with a fixed seed."""
    rng = np.random.default_rng(seed)
    return rng.exponential(scale=1.0, size=(n, k))


def test_compute_theta_aggregates_shapes():
    """Returned dict has the correct shapes for N=200, K=10."""
    gamma = _uniform_gamma(200, 10)
    result = compute_theta_aggregates(gamma)

    assert result["n_patients"] == 200

    hist = result["theta_histogram"]
    assert len(hist) == 10
    for row in hist:
        assert len(row) == 50

    pcts = result["theta_percentiles"]
    assert len(pcts) == 10
    for entry in pcts:
        assert set(entry.keys()) == {"p5", "p25", "p50", "p75", "p95"}

    prev = result["corpus_prevalence"]
    assert len(prev) == 10


def test_compute_theta_aggregates_small_cell_suppression():
    """Bins with count in [1, min_count) → None; count 0 → 0.0; count ≥ min_count → float."""
    # Build γ with N=100 patients, K=3 topics.
    # We control θ_0 (topic index 0) explicitly:
    #   19 patients → θ_0 = 0.35  (bin index 17 with n_bins=50: [0.34, 0.36))
    #   25 patients → θ_0 = 0.85  (bin index 42: [0.84, 0.86))
    #   56 patients → θ_0 = 0.01  (bin index 0: [0.0, 0.02))
    # For other topics the exact distribution doesn't matter.
    #
    # We construct γ so that γ / γ.sum(axis=1, keepdims=True) gives the exact θ_0
    # values we want. Strategy: for each patient, set γ[i, 0] = θ_desired and
    # γ[i, 1] = γ[i, 2] = (1 - θ_desired) / 2.  Row sums to 1 so θ_0 = γ[i,0].
    n = 100
    k = 3
    gamma = np.zeros((n, k))
    theta_0_target = np.empty(n)
    theta_0_target[:19] = 0.35   # 19 patients → suppressed bin
    theta_0_target[19:44] = 0.85  # 25 patients → non-suppressed bin
    theta_0_target[44:] = 0.01   # 56 patients → non-suppressed (bin 0)

    gamma[:, 0] = theta_0_target
    gamma[:, 1] = (1 - theta_0_target) / 2
    gamma[:, 2] = (1 - theta_0_target) / 2

    result = compute_theta_aggregates(gamma, n_bins=50, min_count=20)
    hist_0 = result["theta_histogram"][0]

    # Determine which bin indices our target values land in.
    # n_bins=50 → bin width = 0.02; bin_index = floor(θ / 0.02)
    # 0.35 / 0.02 = 17.5 → bin 17
    # 0.85 / 0.02 = 42.5 → bin 42
    # 0.01 / 0.02 = 0.5  → bin 0
    bin_width = 1.0 / 50
    bin_19 = int(0.35 / bin_width)   # 17
    bin_25 = int(0.85 / bin_width)   # 42
    bin_56 = int(0.01 / bin_width)   # 0

    # 19 patients in bin_19 → suppressed (None)
    assert hist_0[bin_19] is None, f"Expected None for bin {bin_19}, got {hist_0[bin_19]}"

    # 25 patients in bin_25 → fraction 25/100 = 0.25
    assert isinstance(hist_0[bin_25], float)
    assert hist_0[bin_25] == pytest.approx(25 / 100)

    # 56 patients in bin_56 → fraction 56/100 = 0.56
    assert isinstance(hist_0[bin_56], float)
    assert hist_0[bin_56] == pytest.approx(56 / 100)

    # All bins that are not bin_19, bin_25, bin_56 should be 0.0
    for i, val in enumerate(hist_0):
        if i not in {bin_19, bin_25, bin_56}:
            assert val == 0.0, f"Expected 0.0 for empty bin {i}, got {val}"


def test_compute_theta_aggregates_zero_bin_not_suppressed():
    """Empty bins (count == 0) are 0.0, not None — privacy suppression only applies to nonzero small counts."""
    # Concentrate all patients at θ_0 = 0.5 so most bins are empty.
    n = 50
    k = 2
    gamma = np.ones((n, k))
    gamma[:, 0] = 1.0
    gamma[:, 1] = 1.0
    # All patients have θ_0 = 0.5 exactly → one bin has all n patients, rest are 0.

    result = compute_theta_aggregates(gamma, n_bins=50, min_count=20)
    hist_0 = result["theta_histogram"][0]

    zero_bins = [val for val in hist_0 if val == 0.0]
    none_bins = [val for val in hist_0 if val is None]

    # All empty bins must be 0.0, none of them should become None
    assert len(none_bins) == 0, f"Found {len(none_bins)} suppressed bins but all empties should be 0.0"
    # The non-empty bin has n=50 ≥ min_count=20, so it's a float too
    nonzero_bins = [val for val in hist_0 if val is not None and val != 0.0]
    assert len(nonzero_bins) == 1
    assert len(zero_bins) == 49


def test_compute_theta_aggregates_percentile_monotonic():
    """p5 ≤ p25 ≤ p50 ≤ p75 ≤ p95 for every topic."""
    rng = np.random.default_rng(7)
    gamma = rng.exponential(scale=1.0, size=(300, 8))
    result = compute_theta_aggregates(gamma)

    for entry in result["theta_percentiles"]:
        assert entry["p5"] <= entry["p25"]
        assert entry["p25"] <= entry["p50"]
        assert entry["p50"] <= entry["p75"]
        assert entry["p75"] <= entry["p95"]


def test_compute_theta_aggregates_mean_matches_full_precision():
    """corpus_prevalence equals theta.mean(axis=0) to float64 precision, NOT derived from bins."""
    rng = np.random.default_rng(99)
    gamma = rng.exponential(scale=1.0, size=(250, 12))
    theta = gamma / gamma.sum(axis=1, keepdims=True)
    expected_mean = theta.mean(axis=0).tolist()

    result = compute_theta_aggregates(gamma)
    got = result["corpus_prevalence"]

    for k_idx, (exp, got_val) in enumerate(zip(expected_mean, got)):
        assert got_val == pytest.approx(exp, rel=1e-12), (
            f"Topic {k_idx}: expected {exp}, got {got_val}"
        )


def test_compute_theta_aggregates_theta_eq_one_lands_in_last_bin():
    """Patients with θ ≈ 1.0 land in the last bin, not lost to over-the-edge binning."""
    n = 50
    k = 3
    gamma = np.ones((n, k)) * 1e-6  # tiny baseline
    # Make the first 10 patients have θ_1 ≈ 1.0 by giving topic 1 a huge γ
    gamma[:10, 1] = 1e10

    result = compute_theta_aggregates(gamma, n_bins=50, min_count=5)
    hist_1 = result["theta_histogram"][1]

    # The last bin (index 49) should contain the 10 near-unity patients
    last_bin = hist_1[-1]
    assert last_bin is not None, "Last bin is None (suppressed), expected a float ≥ 0"
    assert last_bin > 0.0, f"Last bin is 0.0 — θ≈1 patients were lost to over-the-edge binning"
    assert last_bin == pytest.approx(10 / n)
