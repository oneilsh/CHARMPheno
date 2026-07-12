"""Tests for the MCMC convergence diagnostic (rank-normalized split-R-hat).

Reference: Vehtari, Gelman, Simpson, Carpenter, Bürkner (2021), "Rank-
normalization, folding, and localization: An improved R-hat for assessing
convergence of MCMC", Bayesian Analysis 16(2):667-718. The improved R-hat is
max(rank-normalized split-R-hat, rank-normalized folded split-R-hat); the
folded term is what catches scale (variance) non-stationarity, which is the
failure mode the free-Gibbs-Sigma probe is looking for on scarce-topic
variances.
"""
import numpy as np
import pytest

from spark_vi.models.topic._mcmc_diag import improved_rhat, rank_normalized_rhat


def test_well_mixed_iid_chains_rhat_near_one():
    # Four independent chains from the SAME stationary distribution -> ~1.0.
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 2000))
    assert improved_rhat(x) < 1.01


def test_distinct_chain_means_rhat_large():
    # Chains stuck at different locations (non-stationary across chains) -> >>1.
    rng = np.random.default_rng(1)
    x = rng.standard_normal((4, 2000)) + np.array([[0.0], [10.0], [20.0], [30.0]])
    assert improved_rhat(x) > 2.0


def test_folded_catches_scale_drift_bulk_misses_it():
    # Same mean (0) in every chain, but wildly different SCALES. The bulk
    # (location) R-hat is fooled; the folded term must flag it.
    rng = np.random.default_rng(2)
    scales = np.array([[0.1], [1.0], [5.0], [25.0]])
    x = rng.standard_normal((4, 2000)) * scales
    assert rank_normalized_rhat(x) < 1.3            # location looks fine
    assert improved_rhat(x) > 1.5                   # folded catches the scale drift


def test_random_walk_multichain_is_flagged_nonstationary():
    # Four independent random walks (the realistic multi-chain use) diverge and
    # must not read as converged. (A single walk split in two carries little
    # signal -- each half has high within-variance -- so R-hat correctly needs
    # multiple chains; this mirrors how the free-Gibbs probe runs 4 chains.)
    rng = np.random.default_rng(3)
    walks = np.stack([np.cumsum(rng.standard_normal(2000)) for _ in range(4)])
    assert improved_rhat(walks) > 1.5


def test_too_few_draws_raises():
    with pytest.raises(ValueError):
        improved_rhat(np.zeros((4, 3)))
