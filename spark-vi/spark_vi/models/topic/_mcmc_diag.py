"""Rank-normalized split-R-hat, a validated MCMC convergence diagnostic.

Reference: Vehtari, Gelman, Simpson, Carpenter, Bürkner (2021),
"Rank-normalization, folding, and localization: An improved R-hat for
assessing convergence of MCMC", Bayesian Analysis 16(2):667-718.

The "improved" R-hat is ``max(rank-normalized split-R-hat,
rank-normalized folded split-R-hat)``:

  * rank-normalization (pool all draws -> fractional ranks -> Blom
    inverse-normal transform) makes R-hat robust to heavy tails / non-normal
    marginals, so it is well defined for the variance parameters this project
    samples (log sigma^2_k) without assuming normality;
  * split (halve each chain) catches within-chain non-stationarity a single
    trend would otherwise average away;
  * folding about the median (|x - median|) turns a SCALE non-stationarity into
    a LOCATION one, so the folded term flags a chain whose *variance* wanders
    even when its mean is stable — the exact failure mode of a weakly-identified
    scarce-topic variance under a near-flat prior.

Pure numpy; no external MCMC library dependency (arviz is not on the cluster
image). Input convention: a 2-D array ``(n_chains, n_draws)`` for ONE scalar
parameter, or a 1-D ``(n_draws,)`` single chain (split into two half-chains).
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtri


def _as_chains(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError(f"expected (n_chains, n_draws) or (n_draws,), got shape {x.shape}")
    return x


def _split_chains(x: np.ndarray) -> np.ndarray:
    """Halve each chain: (m, n) -> (2m, n//2), dropping a trailing odd draw."""
    m, n = x.shape
    half = n // 2
    if half < 2:
        raise ValueError(
            f"need >= 4 draws per chain after splitting (each half >= 2); got n_draws={n}")
    first, second = x[:, :half], x[:, half:2 * half]
    return np.concatenate([first, second], axis=0)


def _rank_normalize(x: np.ndarray) -> np.ndarray:
    """Pooled fractional ranks -> Blom (1958) inverse-normal transform.

    Ranks are computed over ALL draws in ALL (split) chains jointly, average
    ranks for ties, then z = Phi^-1((r - 3/8) / (N - 1/4)).
    """
    flat = x.ravel()
    order = np.argsort(flat, kind="stable")
    ranks = np.empty(flat.size, dtype=np.float64)
    ranks[order] = np.arange(1, flat.size + 1, dtype=np.float64)
    # average ranks within tie groups
    _, inv, counts = np.unique(flat, return_inverse=True, return_counts=True)
    csum = np.concatenate([[0.0], np.cumsum(counts)])
    avg = (csum[inv] + csum[inv + 1] + 1.0) / 2.0
    ranks = avg
    n = flat.size
    z = ndtri((ranks - 3.0 / 8.0) / (n - 0.25))
    return z.reshape(x.shape)


def _classic_rhat(split: np.ndarray) -> float:
    """Gelman-Rubin potential scale reduction on already-split chains (m, n)."""
    m, n = split.shape
    chain_means = split.mean(axis=1)
    chain_vars = split.var(axis=1, ddof=1)
    W = chain_vars.mean()
    if W <= 0:
        return 1.0
    B = n * chain_means.var(ddof=1)
    var_plus = (n - 1) / n * W + B / n
    return float(np.sqrt(var_plus / W))


def rank_normalized_rhat(x: np.ndarray) -> float:
    """Rank-normalized split-R-hat (the "bulk" term; location-sensitive)."""
    split = _split_chains(_as_chains(x))
    return _classic_rhat(_rank_normalize(split))


def folded_rhat(x: np.ndarray) -> float:
    """Rank-normalized folded split-R-hat (scale/variance-sensitive term)."""
    xc = _as_chains(x)
    folded = np.abs(xc - np.median(xc))
    split = _split_chains(folded)
    return _classic_rhat(_rank_normalize(split))


def improved_rhat(x: np.ndarray) -> float:
    """Vehtari et al. (2021) improved R-hat = max(bulk, folded).

    Values at or just above 1 indicate a stationary, well-mixed chain (a WIDE
    posterior is fine — R-hat measures mixing, not width); values meaningfully
    above 1 (the paper's ship threshold is 1.01) indicate the chain has not
    converged along that parameter.
    """
    return max(rank_normalized_rhat(x), folded_rhat(x))
