"""Pólya-Gamma variational core for the gated stick-breaking logistic-normal
topic model (design 2026-07-11-pg-stm-inference-core-design.md). Single machine,
full-batch VI + exact Gibbs cross-check. References: Polson/Scott/Windle 2013 (PG);
Linderman/Johnson/Adams 2015 (stick-breaking multinomial + PG); Blei/Lafferty 2007
(logistic-normal topic model)."""
from __future__ import annotations

import numpy as np
from scipy.special import expit  # logistic sigmoid


def stick_to_simplex(psi: np.ndarray) -> np.ndarray:
    """Stick-breaking map: psi (K-1,) -> theta (K,) on the simplex.
    theta_k = sigma(psi_k) * prod_{j<k}(1 - sigma(psi_j)); last topic gets the remainder."""
    psi = np.asarray(psi, dtype=np.float64)
    sig = expit(psi)                          # (K-1,)
    theta = np.empty(psi.shape[0] + 1, dtype=np.float64)
    remaining = 1.0
    for k in range(psi.shape[0]):
        theta[k] = remaining * sig[k]
        remaining *= (1.0 - sig[k])
    theta[-1] = remaining
    return theta


def simplex_to_stick(theta: np.ndarray) -> np.ndarray:
    """Inverse map: theta (K,) -> psi (K-1,). sigma(psi_k) = theta_k / (1 - sum_{j<k} theta_j)."""
    theta = np.asarray(theta, dtype=np.float64)
    psi = np.empty(theta.shape[0] - 1, dtype=np.float64)
    remaining = 1.0
    for k in range(theta.shape[0] - 1):
        frac = np.clip(theta[k] / remaining, 1e-15, 1.0 - 1e-15)
        psi[k] = np.log(frac) - np.log1p(-frac)   # logit(frac)
        remaining -= theta[k]
    return psi


def stick_trials(n: np.ndarray) -> np.ndarray:
    """Per-stick trials-at-risk b (K-1,): b[k] = sum_{j>=k} n[j]."""
    n = np.asarray(n, dtype=np.float64)
    return np.cumsum(n[::-1])[::-1][:-1].copy()
