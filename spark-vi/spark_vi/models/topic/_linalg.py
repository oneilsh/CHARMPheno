"""Pure SPD / correlation helpers for the full-covariance STM (no Spark deps).

safe_inverse mirrors the per-doc Hessian repair (_spd_inverse in stm.py) but for
Sigma. nearest_spd projects an assembled (possibly indefinite) covariance to the
nearest SPD matrix — needed because the gated per-pair M-step stitches Sigma from
inconsistent doc subsets and can break positive-definiteness (design spec C2).
"""
from __future__ import annotations
import numpy as np

def safe_inverse(M: np.ndarray, cond_cap: float = 1e-10) -> np.ndarray:
    """Inverse of a matrix meant to be SPD; eigenvalue-floor repair if not PD."""
    try:
        np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(0.5 * (M + M.T))
        floor = max(w.max() * cond_cap, 1e-12)
        w = np.maximum(w, floor)
        return (V * (1.0 / w)) @ V.T
    return np.linalg.inv(M)

def nearest_spd(M: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    """Symmetrize and floor eigenvalues at `floor`. Identity (within fp) on SPD
    inputs whose eigenvalues already exceed the floor."""
    S = 0.5 * (M + M.T)
    w, V = np.linalg.eigh(S)
    if np.min(w) >= floor:
        return S
    w = np.maximum(w, floor)
    return (V * w) @ V.T

def topic_correlation(Sigma: np.ndarray) -> np.ndarray:
    """Correlation matrix R_ij = Sigma_ij / sqrt(Sigma_ii Sigma_jj); unit diagonal.

    Blei & Lafferty 2007 logistic-normal correlation (eq. 4).
    """
    d = np.sqrt(np.clip(np.diag(Sigma), 1e-300, None))
    R = Sigma / np.outer(d, d)
    np.fill_diagonal(R, 1.0)
    return R
