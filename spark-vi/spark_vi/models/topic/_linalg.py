"""Pure SPD / correlation helpers for the gated STM (no Spark deps).

safe_inverse mirrors the per-doc Hessian repair (_spd_inverse in stm.py) but for
Sigma. nearest_spd projects an assembled (possibly indefinite) covariance to the
nearest SPD matrix — needed because the gated per-pair M-step stitches Sigma from
inconsistent doc subsets and can break positive-definiteness (design spec C2).

Historical note: a max-determinant PD *completion* (pd_complete /
min_frobenius_psd_completion, Dempster 1972 covariance selection) used to fill
unobserved cross-group Sigma cells. It was retired when the block-wise
unit-diagonal correlation M-step (ADR 0034) made completion unnecessary — under
single-label gating the E-step only inverts fully-observed marginal sub-blocks,
so nothing needs completing (see insight 0032 for the journey). The completion
code and its tests were removed 2026-07-09.
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

def topic_correlation_identified(Sigma, n_pairs, min_pair_support):
    """Logistic-normal correlation R (topic_correlation) with an identified mask.

    A cell (i,j) is identified iff n_pairs[i,j] >= min_pair_support — the same
    document-support floor the M-step uses to decide estimated-vs-completed
    (stm.py). Unidentified OFF-diagonal cells are set to NaN in R (no joint data
    supports that correlation); the diagonal is always identified (unit value).
    Domain-agnostic: topic indices only.

    Identifiability by document support: pairs with fewer than min_pair_support
    co-activations lack sufficient joint data to reliably estimate correlation
    (Blei & Lafferty 2007 for the correlation formula; masking unidentified
    entries is a domain-agnostic heuristic this engine applies).

    Returns (R, identified): R is (K,K) float with NaN on unidentified off-diag
    cells; identified is (K,K) bool.
    """
    R = topic_correlation(Sigma)
    identified = np.asarray(n_pairs) >= float(min_pair_support)
    identified = identified | identified.T          # symmetric support
    np.fill_diagonal(identified, True)              # diagonal always identified
    mask_na = ~identified
    np.fill_diagonal(mask_na, False)                # never NaN the unit diagonal
    R = R.copy()
    R[mask_na] = np.nan
    return R, identified
