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

def pd_complete(
    target: np.ndarray,
    observed_mask: np.ndarray,
    *,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> np.ndarray:
    """Maximum-determinant positive-definite completion (covariance selection).

    Given measured covariance entries on `observed_mask` (True = measured), fill the
    free entries so the result is SPD and has zero PRECISION (Sigma-inverse) on every
    free entry. That zero-precision completion is the unique maximum-determinant /
    maximum-entropy PD completion: it is the only completion that imposes conditional
    independence (no spurious partial correlation) where nothing was measured. This is
    the covariance-selection model of Dempster 1972 ("Covariance Selection",
    Biometrics 28:157-175). It replaces the transitively inconsistent "zero the
    free COVARIANCE" pin, which sets unobserved cross-group entries to 0 directly and
    yields a near-singular, inconsistent Sigma.

    Method: iterative proportional scaling (Speed & Kiiveri 1986,
    "Gaussian Markov Distributions over Finite Graphs", Ann. Statist. 14:138-150; the
    IPS / IPF algorithm of Dempster 1972), realized as coordinate ascent over the free
    off-diagonal entries. For each free entry (i,j) the unique value that zeroes the
    precision there, holding the rest fixed, is the regression closed form
    Sigma[i,j] = Sigma[i,R] inv(Sigma[R,R]) Sigma[R,j] over R = all other topics: this is
    the partial-covariance-to-zero update, and it leaves the observed entries (never
    visited) untouched. Sweeping all free entries to convergence drives every free
    precision entry to zero while preserving the measured covariances exactly. A Higham
    2002 PSD projection (`nearest_spd`, "Computing the Nearest Correlation Matrix",
    IMA J. Numer. Anal. 22:329-343) is applied if a sweep loses positive-definiteness,
    providing robustness when the observed part admits no PD completion (then the result
    is an approximate, still SPD, completion rather than the exact max-det one).

    On a decomposable (chordal) observed pattern this converges in one sweep to the
    closed form Sigma[A,B] = Sigma[A,S] inv(Sigma[S,S]) Sigma[S,B] over each separator S
    (Grone, Johnson, Sa & Wolkowicz 1984, "Positive Definite Completions of Partial
    Hermitian Matrices", Linear Algebra Appl. 58:109-124; Lauritzen 1996, "Graphical
    Models", Oxford, Sec. 5.3) — used as a test oracle, not a separate code path.

    The caller guarantees a true diagonal in `observed_mask`; the mask is symmetrized
    defensively (`mask | mask.T`) and the diagonal is always kept observed.

    `tol` / `max_iter` are convergence controls (heuristic stopping rule, not from the
    literature): sweep until the max absolute change in Sigma drops below `tol`.
    """
    mask = np.asarray(observed_mask, dtype=bool)
    mask = mask | mask.T                      # symmetrize defensively
    n = mask.shape[0]
    free = ~mask                              # entries to be completed (off-diagonal)
    free_pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if free[i, j]]

    # Init: measured values on observed entries, 0 on free off-diagonal entries, and the
    # measured diagonal kept exact. One PSD projection so the first inv() is well posed.
    Sigma = np.where(mask, target, 0.0)
    np.fill_diagonal(Sigma, np.diag(target))
    Sigma = nearest_spd(Sigma).copy()

    for _ in range(max_iter):
        Sigma_prev = Sigma.copy()
        # Coordinate ascent: set each free entry to the value that zeroes its precision
        # (regression of i,j on the remaining topics) -> conditional independence.
        for i, j in free_pairs:
            rest = [k for k in range(n) if k != i and k != j]
            x = Sigma[i, rest] @ np.linalg.inv(Sigma[np.ix_(rest, rest)]) @ Sigma[rest, j]
            Sigma[i, j] = Sigma[j, i] = x
        # Higham fallback if the observed part is not PD-completable.
        try:
            np.linalg.cholesky(Sigma)
        except np.linalg.LinAlgError:
            Sigma = nearest_spd(Sigma).copy()
        if np.max(np.abs(Sigma - Sigma_prev)) < tol:
            break

    return 0.5 * (Sigma + Sigma.T)            # symmetrize the output


def topic_correlation(Sigma: np.ndarray) -> np.ndarray:
    """Correlation matrix R_ij = Sigma_ij / sqrt(Sigma_ii Sigma_jj); unit diagonal.

    Blei & Lafferty 2007 logistic-normal correlation (eq. 4).
    """
    d = np.sqrt(np.clip(np.diag(Sigma), 1e-300, None))
    R = Sigma / np.outer(d, d)
    np.fill_diagonal(R, 1.0)
    return R
