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

def min_frobenius_psd_completion(
    target: np.ndarray,
    observed_mask: np.ndarray,
    *,
    eps: float = 1e-8,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> np.ndarray:
    """Minimum-Frobenius strictly-PD matrix preserving observed entries as closely
    as possible — the robust fallback for an observed pattern that admits no PD
    completion.

    This is Dykstra's alternating projection (Dykstra 1983, "An Algorithm for
    Restricted Least Squares Regression", JASA 78:837-842) onto two convex sets of
    symmetric K×K matrices, the construction Higham 2002 ("Computing the Nearest
    Correlation Matrix — a problem from finance", IMA J. Numer. Anal. 22:329-343)
    uses for the nearest correlation matrix, here generalized from Higham's
    unit-diagonal constraint to an arbitrary observed-entry pattern:

      - S_obs = {M symmetric : M_ij = target_ij for every observed (i,j)} — an affine
        set. Projection P_obs(X): symmetrize, reset observed entries to target, leave
        free entries unchanged.
      - S_psd(eps) = {M symmetric : M >= eps·I}. Projection P_psd(X): eigendecompose
        the symmetric X (np.linalg.eigh), clamp eigenvalues to max(lambda, eps),
        rebuild (V * w) @ V.T.

    eps > 0 is a strict-positive-definite safeguard (so the result is invertible for
    downstream precision use) — a HEURISTIC, NOT from the literature; it matches
    nearest_spd's existing floor default 1e-8. Higham 2002 itself uses eps = 0 for
    the PSD cone; the eps > 0 floor is this codebase's deviation.

    Dykstra (not naive POCS): each set keeps a CORRECTION increment that is added
    back before its projection, so the iteration converges to the true closest point
    of the intersection (when non-empty) or to the minimum-Frobenius compromise (when
    empty), rather than cycling near the boundary. When the observed block IS
    PD-completable the iterate enters the intersection (observed exact AND PSD); when
    it is NOT, it settles at the min-Frobenius compromise whose observed-entry drift
    is far below a single eigenvalue floor's.

    Returns the final P_psd projection so the result is always strictly PD (and hence
    invertible). tol / max_iter are convergence controls (heuristic stopping rule):
    iterate until the max absolute change drops below tol.
    """
    mask = np.asarray(observed_mask, dtype=bool)
    mask = mask | mask.T
    tgt = 0.5 * (target + target.T)

    def P_psd(X: np.ndarray) -> np.ndarray:
        w, V = np.linalg.eigh(0.5 * (X + X.T))
        w = np.maximum(w, eps)
        return (V * w) @ V.T

    def P_obs(X: np.ndarray) -> np.ndarray:
        Y = 0.5 * (X + X.T)
        Y[mask] = tgt[mask]
        return Y

    X = P_psd(tgt)                       # start feasible-ish (PSD)
    dA = np.zeros_like(X)                 # Dykstra increment for S_psd
    dB = np.zeros_like(X)                 # Dykstra increment for S_obs
    for _ in range(max_iter):
        Y = P_psd(X + dA)
        dA = (X + dA) - Y
        X_new = P_obs(Y + dB)
        dB = (Y + dB) - X_new
        if np.max(np.abs(X_new - X)) < tol:
            X = X_new
            break
        X = X_new
    return P_psd(X)                       # return a strictly-PD iterate


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
    precision entry to zero while preserving the measured covariances exactly. When the
    observed part admits no PD completion (the IPS sweep loses positive-definiteness, or
    the all-observed init is already indefinite), the routine switches to the Dykstra
    min-Frobenius alternating-projection fallback (`min_frobenius_psd_completion`; Higham
    2002, "Computing the Nearest Correlation Matrix", IMA J. Numer. Anal. 22:329-343;
    Dykstra 1983, JASA 78:837-842) and returns its result: the strictly-PD matrix whose
    observed-entry deviation from `target` is minimized — a far smaller drift than a single
    eigenvalue floor of the indefinite assembly.

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
    # measured diagonal kept exact.
    Sigma = np.where(mask, target, 0.0)
    np.fill_diagonal(Sigma, np.diag(target))

    # Non-PD-completability detector. When the observed entries admit a PD completion,
    # this zero-on-free init is itself PD (a sufficient seed for IPS: zeroing the free
    # off-diagonals does not break a genuinely completable observed block). When the
    # observed entries are mutually inconsistent (no PD matrix matches them all), the
    # init is indefinite — IPS cannot recover the observed entries exactly, and flooring
    # the init (the former behaviour) just returns a single nearest_spd projection that
    # drifts the observed entries badly. Route those straight to the Dykstra
    # min-Frobenius fallback, which preserves the observed entries as closely as a
    # strictly-PD result allows. This covers both the all-observed inconsistent case
    # (no free entries) and the inconsistent-clique-plus-free case.
    try:
        np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        return min_frobenius_psd_completion(target, mask, tol=tol, max_iter=max_iter)

    # All-observed and PD-completable: nothing to complete, return the symmetrized target.
    if not free_pairs:
        return 0.5 * (Sigma + Sigma.T)

    for _ in range(max_iter):
        Sigma_prev = Sigma.copy()
        # Coordinate ascent: set each free entry to the value that zeroes its precision
        # (regression of i,j on the remaining topics) -> conditional independence.
        for i, j in free_pairs:
            rest = [k for k in range(n) if k != i and k != j]
            x = Sigma[i, rest] @ np.linalg.inv(Sigma[np.ix_(rest, rest)]) @ Sigma[rest, j]
            Sigma[i, j] = Sigma[j, i] = x
        # Numerical safety net: the up-front detector already routes inconsistent
        # observed blocks to the fallback, but if a free-entry regression update should
        # ever drive an otherwise-PD iterate indefinite, hand to the same Dykstra
        # min-Frobenius routine rather than flooring once and continuing.
        try:
            np.linalg.cholesky(Sigma)
        except np.linalg.LinAlgError:
            return min_frobenius_psd_completion(target, mask, tol=tol, max_iter=max_iter)
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
