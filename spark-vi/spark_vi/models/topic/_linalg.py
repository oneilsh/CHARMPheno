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
    info: dict | None = None,
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
    precision entry to zero while preserving the measured covariances exactly.

    Seeding and re-pinning. The IPS regression update needs a PD iterate so the inner
    inverse is well posed, but the natural zero-on-free init can be INDEFINITE even for a
    genuinely PD-completable observed block: a strongly-correlated pattern such as the
    gated background-coupled "block-arrow" (one background topic strongly tied to each
    foreground group, cross-group pairs free) is PD-completable to a well-conditioned
    matrix yet has an indefinite zero-fill. So the iterate is seeded with the nearest_spd
    projection of that zero-fill (PD, so the regression inverse is well posed), and after
    every sweep the observed entries are re-pinned to `target` and the diagonal restored.
    The re-pin is what lets IPS converge to the exact maximum-determinant completion from
    an indefinite-but-completable seed, independent of the seed projection.

    Completability is decided AFTER convergence, not from the init. A GENUINELY
    non-PD-completable observed set cannot be matched by any PD matrix, so the converged
    iterate drifts off the observed targets (the PD floor moves them) or fails to be PD;
    only that post-convergence observed-mismatch / non-PD check routes to the Dykstra
    min-Frobenius alternating-projection fallback (`min_frobenius_psd_completion`; Higham
    2002, "Computing the Nearest Correlation Matrix", IMA J. Numer. Anal. 22:329-343;
    Dykstra 1983, JASA 78:837-842), which returns the strictly-PD matrix whose
    observed-entry deviation from `target` is minimized — a far smaller drift than a single
    eigenvalue floor of the indefinite assembly. An indefinite zero-fill init alone does
    NOT trigger the fallback.

    On a decomposable (chordal) observed pattern this converges in one sweep to the
    closed form Sigma[A,B] = Sigma[A,S] inv(Sigma[S,S]) Sigma[S,B] over each separator S
    (Grone, Johnson, Sa & Wolkowicz 1984, "Positive Definite Completions of Partial
    Hermitian Matrices", Linear Algebra Appl. 58:109-124; Lauritzen 1996, "Graphical
    Models", Oxford, Sec. 5.3) — used as a test oracle, not a separate code path.

    The caller guarantees a true diagonal in `observed_mask`; the mask is symmetrized
    defensively (`mask | mask.T`) and the diagonal is always kept observed.

    `tol` / `max_iter` are convergence controls (heuristic stopping rule, not from the
    literature): sweep until the max absolute change in Sigma drops below `tol`.

    `info`, if given, is filled with diagnostic counters and does NOT affect the result:
    `n_free` (free off-diagonal pairs completed), `sweeps` (IPS sweeps actually run,
    <= `max_iter`), `max_iter`, and `fell_back` (True iff the Dykstra min-Frobenius
    fallback was used because the observed block admits no PD completion).
    """
    mask = np.asarray(observed_mask, dtype=bool)
    mask = mask | mask.T                      # symmetrize defensively
    n = mask.shape[0]
    free = ~mask                              # entries to be completed (off-diagonal)
    free_pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if free[i, j]]

    # PD seed: measured values on observed entries, 0 on free off-diagonals, measured
    # diagonal kept exact, then nearest_spd. The zero-fill can be indefinite even for a
    # genuinely completable observed block (e.g. the gated block-arrow); projecting it to
    # the nearest SPD matrix makes the per-pair regression inverse below well posed
    # regardless. The post-sweep re-pin recovers the exact max-det completion from this
    # seed for any completable pattern.
    Sigma = np.where(mask, target, 0.0)
    np.fill_diagonal(Sigma, np.diag(target))
    Sigma = nearest_spd(Sigma).copy()
    if info is not None:
        info["n_free"] = len(free_pairs)
        info["max_iter"] = max_iter

    sweeps = 0
    for _ in range(max_iter):
        sweeps += 1
        Sigma_prev = Sigma.copy()
        # Coordinate ascent: set each free entry to the value that zeroes its precision
        # (regression of i,j on the remaining topics) -> conditional independence.
        for i, j in free_pairs:
            rest = [k for k in range(n) if k != i and k != j]
            x = Sigma[i, rest] @ np.linalg.inv(Sigma[np.ix_(rest, rest)]) @ Sigma[rest, j]
            Sigma[i, j] = Sigma[j, i] = x
        # Re-pin the observed entries (and diagonal) to target each sweep. From an
        # indefinite-but-completable seed, IPS then converges to the exact max-det
        # completion regardless of the seed projection. If a sweep loses PD, floor it so
        # the next regression inverse stays well posed.
        Sigma[mask] = target[mask]
        np.fill_diagonal(Sigma, np.diag(target))
        try:
            np.linalg.cholesky(Sigma)
        except np.linalg.LinAlgError:
            Sigma = nearest_spd(Sigma)
        if np.max(np.abs(Sigma - Sigma_prev)) < tol:
            break

    if info is not None:
        info["sweeps"] = sweeps
    Sigma = 0.5 * (Sigma + Sigma.T)           # symmetrize the output

    # Completability check, decided AFTER convergence (not from the init). A genuinely
    # non-PD-completable observed set cannot be matched by any PD matrix, so the converged
    # result drifts off the observed targets (the PD floor moves them) or fails to be PD.
    # Only then route to the Dykstra min-Frobenius fallback. An indefinite zero-fill alone
    # does NOT land here — a completable block re-pins to its observed targets exactly.
    observed_ok = np.max(np.abs(Sigma[mask] - target[mask])) < 1e-7
    is_pd = True
    try:
        np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        is_pd = False
    if info is not None:
        info["fell_back"] = not (observed_ok and is_pd)
    if observed_ok and is_pd:
        return Sigma
    return min_frobenius_psd_completion(target, mask, tol=tol, max_iter=max_iter)


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
