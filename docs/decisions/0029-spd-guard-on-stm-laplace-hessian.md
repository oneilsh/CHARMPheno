# 0029 — SPD guard on the STM per-document Laplace Hessian

**Status:** Accepted
**Date:** 2026-06-25

## Context

STM's per-document inference (ADR 0023) is a two-step Laplace approximation:
L-BFGS finds the MAP point η̂, then the Laplace covariance is ν = H⁻¹, the
inverse of the neg-log-joint Hessian at η̂. The objective is **not globally
convex** — its data term is a difference of two log-sum-exp functions
(log-sum-exp(η + log β) minus log-sum-exp(η)) — so H is only guaranteed
positive-semidefinite *at a true mode*. With a weak prior (large Σ, whose Σ⁻¹
term is the only thing stabilizing H) or when L-BFGS stops short of the mode
(`lbfgs_max_iter` / `lbfgs_tol`), H can be indefinite. The prior code called
`np.linalg.inv(H)` unconditionally and read `slogdet(ν)` without checking its
sign, so a non-PD H would silently produce a covariance with negative
variances — corrupting `residual_diag_stat` (it adds diag(ν)) and the Gaussian
KL (its log-determinant). Surfaced during the `stm`-branch pre-merge review;
the reference stm R package guards the same step with a Hessian "nugget."

## Decision

Invert through a `_spd_inverse(H)` helper. Fast path: attempt a Cholesky
factorization; if it succeeds H is PD and we return `inv(H)` unchanged —
bit-for-bit identical to the prior code in the overwhelmingly common case.
Repair path: if Cholesky fails, eigendecompose the symmetrized H, floor its
eigenvalues at a condition-number cap (`max(λ_max · 1e-10, 1e-12)`), and
rebuild the inverse. The result is always SPD with bounded variance in
flat/indefinite directions, so the downstream `slogdet` sign is guaranteed
positive.

## Alternatives considered

- **Uniform ridge nugget (H + τI).** Shifts every eigenvalue up, shrinking all
  variances even when only one direction is bad. Eigenvalue flooring is more
  targeted and is a no-op when H is already PD.
- **Always eigendecompose.** Robust but perturbs ν by ~1e-12 on every document
  even in the PD case, needlessly. The Cholesky fast path avoids touching the
  common case at all.
- **Tighten L-BFGS / raise SIGMA_FLOOR instead.** Reduces but does not
  eliminate the non-PD possibility; a structural guard is the correct fix.

## Consequences

- The Laplace covariance is always SPD; the KL and residual statistics are
  always well-defined. Sibling robustness fix to ADR 0027 from the same review.
- **Dirichlet-family models are unaffected.** LDA, HDP, and the forthcoming
  gated LDA (PLDA) use conjugate closed-form CAVI per-document updates with no
  Hessian inversion, so they have no analogous failure mode — the fragility is
  intrinsic to STM's non-conjugate logistic-normal prior (insight 0028).
- A counter of how often the repair path fires (to learn whether non-PD H
  occurs in real fits) is a possible follow-up; not added now to avoid changing
  the `local_update` sufficient-statistic contract.
