# STM gated-Σ positive-definite completion — design

**Date:** 2026-06-30
**Status:** Draft (for review)
**Branch:** stm
**Supersedes (operationally):** the `min_pair_support` zero-pin + post-hoc
`nearest_spd` flooring as the gated-Σ assembly strategy. Governed by ADR 0033;
this design replaces decision 5/7's cross-group handling.

## Goal

Make the gated multi-group full-covariance Σ well-conditioned **by construction**,
with no tunable conditioning knob, by estimating it as the maximum-determinant
positive-definite completion of the well-supported entries (a Gaussian graphical
model in which topic pairs lacking joint support are conditionally independent
given the rest). Replace the current zero-pin of thin cross-group covariance cells,
which produces a near-singular assembled matrix (insight 0032).

## Problem

`OnlineSTM.update_global` assembles the K×K topic covariance Σ from per-pair
sufficient statistics whose support is heterogeneous: background↔background pairs
are informed by every document, within-group pairs by that group's documents, and
cross-group pairs only by comorbid documents that co-activate both groups. The
current M-step
([stm.py:711-740](../../../spark-vi/spark_vi/models/topic/stm.py#L711-L740))
forms each entry as `S_ij / N_ij`, zeros any cross-pair with `N_ij <
min_pair_support`, and repairs the result with `nearest_spd`.

This zero-pin sets an unobserved **covariance** to zero — the strong, usually false
claim "these topics are marginally uncorrelated." Because background topics
correlate with every foreground block, a zeroed cancer↔dementia covariance is
transitively inconsistent with strong background↔cancer and background↔dementia
correlations, so the assembled matrix is indefinite and `nearest_spd` floors a
near-zero eigenvalue. Result: condition number ≈ 3e7, minimum eigenvalue pinned at
`SIGMA_FLOOR=1e-6` (insight
[0032](../../insights/0032-gated-fullcov-recovers-dementia-subphenotypes-and-exposes-spd-assembly-conditioning.md),
Finding 3).

Four post-M-step regularizer experiments failed to fix this (insight 0032,
Findings 4-6): the inverse-Wishart prior is N-weighted and cannot reach the
well-supported entries that drive the singularity (exp 0022); `sigma_diag_shrink`
lifts the min eigenvalue but, by decorrelating, removes the off-diagonal coupling
that stabilizes variances and triggers a max-eigenvalue runaway (exp 0023, e.g.
topic 45 dementia → Σ_ii ≈ 2.6e8); the two together make it worse (exp 0024). The
off-diagonal correlations are simultaneously the source of the near-singularity and
the stabilizer of the variances, so no post-hoc knob conditions the matrix. The
defect is upstream — in what we pin the unobserved cells to.

## Principle

Zero the **precision**, not the **covariance**. Σ⁻¹_ij = 0 encodes conditional
independence of topics i and j given all others — the minimum-assumption
(maximum-entropy) statement for a pair we have no joint data on, and unlike a zeroed
covariance it is consistent with arbitrary marginal correlations through shared
neighbors. The covariance matrix with the observed entries fixed and zero precision
on the unobserved entries is the **maximum-determinant positive-definite completion**
(Dempster 1972; Grone et al. 1984): the unique PD matrix matching the observed
entries whose inverse vanishes on the unobserved positions. It exists and is unique
whenever the observed entries admit any PD completion, and it adds no spurious
cross-correlation — unobserved cross-group covariances take exactly the value implied
by conditional independence given the background (and any other observed links).

This reframes the assembly as fitting a **Gaussian graphical model** whose edges are
the well-supported topic pairs: the non-edges (thin/absent cross-group pairs) are
conditional independencies, and the completion is that model's covariance MLE
(Lauritzen 1996). The non-gated case (exp 0020) is the fully-connected special case —
no missing edges, completion is the identity, which is why it is already
well-conditioned (cond 13.3).

## Solution

Replace the zero-pin + `nearest_spd` repair in `update_global` with a
maximum-determinant PD completion.

### Observed / free partition

After accumulating the scatter `S` (K×K) and support `N` (K×K), partition the
off-diagonal entries:

- **observed:** `N_ij >= min_pair_support` — the entry is `S_ij / N_ij` (the data
  estimate), fixed.
- **free:** `N_ij < min_pair_support` — the entry is unconstrained; the completion
  sets it so that Σ⁻¹_ij = 0.

Diagonals are always observed (every topic's own variance has full within-allowed
support). `min_pair_support` keeps its current meaning — the small-cell/robustness
threshold that decides which cross-pairs are trustworthy — but now selects
observed-vs-free for the completion instead of triggering a zero-pin. Below-floor =
free was chosen over a support-weighted blend (a blend reintroduces an implicit
weighting knob and complicates the PD guarantee).

### Completion algorithm — general, robust to non-PD observed

The completion runs on the K×K matrix on the **driver**; K is the topic count
(tens to low hundreds), fully decoupled from corpus size. The corpus-sized scatter
and support accumulation is already distributed via `treeReduce`
([mllib/topic/stm.py](../../../spark-vi/spark_vi/mllib/topic/stm.py)) and is
unchanged. Cost is O(K³) per global iteration on the driver (microseconds at K=50;
the absurd-K caveat — thousands of topics — is documented, not engineered for).

The algorithm must be **general** (any sparsity pattern, per the multi-group
requirement — a decomposable closed-form is a special case used only as a test
oracle) and **robust to non-PD observed blocks**, which are common in the intended
rare-disease regime: foreground groups are NOT dropped before the fit (only at
export, [gating.py](../../../charmpheno/charmpheno/export/gating.py); fit-time only
warns at <100 docs), so a thin group's within-block estimate, or a clique with
heterogeneous within-clique support, can fail to be PD.

Two literature-grounded families, to be pinned by the implementer with a focused
source check + TDD against the contract below:

1. **Maximum-determinant completion via covariance selection / iterative
   proportional scaling** (Dempster 1972; Speed & Kiiveri 1986). Coordinate ascent
   that drives the free precision entries to zero while matching the observed
   covariance entries; converges to the unique max-det completion when the observed
   part is PD-completable. The principled primary path (gives the exact
   zero-precision, conditional-independence interpretation for the dashboard).

2. **Alternating projection onto the PSD cone and the affine "match-observed" set**
   (Higham 2002; Dykstra's algorithm). Robust by construction: matches the observed
   entries when a PD completion exists, and converges to the minimum-Frobenius PSD
   compromise (perturbing the inconsistent observed entries minimally) when it does
   not. The fallback for the non-PD-observed / rare-disease degenerate case.

Recommended structure: attempt the max-det completion (1); if the observed sub-part
is not PD-completable (detected by solver non-convergence or a non-positive pivot),
fall back to the nearest-PSD projection (2). Both are knob-free up to a numerical
convergence tolerance (a tolerance, not a modeling parameter). The final choice
between a unified alternating-projection solver and the two-stage approach is a spec
decision deferred to implementation, gated on the contract tests.

### Contract (the completion function)

A new pure function in
[_linalg.py](../../../spark-vi/spark_vi/models/topic/_linalg.py):

```
pd_complete(target: (K,K), observed_mask: (K,K) bool, *, tol, max_iter) -> (K,K)
```

- **Input:** `target` carries the observed entries (data estimates) in the
  `observed_mask`-true positions; free positions are ignored on input. The mask is
  symmetric with a true diagonal.
- **Output:** a symmetric PD matrix that (a) matches `target` on observed entries
  (within `tol`, exactly when PD-completable), (b) has near-zero precision on free
  entries, and (c) is PD (minimum eigenvalue > 0).
- **Identity property:** when `observed_mask` is all-true (non-gated, fully
  supported), returns `target` unchanged (the completion is a no-op).
- **Decomposable oracle:** on a chordal observed pattern, matches the closed-form
  completion (free block = Σ_{free,sep} Σ_sep⁻¹ Σ_{sep,free}) within `tol` (a test
  oracle, Grone et al. 1984 / Lauritzen 1996).
- **Robustness:** never raises on a non-PD observed sub-part; returns the
  nearest-PSD compromise.

### M-step integration

In `update_global`, replace the `min_pair_support` zero-pin block and the trailing
`nearest_spd` call with: build the observed mask from `N` and `min_pair_support`,
form the observed entries as `S/N` (with the existing IW-MAP blend retained for the
*observed* entries only — the IW prior remains available as a per-entry variance
regularizer on supported cells, orthogonal to the completion), ρ-blend against the
prior Σ, then `pd_complete`. `sigma_diag_shrink` is **removed from the gated
conditioning path** (it is the wrong lever — insight 0032 Finding 5/6); it may
remain as a parameter for the non-gated case or be deprecated (decided in the plan).
`nearest_spd` stays in the library as the per-document Laplace-Hessian repair
([_spd_inverse](../../../spark-vi/spark_vi/models/topic/stm.py#L203-L228)) and as the
completion's internal PSD projection.

## Validation

- **Unit tests** (TDD, `spark-vi/tests`): the contract above — identity on
  all-observed; decomposable-oracle match; PD output on a constructed inconsistent
  observed pattern (non-PD observed → PSD compromise, no raise); zero-precision on
  free entries; reduces to the prior diagonal/scalar cases. Use the existing
  `tests/_stm_synth.py` planted-covariance fixtures.
- **Engine integration:** a gated `StreamingSTM` fit on synthetic gated data yields
  a PD, well-conditioned Σ with cross-group entries at their CI-implied values.
- **Cluster (exp 0025):** gated comorbid (= exp 0021 config: K=50, background 30,
  cancer 10, dementia 10, `min_pair_support=10`), completion ON, `sigma_diag_shrink`
  and IW prior OFF. Success criteria, head-to-head with 0021-0024:
  - **cond drops to O(1e1-1e3)** with `sigma_eig_min` well off 1e-6 AND
    `sigma_eig_max` O(1-10) — both ends controlled, no flooring;
  - **no variance runaway** (the `stm_sigma_diagnostic` runaway topic Σ_ii back to
    O(1-10), ELBO back to the 0021 regime ≈ −1.59e6, not 0024's −1.9e6);
  - **dementia sub-phenotypes preserved** (topic with Amnesia+Alzheimer's+MCI; topic
    with atherosclerosis+AFib+dementia) and per-block NPMI near 0021 levels;
  - **trustworthy cross-block R:** the cancer↔dementia sub-block of `correlation.npy`
    holds the CI-implied values, interpretable as "correlated only through shared
    background comorbidity."

## Scope

**In:** the `pd_complete` function; the `update_global` integration; removal of the
gated zero-pin and gated `sigma_diag_shrink` conditioning role; unit + integration
tests; exp 0025; an insight + ADR 0033 amendment recording the resolution.

**Out (unchanged / deferred):** the distributed scatter/support accumulation;
dashboard surfacing of R and the measured-vs-imputed annotation (deferred to the
dashboard arc with N-persistence per ADR 0033); the non-gated path (already
well-conditioned, completion is a no-op there); the per-document Laplace inference.

## Decisions baked in

- General iterative completion, not the decomposable closed-form (multi-group
  requirement; closed-form is a test oracle only).
- Full replacement of the zero-pin (no fallback to it; git is the rollback).
- Below-`min_pair_support` = free (binary), not a support-weighted blend.
- Driver-side K×K; corpus-sized work stays distributed.
- Robust to non-PD observed blocks (rare-disease regime) via the alternating-projection
  fallback — no `SIGMA_FLOOR`-style conditioning knob.

## References

- Dempster, A. P. (1972). "Covariance Selection." *Biometrics* 28(1), 157-175. —
  zeroing precision entries = conditional independence; the max-entropy completion.
- Grone, R., Johnson, C. R., Sá, E. M., & Wolkowicz, H. (1984). "Positive definite
  completions of partial Hermitian matrices." *Linear Algebra and its Applications*
  58, 109-124. — existence/uniqueness of the max-determinant PD completion;
  closed-form for chordal patterns.
- Speed, T. P., & Kiiveri, H. T. (1986). "Gaussian Markov distributions over finite
  graphs." *Annals of Statistics* 14(1), 138-150. — iterative proportional scaling
  for covariance selection.
- Lauritzen, S. L. (1996). *Graphical Models*. Oxford University Press. — decomposable
  GGM covariance MLE in closed form (the test oracle).
- Higham, N. J. (2002). "Computing the nearest correlation matrix — a problem from
  finance." *IMA Journal of Numerical Analysis* 22(3), 329-343. — alternating
  projections onto the PSD cone, the robustness fallback for non-PD observed input.
- [ADR 0033](../../decisions/0033-stm-full-covariance-sigma.md) — full-covariance Σ;
  this design replaces its gated cross-group handling (decisions 5/7).
- [insight 0032](../../insights/0032-gated-fullcov-recovers-dementia-subphenotypes-and-exposes-spd-assembly-conditioning.md)
  — the SPD-assembly pathology and the four failed-lever experiments motivating this.
