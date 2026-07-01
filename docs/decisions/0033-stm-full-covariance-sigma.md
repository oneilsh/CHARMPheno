# 0033 — STM replaces diagonal Σ with a full (K−1)×(K−1) covariance (CTM/STM treatment)

**Status:** Accepted + Implemented (this branch, SDD arc)
**Date:** 2026-06-30

> **Amendment (2026-07-01) — the gated M-step is superseded by ADR
> [0034](0034-stm-blockwise-unit-diagonal-correlation-sigma.md).** For the gated
> **single-label** case, the full-covariance-with-`pd_complete` M-step below (decisions
> 3-5, 7) is replaced by a **block-wise unit-diagonal correlation** M-step: standardize
> the observed per-pair scatter to correlations, lazy-keep unsupported pairs, pin the
> diagonal to 1 — no completion. This fixes the variance runaway ADR 0033 left open
> (insight 0033: decision 6 removed the variance anchor, so a weakly-identified topic's
> Σ_kk could run away) by removing the free prior variance entirely, and it retires
> `pd_complete` from the fit path because single-label gating never inverts a
> cross-group entry. `pd_complete` / `min_frobenius_psd_completion` remain as tested
> utilities (kept for a future multi-label/comorbid fit, which still needs completion).
> The material below stands as the full-covariance record; ADR 0034 governs the shipped
> gated engine.

## Context

`OnlineSTM` has always stored Σ as a K-vector of per-topic variances, and every
downstream term is elementwise-diagonal: the Gaussian prior gradient −(η−μ)/σ²,
the prior Hessian block `diag(1/Sigma_diag)`, and the η-KL (in the per-doc
neg-log-joint Hessian and the `local_update` η-KL accumulation — both replaced by
the full-matrix forms by this ADR).
This diagonal mean-field representation is not the CTM/STM model of Blei & Lafferty
2007 or Roberts et al. — it forgoes the signature feature that distinguishes
correlated topic models from LDA: modeled topic correlation. The practical consequence
is that the reference `stm` package's `sigma.prior` (a diagonal-shrink lever ∈ `[0, 1]`)
is a literal no-op in OnlineSTM — there are no off-diagonals to shrink.

The K-1 reference-topic parameterization (ADR 0031) placed Σ over the K−1 free topics;
the full K×K Laplace covariance ν_d and Cholesky-based inverse are already computed
per document ([`_spd_inverse`](../../spark-vi/spark_vi/models/topic/stm.py#L203-L228)).
The diagonal model simply discards the off-diagonals at every M-step. The
computational infrastructure for full-Σ inference is therefore already in place.

The scientific deliverable is the (K−1)×(K−1) topic correlation matrix R: which
phenotypes co-occur. Exp 0015 and 0017 (insight 0030) established the diagonal-Σ
result — bounded, stable Σ, 40 resolved topics — as the prior baseline. This arc
replaces the diagonal with a full Σ and re-validates the whole stabilizer stack
(reference, spectral, gating, Σ-prior) under the genuine model.

## Decision

Replace the diagonal Σ representation with a full (K−1)×(K−1) positive-definite Σ —
the logistic-normal covariance of Blei & Lafferty 2007 — across the entire STM
inference pipeline. Seven design decisions govern the implementation:

**1. Goal and scope: topic correlations as the scientific deliverable.**
The correlation matrix R_ij = Σ_ij / sqrt(Σ_ii·Σ_jj) over free topics is computed
at export alongside Σ and the support matrix N_ij. Dashboard surfacing of R is a
separate downstream arc.

**2. Replace, not toggle.**
The diagonal representation is removed. There is no `covariance_type` knob. The
diagonal results (exp 0015/0017, insight 0030) are the prior baseline against which
the full-Σ runs are measured; they are not preserved as a runtime option.

**3. Gated-doc prior = marginal sub-block.**
A hard-gated document's active-topic set A_d has prior η_{A_d} ~ N(μ_{A_d}, Σ_{A_d,A_d}):
the A,A sub-block of Σ, inactive topics integrated out. This is the correct marginal
(not the conditional on a clamped value). The inverse of this sub-block is precomputed
once per distinct group-combination present in the minibatch and broadcast.

**4. Multi-group membership and cross-group covariance.**
Each document belongs to a set of groups; its allowed set is
background_indices ∪ (union of its groups' foreground blocks). Cross-group covariance
is estimated exclusively from comorbid documents that co-activate both groups. The
M-step is a per-pair lazy update (the per-pair generalization of the per-block lazy
rule, ADR 0027): each document scatter-adds its outer-product + Laplace covariance
into the A_d × A_d block of the Σ accumulator and increments a parallel support
matrix N_ij for every active pair. A cross-group entry gains support only from
documents that co-activate both indices.

**5. SPD-assembly risk and the three-layer mitigation.**
**SUPERSEDED for the gated cross-group case** by the
[gated-Σ PD-completion design](../superpowers/specs/2026-06-30-stm-gated-sigma-pd-completion-design.md)
(2026-06-30). The three-layer zero-pin + nearest-SPD-flooring approach below was
shown not to condition the assembled gated Σ — the four lever-tuning experiments
0022-0024 all failed (insight
[0032](../insights/0032-gated-fullcov-recovers-dementia-subphenotypes-and-exposes-spd-assembly-conditioning.md),
Findings 4-6). The assembly is now reframed as covariance selection: zero the
PRECISION (Σ⁻¹), not the covariance, on unobserved cross-pairs — the
maximum-determinant PD completion (`pd_complete`,
[_linalg.py:32-105](../../spark-vi/spark_vi/models/topic/_linalg.py#L32-L105)),
well-conditioned by construction with no conditioning knob. `nearest_spd` remains as
the per-doc Laplace-Hessian repair and as the completion's internal PSD fallback. The
description below is retained as the historical decision.

**Amendment (2026-07-01) — `pd_complete` is a fit-time prior, not a conditioning
cure; the conditioning diagnostics were a reporting artifact.** The correlation-
reporting arc (insight 0032 Resolution) established that the four-experiment
"conditioning pathology" this decision was written to fix was never really about
fit health: the gated single-document E-step inverts only the marginal sub-block
`Sigma[allowed, allowed]` via
[`safe_inverse`](../../spark-vi/spark_vi/models/topic/stm.py#L779), so the
cross-foreground block — the one whose completion Findings 3-6 chased — never
enters any single-group document's inference at all. `pd_complete` is correctly
retained (it is still what runs at the M-step, [stm.py:709-727](../../spark-vi/spark_vi/models/topic/stm.py#L709-L727)),
but its role is properly understood as giving MULTI-GROUP (comorbid) documents a
coherent cross-foreground prior at fit time — the Dempster (1972) zero-precision /
conditional-independence completion of the unobserved cross-pairs — not as a device
for lowering a full-matrix condition number that the fit does not depend on.
Consequently the `sigma_cond` and `max_abs_offdiag_corr` full-matrix diagnostics
(Consequences, below, and ADR 0030) were REMOVED (commit 35deb1e) as reporting
artifacts rather than fit-health signals. Correlation reporting was changed to
report honestly at the pair level instead: `topic_correlation_identified`
([_linalg.py](../../spark-vi/spark_vi/models/topic/_linalg.py)) returns R with a
support-keyed identified mask, NaN-ing any pair whose document support (`n_pairs`)
falls below `min_pair_support` rather than summarizing the whole matrix with one
scalar. See insight 0032's Resolution section for the full argument and the test
that proves recovery is invariant to the full-matrix condition number.

In the gated case, Σ entries are estimated from different document subsets and some
cross-group cells are pinned to the prior, so the assembled matrix need not be
positive definite — this is guaranteed to arise, not hypothetical. Background topics
appear in every allowed set, so they correlate with every foreground block; pinning
cross-foreground entries while background-foreground correlations are freely estimated
is exactly the inconsistency that breaks positive definiteness. The fix is three
layers, all load-bearing:

- (i) The inverse-Wishart prior (Component 3 below) fills uninformed entries with a
  coherent SPD scale instead of a raw zero or a few-patient estimate.
- (ii) Nearest-SPD eigenvalue-floor projection (generalizing the per-doc Hessian
  repair in [`_spd_inverse`](../../spark-vi/spark_vi/models/topic/stm.py#L203-L228)
  to the global Σ) — also the principled minimum-perturbation imputation of unobserved
  cross-group entries.
- (iii) The `sigma_ridge`·I floor (already present).

Note: a `sigma_ridge` bump was considered as a standalone fix but rejected in favour
of the nearest-SPD eigenvalue floor, which is the minimum perturbation that restores
positive definiteness rather than inflating the diagonal uniformly.

**6. Two opt-in regularizers (both default off).**
**REMOVED** (2026-06-30) by the
[gated-Σ PD-completion design](../superpowers/specs/2026-06-30-stm-gated-sigma-pd-completion-design.md):
the inverse-Wishart prior (`sigma_prior_scale`, `sigma_prior_count`) and
`sigma_diag_shrink` are deleted entirely — parameters, plumbing, flags, and tests.
Neither conditions the gated full-Σ (the IW prior is N-weighted and reaches neither
well-supported end; `sigma_diag_shrink` fixes the min eigenvalue only by
decorrelating, which triggers a max-eigenvalue variance runaway — insight
[0032](../insights/0032-gated-fullcov-recovers-dementia-subphenotypes-and-exposes-spd-assembly-conditioning.md),
Findings 4-6). The PD completion makes both unnecessary. The original decision
text is retained below as historical record (superseded by the PD completion).

--- begin historical decision text ---

Both regularizers apply after the M-step scatter:

- *Inverse-Wishart prior* (Blei & Lafferty 2007): scale matrix Ψ = `sigma_prior_scale`·I,
  pseudo-count ν = `sigma_prior_count`. The MAP M-step becomes
  Σ_ij = (S_ij + Ψ_ij) / (N_ij + ν) per entry — shrinks toward Ψ/ν and regularizes
  thin cross-group cells. This cleanly generalizes the old diagonal inverse-gamma
  (which was the Ψ=ψI special case); parameter names are unchanged. Default Ψ=0, ν=0
  reduces to plain MLE.
- *Diagonal-shrink* (`sigma_diag_shrink` ∈ `[0, 1]`, Roberts et al. `sigma.prior`):
  Σ ← (1−w)·Σ + w·diag(diag(Σ)). Shrinks off-diagonal correlations toward zero; now
  meaningful because there are off-diagonals to shrink. Default w=0 is the identity.

These two regularizers have **distinct, non-interchangeable roles — one per
spectral end** — established empirically by exps 0022/0023/0024 (insight
[0032](../insights/0032-gated-fullcov-recovers-dementia-subphenotypes-and-exposes-spd-assembly-conditioning.md),
Findings 4-5). The diagonal-shrink is N-INDEPENDENT (a fractional (1−w) pull on
every off-diagonal regardless of support); it owns the MIN-eigenvalue / correlation
end — the SPD-assembly near-singularity that lives in well-supported
background↔foreground couplings, which the N-weighted IW prior cannot reach (exp
0022: ν=100 left cond at 3e7; exp 0023: `sigma_diag_shrink` lifted min eig 1e-6 →
1.0). The IW prior's off-diagonal action is N-WEIGHTED (Ψ is diagonal, so
Σ_ij = S_ij/(N_ij + ν) for i≠j), so it is a VARIANCE regularizer: its diagonal MAP
Σ_ii = (S_ii + ν·scale)/(N_ii + ν) is a per-iteration contraction toward `scale`
that owns the MAX-eigenvalue / variance end — capping the runaway a weakly-identified
topic exhibits once decorrelation removes the off-diagonal coupling that had been
stabilizing it (exp 0023: diag-shrink alone let one topic's variance run to 2.2e7;
this is insight 0029's runaway, cured by insight 0031's `sigma_prior_count=2000`).
The nearest-SPD floor (Decision 5) guarantees positive-definiteness but conditions
neither end. A well-conditioned GATED Σ therefore needs BOTH levers; the non-gated
case (exp 0020, every topic fully supported, cond 13.3) needs neither.

Pipeline order: scatter + min_pair_support floor → IW blend → diagonal-shrink →
ridge + SPD-repair. Both knobs at defaults reduce to the Component 1 MLE.

--- end historical decision text ---

**7. min_pair_support floor (robustness and small-cell privacy).**
**Zero-pin SUPERSEDED for the gated cross-group case** (2026-06-30): the
`min_pair_support` threshold survives unchanged, but it no longer ZEROS the scatter
of a thin cross-pair. Under the
[gated-Σ PD-completion design](../superpowers/specs/2026-06-30-stm-gated-sigma-pd-completion-design.md)
the threshold selects observed-vs-free for the completion — a cross-pair with
N ≥ `min_pair_support` is observed (entry = S/N, fixed); a thinner one is FREE and
filled by `pd_complete` with its zero-precision (conditional-independence-implied)
value, not pinned to zero covariance. The robustness/small-cell rationale below is
unchanged; only the below-floor action changes from zero-pin to free-for-completion.

A covariance entry backed by fewer than `min_pair_support` co-activating documents is
statistically unreliable and a small-cell disclosure risk. Below the floor the scatter
contribution is zeroed (S_ij → 0) and the entry falls back to the IW prior or
SPD-completion. Background and within-group cells have massive support and never
trigger the floor; the floor bites exactly the thin cross-group cells.

**Deferred — N persistence and imputed_fraction diagnostic:** Persisting the support
matrix N_ij as a sidecar artifact and computing the `imputed_fraction` summary (share
of entries below the floor) are deferred to the dashboard-surfacing arc. N matters
only in the gated/multi-group case and is needed for dashboard annotation of each
correlation as measured vs imputed. The full-covariance engine computes correlations
correctly now; N-based provenance tracking is future work.

## Alternatives considered

- **Diagonal Σ (status quo).** Retained indefinitely as a toggle. Rejected: the
  diagonal model is structurally wrong for CTM/STM — there is no off-diagonal signal
  to recover regardless of the data. Keeping it as an option would fragment the
  stabilizer validation and signal that the diagonal is a legitimate choice rather
  than a limitation that has been fixed. The diagonal results (exp 0015/0017, insight
  0030) serve as the prior baseline; the option itself is removed.
- **Ridge-only SPD repair for gated Σ.** Rejected as a standalone fix: a scalar
  ridge inflates the diagonal uniformly, distorting all correlations, whereas the
  nearest-SPD eigenvalue floor is the minimum-perturbation completion. The ridge
  remains as the final floor, applied after the eigenvalue repair.
- **Conditional sub-block prior (conditioning on clamped inactive-topic value).**
  Rejected: the hard-masking interpretation integrates out the inactive topics, giving
  the marginal N(μ_A, Σ_AA). Conditioning on a clamped scalar value introduces a
  spurious signal that depends on the arbitrary pin location.
- **Per-block (not per-pair) lazy update for gated Σ.** Would estimate cross-block
  Σ entries from all documents in either block, including non-comorbid ones. Rejected:
  cross-group covariance should reflect co-occurrence, not marginal overlap. The
  per-pair rule is the correct generalization of ADR 0027's per-block rule.

## Consequences

- **ADR 0028-B logistic-normal sampler is now enabled.** The parked alternative B of
  ADR 0028 (sample η ~ N(Γᵀx, Σ), θ = softmax(η) — the faithful STM draw) was
  rejected in that arc because Σ was not exported as a full matrix. Full Σ removes
  that blocker; dashboard wiring is a downstream arc (see
  [ADR 0028](0028-dashboard-conditioned-dirichlet-prior.md)).
- **Σ diagnostic generalizes, then the full-matrix summary is removed (2026-07-01
  amendment).** The `Σ[min…max]` trace (ADR 0030) was extended to eigenvalue range +
  condition number and max |off-diagonal correlation|; see
  [ADR 0030](0030-diagnostic-traces-persist-faithfully-no-size-cap.md). That
  extension was subsequently REMOVED (commit 35deb1e) once insight 0032's
  Resolution established it was a reporting artifact of a block the gated E-step
  never inverts — see the decision 5 amendment above. Per-pair support (`n_pairs`)
  IS persisted (superseding the "deferred" note previously here) and backs the
  `topic_correlation_identified` support-keyed mask
  ([_linalg.py](../../spark-vi/spark_vi/models/topic/_linalg.py)) used for
  correlation reporting instead of a single scalar condition number.
- **No backward compatibility.** Legacy diagonal checkpoints (K-vector `global_params["Sigma"]`)
  do not reload under the full-Σ model. Re-fit under full Σ is required. No
  promote-on-load shim.
- **Storage shape change.** `global_params["Sigma"]` goes from a K-vector to a
  (K−1)×(K−1) matrix. New persisted artifacts: correlation matrix R and a free-topic
  → topic-id map. `.npy` handles the shape change; no format migration. The support
  matrix N_ij is **deferred** — not persisted in this arc.
- **Cost is negligible at K=40.** Σ is 39×39: one Cholesky per global iteration, one
  (K−1)×(K−1) matrix-vector product per document (already paid for the Laplace inverse).
  All V-sized work is unchanged.
- **The whole stabilizer stack re-validates under full Σ** — reference topic, spectral
  init, gating, Σ-priors. The diagonal experiments (exp 0015/0017, insight 0030, and
  the stability follow-ups 0018/0019) are the prior baseline. Exps 0020 (non-gated
  full-Σ cancer) and 0021 (gated multi-group comorbid) are the validation runs.

## References

- Blei, D. M. & Lafferty, J. D. (2007). "A Correlated Topic Model of Science."
  *Annals of Applied Statistics*, 1(1), 17–35. — the logistic-normal full-Σ treatment
  and the conjugate inverse-Wishart prior on Σ.
- Roberts, M. E., Stewart, B. M., & Tingley, D. (2019). `stm`: An R package for
  structural topic models. *Journal of Statistical Software*, 91(2).
  `sigma.prior` ∈ `[0, 1]` diagonal-shrink regularizer; source
  https://github.com/bstewart/stm
- Dempster, A. P. (1972). "Covariance Selection." *Biometrics*, 28(1), 157-175. —
  zeroing precision entries = conditional independence; the maximum-entropy
  completion. Basis for the gated-Σ PD-completion redesign of decisions 5/7.
- Grone, R., Johnson, C. R., Sá, E. M., & Wolkowicz, H. (1984). "Positive definite
  completions of partial Hermitian matrices." *Linear Algebra and its Applications*,
  58, 109-124. — existence/uniqueness of the maximum-determinant PD completion;
  closed form for chordal patterns.
- Speed, T. P., & Kiiveri, H. T. (1986). "Gaussian Markov distributions over finite
  graphs." *Annals of Statistics*, 14(1), 138-150. — iterative proportional scaling
  for covariance selection (the `pd_complete` primary path).
- Higham, N. J. (2002). "Computing the nearest correlation matrix — a problem from
  finance." *IMA Journal of Numerical Analysis*, 22(3), 329-343. — alternating
  projections onto the PSD cone; the `pd_complete` fallback for non-PD observed input.
- [gated-Σ PD-completion design](../superpowers/specs/2026-06-30-stm-gated-sigma-pd-completion-design.md)
  — supersedes decisions 5/7's gated cross-group handling (zero-pin → max-det PD
  completion) and removes decision 6's two regularizers.
- [gated CTM correlation-reporting design](../superpowers/specs/2026-07-01-gated-ctm-correlation-reporting-design.md)
  — the 2026-07-01 amendment to decision 5: reframes `pd_complete` as a fit-time
  cross-foreground prior (not a conditioning cure), removes the full-matrix
  `sigma_cond` / `max_abs_offdiag_corr` diagnostics as reporting artifacts, and
  introduces the support-keyed `topic_correlation_identified` mask for honest
  per-pair correlation reporting.
- [ADR 0027](0027-lazy-block-updates-for-gated-svi-mstep.md) — per-block lazy update,
  generalized here to per-pair for cross-group covariance.
- [ADR 0028](0028-dashboard-conditioned-dirichlet-prior.md) — parked alternative B
  (logistic-normal sampler) now enabled by the full Σ.
- [ADR 0029](0029-spd-guard-on-stm-laplace-hessian.md) — SPD guard on the per-doc
  Laplace Hessian inverse; the nearest-SPD floor is generalized here to the global Σ.
- [ADR 0030](0030-diagnostic-traces-persist-faithfully-no-size-cap.md) — Σ diagnostic
  traces; the Σ[min…max] trace is extended to eigenvalue range, condition number, and
  max off-diagonal correlation.
- [ADR 0031](0031-stm-k1-reference-topic-parameterization.md) — K−1 reference
  parameterization; Σ lives over the K−1 free topics.
- [insight 0029](../insights/0029-stm-sigma-init-collapse-blowup-missing-stabilizers.md)
  — the three missing stabilizers (reference, spectral, Σ-prior).
- [insight 0030](../insights/0030-spectral-init-closes-stm-sigma-blowup-on-real-data.md)
  — diagonal-Σ spectral result, the prior baseline for full-Σ validation.
- [design spec](../superpowers/specs/2026-06-30-stm-full-covariance-sigma-design.md)
  — the full design with component-level math and validation plan for this arc.
