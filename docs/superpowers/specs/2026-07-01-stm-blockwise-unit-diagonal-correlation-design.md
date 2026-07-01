# Block-wise unit-diagonal correlation Σ for gated STM — design

**Date:** 2026-07-01
**Status:** Design approved; spec for implementation
**Supersedes:** the full-covariance-with-PD-completion M-step (ADR 0033 + the
2026-06-30 gated-Σ PD-completion design) for the gated single-label case.

## Context

The gated full-Σ STM (ADR 0033) models topic correlations via a full (K−1)×(K−1)
covariance assembled per-pair and repaired to positive-definite by the maximum-
determinant completion `pd_complete` (Dempster 1972 covariance selection), with a
Dykstra min-Frobenius fallback for non-completable observed blocks.

Two problems surfaced on the real `cancer_or_dementia` cohort:

1. **Variance runaway** (insight 0033, confirmed via exp 0026). A rare-but-coherent
   minority dementia sub-phenotype (too few documents to constrain its variance) drives
   its η-variance to a runaway via the softmax-saturation feedback loop — one topic's
   Σ_kk reached 2.7e5, ELBO −1.56e7. Controlled reproduction established that (a) the
   runaway needs a weakly-identified/under-constrained topic AND a prior variance free
   to grow, and (b) better initialization cannot fix it (the topic is already
   coherently identified; it is document-scarce). The post-M-step variance levers
   (inverse-Wishart prior, `sigma_diag_shrink`) were falsified across exps 0022-0024
   (insight 0032, Findings 4-6).

2. **Completion cost + fallback** (insight 0032 Finding 3). The gated cross-foreground
   block is generically non-PD-completable (thin comorbid support), so `pd_complete`
   runs to its 1000-sweep cap then falls back to Dykstra — ~13-55s per M-step on the
   driver — and pins the assembled Σ's minimum eigenvalue at the floor.

The key structural observation: **under single-label gating (each document belongs to
exactly one group — the current `PatientCohortDocSpec`), no E-step ever inverts a
cross-foreground entry.** Each document's marginal is `Sigma[bg ∪ its-one-group]`
([stm.py:797](../../../spark-vi/spark_vi/models/topic/stm.py#L797)); the cancer↔dementia
block is never sliced in. The completion exists solely to give hypothetical comorbid
(multi-group) documents a coherent cross-foreground prior — documents the single-label
pipeline does not have.

## Goal

Replace the full-covariance-with-completion Σ with a **unit-diagonal correlation
matrix estimated block-wise, without any completion**, for the gated single-label STM.
This fixes the variance runaway by construction, retires `pd_complete` / the Dykstra
fallback / the driver-side completion cost / the min-eigenvalue near-singularity, and
preserves the scientific deliverable (within-group and background↔foreground topic
correlations) that the demo needs. Cross-group correlations are not modeled (reported
NA) — an accepted scope reduction, since the single-label pipeline never observes them.

## Decisions (settled during brainstorming)

1. **Replace, not toggle.** Unit-diagonal correlation Σ everywhere; free-variance
   full-Σ is removed. (Consistent with ADR 0033's "replace not toggle".) Non-gated fits
   have no free pairs and reduce to estimating the full correlation matrix directly.
2. **Single-label gating** (foreground groups non-overlapping). Multi-membership is
   foreclosed by this design; if cross-group correlations are ever wanted, that is a
   separate future arc that re-introduces a completion.
3. **Block-wise M-step, no completion.** Standardize the observed scatter to
   correlations on supported pairs; lazy-keep unsupported pairs; pin the diagonal to 1.
   No `pd_complete`.

## The math (why this works)

**The model.** Per document d, η_d ~ Normal(μ_d, Σ), θ_d = softmax(η_d), words ~ θ_d·β
(Blei & Lafferty 2007 logistic-normal). Σ is the topic covariance; here constrained to
a correlation matrix (unit diagonal).

**Why unit-diagonal stops the runaway.** The runaway is the softmax-saturation feedback
loop: a loose prior on topic k (large Σ_kk) lets η_d,k drift to the softmax-saturation
boundary where the likelihood gradient vanishes, inflating the residual and hence Σ_kk
next M-step — a positive feedback with no fixed point for a document-scarce topic
([stm.py:422-425](../../../spark-vi/spark_vi/models/topic/stm.py#L422-L425); insight
0029/0033). Pinning Σ_kk = 1 severs the first link: the prior variance can never
loosen, so the loop cannot start. This is the tuning-free ν→∞, scale=1 limit of a
variance anchor — at the load-bearing scale insight 0030 identified (1, not 2).

**Why block-wise needs no completion.** Under single-label gating, every document's
allowed set is `bg ∪ (one group)`, and the E-step only ever inverts that marginal
sub-block. Every pair *within* such a marginal (bg↔bg, bg↔fg, fg↔fg) is observed by
that group's own documents, so the marginal is fully specified — nothing to complete.
Only cross-foreground pairs (A↔B) are unobserved, and no marginal contains them. The
full K×K Σ is therefore left block-structured (cross-foreground entries at their zero
init) and is generically *not* globally PD — which is harmless, because no computation
ever inverts the full matrix. Prototype confirmation (gated synthetic, 8% minority):
E-step marginals min-eig A=0.379, B=0.036 (both PD) while the full Σ min-eig = −0.8
(unused); 14/14 topics recovered including the minority arm; max variance pinned at 1.

**Standardization from per-pair support.** The observed correlation estimate is
R_ij = (S_ij/N_ij) / sqrt((S_ii/N_ii)·(S_jj/N_jj)), where S is the residual-outer +
Laplace-covariance scatter and N the per-pair document support. The empirical variances
S_ii/N_ii are used only to standardize; they are never stored as the prior (that is
exactly the runaway degree of freedom being removed). Because supports differ per pair,
a standardized within-marginal correlation can in principle exceed 1 or dip a marginal
slightly non-PD; `safe_inverse`'s existing eigenvalue floor is the graceful per-document
catch (a lightweight clamp, not a global Dykstra completion).

## Components

### 1. M-step: block-wise unit-diagonal Σ (`OnlineSTM.update_global`)

Replaces the covariance-assembly + `pd_complete` block
([stm.py:692-747](../../../spark-vi/spark_vi/models/topic/stm.py#L692-L747)). β and Γ
updates are unchanged. New Σ update:

```
S = residual_outer_stat;  N = n_pairs_stat
supported = N >= min_pair_support
mle       = where(supported, S / max(N, 1), 0.0)      # per-pair covariance
std       = sqrt(where(diag(mle) > 0, diag(mle), 1.0)) # standardize scale (not stored)
R         = mle / outer(std, std)                      # observed correlations
R_target  = where(supported, R, Sigma_prev)            # lazy-keep unsupported (ADR 0027)
Sigma_new = (1 - lr) * Sigma_prev + lr * R_target
fill_diagonal(Sigma_new, 1.0)                          # exact unit diagonal; NO completion
```

`min_pair_support` retains its meaning (observed-vs-lazy threshold, robustness +
small-cell guard). Cross-foreground pairs have N=0 → unsupported → lazy-kept at the
zero init. A group absent from a minibatch keeps its previous entries (lazy-block
invariant, ADR 0027) — no decay toward a floor.

### 2. E-step (unchanged)

`infer_local` / `local_update` continue to invert the marginal
`Sigma_inv_allowed = safe_inverse(Sigma[allowed, allowed])`
([stm.py:797](../../../spark-vi/spark_vi/models/topic/stm.py#L797)). `safe_inverse`
stays as the per-document PD guard. No change.

### 3. Initialization (`initialize_global`)

Initial Σ = identity (unit diagonal). `sigma_init` becomes vestigial (the diagonal is
pinned to 1 at every M-step); retain the parameter as the pre-first-M-step scale for
API stability OR remove it — decided in the plan (prefer removal if no downstream
reader depends on it).

### 4. Reporting (unchanged — already correct)

`topic_correlation_identified(Sigma, n_pairs, min_pair_support)`
([_linalg.py](../../../spark-vi/spark_vi/models/topic/_linalg.py)) already returns R
with a support-keyed identified mask. Since Σ is now itself unit-diagonal,
`topic_correlation(Σ) ≈ Σ`. Cross-group pairs (n_pairs=0 < floor) are NA'd exactly as
desired; within-group and bg↔fg pairs are identified. The charmpheno
`build_correlation_json` export and the dashboard heatmap consume this unchanged.

### 5. Retired surface

- `pd_complete` and `min_frobenius_psd_completion` are no longer called by the M-step.
  Retire them (and their tests) if no other caller remains; the plan verifies callers
  repo-wide before deletion.
- The per-iter `M-step pd_complete: …` driver log and its `time`/`logging`
  instrumentation are removed with the completion.
- The precision-space `pd_complete` rewrite (commit 4d1d114) becomes moot for the
  M-step; it stays only if `pd_complete` is retained as a utility.

## Data flow

`local_update` (per-doc Laplace → scatter S, support N) → `update_global` (block-wise
standardized unit-diagonal Σ, no completion) → next iter's E-step inverts marginals →
… → `save` persists Σ (unit diagonal, block-structured) + n_pairs → correlation export
(within-group/bg identified, cross-group NA) → dashboard heatmap.

## Error handling

- **Non-PD marginal** (rare, from mismatched within-marginal support): `safe_inverse`
  floors eigenvalues per document — existing behavior, no new path.
- **Non-PD full Σ** (expected, cross-foreground zero): harmless; never inverted. Any
  code asserting full-Σ PD-ness must be found and relaxed (the plan audits this — e.g.
  a future ADR-0028-B logistic-normal sampler must sample per-group marginal, not the
  full joint).
- **Document-scarce topic**: variance pinned at 1 (the whole point); its correlations
  are estimated from what little support exists and NA'd below `min_pair_support`.

## Testing

- **Unit (M-step):** output is unit-diagonal; supported off-diagonals equal the
  standardized correlations; unsupported pairs lazy-kept; cross-group stays at init;
  absent group unchanged.
- **Unit (runaway):** feed a scatter whose free-variance M-step would inflate a
  diagonal; assert block-wise pins it to 1.
- **Unit (E-step marginal):** a representative gated Σ yields PD marginals; a
  constructed non-PD marginal is handled by `safe_inverse` without raising.
- **Integration (the prototype as a test):** gated synthetic with a thin minority arm —
  no variance runaway (max diag ≈ 1), planted-topic recovery holds (incl. minority via
  `foreground_recovers_group`), within-group + bg↔fg correlation MAE within tolerance.
- **Characterization (non-gated):** no free pairs → Σ is the directly-standardized full
  correlation matrix.
- **Reporting:** existing correlation-export / dashboard tests stay green (cross-group
  NA, within-group identified).

## Validation (cluster)

**exp 0027** — `cancer_or_dementia`, K=50 = 30 bg + 10 cancer + 10 dementia, single-
label, block-wise unit-diagonal. Success: max Σ_var ≈ 1 (no runaway), Alzheimer's/
amnestic + vascular dementia sub-phenotypes preserved (insight 0032 Finding 2),
within-group + bg↔fg R trustworthy, cross-group NA, ELBO recovered near −1.59e6, and
per-iter wall-clock free of the completion cost.

## Decision record

A new ADR (next number) amends ADR 0033: block-wise unit-diagonal correlation Σ
replaces full-covariance-with-completion for gated single-label STM; records the
runaway fix (insight 0033), the single-label scope, and the retirement of
`pd_complete`/Dykstra. Insight 0032's Resolution and insight 0033 are cross-linked.

## Migration / compatibility

No backward compatibility (consistent with ADR 0033): legacy full-covariance Σ
checkpoints do not reload under the unit-diagonal model; re-fit required. `.npy` handles
the (still K×K) shape; no format change.

## References

- Blei, D. M. & Lafferty, J. D. (2007). "A Correlated Topic Model of Science." *Annals
  of Applied Statistics* 1(1), 17-35. — logistic-normal topic covariance.
- Lewandowski, D., Kurowicka, D., & Joe, H. (2009). "Generating random correlation
  matrices based on vines and extended onion method." *J. Multivariate Analysis*
  100(9), 1989-2001. — the LKJ distribution; the standard way to model a correlation
  matrix directly (conceptual grounding for the unit-diagonal parameterization).
- insight [0033](../../insights/0033-gated-fullcov-variance-runaway-is-an-init-identifiability-failure.md)
  — the runaway is an under-constrained-rare-topic problem; unit-diagonal is the fix.
- insight [0032](../../insights/0032-gated-fullcov-recovers-dementia-subphenotypes-and-exposes-spd-assembly-conditioning.md)
  — the completion, the falsified levers, and the single-group-marginal E-step.
- insight [0030](../../insights/0030-spectral-init-closes-stm-sigma-blowup-on-real-data.md)
  — σ-scale 1 is load-bearing.
- ADR [0033](../../decisions/0033-stm-full-covariance-sigma.md) — full-covariance Σ
  (amended by this design), ADR [0027](../../decisions/0027-lazy-block-updates-for-gated-svi-mstep.md)
  — the lazy-block invariant retained here.
```
