# 0034 — STM Σ M-step becomes block-wise unit-diagonal (correlation matrix, no PD completion)

**Status:** Accepted + Implemented (this branch, SDD arc)
**Date:** 2026-07-01

## Context

ADR [0033](0033-stm-full-covariance-sigma.md) replaced the diagonal Σ with a full
(K−1)×(K−1) covariance and estimated it at the M-step by max-determinant PD completion
(`pd_complete`) of the observed per-pair scatter. That engine had one unresolved
failure mode. On the real comorbid cohort (exp
[0025](../experiments/0025-stm-comorbid-fullcov-gated-pdcompletion.md)) a single topic's
variance Σ_kk climbed to 2.71e5 by iter 28 and the ELBO diverged to −1.56e7 — a
variance runaway. Exp [0026](../experiments/0026-stm-comorbid-fullcov-runaway-diagnosis.md)
attributed it (insight
[0033](../insights/0033-gated-fullcov-variance-runaway-is-an-init-identifiability-failure.md)):
the runaway needs two ingredients — a weakly-identified topic (here a rare but coherent
minority dementia sub-phenotype, under-constrained by document count) and a prior
variance free to grow. Softmax saturation removes the likelihood's restoring force on
that topic, so its residual grows, its M-step variance Σ_kk grows, the prior loosens,
η drifts further into saturation — a positive feedback with no fixed point. Decision 6
of ADR 0033 had removed the inverse-Wishart variance anchor (the levers were falsified
by exps 0022-0024, insight
[0032](../insights/0032-gated-fullcov-recovers-dementia-subphenotypes-and-exposes-spd-assembly-conditioning.md)
Findings 4-6), so nothing bounded the diagonal.

A second observation makes the completion machinery itself unnecessary. Under
**single-label** gating — each document belongs to exactly one foreground group — no
E-step ever inverts a cross-group entry: a document's prior is the marginal
Σ[background ∪ its one group], which is fully observed and therefore naturally PD
([stm.py:760](../../spark-vi/spark_vi/models/topic/stm.py#L760),
`safe_inverse`). The cross-foreground block that `pd_complete` was built to complete
never enters any document's inference. So for single-label gating the completion is
pure fit-path cost (the serial covariance-selection sweep dominated the driver while
executors idled) with no consumer.

Both problems have the same fix: estimate Σ directly as a **correlation matrix** with
the diagonal pinned to 1. Pinning Σ_ii = 1 removes the variance degree of freedom the
runaway rides on (the ν→∞, scale-1 limit of a variance anchor at the load-bearing
η-scale — insight
[0030](../insights/0030-spectral-init-closes-stm-sigma-blowup-on-real-data.md) showed
scale 1 is the load-bearing default, not cosmetic). And a block-wise correlation
estimate needs no completion, because the only sub-blocks the E-step inverts are
already fully observed.

## Decision

Replace the full-covariance-with-completion Σ M-step with a **block-wise
unit-diagonal correlation** M-step for gated single-label STM. Four decisions govern it.

**1. Unit-diagonal correlation parameterization (the runaway fix).**
Σ is a correlation matrix: after every M-step Σ_ii = 1 exactly
([stm.py:710](../../spark-vi/spark_vi/models/topic/stm.py#L710)). The empirical
per-topic variance S_ii/N_ii is used ONLY to standardize the off-diagonals to
correlations; it is never stored as the prior. This severs the softmax-saturation
feedback loop by construction: a weakly-identified topic's prior variance can no longer
grow, so there is no positive-feedback fixed point to run away to (insight 0033). It is
the ν→∞ / scale-1 limit of a variance anchor at the load-bearing scale — LKJ-style
correlation modeling (Lewandowski, Kurowicka & Joe 2009) rather than a full
inverse-Wishart covariance.

**2. Single-label gating (cross-group correlations foreclosed).**
Each document activates background ∪ exactly one foreground group. No allowed set spans
two foreground groups, so no cross-foreground pair is ever sliced into an E-step
marginal. Cross-group correlations are therefore not estimated and are reported NA
(`topic_correlation_identified` already masks any pair whose support n_pairs falls
below min_pair_support — a single-label cross-group pair has n_pairs = 0). Within-group
and background↔foreground correlations — the ones documents actually inform — are
estimated and reported honestly. This is a deliberate narrowing of ADR 0033 decision 4
(multi-group membership + cross-group covariance from comorbid documents): the block-wise
engine is defined for the single-label case.

**3. Block-wise M-step, no completion.**
The M-step standardizes the observed per-pair scatter to correlations and lazy-keeps
unsupported pairs at their prior Σ value
([stm.py:708](../../spark-vi/spark_vi/models/topic/stm.py#L708)):
R_ij = clip( (S_ij/N_ij) / sqrt((S_ii/N_ii)·(S_jj/N_jj)), −1, 1 ) on supported pairs
(N_ij ≥ min_pair_support), previous Σ_ij on unsupported ones (ADR
[0027](0027-lazy-block-updates-for-gated-svi-mstep.md) lazy invariant — an absent
group's entries do not decay), diagonal pinned to 1. No `pd_complete`, no Dykstra
projection. `pd_complete` and `min_frobenius_psd_completion`
([_linalg.py:104](../../spark-vi/spark_vi/models/topic/_linalg.py#L104),
[_linalg.py:32](../../spark-vi/spark_vi/models/topic/_linalg.py#L32)) are RETAINED as
tested linalg utilities (they have direct isolation-test consumers) — only the M-step
call and its per-iteration timing/logging are removed. The full Σ is left
block-structured (cross-group at its 0 init) and is never inverted whole, so it need
not be PD; only the marginals the E-step actually inverts must be, and they are by
construction.

The per-cell standardization is NOT Cauchy-Schwarz-bounded to [−1,1] on its own: the
covariance S_ij/N_ij is averaged over the N_ij co-active documents, while each variance
S_ii/N_ii is averaged over its own (generally larger) active-doc set, so the numerator
and denominator support counts differ and the ratio can exceed 1 (a
background↔foreground pair is the canonical case — N_ii spans all documents, N_ij only
the one group's). The supported off-diagonals are therefore clipped to [−1,1]
([stm.py:707-708](../../spark-vi/spark_vi/models/topic/stm.py#L707-L708)) — the minimal
projection onto the valid correlation range (cf. Higham 2002 nearest-correlation),
applied per entry rather than globally. With the clip plus the pinned diagonal, Σ is a
valid correlation matrix (all |Σ_ij| ≤ 1, Σ_ii = 1) by construction, not merely
empirically. It need not be PD (see above); the reported correlation entries are always
in range.

**4. E-step marginal inversion unchanged.**
The gated prior is still the marginal sub-block Σ[allowed, allowed] (ADR 0033
decision 3), inverted per distinct group-combination via
[`safe_inverse`](../../spark-vi/spark_vi/models/topic/stm.py#L760) — the lightweight
per-doc eigenvalue-floor guard. Under single-label gating each such marginal is fully
observed, so `safe_inverse` almost never has to floor anything; it remains as the
principled guard, not a load-bearing repair.

`sigma_init` is kept unchanged: the default 1.0 already makes `initialize_global`'s
identity a valid unit-diagonal start, and the M-step pins the diagonal to 1 from the
first step, so it is vestigial but harmless — removing it would be invasive churn
across the shim and drivers for no behavioral gain.

## Alternatives considered

- **Keep `pd_complete` + standardize the output to unit diagonal (post-hoc).**
  Rejected: it keeps the completion's per-iteration cost and its Dykstra fallback
  (which fires at the same rate as the block-wise path would, because the fallback is
  inherent to the gating pattern, not to the parameterization) while adding a
  standardization step on top. The block-wise estimate reaches the same correlation
  matrix without ever forming the completion.
- **Reparameterize the M-step directly in correlation space (standardize the scatter
  FIRST, then complete the correlation target).** Rejected after a local measurement.
  The shipped block-wise M-step shares the underlying |r| > 1 property — standardizing
  per-pair scatter with mismatched per-cell support produces off-diagonals above 1 on up
  to ~22% of observed pairs (the variances come from different document subsets, so
  Cauchy-Schwarz does not hold across pairs) — but handles it with a single per-entry
  clip to [−1,1] (Decision 3). This alternative, by ALSO feeding the standardized target
  through the max-determinant completion, additionally drifts the completed diagonal off
  1 and needs a re-standardization pass afterward — the completion and its Dykstra
  fallback plus a re-standardize step, versus block-wise's one clip. More steps, same
  |r|>1 root cause, not fewer.
- **Inverse-Wishart variance anchor / diagonal-shrink (ADR 0033 decision 6 levers).**
  Already falsified: exps 0022-0024 (insight 0032 Findings 4-6) showed the IW prior is
  N-weighted and reaches neither spectral end, and `sigma_diag_shrink` fixes the
  min-eigenvalue end only by decorrelating, which itself triggers the max-eigenvalue
  variance runaway. Unit-diagonal caps the variance end directly and needs no tuning
  knob.
- **Multi-label (overlapping foreground groups).** Deferred, not chosen: with
  overlapping groups a document's allowed set can span two foreground blocks, so the
  cross-foreground block WOULD enter an E-step marginal and would need completion again.
  Single-label gating is the scope where block-wise-without-completion is exact. If
  multi-label is needed later, the completion path (ADR 0033) is the reference; the
  utilities are retained for it.

## Consequences

- **The variance runaway is structurally impossible.** No fit can inflate a prior
  variance because there is no free prior variance — Σ_ii ≡ 1. Validated locally
  (block-wise fit on a gated synthetic with a known unit-diagonal Σ_true recovers 14/14
  topics including the thin minority arm, max Σ_var = 1.0 throughout) and CONFIRMED on
  the real cohort by exp
  [0027](../experiments/0027-stm-comorbid-blockwise-unit-diagonal.md) (converged iter 52,
  ELBO −1.63e6, Σ_var pinned at 1, dementia sub-phenotypes preserved — insight
  [0034](../insights/0034-blockwise-unit-diagonal-fixes-runaway-on-real-data-and-needs-a-correlation-clamp.md)).
- **`pd_complete` leaves the fit path.** The per-iteration `M-step pd_complete: …s
  sweeps=…` driver log line and the `time`/`logging` plumbing are removed. The serial
  covariance-selection completion no longer runs while executors idle. The utility stays
  in `_linalg.py` for its isolation tests and any future multi-label use.
- **Correlation reporting is unchanged and now exact.** The fitted Σ IS a correlation
  matrix, so `topic_correlation(Σ) ≈ Σ`; `topic_correlation_identified` still NA's
  cross-group pairs (n_pairs = 0 under single-label) and reports within-group /
  background↔foreground pairs. No charmpheno-export or dashboard change was needed.
- **Narrows ADR 0033's multi-group decision to single-label.** Cross-group covariance
  from comorbid documents (ADR 0033 decision 4) is out of scope for the block-wise
  engine; a comorbid/multi-label fit would use the retained completion path.
- **No new dependency, negligible cost.** Standardize + `fill_diagonal` is O(K²) per
  M-step, strictly cheaper than the completion it replaces.

## References

- Blei, D. M. & Lafferty, J. D. (2007). "A Correlated Topic Model of Science."
  *Annals of Applied Statistics*, 1(1), 17–35. — the logistic-normal Σ whose diagonal
  this ADR pins to 1.
- Lewandowski, D., Kurowicka, D. & Joe, H. (2009). "Generating random correlation
  matrices based on vines and extended onion method." *Journal of Multivariate
  Analysis*, 100(9), 1989–2001. — the LKJ correlation-matrix parameterization
  (unit-diagonal Σ as the modeling object, variances handled separately).
- ADR [0033](0033-stm-full-covariance-sigma.md) — full-covariance Σ with PD completion;
  superseded for the gated single-label M-step by this ADR (see its amendment note).
- ADR [0027](0027-lazy-block-updates-for-gated-svi-mstep.md) — the lazy-keep invariant
  the unsupported-pair rule preserves.
- ADR [0031](0031-stm-k1-reference-topic-parameterization.md) — K−1 reference
  parameterization; the reference topic's diagonal is pinned to 1 like every other.
- insight [0029](../insights/0029-stm-sigma-init-collapse-blowup-missing-stabilizers.md)
  — σ-init collapse/blowup and the missing stabilizers.
- insight [0030](../insights/0030-spectral-init-closes-stm-sigma-blowup-on-real-data.md)
  — spectral init + small σ_init keeps Σ proper; scale-1 is load-bearing (the limit
  this ADR takes to the boundary).
- insight [0032](../insights/0032-gated-fullcov-recovers-dementia-subphenotypes-and-exposes-spd-assembly-conditioning.md)
  — the falsified IW / diag-shrink levers (Findings 4-6) and the reporting-artifact
  Resolution.
- insight [0033](../insights/0033-gated-fullcov-variance-runaway-is-an-init-identifiability-failure.md)
  — the runaway's two-ingredient diagnosis; unit-diagonal is the fix that kills
  ingredient 2 (free prior variance).
- insight [0034](../insights/0034-blockwise-unit-diagonal-fixes-runaway-on-real-data-and-needs-a-correlation-clamp.md)
  — exp 0027 real-cohort confirmation of this decision, and the empirical finding that
  per-cell standardization produces |r| > 1 on mismatched-support pairs (the clamp).
- [design spec](../superpowers/specs/2026-07-01-stm-blockwise-unit-diagonal-correlation-design.md)
  — the full design and local validation for this arc.
