# ADR 0044 — The supervised λ-correction is a NATURAL gradient (not a raw-gradient step); retire `topic_trust`

**Date:** 2026-08-18
**Status:** Accepted
**Context:** insight 0072 (root cause + numerical confirmation); exps 0090/0091 (the
neutral-PC symptom at whole-population scale).

## Decision

Apply the supervised topic correction to λ as a **natural gradient** in the same space
as the unsupervised sufficient statistics, and **remove the `topic_trust`
trust-region knob**.

Per domain block (`OnlinePCLDA._corrected_lambda_block`):

    eb  = exp(ψ(λ_pre) − ψ(Σ_v λ_pre))                # expElogbeta at the grad's λ
    nat = grad_eb · eb                                # = ∂L/∂E[logβ], the natural grad
    λ  ← max(λ_unsup − ρ · weight_y · nat, 1e-30)

`grad_eb = grad_topics_stat` is corpus-scaled by the runner, matching the unsupervised
sstats that formed `λ_unsup`. The RM ρ step-damping bounds the step exactly as it does
the unsupervised natural-gradient step, so no per-cell trust clip is needed.

## Why

The previous code used the **raw** gradient `∂L/∂λ` — the trigamma transform
`_grad_topics_to_lambda` multiplies by `polygamma(1,λ) ≈ 1/λ` — then *subtracted* it
from λ. That is an `O(1/λ)` count-space move off an `O(λ)` value, i.e. an `O(1/λ²)`
**relative** move. At whole-population λ (`Σλ_k` up to 1.3e6) it is ~5e-11 — below
floating-point resolution — so `weight_y` was a silent no-op and gated topics came out
BIT-IDENTICAL to the unsupervised topics (`corr_relΔλ ≈ 0`). This produced the "PC is
neutral" reading of insights 0064/0066 and exps 0089/0090/0091.

For an exponential family, the natural gradient w.r.t. the natural parameter (λ) equals
the ordinary gradient w.r.t. the **mean** parameter (`E[logβ] = ψ(λ) − ψ(Σλ)`). With
`eb = exp(E[logβ])`, that gradient is `∂L/∂eb · eb` — which **cancels the trigamma**
and is scale-STABLE (numerically flat across 4+ decades of λ; a steady relative topic
move at any corpus size). This is also the correct SVI structure: the unsupervised
update is already a natural gradient (sstats → λ); the supervised term must join it in
the same space, not as a raw-gradient afterthought.

## Consequences

- **`topic_trust` is retired** (a knob removed). It only ever "masked the magnitude" of
  the mis-scaled raw correction. Still accepted by the constructor/shim for config
  compatibility, but ignored; the exp configs' `topic_trust` values are dead.
- **`weight_y` must be re-calibrated.** The old default (50) was compensating for a
  ≈0 correction; with the correction live, `weight_y` is a genuine O(1) supervised
  weight and 50 is likely far too strong (can floor rare-topic λ cells in one step).
  Re-swept starting from the fixed engine (exp 0092+).
- **`_grad_topics_to_lambda` is dead** on the fit path (kept for now with its
  finite-difference test of the raw-gradient identity; remove in a cleanup pass).
- The neutral-PC conclusion is re-opened: PC was never tested at scale. The head's own
  under-prediction (co-fit 0.52 vs oracle 0.68; needs an unpenalized intercept +
  convergence) is a SEPARATE axis, sequenced after this fix.
- Guarded by `test_supervised_correction_is_scale_stable_natural_gradient` (natural is
  flat where the old raw collapses) + the per-iter `corr_relΔλ` driver diagnostic.
