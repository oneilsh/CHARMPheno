# 0072 — "PC is neutral" was a SCALING BUG: the supervised λ-correction vanishes as 1/λ² because it's a RAW gradient step where a NATURAL gradient belongs

**Date:** 2026-08-18
**Topic:** prediction-constrained, gated-PC, SVI, natural gradient, scale, bug

**Status:** PARTIAL — SUPERSEDED by insight 0073. This insight found a real scaling issue
(the natural-gradient direction, ADR 0044) but wrongly concluded it was THE cause of the
neutral-PC result. It was not even the dominant one: exps 0092–0095 showed `corr` still
bit-exact 0 AFTER this fix, because the actual upstream blocker was α-collapse killing the
CAVI Jacobian (`∂θ/∂eb → 2.7e-90`), with a head runaway and an additive-correction ELBO
detonation stacked behind it. See insight 0073 for the full three-bug story and ADR 0045
for the fixes. The `1/λ²` analysis below is still correct for the RAW-vs-natural gradient
question; it just wasn't what made PC read as neutral at scale.

## The observation

Across every whole-population gated-PC run, `weight_y > 0` produced topics
BIT-IDENTICAL to `weight_y = 0` — same ELBO (to 11 figures), same per-node AUC, same
domain λ-mass. A new per-iter diagnostic, `corr_relΔλ = ||λ_sup − λ_unsup|| /
||λ_unsup||`, read **exactly 0**. So the supervised topic correction was doing nothing,
and "PC is neutral" was the conclusion drawn four times.

## The mechanism (numerically confirmed)

The correction is `λ ← λ_unsup − ρ·wy·∂L_sup/∂λ`, with `∂L_sup/∂λ` obtained from the
probability-space autograd gradient via `_grad_topics_to_lambda`, which multiplies by
`polygamma(1, λ) = trigamma(λ) ≈ 1/λ`. So the update is `O(1/λ)` in count space,
subtracted from an `O(λ)` value — a **relative** move of `O(1/λ²)`:

| `Σλ_k` | `corr_relΔλ` |
|---|---|
| 50 | 2.7e-02 (moves topics — PC works) |
| 5,000 | 3.1e-06 |
| 500,000 | 3.1e-10 |
| 1.3e6 (this run) | 4.7e-11 (dead) |

Every 10× in λ shrinks the correction 100×. At small-corpus λ (~50) the correction is
2.7%; at whole-population λ (~1e6, the log's `Σλ_k` max) it underflows to nothing.
**This is the exact small-model-works / large-model-neutral split** — not a property of
the data.

## Why it's a bug, not a fundamental limit

The trigamma transform is the *correct* raw derivative `∂L_sup/∂λ` (it's FD-verified).
The defect is using a **raw** gradient step for the supervised term while the
**unsupervised** SVI update is a **natural** gradient (the sstats go straight into
`λ = η + corpus_scale·sstats`, no trigamma). The Dirichlet Fisher is `≈ trigamma(λ)`,
and

    natural_grad = Fisher⁻¹ · ∂L_sup/∂λ ≈ (∂L_sup/∂eb · eb · trigamma(λ)) / trigamma(λ)
                 = ∂L_sup/∂eb · eb        ← the trigamma CANCELS; scale-stable.

So the `1/λ²` vanishing is entirely an artifact of mixing a raw-gradient supervised
correction with a natural-gradient unsupervised update. `head_lr` is irrelevant to this
(exp 0091: full Newton head, `corr_relΔλ` still 0) — it's the topic side, not the head.

## The fix (ADR to follow)

Fold the supervised term into the sufficient statistics as a **natural gradient**,
corpus-scaled identically to the unsupervised counts, BEFORE the λ update. This
removes `_grad_topics_to_lambda` (the trigamma transform), `_corrected_lambda_block`
(the post-hoc subtract), AND the **`topic_trust`** knob (the per-cell trust-region cap
that only ever "masked the magnitude" of the mis-scaled correction). `weight_y` stays.
More principled, one fewer knob, less code. Guard with a small-λ-vs-large-λ regression
test proving `corr_relΔλ` is now scale-stable, and an FD check on the pseudo-count form.

## Consequences

- The neutral-PC conclusion (0064/0066/0089/0090/0091) is **not** a data truth — PC was
  never actually being tested at whole-population scale. Re-open the thesis after the fix.
- Insight 0069 ("ridge-bounded co-fit head is calibrated at scale, unified model
  works") was measured on tiny-K runs where λ was small enough that the correction fired;
  its scale claim needs re-checking after the fix.
- The head's own under-prediction (co-fit 0.52 vs oracle 0.68; needs an unpenalized
  intercept + convergence — exp 0090 ladder) is a SEPARATE axis (head prediction, not
  topic shaping), sequenced after this fix.
