# insight 0073 — "PC is neutral at whole-Mondo scale" was a stack of THREE scale bugs, not a data truth

**Date:** 2026-08-19
**Status:** Active (supersedes the neutral-PC reading of insights 0064/0066/0072 at scale)
**Context:** exps 0090–0098 (Mondo cardiovascular branch, C=437 nodes, K=444 gated
topics, full AoU population). Refines insight 0072 (which found only the FIRST of the
three bugs). Engine fixes recorded in ADR 0045.

## The claim

Across exps 0089–0091 the supervised Prediction-Constrained (PC) correction produced
`corr_relΔλ ≈ 0` and `gated_pc` topics BIT-IDENTICAL to `unsup_gated` — read as "PC
supervision is neutral at scale." That reading was WRONG. It was three independent
**scale-dependent numerical bugs** stacked in series, each of which alone zeroes or
destabilizes the supervised signal. None is a property of the data. Fixed one at a time,
the method becomes a stable, shaping PC fit at whole-Mondo scale (exp 0098: `corr` a
healthy 2–3%/step, ELBO rising, `|w|` bounded).

Each bug has a distinct diagnostic signature, and they were only separable because we
added the `||grad_y||`, `|w_CK|max`, and `corr_relΔλ` per-iter trajectory diagnostics —
the aggregate readout hid all three as a single flat "Δ≈0."

## The three bugs (in the order the gradient flows, upstream → downstream)

### 1. α-collapse kills the shaping path (`corr` bit-exact 0)

The gated engine hard-coded the doc-topic Dirichlet `α = 1/K` and ignored
`docConcentration`. At whole-Mondo K (`1/444 = 0.0022`), the doc-topic posterior θ
collapses so hard that the **differentiable CAVI Jacobian `∂θ/∂eb` UNDERFLOWS**
(`ψ(0.0022) ≈ −455` underflows the unroll's `exp`): autograd measured
`||∂θ/∂eb|| = 2.7e-90` at α=0.0022 — matching the observed `||grad_y|| ~ 1e-84`. The
supervised gradient `grad_topics = (∂loss/∂θ)·(∂θ/∂eb)` therefore cannot flow back to the
topics AT ALL — upstream of the head, the ridge, everything. A second collapse axis:
`grad_cavi_iters` (ni=30 kills the Jacobian even at α≤0.1); the safe zone is **α ≳ 0.5**
at any ni. Local runs only stayed alive because they used α=0.05 + ni=8. **Signature:**
`corr` and `||grad_y||` bit-exact / underflow-zero at every iter despite a trained head.
**Diagnosed:** exp 0094 refuted head-saturation (bounding `|w|` 273→27 left `||grad_y||`
dead), then autograd on `_cavi_theta_anp` showed the Jacobian-vs-α cliff.

### 2. The newton head runs away (`|w| → 2.4e5`)

Once α=0.5 revived shaping (exp 0095), the co-fit newton head jumped `|w|` 113 → 5133 →
2.38e5 in three iters. The newton solve used the CORPUS-SUMMED gradient/Fisher with an
ABSOLUTE `head_l2` ridge; at ~110k docs/batch the ridge is negligible against the summed
gradient, so the step is the unregularized MLE and `|w|` runs to the logistic tail.
`head_l2`/`head_lr` were silently corpus-dependent. **Signature:** `|w_CK|max` climbing
past ~100 into the thousands within a few iters. **Fix:** per-doc-mean newton (÷ n_docs)
makes `head_l2` a scale-invariant ridge — `|w|` settles at `~|g_mean|/head_l2`,
corpus-independent (the reference's `rescale_total_loss_by_n_tokens`, for the per-doc
head).

### 3. The additive λ-correction detonates the ELBO (`−4.5e27`)

With the head bounded, pushing `weight_y` up (exp 0098, wy=16) drove `corr` to 9%/step and
the ELBO to `−4.5e27`; some topics were STARVED (`Σλ_k min → 12.9`) while others bloated
(`1e7`). The correction was an ADDITIVE subtraction on λ (`λ_unsup − ρ·wy·nat`) with no
simplex constraint, so a large `weight_y` drains a topic's pseudocounts toward 0, and an
empty topic's `E[logβ] → −∞` detonates the ELBO. `weight_y` was a raw, unbounded multiplier
on a move that can leave the valid region. **Signature:** `corr` spiking >~5%/step, ELBO
→ astronomically negative, `Σλ_k` min→tiny / max→huge, while `|w|` stays bounded. **Fix:**
exponentiated-gradient, mass-preserving correction (the reference's simplex-safe update):
multiplicative step, renormalize each topic to its unsupervised total mass — `weight_y`
bounded BY CONSTRUCTION (ADR 0045).

## Why it read as "neutral" and not "broken"

Bug 1 makes the supervised and unsupervised topics BIT-IDENTICAL (grad ≡ 0), which looks
exactly like "supervision doesn't matter" rather than "supervision can't reach the
topics." Bugs 2 and 3 only surface AFTER bug 1 is fixed (you need a live gradient before
the head can run away, and a bounded head before you can push `weight_y` into the
detonation regime). So the three were strictly serial — each invisible until the prior was
fixed — which is why the neutral reading survived multiple runs (0089/0090/0091) and even
a partial fix (0072's natural-gradient, which addressed a fourth, milder scaling issue but
not any of these three).

## Consequences

- The neutral-PC conclusion is RETRACTED at scale. Whether supervision *usefully* lifts the
  readout over the already-strong gated baseline is now a live, testable question (the gate
  itself aligns topics to labels; PC's marginal room is the open empirical question — exp
  0098+ and the per-node rarity split).
- Method-building lesson: at whole-population scale, **every hyperparameter that is `1/K`,
  absolute, or a raw multiplier is a latent scale bug.** α (=1/K), head_l2 (absolute), and
  weight_y (raw) each broke. The fixes make each SCALE-INVARIANT (α floored to a prior
  constant; head ridge per-doc; correction simplex-normalized).
- Diagnostics-building lesson: the per-iter `||grad_y||` / `|w_CK|max` / `corr_relΔλ`
  trajectory (not the aggregate readout) was what separated three bugs that all present as
  one flat delta. Keep them.
