# 0023 — STM inference: two-step Laplace + stochastic-EM mini-batch blending

**Status:** Accepted
**Date:** 2026-05-29
**Related:** ADR 0022 (STM-prevalence as covariate path); design spec [2026-05-29 STM prevalence-only design](../superpowers/specs/2026-05-29-stm-prevalence-design.md); ADR 0005 (LDA concentration optimization — establishes the natural-gradient-blended-with-hyperparameter-update pattern STM extends)

## Context

STM-prevalence (ADR 0022) replaces the Dirichlet prior on θ_d with a logistic-normal prior: η_d ~ N(Γ x_d, Σ); θ_d = softmax(η_d). The softmax couples to the multinomial likelihood non-conjugately, so the per-document update no longer has the closed-form CAVI structure that LDA enjoys. The M-step gains a regression of η_d's variational means on x_d (for Γ) plus a residual-covariance estimate (for Σ). Multiple inference choices need to be locked down before implementation:

1. **How to find the per-doc MAP point η̂_d.** Quasi-Newton (L-BFGS) vs full Newton vs other.
2. **How to compute the Laplace covariance ν_d at η̂_d.** Use the optimizer's running inverse-Hessian approximation, or evaluate the analytic Hessian once at the MAP.
3. **How to blend the M-step updates for Γ and Σ in mini-batch SVI.** Γ and Σ aren't natural parameters of an exponential-family q distribution, so the canonical natural-gradient ρ-blending derivation doesn't apply directly. Choose between full-batch only, full-batch as v1 with mini-batch deferred, or mini-batch with stochastic-EM blending.
4. **Whether to warm-start the per-doc inner loop** across outer iterations. Warm-start is cheaper per outer iter but requires persisting per-doc state, which collides with the stateless `local_update` contract that mini-batch sampling assumes.

These choices are independent of the row-type or covariate-plumbing choices (ADRs 0024, 0025) but they propagate into both the engine's contract and the per-doc cost factor that determines whether STM is viable at scale.

## Decision

### 1. Per-doc MAP via L-BFGS

The per-doc inner loop uses scipy `minimize(method='L-BFGS-B')` with an analytic gradient, cold-started at η_d = 0 each outer iteration. Typical inner iteration count is 20–30 per doc per outer iter at K ~ 80; per-iter gradient cost is O(K + V_d · K), comparable to one LDA CAVI iter.

Cold-start (rather than warm-start) is chosen so the `local_update` contract stays stateless — no per-doc variational state persists across outer iterations. This preserves mini-batch sampling semantics (any doc can be sampled independently on any iteration without inconsistency) and matches the same convention `OnlineLDA` uses for its CAVI initialization today.

### 2. Analytic Hessian at η̂_d for Laplace covariance ν_d

After L-BFGS converges to η̂_d, the analytic Hessian `H(η̂_d) = ∇²L(η̂_d)` is evaluated once in closed form (prior contribution `Σ⁻¹` plus likelihood contribution involving `diag(p) - p pᵀ` where `p = softmax(η̂_d)`, weighted by token counts). The Laplace covariance is `ν_d = (-H)⁻¹`.

We do **not** use L-BFGS's running inverse-Hessian approximation `H̃⁻¹` as ν_d. The L-BFGS approximation is built from gradient-difference history along the directions L-BFGS happened to explore during optimization; its accuracy at the converged point has nothing to do with the second-order expansion that Laplace's approximation requires. Evaluating the analytic Hessian once at the MAP costs ~5% of the per-doc total (one K² evaluation versus L-BFGS's 20–30 gradient evaluations) and gives the exact second derivative.

This is the canonical "two-step Laplace" pattern: find the mode with whatever optimizer is cheapest, then separately compute the analytic Hessian at the mode for the covariance. The R `stm` package uses this pattern; PyMC's `Laplace`, blackjax's Laplace primitives, and most variational-Laplace implementations follow the same recipe.

### 3. M-step: β natural-gradient SVI, Γ and Σ via stochastic-EM ρ-blending

The M-step has three pieces:

- **β** (Dirichlet variational parameter λ on topic-word). Unchanged from `OnlineLDA`: the closed-form natural-gradient SVI step `λ_new = (1 - ρ) · λ + ρ · λ̂_target`. Natural-parameter form, canonical natural gradient.

- **Γ** (P × K regression coefficient matrix). Closed-form ridge regression on aggregated cross-products: `Γ̂_target = (XᵀX + λ_ridge I)⁻¹ XᵀMu`, where `XᵀX` and `XᵀMu` are summed additively across partitions. Default `λ_ridge = 1e-6` for numerical stability against zero-variance or collinear columns. Blended with ρ in mini-batch mode: `Γ_new = (1 - ρ) · Γ + ρ · Γ̂_target`.

- **Σ** (K-diagonal residual covariance). Diagonal sample covariance of (μ_d - Γ x_d), with the Laplace variance correction `+ diag(ν_d)` added. Same ρ-blending: `Σ_new = (1 - ρ) · Σ + ρ · Σ̂_target`.

Γ and Σ are **not natural parameters** of an exponential-family variational q distribution in STM's formulation — they are hyperparameters of the prior on η_d, learned by an M-step rather than coordinate ascent on a q distribution. Their closed-form maximum-likelihood targets do not derive from a Fisher-information geometry. The ρ-blended updates therefore are not "natural gradient SVI" in the strict sense; they are **stochastic approximation of the M-step** (a.k.a. online EM; Cappé & Moulines 2009 establishes Robbins-Monro convergence for this pattern).

This is the same situation `OnlineLDA` already has with α: α is a hyperparameter learned by a Newton-step M-step, ρ-blended in the same outer loop as λ's natural-gradient SVI. The blending shape is identical; the theoretical-pedigree label differs. ADR 0005 introduced this duality at the LDA level; STM extends it from "K-vector α + Newton step" to "P×K matrix Γ + closed-form OLS step + K-diag Σ + closed-form sample-cov step."

### 4. Mini-batch convergence is to a neighborhood, not identity

Stochastic-EM with a Robbins-Monro ρ schedule converges to a *neighborhood* of the full-batch optimum, not to the same point. This is the same posture LDA has today — mini-batch λ does not match full-batch λ identically — and the same posture all SVI-style models in the framework inherit.

Validation criterion for mini-batch STM (phase 2 of the implementation in the design spec) is qualitative agreement, not identity:

- ELBO at convergence within ~1% of full-batch.
- β: top-N tokens per topic substantially overlap; high topic-level cosine similarity.
- Γ̂: signs agree with full-batch; magnitudes within a small constant factor.

A failure mode that would force mini-batch off the v1 release: Γ̂ flipping signs vs full-batch on the validation corpus, or β topics fragmenting unrecognizably. If that happens, STM ships full-batch-only and we open a separate investigation.

## Alternatives considered

1. **Newton's method for MAP (instead of L-BFGS).** Per-iter computes the analytic Hessian and solves a K-dim linear system; quadratic convergence near the optimum, typically 5–10 inner iters vs L-BFGS's 20–30. Rejected for v1 because the per-iter K² Hessian build plus K³ linear solve dominates: at K = 80, V_d = 20 (typical), a Newton iter is ~100× more expensive per doc than an L-BFGS iter. The analytic Hessian at the converged point — which is what Laplace's approximation actually needs — is computed once anyway (decision 2 above), so paying for it per iter buys no accuracy. Worth revisiting only if L-BFGS proves inadequate for convergence.

2. **L-BFGS's running H̃⁻¹ as ν_d.** Lazier than evaluating the analytic Hessian. Rejected because the approximation can be quite rough along directions L-BFGS didn't traverse, with no relationship to the second-order expansion Laplace's approximation specifies. The analytic Hessian costs ~5% of per-doc total; not worth the corner-cutting.

3. **Full Newton's method + analytic Hessian everywhere.** Combines alternatives 1 and 2; rejected for the same per-iter cost reason as 1.

4. **Full-batch STM only (no mini-batch).** Viable fallback if mini-batch convergence agreement fails empirically. Useful for moderately-sized cohorts (≲1M patients fits on a beefy worker) but does not match what `spark-vi` is for — scale via mini-batch SVI is the framework's raison d'être. Decision: ship mini-batch in v1 and fall back to full-batch-only only if phase 2 validation breaks.

5. **Per-doc warm-starting of L-BFGS** (persist (μ_d, ν_d) per doc across outer iters; L-BFGS resumes from previous outer iter's solution). Would cut per-doc inner iters substantially. Rejected for v1 because it requires per-doc state to persist across outer iterations, which:
   - Breaks the stateless `local_update` contract (each call would need to read and write per-doc state).
   - Conflicts with mini-batch sampling semantics (sampled docs would need their previous state; unsampled docs would carry stale state).
   - Would require a corpus-sized per-doc state side table, growing storage requirements.
   Tracked as a v1.x optimization to revisit only if per-doc cost proves blocking. R `stm` warm-starts, but R `stm` is single-process and full-batch.

6. **Full K × K Σ** (vs K-diagonal). Captures topic-correlation structure in the prior. Larger sufficient stats (K² instead of K) and a non-trivial M-step (Wishart-style covariance estimation with the Laplace ν_d correction). Rejected for v1 because R `stm` package defaults to K-diagonal and the literature supports it; K-diagonal also keeps the per-partition residual stat small. Revisit if residual diagonal analysis shows off-diagonal mass worth modeling.

7. **Diagonal-only optimization for the per-doc Laplace** (treat ν_d as K-vector by setting H to its diagonal). Skips the K × K solve in ν_d = (-H)⁻¹. Rejected: the analytic K × K Hessian is cheap to invert at K ~ 80 (~K³/3 = ~170K ops), and using only the diagonal would degrade the ELBO's accuracy in the doc-level KL term (`KL(N(μ_d, ν_d) || N(Γx_d, Σ))`) in a way that's hard to predict.

## Consequences

- **Per-doc cost factor** vs LDA's CAVI is claimed at ~2–3× per outer iteration; phase-1 implementation benchmarks this on real corpora. The 2–3× factor is the basis for declaring STM viable at scale. If empirical measurement shows substantially higher (say 10×), the warm-start question (alternative 5) gets reopened.

- **The stateless `local_update` contract is preserved.** No per-doc state persists across outer iters. Mini-batch sampling semantics are the same as LDA's today.

- **Mini-batch SVI is part of v1.** Full-batch is a fallback path if phase 2 validation surfaces a problem. The recipe is committed; the empirical validation is the genuine risk in the design spec.

- **Stochastic-EM framing is explicit in code comments and docstrings.** Γ and Σ updates look like SVI updates structurally, but the theoretical justification is stochastic-EM (Cappé & Moulines 2009), not natural-gradient SVI in the strict Fisher sense. ADR 0005 already established this duality at the LDA level for α; STM inherits and extends it.

- **The two-step Laplace pattern (L-BFGS + analytic Hessian at MAP) is the canonical recipe** documented in the engine. Anyone refactoring `_stm_doc_inference` should preserve the separation between MAP-finding (optimizer's job, approximation OK) and covariance computation (analytic Hessian required).

## Open follow-ups

- Phase 1 benchmark of L-BFGS per-doc cost factor vs LDA's CAVI on real OMOP cohorts. The ~2–3× claim is a complexity-analysis estimate, not a measurement.
- Phase 2 empirical validation of mini-batch vs full-batch convergence agreement. Recorded as a risk in the design spec; the criterion is qualitative agreement, not identity.
- Decision (deferred until empirical: K-diagonal Σ residuals show off-diagonal structure worth capturing) on whether v1.x should add full K × K Σ.
- Decision (deferred until per-doc cost proves blocking) on whether to add per-doc warm-starting; would require revising the stateless `local_update` contract.
