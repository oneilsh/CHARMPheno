# Insight 0047 — The Σ eigmin-floor is a mean-field attenuation artifact: VI collapses the block-scatter rank (and scale), exact Gibbs recovers the full-rank truth; the fix is a better posterior, not a low-rank Σ

**Date:** 2026-07-12
**Branch:** pg-stm
**Resolves the open question in:** insight 0046 (is the low fitted rank a real low-rank signal or a mean-field artifact?). **Extends:** insight 0044 (mean-field reads the wrong Σ correlation sign) from *sign* to *rank / conditioning / scale*. **Relates to:** insights 0037/0038 (fit-anchored scales under-concentrate), the at-scale null (exp 0050/0051, eigmin at the 1e-8 floor). Queue item (2)→(3) bridge from the Fable exchange.

## Question

Insight 0046 showed the at-scale eigmin-at-floor is reproduced by *rank-deficient* block scatter, and insight 0044 says mean-field attenuation compresses the per-doc logits. So: is the low effective rank of the fitted block scatter a genuine low-rank comorbidity signal (→ low-rank Σ=ΛΛᵀ+D is the right model) or an **attenuation artifact** of mean-field VI (→ the fix is a better posterior)?

## Method

On a stick-native gated corpus whose TRUE block covariance is full-rank and well-conditioned (Σ_true block eigmin ≈ eta_scale·(1−rho) ≈ 2.8), compare the eigenspectrum of a group's [gate, fg] block (11-dim) across three estimates of the per-doc logit covariance:

1. **Truth** — Σ_true block (the generative covariance).
2. **VI mean-field** — empirical covariance of the fitted per-doc variational means `psi_mean` (attenuated).
3. **Gibbs fixed-β** — the assembled block from `pg_stm_gibbs(beta_fixed=…)` (labels pinned per insight 0045, so the comparison is not contaminated by label-switching), un-attenuated (sampled ψ).

Effective-rank metric: participation ratio PR = (Σλ)² / Σλ² (1..p). Corpus: D=1600 (800 group-A docs), K=30, bg_k=10, fg=10, eta_scale=4.0.

## Result — the rank collapse is a VI artifact, decisively

| estimate | PR (of 10) | eigmax | eigmin | cond |
|---|---|---|---|---|
| Truth (Σ_true block) | 5.52 | 14.8 | 2.80 | 5.3 |
| **VI mean-field** (psi_mean cov) | **2.37** | 6.8 | **2.8e-6** | **2.5e6** |
| **Gibbs fixed-β** (un-attenuated) | **6.71** | 26.3 | **3.59** | 7.3 |

Mean-field VI **halves the participation ratio** (5.5 → 2.4) and **crashes the minimum eigenvalue by six orders of magnitude** (2.80 → 2.8e-6, cond → 2.5e6) — reproducing the at-scale near-singularity on a corpus whose truth is well-conditioned. Exact Gibbs (labels pinned) **recovers the full-rank, well-conditioned truth** (PR 6.71, eigmin 3.59, cond 7.3). The background block shows the same pattern (VI PR 2.08 / eigmin 2.6e-6; Gibbs PR 6.63 / eigmin 4.0). So the low effective rank — hence the eigmin-at-floor — is a **mean-field attenuation artifact**, not a property of the data.

Note VI also **under-estimates scale** (eigmax 6.8 vs truth 14.8): attenuation shrinks the whole spectrum, consistent with the fit-anchored-scales-under-concentrate finding (insights 0037/0038). This suggests the at-scale VI max|Σ|≈6.5 is an *underestimate* of the true generative scale. (Gibbs eigmax 26.3 *over*-shoots the truth because the sampled-ψ scatter adds the per-doc posterior variance on top of the point scatter — an expected offset; the load-bearing comparison is rank/conditioning, where Gibbs is healthy and VI collapses.)

## What this changes — re-sequences the queue

1. **Low-rank Σ=ΛΛᵀ+D (item 3) would model the METHOD, not the data.** Fitting a low-rank factor to VI's attenuated scatter bakes in the artifact. Low-rank is therefore **not** the fix for the at-scale eigmin floor; it is at most an optional modeling choice for genuine structure, to be decided *after* the posterior is fixed. Demote item (3) from "the floor fix."

2. **The fix is a better posterior = the condition-on-VI-β Gibbs read-out (item 4).** VI's β is trustworthy (insight 0044); Gibbs's un-attenuated ψ,Σ recovers the full-rank, well-conditioned covariance. The item-4 architecture (condition on VI β, Gibbs-sample η and Σ) gets both. **Promote item (4) to next.** This diagnostic is direct evidence it will produce a properly-conditioned Σ, not just the right correlation sign.

3. **The whole Σ read-out is a mean-field casualty.** Insight 0044 (wrong correlation sign) + this (collapsed rank/conditioning, under-estimated scale) means correlation, conditioning, and scale of Σ are all unreliable under mean-field VI. β/topic content is unaffected. The comorbidity deliverable and the KG rare-disease thesis must read Σ off the Gibbs (or a structured/collapsed variational) posterior, never mean-field.

## Caveats

- Synthetic, well-specified corpus with a full-rank truth. On real data there may be a *genuine* low-rank component on top of the attenuation; this shows attenuation *alone* produces the collapse, not that no real low-rank signal exists. The clean way to separate them at scale is the item-4 Gibbs read-out's own eigmin vs VI's.
- The fixed-β Gibbs block over-estimates scale by the average per-doc posterior variance (sampled-ψ scatter = point scatter + posterior variance); the item-4 read-out should decide whether it reports the covariance of the posterior-mean logits or the full posterior covariance.
- eta_scale=4 and moderate correlations chosen; the qualitative collapse (VI PR≈2, eigmin≈1e-6 vs Gibbs full-rank) is robust, exact numbers config-specific.
