# Insight 0046 — The at-scale Σ eigmin-at-floor is statistical (rank-deficient / compressed block scatter), not structural to the gated stick parameterization; low-rank Σ=ΛΛᵀ+D is the matched fix, and the IW ridge may lift the floor (a possible estimator contrast the max|Σ| headline missed)

**Date:** 2026-07-12
**Branch:** pg-stm
**Relates to:** the at-scale null result (exp 0050 iw / 0051 mle: both plateau max|Σ|≈6.5, PD, **eigmin≈1e-8 at the PD floor**), insight 0044 (mean-field attenuation compresses the per-doc logits), ADR 0034 (block-Σ / pd_complete), insight 0045 (free-Gibbs probe). Queue item (2) from the Fable exchange: diagnose the near-singularity before prescribing (low-rank Σ).

## Question

At scale the assembled block Σ has eigmin pinned at the PD floor (~1e-8) on **both**
the un-regularized MLE and the inverse-Wishart arm. Is that near-singularity
**structural** to the gated nested-stick parameterization + the max-determinant PD
completion of the never-co-active group×group′ cross-blocks (→ low-rank Σ=ΛΛᵀ+D is the
right fix, the rank deficiency is inherent), or **statistical** near-singularity from
the fitted data (→ a prior / better inference could resolve it)?

## Method

`assemble_sigma`'s only unobserved entries are the group×group′ foreground+gate
cross-blocks, filled by `pd_complete` (the max-determinant = **least-singular**
completion). If even that completion is near-singular, no PD completion is
well-conditioned → the *observed* blocks over-constrain the matrix = structural. The
generator `gated_ln_corpus_stick` builds Σ_true by the same construction, so the
structural floor is inspectable directly on Σ_true (no fit). Then two mechanism tests
feed `assemble_sigma` controlled scatter.

## Finding 1 — NOT structural: Σ_true is well-conditioned at moderate correlations

The gated nested-stick + pd_complete Σ_true is nowhere near singular:

| config | K−1 | eigmin | cond |
|---|---|---|---|
| at-scale-like (bg30, 2 groups × fg10) | 49 | **2.80** | 17.8 |
| bg8, 2 groups × fg4 | 15 | 2.80 | 5.8 |
| bg8, **5** groups × fg4 | 27 | 2.80 | 7.9 |

The minimum eigenvalue is `eta_scale·(1 − rho_grp) = 4·0.7 = 2.80` — the **within-group
foreground-contrast** direction — and it is **invariant to the number of groups** and
**not** a cross-group direction (the floor eigenvector's mass splits across groups with
the same sign, CROSS-GROUP=False). So the multi-group max-det completion does *not*
create the near-singularity; the pd_complete of never-co-active cross-blocks is benign
at these correlation levels. **The structural hypothesis is refuted.**

## Finding 2 — statistical: rank-deficient / compressed block scatter reproduces the exact 1e-8 floor

Two data-side mechanisms, feeding `assemble_sigma` controlled scatter:

**(a) Strong correlation** shrinks eigmin smoothly but cannot reach the floor alone:
eigmin = 2.22 → 0.90 → 0.20 → **0.040** as ρ (all blocks) = 0.3 → 0.7 → 0.95 → 0.99.
Even ρ=0.99 lands at 0.04, not 1e-8. Real strong comorbidity contributes conditioning
loss but is not sufficient.

**(b) Rank-deficient block scatter is the smoking gun.** Compressing each foreground
block's scatter to rank-2 (a proxy for collinear / attenuated per-doc logits) collapses
the assembled **MLE** Σ eigmin to **exactly 1.0e-8** (cond 2.5e9) — the at-scale
signature reproduced precisely. The hard floor is `nearest_spd(floor=1e-8)` /
`_jitter_to_pd(jit=1e-8)` clamping the zero-variance directions.

**Interpretation:** the at-scale eigmin-at-floor is a **rank deficiency in the fitted
block scatter**, i.e. the fitted per-doc logits are (near-)collinear so the block
covariance estimate has low effective rank. This is exactly what mean-field
**attenuation** (insight 0044) produces — the per-doc posterior logits are shrunk
toward the covariate mean, collapsing their scatter onto a low-dimensional subspace.
Whether the real low rank is a *true* low-rank comorbidity signal or an *attenuation
artifact* is not resolved here (needs the real Σ, or a Gibbs-vs-VI scatter-rank
comparison — the VI scatter would be artificially compressed, the exact-Gibbs scatter
would not).

## Finding 3 — the IW ridge lifts the floor (small-block shrinkage) — a possible estimator contrast

On the same rank-deficient scatter, the inverse-Wishart arm does **not** floor: its
diagonal prior ridge gives the zero-variance directions variance
≈ Psi0_scale/(nu0+n−p−1):

| estimator on rank-deficient scatter | eigmin |
|---|---|
| MLE (scatter/n) | 1.0e-8 (clamped) |
| IW, Psi0_scale=1 | 2.0e-3 |
| IW, Psi0_scale=10 | 2.0e-2 |
| IW, Psi0_scale=50 | 9.9e-2 |

This is Fable's cited full-Bayes merit (small-block / rank-deficient shrinkage) shown
concretely. **It complicates the "estimator contrast is null" headline:** the at-scale
max|Σ| was identical across arms, but the *conditioning* (eigmin) should differ — the
IW ridge should have lifted eigmin off the 1e-8 floor. The report recorded both arms at
≈1e-8; that either (i) under-resolved the IW eigmin, (ii) reflects a real scatter so
large that Psi0_scale=1 is negligible, or (iii) a downstream clamp masked it. **Open
check (needs the exp 0050/0051 Σ npz from the cluster):** is exp 0050 (iw) eigmin
actually above the floor? If so, the estimator contrast is *not* null on conditioning,
only on max scale.

## What this changes

1. **Low-rank Σ=ΛΛᵀ+D (item 3) is the matched fix — for the right reason.** The near-
   singularity is a low effective rank in the fitted covariance, so a low-rank factor
   ΛΛᵀ plus a proper learned diagonal floor D directly models the signal and supplies a
   principled PD floor, replacing the artificial 1e-8 jitter. It is not a repair for a
   structurally rank-deficient parameterization (there is none) but a match to the
   data's fitted rank.
2. **Re-examine the estimator contrast on conditioning, not just scale.** Pull the real
   Σ and compare eigmin(iw) vs eigmin(mle); the IW ridge may be a genuine merit the
   max|Σ| headline missed. Feeds Fable's Q1 (the case for full-Bayes IW).
3. **Diagnose the rank source (feeds items 3/4).** Compare the *rank* of the VI block
   scatter vs the exact-Gibbs block scatter on the same corpus: if VI is compressed and
   Gibbs is not, the low rank is partly an attenuation artifact and the fix is a better
   posterior; if both are low-rank, the signal is genuinely low-rank and ΛΛᵀ+D is right.

## Caveats

- Synthetic throughout; the real at-scale Σ was not available locally (cluster npz).
- The rank-2 compression is a *proxy* for attenuation/collinearity, not a measured VI
  scatter rank.
- eta_scale=4 and the moderate planted correlations are chosen; the qualitative
  separation (structural healthy vs statistical floor) is robust but exact numbers are
  config-specific.
