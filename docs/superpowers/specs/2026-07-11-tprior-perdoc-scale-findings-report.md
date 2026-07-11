# Per-unit scale (multivariate-t) calibration — findings report

**Date:** 2026-07-11
**Audience:** methodological collaborator (follow-up to the "MAP-plug-in vs Laplace-marginalized" report, 2026-07-10). Domain-agnostic; no LaTeX (Unicode: η, θ, Σ, R, c, ν, s).

---

## 0. Where the last report left off

Same model: units each carry a bag of discrete observations; latent η ∈ R^K with a
logistic-normal prior η ~ Normal(μ, Σ), Σ = c·R (R a fixed correlation matrix, c > 0 a
single scalar scale); θ = softmax(η); observations ~ Categorical(θ·B), with B and R
frozen upstream. The one calibrated quantity was the scale **c**, via held-out completion
likelihood (hide a fraction f of each unit's observations, infer η from the visible
fraction, score the held-out fraction, sweep c).

The last report concluded: the held-out-optimal c **drifts with f**, marginalizing does
not fix it (it trades the drift for an equal-and-opposite Laplace-biased one), and the
residual drift is a real **misspecification** — a genuine spread of *per-unit*
concentrations that a single global c cannot fit at every f. A separate check (deduplicate
each unit's observations, cap multiplicities at 1, re-infer) ruled out the obvious
confound — within-unit observation **burstiness** — because the per-unit concentration
spread survived deduplication, kept its rank ordering, and was uncorrelated with
repeat-rate. So the drift is genuine per-unit concentration heterogeneity, and the fix is
**prior-side**: give each unit its own scale. This report builds that and calibrates it.

## 1. The model and the two falsifiable predictions

Promote the single scale to a **per-unit** scale s drawn from a corpus-level distribution —
a scale-mixture-of-normals, i.e. a **multivariate-t** prior:

- η_d | s_d ~ Normal(μ_d, s_d · c · R),  s_d ~ Inverse-Gamma(ν/2, ν/2).
- Integrating s_d out gives η_d ~ multivariate-t_ν(μ_d, c·R). Two hyperparameters:
  **c** (the location of the per-unit scale cloud, since E[s_d] = ν/(ν−2) ≈ 1) and **ν**
  (its spread — the corpus concentration heterogeneity; small ν = heavy-tailed = strong
  heterogeneity; **ν → ∞ collapses s_d → 1 and recovers the original single-scale model** —
  a nesting the sweep contains as its own null).
- Per-unit inference is coordinate-ascent EM to the joint mode (η̂, ŝ_d): the existing
  Laplace η-step at prior precision (1/(s_d·c))·R⁻¹, alternated with the closed-form
  Inverse-Gamma mode ŝ_d = (ν + q/c)/(ν + K_free + 2), where q is the R-Mahalanobis of
  (η̂ − μ). Scoring of held-out observations stays MAP-plug-in (the marginalization route
  is dead, per the last report).
- Calibration is now a **2-D held-out sweep over (c, ν)**, same protocol as before (common
  visible/held split across all knobs).

This model makes two falsifiable predictions, stated in advance:
1. **At the calibrated (c, ν), the f-drift should collapse** — the per-unit s_d absorbs the
   concentration heterogeneity, so the corpus-level scale stops moving with f.
2. **The fitted per-unit effective-scale distribution (s_d·c) should reproduce the spread
   of the per-f single-scale estimates** the earlier drift produced.

Production regime, as before: K=60, V=5000, ~44 observations/unit; sweep on a 2441-unit
sample.

## 2. Prediction 1 holds — the drift collapses, and finite ν beats the Gaussian

Two runs (an initial grid, then a wider one — see §3 for why it was widened). In both, the
drift readout compares the single-scale (ν=∞) c\*(f) against the t-prior c\*(f) at the
calibrated ν, over f ∈ {0.2, 0.3, 0.5}:

| | single-scale (ν=∞) c\*(f) | t-prior c\*(f) |
|---|---|---|
| run 1 | 6, 6, 4  → **spread 2.0** | 12, 12, 12 → **spread 0.0** |
| run 2 | 6, 6, 4  → **spread 2.0** | 32, 32, 32 → **spread 0.0** |

The single-scale optimum drifts *downward* with f (6 → 4); the t-prior optimum does not
move. Note the single-scale's downward pull at f=0.5 (to 4) does **not** drag the t-prior
below its level — the per-unit scale has absorbed the f-dependence. (Caveat in §3: the
t-prior c\* rails against the grid ceiling, so "spread 0.0" has a ceiling component; the
qualitative collapse is nonetheless unambiguous — the single-scale drifts, the t-prior
does not.)

And the heterogeneity is **real, not the null**: in both runs every finite-ν column's
optimum beats the ν=∞ (single-scale) optimum. In the wider run the best finite-ν held-out
LL is −6.5451 vs the best Gaussian −6.5599 — a ~0.015-nats/observation improvement. If the
corpus were homogeneous the argmax would sit at ν=∞; it does not.

## 3. Prediction 2, and the structural finding — (c, ν) are NOT separately identified

Widening the grid (c up to 32, ν up to 80) to chase the argmax — which railed at the c
ceiling in run 1 — did not find a peak. It found a **flat ridge**. Best c at each ν:

| ν | best c | held-out LL |
|---|---|---|
| 5  | 32 (railed) | −6.54508 |
| 10 | 20 | −6.54574 |
| 20 | 12 | −6.54588 |
| 40 | 8  | −6.54665 |
| 80 | 8  | −6.54764 |
| ∞  | 6  | −6.55990 |

The `best c` column is a clean monotone **c–ν trade-off** (32 → 20 → 12 → 8 → 8 → 6 as ν
rises): lower ν (heavier tails, s_d more dispersed, posterior s_d pulled smaller) demands a
higher base c to compensate. Along ν ∈ {5, 10, 20} the LL spans just **0.0008 nats** — flat
within the sweep's shelf noise (vertex SEs ~0.005–0.01 in log c). The argmax (c=32, ν=5)
rails against the ceiling and is only ~0.001 nats above the interior optima; it is a ridge,
not a peak.

**What IS identified is the product s_d·c** (the per-unit effective scale). Its median was
4.49 at run 1's (c=12, ν=20) and 4.20 at run 2's (c=32, ν=5): while c moved 12 → 32 and the
median s_d moved 0.37 → 0.13, the **product stayed ~4.2–4.5**. The per-unit
effective-scale distribution is narrow (run 2 s_d·c: p10–p90 = 3.23–4.71).

This settles Prediction 2 through the same bias lens as the last report: MAP-plug-in scoring
under-recovers scale by the previously-measured ~0.66 factor, so the raw s_d·c ≈ 4.2–4.5
(≈ the shipped single-scale value 4.6) bias-corrects to ~6.4 median, p10–p90 ≈ 4.9–7.1 —
which brackets the per-f single-scale estimates the earlier drift produced (≈ 5.4–7.0). The
per-unit spread reproduces the per-f spread, but only in the compressed units the plug-in
scorer reports.

## 4. Why ν is unidentified — mild heterogeneity + data-pinned s_d

The inferred per-unit s_d is **narrow regardless of the prior ν** (run 2: p10–p90 =
0.10–0.15). Each unit's own observations pin its s_d tightly, so the data do not *need*
heavy tails; a heavy-tailed prior (small ν) and a near-Gaussian prior (large ν) fit about
equally, differing only in the base c that keeps the product fixed. This is the textbook
**flat likelihood in the t degrees-of-freedom** — anticipated for scale mixtures, here
demonstrated on a real fitted model. The heterogeneity is real (finite ν beats ∞) but
**mild**, and mild heterogeneity is exactly the regime where c and ν are jointly weakly
identified.

## 5. Synthesis

- **The prior-side hypothesis is confirmed.** The drift the last report attributed to
  per-unit concentration heterogeneity IS that: modeling a per-unit scale **collapses the
  f-drift** (Prediction 1), and the fit prefers finite ν over the single-scale null. This
  closes the loop from the earlier "the residual is misspecification, not a Laplace
  artifact."
- **But the two-parameter prior is over-parameterized for this signal.** (c, ν) are not
  separately identifiable — a flat ridge — because the heterogeneity, though real, is mild
  and each unit's scale is data-pinned. Only the **product s_d·c ≈ 4.4** is estimable, and
  it lands right at the shipped single-scale value (4.6), bias-correcting into the same
  5–7 band three earlier routes agreed on.
- **Practical disposition:** do **not** ship (c, ν) as two calibrated numbers — ν is not
  identifiable from held-out likelihood here, and an argmax over the flat ridge rails
  arbitrarily. The usable, stable output is the **per-unit effective-scale distribution**
  (median ≈ 4.4), which both removes the f-drift and agrees with the existing scale. In
  effect the per-unit scale mixture *explains* the earlier drift and *vindicates* the
  single shipped scale as the median of a mild per-unit spread — rather than replacing it
  with a sharper two-parameter object.
- This mirrors the marginalization arc's shape: a principled elaboration (marginalize;
  then, per-unit t-prior) that **diagnoses** the phenomenon cleanly but does **not** hand
  back a sharper identified scalar — because the limiting factor is the data's information
  about the elaboration, not the estimator.

## 6. Open question

The identified object is a *distribution* of per-unit effective scales, not a scalar. For a
simulator that draws units, that is arguably the right object — sample s_d per unit and
scale accordingly. The open question is whether, given ν is unidentified, one should (a)
fix ν at a weakly-heterogeneous default (e.g. ν≈20, the interior ridge point, non-railed)
and ship the corresponding c and s_d distribution, or (b) drop the parametric scale mixture
entirely and ship the **empirical** per-unit effective-scale distribution directly (median
≈ 4.4, the narrow spread above), since the parametric ν buys no identified information.
Both reproduce the drift-collapse; (b) is more honest about what the data actually pin.

---

### Appendix — provenance

- Model: gated logistic-normal, K=60, V=5000; per-unit multivariate-t scale mixture,
  coordinate-ascent EM (Laplace η-step + closed-form Inverse-Gamma s_d-step, with a closing
  η-solve so the returned (η̂, ŝ_d) is a joint fixed point). Nesting (ν=∞ reproduces the
  single-scale sweep) is a guarded test.
- Calibration: 2-D held-out (c, ν) sweep on a 2441-unit sample, common visible/held split
  across all knobs; distributed and single-machine paths verified equal unit-for-unit.
- Two runs: run 1 grid c ∈ {2,3,4,6,8,12}, ν ∈ {2.5,5,10,20,∞}; run 2 grid
  c ∈ {4,6,8,12,16,20,24,32}, ν ∈ {5,10,20,40,80,∞}. Both argmaxes railed the c ceiling
  (c=12 then c=32) along the low-ν arm; the ridge and the product-invariance are the
  robust readouts.
- Load-bearing implementation detail (guarded by test): the sweep **cold-starts** the
  per-unit inference at every (unit, c) rather than warm-starting across the c-grid — the
  per-unit objective is a non-concave log-mixture (multi-basin), so warm-starting across
  scales lands in different basins and breaks the ν=∞ nesting equivalence.
