# 0044 — Marginalizing the held-out predictive does NOT fix the generative-scale holdout-drift: it is a low-dimensional fix that inverts to a dominant Laplace-approximation bias at production dimension, and the residual real-corpus drift is genuine per-document concentration heterogeneity (misspecification), not an estimator artifact — so the f-drift becomes a misspecification meter and the real fix is a per-document scale, not a better point-scale

**Date:** 2026-07-10
**Topic:** stm | generation | concentration | calibration | diagnostics | misspecification
**Status:** Confirmed (synthetic exp 0046 + controller bias/variance probe; real-corpus exp 0047 population_cancer). Supersedes the working hypothesis of the design spec `docs/superpowers/specs/2026-07-10-marginalized-heldout-scale-calibration-design.md`.

The exported generative scale c (eta_scale, Σ_gen = c·R) is calibrated by held-out
within-unit completion likelihood: hide a fraction f of each document's tokens, infer η
from the visible fraction at scale c, score the held-out tokens, sweep c, take the
smoothed argmax. The **MAP-plug-in** scorer's recovered c\* drifts with f (observed
4.61→3.65 across f on the real cancer corpus). The proposed fix (insight
[0037](0037-in-band-pooled-scale-converges-below-export-scale-faithful-scale-not-fit-recoverable.md)/[0038](0038-heldout-ll-recovers-true-concentration-and-lda-alpha-opt-is-not-hot.md)
lineage) was to **marginalize** the predictive over the per-doc Laplace posterior
(log-of-average of θ(η_s)·β over S draws η_s ~ Normal(η̂, H⁻¹)), which under a
well-specified model is f-independent and equals the true scale. This insight records
that the fix does NOT hold on the production problem, why, and the reframe it forces.

## Finding 1 — the fix works in low dimension and INVERTS at production dimension (synthetic, well-specified)

Plant-and-recover, β frozen at truth, single planted scale (level 5), sweep both scorers
at f ∈ {0.5, 0.7, 0.95} (exp 0046):

| regime | MAP c\* drift (max−min over f) | marginalized c\* drift |
|---|---|---|
| low-dim (K=8, V=400, len=60) | **0.83** (3.85→3.01) | **0.13** (flat ≈4.0) |
| production (K=60, V=5000, len=44) | **0.12** (flat ≈3.4) | **0.99** (2.60→3.59) |

The MAP regularization-drift artifact is a **low-dimensional, data-rich phenomenon**. At
K=60 with ~22 visible tokens over 60 components the point estimate is prior-dominated at
every f, so the artifact vanishes — and instead the *marginalized* estimator drifts,
because a Gaussian-at-the-mode is a poor approximation to a 60-dim logistic-normal
posterior informed by ~22 tokens (strongly non-Gaussian, skewed against the simplex
boundary, plausibly multi-basin), and that approximation error grows as f grows (less
visible data → wider, less-Gaussian truth), which is exactly what turns an approximation
error into an f-drift.

## Finding 2 — the production marginalized drift is BIAS, not Monte-Carlo variance

Controller probe (K=60, D=600): the marginalized drift is **stable across RNG seeds**
(1.06 vs 1.14 at S=128) and **grows with sample count S** (S=128→1.14, S=512→1.51). More
samples make it worse, converging to the *biased* Laplace-marginal rather than to truth.
"Stable across seeds, grows with S" is the clean signature separating bias from variance;
it kills the "just add samples" escape. (Vertex SEs from the quadratic smoother are
~0.005–0.01 in log c, so the curves are precise — a smooth biased objective, not a jittery
one.)

## Finding 3 — on the real (misspecified) corpus BOTH estimators drift, ~equally, in OPPOSITE directions

exp 0047 (re-export of the exp 0028 population_cancer gated STM fit; MAP sweep full-corpus,
marginalized sweep on a 967-doc sample at S=64), f ∈ {0.5, 0.8, 0.95}:

| f | MAP c\* (full) | MAP c\* (sample) | marginalized c\* (sample) |
|---|---|---|---|
| 0.5 | 4.61 | 5.30 | 2.36 |
| 0.8 | 3.75 | 3.90 | 2.65 |
| 0.95 | 3.65 | 3.80 | 3.76 |
| drift | 0.96 | 1.49 | 1.40 |

MAP drifts substantially on real data (unlike the well-specified K=60 synthetic where it
was flat), *decreasing* with f. Marginalization does not remove it: the marginalized drift
(1.40) ≈ the MAP drift (1.49) and runs **opposite** (increasing with f); the two cross near
f=0.95 (≈3.8) and diverge maximally at f=0.5 (5.30 vs 2.36). Marginalization trades one
drift for an equal-and-opposite one.

## Interpretation — the f-drift is a misspecification meter

The residual real-corpus drift is **genuine per-document concentration heterogeneity**: the
corpus mixes one/two-topic units with handful-of-topic units, and different conditioning
regimes (holdout fractions) weight that heterogeneity differently, so no single c is optimal
for all f under a prior that assumes a homogeneous scale. This is finding-4 (the
concentration-heterogeneity hypothesis) from the 2026-07-04/05 diagnostics, now cleanly
isolated from plug-in bias for the first time. The f-drift, which began as a nuisance to
engineer away, is actually the **observable signature of the misspecification** — and it is
the closest thing to ground truth this problem admits (a document's true concentration is
unobservable, but whether a heterogeneity-permitting model stops needing a different answer
per regime is observable).

## Consequences

- **Do not ship marginalization.** It is more expensive (S×), biased at production
  dimension (Findings 1–2), and does not reduce the real drift (Finding 3). Keep the
  MAP-plug-in scale. Importance-sampling de-bias (the parked Task 7) would only shave the
  Laplace term, which is not the real problem.
- **Not a model-family problem.** LDA has the identical defect — a symmetric/asymmetric
  Dirichlet is a single corpus-wide concentration dial, misspecified for a
  heterogeneous-concentration corpus in exactly the same way; the user's asym/sym LDA
  (Wallach–Mimno–McCallum) reads too peaky under held-out-LL, consistent with insight 0038's
  finding that LDA peakiness is co-fit-β sharpening, not a θ-prior effect. Switching families
  trades away gating/covariates/correlation (load-bearing, and what makes rare-community
  topics and the scale itself identifiable) to buy a model with the same missing piece.
- **The missing piece is orthogonal to family: a per-document concentration.** Minimal
  logistic-normal form: η_d ~ Normal(μ_d, s_d·c·R) with s_d a per-document latent scalar from
  a unit-median prior (inverse-gamma → a multivariate-t prior t_ν(μ_d, c·R)). Two
  hyperparameters (c median scale, ν heterogeneity), ν→∞ recovers the current model exactly
  (nesting, not replacement); R/gating/covariates survive untouched. s_d is the
  **best-identified latent in the model** (one scalar per doc informed by that doc's ~44
  tokens, pooled across ~48k docs) — the opposite corner of the identification table from the
  per-topic-variance runaways (insight 0033), hence runaway-immune, and can be introduced at
  inference/export/generation only (conditionally conjugate given η) with no refit. Calibrate
  (c, ν) in 2D with the existing held-out sweep. **Falsifiable prediction:** if concentration
  heterogeneity is the source of the f-drift, the drift shrinks at the calibrated (c, ν).
- **Burstiness confound — check BEFORE modeling heterogeneity.** Within-document token
  repetition (e.g. one hashtag repeated ten times) reads as high concentration under every
  model in this family, but is repeat-rate, not cross-topic concentration. The dedup-variant
  predictive-gain diagnostic already exists: measure how much apparent concentration
  heterogeneity survives deduplication. If much of the s_d spread is really repeat-rate
  spread, the principled fix is likelihood-side — a Dirichlet-compound-multinomial (Pólya-urn)
  emission (Madsen, Kauchak & Elkan 2005; Doyle & Elkan DCM-LDA), whose log-likelihood
  increments log(s·p_w + i) for the i-th repeat make each repeat count less (growth ~log(n_w),
  first occurrence undamped, damping governed by an estimated per-topic s) — NOT the ad-hoc
  count-dampening log(1+n), and NOT a heavy-tailed prior absorbing a likelihood pathology.
- **Disposition for what ships now:** calibrate at the f matching the tool's dominant
  conditioning regime (matched-f calibration), ship that single number with the drift band
  quoted (MAP-side ≈ 3.7–4.6), and do NOT expose a per-request scale (a query-dependent
  parameter is the hack the original "yuck" was warning about; a regime-matched single value
  inside a narrow band is the honest pseudo-true scale for the deployment regime). A companion
  consistency principle: **calibrate under the inference the tool actually runs** — if the
  completion path samples η from the Laplace posterior, the marginalized scorer is the
  self-consistent calibration for that simulator even though it is a biased estimator of the
  true scale; if the completion path uses the MAP mode, the MAP-calibrated scale matches.
  "Which estimator finds the true c" and "which c makes the shipped system behave" are
  different questions; under known misspecification the second is the one with an operational
  answer.
- **§5 "is 4.6 too high" — RESOLVED by the bias inversion (run 2026-07-10,
  `scripts/marginalized_scale_bias_inversion.py`, results
  `docs/experiments/0046-marginalized-scale-decomposition/bias-inversion.md`).** Planting at
  c ∈ {2, 3.5, 5, 7, 10} at the production regime and measuring ĉ_est(c_true) per f gives a
  clean, near-linear, f-stable, invertible **MAP** map (ĉ/c_true ≈ 0.6–0.85, a constant
  under-recovery) and a **SATURATED marginalized** map (at f=0.5 c_true 2→10 moves ĉ only
  2.12→2.51 — nearly uninvertible). Three results: (1) the two instruments do NOT reconcile
  (where marginalized is invertible at all they disagree 1.8–2.6×) — the napkin ratio-transfer
  was coincidental because the marginalized map is saturated, not linear; (2) the shipped raw
  MAP scale (4.6) is NOT too high — bias-corrected through the MAP instrument the true scale is
  **~5.4–7** (MAP_full inverts to 6.95/5.60/5.41 across f), *higher* than shipped and near the
  ~7.6 natural scale, so the marginalized "low" reading (2.36) was Laplace saturation, not a
  genuine lower-scale signal; (3) the bias-corrected scale STILL drifts across f (6.95→5.41,
  drift 1.55) — the genuine per-document concentration heterogeneity, now cleanly isolated from
  estimator bias and quantified (~1.5 corrected-scale units). So marginalization is doubly dead
  (biased AND uninvertible), the shipped scale is if anything conservative, and the
  heterogeneity is real. **Real-β transfer check (`scripts/realbeta_bias_transfer_check.py`):**
  re-measuring the MAP map under the DEPLOYED β (from the exp 0047 bundle, only β swapped)
  gives an under-recovery ratio of 0.63–0.80 that tracks the synthetic 0.66–0.86 in shape and
  magnitude — so the ~0.66 is estimator geometry, not a synthetic-β artifact, and the
  correction is EARNED with provenance under the deployed emission matrix (corrected scale
  ~6–7.3, converging with the 5–7 faithful band and 7.6 natural scale — three independent
  routes agree). **But the "ship a bias-corrected ~6" decision is demoted to the FALLBACK
  branch (per Fable):** because the residual f-drift is a genuine distribution of per-document
  scales, a single corrected c is just a different compromise point, not a recovered truth. On
  the main line the fix is the per-document scale (multivariate-t), and the bias-corrected
  inversion becomes its **validation target** — at the calibrated (c, ν) the model's own s_d
  spread should REPRODUCE the per-f implied scales (6.95/5.60/5.41), a sharper falsifiable test
  than "the drift shrinks." Sequence: (1) dedup/burstiness gate, (2) real-β bias map [done],
  (3) the gate-selected heterogeneity fit, (4) scale decision — ship (c, ν) on the main line,
  or the bias-corrected single-c only as fallback.

## What this does NOT claim

- It does not claim marginalization is wrong in general — it is correct where its
  assumptions hold (low-dim, data-rich; Finding 1 low-dim column), and the log-of-average
  primitive is validated and retained as a diagnostic.
- It does not measure a document's true concentration — the heterogeneity claim rests on the
  f-drift signature, and its verdict is the falsifiable drift-shrinkage test above, not a
  direct observation.
- It does not reopen a fit-time free variance (still falsified; insight 0033). The per-doc
  scale is proposed at inference/export first, precisely because s_d is well-identified where
  the per-topic variance was not.

**Related:** design spec + domain-agnostic findings report
`docs/superpowers/specs/2026-07-10-marginalized-heldout-scale-calibration-design.md`,
`docs/superpowers/specs/2026-07-10-marginalized-heldout-scale-findings-report.md`;
experiments 0046 (synthetic decomposition) and 0047 (real-corpus diagnostic); insights 0033
(gated runaway = document scarcity), 0037/0038 (held-out-LL scale calibration lineage this
qualifies), 0036 (export decoupling), ADR 0034 (unit-diagonal Σ). Instrument commits
32fdb9f..3aa961f on branch stm. References: Madsen, Kauchak & Elkan 2005 ("Modeling word
burstiness", ICML) and Doyle & Elkan (DCM-LDA) for the burstiness emission; Wallach, Mimno &
McCallum 2009 ("Rethinking LDA: Why Priors Matter") for the asym-θ/sym-β configuration.
