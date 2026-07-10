# Held-out scale calibration: MAP-plug-in vs Laplace-marginalized — findings report

**Date:** 2026-07-10
**Audience:** methodological collaborator (follow-up to the earlier "the sweep's c drifts with holdout fraction" diagnosis). Domain-agnostic; no LaTeX (Unicode: η, θ, Σ, R, c).

---

## 0. Setup (general terms)

A latent-variable model over "units," each carrying a bag of discrete "observations":

- Each unit has a latent vector η ∈ R^K with a logistic-normal prior η ~ Normal(μ, Σ),
  Σ = c·R, where **R is a fixed correlation matrix** and **c > 0 is a single scalar
  scale (concentration) hyperparameter**. θ = softmax(η) is a distribution over K
  components; observations are drawn Categorical(θ·B) with **B (K × V) the fixed
  component→observation emission matrix**.
- B and R are frozen (learned upstream). The one quantity being calibrated is **c** —
  it sets how concentrated a unit's θ is, i.e. the generative spread of simulated units.

**Calibration objective — held-out completion likelihood.** For each unit, hide a
random fraction f of its observations; infer the latent from the *visible* fraction
under prior scale c; score the predictive likelihood of the *held-out* fraction; sweep
c; take the (quadratic-smoothed) argmax. Two scorers:

- **MAP plug-in:** infer the posterior mode η̂ (from visible obs, at scale c); score
  held-out obs under the point θ̂ = softmax(η̂).
- **Marginalized (Laplace):** additionally form the Laplace covariance H⁻¹ at η̂; draw
  S samples η_s ~ Normal(η̂, H⁻¹); score each held-out observation w by the **log of the
  average** predictive, log[(1/S) Σ_s (θ(η_s)·B)_w] — the log-of-average, not the
  average-of-log.

**The prior claim under test.** The MAP-plug-in optimal c drifts with the holdout
fraction f, because a point estimate's optimal *regularization strength* depends on how
much data it conditions on (bias–variance). Marginalizing should remove that: under a
well-specified model the marginal-likelihood optimum is f-independent and equals the
true scale, because the posterior width already carries the "few visible obs"
uncertainty the MAP was forcing c to fake. Two caveats were flagged in advance:
(1) under **misspecification** the marginalized optimum is only a pseudo-true scale and
can still drift; (2) the Laplace posterior **under-disperses**, so a second-order
residual drift survives, same direction, de-biasable by importance sampling if material.

The three experiments below test this. Production regime is **K=60, V=5000, ~44
observations/unit**.

---

## 1. Synthetic, well-specified — the fix works in low dimension, INVERTS in high dimension

Plant units at a KNOWN scale (logistic-normal, level=5) over a shared-vocabulary B (so
inference must disambiguate overlapping components), freeze B at truth, sweep c under
both scorers at holdout f ∈ {0.5, 0.7, 0.95}. Recovered c\* via quadratic-smoothed argmax.

| regime | MAP c\* drift (max−min over f) | Marginalized c\* drift |
|---|---|---|
| **low-dim** (K=8, V=400, len=60, D=1000) | **0.83** (3.85 → 3.46 → 3.01) | **0.13** (flat ≈ 4.0) |
| **production** (K=60, V=5000, len=44, D=1500) | **0.12** (flat ≈ 3.4) | **0.99** (2.60 → 2.82 → 3.59) |

- **Low dimension: exactly as predicted.** MAP drifts monotonically with f; marginalized
  is flat — the plug-in artifact is real and marginalization removes it.
- **Production dimension: the result inverts.** The MAP is already flat (drift 0.12), and
  it is the *marginalized* estimator that drifts hard (0.99). At K=60 with ~22 visible
  obs over 60 components the point estimate is prior-dominated at *every* f — so the MAP
  regularization artifact vanishes — while the Laplace approximation to a 60-dimensional
  logistic-normal posterior becomes poor enough that its error, which varies with f
  (posterior width grows as visible obs shrink), *is* the drift.

This is caveat (2) — Laplace under-dispersion — escalating from the predicted "small
second-order residual, same direction" to the **dominant, drift-reversing term** at
production dimension. It is stronger than anticipated.

## 2. Is the production marginalized drift bias or Monte-Carlo variance? — BIAS

If the drift were MC noise from finite S, more samples would shrink it. Probe (K=60,
D=600):

| S (samples) | marginalized c\* drift (f=0.5 vs 0.95) |
|---|---|
| 128 (seed A) | 1.14 |
| 128 (seed B) | 1.06 |
| **512** (seed A) | **1.51** |

Drift is **stable across seeds** (1.06 vs 1.14) and **grows with S** (1.14 → 1.51). So it
is **bias, not variance**: more samples converge to the *biased* Laplace-marginal, not to
truth. More compute makes it worse, not better. (The vertex standard errors from the
quadratic smoother are ~0.005–0.01 in log c, so the curves are precise — this is a smooth
biased objective, not a jittery one.)

## 3. Real (misspecified) corpus — BOTH estimators drift, ~equally, in OPPOSITE directions

The decisive test: run both scorers on a real fitted model (production regime, K=60),
holdouts {0.5, 0.8, 0.95}. The full-corpus MAP sweep is the shipped calibration; the
marginalized sweep ran on a 967-unit sample (S=64) as an in-export diagnostic.

| holdout f | MAP c\* (full) | MAP c\* (sample) | **Marginalized** c\* (sample) |
|---|---|---|---|
| 0.5 | 4.61 | 5.30 | **2.36** |
| 0.8 | 3.75 | 3.90 | **2.65** |
| 0.95 | 3.65 | 3.80 | **3.76** |
| **drift** | **0.96** | **1.49** | **1.40** |

- **On real (misspecified) data the MAP drifts substantially** (0.96 full / 1.49 sample),
  *decreasing* with f — unlike the well-specified K=60 synthetic where MAP was flat. So the
  real MAP drift is not the plug-in regularization artifact (absent at this K); it is a
  property of the data.
- **Marginalization does not remove it.** Marginalized drift (1.40) ≈ MAP drift (1.49),
  and it runs in the **opposite direction** (*increasing* with f). The two cross near
  f=0.95 (≈ 3.8) and diverge maximally at f=0.5 (MAP 5.30 vs marginalized 2.36).

This is caveat (1) — misspecification — confirmed and *dominant*: the residual drift that
survives marginalization is a measurement of the model's misspecification (a genuine
spread of true per-unit concentrations, weighted differently at different f), and it is
**comparable in magnitude to the MAP drift it was meant to remove**. Marginalization
trades one drift for an equal-and-opposite one.

---

## 4. Synthesis

- The plug-in regularization artifact that motivated marginalizing is a **low-dimensional,
  data-rich phenomenon**. At production dimension with short units, the MAP is
  prior-dominated and the artifact is already absent.
- Plain Laplace-marginalization at production dimension introduces a **dominant Laplace
  bias** (Section 1–2), and importance-sampling de-biasing would target only that term.
- But even a perfectly de-biased marginal likelihood would **not** yield an
  f-independent scale here, because the real-data drift is **misspecification** (Section 3),
  not a Laplace artifact. So the marginalization program does not deliver its promised
  payoff — one prefix-independent scale — on this problem.
- **Practical disposition:** keep the MAP-plug-in scale; do not ship marginalization
  (more expensive, biased at scale, and does not reduce the real drift). The held-out
  scale is genuinely f-dependent — an honest calibration must *choose* an f (matched to
  the intended conditioning regime) rather than expect a single identified number.

## 5. Open question

At f=0.5 — the most-visible-data, least-Laplace-biased regime — the marginalized estimate
(≈ 2.4) sits far *below* the MAP (≈ 5.3), consistent with "proper posterior uncertainty
⇒ lower point-concentration than a MAP plug-in." Read one way, that argues the shipped
MAP scale (4.6) is too high and the generative scale should be lower. Read another way, it
is just the Laplace bias pulling the marginalized objective down where the posterior is
tightest. **Which is it — and does the answer imply the generative scale for a
prefix-conditioned simulator should be the marginalized-at-matched-f value rather than the
MAP-at-f=0.5 value?** That is the remaining calibration-philosophy question this test
surfaces but does not settle.

---

### Appendix — provenance

- Sections 1–2: synthetic plant-and-recover, controlled (frozen true B, single planted
  scale, quadratic-smoothed argmax). Well-specified by construction.
- Section 3: real fitted model, gated logistic-normal, K=60, V=5000; MAP sweep on the full
  corpus, marginalized sweep on a 967-unit sample at S=64. The marginalized numbers carry
  sample + MC noise but the qualitative pattern (opposite-direction drift of comparable
  magnitude) is unambiguous.
- Estimator detail that is load-bearing and was guarded by a dedicated test: the
  marginalized score is **log-of-average** per held-out observation, not average-of-log
  (the latter reproduces a plug-in-flavored objective by Jensen).
