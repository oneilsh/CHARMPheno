# Insight 0052 — Two-engine architecture: mean-field VI is a sound point-and-ordering engine but an unsound calibration engine (the single mechanism behind 0044/0047/0048/0051)

**Date:** 2026-07-13
**Branch:** pg-stm
**Topic:** pg-stm | mean-field | variational-inference | calibration | uncertainty | theory
**Status:** Observed (synthesis of 0044/0047/0048/0051; framed with Fable as the arc's headline general result)

**The synthesis.** Four separate findings in this project are one result with four
confirmations:

| # | read-out | mean-field failure |
|---|---|---|
| 0044 | Σ correlation (sign) | wrong sign under mean-field; Gibbs recovers |
| 0047 | Σ rank / scale / conditioning | attenuated (rank collapse, eigmin floor); Gibbs recovers |
| 0048 | β-conditioned Σ read-out | corrupted by mean-field β-sharpening under overlap |
| 0051 | additive-offset intervals | ordering correct, absolute coverage 0.13 (biased means) |

**The one mechanism.** Mean-field's `KL(q‖p)` direction makes `q` mode-seeking and
under-dispersed, so (i) the point estimates are biased toward sharp / decoupled
configurations (β sharpening, Σ sign flips, rank collapse, offset attenuation); and
(ii) the self-reported variance measures scatter *around the biased point* and is
structurally **blind to the bias itself**. Clause (ii) is load-bearing: it is why no
within-VI fix — more iterations, better schedules, wider grids — can ever produce
calibrated *absolute* uncertainty, and why the fix is identical in all four cases:
route absolute uncertainty through an **exact-conditional (Gibbs) pass**.

**The consequence — a measured division of labor, not a workaround.** In gated
logistic-normal topic models, mean-field VI is a **sound point-and-ordering engine**
(point estimates and relative/ordinal signals are trustworthy — e.g. 0051's
scarce/populated width ratio 2.0 is a design-moment property, correct without any
calibration) and an **unsound calibration engine**. Ship the two-engine architecture:
VI for points and ordering; an exact-conditional pass for anything that must carry an
absolute coverage number.

**Two walls that were being conflated (Fable's separation).** The forward plan
depends on not confusing them:
- **Information wall** — exact Gibbs with the *correct* β still cannot identify
  scarce, overlapping structure from this much data (insight 0049's scarce-gated
  block). No estimator fixes this; only more data or pooling does (this is why the
  DAG / partial-pooling layer exists).
- **Estimator wall** — the information *is* in the data (the oracle rows and Gibbs
  recoveries prove it four times over) and mean-field simply cannot carry it.
The read-out engine only needs to climb the **estimator** wall — which the warm-started
co-sampled Gibbs pass already did for the well-populated blocks (shared-block MAE ≈
oracle at η_β=1.0, insight 0049). Scarce blocks correctly come back **wide** under it;
wide-but-covering is the right answer, and the engine's job is to make the reported
uncertainty *true*, not to make scarce things certain.

**Why it matters for the build order.** Because the offset layer is realized as an
augmented covariate (offsets = covariate coefficients, insight 0050 / model core v1),
the exact-conditional engine for offsets is just the Gibbs pass's **Γ-block draw** —
conditionally Gaussian, conjugate, one more block in the sweep already validated. So
"the calibration engine" is a scoped engineering push (productionize + extend the
validated co-sampling read-out to emit offset-increment draws + a coverage plant as
its acceptance test), not a research program. If scarce-node coverage fails even under
Gibbs, that is precisely when the LKJ / half-t provenance priors earn their place.

**Paper framing.** This is the strongest general contribution of the estimator arc:
*mean-field VI in gated logistic-normal topic models is a sound point-and-ordering
engine and an unsound calibration engine; absolute uncertainty must be routed through
an exact-conditional pass.* Stated once, it turns four empirical findings into one
theorem-shaped claim with four confirmations.
