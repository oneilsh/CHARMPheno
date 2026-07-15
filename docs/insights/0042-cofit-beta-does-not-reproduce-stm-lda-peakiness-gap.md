# 0042 — Co-fitting beta does NOT reproduce the STM-vs-LDA peakiness gap: LDA's Dirichlet pressure carves a marginally sharper beta, but a sharper beta does not make patients peakier — so neither alpha-inference (0038) nor beta-co-adaptation explains the real-data gap

**Date:** 2026-07-09
**Topic:** stm | lda | generation | concentration | beta-learning | calibration | diagnostics
**Status:** Confirmed (synthetic co-fit-beta plant-and-recover, `spark_vi/eval/topic/concentration_recovery.py` + `scripts/cofit_beta_concentration_experiment.py`; results in `docs/experiments/0045-cofit-beta-concentration/`)

Insight [0038](0038-heldout-ll-recovers-true-concentration-and-lda-alpha-opt-is-not-hot.md)
validated held-out predictive-LL as a ground-truth-free concentration calibrator
with beta FROZEN at truth, and refuted the "LDA alpha-optimization reads hot"
story — reattributing the real-data STM-vs-LDA peakiness gap (median top_mass:
STM 0.269 vs LDA 0.513, exps 0033/0034) to two candidates it could not test on
frozen beta: STM's fit scale being too low, and LDA CO-FITTING a sharper, more
document-specific beta. This insight tests the second candidate directly, the
one experiment 0038 explicitly left un-run.

## The experiment

Same plant-and-recover as 0038 (shared-vocab beta via `make_shared_beta`, two
planting mechanisms — logistic-normal and Dirichlet — four concentration levels,
clean K=8/V=400 and real K=60/V=5000 regimes), but each model now LEARNS its own
beta by full-batch variational EM instead of freezing it at truth. STM learns
beta under a fixed logistic-normal prior N(0, c·I) and calibrates c by the
held-out-LL sweep; LDA learns beta at Dirichlet alpha (both a held-out-LL alpha
sweep and its own empirical-Bayes alpha-optimization). Beta is trained on a TRAIN
corpus and scored by document-completion held-out-LL on a disjoint TEST corpus
(Wallach et al. 2009). Topic recovery is measured permutation-invariantly
(Hungarian match, Kuhn 1955), with a beta-sharpness readout (top-k mass;
inverse-Simpson effective vocabulary, Hill 1973 / Jost 2006). A 5x-larger
training corpus (D=3000) control isolates small-sample effects.

## Finding 1 — co-fitting beta does not reproduce the gap; the sign even flips

At the held-out-LL-calibrated knob, co-fit LDA theta is LESS peaky than co-fit
STM theta in BOTH real regimes (top_mass 0.032 vs 0.063 at D=600; 0.127 vs 0.154
at D=3000) — the OPPOSITE ordering to the real data, where LDA is the peakier
family (0.513 vs 0.269). The real-data-matching LDA config (co-fit beta + alpha
optimized) collapses hardest, to top_mass 0.030 at large D. Co-fitting beta,
under this generative model, does not open an LDA>STM peakiness gap; it does the
reverse.

## Finding 2 — LDA's beta IS marginally sharper, but a sharper beta does not make patients peakier

The beta-sharpening half of the hypothesis is real: LDA's co-fit beta is
consistently a little sharper than STM's (mean top-k mass 0.213 vs 0.198 at
D=600, 0.158 vs 0.120 at D=3000; effective vocabulary always lower for LDA, e.g.
227 vs 408 terms at D=3000), and the effect GROWS with data. But it is causally
inert for concentration: despite the sharper beta, LDA's per-document theta stays
LESS peaky than STM's at the calibrated knob. The hypothesized chain "Dirichlet
pressure → sharper, more document-specific beta → peakier patients" breaks at the
last link — sharper topics did not yield peakier documents.

## Finding 3 — co-fitting beta collapses concentration, and it is largely a small-sample beta-identifiability failure

With beta frozen at truth, both families recover the planted concentration well
(real: STM 0.183 / LDA 0.166 vs planted 0.201). Learning beta on the SAME
documents drops recovered top_mass to 0.03–0.06 at D=600. The mechanism is
visible in the held-out-LL curves: at D=600 the argmax sits at the DIFFUSE grid
BOUNDARY (STM c=1, LDA alpha=3.0 in 7/8 cells) — a poorly-identified beta cannot
support confident, generalizing per-document assignments, so held-out prediction
prefers a near-uniform theta and there is NO interior concentration optimum
(contrast 0038's interior frozen-beta argmax). At 5x data the argmax moves back
INTERIOR (STM c=5, LDA alpha=0.3) and recovered top_mass climbs to 0.13–0.16, so
the collapse is largely curable by data. Even cured, Findings 1–2 hold: LDA
stays cooler than STM.

## Which family recovers ground truth better? — two axes, only one has a clean answer

"Better recovery" splits by what you measure:

- **Per-document concentration (θ variance) — STM, robustly.** Recovered top_mass is
  closer to the planted value for STM than for LDA in every measured cell, frozen and
  co-fit, at both corpus sizes tested (frozen real: STM 0.183 vs LDA 0.166 vs planted
  0.201; co-fit real D=600: 0.063 vs 0.032; D=3000: 0.154 vs 0.127 vs planted 0.288).
  STM's logistic-normal is the better concentration recoverer; LDA reads cooler. This is
  the one categorical claim the data supports.
- **Topic sharpness (β top-k mass) — LDA is closer to truth here.** LDA's co-fit β
  top-k mass tracks the planted sharpness (~0.211) across cells, while STM under-sharpens
  at higher concentration (e.g. 0.161 vs true 0.211). LDA's Dirichlet carves topics whose
  peakiness sits nearer the true topics'.
- **Topic direction (β 1−cos to truth) — NOT established.** STM's Hungarian-matched
  β-cosine error was lower than LDA's at D=600 (all 8 cells) and higher at D=3000 — the
  ordering flipped between the two sizes tested. This is an observation at D=600 vs D=3000
  ONLY, not a large-scale claim: D=3000 is far below the ~10.8k-document real corpus, let
  alone production, and the co-fit optimizer settings (learning rate, EM iteration count,
  convergence tolerance) were held fixed rather than tuned per size, so the flip is
  confounded by optimization. Read it as "the β-direction ordering is size-sensitive under
  these settings," not "LDA recovers β better at scale."

The takeaway is the decoupling: θ-concentration and β-sharpness are separate competencies
— STM the better recoverer of the first, LDA of the second — which is exactly why a
sharper LDA β (Finding 2) did not buy peakier θ. No single family "recovers ground truth
better"; it depends on the axis.

## Consequence — the real gap is not a modeling artifact of either inference; pin it on the real corpus

0038 ruled out alpha-inference; this insight rules out beta-co-adaptation.
Neither synthetic mechanism reproduces the real-data STM-vs-LDA peakiness gap —
in fact both point the other way. So the gap is not an artifact of how either
family infers alpha or learns beta on data of this structure; it is a property of
the real corpus and/or the models' scale calibration that this
shared-vocab-planting synthetic does not capture (real topics may be far sharper
and more separated than the 50%-shared-pool construction, and the real corpus is
far larger and better-identified). The decision-critical step is unchanged and
reinforced: pin the faithful per-document concentration by running the validated
held-out-LL sweep on the REAL corpus, rather than attributing the gap to an
inference or beta-learning artifact.

## What this does NOT claim

- It does not claim LDA never sharpens beta — it does (Finding 2), just without a
  peakiness payoff here.
- It does not measure the real corpus's concentration — the synthetic beta and
  planting are a MODEL of the problem; the shared-vocab construction and the
  D=600/3000 corpus sizes are almost certainly less identifiable and less sharp
  than the production corpus, which is exactly why the real gap may live outside
  this model's reach.
- It does not reopen a fit-time free variance (still falsified, exps 0022/0024/0032).

**Setting context:** Local pure-numpy plant-and-recover, no Spark. K=8/V=400
(clean) and K=60/V=5000/doc_len=44 (real, matching production), full-batch VB
co-fit of beta (~50–60 EM sweeps), held-out document-completion at 30% masked
tokens, deterministic seeds. Extends the 0038 machinery additively; the
frozen-beta path and its tests are unchanged.

**Related:** insight 0038 (the frozen-beta validation this extends), 0037 (the
concentration bracket), exps 0033 (STM A1) / 0034 (LDA alpha-opt) for the
real-data gap. Code: `spark_vi/eval/topic/concentration_recovery.py`
(`stm_cofit_beta`, `lda_cofit_beta`, `beta_recovery_error`, `beta_sharpness`,
`sweep_heldout_cofit`), `scripts/cofit_beta_concentration_experiment.py`; results
`docs/experiments/0045-cofit-beta-concentration/`. Metric: Hill 1973 / Jost 2006;
Hungarian assignment Kuhn 1955; document-completion Wallach et al. 2009.
