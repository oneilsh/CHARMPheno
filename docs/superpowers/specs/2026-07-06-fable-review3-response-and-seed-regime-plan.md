# Fable review-3 response + the seed-regime reclassification (actionable plan)

Fable's reply to Update 3. The generative-scale arc is CLOSED; the one residual problem (tiny-seed
conditioned completion) is REDIAGNOSED away from the scale and given two concrete fixes. This doc
records the verdicts and the plan so it survives context compaction.

## Verdicts on our three questions

- **Q1 (don't chase the refit loop): AGREE.** Fit-scale == generation-scale self-consistency has NO
  intrinsic value -- it's aesthetic. Confirmed by the data: the best predictor is a deliberately
  INCONSISTENT pair (fit at 5, generate at 12). This is the "two roles of Sigma" point: fitting wants a
  moderate prior (enough regularization that beta learns from stable responsibilities); generation/
  prediction wants the wider scale that matches true document concentration. Different optima; the fixed
  point is a compromise serving neither best. Caveat Fable flags: "the fixed point lies between 0041 and
  0042 and can't beat 0041" is an INFERENCE (monotone interpolation between two points), not a
  measurement -- beta fit at 9 was not run. He would not run it either; just don't overstate it as
  measured. And: IF anyone wants 0041's +0.038 nats, generation at c=12 MUST pass the seed-panel
  over-commitment check at 12 first (we only probed to 8) -- a real gate.

- **Q2 (conditioning-aware scale): WRONG DIAGNOSIS.** The tiny-seed failure is NOT about c. It is (i) the
  prior MEAN and (ii) MAP mode-selection. See the two fixes below. A seed-size-dependent scale is to be
  AVOIDED: a prior that weakens as data shrinks is anti-Bayesian (the prior should matter MORE with less
  data, and its job is to be RIGHT, not weak), and it bakes an inference-regime artifact into the
  generative model. The product-rule guardrail ("seed rare topics with >=2 codes / non-mean covariates")
  is fine as an interim workaround but it is papering over the plug-in-mean bug.

- **Q3 (flat held-out LL curves): fine for band-selection, but do two things** (below).

## The seed-regime fix (this REPLACES the "conditioning-aware scale" line of work)

The observed failure: at c in {3,4,5}, most rare foreground seeds do NOT recover their own topic
(self-recovery 5-15% single-token / 25-65% two-token); they get pulled to common background topics,
because Gamma^T x is evaluated at the population-MEAN covariate default and rare foreground topics have
strongly negative population-mean intercepts (e.g. -4.31 vs a common topic's +2.95). Two independent,
complementary, cheap fixes:

### Fix 1 (primary): marginalize over covariates instead of plugging in x-bar
The bug is the plug-in mean x-bar. A user/document that expresses a rare seed is overwhelmingly NOT a
population-mean member. Bayesian-correct treatment: the generative population is a mixture over the
empirical covariate distribution, so completion should compute
  p(x | seed) proportional to p(seed | x) * p_hat(x)
and infer eta under the REWEIGHTED prior mean. In practice: sample or enumerate covariate profiles, run
the cheap per-profile E-step under each Gamma^T x, weight each by its marginal seed likelihood, and mix.
A rare-community seed concentrates p(x | seed) onto the covariate cells where that community lives -> the
prior mean shifts to where that topic's intercept is NOT -4.31 -> the seed recovers its topic at the
already-calibrated c = 5. No scale surgery, no data-dependent prior. It is also correct at the product
level: "complete a profile from a rare-community token" SHOULD imply "probably a member of that
community" -- the plug-in mean was silently denying that inference.

### Fix 2 (complementary): multi-start E-step with basin scoring
With a 2-token seed the posterior over eta is genuinely MULTIMODAL -- one basin near the
background-heavy prior mean, one near the seed's topic. A single MAP/Laplace run initialized at the
prior mean deterministically finds the WRONG basin when the prior basin is deeper. This is a
mode-SELECTION failure of the inference, not prior stiffness. Fix: initialize the Newton E-step once at
the prior mean AND once per topic carrying meaningful seed-token responsibility (the seed token has
~0.835 responsibility on its own topic -- we have these), run to convergence from each start, score each
converged mode by its joint density (or Laplace-approximate marginal), then pick or mixture-weight. Cost:
a few extra Newton solves on a (K-1)-dim problem per completion -- trivial. You do not DRAG eta out of the
wrong basin; you START in the right basin and check whether it is competitive.

**Predicted outcome:** Fix 1 alone recovers most of the 5-15% -> acceptable self-recovery gap; Fix 2
covers the residue; the combination makes "c ~ 20-50" unnecessary. Reclassification headline for whoever
picks this up: **the seed problem was never about c -- do not let it reopen the scale question.**

## Q3 follow-ups (do when convenient; not blocking)

1. **Bootstrap error bars on the held-out LL.** Document-level bootstrap (resample documents, recompute
   the per-token mean LL) to test whether the 0.001-0.01-nat differences across c in 2-6 exceed split
   noise. Fable suspects much of the flat region IS within noise -- which would also explain the
   oscillation amplitude (5 -> 12 -> 8 argmax wander is what a loosely-identified optimum does on a
   plateau under resampling). Read "fixed point ~9-10" as "somewhere on a broad shelf"; temper how
   precisely the number is quoted. Not a shipping problem (flat objective => c insensitive => any choice
   in-band defensible). Requires the sweep to return per-document held-out LL contributions (it currently
   returns only the corpus mean) -> small code change, then bootstrap.
2. **Concentration-matching auxiliary objective (deferred until something downstream demands tighter
   pinning).** Held-out LL is flat in c, but GENERATED concentration is NOT (top_mass / eff-#topics move
   materially across c in 2-6 -- the seed-panel table shows it). If the scale ever needs a POINT not a
   band: choose c so the distribution of generated-document theta concentration matches the real corpus's
   theta-hat concentration distribution under matched conditioning (this is the old B2-flavored idea).
   Prediction picks the band; concentration-matching picks the point within it. Do NOT build until needed.

## The completed arc (Fable's synthesis)

The original problem was "the fit discards the scale and nothing recovers it safely." The layered answer
that survived all the data:
- unit-pinned fit (stability),
- held-out-calibrated GENERATION scale (c = 5; honest; loosely but sufficiently identified -- a broad
  shelf, not a sharp point),
- co-fitting as an OPTIONAL, bounded, non-monotone +0.04-nat refinement (skippable; ship default is the
  unit fit + generation-only c = 5),
- and the residual seed-regime problem RECLASSIFIED from "scale" to "prior-mean + mode-selection", with
  Fix 1 + Fix 2 above leaving the scale architecture alone.

Coherence/prediction decoupling is the general lesson worth keeping: NPMI sees only a topic's top tokens;
held-out LL sees the whole model; a beta can redistribute mid-tail mass to help prediction without moving
NPMI. Never conclude "beta didn't change" from coherence alone.
