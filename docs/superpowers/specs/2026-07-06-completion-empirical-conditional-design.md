# Conditioned-completion: the empirical-conditional contract (Fable review-3 follow-up, design)

Supersedes the Fix-1/Fix-2 sketch in `2026-07-06-fable-review3-response-and-seed-regime-plan.md`.
Records the design Fable and we converged on for the tool's conditioned-completion feature (user
supplies a few tokens -> tool generates a synthetic user-day conditioned on them). Data shape: documents
are user-day bag-of-words aggregates; there is no "seed" in the data -- the seed is purely this
use-case.

## Decision (user, 2026-07-06): the honest contract

Two legitimate contracts for the completion tool; we chose the first:
- **"A plausible user-day CONTAINING these tokens"** (CHOSEN) -- the Bayesian conditional, faithful to
  the corpus even when faithfulness means a rare token stays a passing mention. Keeps the synthetic
  cohort a real sample whose statistics downstream consumers can trust.
- "A user-day EXEMPLIFYING this theme" (rejected as the default) -- deliberate steering past the
  empirical conditional. Not shipped as a hidden second behavior; see "showcase" below.

## The spec, in one sentence

The completion posterior for a seed token w must match the distribution of theta_hat_k among the real
user-days that CONTAIN w. That is the whole contract; everything below is mechanism. Accept the
consequence: for a rare token whose real containing-days mostly treat it as incidental, the honest
completion is diffuse too -- the SS-A "failure" is fixed exactly to the degree the corpus says it is one.

## Run the diagnostic FIRST -- it may dissolve the problem

Before building anything: for a panel of rare seed tokens (spanning frequency bands), compute the
empirical conditional -- the distribution of theta_hat_k among containing user-days. This adjudicates
whether SS-A is even a real failure:
- If containing-days have theta_hat_k ~ 0.4 (genuinely about the theme) -> the current diffuse
  completion IS miscalibrated, and the diagnostic hands us a quantified target.
- If containing-days have theta_hat_k ~ 0.08 (the token is mostly a passing mention) -> the "failure"
  is FIDELITY; the tool needs a knob/documentation, not a fix, and much of this work evaporates.

**Critical build note -- do NOT calibrate to compressed theta_hat.** The bundle's fitted theta_hat were
inferred under the UNIT prior and are systematically over-diffuse (the compression we spent three
reports establishing). Re-infer theta_hat for the containing documents at c = 5 with beta FROZEN -- one
cheap gated E-step pass over the (small) set of containing docs per panel token -- and build the target
from THOSE. The unit-prior fit's per-document outputs are not measurements.

## The mechanism: multi-start mixture IS the (bimodal) conditional

The empirical target is expected to be BIMODAL: real containing-days split into an "about-the-theme"
population (theta_hat_k substantial) and an "incidental/passing-mention" population (theta_hat_k small,
mass on background). A single tilted Gaussian cannot reproduce that -- but the multi-start E-step
already produces exactly the right object:
- **Find the basins:** start Newton from the prior mean (prior basin -> common topics) AND from each
  topic carrying meaningful prefix responsibility (token basin(s) -> the theme). The conditioned
  log-posterior is non-concave (softmax-mixture likelihood = difference of log-sum-exps), so these are
  genuine distinct modes. Each converges to a Laplace Gaussian.
- **Represent, don't collapse:** SAMPLE a basin per completion (do NOT argmax), then sample eta within
  it. Repeated completions from one seed then produce a MIX of focused days and passing-mention days --
  matching what the corpus does. Fix 2 is thus the REPRESENTATION of the conditional, not a safety net.

## The calibration knob is the basin WEIGHTS, not the location

- Score every mode's mass under the SAME joint (prior x likelihood), on equal footing.
- Be suspicious of the Laplace width term log|H^-1| at the token basin: with a 1-2 token prefix that
  basin is shallow/skewed, and Gaussian basin-width there is the same Laplace bias that haunted the
  scale EM. De-bias with a handful of IMPORTANCE SAMPLES per basin (proposal = the basin's Laplace
  Gaussian) -- cheap at (K-1) dims.
- Then CALIBRATE the mixture proportion (one scalar per token-frequency band) so the about-vs-mention
  split matches the empirical conditional. This is lambda's job now: correct the (biased) basin weights
  against an observable, NOT move the mode's location. Same spirit as the held-out scale calibration --
  matched to an observable, no hand-set constant.

## Rejected mechanism: the token up-weight multiplier / power likelihood

Do NOT up-weight the prefix token to n effective observations. A power likelihood (tempered posterior)
does move the mode toward the token basin, but it also SHRINKS the posterior covariance -- n
pseudo-replicates manufacture false certainty, so every completion from the same seed collapses to the
same focused day (dead diversity). We want mass in the right basin, not confidence we do not have. The
mixture-weight calibration above gets the location/weight effect without the covariance side effect.
(Exponential tilting in eta-space -- a conjugate prior-mean shift N(Gamma^T x + lambda*c*R e_k, cR),
covariance untouched -- was Fable's clean alternative for MOVING a mode; the bimodal honest contract
subsumes it, since we keep both basins at their natural locations and calibrate weights instead.)

## Validation: generate-then-re-infer (bundle-build job, not new infra)

For the seed panel across frequency bands: generate N completions per seed, push them back through the
(c = 5, gated) E-step, and compare the completed-theta DISTRIBUTION to the re-inferred empirical
conditional -- mixture proportion, component locations, and overall shape, not just a mean. Same
generate-then-re-infer pattern as the held-out scale calibration, so it reuses that machinery. Match
across bands => the tool is honest by construction and can be documented as such with evidence attached.

## Showcase need: better conditioning, not a steering knob

Someone will still want "a user-day exemplifying this niche theme." Under the chosen contract the honest
answer is NOT a steering code path -- it is BETTER CONDITIONING: a 3-4 token seed drawn from the topic's
signature is legitimate evidence, and the honest posterior concentrates on the theme because the data
genuinely says so (SS-A: self-recovery climbs steeply from one token to two). "If you want a focused day,
give a focused seed." A documentation sentence, not a second code path -- so every completion stays a
draw from the model's conditional and the synthetic cohort's statistics remain trustworthy for every
downstream consumer. Protecting that trustworthiness is the property the gated design existed for from
the first brief.

## Build order (post-compaction)

1. **Diagnostic** (adjudicates + sets targets): re-infer theta_hat at c=5 (beta frozen) for containing
   docs of a rare-token panel; plot the empirical conditional per frequency band. If it says "passing
   mention," stop and document -- there is no bug.
2. **Multi-start E-step + mixture sampling** in the completion path (recordPosterior.ts / the gated
   conditioned E-step): multi-start Newton, importance-sampled basin weights, sample-a-basin-then-eta.
3. **Basin-weight calibration** (one lambda per frequency band) to the empirical about-vs-mention split.
4. **Validation** (generate-then-re-infer, distribution match across bands) as a bundle-build check.
5. **Docs:** the plausible-completion contract as default; "focused seed for a focused day" as the
   showcase guidance.

## What changed from the earlier Fix-1/Fix-2 sketch
- Fix 1 (covariate marginalization): demoted -- weak given thin covariates (+/-1-3 nats vs ~7-nat gap).
- Fix 2 (multi-start): PROMOTED from "enabling half / safety net" to the representation of the conditional.
- Token up-weight multiplier: DROPPED (covariance-shrink -> collapsed completion diversity).
- NEW spine: empirical-conditional calibration (the spec), re-inference at c=5 (avoid the compression
  artifact), basin-weight (not location) tuning, generate-then-re-infer validation, "focused seed" for
  showcase.
