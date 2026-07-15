---
id: 41
slug: stm-population-cancer-cofit-beta-pin5
status: pending
model_class: stm
cohort: population_cancer
cohort_def: population_cancer
prior_obs_days: 0
person_mod: 4
doc_unit: patient_cohort
doc_min_length: 10
covariate_formula: "~ C(sex) + age"
categorical_cols: [sex]
continuous_cols: [age]
known_sex_only: true
random_seed: 42
cache_uri: hdfs:///user/dataproc/charm/covariates_cache
K: 60
background_k: 40
foreground: "cancer:20"
group_var: source_cohort
max_iter: 200
sigma_init: 1.0
sigma_diagonal_pin: 5.0
---

# Experiment 0041 — co-fit-beta: population+cancer gated STM fit AT the calibrated scale (Sigma = 5*R)

**Clean A/B against exp 0028.** Byte-identical corpus, covariates, gating, K, seed
(random_seed=42) — the ONLY difference is `sigma_diagonal_pin: 5.0`, so the fit
holds Sigma_ii = 5 (Sigma = 5*R, ADR 0036) every M-step instead of the unit pin
(ADR 0034). This is a fresh fit (full learning-rate schedule) rather than a
tail-resume of 0028, so beta actually adapts to the wider prior — a
tail-continuation at iter 200 would move beta negligibly (step ~0.017).

## Question (Fable Q5 / co-fit-beta, on the real corpus)

Does fitting at the calibrated generative scale c*=5 (rather than the unit pin)
sharpen beta / improve topics? The held-out-LL calibration selected c*=5 as the
generative scale for exp 0028's *unit-fit* beta; the hypothesis is that a beta
CO-FIT under Sigma=5*R — where the E-step's weaker prior lets each document's eta
spread, concentrating per-doc theta and sharpening topic-token responsibilities —
predicts held-out tokens better and yields more coherent topics. The synthetic
refit-dynamics probe (exp 0040) was confounded for this question (readout-prior
confound, scale-relative-to-R, flat identifiability), so it is answered here on
the real corpus where beta is real and the scale is in the model's own basis.

## What to compare (0041 vs 0028)

- **Topic coherence (NPMI, Roder et al. 2015):** beta-only, no Sigma confound —
  the primary signal. NPMI(0041) > NPMI(0028) => co-fit at c=5 sharpened topics.
- **Held-out predictive-LL curve + recalibrated c\*** (export's
  corpus_heldout_scale_sweep_gated): if the co-fit beta predicts held-out tokens
  better, its LL curve sits higher; its argmax is the RE-calibrated scale — Fable
  predicted the second calibration "moves much less than the first", so expect it
  to stay near 5 (the (beta, c*) fixed point) rather than ratchet.
- **Concentration (top_mass / eff_topics):** reported for context but Sigma-
  confounded across the two fits (0028 reads out under Sigma=R, 0041 under 5*R) —
  do NOT use as the primary comparison; lead with NPMI + held-out LL.

## Run

Fresh fit (NOT a resume of 0028):

```
make exp ID=41                    # fits from scratch at sigma_diagonal_pin=5.0
make build-dashboard-exp ID=41    # export -> recalibrated c* + NPMI (covariate cache
                                  #   HITs: same corpus/formula/cohort as 0028)
```

Then compare 0041's summary NPMI + correlation.json eta_scale/eta_scale_diagnostic
against 0028's. (If the covariate cache misses, `make build-covariates EXP=0041`
first — but the key is identical to 0028's, so it should HIT.)

## Result — co-fit helps prediction; recalibrates UP to 12 (see 0042 for the ratchet resolution)

Recalibrated c* = **12** (interior; robustness {0.5: 12, 0.8: 8, 0.95: 8}) -- NOT back to 5, so 5 is not
a fixed point of the refit map. The co-fit beta predicts held-out tokens uniformly better than the unit
fit: peak per-token LL -6.577 (at c=12) vs 0028's -6.615 (at c=5), +0.038 nats; better at every
comparable scale (c=5: -6.601 vs -6.615). **NPMI (coherence) is flat**, though: background mean 0.1816
-> 0.1912, foreground 0.2284 -> 0.2298 (both within noise). So co-fitting improved PREDICTION, not
coherence -- the two axes diverged, and an NPMI-only read (a first draft) wrongly called it "no
benefit." The 5 -> 12 move looked like a ratchet; round 2 (exp 0042, fit @12) recalibrated back to 8,
showing a damped oscillation (5 -> 12 -> 8, fixed point ~9-10), NOT a runaway. This 0041 model (fit @5,
generate @12) is the best predictor observed. Full writeup:
`docs/superpowers/specs/2026-07-05-update3-seed-panel-refit-and-cofit-beta.md`.
