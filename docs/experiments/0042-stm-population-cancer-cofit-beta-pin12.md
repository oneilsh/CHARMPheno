---
id: 42
slug: stm-population-cancer-cofit-beta-pin12
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
sigma_diagonal_pin: 12.0
---

# Experiment 0042 — refit round 2: gated STM fit AT Sigma = 12*R (ratchet vs fixed point)

**Round 2 of the refit loop.** Byte-identical to exp 0028 / 0041 (same corpus,
covariates, gating, K, seed=42); the ONLY change is `sigma_diagonal_pin: 12.0`.

## Why

The refit sequence so far, calibrating the generative scale by held-out predictive
LL after each fit:
- Round 0 (exp 0028, unit pin Sigma=R):  held-out c* = 5.
- Round 1 (exp 0041, pin Sigma=5*R):     held-out c* = 12  (and the co-fit beta
  predicts held-out tokens uniformly BETTER: peak per-token LL -6.577 vs 0028's
  -6.615, +0.038 nats).

So the recalibration MOVED UP 5 -> 12 (not back to 5): fitting at a higher scale
makes the model prefer a higher scale still. One step cannot distinguish a ONE-STEP
SHIFT to a new fixed point near 12 from a genuine RATCHET that keeps climbing
(Fable Q3). This round settles it: fit at pin=12, recalibrate.

- Recalibrated c* ~ 12  => FIXED POINT at 12 (the refit map contracts; 12 is the
  self-consistent generative scale, and the co-fit model @ 12 is the one to ship
  if it also passes the seed-panel over-commitment check at that scale).
- Recalibrated c* > 12 / at the grid boundary => RATCHET (the scale runs away;
  the refit loop is unstable on real data and must NOT be shipped as a training
  step — keep c a generation-only knob calibrated once off the unit fit).

The export grid was widened to [1,2,3,5,8,12,16,20,28] so this round can LAND at
an interior value above 12 (16 / 20 / 28) rather than being pinned at the old
top-of-grid 20 — necessary to read settle-vs-climb cleanly.

## What to compare (0042 vs 0041 vs 0028)

- **Recalibrated held-out c\*** (export's corpus_heldout_scale_sweep_gated): the
  headline — does it settle near 12 or climb past.
- **Peak held-out per-token LL**: 0028 -6.615 -> 0041 -6.577 -> 0042 ?. If it keeps
  rising, the co-fit is still improving prediction; if it plateaus, we are near the
  fixed point.
- **NPMI** (beta-only coherence): expected roughly flat again (0028/0041 were flat),
  reported for completeness.

## Run

```
git pull                          # gets the widened export grid + this experiment
make exp ID=42                    # fresh gated STM fit at sigma_diagonal_pin=12.0
make build-dashboard-exp ID=42    # export -> recalibrated c* + widened held-out LL curve
```

Then send the 0042 correlation.json (eta_scale + eta_scale_diagnostic) + summary
NPMI; compare against the 0028 -> 0041 -> 0042 trajectory.

## Result

(pending cluster run)
