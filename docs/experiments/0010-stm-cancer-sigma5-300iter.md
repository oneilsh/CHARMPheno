---
id: 10
slug: stm-cancer-sigma5-300iter
status: pending
model_class: stm
cohort: cancer
cohort_def: first_cancer_year
prior_obs_days: 0
person_mod: 4
doc_unit: patient
covariate_formula: "~ C(sex) + age"
categorical_cols: [sex]
continuous_cols: [age]
random_seed: 42
cache_uri: hdfs:///user/dataproc/charm/covariates_cache
K: 40
max_iter: 300
sigma_init: 5.0
---

# STM cancer demo: sigma=5 stability probe (300 iterations)

Stability check on experiment 0009. **Only `max_iter` changes** (100 → 300);
cohort, covariates, K, seed, and `sigma_init: 5.0` are identical, so iterations
0–99 reproduce 0009's trajectory exactly (deterministic, seed 42) and
iterations 100–299 reveal whether the fit has settled.

## Why

Exp 0009 (sigma_init=5) was a dramatic success — NPMI mean tripled to +0.204,
~28 distinct phenotypes, sex-specific cancers cleanly separated. But two facts
left its stability unconfirmed:

1. **Σ ran up to ~10^10** (from init 5.0) — a near-improper prior. The
   residual-variance M-step can be self-reinforcing: larger Σ → more peaked
   θ → larger η residuals → larger Σ. It may not have a finite fixed point.
2. **The fit hit max_iter (100/100)** rather than triggering the ELBO
   convergence criterion (unlike 0008, which "converged" at iter 26 into its
   degenerate basin). So Σ may still have been climbing at iteration 100.

## What to watch

- **Σ[min … max] per iter** — does it plateau by iter ~150–200, or keep
  climbing decade by decade? A plateau means a finite (if large) fixed point.
- **ELBO** — does it flatten (convergence) or keep creeping?
- **Convergence iteration** — if it converges before 300, the fit is settled.
- **Topic quality** — compare the iter-300 topic list + NPMI to 0009's iter
  100. Do the crisp phenotypes (breast/prostate/thyroid/melanoma/AF/HF/CKD)
  hold, sharpen, or smear back toward the marginal?
- The saved 2-D Σ trace (ADR 0030) captures the full per-iter trajectory for
  offline inspection.

## Decision

- Σ plateaus + topics hold → **0010 is the demo run** (better converged than
  0009); proceed to export → label → dashboard.
- Σ climbs but topics hold → benign; ship 0009 or 0010, document the regime.
- Σ climbs and topics degrade → bound Σ via `sigma_ridge` (currently 1e-6)
  before demoing; the σ=5 win was a transient, not a fixed point.

## Run

```
make exp ID=10
```

Result feeds the insight write-up (STM's acute sigma_init sensitivity and the
near-improper Σ operating point).
