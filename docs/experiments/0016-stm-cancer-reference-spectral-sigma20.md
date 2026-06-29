---
id: 16
slug: stm-cancer-reference-spectral-sigma20
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
sigma_init: 20.0
reference_topic: true
spectral_init: true
---

# STM cancer demo: reference + spectral init at sigma_init=20 (high-side init robustness)

The high-σ_init companion to exp 0015. **Identical to exp 0015 except
`sigma_init: 20.0` (vs 1.0).** Same cohort, covariates, K=40, seed 42,
reference_topic + spectral_init on, max_iter 300. The spectral analog of the
now-moot exp 0014 (which was reference-ONLY at σ=10 and was superseded once 0013
showed reference alone does not tame Σ).

## Why

exp 0015 (reference + spectral, σ_init=1) converged with **Σ bounded at 7.56**,
all 40 topics resolved, and the reference topic revived (insight 0030). The
claim that follows is that, with spectral init, Σ is no longer init-selected: the
residual-variance M-step has become a proper estimator that settles at the
data-determined scale (≈ O(1–10) here) *regardless of where σ_init starts*.

0015 started at σ_init=1 and Σ rose to 7.56 (it came UP to the natural scale).
This run starts an order of magnitude ABOVE that natural scale to test the mirror
direction: does Σ come DOWN to the same ~O(1–10) fixed point? Under the old
random-init regime, a large σ_init pushed Σ to the ~10^10 saturation boundary and
it never returned (exp 0009/0010/0011). If spectral has genuinely removed the
saturation feedback, σ_init=20 should decay toward 0015's fixed point instead.

## Hypothesis

`sigma_init=20` + reference + spectral gives the SAME outcome as 0015
(sigma_init=1): Σ settles to ~O(1–10) (coming DOWN from 20, not blowing up to
10^10), all 40 topics resolve, the reference topic stays alive, and the phenotype
list / ELBO match 0015. Quality and Σ are flat across σ_init ∈ {1, 20} — the
knife-edge is gone on the high side too.

## What to watch (vs exp 0015, the σ_init=1 spectral run)

- **Σ[min … max] trace** (ADR 0030) — THE test. Does Σ *descend* from 20 toward
  ~7-ish over the iterations (proper estimator), or stall high near the init
  (residual feedback not fully gone)? Watch the trajectory, not just the final
  value: a monotone decay to 0015's scale is the clean confirmation.
- **Final Σ max** — target ~O(1–10), close to 0015's 7.56. A final Σ near 20
  (stuck at init) or climbing toward 10^10 would mean spectral tamed σ=1 but not
  σ=20 — investigate the M-step.
- **Topic resolution** — all 40 active (min Σλ well above the 93.7 dead-floor),
  matching 0015. Reference topic 0 alive (E[β] ≫ 0.0001).
- **Phenotype list + NPMI + ELBO** — flat vs 0015 (NPMI mean ≈ +0.17, ELBO
  ≈ −1.10e6). Read peak β / Σλ, not NPMI mean (insight 0026/0030 caveat).

## Decision

- **Σ descends to ~O(1–10), quality flat vs 0015** → spectral makes Σ a proper,
  init-independent estimate on BOTH sides of the old knife-edge. Confirms insight
  0030 and finalizes the case: ship reference + spectral as the default stack and
  treat Σ as a real covariance (un-park the faithful sampler, ADR 0028-B).
- **Σ stuck high near 20 (doesn't descend) but topics fine** → spectral fixes the
  blowup-from-below but the M-step still can't pull Σ *down* from an over-large
  init; document the asymmetry and prefer a small default sigma_init (1) with
  spectral.
- **Σ blows toward 10^10** → spectral's taming is σ_init-bounded on real data
  (diverges from synthetic on the high side); revisit the residual-variance
  M-step / add a Σ-prior top-up.

## Run

```
make exp ID=16
```

Compare head-to-head with exp 0015 (σ_init=1, the same stack). Result finalizes
the σ_init-robustness claim of insight 0030.
