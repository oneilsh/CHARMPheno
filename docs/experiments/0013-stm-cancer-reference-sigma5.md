---
id: 13
slug: stm-cancer-reference-sigma5
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
reference_topic: true
---

# STM cancer demo: K−1 reference at sigma_init=5 (init-robustness sweep)

Second cell of the `reference_topic` default-decision grid (insight 0029
Ablation 2, ADR 0031). **Identical to exp 0010 (full-K, sigma_init=5, 300 iter)
except `reference_topic: true`.** Same cohort, covariates, K=40, seed 42.

## Why

Full-K at sigma_init=5 (exp 0009/0010) gave the usable phenotypes but drove Σ to
~10^10 — a near-improper prior, crisp topics achieved as a byproduct of softmax
saturation, not a real covariance (insight 0029). The reference parameterization
should reach the *same* crisp regime **without** the Σ blowup, because pinning
topic 0's η removes the translation freedom that the saturation feedback loop
exploits.

## Hypothesis

`reference_topic=True` at sigma_init=5 matches full-K 0010's topic quality
(NPMI ≈ +0.216, ~28 distinct phenotypes) with Σ bounded to O(1–100) instead of
10^10.

## What to watch (vs exp 0010, full-K sigma_init=5)

- **Σ[min … max] trace** — the key contrast: does reference keep Σ bounded where
  0010 ran to ~2.2e10?
- **NPMI mean + phenotype list** — at least as crisp as 0010.
- **Comparison with 0012** — quality and Σ should be roughly flat between
  sigma_init=1 (0012) and =5 (0013) if reference is genuinely init-robust; full-K
  was a knife-edge between these two (0008 collapse vs 0010 escape).

## Decision

Together with 0012 and 0014, establishes whether reference flattens the
sigma_init sensitivity. Flat quality + bounded Σ across {1, 5, 10} → default
reference on. See exp 0012 for the full decision rule.

## Run

```
make exp ID=13
```
