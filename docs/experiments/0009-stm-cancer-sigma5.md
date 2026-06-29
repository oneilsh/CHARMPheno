---
id: 9
slug: stm-cancer-sigma5
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
max_iter: 100
sigma_init: 5.0
---

# STM cancer demo: higher sigma_init (collapse probe)

> **Defaults note:** the engine defaults have since flipped to `reference_topic=True`
> and `spectral_init=True` (validated default stack; see insight 0030). To reproduce
> this run exactly, pass `--no-reference-topic --no-spectral-init` (or set
> `reference_topic: false, spectral_init: false` in the run frontmatter).

Single-knob A/B against experiment 0008. **Only `sigma_init` changes**
(1.0 → 5.0); cohort, covariates, K, seed, and sampling are all identical, so
any difference in topic differentiation is attributable to the prior-variance
initialization.

## Hypothesis

Exp 0008 collapsed: 2 catch-all topics held ~77% of corpus mass and ~36 of 40
topics relaxed to the corpus marginal at nearly identical E[β] ≈ 0.0057. The
diagnostic was Σ — almost every diagonal stayed at its init value (≈1.0),
because Σ is updated from the residual variance of η_d around Γᵀx_d, and once
β is undifferentiated, η_d sits near its mean, residual variance stays ≈ init,
the prior stays tight, θ_d stays near-uniform, and β never differentiates. A
self-reinforcing symmetric fixed point (the fit "converged" at iteration 26).

The logistic-normal θ = softmax(η) has no Dirichlet-style document sparsity to
prune surplus slots, so this is partly fundamental (see insight 0028). But a
larger `sigma_init` gives η room to spread across documents *before* the
fixed point locks in, which may let topics differentiate enough to escape the
fully-degenerate basin. Expected outcome: partial improvement (more than ~4
distinct topics, lower mass concentration in the 2 catch-alls), not a cure.

## What to watch in the fit log

- `Σ[min ... max]` — does Σ grow well above the 5.0 init, or stay pinned there?
- Topic E[β] spread — is the long tail still ~36 identical topics at one E[β],
  or do more topics carve distinct vocabulary?
- Iteration at convergence — later than 0008's 26 would suggest the model
  explored more before settling.

## Run

```
make exp ID=9
```

If σ=5 sharpens topics, sweep further (σ=10) and/or combine with smaller K;
if it does nothing, that's strong evidence the collapse is fundamental to the
logistic-normal prior (reinforcing insight 0028) and the path is PLDA, not STM
tuning. Either way the result feeds the insight write-up.
