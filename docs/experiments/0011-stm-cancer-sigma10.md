---
id: 11
slug: stm-cancer-sigma10
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
sigma_init: 10.0
---

# STM cancer demo: sigma_init=10 sweep point

Third point on the sigma_init sweep (after 1=collapse, 5=escape). Only
`sigma_init` changes vs 0010 (5.0 → 10.0); `max_iter: 300` so it can converge.

## Hypothesis (and what it tests)

If sigma_init=10 converges to the SAME place as sigma_init=5 — Σ settling at
~10^10, NPMI ≈ 0.21, the same phenotypes — then sigma_init=5 was never a tuned
operating point; it was just "any init past the collapse threshold," and the
model self-selects its (degenerate, boundary-pinned) Σ. That de-magics the 5.

The deeper read (see the stm.py code review): Σ → 10^10 is NOT a meaningful
covariance — it is the unregularized residual-variance update running η toward
the softmax-saturation boundary, capped only by numerics + ELBO convergence,
not by any prior on Σ (sigma_ridge=1e-6). Full-K η (no reference topic) leaves
a likelihood-flat translation direction that only the weak prior controls. So
sigma_init=10 is expected to blow up to the same ~10^10 boundary. The principled
fix is not a better sigma_init but REGULARIZING Σ (a real sigma_ridge / prior,
shrinkage toward diagonal), and/or the K-1 reference-topic parameterization and
spectral init that the original CTM/STM use.

## Watch

- Final Σ[min … max] — same ~10^10 ballpark as 0010 (→ boundary phenomenon,
  init-independent), or materially different (→ sigma_init genuinely tunes Σ)?
- Convergence iteration + NPMI — do the phenotypes match 0010's?

## Run

```
make exp ID=11
```

Feeds the sigma_init sweep + the insight. If this confirms the boundary read,
the next experiment should test Σ regularization (a larger sigma_ridge), not
more sigma_init points.
