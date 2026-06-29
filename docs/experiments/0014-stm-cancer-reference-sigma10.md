---
id: 14
slug: stm-cancer-reference-sigma10
status: superseded
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
reference_topic: true
---

# STM cancer demo: K−1 reference at sigma_init=10 (init-robustness sweep)

> **SUPERSEDED — never run.** This was the high-σ cell of the *reference-only*
> grid. exp 0013 (reference, σ=5) already showed reference alone does not tame Σ
> on real data (Σ→7e9, reference topic dead) — so a σ=10 reference-only cell
> would only confirm "still blows up" and changes no decision (insight 0030,
> Finding 1). The high-σ init-robustness question is now asked properly *with
> spectral init* in **exp 0016** (reference + spectral, σ=20). Kept for the record.

Third cell of the `reference_topic` default-decision grid (insight 0029
Ablation 2, ADR 0031). **Identical to exp 0011 (full-K, sigma_init=10) except
`reference_topic: true`.** Same cohort, covariates, K=40, seed 42, max_iter 300.

## Why

The high-sigma_init end of the sweep. For full-K, sigma_init=10 was predicted
(insight 0029) to also stall Σ at ~10^10 — i.e., the escape regime is
init-independent on the bad side: once large enough to escape collapse, Σ runs to
the saturation boundary regardless. This cell checks the mirror claim for
reference: that a *larger* init does not push Σ up, because the degeneracy that
fed the blowup is gone.

## Hypothesis

`reference_topic=True` at sigma_init=10 gives the same crisp phenotypes and the
same bounded Σ as 0012 (sigma_init=1) and 0013 (sigma_init=5) — i.e., quality and
Σ are flat across the whole sigma_init range, demonstrating the knife-edge is
removed.

## What to watch

- **Σ[min … max] trace** — bounded, and close to 0013's, NOT climbing with
  sigma_init.
- **NPMI + phenotype list** — flat vs 0012 / 0013.
- **The three-cell picture** (0012 / 0013 / 0014) side by side: if reference is
  init-robust, the rows are interchangeable; full-K's corresponding row
  (0008 collapse / 0010 blowup / 0011) is not.

## Decision

Completes the reference arm of the 2×3 grid. Init-robust + at least matching
full-K's best at every sigma_init → default `reference_topic` on and retire
full-K (toggle kept for research/repro). Full decision rule in exp 0012.

## Run

```
make exp ID=14
```
