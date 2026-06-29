---
id: 12
slug: stm-cancer-reference-sigma1
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
sigma_init: 1.0
reference_topic: true
---

# STM cancer demo: K−1 reference at the default sigma_init=1 (decisive cell)

The load-bearing test for whether `reference_topic` should become the default
(ADR 0031, insight 0029 Ablation 2). **Everything is identical to exp 0008
(the full-K baseline) except `reference_topic: true`** — same cohort,
covariates, K=40, seed 42, sigma_init=1.0. Differs from 0010 only by
sigma_init (1.0 vs 5.0) and reference_topic. The covariate cache is reused
(reference does not change covariates).

## Why

Exp 0008 established that full-K STM at the **default** sigma_init=1 *collapses*:
2 catch-all topics hold ~77% of the mass, ~36 topics clone the corpus marginal
at E[β]≈0.0057, NPMI mean a weak +0.055, "converged" at iter 26 into a
degenerate fixed point. The usable phenotypes only appeared at sigma_init=5
(exp 0009/0010) — but at the cost of Σ running to ~10^10 (a near-improper
prior) and a hand-tuned, off-default knob.

Insight 0029 traced the collapse to the softmax translation degeneracy:
softmax(η) = softmax(η + c·1) leaves η's overall level unidentified, and a tight
Σ≈1 prior then pins θ near-uniform. The K−1 reference parameterization removes
that degeneracy by pinning topic 0's η ≡ 0. On synthetic corpora it lifted
recovery at sigma_init=1 from 0/8 to 2/8 and kept Σ bounded. **This run tests
the same claim on real cancer data at the default sigma_init.**

## Hypothesis

`reference_topic=True` at sigma_init=1 escapes the 0008 collapse and yields
crisp, specialized phenotypes — comparable to full-K's sigma_init=5 win
(0010) — with Σ bounded (no 10^10), all at the out-of-the-box sigma_init.

## What to watch (vs exp 0008, the full-K sigma_init=1 baseline)

- **Topic specialization** — do distinct cancers separate (breast / prostate /
  thyroid / melanoma / lung / AF / HF / CKD, as in 0009/0010), or do ~36 topics
  clone the corpus marginal as in 0008? Topic 0 is the pinned baseline
  ("+ baseline comorbidity") and is expected to absorb the corpus-wide mass —
  the other 39 are the free content topics.
- **NPMI mean** — relative to 0008's +0.055 and 0010's +0.216. Crossing well
  above +0.055 is the headline.
- **Σ[min … max] trace** (ADR 0030) — does it stay bounded (O(1–100)) rather
  than collapsing toward init (0008) or blowing toward 10^10 (0009/0010)?
- **Convergence** — does it reach the ELBO criterion at a non-degenerate point,
  rather than the iter-26 collapse of 0008?
- **Peak word prob** per topic — 0008's clones peaked at ~0.003–0.006; usable
  phenotypes (the dementia-cohort LDA, insight 0028) reached 0.01–0.14.

## Decision

- 0012 escapes collapse with crisp phenotypes + bounded Σ → strong case for
  defaulting `reference_topic` on; confirm init-robustness with 0013/0014, then
  retire full-K (toggle kept for research/repro only).
- 0012 partially escapes (some specialization, weak NPMI) → reference helps but
  is not alone sufficient on real V; likely needs scalable spectral init too
  (separate arc). Keep `reference_topic` opt-in.
- 0012 still collapses → reference does not transfer from synthetic to this real
  corpus at sigma_init=1; investigate before any default flip.

## Run

```
make exp ID=12
```

Result feeds the `reference_topic` default decision and the insight 0029 / ADR
0031 follow-up. Compare head-to-head with exp 0008 (full-K, same sigma_init).
