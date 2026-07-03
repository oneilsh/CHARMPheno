---
id: 32
slug: stm-population-cancer-estimated-sigma-diagonal
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
subsampling_rate: 0.1
tau0: 128
kappa: 0.7
sigma_init: 1.0
reference_topic: true
spectral_init: true
spectral_method: dense
min_pair_support: 10
estimate_sigma_diagonal: true
---

# Experiment 0032 — Population-cancer gated STM with estimated Σ diagonal

## Goal

Re-fit exp 0028's config (population background + cancer foreground, gated
STM) with the block-wise Σ M-step's diagonal **estimated** instead of pinned
to 1 (`estimate_sigma_diagonal: true`). This tests whether, WITH the
stabilizers exp-0028 already uses (reference topic + dense spectral init),
the per-topic variance recovers the natural η-scale STABLY — i.e. bounded,
not runaway — rather than needing the ADR 0034 unit-diagonal pin.

## Why

ADR 0034 pins Σ_ii = 1, removing the variance degree of freedom that drove
the softmax-saturation runaway (insight 0033). That pin is safe but discards
the per-topic η-variance scale entirely, and that scale is exactly what the
generative record-completion simulator needs: fitting under the unit-diagonal
pin and trying to recover a variance scale post-hoc leaves the simulator's
patients too concentrated (the exported η-variance comes out ~10x too small,
unrecoverably, since the pin never estimated it in the first place).

Insight 0030 (exp 0015) showed that reference-topic + spectral-init keeps Σ
**bounded** at a proper natural scale (~7.56) even WITHOUT pinning — the
runaway insight 0033 diagnosed was a symptom of missing those stabilizers,
not of estimating the diagonal per se. Since exp 0028 already fits with both
stabilizers on, this experiment tests whether estimating the diagonal here
also stays bounded and stable, recovering a usable η-scale for the simulator.

## Configuration

Identical to exp 0028 (population background, cancer foreground, K=60 =
40 background + 20 foreground, `~ C(sex) + age`, dense spectral init,
reference topic, `min_pair_support: 10`), with one change:

| Field | Value | Note |
|---|---|---|
| `estimate_sigma_diagonal` | true | keep the estimated per-topic variance on the Σ diagonal (block-wise covariance) instead of pinning to 1 |

All other fields (K, background_k, sigma_init, reference_topic,
spectral_init, spectral_method, min_pair_support, cohort, cache_uri,
covariate_formula, schedule) are unchanged from exp 0028.

## Success criteria

- Fit completes without the softmax-saturation runaway (insight 0033) that
  motivated ADR 0034 — i.e. Σ_ii settles near the insight-0030 natural scale
  (~7.6) and stays there across iterations, relying on the spectral init +
  reference topic stabilizers rather than a hard ceiling. The logged Σ
  min/max trace is checked per iteration for blowup visibility.
- Correlation structure (off-diagonal Σ_ij / sqrt(Σ_ii Σ_jj)) reads
  comparably to exp 0028's block-wise unit-diagonal fit — dementia/cancer
  sub-phenotype correlations preserved, no new spurious cross-block signal.
- Exported η-variance is usable directly by the record-completion generative
  simulator (no post-hoc rescaling needed) and produces concentrated,
  coherent synthetic patients (the problem this experiment exists to fix).

## Related

Builds on exp 0028 (population-cancer gated STM, config source), insight
0030 (reference + spectral stack keeps Σ bounded ~7.56 without pinning),
insight 0033 (softmax-saturation runaway without stabilizers), ADR 0034
(block-wise unit-diagonal Σ — the pin this run's `estimate_sigma_diagonal`
mode overrides, opt-in only).
