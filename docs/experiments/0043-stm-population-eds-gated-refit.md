---
id: 43
slug: stm-population-eds-gated-refit
status: pending
model_class: stm
cohort: population_eds
cohort_def: population_eds
prior_obs_days: 0
person_mod: 1
doc_unit: patient_cohort
doc_min_length: 10
covariate_formula: "~ C(sex) + age"
categorical_cols: [sex]
continuous_cols: [age]
known_sex_only: true
random_seed: 42
cache_uri: hdfs:///user/dataproc/charm/covariates_cache
K: 100
background_k: 80
foreground: "eds:20"
group_var: source_cohort
max_iter: 300
subsampling_rate: 0.1
tau0: 256
kappa: 0.7
sigma_init: 1.0
reference_topic: true
spectral_init: true
spectral_method: dense
min_pair_support: 10
---

# Experiment 0043 — Population + Ehlers-Danlos gated STM (re-fit of 0030)

## Goal

Re-fit the `population_eds` cohort from exp
[0030](0030-stm-population-eds-gated.md) under a new id — so 0030's artifact is
preserved — with two changes:

1. **Bigger background** — `background_k` 40 → 80 (K = 100 = 80 background + 20
   EDS). 0030 resolved 191,872 background documents against only 40 topics; the
   extra capacity should split the general-population comorbidity atlas more
   finely instead of packing multiple comorbidity clusters into one topic.
2. **Slower schedule** — `tau0` 128 → 256 and `max_iter` 200 → 300, because 0030
   converged early (iter 89). A gentler Robbins-Monro warm-up (ρ_t = (τ0+t)^−κ)
   lets the larger corpus + larger K settle more gradually.

Plus the latest model-engine updates on `stm` since 0030 was fit (e.g. the
export-time generative-scale / eta_scale calibration and predictive-gain
smoother work) — the re-fit + re-export picks those up.

## Cohort

Unchanged from 0030 — `population_eds` (generalized
`apply_population_disease_cohort`, disease registry key `eds`, OMOP ancestor
79145, no exclusions), one document per person, `source_cohort ∈ {eds, general}`,
365-day windows, `prior_obs_days: 0`. The corpus + covariate caches are identical
to 0030 (same cohort / `person_mod` / `doc_min_length` / vocab), so only the fit
differs — no `build-covariates` rebuild required.

## Configuration

Full population (`person_mod: 1`). K=100 = 80 background + 20 EDS. Slowed schedule
(tau0 256, max_iter 300). Otherwise the exp 0028 hardened stack (subsample 0.1,
kappa 0.7, reference + dense spectral, sigma_init 1, min_pair_support 10,
block-wise unit-diagonal Σ / ADR 0034), `~ C(sex) + age`, `known_sex_only`.

## Success criteria

- EDS foreground recovers the same recognizable sub-phenotypes 0030 found (POTS /
  dysautonomia, MCAS, joint instability, vascular EDS, GI dysmotility) — the
  larger K should not disturb the foreground.
- The 80-topic background splits the comorbidity atlas more finely than 0030's 40
  without producing near-empty topics (watch for very low Σλ topics — a signal
  that 80 is a touch generous).
- Σ variance bounded (no runaway); honest correlation report.

## Related

Re-fit of exp [0030](0030-stm-population-eds-gated.md) (which stays the preserved
K=60 artifact). Cohort + rare-disease finding: insight
[0035](../insights/0035-rare-disease-gated-foreground-recovers-eds-subphenotypes-on-full-population.md).
