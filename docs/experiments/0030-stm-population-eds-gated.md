---
id: 30
slug: stm-population-eds-gated
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
K: 60
background_k: 40
foreground: "eds:20"
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
---

# Experiment 0030 — Population-background + Ehlers-Danlos-foreground gated STM

## Goal

The rare-disease-on-a-background-population case: a gated STM whose background
is the whole population and whose single foreground group is Ehlers-Danlos
syndrome (EDS). Same asymmetric gated architecture as exp 0028 (population +
cancer), swapping the cancer anchor for a much rarer disease to test that the
foreground block recovers a recognizable EDS comorbidity signature (e.g. POTS /
dysautonomia, GI dysmotility, chronic pain, joint hypermobility / connective
tissue) rather than collapsing into the background.

## Cohort

New `population_eds` cohort (built on the generalized
`apply_population_disease_cohort`, disease registry key `eds`, OMOP ancestor
79145, no exclusions):

- **eds** (`source_cohort='eds'`): persons with a first EDS diagnosis, windowed
  to the 365 days after that diagnosis; carries the 20 EDS foreground topics.
- **general** (`source_cohort='general'`): every other person, windowed to a
  deterministic random 365-day event-anchored span; background-only.

`prior_obs_days: 0` admits prevalent EDS cases to maximize a rare arm.

## Configuration

Full population (`person_mod: 1`) — the heaviest fit in this batch — to maximize
the EDS foreground and give a rich background. K=60 = 40 background + 20 EDS.
Otherwise the exp 0028 gentle + hardened stack (subsample 0.1, tau0 128, 200
iter, reference + dense spectral, sigma_init 1, min_pair_support 10, block-wise
unit-diagonal Σ / ADR 0034), `~ C(sex) + age`, `known_sex_only`.

## Success criteria

- Covariate diagnostics show a realistic 2-level sex distribution.
- EDS foreground topics recover recognizable EDS-associated phenotypes; the EDS
  arm has enough documents (check corpus diagnostics — if thin, revisit
  `person_mod`).
- Σ variance bounded (no runaway); honest correlation report.

## Related

Follows exp 0028 (population + cancer gated). First cohort on the generalized
population+disease registry.
