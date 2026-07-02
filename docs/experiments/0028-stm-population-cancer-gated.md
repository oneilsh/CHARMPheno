---
id: 28
slug: stm-population-cancer-gated
status: pending
model_class: stm
cohort: population_cancer
cohort_def: population_cancer
prior_obs_days: 0
person_mod: 4
doc_unit: patient_cohort
covariate_formula: "~ C(sex) + age"
categorical_cols: [sex]
continuous_cols: [age]
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
---

# Experiment 0028 — Population-background + cancer-foreground gated STM

## Goal

Fit a gated STM whose **background is the whole (sampled) population** and whose
single **foreground group is cancer**. This tests the gated architecture in its
intended asymmetric regime (a large common cohort informing the shared
background, a rarer subgroup carrying its own foreground topics — see
`TopicBlockPartition.allowed_indices`) rather than the balanced two-foreground
setup of exp 0025–0027 (cancer_or_dementia).

## Cohort

New `population_cancer` cohort
([cohorts.py](../../charmpheno/charmpheno/omop/cohorts.py)), disjoint and one
document per person:

- **cancer** (`source_cohort='cancer'`): patients with a first malignant-cancer
  diagnosis (SNOMED 443392 and descendants, excluding non-melanoma skin cancer
  and carcinoma in situ), windowed to the 365 days after that diagnosis. These
  documents carry the 20 cancer foreground topics.
- **general** (`source_cohort='general'`): every other person, windowed to a
  deterministic random 365-day span **anchored on one of their own
  condition-eras** whose forward year is fully observed (min
  `hash(person_id, event_date)` pick — reproducible, not `F.rand()`).
  `'general'` is not a foreground group, so these documents resolve to
  background-only.

  A random *calendar* window was tried first and collapsed the general arm
  (~12k docs) because EHR coding is bursty over long observation periods — a
  random year usually lands in a quiet stretch, so the document falls below
  `doc_min_length` and is dropped. Anchoring on the person's own coding
  guarantees the window contains real activity and recovers the population.

The general arm's window ignores `prior_obs_days` (there is no diagnosis index
to be "first" of); the cancer arm uses it. `prior_obs_days: 0` here admits
prevalent cancer cases (maximizing the cancer arm on the 25% sample); flip to
365 for an incident-only cancer definition.

## Configuration

| Field | Value | Note |
|---|---|---|
| `person_mod` | 4 | 25% sample (person_id % 4 == 0) |
| K | 60 | 40 background + 20 cancer foreground |
| `covariate_formula` | `~ C(sex) + age` | prevalence-only STM |
| `subsampling_rate` | 0.1 | halved from 0.2 |
| `tau0` | 128 | doubled from 64 (gentler Robbins-Monro warm-up) |
| `max_iter` | 200 | doubled from 100 |
| hardening | reference + dense spectral, sigma_init 1, min_pair_support 10 | validated stack (insight 0030) |

The slower schedule (smaller minibatches + larger `tau0` + twice the iterations)
is a deliberately gentle fit for the larger, more heterogeneous whole-population
corpus.

## Gender covariate

Exercises the `~ C(sex)` term on a fresh covariate cache. `decode_sex`
([bigquery.py](../../charmpheno/charmpheno/omop/bigquery.py)) maps
8507→M / 8532→F / else→Unknown; a prior comorbid bundle showed sex collapsed to
F-only from a stale pre-fix covariate cache, so verify the fit's covariate
diagnostics report a realistic M/F/Unknown split (not F=100%).

## Success criteria

- Covariate diagnostics show a realistic sex distribution (not F-only).
- Background topics read as general-population comorbidity structure; the 20
  cancer foreground topics recover recognizable cancer sub-phenotypes.
- Σ variance bounded (block-wise unit-diagonal estimator, ADR 0034); no runaway.
- Honest correlation report with cross-block NA where unsupported.

## Caveats

- General-population documents are filtered by `doc_min_length` (20, from
  `_base.yaml`), so the background is trained on persons with substantial coding
  activity in their sampled year — a mild "sicker/more-coded" skew relative to
  the true general population. Lower `doc_min_length` to broaden it.

## Related

Builds on exp 0027 (block-wise unit-diagonal Σ, gated comorbid), insight 0030
(reference + spectral default stack), ADR 0034 (block-wise correlation Σ).
