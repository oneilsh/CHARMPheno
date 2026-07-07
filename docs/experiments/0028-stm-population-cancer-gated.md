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

## Sex covariate

Exercises the `~ C(sex)` term. Sex is read from
`person.sex_at_birth_concept_id` (standard OMOP 8507 Male / 8532 Female) and
decoded by concept name via `decode_sex_from_name`
([bigquery.py](../../charmpheno/charmpheno/omop/bigquery.py)) — NOT from
`gender_concept_id`, which in the AoU CDR holds *gender identity* (custom
concepts 45878463 / 45880669 / 1585841 / 2000000002 / ...), so an id-based
decoder collapsed every person to a single 'Unknown' level and dropped C(sex)
(the exp 0027 symptom; surfaced here by the covariate level diagnostic).
`known_sex_only: true` restricts the fit to persons with a decoded binary
sex M/F (dropping Unknown/other) via the inner corpus⋈covariate join. Verify
the `covariate level diagnostics` phase reports `sex_at_birth_concept_id`
{8507, 8532, ...} and a 2-level `sex` {M, F}.

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

## Certification (fast-downdate reliability)

The predictive-gain panel's per-topic aggregates (presence, mean_gain, depth,
prominence, dedup_gain) are computed with a fast one-Newton-step "downdate"
approximation (`fast=True`), not the exact per-document cold solve. A small
in-memory sample is checked cold-vs-fast by `predictive_gain_downdate_audit`
([predictive_gain.py](../../spark-vi/spark_vi/mllib/topic/predictive_gain.py#L1192-L1197))
and the audit's `max_abs_overall` and `mean_abs_overall` ship into
`phenotypes.json`'s `predictive_gain.downdate_audit`. This is a
re-certification of that fast path against the actual 0028 corpus — no
re-fit, just a fresh export.

Run (from `analysis/cloud` on the Dataproc master):

```
git pull && git rev-parse --short HEAD    # confirm current
export BUILD_ETA_SCALE_OVERRIDE=4.6       # pins c*, SKIPS the ~13-min held-out
                                           #   sweep -- only needs to be a
                                           #   stable, reasonable scale for
                                           #   this check, not a fresh calibration
make build-dashboard-exp ID=28 2>&1 | tee ~/build_0028.log
```

What to confirm in the log:

- `eta_scale: OVERRIDE=4.6 (BUILD_ETA_SCALE_OVERRIDE set; skipping the
  held-out sweep)`
- `predictive_gain: smoothed score active (lambda=1.0)`
- `BUNDLE WRITTEN: <path> (mtime ...)` — and the shell returns immediately
  (no teardown hang). `cp` the bundle to a uniquely-named file before
  downloading it (repeated runs overwrite the default path).

What to read in `phenotypes.json` → `predictive_gain`:

- `smoothing.active` must be `true` (the marginal-backoff smoother engaged).
- `downdate_audit.mean_abs_overall` vs `downdate_audit.max_abs_overall` — the
  certification itself. `max_abs_overall` is a single worst-case document;
  `mean_abs_overall` averages the audit's per-topic mean discrepancy over
  finite entries. If `mean_abs_overall` is SMALL (much less than 1 nat) while
  `max_abs_overall` is large, the fast downdate disagrees with the exact cold
  solve only on a handful of pathological documents — the per-topic
  aggregates (means over the corpus's ~48k docs) are trustworthy as-is. If
  `mean_abs_overall` is itself large (order-nats), the fast path is broadly
  biased and the aggregates are suspect — fall back to the exact cold solve
  (or a hardened multi-step downdate) before trusting them.

This run also exercises the smoothed predictive score, the uniform marginal
floor, and the bundle provenance stamp — all already shipped, not new here.

## Related

Builds on exp 0027 (block-wise unit-diagonal Σ, gated comorbid), insight 0030
(reference + spectral default stack), ADR 0034 (block-wise correlation Σ).
