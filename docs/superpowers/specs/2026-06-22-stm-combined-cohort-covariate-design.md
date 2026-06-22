# Combined cancer/dementia cohort with a source covariate — Design

**Date:** 2026-06-22
**Status:** Brainstorm-grade design, awaiting user review.
**Scope:** Add a combined cancer-or-dementia corpus with a binary `source_cohort` covariate, as an STM validation experiment. Validates that STM prevalence covariates produce strongly-separable topic structure. All work is confined to charmpheno (OMOP corpus build, covariate build, fit-driver join, experiment config). The spark-vi engine and MLlib shim are untouched; the dashboard bundle stays fully aggregate.

---

## Context

STM (prevalence-only) shipped on the `stm` branch but has never been cluster-validated, and the faithful corpus-mean `corpus_prevalence` was just implemented. We want a first STM experiment whose covariate effect is large and interpretable, so a successful run is unambiguous evidence the prevalence covariates work end to end.

The chosen design: merge the existing `first_cancer_year` and `first_dementia_year` cohorts into one corpus and add a `source_cohort` covariate indicating which cohort each document came from. Cancer and dementia post-onset phenotype cascades are clinically distinct, so a working STM should place strong, separable mass on cancer topics for `source_cohort = cancer` documents and dementia topics for `source_cohort = dementia`. The Γ matrix becomes the readout.

Two cohort functions already exist in [cohorts.py](../../../charmpheno/charmpheno/omop/cohorts.py), each performing inclusion + a 365-day post-index window + observation-period bracketing. This design composes them rather than reinventing cohort logic.

## Decisions made during brainstorming

- **Comorbid patients contribute two documents**, one per cohort, each labeled by its source. Not dropped, not merged. Rationale: keeps all data; each document is an independent STM observation; the "so be it" on shared-window codes is acceptable for a validation.
- **`source_cohort` is therefore a per-document covariate**, not per-person. A comorbid patient has a cancer-labeled document and a dementia-labeled document.
- **Composite join key** `[person_id, source_cohort]` rather than remapping `person_id` to a synthetic string. Keeps `person_id` a clean integer in-enclave; the covariate build emits one row per `(person_id, source_cohort)`.
- **Covariate formula:** `~ C(source_cohort) + C(sex) + age`. Exercises multi-covariate STM (a categorical of interest plus the standard age/sex adjustment).
- **Layering:** all changes are charmpheno-side and upstream of the join. The spark-vi shim consumes only the `features` and `covariates` columns and never sees ids; the engine sees only `STMDocument`. Confirmed: no `person_id`/`doc_id` reference exists anywhere in `spark-vi/`.
- **Privacy boundary unchanged.** `person_id` (real or composite) is in-enclave only. The dashboard bundle is five aggregate JSON files and a synthetic-patient simulator; no patient-level identifiers leave the enclave. The composite key does not change what crosses the boundary.

## Components

### 1. Combined cohort — [cohorts.py](../../../charmpheno/charmpheno/omop/cohorts.py)

New cohort registered in `SUPPORTED_COHORTS` (name `cancer_or_dementia`):

- Runs the existing `first_cancer_year` and `first_dementia_year` filters.
- Tags each cohort's surviving events with a `source_cohort` string column ("cancer" / "dementia").
- Unions the two tagged event sets. No overlap removal: a comorbid patient's cancer-window events (tagged cancer) and dementia-window events (tagged dementia) both appear.
- Adds `COHORT_METADATA` and label entries to keep the registry the single source of truth.

The membership/label derivation (which `person_id`s are cancer, which are dementia) flows from the same two cohort functions, so the corpus and the covariate build cannot disagree on labels.

### 2. `doc_id` encodes `source_cohort` — corpus build path

A comorbid patient's two documents must survive the corpus `groupBy(doc_id)` in [topic_prep.py](../../../charmpheno/charmpheno/omop/topic_prep.py) as two distinct documents. Today `doc_id` is `person_id` (patient) or `"{person_id}:{year}"` (patient_year) — neither encodes cohort, so two windows touching the same year would silently merge into one ambiguously-labeled document.

Fix: in the combined-cohort corpus path, fold `source_cohort` into `doc_id` (for example `"{source_cohort}:{spec_doc_id}"`), and preserve `source_cohort` through the BOW aggregation so the output is `bow_df = (person_id, doc_id, source_cohort, features)`. A single joint vocabulary is fit over the whole combined corpus so cancer and dementia BOW vectors share a feature index.

### 3. Per-(person, cohort) covariates — [covariates.py](../../../charmpheno/charmpheno/omop/covariates.py) and person load

The person table (age from `year_of_birth`, sex from `gender_concept_id`) is expanded to one row per `(person_id, source_cohort)` by joining the cohort labels — a comorbid person appears twice, age/sex repeated. `build_patient_covariate_df` then applies `~ C(source_cohort) + C(sex) + age`, producing `cov_df = (person_id, source_cohort, covariates)`.

### 4. Composite join — [stm_bigquery_cloud.py](../../../analysis/cloud/stm_bigquery_cloud.py)

Change the broadcast join from `on="person_id"` to `on=["person_id", "source_cohort"]`, so each document receives the covariate row matching its label. Both frames carry `source_cohort` after components 2 and 3.

### 5. Experiment + config

A new experiment markdown `docs/experiments/NNNN-stm-cancer-dementia.md` with frontmatter: `model_class: stm`, `cohort: cancer_or_dementia`, `covariate_formula: "~ C(source_cohort) + C(sex) + age"`, `categorical_cols: [source_cohort, sex]`, `continuous_cols: [age]`, `random_seed` set, and `cache_uri`. `cache_uri` must reach the merged config so the STM fit persists the covariate sidecar and the dashboard build computes the faithful `corpus_prevalence`; mirror the corpus-cache convention (`hdfs:///user/dataproc/charm/covariates_cache`).

## Data flow

```
OMOP --> cancer_or_dementia (tag source_cohort) --> joint vocab --> BOW (doc_id encodes cohort)  --+
                                                                                                   +-- join on [person_id, source_cohort] --> STMDocument --> fit
OMOP --> person table --> x {cohort labels} --> build_patient_covariate_df ------------------------+
```

## Testing

- Combined cohort: source tags correct; union counts = cancer count + dementia count (comorbid counted in both).
- `doc_id` uniqueness: a synthetic comorbid patient whose two windows fall in the same year yields two distinct `doc_id`s, not one merged document.
- Covariates: one row per `(person_id, source_cohort)`; comorbid person duplicated; formula columns present.
- Composite join cardinality: each document joins to exactly its labeled covariate row.
- STM smoke extension: add a comorbid patient; assert two documents with correct labels and that the fitted Γ shows opposite-sign source-cohort loadings on cancer-leaning vs dementia-leaning topics.

## Out of scope

- A three-way "both" covariate level (rejected in favor of two documents per comorbid patient).
- Per-document covariates beyond `source_cohort` (age/sex remain per-person, repeated across a person's documents).
- Dashboard frontend Γ visualization (its own brainstorm).
- Splines / content covariates.
