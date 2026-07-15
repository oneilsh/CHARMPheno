# 0025 — Charmpheno covariate sidecar parquet (separate artifact from corpus parquet)

**Status:** Accepted
**Date:** 2026-05-29
**Related:** ADR 0022 (STM-prevalence as covariate path); ADR 0024 (formulaic in MLlib shim); design spec [2026-05-29 STM prevalence-only design](../superpowers/specs/2026-05-29-stm-prevalence-design.md); ADR 0018 (document unit abstraction); ADR 0021 (producer-side vocab trim — establishes the corpus-artifact pattern this ADR extends)

## Scope

This decision applies to **the charmpheno corpus-artifact layout**: how patient-level covariates are persisted on disk alongside the BOW corpus parquet, how the `corpus_manifest.json` references them, and how the fit driver wires them together at fit time.

**It does not apply to the spark-vi side.** On the spark-vi side covariates are bundled per-row into the `STMDocument` type (`indices`, `counts`, `length`, `x`); there is no analogous "sidecar vs baked-in" choice because the engine has no notion of an artifact-layout layer at all. The engine receives pre-joined rows; the join happens upstream in the MLlib shim's row-conversion code.

## Context

Charmpheno's existing corpus pipeline produces a single parquet artifact per run, written by `charmpheno.omop.topic_prep.fit_bow_pipeline`. The parquet schema is `(person_id, doc_id, features: SparseVector)`, with a `corpus_manifest.json` capturing the build's effective config (`cohort_def`, `doc_spec`, `min_df`, vocab map, run UUID, source timestamps). The build is expensive: a BigQuery scan over the cohort's clinical events, an aggregation pass to produce per-doc bags, and a `CountVectorizer.fit` over the corpus to learn the vocab.

STM (ADR 0022) needs an additional per-document covariate vector `x_d`. The covariate values are produced by applying a formula (ADR 0024) to a patient-level covariate DataFrame — typically a BigQuery query over the OMOP `person` table plus any condition / observation lookups the formula references.

Two artifacts now want to coexist:

- The corpus parquet (`(person_id, doc_id, features)`) — expensive to build, stable across experiments with the same cohort definition and tokenization.
- The covariate data (`(person_id, x_d)`) — cheap to rebuild, varies with the formula choice.

The shapes are different: corpus rows are per-document; covariates are typically per-patient (a patient with 100 docs has one age, one sex, one cohort assignment). They join by `person_id` to produce per-document covariate rows at fit time.

Three persistence layouts are plausible:

1. **Baked-in**: corpus parquet schema becomes `(person_id, doc_id, features, x)` — covariates embedded per row.
2. **Sidecar**: covariates live in a separate parquet (`patient_covariates.parquet`) in the same run directory; corpus parquet is unchanged. Joined by `person_id` at fit time.
3. **Runtime-only**: covariates are computed from a fresh BigQuery query at each fit invocation; never persisted.

## Decision

CharmPheno uses **layout 2 — the covariate sidecar parquet**.

### Run directory layout

```
runs/<run_uuid>/
  corpus_manifest.json          # gains a covariate_sidecar field
  corpus.parquet                # unchanged: (person_id, doc_id, features)
  patient_covariates.parquet    # NEW: (person_id, x: DenseVector)
  model_spec.pkl                # NEW: formulaic ModelSpec for round-trip
  ...                           # vi_result, fit_log, dashboard_bundle, etc.
```

### Schema additions

- **`patient_covariates.parquet`**: schema `(person_id: long, x: DenseVector)`. One row per distinct `person_id` appearing in the corpus. Sidecar may have a superset of person_ids if the patient table covers patients with no docs; the join drops them.

- **`corpus_manifest.json`**: gains a `covariate_sidecar` field. Set to a relative path (e.g., `"patient_covariates.parquet"`) when covariates are present; `null` for LDA / HDP runs. Backward-compatible: existing readers ignore unknown fields.

- **`model_spec.pkl`** (or equivalent formulaic-native serialization): persists the formulaic `ModelSpec` so transform-time inputs and the dashboard adapter can apply the same encoding without re-fitting.

### Build pipeline

Two helpers in charmpheno:

- `charmpheno.omop.covariates.build_patient_covariate_sidecar(spark, person_df, formula, out_path)`:
  1. Selects columns referenced by the formula from `person_df` (sourced from BigQuery via the same `_corpus_load` machinery the corpus pipeline uses).
  2. Invokes the MLlib shim's formula-spec fitting helper (ADR 0024's schema-frame discovery) to produce a `ModelSpec`.
  3. Applies the `ModelSpec` per partition to produce per-person `x` vectors.
  4. Writes `(person_id, x)` to parquet.
  5. Persists the `ModelSpec` alongside.

- Fit driver (`analysis/cloud/stm_bigquery_cloud.py`, ADR 0022-scope):
  1. Loads the corpus parquet via `_corpus_load`.
  2. Loads the sidecar parquet.
  3. Broadcast-joins sidecar to corpus by `person_id` (sidecar is small — millions of patients × P float64s = MB-scale, fits comfortably in driver memory and broadcasts cleanly).
  4. Constructs `StreamingSTM` via the MLlib shim's pre-built-`covariates_col` path (ADR 0024's Path A).
  5. Fits via `VIRunner`.

### Experiment-tracking integration

The experiment-tracking wrapper (`scripts/run_experiment.py`) needs to know whether to build a sidecar before fit. Recommendation: **auto-build sidecar if missing**. The fit driver checks the manifest's `covariate_sidecar` field; if null but the experiment's defaults YAML specifies `covariate_formula`, the driver builds the sidecar inline, updates the manifest, and proceeds. This keeps the user's command surface to `make exp ID=N` without an extra build step.

A separate `make build-covariates EXP=N` target is available for explicit re-builds (e.g., when the formula changes but the corpus is unchanged).

## Why a sidecar and not the alternatives

### vs. baked-in (layout 1)

The corpus parquet is **expensive to build** and **stable across experiments**. The covariate vector is **cheap to compute** and **varies with the formula**. Coupling them is asymmetric:

- A user iterating on covariate formulas (`~ age + sex` → `~ age + sex + cohort` → `~ age + sex + cohort + sex:cohort`) would have to rebuild the corpus three times. The corpus build dominates total wall-clock; the covariate build is negligible.
- A user iterating on `min_df` or `doc_spec` rebuilds the corpus naturally, but should not also have to recompute covariates that don't depend on those choices.
- Multiple experiments can share the same corpus parquet while using different sidecars (e.g., the cancer cohort with `~ age + sex` and the cancer cohort with `~ age * stage`).

The sidecar layout respects the asymmetry. Baking-in does not.

### vs. runtime-only (layout 3)

Three problems:

- **Reproducibility.** The formula and the discovered factor levels are part of the fit's effective config. Persisting the sidecar ensures a later re-run of the same experiment produces bit-identical covariate vectors; runtime-only re-queries BigQuery, which may return different rows over time (new patients, updated cohort definitions).
- **The `ModelSpec` needs to round-trip anyway.** Transform-time inputs and the dashboard adapter need the factor level mapping. We need a persistent artifact for the ModelSpec regardless; making the covariate values match what the ModelSpec was fit against — by persisting them in the same place — is cheap.
- **Compute cost.** Re-querying BigQuery on every fit invocation is a real cost (egress, query slot, latency). Persisting the sidecar amortizes that over multiple experiments per cohort.

## Alternatives considered

1. **Baked-in covariates** (layout 1 above). Rejected per the asymmetry argument: couples expensive corpus build to cheap covariate iteration.

2. **Runtime-only computation** (layout 3 above). Rejected per the reproducibility and cost arguments.

3. **Single sidecar per cohort, shared across runs.** Place `patient_covariates.parquet` at the *cohort* level rather than the *run* level (`runs/<cohort_id>/patient_covariates.parquet` instead of `runs/<run_uuid>/patient_covariates.parquet`). Rejected for v1 because the formula varies per experiment, and a per-cohort sidecar would either need to encode all possible formula outputs (impossible) or be tied to a single formula choice per cohort (defeats the point). A per-run sidecar trivially supports multiple formulas per cohort. If two runs use the same formula and same cohort, they get identical sidecars — that's redundant storage but cheap relative to anything else in the run directory.

4. **Embed covariates as a column inside `corpus_manifest.json` metadata.** Rejected: `x_d` is per-document data, not metadata. `corpus_manifest.json` is small JSON and not a parquet column; embedding millions of `x_d` vectors would inflate the manifest from KB to GB.

5. **Per-doc covariates in the sidecar** (sidecar schema `(person_id, doc_id, x)` instead of `(person_id, x)`). Rejected for v1 because the supported covariate space (patient-level: age, sex, cohort) is per-patient, not per-doc. The per-doc join trivially replicates the patient row across the patient's docs. A v1.x extension for doc-level covariates (e.g., "visit type") would change the sidecar schema to `(person_id, doc_id, x)` and key the join differently; the manifest field would still point to the sidecar regardless.

## Consequences

- **The corpus parquet schema is unchanged.** Existing LDA / HDP runs continue to work with no migration. The manifest gains an optional field; readers tolerate its absence.

- **A new charmpheno helper module** (`charmpheno.omop.covariates`) holds the sidecar builder. Its inputs are the formula and a person-level DataFrame; its outputs are the sidecar parquet and the persisted `ModelSpec`.

- **The fit driver does a broadcast join** at fit start. The sidecar is small (patients × P × 8 bytes; for 10M patients × 20 covariates ≈ 1.6 GB driver-side memory, well within broadcast limits at typical cluster sizes; for cohort fits more like 100K patients × 20 covariates ≈ 16 MB, trivial). If a cohort ever pushes past broadcast limits, the join falls back to a shuffled join with a one-line code change.

- **The dashboard bundle gains the `ModelSpec`** (in adapted form — labels and term names for Γ's rows) so the dashboard can label coefficients correctly. The `adapt_stm` adapter (design spec phase 6) reads `ModelSpec` from the run directory.

- **Two parquet files per run** that must stay together for STM runs. The run directory structure makes this manageable; the manifest's `covariate_sidecar` pointer documents the relationship.

- **Sidecar regeneration is cheap.** Same formula, same `person_df` → identical sidecar (the ModelSpec discovery is deterministic given the sorted level set). A user can `make build-covariates EXP=N` to refresh after a formula change.

- **No spark-vi-side change.** The engine's stateless contract is unchanged. The covariate sidecar is purely a charmpheno persistence concern.

## Open follow-ups

- Per-cohort vs per-run sidecar deduplication: if storage proves a concern, a content-addressable layout (`runs/<run_uuid>/patient_covariates.parquet` → symlink or pointer to `cohorts/<cohort_id>/<formula_hash>/patient_covariates.parquet`) could deduplicate identical sidecars across runs. Not done in v1; revisit if a deployment surfaces the cost.

- **Doc-level covariates** (v1.x extension): sidecar schema becomes `(person_id, doc_id, x)` and the join keys differently. The manifest field and driver wiring are unchanged.

- **Sidecar build performance** if formula expansion produces high-cardinality categoricals: the schema-frame discovery's per-column `select distinct` is one shuffle per categorical. v1.x optimization could fuse into one `df.agg(F.collect_set(c1), F.collect_set(c2), ...)` pass (mentioned in ADR 0024).

- **Cross-cohort sidecar reuse**: two cohorts with overlapping patient sets and the same formula could share a single sidecar. Possible but not designed; out of scope for v1.
