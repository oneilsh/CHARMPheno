# 0003 — Explicit OMOP I/O primitives (no environment sniffing)

**Status:** Accepted
**Date:** 2026-04-22

## Context
Code must run in two environments: local dev (parquet files) and a cloud
notebook environment (BigQuery). A natural temptation is a single
`load_omop(...)` function that sniffs the environment and dispatches.

## Decision
Expose two explicit, narrow loaders in `charmpheno.omop`:
- `load_omop_parquet(path, *, spark) -> DataFrame`
- `load_omop_bigquery(*, spark, cdr_dataset, ...) -> DataFrame`

Neither sniffs the environment. The caller imports and invokes the one it
wants. Both return the canonical OMOP-shaped Spark DataFrame:
`person_id, visit_occurrence_id, concept_id, concept_name`.

A `charmpheno.omop.validate(df)` function asserts shape and fails loudly.

## Alternatives considered
- Environment-switched single loader (rejected: hides behavior, invites
  magic, complicates testing, makes I/O bugs much harder to localize).

## Consequences
- Scripts carry one extra import line declaring their environment.
- Library behavior is straightforward to reason about without needing to
  know which env vars were set.
- Caller code visually declares its target environment.
