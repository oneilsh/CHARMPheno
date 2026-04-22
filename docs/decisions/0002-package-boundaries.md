# 0002 — Package boundaries: generic framework + clinical wrapper

**Status:** Accepted
**Date:** 2026-04-22

## Context
Both `spark-vi` and `charmpheno` have a claim on the Online HDP model. The
framework docs position `spark-vi` as reusable beyond clinical use; the
research docs position `charmpheno` around HDP-based phenotype discovery.

## Decision
- `spark-vi` ships a **generic, domain-agnostic** `OnlineHDP` suitable for any
  bag-of-words / topic-model workload. It has no clinical or OMOP semantics.
- `charmpheno` wraps the generic HDP with OMOP-shaped I/O, concept-vocabulary
  handling, recovery-metric evaluation, downstream export, and per-patient
  profile construction.
- `spark-vi` must **never import `charmpheno`** or any clinical / BigQuery
  code. The dependency direction is one-way.

## Alternatives considered
- Clinical HDP in `spark-vi` directly (rejected: conflates framework with
  application; a non-clinical user would pull in OMOP code they don't need).
- Pure framework with no model (rejected: every model author would
  re-implement the HDP orchestration, defeating the reuse goal).

## Consequences
- OU or other future models can be added in `charmpheno` as demonstrations
  of extending `VIModel`, without destabilizing the framework.
- The `spark-vi` public API must stay stable enough to be depended upon.
