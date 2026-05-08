# ADR 0014 — Rename `VanillaLDA` to `OnlineLDA`

**Status:** Accepted
**Date:** 2026-05-08
**Related:** ADR 0008 (original Vanilla LDA design),
ADR 0011 (Online HDP design),
ADR 0012 (HDP MLlib shim).

## Context

ADR 0008 introduced the framework's first model under the name
`VanillaLDA` to signal "the basic Hoffman-2010 SVI LDA, no
optimizations" — distinguishing it from hypothetical fancier variants
within our own codebase. At the time there was no naming pressure from a
sibling implementation.

That pressure now exists. `OnlineHDP` (ADR 0011) entered the codebase
under the conventional literature name. Pairing it with `VanillaLDA`
reads as if the two are different *classes* of algorithm (a "vanilla"
one and an "online" one) when in fact both are stochastic variational
inference: Hoffman 2010 for LDA, Wang/Paisley/Blei 2011 for HDP. Both
papers self-identify in their titles as "Online" methods.

## Decision

Rename:

- `VanillaLDA` → `OnlineLDA` (model class in `spark_vi/models/lda.py`)
- `VanillaLDAEstimator` → `OnlineLDAEstimator`
- `VanillaLDAModel` → `OnlineLDAModel`
- `_VanillaLDAParams` → `_OnlineLDAParams`
- snake_case `vanilla_lda` → `online_lda` (variable names, test
  function names, etc.)
- prose "Vanilla LDA" → "Online LDA" in active documentation

Active code, tests, driver scripts, and the architecture docs are
updated. Historical records — `REVIEW_LOG.md`, ADR 0008–0013 bodies,
and `docs/superpowers/plans/` and `docs/superpowers/specs/` from past
work — are intentionally left at the old name as point-in-time
artifacts of decisions made under that name. ADR 0008 gains a one-line
"updated by ADR 0014" pointer at the top so a reader can follow the
trail.

## Alternatives considered

- **Keep `VanillaLDA`.** Avoids any rename work and the modest
  conceptual-overlap risk with Spark MLlib's `OnlineLDAOptimizer`
  (discussed below). Rejected: leaves the inconsistency with `OnlineHDP`
  permanent, and "vanilla" doesn't appear anywhere in the literature we
  cite.
- **`SVILDA` / `SVIHDP`.** Names the algorithm class precisely
  (Stochastic Variational Inference). Pairs symmetrically and
  disambiguates fully from any MLlib name. Rejected: "SVI" is jargon
  that newcomers and tutorial readers have to decode, and "Online LDA"
  is the canonical name in the source paper anyway.
- **`HoffmanLDA`.** Author-name disambiguation. Unambiguous. Rejected
  as out of style for this codebase (no `BleiSomething` elsewhere).

## Naming-collision note

Spark MLlib has a JVM-side optimizer `OnlineLDAOptimizer`, but it is
**not** exposed as a Python class — it surfaces only as a string value
of the `optimizer` Param on `pyspark.ml.clustering.LDA` (`"online"` vs
`"em"`). So the literal Python-name collision for `OnlineLDA` is zero.
What remains is mild conceptual proximity: a reader could initially
wonder whether `OnlineLDA` wraps MLlib's optimizer. It doesn't —
`spark_vi/models/lda.py` is an independent reimplementation of the
Hoffman 2010 algorithm in pure Python on top of Spark RDDs. Class
docstrings make this explicit.

If both implementations need to be referenced in the same script, the
standard import-time alias works: `from spark_vi.mllib.lda import
OnlineLDAEstimator as SparkViOnlineLDA` (or similar).

## Consequences

- Any external consumer pinned to `from spark_vi.mllib.lda import
  VanillaLDAEstimator` (or the model class) has to update their import.
  No compatibility shim; the framework has no external users yet.
- Saved checkpoints written before this rename carry
  `metadata["model_class"] == "VanillaLDA"`. There are no production
  checkpoints predating the rename, so we do not implement a migration.
  If one surfaces, the manifest field is plain JSON and trivially
  editable.
- ADR 0008's title remains "Vanilla LDA design choices" as a historical
  artifact. The rename rationale and the new name live here.
