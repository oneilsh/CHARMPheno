# ADR 0009 — MLlib Estimator/Transformer shim for VanillaLDA

**Status:** Accepted
**Date:** 2026-05-04
**Context:** [docs/superpowers/specs/2026-05-04-mllib-shim-design.md](../superpowers/specs/2026-05-04-mllib-shim-design.md)

## Context

`VanillaLDA` (ADR 0008) is fittable via `VIRunner` on an `RDD[BOWDocument]`.
That API is correct but unfamiliar to anyone coming from MLlib, and the
head-to-head comparison driver
(`charmpheno/charmpheno/evaluate/lda_compare.py`) had to maintain two
non-symmetric run wrappers (`run_ours` for our framework, `run_mllib` for
`pyspark.ml.clustering.LDA`). Both pain points point to the same fix: an
Estimator/Model pair that wraps `VanillaLDA` in the MLlib API shape.

## Decisions

### Subclass `pyspark.ml.base.{Estimator, Model}`

The shim subclasses MLlib's base Estimator and Model (not the concrete
`pyspark.ml.clustering.LDA`). This gives `Pipeline` integration and Param
introspection without committing to mirroring every MLlib LDA Param. We
mirror the surface that maps cleanly to our implementation; the rest is
rejected at fit time with a clear error.

### LDA-specific shim now; defer generic `VIModel` adapter

We write `VanillaLDAEstimator` / `VanillaLDAModel` concretely. When
`OnlineHDP` lands, its second data point will inform whether the abstraction
boundary is real. Designing a generic adapter from one model is likely to
produce an LDA-shaped abstraction with HDP-shaped duct tape later.

### MLlib param names on the shim; our extras camelCased

Shared params use MLlib's names exactly (`k`, `maxIter`, `learningOffset`,
`learningDecay`, `subsamplingRate`, `docConcentration`, `topicConcentration`,
`featuresCol`, `topicDistributionCol`, `optimizer`, `optimizeDocConcentration`).
Our framework-specific extras adopt MLlib's camelCase convention for
consistency on the shim's surface (`gammaShape`, `caviMaxIter`, `caviTol`).
Internal `VanillaLDA` and `VIConfig` keep their snake_case names.

### Shared Param mixin for Estimator and Model

The 12 custom Params plus the three inherited mixins (`HasFeaturesCol`,
`HasMaxIter`, `HasSeed`) live on a single private class `_VanillaLDAParams`.
Both `VanillaLDAEstimator` and `VanillaLDAModel` inherit from it, so the
Param surface is declared once and stays in sync between Estimator and
Model. This mirrors MLlib's own `_LDAParams` pattern.

### Vector-column DataFrames in/out

The shim accepts a single Vector column (default `featuresCol="features"`) —
the standard MLlib `CountVectorizer` output shape — and emits a Vector
column (`topicDistributionCol`). It does not duplicate `CountVectorizer`'s
work. Anyone who wants to feed `RDD[BOWDocument]` directly continues to use
`VIRunner` straight.

### Reject unsupported configurations explicitly

Three rejections at fit time, each with a clear message pointing to ADR 0008
where applicable:

- `optimizer != "online"` (we are SVI-only).
- `optimizeDocConcentration=True` (deferred per ADR 0008 future work).
- Vector `docConcentration` (symmetric-α-only per ADR 0008).

Silent fallback would mislead about what users are getting. We deliberately
default `optimizeDocConcentration=False` even though MLlib defaults it to
True — defaulting to True would mean any `VanillaLDAEstimator().fit(df)`
trips the validator on the no-arg path.

### Persistence (`MLReadable` / `MLWritable`) deferred

The driving v1 use case is comparison and Pipeline ergonomics, not
`Pipeline.save()`. When a concrete user appears, the question of which
Param values and which parts of `VIResult` to round-trip becomes specific
instead of speculative. Until then, users persist via `VIResult.export_zip`
and reconstruct.

### `_transform` via Python UDF

The Model's `transform` applies a UDF over the Vector column rather than
routing through `VIRunner.transform`. Reattaching `VIRunner.transform`'s
`RDD[dict]` output to the DataFrame's other columns is awkward; a UDF
preserves all columns trivially and matches MLlib's own LDA transform
implementation pattern.

### `logLikelihood` / `logPerplexity` stubbed

Held-out perplexity for variational LDA requires deriving a held-out ELBO
bound; non-trivial and there is no concrete user. Stubs raise
`NotImplementedError` and point to `VIResult.elbo_trace` for the closest
existing analog.

## Relation to prior ADRs

- [ADR 0007](0007-vimodel-inference-capability.md) —
  `VanillaLDA.infer_local` (the optional capability) is what makes
  `_transform` possible. The shim's UDF is a thin wrapper around the same
  CAVI inner loop.
- [ADR 0008](0008-vanilla-lda-design.md) — the symmetric-α-only constraint
  and the deferral of `optimizeDocConcentration` carry forward unchanged;
  the shim makes those constraints visible at the API boundary.

## Future work

- `MLReadable` / `MLWritable` when a concrete user wants `Pipeline.save()`.
- Generic `VIModel` → MLlib adapter, when OnlineHDP gives us the second
  data point.
- Held-out `logLikelihood` / `logPerplexity`.
- Empirical-Bayes α (the missing path that would let us flip
  `optimizeDocConcentration=True` from rejection to support).
