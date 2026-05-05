# ADR 0010 — Concentration-parameter optimization for VanillaLDA

**Status:** Accepted
**Date:** 2026-05-05
**Context:** [docs/superpowers/specs/2026-05-05-lda-concentration-optimization-design.md](../superpowers/specs/2026-05-05-lda-concentration-optimization-design.md)

## Context

[ADR 0009](0009-mllib-shim.md) shipped the MLlib shim with `optimizeDocConcentration=True` rejected and listed empirical-Bayes α as future work. We are now adding it because the next model on the roadmap is online HDP (Wang, Paisley, Blei 2011), which has *two* concentration parameters — γ on the corpus stick and α on the doc stick — at the heart of its appeal. The Newton machinery for HDP's γ and α directly templates from LDA's η and α updates respectively. Building it on LDA first is cheaper because the math is well-trodden, MLlib has a reference implementation to bit-match, and our existing ELBO trend test gates regressions.

## Decision

- **Asymmetric α** (length K) via Blei et al 2003 §5.4 closed-form Newton (diagonal+rank-1 Hessian via Sherman-Morrison). Mini-batch scaling per [ADR 0005](0005-mini-batch-sampling.md); ρ_t damping reused from λ.
- **Symmetric scalar η** via Hoffman 2010 §3.4 scalar Newton. Cheaper than α: the η stat is computable from current global λ, so `local_update` returns nothing new for η.
- Flip `optimizeDocConcentration` default to `True` (MLlib parity, drops our v0 divergence). Add `optimizeTopicConcentration`, default `False`.
- Numerical floor: clip α and η to `[1e-3, ∞)` after each Newton step.

## Alternatives considered

- **α only, skip η.** Cheaper, but doesn't suss out the global-stat optimization pattern HDP needs for γ. Rejected — building both flavors here is the de-risking step.
- **Asymmetric η** (per-vocab, length V). MLlib doesn't do this; mini-batch SVI is least stable on η; tractable but real numerical work. Deferred.
- **Concentration-specific learning rate.** Hoffman 2010 derives shared ρ_t from the natural-gradient view; no evidence yet we need separate damping. Reuse λ's ρ_t.
- **Minka 2003 fixed-point** instead of Blei's Newton. Both work; Blei's matches MLlib's reference implementation, so cross-checking is mechanical.

## Consequences

- The "deliberate divergence from MLlib defaults" recorded in ADR 0009 is gone: `VanillaLDAEstimator()` now optimizes α by default, matching `pyspark.ml.clustering.LDA`. The `test_optimize_doc_concentration_defaults_false_diverging_from_mllib` test is replaced with a positive parity assertion.
- The validator now accepts vector `docConcentration` and `optimizeDocConcentration=True`. Both rejections from ADR 0009 §"Reject unsupported configurations explicitly" are removed.
- `VanillaLDAModel` gains `alpha` and `topicConcentration` accessors so callers can introspect optimized values.
- `MLWritable` round-trip of optimized α/η is still deferred (ADR 0009's persistence punt is unchanged).
- HDP (next ADR) becomes a translation exercise: γ uses η's machinery (global stat), α uses LDA-α's per-doc form.
- The integration test gating α optimization regressions ships as a smoke test (`test_alpha_optimization_runs_end_to_end_without_regression`) rather than the originally-planned "L1 drift toward truth" recovery test. Empirical diagnostics during implementation showed truth-recovery is unverifiable at synthetic-corpus scales (D≪10K) because of well-known topic-collapse SVI behavior — Hoffman 2010 §4 used D=100K–352K to validate recovery. Strict math validation is upstream: helper unit tests with idealized inputs plus the existing ELBO smoothed-trend gate.
