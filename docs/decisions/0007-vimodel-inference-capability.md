# ADR 0007 — VIModel inference capability via optional `infer_local`

**Status:** Accepted
**Date:** 2026-04-30
**Context:** [docs/superpowers/specs/2026-04-30-vanilla-lda-design.md](../superpowers/specs/2026-04-30-vanilla-lda-design.md)

## Context

`VIModel` until now exposed only training-side hooks: `initialize_global`,
`local_update`, `update_global`, `combine_stats`, `compute_elbo`,
`has_converged`. Models with local latent variables (LDA, HDP) need a way
to expose per-row inference (e.g. theta_d for LDA) under fixed trained
global params. The framework had no first-class slot for it; clinical
wrappers like `CharmPhenoHDP` added thin per-class `.transform` methods
ad hoc.

## Decision

Add an optional `infer_local(row, global_params)` method to `VIModel`:

- **Optional**, not abstract. Default raises `NotImplementedError` with a
  message naming the concrete subclass.
- **Pure function** of `(row, global_params)`. No dependence on instance
  state from training. Encoded in the docstring as a hard contract.
- New `VIRunner.transform(rdd, global_params)` orchestrator that broadcasts
  global_params and maps `infer_local` row-by-row, with the same broadcast-
  unpersist discipline as `fit`.

Models without per-row latent variables (e.g. `CountingModel`) leave
`infer_local` unimplemented; calling `VIRunner.transform` on them produces
a clear, named error rather than NaN or None.

## Why default-raise rather than default-NaN

Silent fallback (returning None or NaN) would mask a real user error: "I
called transform on a model that can't do inference." A loud error names
the concrete class and points at `transform`, making the misuse obvious in
the stack trace.

## MLlib compatibility invariant

The `(row, global_params)` purity invariant is what keeps a future MLlib
`Estimator/Transformer` shim mechanical. Such a shim would:

- Wrap `VIRunner` as the `Estimator.fit(df)` body.
- Hold trained `VIResult.global_params` inside a `Transformer`.
- In `Transformer.transform(df)`, call `df.rdd.map(lambda r:
  model.infer_local(r, captured_global_params))`.

Nothing about `infer_local` needs to know about MLlib for this to work, as
long as it never reads `self.<post-fit-state>`. The compat layer is left
clean but unused for v1.

## Alternatives considered

- **Add `infer_local` as an abstract method.** Rejected: `CountingModel`
  has no meaningful per-row inference, and forcing every model author to
  implement an `infer_local` that raises is worse than a single base-class
  default that does the same.
- **Adopt MLlib `Estimator/Transformer` directly.** Rejected as premature:
  significant framework rewrite (DataFrame as the unit of work, param
  plumbing, serialization conformance) before we even have a second real
  model. The compat path is left non-foreclosed.
- **Bundle inference into `local_update` and have transform read it from
  the suff-stat dict.** Rejected: `local_update` already aggregates per-
  partition stats, so per-row outputs would have to escape via a side
  channel — twisting one method's contract to serve two purposes.

## Consequences

- New optional method on `VIModel`. Existing models keep working unchanged.
- `VIRunner` gains one new public method.
- Future models with per-row latents have a first-class slot for inference.
- Future MLlib compat shim is a wrapper layer, not a framework rewrite.
