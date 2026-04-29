# 0006 — Unified persistence format: VIResult as canonical state for completed runs and in-progress checkpoints

**Status:** Accepted
**Date:** 2026-04-29

## Context
The bootstrap-phase persistence layer had three issues that compounded: two
near-duplicate save/load implementations (`spark_vi/io/export.py` for completed
results, `spark_vi/diagnostics/checkpoint.py` for in-progress state) sharing
the same on-disk layout but different manifest schemas; a `VIConfig.checkpoint_interval`
field that was validated but never read by `VIRunner.fit`; and no first-class
resume API — the only available path required monkey-patching
`model.initialize_global` to inject loaded parameters, with `start_iteration=`
set manually for Robbins-Monro continuity. The architecture docs claimed
checkpointing was a working feature; the reality was a half-wired field with
no glue.

The natural consolidation: `VIResult` already records everything needed for
either lifecycle. A "completed run" and an "in-progress checkpoint" differ only
in whether `converged=True` and how many iterations have happened. One
dataclass, one save/load pair, one format covers both.

## Decision
- One save/load pair: `save_result(result, path)` / `load_result(path) -> VIResult`,
  in `spark_vi/io/export.py`. `VIResult` becomes the canonical record for both
  completed runs and in-progress checkpoints. `converged=True` indicates a
  finished, converged run; `converged=False` covers both runs that exhausted
  `max_iterations` without converging and interim checkpoints written during a
  fit.
- Manifest gains `format_version: 1`. `load_result` raises `ValueError` for
  unknown versions, providing a migration handle for future format changes.
- `VIConfig.checkpoint_dir: Path | str | None` is added alongside
  `checkpoint_interval`. The two fields are coupled — both-or-neither, enforced
  in `__post_init__`. When set, `VIRunner.fit` auto-saves a `VIResult` to
  `checkpoint_dir` every `checkpoint_interval` iterations, overwriting the
  previous checkpoint (the last one is the only one needed for resume).
- `VIRunner.fit(rdd, *, resume_from: Path | str | None = None)` is added. When
  set, the runner loads the saved `VIResult`, uses its `global_params` instead
  of `model.initialize_global`, sets the Robbins-Monro counter to match its
  `n_iterations`, and seeds `elbo_trace` with the loaded trace so post-resume
  convergence checks see continuity.
- `start_iteration` retains its current behavior but becomes an internal kwarg;
  the user-facing path is `resume_from`.
- `spark_vi/diagnostics/checkpoint.py`, `spark_vi/diagnostics/__init__.py`, and
  the `spark_vi/diagnostics/` directory are deleted. `save_checkpoint` and
  `load_checkpoint` are removed from the public API; only the test was a
  caller, and it is rewritten to use `save_result` + `resume_from`.

## Alternatives considered
- **Keep two separate save/load implementations with a shared backbone.**
  Rejected: the manifest schemas were the only real difference, and that
  difference is illusory once `VIResult` covers both lifecycles. Maintaining
  two near-identical formats invites future drift.
- **`VIRunner.from_checkpoint(path, model)` constructor instead of
  `resume_from` kwarg.** Rejected: the kwarg is symmetric with the existing
  `start_iteration` and `data_summary` kwargs and keeps the constructor
  minimal. The class method would force users to remember whether to construct
  via the alternate constructor or to pass the kwarg.
- **`checkpoint_dir` as a kwarg on `fit()` instead of a `VIConfig` field.**
  Rejected: it is tightly coupled to `checkpoint_interval` (which is in
  `VIConfig`), and the two belong together. Splitting them across
  `VIConfig`/`fit` would let inconsistent calls slip through.

## Consequences
- The `spark_vi.diagnostics` namespace is removed. Anything depending on
  `from spark_vi.diagnostics.checkpoint import save_checkpoint` will break;
  no production code does, and the only test caller is rewritten.
- `VIResult.n_iterations` now reflects total iterations including any
  resume offset. A run resumed at iteration 3 and run for 3 more reports
  `n_iterations=6`. Pre-existing tests that asserted on `n_iterations` for
  resumed runs were silent on this; the value is now meaningful.
- The `format_version` field is mandatory in newly-written manifests. Manifests
  written before this change (none in production) would load successfully
  because `load_result` defaults missing-version to 1.
- The auto-checkpoint mechanism overwrites the previous checkpoint each time.
  This is the right default for "recover from crash" use cases — only the most
  recent state matters. Versioned-checkpoints (one per N iterations preserved
  separately) is intentionally not built in; users who need it can call
  `save_result` manually or wrap the runner.
- Documentation is brought into agreement with code: `SPARK_VI_FRAMEWORK.md`
  had a wrong VIResult dataclass body (showing `model` and `history` fields
  that don't exist) which is now corrected to the actual fields.
