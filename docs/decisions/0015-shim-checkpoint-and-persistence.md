# ADR 0015 — Shim checkpoint, save, and load

**Status:** Accepted
**Date:** 2026-05-08
**Related:** ADR 0006 (canonical persistence format),
ADR 0009 (LDA MLlib shim),
ADR 0012 (HDP MLlib shim).

## Context

ADR 0006 wired the framework's persistence layer: `save_result` /
`load_result` round-trip a `VIResult`, and `VIRunner.fit` already accepts
`cfg.checkpoint_dir` / `cfg.checkpoint_interval` / `resume_from` for
crash-recovery during long fits. ADR 0009 and ADR 0012 each explicitly
deferred the *shim* end of persistence (`MLReadable` / `MLWritable`,
shim-level save/load) to "v2 when a concrete user appears." A concrete
user has now appeared: a multi-hour cloud HDP fit lost its result when
the SSH session terminated, and the runner-level checkpoint machinery
sitting in `VIConfig` was never plumbed through the MLlib shim that the
cloud driver actually uses.

Two further gaps surfaced while wiring this up:

- Per-iteration trajectories of the optimized hyperparameters (γ, α, η —
  ADR 0010, ADR 0013) were lost on overwrite. Only the final values
  landed in `global_params`; the iteration-by-iteration path was visible
  only in the live log and gone after the run.
- HDP's doc-truncation `K` is not recoverable from `global_params` alone
  (the per-doc Beta factors live in local state, not global), so a saved
  `VIResult` could not by itself reconstruct an `OnlineHDPModel` of the
  right shape.

This ADR records the decision to surface the existing framework
machinery through the shim and to fill those two gaps in the
`VIResult` / `VIModel` contracts.

## Decisions

### Lightweight wrapper, not `MLWritable`

No `Pipeline.save` round-trip. The shim's save/load delegates to
`save_result` / `load_result` directly and reconstructs the Estimator-side
Model from a `VIResult`. `CountVectorizer` already round-trips its own
vocabulary natively if a Pipeline.save consumer ever materializes; the
shim layer doesn't need to participate. ADR 0009 and ADR 0012's
"persistence deferred" stays deferred for the MLlib-JSON-metadata layout
specifically; what we add here is the simpler in-house format.

### Three new Estimator Params on both shims

- `saveInterval: int` (default `-1`, off)
- `saveDir: str` (default `""`, off)
- `resumeFrom: str` (default `""`, fresh fit)

Live on the shared Params mixin so Estimator and Model agree. See
[`spark-vi/spark_vi/mllib/lda.py`](../../spark-vi/spark_vi/mllib/lda.py)
and [`spark-vi/spark_vi/mllib/hdp.py`](../../spark-vi/spark_vi/mllib/hdp.py).

### `save*` rather than `checkpoint*`

MLlib already has a `checkpointInterval` Param on several Estimators
(LDA included). Its meaning is RDD-lineage checkpointing via
`RDD.checkpoint()` — a Spark execution-engine concern that breaks long
DAGs by materializing intermediate state to a checkpoint dir. It has
nothing to do with model-state persistence. Reusing the name on the
shim would be a footgun for any reader fluent in MLlib. We use `save*`
so the distinction is unmistakable. (Internally on the framework side
ADR 0006 already used `checkpoint_dir` / `checkpoint_interval`; that
naming predates this ADR and stays as-is — it's not Spark-facing.)

### Single-dir, multi-purpose

`saveDir` is the working area during fit *and* the final artifact.
Latest state always overwrites the previous interim save; the runner's
end-of-fit final save also overwrites. To keep multiple historical
fits, point `saveDir` at distinct paths per run. This matches ADR 0006's
"latest-only" choice for `checkpoint_dir` and avoids inventing a second
on-disk layout for the shim.

### Resume is explicit

`resumeFrom` is a separate Param from `saveDir`. A user pointing
`saveDir` at an existing directory does *not* auto-resume; they have to
opt in via `resumeFrom`. This guards against the "I wanted a fresh fit
but the dir already had something in it" failure mode.

### Final-save guarantee at the runner level

When `cfg.checkpoint_dir` is set, [`VIRunner.fit`](../../spark-vi/spark_vi/core/runner.py)
writes a final `save_result` to that dir before every return path
(convergence and max-iter exhausted), regardless of where the
iteration count falls relative to `checkpoint_interval`. (The runner
has no `try/finally` around the fit loop today, so an exception still
terminates without a final save — an explicit save on exception
teardown is a possible future enhancement.)
Before this work the dir reflected only the last interim-save iteration
— not what callers reasonably expect. The fix lives at the runner so
all `VIRunner.fit` callers benefit, not just the shim.

### Two new model-contract methods, both default no-op

Added to `VIModel` in [`spark-vi/spark_vi/core/model.py`](../../spark-vi/spark_vi/core/model.py).
Neither is `@abstractmethod`; existing models that override neither
keep working unchanged.

- **`get_metadata() -> dict`** — returns shape constants needed for
  reconstruction. Concrete override on `OnlineLDA` returns
  `{"K": self.K, "V": self.V}`; on `OnlineHDP` returns
  `{"T": self.T, "K": self.K, "V": self.V}`. Note the layering:
  models contribute shape constants only. The runner's
  `_runner_metadata` helper merges in `model_class` (and the
  `checkpoint` flag for interim saves) on top of whatever the model
  returns, with runner-set keys winning on conflict. So a loaded
  `VIResult.metadata` carries both the shape constants the model
  supplied and the `model_class` the runner stamped — together that
  is what `Model.load` needs to reconstruct.
- **`iteration_diagnostics(global_params) -> dict`** — returns a flat
  dict of scalars and small 1-D arrays computed once per iteration. The
  runner accumulates these into `VIResult.diagnostic_traces` and emits
  a compact `key=value` line in the iteration log. Designed for the
  γ/α/η trajectories now and for future per-iter diagnostics
  (topic coherence, held-out perplexity) without further contract
  changes.

### `VIResult.diagnostic_traces`

New field on [`VIResult`](../../spark-vi/spark_vi/core/result.py),
typed `dict[str, list[float | np.ndarray]]`. `save_result` /
`load_result` in [`spark-vi/spark_vi/io/export.py`](../../spark-vi/spark_vi/io/export.py)
round-trip both shapes: scalar-valued traces inline-JSON in the
manifest, 1-D-array-valued traces as sidecar `traces/<name>.npy`. No
`format_version` bump — pre-release codebase, no production checkpoints
to migrate. Per-key sidecars match the existing `params/<name>.npy`
convention (one file per logical quantity, individually inspectable)
rather than packing heterogeneous keys into one NPY.

### Type-aware `Model.load`

Both `OnlineLDAModel.load` and `OnlineHDPModel.load` validate
`metadata["model_class"]` against the calling class and raise a clear
error on mismatch (e.g. loading an HDP checkpoint via
`OnlineLDAModel.load` fails at the metadata gate, not deep inside
shape reconstruction).

### `OnlineHDPModel` constructor simplification

Pre-this-work the model was constructed as `OnlineHDPModel(result, T, K)`.
The shim's `_fit` was the only caller and it had T from the Estimator
Param and K from the OnlineHDP instance. Now the constructor takes
`(result,)` only; T and K come from `result.metadata` via the new
`get_metadata` machinery. This removes the failure mode where
`Model.load` would have had to guess K.

## Alternatives considered

- **Full `MLWritable` / `MLReadable` for `Pipeline.save` round-trip.**
  Rejected: no concrete user, and MLlib's per-stage JSON-metadata
  layout is significantly more work than the simple delegation pattern
  we adopt. Reopen if a Pipeline.save consumer ever materializes — it
  can sit on top of the new save/load without disturbing it.
- **`fsspec` / `gcsfs` to make persistence GCS-native.** Rejected: the
  current cloud environment auto-mounts GCS buckets as local paths, so
  `pathlib.Path` and `np.save` work end-to-end without an abstraction
  layer. Add fsspec only if the mount layer is ever absent.
- **Reusing MLlib's `checkpointInterval` Param name.** Rejected — see
  Decision 3. Name collision with an unrelated Spark concept.
- **Versioned checkpoints (one per N iters preserved separately).**
  Rejected per ADR 0006's existing latest-only choice. Users who want
  historical snapshots can per-run `saveDir` paths.
- **Auto-resume from an existing `saveDir`.** Rejected: too easy to
  accidentally continue a run when you wanted a fresh start. Explicit
  `resumeFrom` is safer.
- **Per-iter traces packed into one heterogeneous NPY** vs per-key
  sidecars. Rejected: per-key matches the existing
  `params/<name>.npy` convention and lets each trace be loaded /
  inspected independently.
- **Putting `get_metadata` content directly in `global_params` as
  scalars.** Rejected: blurs the semantic — `global_params` is the
  variational state we update each iteration; metadata is fixed shape.
  Different things, different homes.
- **Persisting Estimator Params alongside the trained Model.**
  Rejected: that's the start of `MLWritable` territory and we are
  explicitly not building it. Loaded Models carry default Param
  values; users re-set them if needed.

## Consequences

- Both Estimator shims now accept `setSaveDir` / `setSaveInterval` /
  `setResumeFrom`. Existing call sites that don't set these keep
  working unchanged.
- Runner behavior shifts subtly: when `cfg.checkpoint_dir` is set, the
  dir is now authoritative as the post-fit artifact (final-save
  guarantee). Pre-this-work it reflected only the last interim-save
  iteration. Callers that consumed `checkpoint_dir` mid-fit see no
  change.
- `VIResult.metadata["checkpoint"]: True` previously appeared only on
  interim saves. After this work the dir overwrites with the final
  save at end-of-fit, so post-fit consumers should distinguish
  "fit running" vs "fit done" via `metadata["model_class"]` plus
  `converged`, *not* via the `checkpoint` flag.
- New `get_metadata` / `iteration_diagnostics` overrides on `OnlineLDA`
  and `OnlineHDP` are transparent — existing code paths are unaffected
  because the base methods are no-op defaults.
- `OnlineHDPModel` constructor changed from `(result, T, K)` to
  `(result,)`. There were no external callers; the in-tree `_fit` was
  the only one, and it is updated.
- Cloud drivers
  [`lda_bigquery_cloud.py`](../../analysis/cloud/lda_bigquery_cloud.py)
  and [`hdp_bigquery_cloud.py`](../../analysis/cloud/hdp_bigquery_cloud.py)
  gained `--save-dir` / `--save-interval` / `--resume-from` CLI flags;
  the cloud Makefile gained matching `SAVE_DIR` / `SAVE_INTERVAL` /
  `RESUME_FROM` overrides. Long-running fits can now survive
  driver-process death.
- Test count: ~194 (pre-branch) → 215 passing default + 2 default-skipped
  (zip tests, depend on a built wheel) + 15 `@slow` = 232 total
  collected. `cd spark-vi && make test` still green.

## Implementation

Landed on `feat/shim-checkpoint-export-import` as a series of focused
commits across the layers touched: core (`VIModel` /
`VIResult` contract additions), io (`save_result` /
`load_result` round-trip of `diagnostic_traces`), runner (metadata
merge, per-iter diagnostic accumulation, final-save guarantee),
models (`get_metadata` / `iteration_diagnostics` overrides on
`OnlineLDA` and `OnlineHDP`), mllib (shim Params +
`Model.save` / `Model.load`), cloud (BigQuery driver CLI flags +
Makefile overrides), tests (integration coverage for save/load
round-trip + diagnostic traces), and docs (this ADR). See
`git log main..HEAD` on the branch for the canonical commit list.
