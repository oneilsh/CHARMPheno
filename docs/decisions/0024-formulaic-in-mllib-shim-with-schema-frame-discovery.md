# 0024 — STM covariate API: formulaic in the MLlib shim layer, Spark-native categorical discovery via schema-frame

**Status:** Accepted
**Date:** 2026-05-29
**Related:** ADR 0022 (STM-prevalence as covariate path); ADR 0019 (MLlib shim and BOWDocument namespace); design spec [2026-05-29 STM prevalence-only design](../superpowers/specs/2026-05-29-stm-prevalence-design.md)

## Context

STM-prevalence needs a covariate-specification API. Three user groups want different things:

1. **Pure-numpy spark-vi users** — want to build their own design matrix in numpy and pass it. No DataFrame layer, no formula DSL.
2. **PySpark MLlib users with their own preprocessing** — already using `StringIndexer`, `OneHotEncoder`, `VectorAssembler` Pipelines; want to plug a Vector column into the STM estimator the same way they plug a `features` column into MLlib regressors.
3. **R-stm-familiar users** — want R-style formula syntax (`~ age + sex + cohort + sex:cohort`); expect dummy coding, interactions, intercept handling, reference-level control to "just work."

Four design questions follow:

1. Where does formula-handling code physically live in the spark-vi codebase? Options: pure-numpy engine layer (`spark_vi.models.topic.stm`), MLlib shim layer (`spark_vi.mllib.topic.stm`), or downstream in charmpheno.

2. What library handles formula parsing? Options: PySpark's built-in `pyspark.ml.feature.RFormula`, `patsy` (older Python formula library), `formulaic` (modern Python formula library, Apache 2.0).

3. How are categorical levels discovered without breaking spark-vi's "no driver-side materialization of full data" posture? Options: let formulaic sample the data, do AST rewriting on the formula, or feed formulaic a controlled stand-in DataFrame.

4. What gets persisted across fit → transform? The factor level mapping, intercept handling, and interaction expansion need to round-trip so transform-time inputs apply the same encoding.

## Decision

### 1. formulaic lives in the MLlib shim layer (`spark_vi.mllib.topic.stm`)

Not in the engine, not in charmpheno. Becomes an **optional** spark-vi dependency declared under `[project.optional-dependencies]` with extra name `formula`. The pure-numpy engine (`OnlineSTM`) has no new dependency; only users who reach for the shim's formula path install it.

This placement matches the existing LDA layering:

- `spark_vi.models.topic.lda.OnlineLDA` — pure numpy, no DataFrames, consumes `BOWDocument`.
- `spark_vi.mllib.topic.lda` — DataFrame-aware estimator, consumes a `features` Vector column, converts internally via `_vector_to_bow_document`.

The MLlib shim is explicitly the "spark-vi speaks DataFrames" layer; adding formula support is filling out the convenience that layer is for, not expanding scope. Pure-numpy users continue to consume `OnlineSTM` directly with no formulaic dependency. PySpark users who prefer to build their own covariate Vector column use the shim's pre-built path. R-stm-familiar users use the shim's formula path.

### 2. Library: `formulaic`, not RFormula or patsy

`formulaic` (https://matthewwardrop.github.io/formulaic/) is a modern Apache-2.0 Python implementation of R-style formula syntax. Active maintenance, faster than patsy, used by `linearmodels` and other established Python statistics packages.

PySpark's built-in `pyspark.ml.feature.RFormula` is rejected because its formula grammar is a subset of R's: `~ a + b + a:b` works, but `bs(x, df=4)` (deferred to v1.x anyway), `I(x**2)`, explicit reference-level control via `C(x, contr.treatment(reference="..."))`, and several other features R-stm users expect are absent. RFormula is slow-moving and owned by the Spark project; we cannot extend it. formulaic is the right shape for a v1 that wants to be a real STM port.

`patsy` (the older Python formula library) is functionally similar to formulaic but slower, no longer actively developed, and has accumulated edge cases. formulaic is patsy's modern successor in the Python ecosystem.

### 3. Categorical level discovery uses the schema-frame trick

formulaic learns categorical levels at fit time by scanning the input DataFrame — this is a "stateful transform" in formulaic's terminology, captured in the `ModelSpec.transform_state`. For STM at Spark scale we cannot hand formulaic the full corpus DataFrame: the corpus has millions to billions of rows and lives distributed across workers.

The discovery mechanism is:

1. Parse the formula via `Formula(formula_str)`. Walk the term tree to identify categorical columns — either explicit (`C(col)` in the formula) or dtype-inferred (columns whose Spark dtype is `StringType` or `BooleanType`).

2. For each categorical, **discover levels via Spark**:
   - Bound cardinality first via `approxCountDistinct`; if over `max_levels` (default 10000), raise with a clear error pointing to manual binning.
   - Otherwise materialize the level set via `df.select(col).distinct().collect()`, sort lexicographically for determinism across runs.

3. Build a small pandas **schema-frame** that contains each level of each categorical at least once, plus placeholder rows (value 0.0) for continuous columns. For 5 categoricals × 10 levels each, ~10 rows. For 20 categoricals × 100 levels each, ~2000 rows. Trivial.

4. Hand the schema-frame to formulaic the normal way: `model_spec = Formula(formula_str).get_model_matrix(schema_frame).model_spec`. formulaic captures the level set in `transform_state`; the resulting `ModelSpec` is now data-independent at application time.

5. **Validate that nothing else stateful slipped in.** After fitting against the schema-frame, assert `model_spec.transform_state` contains *only* the level mappings we pre-seeded. Anything else (spline knots, mean/sd for standardization) would have been learned from our schema-frame's bogus placeholder values; we reject the formula at this point with an error pointing to the v1 scope decision (ADR 0022) and the documented escape hatches.

6. Broadcast the `ModelSpec` to workers. In each partition's `mapPartitions` block, materialize the per-row design matrix via `model_spec.get_model_matrix(partition_df)` and construct `STMDocument` rows.

The schema-frame trick uses formulaic's normal stateful path while keeping the data-dependent step under Spark's control. No AST rewriting; no sampling; no fork of formulaic.

### 4. ModelSpec is persisted as part of the fitted MLlib shim model

`OnlineSTM` (engine) knows about P (covariate dim) but nothing about what the covariates mean. The fitted `ModelSpec` — factor level mappings, intercept handling, interaction expansion — is part of the **MLlib shim's fitted model state**, not the engine's `VIResult`. Same layering as the existing shim's `LDAModel` carrying vocab metadata while `OnlineLDA` carries only `V`.

Two artifacts on save:

- **VIResult** (spark-vi-native): carries λ, Γ, Σ, structural metadata (K, V, P). Round-trips via spark-vi's `io.export`. Pure numpy.
- **MLlib shim's STMModel**: wraps `VIResult` plus `ModelSpec` plus covariate term names. Persists ModelSpec via formulaic's standard serialization. Loads from disk by restoring both.

Charmpheno's dashboard adapter (ADR 0025-scope) receives the shim's STMModel, not raw VIResult; it reads Γ from VIResult and labels rows from the ModelSpec.

## Alternatives considered

1. **formulaic in the engine layer (`spark_vi.models.topic.stm`).** Rejected: violates the existing posture that engine code is pure numpy with no DataFrame or domain-specific dependencies. The engine becomes harder to use for non-formula consumers (pure-numpy users would inherit a formula-library transitive dependency) and the layering contract documented in [`BOWDocument`'s module docstring](../../spark-vi/spark_vi/models/topic/types.py) ("Generic framework primitives live in spark_vi.core; topic-specific types live alongside the topic models that consume them") breaks down. STMDocument carrying `x: np.ndarray` already encodes the "engine is covariate-agnostic" contract; placing formulaic in the engine would contradict it.

2. **formulaic in charmpheno only.** Rejected: any non-clinical user of spark-vi who wants STM would either have to write their own formula plumbing (rebuilding what we already shipped in charmpheno) or adopt charmpheno (which is clinical-OMOP-specific and not appropriate for non-clinical use cases). The MLlib shim layer exists specifically to be the "convenience for Spark/MLlib users" layer that's framework-agnostic; formulas belong there.

3. **`pyspark.ml.feature.RFormula` instead of formulaic.** Rejected per the library-choice section above. RFormula's grammar is too thin for serious STM use (no `I(...)`, no explicit reference-level control via the `C()` syntax R-stm users expect, no path to splines in v1.x). We would either ship a limited shim now and migrate later, or extend RFormula in our own code; both are worse than just depending on formulaic.

4. **patsy instead of formulaic.** Slower, no longer actively maintained. formulaic is patsy's modern successor and is now what new Python statistics packages reach for.

5. **Sampling-based level discovery** (hand formulaic a random sample of the corpus DataFrame; let it learn levels from the sample). Rejected because sample size becomes a sharp corner that affects the fit: a small sample may miss rare levels entirely, and the user has to tune sample size against an opaque relationship between coverage probability and downstream Γ estimates. Spark `select distinct` + cardinality bound is exact and has no tuning knob.

6. **AST rewriting on the formula** (parse the formula, programmatically substitute `C(col)` → `C(col, levels=[...])`, hand the rewritten formula to formulaic). Possible but requires understanding formulaic's internal AST representation and is fragile across formulaic versions. The schema-frame trick uses only formulaic's public API and has no version-coupling risk.

7. **Fork formulaic** and add a Spark-native discovery hook. Heavy maintenance burden for no clear advantage over the schema-frame trick.

## Consequences

- **`formulaic` becomes an optional spark-vi dependency.** Extras name `formula`. Pure-numpy `OnlineSTM` users are unaffected; PySpark users opting into the shim's formula path install with `pip install spark-vi[formula]`.

- **The R-style formula DSL becomes part of the spark-vi public contract** via the MLlib shim. Documenting which formulaic features are supported (and which the v1-scope validator rejects) is now a permanent documentation obligation; the design spec's "Sharp corners" section is the seed of that documentation.

- **The cardinality bound (`max_levels`, default 10000) is a fit-fail point.** Users with high-cardinality categoricals (e.g., a per-NPI-code factor) get a clear error pointing to manual binning. Configurable on the shim if a legitimate use case needs higher.

- **formulaic version pinning is now a release concern.** The schema-frame trick relies on the public API (`Formula`, `get_model_matrix`, `ModelSpec.transform_state`), not internals, but we should still pin a minimum version (≥1.0) and bump the lower bound only when we test against the new version.

- **Spline support (v1.x)** has a clear extension path: add a Spark `approxQuantile` step that computes knot positions for each `bs(...)` term, rewrite the formula to inject explicit knots before handing to formulaic. Same shape of mechanism, scoped to spline-specific terms. The schema-frame trick generalizes — it stops short of splines in v1 only because we explicitly haven't built the quantile + rewrite step.

- **The MLlib shim has three logical input paths in code** that need clean docstrings: (a) pre-built `covariates_col` Vector column with caller-supplied term names; (b) formula + covariate DataFrame; (c) the internal helper used by the sidecar builder (ADR 0025), which is path (b) decomposed into "fit ModelSpec" and "apply ModelSpec" so charmpheno can persist the spec to a sidecar parquet.

- **The two-pass discovery cost is one extra Spark `select distinct` per categorical column** at fit start (the cardinality bound check is `approxCountDistinct`, single pass; the level materialization is a shuffle). Optimization to fuse into a single `df.agg(F.collect_set(c1), F.collect_set(c2), ...)` is a v1.x perf improvement.

## Open follow-ups

- formulaic version pinning policy: pin minimum (likely ≥1.0); decide on a process for testing against newer versions before bumping the lower bound.
- v1.x spline support: design the `approxQuantile` + formula-rewrite step. Already mapped in the spec; not implemented.
- Fuse per-column distinct queries into a single `df.agg(...)` pass for fit-start latency improvement on formulas with many categoricals.
- Determine whether high-cardinality categoricals (above the default `max_levels`) should fail or auto-bucket-low-frequency-as-other. v1 fails; auto-bucketing is a v1.x choice.
