# MLlib Estimator/Transformer shim for VanillaLDA — Design

**Date:** 2026-05-04
**Branch:** `mllib-compat`
**Related ADRs:** 0008 (Vanilla LDA design), 0009 (this work, to be written)

## Goal

Make `spark_vi.models.lda.VanillaLDA` usable from any code that expects an
`pyspark.ml.clustering.LDA`-shaped API: a fitted `Estimator` produces a
`Model`, both subclass `pyspark.ml.base.{Estimator,Model}`, both declare their
configuration via `Param` objects, and they accept and emit Spark DataFrames
with a Vector column.

The shim is a translation layer only — it owns no SVI logic. All algorithmic
behavior is delegated to the existing `VanillaLDA` + `VIRunner` stack.

Driving use cases:
- Plug `VanillaLDA` into `pyspark.ml.Pipeline` without writing custom glue.
- Replace the bespoke `run_ours` wrapper in
  `charmpheno/charmpheno/evaluate/lda_compare.py` with an estimator that
  matches `run_mllib`'s shape, tightening the head-to-head comparison.
- Provide a familiar API surface for downstream users who already know
  MLlib's LDA conventions.

Non-goals for v1:
- Drop-in replacement for `pyspark.ml.clustering.LDA` (every param, every
  method). The shim mirrors the surface that maps cleanly to our
  implementation; the rest is rejected explicitly or stubbed with
  `NotImplementedError`.
- Persistence (`MLReadable` / `MLWritable`).
- A generic `VIModel` → MLlib adapter. We write the LDA shim concretely;
  when `OnlineHDP` lands, its second data point will inform whether/how to
  generalize.

## Architecture

Two new classes in a new subpackage `spark_vi.mllib`:

- **`VanillaLDAEstimator(pyspark.ml.base.Estimator)`** — declares MLlib-shaped
  Params, implements `_fit(dataset)` by translating params into a `VanillaLDA`
  constructor call plus a `VIConfig`, materializing a `BOWDocument` RDD from
  the DataFrame's Vector column, and running `VIRunner.fit`. Returns a
  `VanillaLDAModel` that carries the trained `VIResult` plus a copy of every
  Param value.

- **`VanillaLDAModel(pyspark.ml.base.Model)`** — wraps a single trained
  `VIResult`. Implements `_transform(dataset)`, `topicsMatrix()`,
  `describeTopics(maxTermsPerTopic)`, `vocabSize()`. Stubs
  `logLikelihood` / `logPerplexity` with `NotImplementedError`.

The shim's only state-bearing object is the `VIResult` carried by the Model;
everything else is plain Param values.

### Internal data flow

```
fit:
  DataFrame[<featuresCol>: Vector]
    → mapPartitions(SparseVector | DenseVector → BOWDocument)
    → RDD[BOWDocument]
    → VIRunner(VanillaLDA(K, V, alpha, eta, gamma_shape, ...),
               VIConfig(max_iterations, learning_rate_tau0, ...,
                        mini_batch_fraction, random_seed)).fit(rdd, ...)
    → VIResult
    → VanillaLDAModel (carrying the VIResult and a copy of all Params)

transform:
  DataFrame[<featuresCol>: Vector]
    → mapPartitions(SparseVector | DenseVector → BOWDocument)
    → RDD[BOWDocument]
    → VIRunner(reconstructed_VanillaLDA).transform(
          rdd, {"lambda": result.global_params["lambda"]}
      )
    → RDD[(gamma, expElogthetad, phi_norm, n_iter)]   # VanillaLDA.infer_local return
    → mapPartitions(gamma → DenseVector(gamma / gamma.sum()))
    → DataFrame[<featuresCol>, <topicDistributionCol>: Vector]
```

`VIRunner.transform` already handles the broadcast lifecycle for the global
params; the shim does not reimplement broadcast/unpersist logic.

## Module layout

New files in `spark-vi/`:

- `spark-vi/spark_vi/mllib/__init__.py` — re-exports `VanillaLDAEstimator`,
  `VanillaLDAModel`.
- `spark-vi/spark_vi/mllib/lda.py` — the two shim classes plus private helpers:
  - `_vector_to_bow_document(v: Vector) -> BOWDocument`
  - `_build_model_and_config(params) -> (VanillaLDA, VIConfig)`
  - `_validate_unsupported_params(params) -> None`
- `spark-vi/tests/test_mllib_lda.py` — unit tests, fast (no `@pytest.mark.slow`).

Updated:

- `spark-vi/spark_vi/__init__.py` — **no** top-level re-export of `mllib`. Users
  opt in via `from spark_vi.mllib.lda import VanillaLDAEstimator`. This keeps
  `import spark_vi` independent of `pyspark.ml` (which is already on the
  classpath as a transitive dep, but the conceptual boundary matters).
- `charmpheno/charmpheno/evaluate/lda_compare.py` — `run_ours` rewritten to use
  the shim. Function signature and `LDARunArtifacts` return type unchanged.
- `docs/decisions/0009-mllib-shim.md` — new ADR documenting the design choices
  in this spec.

## Estimator API

### Param surface

The Estimator declares the following Params (camelCase to match MLlib
convention; defaults match `pyspark.ml.clustering.LDA` for the shared subset
and ADR 0008 for our extras):

| Param                       | Default                | Translates to                          |
|-----------------------------|------------------------|----------------------------------------|
| `k`                         | 10                     | `VanillaLDA(K=...)`                    |
| `maxIter`                   | 20                     | `VIConfig(max_iterations=...)`         |
| `seed`                      | `None`                 | `VIConfig(random_seed=...)`            |
| `featuresCol`               | `"features"`           | (consumed by shim)                     |
| `topicDistributionCol`      | `"topicDistribution"`  | (consumed by shim)                     |
| `optimizer`                 | `"online"`             | (validated; `"em"` raises)             |
| `learningOffset`            | 1024.0                 | `VIConfig(learning_rate_tau0=...)`     |
| `learningDecay`             | 0.51                   | `VIConfig(learning_rate_kappa=...)`    |
| `subsamplingRate`           | 0.05                   | `VIConfig(mini_batch_fraction=...)`    |
| `docConcentration`          | `None` (→ 1/k)         | `VanillaLDA(alpha=...)` (scalar only)  |
| `topicConcentration`        | `None` (→ 1/k)         | `VanillaLDA(eta=...)`                  |
| `optimizeDocConcentration`  | `False`                | (`True` raises)                        |
| `gammaShape`                | 100.0                  | `VanillaLDA(gamma_shape=...)`          |
| `caviMaxIter`               | 100                    | `VanillaLDA(cavi_max_iter=...)`        |
| `caviTol`                   | 1e-3                   | `VanillaLDA(cavi_tol=...)`             |

`vocab_size` for the `VanillaLDA` constructor is determined at fit time from
the Vector column dimensionality (the shim asserts the first row's vector
size matches the dataset's claimed size, then passes it through).

### `__init__` and `setParams`

Standard MLlib pattern: `__init__` accepts every Param as a keyword arg
(camelCase), defaults declared on `Param` objects, `setParams(**kwargs)`
exists for post-construction updates. Mirrors
`pyspark.ml.clustering.LDA.__init__` exactly.

### `_fit(dataset: DataFrame) -> VanillaLDAModel`

1. Call `_validate_unsupported_params(self)`. Raises `ValueError` for any of
   the rejection cases (see "Unsupported configurations" below).
2. Resolve `None` defaults: if `docConcentration is None`, use `1.0 / k`;
   same for `topicConcentration`.
3. Determine `vocab_size = V` from the Vector column. The shim takes the
   size from the first row of the dataset. It does not scan further rows for
   agreement — Vector columns produced by `CountVectorizer` (or any standard
   producer) have homogeneous size by construction, and `_vector_to_bow_document`
   only reads `indices`/`values`, so a mis-sized later row would either
   surface as an out-of-bounds index in CAVI (loud failure) or as a silent
   index-into-the-wrong-vocabulary error (a CountVectorizer-misuse bug at
   the boundary, not the shim's job to catch).
4. Build `(VanillaLDA, VIConfig)` via `_build_model_and_config(self)`.
5. Convert `dataset.select(featuresCol)` to `RDD[BOWDocument]` via
   `mapPartitions`, applying `_vector_to_bow_document` per row.
6. Run `VIRunner(model, config).fit(rdd, ...)`, capture the `VIResult`.
7. Construct `VanillaLDAModel(result)`, copy every Param value from `self`
   to the model so its getters reflect the same configuration.

## Model API

### State

A `VanillaLDAModel` carries:
- `result: VIResult` — trained globals, including `global_params["lambda"]`.
- A copy of every Param declared on the Estimator (the Model is itself a
  `Params` instance, since it inherits from `pyspark.ml.base.Model`).

### `_transform(dataset: DataFrame) -> DataFrame`

Returns the input DataFrame with one new column `<topicDistributionCol>` of
type Vector (DenseVector, length K).

Implementation:
1. Convert `dataset.select(featuresCol)` → `RDD[BOWDocument]` (same helper
   as `_fit`).
2. Reconstruct a `VanillaLDA` instance from stored Param values (same
   `_build_model_and_config` helper, discarding the `VIConfig`).
3. Build `runner = VIRunner(model)` and call
   `runner.transform(rdd, {"lambda": self.result.global_params["lambda"]})`.
4. Map the resulting RDD's per-row tuple `(gamma, ..., ...)` to
   `DenseVector(gamma / gamma.sum())`.
5. Reattach the topic-distribution column to the original DataFrame. The
   simplest correct implementation is `mapInPandas` — converts back through
   a typed schema and preserves all original columns.

### `topicsMatrix() -> pyspark.ml.linalg.Matrix`

Returns the normalized topic-word matrix as an MLlib `DenseMatrix`,
transposed to (V × K) — the orientation MLlib uses (rows are vocabulary,
columns are topics). Internally we store λ as (K × V), so this is one
transpose plus row-normalization.

```python
lam = self.result.global_params["lambda"]
beta = lam / lam.sum(axis=1, keepdims=True)   # (K, V) row-stochastic
return DenseMatrix(numRows=V, numCols=K, values=beta.T.flatten("F"))
```

### `describeTopics(maxTermsPerTopic: int = 10) -> DataFrame`

Returns a DataFrame with schema
`(topic: int, termIndices: array<int>, termWeights: array<double>)`. For
each row k of normalized λ, take argsort descending, keep the top
`maxTermsPerTopic` indices and their weights. Same format MLlib uses.

### `vocabSize() -> int`

Returns `self.result.global_params["lambda"].shape[1]`.

### Stubs

Both raise `NotImplementedError` with a message pointing users to the
training-time ELBO trace (`VIRunner.fit` records this) for the closest
existing analog:

- `logLikelihood(dataset)`
- `logPerplexity(dataset)`

## Unsupported configurations

The shim raises `ValueError` with a clear, actionable message at fit time
(not silently falls back) for:

- `optimizer != "online"` — we are SVI-only. The error message names the
  unsupported value and suggests the user switch back to `"online"`.
- `optimizeDocConcentration is True` — we are symmetric-α-only per ADR 0008.
  The error message points to ADR 0008's "Future work" section.
- `docConcentration` is a vector (length-K array) — same reasoning. Scalar
  `docConcentration` is fine.

These are the only three cases. Any other shape that we accept but cannot
honor would be a bug.

## Testing

### `spark-vi/tests/test_mllib_lda.py` — new

Fast unit tests, no `@pytest.mark.slow` marker. Fixture: a tiny well-separated
3-topic corpus, ~50 docs, vocab size ~12. Constructed in a module-level
fixture and reused.

- `test_default_params_match_mllib_lda` — instantiate `VanillaLDAEstimator()`,
  verify every shared Param has the same default as `pyspark.ml.clustering.LDA()`.
  (Subset of: `k`, `maxIter`, `featuresCol`, `topicDistributionCol`,
  `optimizer`, `learningOffset`, `learningDecay`, `subsamplingRate`,
  `optimizeDocConcentration`.)
- `test_param_translation_to_model_and_config` — set non-default values via
  kwargs, call `_build_model_and_config(estimator)` directly, assert each
  field maps to the right `VanillaLDA` / `VIConfig` attribute.
- `test_fit_returns_model_with_correct_shape` — fit on the fixture, assert
  returned object is `VanillaLDAModel`, `topicsMatrix()` is a Matrix of size
  (V × K), `vocabSize()` equals the fixture's V.
- `test_transform_adds_topic_distribution_column` — transform the fixture,
  assert the new column exists with type Vector and length K, and that all
  rows sum to ≈1.0.
- `test_describe_topics_returns_top_k_per_topic` — call with
  `maxTermsPerTopic=3`, assert schema, length, and that each row's
  `termWeights` is descending-sorted.
- `test_model_param_getters_reflect_estimator_config` — set non-default
  Params on the Estimator, fit, verify `model.getK()`, `model.getMaxIter()`,
  `model.getLearningOffset()`, etc. return the same values that the Estimator
  was configured with.
- `test_unsupported_optimizer_em_raises` — `setOptimizer("em")` then `fit`
  raises `ValueError` whose message names `"em"` and the supported value
  `"online"`.
- `test_optimize_doc_concentration_raises` — `setOptimizeDocConcentration(True)`
  then `fit` raises `ValueError` referencing ADR 0008.
- `test_vector_doc_concentration_raises` — `setDocConcentration([0.1] * k)`
  then `fit` raises `ValueError`.
- `test_log_perplexity_raises_not_implemented` — calling on a fitted model
  raises `NotImplementedError`.

### Existing slow parity test stays as the source-of-truth gate

`charmpheno/tests/test_lda_compare.py::test_vanilla_lda_matches_mllib_on_well_separated_corpus`
continues to be the rigorous parity gate. The test body does not change —
only its `run_ours` callee is now a thin wrapper around the shim. Threshold
(`best_diag < 0.20 nats`, typical observed ~0.01) unchanged.

### `charmpheno/.../evaluate/lda_compare.py::run_ours` rewrite

Replace the body that builds a `BOWDocument` RDD + calls `VIRunner.fit`
directly, with a `VanillaLDAEstimator` invocation against the same DataFrame
that `run_mllib` already accepts. Net effect: the comparison driver becomes
"fit two estimators with matched params, compare outputs," which is exactly
what the shim is designed for. Function signature and `LDARunArtifacts`
return type unchanged so existing callers (`compare_lda_local.py` and the
slow parity test) need no modification.

## Migration & rollout

Single-branch implementation, no feature flag. The shim adds a module rather
than changing any existing one (apart from `lda_compare.py::run_ours`, which
becomes thinner). Existing direct uses of `VanillaLDA` + `VIRunner` continue
to work unchanged.

## Open questions deferred to follow-up work

These are explicitly *not* in scope for this spec, but worth recording:

- **Persistence (`MLReadable`/`MLWritable`).** Day-1 driving use case is
  comparison and pipeline ergonomics, not `Pipeline.save()`. When a
  concrete user wants this, the question of which Param values and which
  parts of `VIResult` to round-trip becomes specific instead of speculative.
- **Generic `VIModel` → MLlib adapter.** Punted until OnlineHDP arrives as a
  second data point on what differs across models.
- **`logLikelihood` / `logPerplexity` proper.** Held-out ELBO bound; can be
  derived but takes care, and we don't have a use case yet.
- **Asymmetric α + `optimizeDocConcentration`.** Already ADR 0008 future
  work; the shim raises clear errors when these are requested.
