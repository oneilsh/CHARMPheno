# 0056 — STM is an MLlib Estimator and Model, with covariates from DataFrame columns and no pickle

**Status:** Accepted
**Date:** 2026-11-25
**Supersedes:** ADR 0024 §4 (the ModelSpec persisted "via formulaic's standard
serialization", i.e. a pickle) and its consequence naming a "formula + covariate
DataFrame" input path; the STM exclusion recorded in ADR 0055's alternatives.

## Context

ADR 0055 put the LDA, HDP and DAG-gated topic models on one MLlib persistence
(Params metadata plus a `result/` VIResult, Pipeline-capable) and left STM out:
`StreamingSTM` was not an `Estimator` (snake_case constructor, fit-time keyword
arguments), `STMModel` was not a `Model` (no Params, no `transform`), it saved its own
layout, and it pickled the formulaic `ModelSpec` beside it. The formula path also took
a separate `covariate_df` + `join_key` that `fit` never used: a caller had to resolve
the spec from pandas first and still supply a pre-built covariate column. Publishing
spark-vi (split scoping report, WP2) needs STM in line with the others.

## Decision

1. `OnlineSTMEstimator` / `OnlineSTMModel` (`spark_vi.mllib.topic.stm`; the old names
   are gone) are MLlib stages: camelCase Params with defaults (the old keyword
   arguments, renamed; the SVI schedule defaults stay STM's own: `learningOffset` 64,
   `learningDecay` 0.7, `subsamplingRate` 0.2), `setOnIteration`, the persistence
   Params, `_ResultStagePersistence`. `topicBlocks` is held as a JSON string so it
   round-trips through the metadata. No math changed.
2. Covariates are columns of the input DataFrame, given one of two ways: a pre-built
   vector column with its names (`covariatesCol` + `covariateNames`), or a formula
   over the DataFrame's columns (`covariateFormula`). For a formula, `fit` discovers
   the categorical levels with Spark (ADR 0024's schema-frame; a variable is
   categorical when it is the first argument of a `C(...)` or a string / boolean
   column, or as `categoricalCols` says), fits the spec on the schema-frame, and
   encodes each partition with it. Rows with a null formula column are dropped at
   fit (logged) and get a null at transform; a level the fit never saw fails the
   transform. The separate covariate frame and join key are dropped: the caller joins.
3. No pickle in a saved model. The spec was fitted on the schema-frame alone, so the
   formula plus that frame is its whole learned state: `covariate_spec.json` holds
   both (column order, dtypes, values) and the design-column names; `load` refits the
   formula on the stored frame and refuses a result whose columns differ. A loaded
   model encodes any row bit-identically to the fitted one (tested, unseen levels
   included). formulaic has no serialization of its own besides pickle (its encoder
   state holds contrast objects).
4. `OnlineSTMModel.transform` appends `topicDistributionCol`: per document, the gated
   mode θ (softmax of the MAP η over the document's allowed topics) of the cold-
   started Laplace E-step at the fitted Γ and Σ, as the corpus helpers compute it.
   E[log β] and Σ go to executors as memory-mapped files (`ship_array`).
5. No backward compatibility (maintainer, for all models): the old `STMModel` layout
   does not load.

## Alternatives considered

- Keep the pickle: not a portable format, and unsafe to load from an untrusted file.
- Persist formulaic's encoder state field by field: it holds contrast objects and is
  internal; refitting on the schema-frame uses only the public API, as ADR 0024 did.

## Consequences

- Breaking: the class names, the constructor and `fit` keywords, and the saved layout.
- The CHARMPheno drivers (`stm_bigquery_cloud.py`, `fit_stm_local.py`) fit through the
  Estimator on the pre-built covariate column (the covariate cache builds it) and keep
  writing their run dir as a bare VIResult checkpoint (plus `params/correlation.npy`,
  `covariate_names.json`), the layout the dashboard builders, `--resume-from` and
  `run_experiment` read; the run dir no longer carries `model_spec.pkl` (nothing read
  it). The workspace covariate cache still pickles its spec (`_covariates_cache.py`):
  a cluster-local cache, not a saved model.
- STM has no Spark-free (LocalRunner) shim, as before.
