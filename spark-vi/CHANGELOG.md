# Changelog

All notable changes to spark-vi. The project follows [semantic versioning](https://semver.org/):
while the version is 0.x, a minor release may change the public API, and every such
change is listed here.

## [Unreleased]

### Breaking
- No backward compatibility for saved models. `OnlineLDAModel.load` /
  `OnlineHDPModel.load` read only the new layout (`metadata/` + `result/`); a model
  saved by an earlier version, and a `saveDir` checkpoint, are bare VIResult
  directories: wrap one with `OnlineLDAModel(load_result(dir))` (Params at their
  defaults). A model saved now does not load in earlier versions.
- STM is an MLlib Estimator / Model pair (ADR 0056): `StreamingSTM` and `STMModel` are
  replaced by `OnlineSTMEstimator` and `OnlineSTMModel` (also exported from
  `spark_vi.mllib`). The constructor keywords and `fit`'s keywords become camelCase
  Params (`K` -> `k`, `features_col` -> `featuresCol`, `covariates_col` ->
  `covariatesCol`, `covariate_names` -> `covariateNames`, `covariate_formula` ->
  `covariateFormula`, `random_seed` -> `seed`, `max_iter` -> `maxIter`,
  `subsampling_rate` -> `subsamplingRate`, `tau0` -> `learningOffset`, `kappa` ->
  `learningDecay`, `topic_blocks` -> `topicBlocks`, `doc_group_col` -> `docGroupCol`,
  and likewise every hardening and spectral-init option; `resume_from` ->
  `resumeFrom`, `on_iteration` -> `setOnIteration`); defaults unchanged. The formula
  reads the input DataFrame's columns; the separate `covariate_df` / `join_key` and
  `_resolve_model_spec_from_pandas` are gone. The fitted model is `model.result`
  (a VIResult) plus Params. Saved layout: `metadata/` + `result/` (+
  `covariate_spec.json`); the old layout (`manifest.json` + `model_spec.pkl` +
  `covariate_names.json` at the top) does not load.

### Removed
- `spark_vi.mllib._common._PersistableModel` (internal; replaced by
  `_ResultStagePersistence`).
- The per-node supervised readout (`spark_vi.readout`, `spark_vi.mllib.readout`'s
  `DagHeadsEstimator` / `DagHeadsModel`) and the exact ranking metrics
  (`spark_vi.eval.metrics`) left the package; they belong to the application that
  uses them. Models saved through MLlib persistence with those classes do not load.
- The simulator-study helpers (`spark_vi.eval.topic.seed_panel`, the recovery parts of
  `concentration_recovery`).

### Added
- `OnlineSTMModel.transform`: each document's topic proportions (the gated mode of its
  logistic-normal posterior) from its words and covariates; rows with a null
  covariate get a null, an unseen categorical level fails the job. Also
  `describeTopics`, `topicCorrelation`, `topicBlockPartition`, `encodeCovariates`.
- A formula model saves its covariate spec as JSON (the formula and the schema-frame
  it was fitted on) and rebuilds it on load; no pickle in any saved model.

### Changed
- `OnlineLDAModel` and `OnlineHDPModel` use MLlib persistence (ADR 0055): `save(path)`
  writes `metadata/` (Params) and `result/` (the VIResult), refuses an existing path
  (`write().overwrite().save(path)` replaces it), and the models work inside a saved
  `PipelineModel`.
- Params metadata is written and read by spark-vi's own code in MLlib's format
  (`params_metadata_json`), no longer through pyspark private API; pyspark is declared
  `>=3.5` with no upper bound.
- The held-out split and predictive log-likelihood used by STM and `predictive_gain`
  live in `spark_vi.eval.topic.heldout`.
- `DagGatedLDAModel.transform` ships its scoring table (E[log β]) to the executors as a
  memory-mapped `.npy` (`spark_vi.core.broadcast.ship_array`) instead of a broadcast:
  one copy per executor rather than one per Python worker, and no pickled object near
  the 2 GB cap. Scores are bit-identical.
- DAG node ids and term ids are strings; a non-string label raises `TypeError`.
- `DagGatedLDA`: `topicConcentration` accepts one value per domain; a node block is
  flat only when every domain of every one of its topics is flat
  (`nodeStatistics` reports `supportFractionByDomain` and `dataMassByDomain`).

## [0.1.0]

The version the package has carried during development; never released.
