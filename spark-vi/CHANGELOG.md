# Changelog

All notable changes to spark-vi. The project follows [semantic versioning](https://semver.org/):
while the version is 0.x, a minor release may change the public API, and every such
change is listed here.

## [Unreleased]

### Removed
- `spark_vi.mllib._common._PersistableModel` (internal; replaced by
  `_ResultStagePersistence`).
- The per-node supervised readout (`spark_vi.readout`, `spark_vi.mllib.readout`'s
  `DagHeadsEstimator` / `DagHeadsModel`) and the exact ranking metrics
  (`spark_vi.eval.metrics`) left the package; they belong to the application that
  uses them. Models saved through MLlib persistence with those classes do not load.
- The simulator-study helpers (`spark_vi.eval.topic.seed_panel`, the recovery parts of
  `concentration_recovery`).

### Changed
- `OnlineLDAModel` and `OnlineHDPModel` use MLlib persistence (ADR 0055): `save(path)`
  writes `metadata/` (Params) and `result/` (the VIResult), refuses an existing path
  (`write().overwrite().save(path)` replaces it), and the models work inside a saved
  `PipelineModel`. `load` still reads the old layout (a bare VIResult directory, also
  every `saveDir` checkpoint), with the Params at their defaults. A model saved this
  way does not load in earlier versions.
- Params metadata is written and read by spark-vi's own code in MLlib's format
  (`params_metadata_json`), no longer through pyspark private API; pyspark is declared
  `>=3.5` with no upper bound.
- The held-out split and predictive log-likelihood used by STM and `predictive_gain`
  live in `spark_vi.eval.topic.heldout`.
- DAG node ids and term ids are strings; a non-string label raises `TypeError`.
- `DagGatedLDA`: `topicConcentration` accepts one value per domain; a node block is
  flat only when every domain of every one of its topics is flat
  (`nodeStatistics` reports `supportFractionByDomain` and `dataMassByDomain`).

## [0.1.0]

The version the package has carried during development; never released.
