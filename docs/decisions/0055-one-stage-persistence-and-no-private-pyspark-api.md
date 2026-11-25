# 0055 — One MLlib persistence for the topic models, and our own Params metadata writer

**Status:** Accepted; the STM exclusion (Alternatives) superseded by ADR 0056
**Date:** 2026-11-25

## Context

Ahead of publishing spark-vi on its own (`docs/reports/2026-11-24-spark-vi-split-scoping.md`,
WP2 and WP3), two things stood out. Persistence came in two styles: the DAG-gated
stages used MLlib's (`_StagePersistence`: Params metadata plus our data files,
Pipeline-capable), while `OnlineLDAModel` / `OnlineHDPModel` kept `_PersistableModel`,
a bare `VIResult` directory with no Params, as ADRs 0009 and 0012 deferred MLlib
persistence "until a concrete user appears". And the metadata writer and reader called
pyspark's private `DefaultParamsWriter._get_metadata_to_save` and
`DefaultParamsReader._parseMetaData`, which any pyspark release may change: an
obstacle to declaring pyspark as a lower bound only (the maintainer's choice).

## Decision

1. `spark_vi.mllib._common.params_metadata_json` / `parse_params_metadata` build and
   parse MLlib's metadata document (`class`, `timestamp`, `sparkVersion`, `uid`,
   `paramMap`, `defaultParamMap`, extra keys; same key order, compact separators) from
   the public `Params` API, in MLlib's file layout (`metadata/part-00000` plus
   `_SUCCESS`). A fixture written by the private writer (`spark-vi/tests/data/
   metadata_v0`) pins the document byte for byte. pyspark is declared
   `>=3.5`, no upper bound.
2. `_ResultStagePersistence` (`metadata/` + `result/`, the VIResult in ADR 0006's
   format) backs `OnlineLDAModel`, `OnlineHDPModel` and `DagGatedLDAModel`: Params
   round-trip, `PipelineModel` save / load, MLlib's refuse-unless-overwrite rule.
   `_PersistableModel` is removed. Supersedes the persistence deferral of ADR 0009 and
   ADR 0012 (the Estimators of LDA / HDP stay unpersisted: nobody has asked).
3. No backward compatibility for saved model layouts (maintainer, 2026-11-25): `load`
   reads only the new layout. A bare VIResult directory (the old LDA / HDP `save`
   layout, every `saveDir` checkpoint, the drivers' `save_result` output) is not a
   saved stage; `OnlineLDAModel(load_result(dir))` wraps one, Params at their
   defaults. The metadata fixture stays: it pins our writer to MLlib's document, which
   is what keeps a saved stage readable by MLlib, not an old spark-vi format.

## Alternatives considered

- Keep the private calls and pin pyspark `<4`: rejected; the pin is the problem.
- Convert `STMModel` too: it is not an MLlib `Model` (no Params, no `transform`) and
  `StreamingSTM` is not an `Estimator`; making them stages is a redesign, recorded in
  the scoping report as follow-up work.

## Consequences

- Breaking, both ways: an LDA / HDP model saved before this ADR does not `load` (wrap
  it with `load_result`), and one saved now does not load in older spark-vi.
- `save` now refuses an existing path (`write().overwrite().save` replaces it); the old
  `save` wrote into it.
- Remaining underscore pyspark API, all deliberate: `Params._dummy`, `_set`,
  `_setDefault`, `_resetUid`, `_input_kwargs` (the documented pattern for Python stages,
  used by pyspark's own); `sc._jvm` / `sc._jsc` / `SparkContext._active_spark_context`
  for Hadoop-filesystem access and context checks in `spark_vi.io.fs`, `core.retry` and
  `diagnostics.persist`, which PySpark exposes no public way to do.
