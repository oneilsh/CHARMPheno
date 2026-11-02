# Plan: the DAG-gated user API, the parked engine ideas, and the clinical migration

**Date:** 2026-11-02
**Spec:** [2026-10-30-dag-gated-user-api-design.md](../specs/2026-10-30-dag-gated-user-api-design.md)
(R1-R13, decisions 1-5).
**Branch:** `claude/dismech-ribbon`. No backward compatibility (maintainer, 2026-11-02).
**Handoff notes:** [reports/2026-11-02-walkthrough-and-redesign-handoff.md](../../reports/2026-11-02-walkthrough-and-redesign-handoff.md).
**Out of scope:** pooled multi-domain normalization (a modeling change, its own
experiment); the readout's home (decided after this work).

Each work package (WP) is one or a few commits, each with tests; the spark-vi unit tier
runs on every commit, the package's slow tier on any change to `spark_vi.core`,
`spark_vi.readout` or a model's `local_update` / `update_global`. Dispositions are
filled in as packages land.

## Order and packages

| WP | what | spec | validates locally | disposition |
|---|---|---|---|---|
| 1 | **Any-URI I/O** (`spark_vi.io.fs`: scheme dispatch, plain Python for local paths, Hadoop FS via py4j otherwise; manifest-last on object stores). Callers: `save_result` / `load_result`, readout checkpoint + heads file, `DagHeadsModel.save` / `load`, shims' `resumeFrom` check. | R13 L1 | yes (`file://` URIs on local Spark) | done (`spark_vi/io/fs.py`, tests `test_io_fs.py`) |
| 2 | **Broadcast expElogbeta, not λ**, in the fit: a `VIModel` hook (what the E-step needs per iteration; default `global_params`), used by the gated engine and OnlineLDA. | parked | yes (Spark-vs-local agreement, full-batch gate) | done (`VIModel.broadcast_params`; slow tier pending at commit) |
| 3 | **Sparse partition statistics** in the gated E-step: accumulate only touched (topic, word) cells, densify at combine. | parked | yes (bit-identical stats on fixtures) | done (row-sparse, not cells: `spark_vi.core.row_sparse.RowSparseStats` holds the touched topic rows full-V wide, since every doc touches the whole background block; gated `combine_stats` unions rows and densifies past K/2; `update_global` densifies; bit-identical to the dense path, tests `test_gated_row_sparse_stats.py`) |
| 4 | **DAG input + node-set learning** (`spark_vi.dag` internal helpers): R1 parsing (dict / Spark DF / pandas DF, mixed ids), validation, support pass, prune, land, reduce, splice; `DagLayout` / `frontier_from_coded` made private. | R1, R3, R4, dec. 5 | yes | done (`spark_vi/dag/nodeset.py`: `parse_dag`, `count_support` / `count_support_spark`, `learn_node_set` / `fit_node_set` -> `NodeSet`; tests `test_dag_nodeset.py`). `DagLayout` / `frontier_from_coded` dropped from `spark_vi.dag` exports, every importer reads `spark_vi.dag.layout`; `layout.py` / `ops.py` untouched, but the import move in `case_finding_assembly` / `multi_domain` is an announced full bundle-cache drop (tripwires re-pinned). |
| 5 | **`DagGatedLDAEstimator` on the new inputs**: `labelsCol`, DAG setter, `minDocsPerNode`, `unknownLabels`, `topicConcentration`; engine ids internal; frontier reduction internal. **Model**: `dag()`, `prunedNodes()`, `splicedNodes()`, `nodeTopics()`, `backgroundTopics()`, `describeNodes()`. Old `gateParent` / `frontierCol` removed. | R2-R5, R9 | yes | done (`spark_vi/mllib/topic/dag_gated_lda.py`). The DAG Param is `dagEdges` (a validated JSON edge list keeping str / int ids; set with `setDag` or the `dag=` kwarg), because the model's `dag()` method owns the name. `fit` checks the labels column's element type against the DAG's id types, counts support (`count_support_spark`), learns the node set, builds the layout from `engine_parents` and maps labels to the frontier through a broadcast `FrontierFn`; one log line (docs; given / kept / pruned / spliced). `domainBounds` removed: widths are the vectors' sizes (ML attribute size when present, else the first row), every row checked. Model: `dag()`, `prunedNodes()`, `splicedNodes()`, `nodeSupport()`, `nodeTopics()`, `backgroundTopics()`, `describeNodes()` (node column long when all ids are int, string otherwise; background rows have a null node); internal `_frontier_fn()` / `_layout()` for WP8. The node set rides `VIResult.metadata["dag_node_set"]` (engine `extra_metadata`; `NodeSet.to_dict` / `from_dict`), so `save` / `load` keep it. `topicConcentration` (unset = 1/K). Tests `test_mllib_dag_gated_lda.py`. Driver tests that set the old Params break until WP12: `scripts/tests/test_run_experiment_dag_gated.py::test_config_round_trips_frontmatter_to_the_estimator`, `tests/scripts/test_gate_schedule_args.py::test_driver_flags_reach_the_estimator`. |
| 6 | **Seed extras in user ids**: `nodeTerms`, profile keyed by node and term; vocabularies per feature column; `gateSeedDomain` removed. | R7 | yes | done (`nodeTerms`, `gateSeedProfile`, `vocabularies`; `gateSeedOwnCodes` / `gateSeedDomain` removed; engine API unchanged: translated to `{engine id: [concat idx]}` / `{engine id: {concat idx: w}}`). Terms match exactly as given (vectorizer vocabularies are strings; a miss whose str/int twin is present raises). A term in two vocabularies must be written `(featureColumn, term)`, else fit refuses it. Entries of unkept nodes and terms in no vocabulary are dropped and counted in the log. A profile replaces a node's seed in each domain its terms touch. |
| 7 | **Prep stage** `DagGatedFeaturizer` (array-of-terms or long tables per domain, labels from an array or long table, vocabularies carried); `domainBounds` removed. | R12 | yes | done (prep stage: `spark_vi/mllib/topic/featurizer.py`, `DagGatedFeaturizer` -> `DagGatedFeaturizerModel`, exported from `spark_vi.mllib[.topic]`; domains via `addArrayDomain` / `addTableDomain`, labels via `setLabelsInputCol` / `setLabelsTable`; one `CountVectorizer` per domain (`minDF` / `maxDF` / `vocabSize` / `minTF`), long-table counts expanded with `array_repeat` (non-integer counts fail by name); model gives `domains()` / `featuresCols()` / `vocabularies()` / `labelsCol()`. Long tables are instance state, not Params: carried by `copy()` and into the model, replaced with `setDomainTable` for new documents, not persisted. MLWritable / MLReadable done here (Params metadata + vocabulary parquet, any Hadoop URI; `PipelineModel` round trip tested), ahead of WP 10. Tests `test_mllib_featurizer.py`. The `domainBounds` removal is the estimator's, in WP 5/6.) |
| 8 | **`DagHeadsEstimator` on the same inputs**: `labelsCol` + DAG or a fitted `DagGatedLDAModel`; `labelMaskMode`; outputs by node id; `degenerateNodes()`; `saveDir` / `saveInterval` / `resumeFrom`. Local/Spark data interfaces aligned. | R6, R8, R13 | yes | |
| 9 | **Defaults and preset**: MLlib-like defaults; a documented working preset; help text final. | R10, R11 | yes | |
| 10 | **MLlib persistence**: `MLWritable` / `MLReadable` for both models (DefaultParamsWriter metadata + our data files through WP1); `Pipeline.save` / `PipelineModel.load` round trip. | R13 L2 | yes | |
| 11 | **charmpheno: the Mondo build hands over the subgraph** (coded terms + ancestors, labels at most-specific terms); powering and `collapse` leave the build. **Bundle** stores attested node-id arrays (gate and label-year) instead of `frontier` / `label` / `labelMask`. Hashed-module edit: ONE announced full cache drop, tripwires re-pinned. | "outside spark-vi" | partly (tripwires, unit) | |
| 12 | **Drivers**: fit driver sets the new Params (`minDocsPerNode=100`), saves the fitted DAG and id map; readout driver builds DagHeads inputs from the model; readout DAG, framings member lists, own codes and sidecar mappings read the model's node set. Workup / conversion sidecars follow by ancestry. | "outside spark-vi" | driver tier | |
| 13 | **Digest measurement domain** (`inspect_topics --digest-domains`). | parked | yes | done |
| 14 | **Full test tier** (`make test-all` equivalent), docs (SPARK_VI_FRAMEWORK, LABEL_RULES, AGENTS), then the **cluster validation** below. | | | |

Dependencies: 4 before 5, 6, 8, 11, 12; 5 before 6, 8, 9, 10; 1 before 8 and 10;
7 is independent of 8. 11-12 last, as a pair.

## Cache-key impact

- WP 2-10: spark-vi only. `spark_vi/dag/{layout,ops}.py` are source-hashed; WP 4 edits
  them (making `DagLayout` private, adding helpers), so WP 4 is an announced cache drop
  that re-pins the tripwires in the same commit (run
  `tests/scripts/test_case_finding_cache_mondo.py` on every `spark_vi/dag` edit).
  As built, WP 4 left `layout.py` / `ops.py` untouched (the helpers live in the
  unhashed `nodeset.py`); the drop came from the hashed importers
  (`case_finding_assembly.py`, `multi_domain.py`) switching to
  `from spark_vi.dag.layout import ...`: all four tripwire keys moved.
- WP 11: `charmpheno/mondo/native_dag.py`, `collapse.py` and the assembler modules:
  one announced full drop. The conversion and workup sidecar keys move if their modules
  change.

## Cluster validation (after WP 14)

1. `--bundle-only` rebuild: the corpus counts (documents, tokens per domain) match
   0163's `bundle_meta.json`; the node set is the new estimator's, reported beside
   0163's (kept / pruned / spliced counts).
2. A fit with the 0163 configuration (the working preset, `minDocsPerNode=100`) and its
   readout: per-node AUC paired against 0163 over the shared nodes; stacked detection;
   legibility census.
3. The framings survey and the workup run on the new readout (heads tag per the new
   names).
