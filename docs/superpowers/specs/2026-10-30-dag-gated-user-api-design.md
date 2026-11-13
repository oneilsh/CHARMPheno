# The DAG-gated estimators take a DAG and labelled documents; the rest is internal

**Date:** 2026-10-30
**Status:** accepted 2026-11-02 (decisions below); plan: docs/superpowers/plans/2026-11-02-dag-gated-user-api-plan.md
**Touches:** `spark_vi.mllib.topic.dag_gated_lda`, `spark_vi.mllib.readout` (DagHeads),
`charmpheno.omop` assembler (bundle columns), `charmpheno.mondo.native_dag` (pruning),
the fit and readout drivers.

## Why

A user of `DagGatedLDAEstimator` today must do the estimator's bookkeeping first:

1. renumber their ontology into dense engine ids `1..N`, root `0` implicit, and pass it
   as a JSON string (`gateParent`);
2. reduce each document's labels to its frontier, the most-specific attested nodes, in
   those engine ids (`frontierCol`; `spark_vi.dag.frontier_from_coded`);
3. for the heads, build dense per-node `label` / `labelMask` arrays from the same
   frontier (`charmpheno` `attach_labels`);
4. key the seed extras (`gateSeedOwnCodes`, `gateSeedProfile`) by engine node id AND
   by an index into one domain's slice of the concatenated vocabulary;
5. after the fit, rebuild a `DagLayout` to learn which topics belong to which node
   (`topicsMatrix` / `describeTopics` are keyed by topic index only).

Pruning rare nodes (with the transitive reduction and the landing of a dropped node's
attestations on its nearest kept ancestors) is done before any of this, in
`charmpheno.mondo.native_dag`, although it is generic DAG work that depends only on the
training data's support.

The goal: **the user supplies the DAG and documents labelled with DAG nodes, in their
own ids; everything else is automatic.** Domain work stays outside: mapping raw codes
to ontology nodes, deciding which events attest a node (the label window, the strip;
`docs/architecture/LABEL_RULES.md`), and what a node's "own codes" are.

## Requirements

**R1. The DAG** is given once, in the user's node ids (any `str` or `int`), as any of:
- a `dict` `{child: parent | [parents]}`;
- a Spark DataFrame with columns `(child, parent)`, one row per edge (collected once
  on the driver: a label DAG is small);
- a pandas DataFrame with the same columns.

Nodes with no parent hang from a synthetic root the estimator adds. The estimator
validates the DAG (acyclic; every label in the data is a DAG node, or is refused by
name) before any Spark work.

**R2. The labels** are one column, `labelsCol`: an ARRAY column (`array<string>` or
`array<int>`, as `CountVectorizer` takes) of the node ids each document attests; the
prep stage (R12) builds it from a long table for users who have one. Any nodes: not reduced to the most specific, ancestors allowed, duplicates
allowed. Empty or null = a background document. The estimator computes the gate from
it (the frontier reduction is internal; `allowed_set(frontier)` equals
`allowed_set(labels)` because an ancestor's closure is inside its descendant's).

**R3. The node set is learned at fit time** (like `StringIndexer` learning its
vocabulary), from the DataFrame passed to `fit` and nothing else. In order:
1. **Validate**: acyclic (refuse); self-loops and duplicate edges dropped; every
   parentless node hangs from a synthetic root; a label not in the DAG is refused by
   name (`unknownLabels="error"`, default) or dropped (`"ignore"`).
2. **Support**: one pass; a node's support is the number of documents whose labels'
   closure contains it.
3. **Prune**: drop nodes with support < `minDocsPerNode` (default 1, so a node no
   document reaches never gets a topic block).
4. **Land labels**: a dropped node's labels move to its nearest kept ancestors
   (`nearest_kept_ancestors`); with none, the document is background.
5. **Reduce**: the transitive reduction over the kept nodes
   (`transitive_reduction_parents`).
6. **Splice chains** (to a fixpoint): remove a kept node with exactly one kept child and
   no documents labelled at the node itself; it admits exactly its child's documents,
   so the two blocks would be trained on identical data. (Generic, data-driven; it
   subsumes the Mondo build's `collapse` splice.)
The model reports `prunedNodes()` (dropped id -> landing ids) and `splicedNodes()`.

**No train/test notion in the shims.** As in MLlib, `fit(df)` learns from what it is
given and the caller splits (`randomSplit`); support is therefore counted on the
training data by construction. The split, the test-feature strip, the incident rule and
the per-node evaluation stay in the consuming package (`LABEL_RULES.md`).

**R4. Engine ids are internal.** The fitted model holds the mapping (user id <-> engine
id) and the kept DAG, both persisted with it.

**R5. Node-keyed outputs** on the fitted model, in user ids:
- `dag()`: the kept DAG (dict, user ids) and `prunedNodes()`: dropped id -> landing ids;
- `nodeTopics(node)`: the topic indices of a node's block; `backgroundTopics()`;
- `describeNodes(maxTerms)`: DataFrame `(node, topic, termIndices, termWeights)`,
  plus `describeTopics` as today;
- `transform` unchanged (ungated theta over all K).

**R6. DagHeads takes the same inputs.** `DagHeadsEstimator` accepts a `labelsCol` and a
DAG (or a fitted `DagGatedLDAModel`, whose kept DAG and id map it reuses) and builds the
per-node label / mask vectors itself (closure labels, `label_mask_mode` as a Param). Its
outputs are keyed by user node id too. The dense `label` / `labelMask` columns stop being
an input contract.

Nodes the heads cannot fit are REPORTED, not silent: a node whose observed training
rows are empty, all-negative or all-positive has no finite optimum, so the readout
predicts its lone class as a constant (`readout.heads.fit_heads`, the sklearn oracle's
fallback). The fitted model exposes `degenerateNodes()` -> `{node: constant}` beside
`prunedNodes()` (R5), and `fit` logs the count.

**R7. Seed extras in user ids.** The node -> attesting-terms relation (`nodeTerms`
`{node: [term]}`, today `gateSeedOwnCodes`) and the external profile (`gateSeedProfile`
`{node: {term: weight}}`) are keyed by user node ids and TERM ids: the identifiers of the
feature vocabulary. The estimator gets the vocabulary of each feature column from the
prep stage (R12) and translates; the domain is implied by which vocabulary a term is
in, so `gateSeedDomain` goes. Both stay optional: a generic user has neither.

Labels and terms may be one vocabulary or two. When the labels ARE terms (e.g. the
DAG is the ICD hierarchy and a document's labels are its own diagnosis codes),
`nodeTerms` is the identity on the shared ids. When they are a separate vocabulary
linked by a domain mapping (Mondo nodes attested by OMOP condition concepts through
charmpheno's code ladder), the consuming package supplies `nodeTerms`. spark-vi only
reads the relation, for the balanced seed; it never decides it.

**Stripping a label's attesting terms from TEST features stays out of spark-vi.** It is
an evaluation-protocol rule (leakage), stated in `docs/architecture/LABEL_RULES.md` and
applied by the consuming package's data prep.

**R8. Separate gate labels and readout labels stay possible.** The gate may be fit on
one attestation (e.g. all-ever, exp 0146) and the heads on another (the label year):
`labelsCol` on each estimator, set independently.

**R9. `topicConcentration` (eta) becomes a Param.** Today it is fixed at `1/K` inside
`_fit`, not settable; MLlib's `LDA` exposes it under that name. Default unchanged.

**R10. Defaults are a working configuration.** Decide (see open questions) whether the
estimator's defaults become the configuration the experiments use (`gateNBg` 200,
`learningOffset` 64, `gateWarm` 2 with the balanced seed and leaf seeds) or stay
MLlib-like with a documented "recommended" preset.

**R11. Help text is domain-neutral** (no Mondo / HPO / experiment numbers in spark-vi;
those references move to the charmpheno driver that sets the Params).

**R12. A prep stage** (`DagGatedFeaturizer`, an Estimator -> Model, the role
`Tokenizer` + `CountVectorizer` play before MLlib's `LDA`) builds the estimator's inputs
from tables a non-specialist has:
- per domain, either an `array<string>` column of terms per document or a long table
  `(doc_id, term[, count])`; it learns each domain's vocabulary (min document frequency,
  max size: `CountVectorizer`'s Params) and emits one sparse vector column per domain;
- labels from an `array` column or a long table `(doc_id, node)`, collected to the R2
  array column;
- the fitted prep model carries the vocabularies (R7) and the domain names, and the
  domain widths are the vectors' own sizes, so `domainBounds` goes (a Spark vector's
  `size` is fixed per column; the "unrepresentative first row" it guarded cannot occur
  for vectorizer output).
Users with vector columns already skip it and pass `featuresCols` + `vocabularies`.

Multi-domain is the DAG-gated engine's alone today (`OnlineLDA`, `OnlineHDP` and STM take
one `featuresCol`); the prep stage is written so another engine could take the same
multi-domain input later.

**R13. Persistence follows one convention, on any Hadoop filesystem.** Today every
file spark-vi writes is opt-in (nothing is written unless a path Param or argument is
set): the topic shims' `saveDir` / `saveInterval` / `resumeFrom`, `DagHeadsModel.save`,
and `fit_heads(checkpoint_path=)` (set only by the readout driver). But the paths are
LOCAL (`pathlib` + `open` on the driver), and the readout checkpoint has its own format
and no shim Params. The redesign:
- `DagHeadsEstimator` gains the same `saveDir` / `saveInterval` / `resumeFrom` Params
  as the topic shims, mapped onto `fit_heads`' checkpoint;
- model `save` / `load` (and the checkpoints) accept any Hadoop-filesystem URI
  (`hdfs://`, `gs://`, `file://`), as MLlib's own `MLWriter` does: a driver-local path
  is wrong under cluster deploy mode, where the driver runs on an arbitrary worker;
- every write stays atomic (stage, then rename / commit), as `save_result` and the
  heads file already are locally.

Two levels, built in this order (maintainer, 2026-10-30):
- **Level 1, any Hadoop-filesystem URI.** One small I/O module (`spark_vi.io.fs`:
  open for write / read, exists, rename, delete, list) dispatching on the URI scheme:
  no scheme or `file://` -> plain Python I/O, exactly as today (so the gcsfuse-mounted
  run dirs behave identically, and no Hadoop `LocalFileSystem` `.crc` sidecars appear);
  `hdfs://`, `gs://`, `s3a://`, ... -> the running SparkContext's Hadoop `FileSystem`
  through py4j (no new dependency; `gs://` through the cluster's GCS connector). Remote
  reads load into memory (np.load needs a seekable file; the objects are driver-sized
  anyway). On object stores a single object is never torn (it becomes visible only when
  its upload completes), so atomicity is "write the manifest last", with no directory
  renames (those are copies there). Callers: `save_result` / `load_result`, the readout
  checkpoint and heads file, `DagHeadsModel.save` / `load`, the shims' `resumeFrom`
  check. A run may then write straight to an in-boundary bucket instead of the fuse
  mount (an option; the bucket is the maintainer's to name).
- **Level 2, MLlib persistence.** Each model implements `MLWritable` / `MLReadable` with
  a small `MLWriter` / `MLReader`: MLlib's standard params metadata
  (`DefaultParamsWriter`) plus our own data files (npy / npz, through Level 1) under the
  stage's directory. Then `Pipeline([DagGatedLDA, DagHeads]).fit(df).save(uri)` and
  `PipelineModel.load(uri)` work like any built-in stage's (today fitting the pipeline
  works but saving it does not: the models' `save(path)` is custom). Built once, on the
  redesigned estimators.
- **Where a save may go (as built; corrected 2026-11-13, code review C2).** A single
  stage's `save` / `load` writes and reads its Params metadata FROM THE DRIVER
  (`spark_vi.mllib._common.write_params_metadata`), so it works on any path, a
  driver-local one included. `Pipeline.save` / `PipelineModel.load` are MLlib's own:
  they write and read the pipeline's own metadata with a Spark job (`saveAsTextFile` /
  `textFile`), i.e. on EXECUTORS (the stages inside still use our driver-side writers),
  so on a multi-node cluster a pipeline needs a shared filesystem (HDFS, GCS); a
  driver-local path fails there.

**R14. Node statistics, and dropping the nodes the data did not sustain** (addendum
2026-11-10, maintainer; evidence: exps 0169 / 0171). A fit long enough to forget its
seeds leaves some node blocks flat at the η prior: blocks seeded at opening that the
data never renews (0171: none starved from iteration 30 to 70, then 205 of 2,845 cross
together 45-60 iterations after the full gate opens, then the count plateaus; support
decides it, 100-1,000 documents). Whether to act on that is the user's call, as with the
other models, whose topic statistics are reported (the dashboard build's prevalence and
coherence) and nothing is removed. So:
1. **Reporting, by default.** The fitted model gets `nodeStatistics(flatThreshold=0.5)`:
   a DataFrame, one row per kept node (node id typed as in `describeNodes`): `depth`,
   `support` (documents whose labels' closure reaches it, as in `nodeSupport()`),
   `supportFraction` (the block's effective support exp(H(β)) over its dominant
   domain's vocabulary, 0 = sharp, 1 = flat; the minimum over the block's topics when
   `gateTpn > 1`), `mass` (the block's λ mass), `massToParent` (its mass over its
   heaviest kept parent's; null for a top-level node) and `flat` (`supportFraction >
   flatThreshold`). Domain-neutral; the census's "starved" is `flat` at 0.5. The model
   and its node set are unchanged; nothing is dropped. No iteration-count warning:
   whether a fit has settled is read off its standard diagnostics (the ELBO trace).
2. **Dropping, explicitly.** `model.dropUnsupportedNodes(maxSupportFraction=0.5)`
   returns a NEW model without the flat nodes, built with the node-set machinery of R3
   so it means the same thing as pruning there: each dropped node's labels land on its
   nearest kept ancestors (background when none); its kept children are re-attached to
   its nearest kept ancestors and the kept DAG is re-reduced; pruned and spliced nodes
   that landed on a dropped node land further up; engine ids are renumbered and every
   K-sized fitted array (λ per domain, α, the gate state) keeps only the surviving
   topics, in the new order. No re-splice and no refit: only blocks the data did not
   sustain are removed. The new model reports them in `unsupportedNodes()` -> `{node:
   [landing nodes]}` beside `prunedNodes()` / `splicedNodes()`, persisted with the node
   set (R13). A dropped node gets no topic block, no gate and no readout head
   (DagHeads built from the cleaned model by `setTopicModel` uses its node set, R6).
   Because a starved node may have fed children, the kept set is no longer closed
   upward; the landing and reduction must not assume it is.

## What changes outside spark-vi

- **The bundle** stores the attested node ids per document (two columns when R8 is
  used: gate attestation and label-year attestation) instead of `frontier` + dense
  `label` / `labelMask`. An assembler edit: one announced cache drop. The readout
  reads the label vectors from DagHeads (R6), not from the bundle.
- **`charmpheno.mondo.native_dag`** keeps building the Mondo graph and the code ladder
  (codes -> Mondo terms) and hands over the relevant subgraph (every coded term and its
  ancestors, within `branch_root` when set) with labels at the most specific terms; its
  powering (`min_positives`) and `collapse` splice move into the estimator (R3). The
  bundle no longer depends on the threshold. The readout DAG, the framings' member
  lists, the own codes and the sidecar mappings read the node set from the fitted
  model. The clinical driver sets `minDocsPerNode=100` (>= the egress floor; now
  counted on training documents only, so the node set shifts slightly from 0163's and
  validation compares over the shared nodes).
- **The drivers** set `labelsCol` / the DAG instead of `frontierCol` / `gateParent`, and
  read node topics from the model (R5) instead of rebuilding a `DagLayout`.

## Decisions (maintainer, 2026-10-30)

1. **Defaults (R10):** MLlib-like defaults, plus a documented preset for the working
   configuration (`gateNBg` 200, `learningOffset` 64, `gateWarm` 2, balanced + leaf seeds).
2. **Node ids:** mixed `str` and `int` accepted, kept as given (no normalization: `1`
   and `"1"` are different nodes); outputs use the same values.
3. **Where Mondo pruning lives:** in the estimator (option (b), 2026-11-02): R3's
   generic validate / support / prune / land / reduce / splice. `minDocsPerNode`
   defaults to 1; unknown labels are refused by default (`unknownLabels="ignore"` to
   drop them). The shims carry no train/test notion.
4. **Back-compat:** none. Old Params are removed, not aliased.
5. **`DagLayout` / `frontier_from_coded`:** internal (not exported from `spark_vi.dag`);
   the model's R5 accessors are the public surface.

## Parked with this spec

- **Where the readout lives.** The readout (`spark_vi.readout`, DagHeads) is generic
  code; the maintainer may move it back to charmpheno given the framing results. To be
  decided after the readout walkthrough; R6 is written so DagHeads works either way.

## Not in scope

- Multi-domain modelling changes (pooled normalization, per-domain weighting): parked as
  a separate experiment candidate.
- The broadcast-expElogbeta and sparse-statistics performance ideas: parked.
