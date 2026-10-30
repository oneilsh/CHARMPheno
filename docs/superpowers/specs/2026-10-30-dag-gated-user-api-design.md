# The DAG-gated estimators take a DAG and labelled documents; the rest is internal

**Date:** 2026-10-30
**Status:** draft for maintainer review (from the 2026-10-29/30 walkthrough of `spark_vi`)
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

**R2. The labels** are one column, `labelsCol`: an array of the node ids each document
attests. Any nodes: not reduced to the most specific, ancestors allowed, duplicates
allowed. Empty or null = a background document. The estimator computes the gate from
it (the frontier reduction is internal; `allowed_set(frontier)` equals
`allowed_set(labels)` because an ancestor's closure is inside its descendant's).

**R3. Pruning is a fit-time step** (like `StringIndexer` learning its vocabulary):
`minDocsPerNode` (default 0 = keep every node). A node attested, in its closure, by
fewer training documents than that is dropped; the kept DAG is the transitive
reduction of the order restricted to kept nodes (`spark_vi.dag.ops.
transitive_reduction_parents`), and a dropped node's labels land on its nearest kept
ancestors (`nearest_kept_ancestors`). One extra pass over the labels column counts the
support.

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

**R7. Seed extras in user ids.** `gateSeedOwnCodes` `{node: [term]}` and
`gateSeedProfile` `{node: {term: weight}}` are keyed by user node ids and TERM ids: the
identifiers of the feature vocabulary. The estimator takes the vocabulary per feature
column (`vocabularies`, e.g. `CountVectorizerModel.vocabulary`) and translates; the
domain is implied by which vocabulary the term is in, so `gateSeedDomain` goes. Both stay
optional: a generic user has neither.

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

## What changes outside spark-vi

- **The bundle** stores the attested node ids per document (two columns when R8 is
  used: gate attestation and label-year attestation) instead of `frontier` + dense
  `label` / `labelMask`. An assembler edit: one announced cache drop. The readout
  reads the label vectors from DagHeads (R6), not from the bundle.
- **`charmpheno.mondo.native_dag`** keeps building the Mondo DAG and the code ladder
  (codes -> nodes); its support pruning moves into the estimator (R3), or stays as an
  optional pre-step if the Mondo build needs it for its own reasons (open question 3).
- **The drivers** set `labelsCol` / the DAG instead of `frontierCol` / `gateParent`, and
  read node topics from the model (R5) instead of rebuilding a `DagLayout`.

## Open questions

1. **Defaults (R10):** working configuration as defaults, or MLlib-like defaults plus a
   documented preset?
2. **Node id type:** accept mixed `str`/`int` and normalize to `str` internally, or require
   one type per DAG?
3. **Where Mondo pruning lives:** does the Mondo build need pruning before the estimator
   (e.g. the conversion sidecar's code map is built on the pruned node set), or can the
   sidecars follow the model's kept DAG?
4. **Back-compat:** none is required (pre-release). The old Params (`gateParent` JSON of
   engine ids, `frontierCol`) are removed, not aliased; existing run dirs stay readable
   through the readout's own reconstruction (it reads the saved npz + bundle meta).
5. **`frontier_from_coded` and `DagLayout` in the public `spark_vi.dag` API:** keep them
   public for power users, or make the layout an internal of the model with only the
   R5 accessors public?

## Not in scope

- Multi-domain modelling changes (pooled normalization, per-domain weighting): parked as
  a separate experiment candidate.
- The broadcast-expElogbeta and sparse-statistics performance ideas: parked.
