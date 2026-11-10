# 0052 — The DAG-gated estimators take a user DAG and labelled documents; the node set is learned at fit time

**Status:** Accepted
**Date:** 2026-11-02

## Context

Using `DagGatedLDAEstimator` meant doing its bookkeeping first: renumber the ontology
into dense engine ids (`gateParent`), reduce each document's labels to its frontier
(`frontierCol`), build dense per-node `label` / `labelMask` columns for the heads, key
seed extras by engine id and by an index into one domain's vocabulary slice, and rebuild
a `DagLayout` after the fit to find a node's topics. Pruning rare nodes and splicing
chains lived in `charmpheno.mondo` although it is generic DAG work. Spec
`docs/superpowers/specs/2026-10-30-dag-gated-user-api-design.md`.

## Decision

1. The user gives the DAG once, in their own ids (str or int, kept as given), as a dict,
   Spark DataFrame or pandas DataFrame (`setDag`), and an array column of attested node
   ids (`labelsCol`).
2. `fit` learns the node set from the data it is given (`spark_vi.dag.nodeset`):
   validate, support, prune (`minDocsPerNode`), land, reduce, splice. Engine ids and
   the frontier reduction are internal; `DagLayout` / `frontier_from_coded` leave the
   public `spark_vi.dag` surface.
3. Outputs are keyed by user ids (`dag()`, `prunedNodes()`, `nodeTopics()`,
   `describeNodes()`, `DagHeadsModel.degenerateNodes()`); seed extras (`nodeTerms`,
   `gateSeedProfile`) by user node and term ids, with `vocabularies` per feature column.
4. `DagHeadsEstimator` takes the same inputs (or a fitted topic model) and builds its
   targets itself. A `DagGatedFeaturizer` prep stage builds the inputs from term arrays
   or long tables.
5. All stages are MLlib-persistable on any Hadoop-filesystem URI (`spark_vi.io.fs`).
6. MLlib-like defaults, except `optimizeDocConcentration` (False: learned α starves deep
   blocks), plus a documented preset (`DagGatedLDAEstimator.recommended()`).
7. No train/test notion in the shims; the consuming package splits, strips and
   evaluates (`docs/architecture/LABEL_RULES.md`). No backward compatibility.

## Alternatives considered

- Keep pruning in `charmpheno.mondo` (spec option a): leaves generic work in the
  clinical package and makes the bundle depend on the threshold.
- Public `DagLayout`: exposes engine ids, which the accessors make unnecessary.

## Consequences

- Supersedes the Param surface named in ADR 0050 §3 (`gateParent`, `frontierCol`,
  `domainBounds`, `gateSeedOwnCodes`, `gateSeedDomain`) and ADR 0049's
  `gateSeedOwnCodes` (now `nodeTerms`); their designs otherwise stand.
- `charmpheno.mondo.collapse` and the Mondo build's powering are removed; the bundle
  stores attested node ids. One full cache drop (bundle and sidecar keys).
- Node sets are counted on training documents only, so they shift slightly from runs
  before this ADR; comparisons go by user node id over the shared nodes.
- Addendum (spec R14, 2026-11-10): the fitted node set is no longer always closed
  upward. `dropUnsupportedNodes` (explicit; `nodeStatistics` reports by default)
  removes nodes whose blocks are flat with the same landing and re-reduction as
  pruning, so a dropped node may have kept children; `NodeSet.unsupported` records
  them. Code reading the node set may not assume a kept node's ancestors in the
  GIVEN DAG are kept (the kept DAG itself holds only kept nodes, as before).
