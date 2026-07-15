# Condition DAG Builder — Design

**Date:** 2026-07-15
**Status:** Design approved (brainstorm), pre-implementation
**Piece:** 1 of 3 of the cluster driver (this piece; then cohort+frontier-label assembly; then the cloud driver + run_experiment wiring)
**Consumes:** OMOP `concept_ancestor` + `concept`
**Produces:** the multi-parent label DAG that the placement engine (`spark_vi.models.topic.dag_placement.DagLayout`) consumes.

## Goal

Build the anchor-first label DAG for hierarchical placement from OMOP `concept_ancestor`, as a
pure, offline-testable transformation. The engine is domain-agnostic (integer ids); this piece is
the domain-specific bridge from OMOP concept ids to the engine's `{child: [parents]}` map, plus the
attestation prune and the pruning ledger that let us cap the DAG *principledly* and *see what gets
chucked*.

## Findings that shaped this (from offline exploration of the real vocab)

- Anchor **201820 "Diabetes mellitus"** yields **127** standard-condition descendants — the
  diabetes **type/etiology/status taxonomy** (T1, T2, MODY, neonatal, gestational, secondary, …
  crossed with remission / control / pregnancy). 12 are multi-parent (type x status axes); max
  depth 4.
- The classic **complications** (nephropathy, retinopathy, neuropathy, diabetic CKD) are **NOT**
  is-a descendants of 201820 — they live under **442793 "Complication due to diabetes mellitus"**
  (470 descendants) and the organ hierarchies, multi-parent across organ x type x complication axes.
- **Decision (v1):** anchor on **201820** (type taxonomy). Hypothesis: complication codes ride
  along as *learned vocabulary* inside each type node's topic (they are in the patients' documents),
  so they need not be DAG nodes. Testable post-fit by inspecting a type node's top codes. The
  complication anchor (442793) and a combined type+complication forest are documented future
  options — a one-line anchor change.

## Architecture

Pure-Python over a small in-memory edge list (the anchor's descendant edges are ~140 rows, so we
extract the relevant slice — via CSV locally or a tiny BigQuery read on-cluster — then build in
plain Python; no Spark needed for this piece).

Units (new module `charmpheno/charmpheno/omop/condition_dag.py`):

- `ConditionDag` — holds `parents: {cid: [parent_cids]}` (concept-id space, root = anchor), `anchor`,
  `names: {cid: str}`. Methods: `nodes()`, `children()`, `depth(cid)` (longest path from anchor),
  `descendants(cid)`, `to_engine()`.
- `build_condition_dag(edges, anchor, node_ids, names) -> ConditionDag` — from `edges` (an iterable
  of `(ancestor_cid, descendant_cid)` at min-levels-of-separation 1) restricted to `node_ids` (the
  standard-condition descendant set incl. anchor), assemble the multi-parent parent map. Drops
  self-loops and any edge touching a non-node.
- `collapse_chains(dag) -> (ConditionDag, list[CollapseRecord])` — losslessly merge single-child
  pass-through nodes: a non-anchor node with exactly one child and (optionally) no independent role
  is spliced out, its child reattached to its parents. Returns the records of what merged (kept for
  the ledger). Lossless because a no-branch level carries no distinguishable placement information.
- `prune_by_attestation(dag, counts, min_n) -> (ConditionDag, PruneReport)` — drop every non-anchor
  node with `counts.get(cid, 0) < min_n`; reattach each dropped node's surviving descendants to the
  dropped node's parents (transitive rewire). `counts` is `{cid: n_attesting_patients}` from the
  cohort. The anchor (root) is never dropped.
- `pruning_ledger(dag_before, dag_after, counts, *, cohort_frontiers=None) -> dict` — the readout:
  `kept`, `dropped`, `kept_by_depth`, `dropped_by_depth`, `K` (= #nodes after = engine topic count
  driver), `min_count_kept`. When `cohort_frontiers` (per-patient most-specific attested node sets,
  from assembly) is supplied, also `coarsening_rate` (fraction of patients whose most-specific node
  was pruned so their frontier rolled up) and `mean_depth_drop` for those patients. Structural stats
  need only the DAG + counts; the coarsening stats need the cohort and are filled in at assembly.
- `ConditionDag.to_engine() -> (parent_int: {int: [int]}, int2cid: {int: cid}, cid2int: {cid: int})`
  — remap concept ids to contiguous engine ids with **anchor -> 0** (root) and descendants ->
  1..N in a topological (depth, cid) order. `parent_int` is exactly what `DagLayout(parent, ...)`
  consumes; `int2cid` carries names/interpretation back for `render_profile`.

## Interfaces (boundaries)

- **In:** `edges` (min-sep-1 ancestor->descendant pairs within the subtree), `node_ids` (standard
  Condition descendants of the anchor, incl. anchor), `names`, and — for pruning — `counts`
  (`{cid: patient_count}`). A thin loader (out of this module's core, tested separately) produces
  `edges`/`node_ids`/`names` from the local CSV or from BigQuery; the core is pure over those.
- **Out:** a `DagLayout`-ready `{int: [int]}` parent map + the `int<->cid` maps + the pruning ledger.
- The **attestation counts** and the **coarsening** portion of the ledger are owned by the cohort
  assembly (piece 2); this module defines the functions and consumes the counts.

## Testing

- Unit (domain-agnostic, synthetic): a tiny hand-built edge list forming a diamond — verify the
  multi-parent `parents` map, `depth` = longest path, `collapse_chains` removes a planted
  single-child chain losslessly, `prune_by_attestation` drops a sub-threshold node and rewires its
  child to the grandparent, `pruning_ledger` reports the expected kept/dropped/K, and `to_engine`
  yields anchor->0 with a contiguous, `DagLayout`-loadable map.
- Real-data smoke (skipped when the vocab CSVs are absent, e.g. CI): build from anchor 201820, assert
  ~127 nodes / >0 multi-parent / max depth 4, and that `to_engine()` output loads into `DagLayout`.
- No clinical vocabulary in the *engine*; this module is the domain layer, so concept ids/names are
  expected here — but the unit tests use synthetic integer ids only.

## Scope / deferred

- **In scope:** the pure builder + collapse + prune + ledger + engine remap, and their unit tests +
  a real-data smoke test.
- **Deferred (piece 2/3):** the BigQuery/CSV loader wiring on-cluster; the cohort + frontier-label
  assembly that supplies `counts` and the per-patient frontiers; the cloud driver + `model_class:
  dag_placement` branch in `run_experiment.py`; the complication (442793) and combined anchors; the
  fit-scale (subsample) decision.
