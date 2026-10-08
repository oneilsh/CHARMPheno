# ADR 0048 — The hierarchy lives in the READ: a Mondo readout DAG over a flat label-set fit

**Date:** 2026-10-08
**Status:** Accepted

## Context

The DisMech-ribbon design (spec 2026-10-07, R1) fits the gated topic model over a FLAT
forest of disorders so that no parent block can deflate a child's and no patient feeds
two nested nodes (R1a). Exp 0135 confirmed the fit-side win (DCM rescued; insight 0096)
and exposed the cost on the read: the bundle's closure mask — "observe the active closure
and its DAG siblings" — on a forest whose only parent is the root observes every node on
every foreground document. 70.7M cells, 70 s per L-BFGS iteration, and a "within-cohort"
head whose cohort is the whole foreground. The negatives the readout needs (patients with
the sibling diseases) and the ancestor heads the stacked product multiplies both come
from a hierarchy the fit must not see.

## Decision

The hierarchy is applied at READOUT time, by a driver-owned module (`readout_dag.py`),
to the SCORED frames, and nowhere else:

- readout nodes = the fit's label nodes (engine ids kept) ∪ ancestor heads appended
  after them: every Mondo ancestor grouping ≥ 2 label nodes, with root aliases (an
  ancestor over every label node) folded into the root and rungs (ancestors with the
  same label-descendant set) collapsed to the most specific;
- readout DAG = `induced_hasse_parents` over those nodes (Mondo's order, transitively
  reduced); `y_r[n]` = closure-max over `n`'s label descendants; `mask_r[n]` = a
  readout parent of `n` is active — the closure policy on the readout DAG, so a node
  is observed exactly where it or a sibling is positive;
- both columns are column arithmetic over the fit's `label` array (ADR 0047: no UDF,
  nothing array-shaped on a task closure). The fit, the bundle on disk, the cache key,
  every hashed module and the manifest's `C` are untouched; the readout node space is
  recorded in `<run>/readout_dag.json`.

`--readout-hierarchy auto` turns this on exactly for label-set fits (`corpus_manifest
.label_set` set), so every nested run's readout is byte-identical to before.

## Alternatives considered

- **Nest the fit DAG** (0127–0134): the parent block becomes the child's stratum
  (insight 0096) — the thing the ribbon exists to avoid.
- **Rebuild the bundle with a readout-specific mask**: moves the cache key, puts a
  readout policy in a hashed module, and still leaves the fit's `label` nested or flat
  but not both.
- **Per-node negatives sampled from the background**: answers the de-novo question
  (which the stacked root head already asks), not the sibling-contrast question that
  conditional diagnosis (spec §D4) needs.

## Consequences

- The record readout for a ribbon run is hierarchical; a flat read of one is tagged
  `_flat` and is the ablation.
- Per-node results for engine ids `< C_fit` are comparable across hierarchy on/off and
  across runs; ids `≥ C_fit` are ancestor heads and resolve through `readout_dag.json`.
- The co-fit head, `dag_head` and incident arms are defined in the fit's node space and
  are skipped under the hierarchy (printed, never silent). Widening the pre-index
  closure column is a follow-up if the incident read is ever wanted on a ribbon run.
- A later ablation can prune the ancestor set (spec §D1); the construction is
  parameterised by `min_descendants`.
