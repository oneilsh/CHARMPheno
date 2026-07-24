# DAG-Native + Set-Valued Frontier Truth — Design

**Date:** 2026-07-15
**Status:** Design approved (brainstorm), pre-implementation
**Amends:** `docs/superpowers/specs/2026-07-15-anchor-first-hierarchical-case-finding-design.md`
**Extends:** the shipped engine `spark-vi/spark_vi/models/topic/dag_placement.py` (14 tests green on branch `case-finding`)

## Motivation

Real ontologies (OMOP/MONDO/SNOMED) are **multi-parent DAGs**, not trees: a node is classified along
several axes at once (e.g. "T2DM with renal complications" is both a *kind of T2DM* and a *kind of
diabetes-with-renal-complication*). The shipped engine assumes a single-parent tree — `DagLayout`
takes `{child: parent}` and cannot even *express* a diamond. It also collapses each held-out
patient to a single label via a lowest-common-ancestor (LCA) rule.

Two problems surface together:
1. **The engine can't represent a DAG.** `closure` walks a single parent chain; `depth` is a single
   path length.
2. **The single label is an eval crutch, not a truth.** The engine's OUTPUT is a graded affinity
   profile over *all* nodes. The truth should be multi-node too. LCA existed only to feed
   single-target metrics, and in a DAG the LCA is ambiguous (multiple incomparable common ancestors).

## Two decisions (approved)

### A. DAG-native `DagLayout`

- `DagLayout` accepts `parent` as `{child: parent}` OR `{child: [parents]}` (a scalar is treated as a
  one-element list — backward compatible with every existing test).
- `closure(v)` = **all ancestors of v, plus v** (the full ancestor set, not one path).
- `allowed(v)` = background ∪ blocks of every node in `closure(v)` (excluding root) — unchanged in
  spirit, now over all ancestors.
- `depth(v)` = **longest** path from root to v (`0` for root; `1 + max(depth(p) for p in parents(v))`).
  Chosen because depth reads as *specificity*: a node reachable via a long chain is genuinely
  specific. Memoized.
- `subtree(u)`, `block`, `K`, `children` are unchanged in definition (`children` is the reverse of the
  multi-parent map; a node may appear as a child of several parents).
- The diamond becomes a non-issue: X and Y with parents {P, R} simply have `closure = {X or Y, P, R,
  root}`; there is no single-parent tie to break.

### B. Set-valued frontier truth

- Held-out (and training) truth is the **frontier**: the most-specific attested nodes = attested
  nodes with **no attested descendant**.
  `frontier(C) = { c in C : no other c' in C lies in subtree(c) }`.
  - Same-path (diabetes + T2DM) → diabetes is an ancestor of T2DM → dropped → `{T2DM}`. This
    preserves the original LCA motivation (ancestor+descendant → most-specific) with no LCA needed —
    just "drop attested ancestors."
  - Comorbid (renal + eye, incomparable) → `{renal, eye}` kept as a set. No roll-up.
  - Contradictory (T1DM + T2DM, incomparable) → `{T1DM, T2DM}` kept as a set.
- **The DAG cannot distinguish comorbid from contradictory** (both are two incomparable frontier
  nodes), so the engine does **not** adjudicate structurally. It keeps the set and **instruments the
  multi-frontier rate**. Suppressing known clinical mutual-exclusions (T1 vs T2) is an **optional
  cohort-assembly input** (a curated exclusion list), never inferred from DAG shape. This keeps the
  contradiction *observable* instead of silently rolled up.

## Interface changes (engine, `dag_placement.py`)

- **`DagLayout(parent, n_bg=2, tpn=1)`** — `parent` values may be int or list; add `self.parents:
  {node: [parent ids]}`. `closure`, `depth`, `allowed` as in Decision A. Add
  `allowed_set(frontier) -> np.ndarray` = background ∪ blocks over the union of closures of the
  frontier nodes.
- **`frontier_from_coded(coded_nodes, lay) -> frozenset[int]`** — the frontier rule above. (Keep the
  existing `label_from_coded` for the single-label path; `frontier_from_coded` is the new primary.)
- **`fit_gated(train_docs, train_labels, lay, V, ...)`** — `train_labels` may now be per-doc
  **frontier sets** (iterables of node ids) OR scalars (treated as singletons). The gate masks each
  doc to `allowed_set(frontier)` = background ∪ union-of-closures over the frontier. A comorbid
  training patient thus trains *all* its attested blocks (strictly better use of data than the
  single-label gate).
- **`profile(doc, beta_hat, lay, ...)`** — unchanged (already unmasked, already returns a profile
  over all nodes).
- **`evaluate(profiles, test_labels, lay) -> dict`** — `test_labels` may be per-doc frontier sets or
  scalars. Metrics generalize:
  - `node_auc[u]`: positive for u = `frontier ∩ subtree(u) != empty` (patient attests to u or a
    descendant). Unambiguous under DAGs.
  - `auc_by_depth`: bucketed by the longest-path `depth`.
  - `mrr`: per doc, reciprocal of the **best** (smallest) rank among the true frontier nodes; mean
    over docs with a non-empty rankable frontier (root-only / empty → excluded, as today).
  - `top2`: fraction of docs whose best-ranked true frontier node is in the top 2 by affinity.
  - `mean_hops` (DAG-distance): mean over docs of the minimum hop-distance from the argmax-affinity
    node to any true frontier node (0 if the argmax is itself true).
  - `frontier_size_mean`, `multi_frontier_rate`: instrumentation — mean |frontier| and the fraction
    of docs with |frontier| > 1 (the comorbid/contradictory-ambiguity signal).
- **`identifiability_annotation(beta_hat, lay, *, tol=0.9)`** — candidate pairs generalize:
  parent↔child over **every** edge of the multi-parent map, and siblings = any two nodes sharing **at
  least one** common parent. Cross-branch (no shared parent, not in an ancestor relation) still never
  reported.
- **`render_profile(...)`** — walks the DAG from root via `children`; a multi-parent node is rendered
  **once** (dedup visited set) at first encounter, annotated so its shared status is visible. No
  double-render.

## Synthetic generator (`tests/_stm_synth.py`)

- Extend `dag_placement_corpus` (or add `dag_placement_corpus_multi`) to:
  - accept a **multi-parent** `parent` map (build a diamond: two axis-parents with shared children);
  - emit a fraction of **comorbid** patients whose truth is a frontier **set** of incomparable nodes
    (drawing tokens from the union of their frontier closures);
  - return `labels` as a list of **frozensets** (frontier per doc) and the per-node `node_codes`.

## Evaluation & validation

- Unit: multi-parent `DagLayout` (diamond) — `closure` = all ancestors, `depth` = longest path,
  `allowed_set` over a frontier; `frontier_from_coded` on same-path (→ most-specific), comorbid (→
  set), contradictory (→ set); set-valued `evaluate` positives; identifiability siblings sharing one
  of two parents; `render_profile` renders a diamond node once.
- Behavioral (plant → gated-train → profile → evaluate) on a multi-parent DAG with comorbid patients:
  per-node AUC clears loose floors; `multi_frontier_rate` is reported and non-zero; the identifiability
  annotation flags a deliberately non-separable within-structure pair.
- Domain-agnostic: engine + tests use integer ids only; no clinical vocabulary.

## Scope / deferred

- **In scope:** the engine changes + set-valued generator + tests above (a self-contained extension
  of the shipped module on `case-finding`).
- **Deferred:** true multi-*label* output (the profile already carries it; we only change TRUTH
  representation); the optional cohort-assembly mutual-exclusion list; the OMOP DAG-builder and
  cohort assembly (separate, cluster-side, consuming this engine's `(docs, frontier-labels, dag)`
  interface); `tpn > 1`; distributed SVI.

## Backward compatibility

Every change is additive/compatible: scalar labels still work (treated as singletons), single-parent
`{child: parent}` maps still work (one-element parent lists), and the 14 existing tests must stay
green. `label_from_coded` is retained; `frontier_from_coded` is the new primary truth function.
