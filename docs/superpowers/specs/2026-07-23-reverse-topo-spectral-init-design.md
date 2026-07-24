# Reverse-topological spectral init (leaves-first deflation) — design

**Status:** design (brainstormed 2026-07-23, branch `case-finding`)

## Problem

The gated spectral init (`gated_init.py`) recovers each DAG node's topic block by
anchor-word spectral recovery (Arora et al. 2013) DEFLATED against its already-
recovered closure-ancestors, in FORWARD topological order (ancestors first). So a
node's topic is "the increment it adds over its ancestors": the ancestor claims the
family-generic signal, the leaf gets the subtype-specific residual.

Hypothesis (worth an A/B): for case-finding the most-specific, most-discriminative
nodes are the leaves, and they are what placement scores on. Recovering them FIRST —
leaves-first, deflating each node against its already-recovered DESCENDANTS — would
let leaf topics claim their full defining signal and leave ancestors as genuinely
within-family background. This may sharpen the leaf topics that matter.

## Idea: a topo_order knob

Add `topo_order ∈ {"forward", "reverse"}` (default `"forward"`, so zero behavior
change) to both spectral init functions:

- **forward** (current): iterate nodes ascending depth; deflate u against
  `background ∪ proper-ancestor anchors`. Node topic = increment over ancestors.
- **reverse** (new): iterate nodes descending depth; deflate u against
  `background ∪ proper-descendant anchors`. Node topic = residual after ALL
  descendants claim their signal; leaves recover their full signal first.

The deflation set in reverse mode is ALL proper descendants (the exact mirror of
forward's all-proper-ancestors via closure), not just direct children.

Correctness of the ordering: a proper descendant v of u always has strictly greater
longest-path depth than u (u lies on a path to v, so u's longest path extends to a
longer path to v). So descending-depth order guarantees every proper descendant of u
is recovered before u — exactly as ascending-depth guarantees ancestors-first today.

`anchor_scope` (which docs train a node) is ORTHOGONAL to `topo_order` (what a node
is deflated against): the two combine freely. Only `seed_rows` and the iteration
order change; the background step, anchor recovery, floor/warn behavior, and the
scalable streaming structure are untouched.

## The math (what changes)

Notation: node_anchors[p] = the recovered anchor rows of a previously-processed node
p; bg_anchors from the background step (unchanged, always recovered first).

Forward (current):
    order      = sorted(nodes, key=(depth, id))                       # ascending
    relatives(u) = [a for a in lay.closure(u) if a not in (u, 0)]     # proper ancestors
Reverse (new):
    order      = sorted(nodes, key=(depth, id), reverse=True)         # descending
    relatives(u) = [a for a in lay.descendants(u) if a != u]          # proper descendants
                   # (0 is the root, never a descendant, so no explicit 0-drop needed;
                   #  the `a != u` guard mirrors the forward `not in (u, 0)`)

Both then build, unchanged:
    seed = list(bg_anchors) + [a for p in relatives(u) for a in node_anchors.get(p, [])]
    fg_anchors = find_anchors(Q_u, lay.tpn, seed_rows=seed)
    node_anchors[u] = list(fg_anchors)
    combined = recover_beta(Q_u, list(seed) + list(fg_anchors))
    fg_beta  = combined[len(seed):]  ->  node u's block

## Components

### Engine — `spark_vi/models/topic/dag_placement.py` (id-agnostic)

- `DagLayout.descendants(u) -> list[int]`: all nodes v with u in `self.closure(v)`
  and v != u, sorted by (depth, id). Mirror of `closure` (which returns ancestors +
  self); descendants reuses closure so there is no separate child-adjacency to keep
  in sync. Cycle-safe (closure already guards cycles).

### Engine — `spark_vi/models/topic/gated_init.py` (id-agnostic)

- `spectral_block_aligned_lambda(..., topo_order="forward")`: parametrize the node
  iteration order + the `relatives(u)` deflation set as above. A small local helper
  (order + relatives) keeps both branches DRY. `_validate_topo_order`.
- `scalable_block_aligned_lambda(..., topo_order="forward")`: identical parametrization
  (the streaming per-node pass structure is unchanged; only the loop order and
  `seed_rows` change). The dense/scalable numerical-parity contract must hold under
  reverse too.
- `TOPO_ORDERS = ("forward", "reverse")`.

### Wiring (mirrors the existing `anchor_scope` threading)

- `GatedOnlineLDA.initialize_global`: pass `topo_order` from `data_summary`
  (dense path) exactly as `anchor_scope` is passed; store on the model for the
  scalable path.
- `mllib/topic/gated_lda.py`: new Param `spectralTopoOrder` (default "forward");
  pass to `scalable_block_aligned_lambda`.
- `analysis/cloud/dag_placement_cloud.py`: `--spectral-topo-order` arg
  (choices forward|reverse, default forward), threaded to the estimator.
- `scripts/run_experiment.py`: `spectral_topo_order` manifest field ->
  `--spectral-topo-order` in the dag_placement arg builder.

### Experiment

- `docs/experiments/0069-*.md`: clone exp 0067's recipe (rare6, 1yr lookback,
  learned alpha fit + symmetric deploy, n_bg 40, frontier anchors) with
  `spectral_topo_order: reverse`. A/B vs 0067 (forward). Read placement
  mrr/auc_by_depth, LR + explain-away detection, and the error-class totals.

## Validation

- **`descendants` unit test** in `test_dag_placement.py`: on a small multi-parent
  DAG, assert descendants(u) is exactly the proper-descendant set (and disjoint from
  closure(u)\{u}); a leaf has empty descendants; the anchor's descendants = all nodes.
- **Semantic flip test** in `test_gated_init.py`: a 2-level plant — parent P with a
  private word p_word + a shared word s_word; child C with a private word c_word +
  the same s_word. Under `topo_order="forward"`, s_word's mass lands in P's block
  (P recovered first, C deflated against P); under `topo_order="reverse"`, s_word's
  mass lands in C's block (C recovered first, P deflated against C). Assert the
  block-argmax of s_word flips between the two modes. This proves the deflation
  direction actually changed the recovered topics, not just the loop order.
- **Dense/scalable parity under reverse** in `test_gated_init.py`: extend the
  existing forward parity check so `spectral_block_aligned_lambda(topo_order="reverse")`
  and `scalable_block_aligned_lambda(topo_order="reverse")` agree (same tolerance as
  the forward parity test).
- **Default-unchanged guard**: `topo_order="forward"` reproduces the current output
  bit-for-bit (the existing forward tests must stay green with the new default).

## Interactions & risks

- **Frontier anchor_scope corollary** (existing, unchanged): in `anchor_scope="frontier"`
  an internal node with no own-frontier docs stays at the floor. Under reverse order
  that node is skipped just as under forward, and a node deflated against it simply
  omits it from `seed` (node_anchors.get(p, []) -> []). No new failure mode; the
  "isolate the residual" guarantee likewise holds only for descendants that have
  their own docs — the reverse-mode mirror of the existing forward caveat. State it
  in the docstring.
- **Reverse is a hypothesis, not a presumed win.** Prototype findings showed spectral
  init did not beat random on synthetic plants (the gate breaks symmetry); reverse may
  be null too. The A/B is the point. Default stays forward.

## Out of scope

- Changing the default init (`random` stays the GatedOnlineLDA default; spectral stays
  optional).
- Direct-children-only deflation (rejected in brainstorming in favor of the
  all-proper-descendants mirror).
- A per-node / per-depth mixed order (only the two global orders in v1).
- Any change to the anchor recovery math or the background step.
