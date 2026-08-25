# Mondo DAG browser — reachability-based visualization redesign

**Date:** 2026-08-24
**Scope:** `mondo-usage-dashboard/index.html` (self-contained standalone dashboard)
**Status:** Approved (brainstorm), building directly.

## Problem

The DAG explorer unfolds a multi-parent ontology with tree-style expand/collapse.
Because a node can have several parents, the current rules feel unruly:

- **Visibility** is a global reachability property (a node shows if *any* parent above
  it is expanded), but **expand/collapse** is a node-local toggle. In a tree these
  coincide; in a DAG they diverge.
- **Edges** are drawn only between *directly adjacent* visible nodes, so collapsing an
  intermediate node visually orphans downstream nodes that are still reachable in the
  full graph.

User's governing requirement: *if two shown nodes are reachable in the full DAG, there
must be a path between them in the shown graph.* Plus a cleaner interaction model
(menus instead of +/−/✕ icons; multi-select; isolate).

## Design

### 1. Two independent layers

- **Visibility** — which nodes are on screen. Driven purely by *expansion state*.
  Today's downward "open" set is joined by an **upward** open set so ancestors can be
  expanded/collapsed symmetrically. Root is always shown.
- **Edges** — a *derived* view of reachability among the visible nodes. Never stored;
  recomputed every render.

### 2. Edge engine — reachability + transitive reduction

Each render, over the shown set `S`:

- For each shown node `A`, `shownDesc(A) = descendantSet(A) ∩ S` (full-DAG descendants
  intersected with what's on screen).
- The **reduced successors** of `A` are the reachability-minimal members of
  `shownDesc(A)`: `{ B ∈ shownDesc(A) : ∄ C ∈ shownDesc(A), C≠B, B ∈ shownDesc(C) }`.
  This is the transitive reduction of full-DAG reachability restricted to `S`, so
  reachability among shown nodes is preserved exactly with a minimal edge set.
- Draw an edge `A→B` for each reduced successor. **Solid/full-strength** when `A` is a
  real DAG parent of `B` (`A ∈ B.parents`); **solid but dimmer/thinner** when the edge
  summarizes a path through hidden nodes.

Consequences: every shown node always has a path to root (nothing floats); collapsing
never disconnects — a solid chain becomes a quiet shortcut. Cost is trivial (|S| ~ tens).

### 3. Visibility model

- `graphOpen` (existing) — nodes whose **children** are shown (downward).
- `graphOpenUp` (new) — nodes whose **parents** are shown (upward).
- A node is visible iff: it is root/focus; OR a DAG-child of a visible node in
  `graphOpen`; OR a DAG-parent of a visible node in `graphOpenUp`; subject to the
  category filter, `graphHidden` prune set, and per-parent fan-out cap.
- Visibility BFS runs both directions to a fixpoint.

### 4. Per-node menu (hover menu-dot + right-click, same menu)

- *Expand children* / *Expand parents* — `graphOpen.add` / `graphOpenUp.add`, one level.
- *Expand all descendants* / *Expand all ancestors* — add the full closure, subject to
  the node budget in §6.
- *Collapse children* / *Collapse parents* — remove from `graphOpen` / `graphOpenUp`
  and recurse; multi-parent survivors stay (kept connected by dimmer edges).
- *Isolate* — set focus/re-root to this node (replaces shift-click focus).
- *Hide* — `graphHidden.add` (replaces the ✕ control).

### 5. Selection & isolate

- **Plain click** — select + expand one level (if it has unshown children) + open drawer.
- **Shift-click** — toggle the node in a multi-selection Set; highlight = **union of
  each selected node's ancestors ∪ descendants**, rest dimmed.
- **Isolate selection** — toolbar button, visible when ≥1 selected; re-roots to the
  selected nodes plus the paths connecting them (their induced reachability subgraph).
- Empty-canvas click clears selection.

### 6. Guards

- *Expand all descendants* respects a node budget (~150): expand breadth-first until the
  budget is hit, then surface a "reached limit — N not shown" notice. Prevents swamping
  on large subtrees (e.g. digestive).
- Per-parent fan-out keeps the existing top-few/bottom-few "＋N more" behavior.

### 7. Removed / changed

- +/− expand badges and the ✕ icon → folded into the menu. Nodes keep a subtle marker
  when they still have unshown relations.
- Shift-click-to-focus → now multi-select; focus is the menu's *Isolate*.

## Testing

Pure-logic helpers get unit tests in a small JS-independent harness or by mirroring the
reduction logic; interaction verified with headless-Chrome screenshots (both modes,
menu open, multi-select highlight, isolate) as in prior iterations. Key invariants to
check: reachability preservation (every shown non-root node reaches root), transitive
reduction minimality, collapse-survivor rule, isolate induced-set correctness.
