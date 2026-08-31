# Native Mondo label space: map-and-roll labels, keep what's supported

**Date:** 2026-08-31
**Status:** Proposed (direction approved; sequencing below)
**Decision owner:** Shawn. **Scope decision already made:** LABELS move to main's
map-and-roll machinery; the FEATURE vocabulary (the BOW) stays as-is for now — the
source-vocab feature change is explicitly deferred, even though the same machinery
could serve it. Publishing/egress (banding, suppression, dashboards) is untouched.

## 1. Why: the anchor construction is a patch magnet, and both patches point here

Exp 0104's whole-Mondo readout carried 763 degenerate (constant-fallback) label heads
out of C=3,820. A week of predict-refute-diagnose (exp 0109's run log has the full
trail, with figure) decomposed them exactly:

- **1** root (structural, by design);
- **143** structural only-child class nodes — an artifact of
  `reduce_to_anchor_hierarchy`'s single-nearest-cover flattening ("terminal
  stealing"); fixed by exp 0109's splice-to-fixpoint, which removed precisely those;
- **619** SUBSUMED CATEGORY-ANCHORS — the deep one: the climb attests every powered
  ancestor of a coded concept, the anchor hierarchy nests anchors only under class
  covers (so a category-anchor sits as a SIBLING of its own specific descendants),
  and closure masking hands out negatives only via siblings — so the category is
  co-attested on every doc that fires a "sibling" and is NEVER observed as a
  negative. All-positive cell, constant head, on 619 common clinical category labels.

The proposed patch #2 (nest anchor-under-anchor) is a hand-rolled local
approximation of what Mondo's own graph does globally: in a TRANSITIVE REDUCTION of
the real ontology restricted to kept nodes, a subsumed sibling cannot exist — the
redundant sibling edge is exactly what reduction deletes. The limit of the patch
sequence is: stop reconstructing Mondo's hierarchy from covers, and use Mondo's
hierarchy. That is this plan.

## 2. What main already provides (surveyed 2026-08-31 against origin/main)

The convergence is stronger than expected — main independently built both of this
branch's DAG repairs, as tested pure functions:

- **`analysis/cloud/mondo_to_omop_mapping.py`** — the Mondo↔OMOP xref seam
  (`seed_source_xrefs`, `build_mondo_to_omop`, `_disease_child_adjacency`,
  `_descendants`). Present and BYTE-IDENTICAL on both branches: the one shared seam.
- **`analysis/cloud/mondo_usage_cloud.py`** (main-only, 1,789 lines) — the
  map-and-roll driver plus a pure core (33 unit tests in `tests/test_mondo_usage.py`):
  - the **3-tier `source_climb` ladder** (source-exact → standard-exact →
    `concept_ancestor` climb with `nearest_mapped_standard_ancestors` +
    `reduce_tie_map` keeping tie leaves), emitting an attribution frame
    `(person_id, mondo_id, origin_cid, via, k_src, k_std)` with fractional 1/m
    attribution — measured 2.1% persons unmapped vs 5.7% for standard-space;
  - **`nearest_mapped_parents(mapped_ids, parent_adj)`** — induced Hasse edges over
    kept terms, collapsing unmapped intermediates, PRESERVING multi-parenthood
    (~50% of terms have >1 parent). This is the anchor-nesting fix, done natively;
  - **`meaningful_skeleton(seed_ids, parent_adj)`** — collapse linear chains, keep
    genuine branch points. This is the 0109 splice's role, done at build time;
  - multi-parent-safe `_depths`, `dag_structures`, collision classification, rare
    flags, and the dashboard payload assembly.
- **What main does NOT do** (and this plan must add): roll SUPPORT up the hierarchy.
  `meta.rollup: false` — per-term counts are direct-tier only; roll-up happens
  client-side in the dashboard. Labels need producer-side closure support (§3).
- **Egress vs internal**: main's `min_cell=20` floor, volume bands, and
  complementary suppression are PUBLISHING rules. Model-internal label powering
  (min_positives=100 on closure support) is a different dial; the two must not be
  conflated in the port.

## 3. Design

**Attestation (the frontier).** Per person, the source_climb attribution frame's
`mondo_id` set IS the frontier — the most-specific mapped Mondo terms for their
codes (`via ∈ {source_exact, standard_exact, climbed}`; the HPO rung is out of scope
with the feature change). This replaces `powered_anchor_climb` and the OMOP-cid
attestation provider outright. Note the semantic improvement inherited for free:
`reduce_tie_map` already resolves climb ties to most-specific terms, where the old
climb attested every powered ancestor (the co-attestation half of the 619 bug).

**Labels.** Unchanged in kind: `label[c] = 1` iff c is in the is-a CLOSURE of the
frontier — but the closure now runs over Mondo's own graph restricted to kept nodes.

**Keep what's supported (powering).** Producer-side closure support: distinct
persons per node rolled through the closure (one aggregation over the attribution
frame joined to the closure — new code, small). Keep nodes with closure support ≥
`min_positives` (100, as today). This replaces terminal powering + class covers with
one uniform rule; a kept node is just "a Mondo disease with enough patients under
it," whether or not any code maps to it directly ("directly coded" becomes a node
property, not a node type).

**The label DAG.** `nearest_mapped_parents(kept_ids, parent_adj)` — the induced
multi-parent Hasse relation — then the 0109 `mondo_collapse` splice as a generic
thin-chain post-pass (it already operates on `{child: [parents]}` and preserves
multi-parenthood). Acceptance property, by construction: no kept node's sibling set
can contain its own descendant (Hasse/transitive reduction), so the 619-class trap
is structurally impossible; residual degeneracy should be `1 (root) + small
thin-chain residue`, verified by the sibling-support diagnostic (§5).

**Engine id space.** All nodes carry Mondo ids. This retires the
positive-OMOP-cid-terminal convention baked into `mondo_dag.py`,
`mondo_collapse._default_is_terminal`, `int2cid`, and the attestation matching. The
shared `charmpheno/omop/condition_dag.py` (`ConditionDag`,
`build_condition_dag(edges, anchor, node_ids, names)`) is id-agnostic already —
byte-identical on both branches — so the spine survives; the terminal test becomes
a property lookup. `int2cid` maps engine ints to Mondo curies; per-node reports and
the drift gate follow mechanically.

**Multi-parenthood, verified not assumed.** The layout and conditional readout take
parent LISTS, but the accidental tree meant real diamonds were never exercised.
Pre-flight checklist: (a) `DagLayout.closure`/`allowed_set` on diamonds (no
double-visit, no cycle assumptions); (b) closure masking's sibling expansion over
multi-parent nodes (union over all parents — `frontier_to_label` already written
that way); (c) `conditional_readout` cohorts when a child has two kept parents;
(d) `_dag_children_and_depth` (already multi-parent-safe per its tests);
(e) `mondo_collapse` rewiring (already preserves multi-parents).

**Scale expectations.** Main's shipped source_climb stats: 9,927 mapped terms,
4,822 used, 3,063 with >20 persons. Closure-rolled support at ≥100 will land C
somewhere above today's 3,677 (closure support ≥ direct support, and mid-level
Mondo terms with no direct codes now qualify) — expect C and K = n_bg + tpn·C to
grow; measure at build time and record, don't guess.

## 4. Porting plan (there is no merge)

`git merge-base` between main and this branch FAILS — unrelated histories (1,410 vs
97 commits). "Reconciliation" is a port, and the port direction is a strategic
choice deferred to Shawn; this plan only needs the MINIMAL port into this branch to
unblock the experiment, keeping full unification separate:

1. Port `mondo_usage_cloud.py`'s pure core + its 33 tests (pandas/pure-python,
   no Spark entanglement) — or import-from-a-ported-module; do NOT fork-and-edit.
2. Build the new label front-end (`mondo_native_dag.py` or similar): attribution
   frame → frontier provider; closure-support powering; kept-set Hasse via
   `nearest_mapped_parents`; splice post-pass; ConditionDag assembly on Mondo ids.
3. Thread as `dag_source: mondo_native` (or `dag_collapse`-style versioned flag) —
   the same front-matter → driver → readout-recovery → cache-key template the last
   two flags used; SNOMED and legacy-mondo keys stay byte-identical (pinned tests).
4. The experiment doc: NOTE the id collision — main and this branch both have an
   0109 (`mondo-hpo-dual-axis` vs `whole-mondo-collapsed-dag`). This experiment
   takes **0110**; the dual-0109 gets resolved (rename ours, or accept dual with a
   cross-reference) when the histories unify.

## 5. Acceptance tests and comparison protocol

- **Degeneracy gate** (the week's diagnostic becomes the acceptance test):
  `diag-sibling-support` on the new bundle must report `allpos_* ≈ 0` beyond root
  plus a small, named thin-chain residue. The 0104→0109→0110 degenerate trajectory
  (763 → 620 → ~1) is the headline structural claim.
- **Detection**: real AUC (the constant-column fix is already always-on).
- **Metrics protocol**: macro AUC/AP reported BOTH on the node set shared with
  0104/0109 (comparability) and on the full new label space (the deliverable claim
  — every formerly-constant category head that becomes genuinely scoreable counts).
- **Reproducibility**: bundle cache key folds the new flag + module source hashes,
  collapse-off/legacy keys pinned byte-identical; manifest self-sufficiency and
  readout-recovery overrides follow the existing template.
- **0104 record (0.6978/0.4845) and 0109's splice numbers are the controls.**

## 6. Non-goals (recorded so nobody "helpfully" bundles them)

- Source-vocab FEATURES (the BOW): deferred by explicit decision, despite the same
  machinery serving it. Labels first; features are their own arc with their own
  vocab-blowup and era-merge questions.
- HPO axis: later, same pattern (the ladder's HPO rung stays off).
- Publishing/egress: bands, suppression, dashboards unchanged; internal label
  powering never leaks into payload rules or vice versa.
- PC: still dead.

## 7. Sequencing

1. Exp 0109's smoke readout lands (in flight) — closes the splice's report card.
2. Port (§4 steps 1–2) — agent-buildable against this plan; review gates as usual.
3. Exp 0110 dev smoke: build the native bundle, run `diag-sibling-support` BEFORE
   any fit (the degeneracy gate is a corpus property — cheap to check first), then
   the 30-iter smoke.
4. 0110 record run; then the strategic unification decision (which trunk absorbs
   which) with real numbers in hand.
