# Whole-Mondo gated-PC fit — integration plan

**Date:** 2026-08-17
**Goal:** the first real Mondo-era fit — a gated-PC (multi-domain, localized head) over the
**whole-Mondo powered DAG** (exp 0088), where detection is the top edge of one
all-conditional tree. De-risked end to end already; this is the plumbing to make a fit
*use* the Mondo hierarchy instead of the SNOMED `concept_ancestor` DAG.

## What is already done (de-risking + reusable seams)

- **Reaches the population** — exp 0087: whole-Mondo places 97.9% of coded AoU patients
  (insight 0070). Unplaced 1.18% are HPO-domain symptoms (correctly unplaced).
- **The tree** — exp 0088 (`analysis/cloud/mondo_hierarchy_cloud.py`): 9,164 anchors →
  **2,513 powered (≥100 pts) + 1,306 branch-point class nodes**, K≈3,800, clean
  body-system structure. Emits `mondo_powered_hierarchy.tsv` (a ready `parent_of`).
  Insight 0071.
- **Fittable as one co-fit** — exp 0089: the **localized head** (`DagLayout.allowed_with_
  siblings`, `localize_head` flag) ≈ dense within ~0.01–0.02 AUC, recovers the collapse.
  The dense head is an 850 GB wall at K≈3,800; localized is O(C·depth³).
- **Pre-flight cost profile** — `DagLayout.cost_report` logs fan-out / support / head
  matrix memory+compute at the build boundary (watch high-fan-out parents).
- **Seams to REUSE (do not rebuild):**
  - MONDO→OMOP mapping: `analysis/cloud/mondo_to_omop_mapping.py:build_mondo_to_omop`.
  - Mondo powered DAG: `mondo_hierarchy_cloud.py` (parent_of + powered anchor set).
  - Engine DAG: `charmpheno/omop/condition_dag.py` (`ConditionDag`, `.to_engine()` →
    `parent_int, int2cid, cid2int`) — edge-source-agnostic (`build_condition_dag(edges,
    anchor, node_ids)` takes ANY edge list; today fed `concept_ancestor` pairs at
    `case_finding_assembly.py:522`).
  - Corpus assembly: `charmpheno/omop/multi_domain.py:assemble_multidomain_case_finding_
    corpus` + `case_finding_assembly.py` (`frontier_to_label`, `attach_labels`,
    `attach_frontiers`, `frontier_from_coded`).
  - Localized head + cost profile (above); `gated_pc_cloud.py` driver; the exp harness
    (`model_class=gated_pc`, `localize_head: true`).

## What is NEW (the build)

### Piece 1 — Mondo engine DAG (branch-agnostic)
Turn the Mondo powered hierarchy into `(parent_int, int2cid, cid2int)`.
- Source: the powered anchors + class nodes + `parent_of` from `mondo_hierarchy_cloud`
  (or recompute inline from `build_mondo_to_omop` + `reduce_to_anchor_hierarchy`).
- The node id space is **Mondo/OMOP concept ids**; feed the Mondo `is-a` edges (not
  `concept_ancestor`) into `build_condition_dag` (it's edge-agnostic), or write a thin
  `mondo_dag.py` that mirrors `ConditionDag`. Prefer reusing `ConditionDag`.
- Output identical shape to today's `parent_int` → `DagLayout` just works, and
  `localize_head`/`cost_report` come for free.

### Piece 2 — per-patient Mondo frontier (the substantive new data logic)
Today a patient's attested nodes = their condition codes that are `concept_ancestor`
descendants of an anchor. For Mondo, a patient's attested Mondo nodes = **their SNOMED
codes rolled up to mapped Mondo nodes** (the SNOMED-climb): `condition ⋈ concept_ancestor
[ancestor ∈ mapped-node-set] on descendant=condition_concept_id → the Mondo nodes`, then
`frontier_from_coded` for most-specific. This is a new branch in the assembly (the
mapping-based node assignment vs the anchor-subtree assignment). Reuse the broadcast-join
pattern from `mondo_completeness_cloud.py`.

### Piece 3 — assembly integration
Add a `dag_source="mondo"` path (or a sibling assembler) to
`assemble_multidomain_case_finding_corpus`: build the Mondo DAG (Piece 1), compute Mondo
frontiers (Piece 2), then the existing label/mask/BOW/domain machinery is UNCHANGED
(features are still SNOMED condition/measurement/drug tokens). Keep `label_mask_mode`,
`extra_domains`, binary-measurement, etc. as-is.

### Piece 4 — cohort / residual decision (from insight 0070)
Who is in the corpus, and how to treat the residual:
- **coded population** = patients with ≥1 condition code (55.85%).
- **no-code (44%)** and **symptom-only (1.18%)** = the "no disease dx" residual. DECIDE:
  explicit NOS top-level class vs exclude. Recommendation (insight 0070): keep as an
  explicit top-level "no-disease" class (symptom-only is the case-finding target), not
  exclude. This affects the top-of-tree contrast (detection factor).

### Piece 5 — scale strategy: TEMPLATE BRANCH FIRST, then full
Do NOT jump to K≈3,800. Pieces 1–3 are branch-agnostic, so:
- **Step A (template):** restrict the Mondo DAG to ONE body-system branch (e.g.
  cardiovascular, ~273 nodes → K~300) and fit gated-PC + localized head. Validates the
  Mondo-DAG assembly + frontier mapping + localized head at real-but-bounded scale, at the
  K we already run comfortably. Read the `[cost]` profile + conditional readout.
- **Step B (full):** the whole K≈3,800 DAG, `localize_head`, cost-profile watch, `|w|max`
  watch (0089 saw a transient excursion), first big fit.

## Open decisions to resolve (fresh context, confirm with user)
1. Residual: NOS class vs exclude (Piece 4). Lean NOS.
2. Template branch choice for Step A (cardiovascular is the natural first — big, adult-
   powered, clinically legible).
3. `min_positives` floor (K dial: 100 → 2,513 powered; raise to shrink K if needed).
4. High-fan-out cap: if `cost_report` shows a parent with huge fan-out inflating the
   localized support (≈K), cap sibling-inclusion for that parent (abstract meta-classes
   like "disease by body system" should be pruned via `max_class_fraction` anyway).
5. `label_mask_mode`: closure (conditional/diagnosis) is the target; full for detection.
   Two-stage compose later.

## Build order
1. Piece 1 (Mondo engine DAG) + unit test (reuse `ConditionDag`).
2. Piece 2 (Mondo frontier) — BQ logic, broadcast-join; validate against exp 0087 counts.
3. Piece 3 (assembly `dag_source="mondo"`) + a cohort/registry entry.
4. Step A template fit (cardiovascular) via `make exp` — read cost profile + conditional.
5. Piece 4 residual handling (once Step A validates the mechanics).
6. Step B full whole-Mondo fit.

## Risks
- **Scale/perf**: whole-population frontier join + K≈3,800 fit; first big run. Mitigate
  with the template branch (Step A) + the cost profile.
- **|w| stability** at large C (0089 excursion); ridge caught it — watch, bump `head_l2`.
- **Mapping coverage / cohort size**: the coded population is large; the residual decision
  changes the top-level contrast.
- **High-fan-out parents** inflating localized support — the cost profile surfaces this.

## Status
De-risked; ready to build in fresh context after compaction. Start at Build order step 1;
Step A (template branch) is the first runnable milestone.
