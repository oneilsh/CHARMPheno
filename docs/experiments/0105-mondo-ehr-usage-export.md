---
id: 105
slug: mondo-ehr-usage-export
status: pending
model_class: mondo_usage
# NOT a fit — exports the whole-Mondo EHR-USAGE report (analysis/cloud/
# mondo_usage_cloud.py): how much of the Mondo disease ontology an EHR (AoU)
# actually touches, counted by EXACT diagnosis code with NO roll-up. Distinct
# from 0088 (mondo_hierarchy), which power-counts by climbing concept_ancestor.
# Each Mondo term = its own OMOP standard Condition concept(s); its count = the
# distinct persons coded exactly at that concept. Mid-level/abstract terms keep
# their own count (may be <20 or 0 while descendants are common — a finding, not
# an artifact). Feeds mondo-usage-dashboard/.
cohort: population_rare_priority   # vestigial (validation only; driver ignores it)
source_table: condition_occurrence
min_cell: 20
mondo_version: "2026-06-02"
mondo_cache_dir: "data/mondo"
---

# 0105 — Whole-Mondo EHR usage (exact map, no roll-up)

> **Tip:** exp **0108** runs this space together with `source` (0106) and `source_climb`
> (0107) in one job (`--count-space all`) and writes a safe cross-space summary. Run 0105
> alone only for the standard space in isolation.

Answers "how much of Mondo does a real EHR use, and where?" — a standalone,
publishable output, not a modeling step. Follows 0087 (whole-Mondo places 97.9% of
coded patients via roll-up) but deliberately **drops the roll-up**: it reports each
Mondo term's *own* exact-code usage so recording granularity stays visible.

## What it does

1. **Map** whole Mondo → OMOP standard Condition anchors (faithful `mondo2omop`,
   `restrict=None`, broadcast-join scale fix) — same mapping as 0087/0088.
2. **Count, NO roll-up.** For each Mondo term, the distinct persons whose
   `condition_occurrence` standard concept EXACTLY equals one of that term's mapped
   standard concepts. No `concept_ancestor` climb. One person per term (distinct);
   comorbidities counted under each distinct term.
3. **Flag collisions.** From the mapping frame, any standard concept mapped from >1
   Mondo term (an OMOP `Maps to` convergence — the only cross-term multi-mapping;
   the `same_as` layer is source-injective, insight 0075). Each affected term keeps
   its full count and is flagged with its co-mapped siblings.
4. **Structure.** Arrange mapped terms in the Mondo is-a hierarchy via
   nearest-mapped-ancestor edges (unmapped intermediates collapsed).
5. **Suppress (three-state).** Per term: `unused` (0), `used <20` (kept & flagged,
   exact count withheld), or exact (≥20). A 0-count term that is an ancestor of a
   used term is a `used_branch` (kept as structural context); a 0-count term above
   nothing used is `other`. Only per-term floored counts are published — nothing
   can be differenced back. Term/node counts in the headline are exact (not
   patients).

Artifacts: `mondo_usage.json` (dashboard payload) + `mondo_usage_nodes.tsv`.

## What to read

- **used (≥1 patient) fraction** — how much of mapped Mondo the EHR touches at all.
- **used-small (<20)** — the long tail: terms genuinely used but below the AoU
  floor. Kept and flagged, never dropped or shown as 0.
- **used-branch** — 0-direct-count branch points sitting above real usage (the
  "used skeleton"); distinguished from the rest of Mondo.
- **collision-flagged** — the real OMOP `Maps to` multi-mapping rate on AoU.
- **internal (non-leaf) used** — mid-level terms carrying their own usage; the
  un-rolled granularity story.

## Run

```bash
cd ~/repos/CHARMPheno && git pull && make -C analysis/cloud exp ID=105
```

Copy the `WHOLE-MONDO EHR USAGE` block + `[ladder]` line back. Then load the
written `mondo_usage.json` into `mondo-usage-dashboard/` (drag-drop via "Open
data", or serve it beside `index.html`).

## Run log

_(pending first cluster run)_
