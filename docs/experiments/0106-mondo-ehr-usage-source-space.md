---
id: 106
slug: mondo-ehr-usage-source-space
status: pending
model_class: mondo_usage
# Source-concept-space twin of exp 0105 (analysis/cloud/mondo_usage_cloud.py with
# --count-space source). Counts a Mondo term by distinct persons whose
# condition_source_concept_id EXACTLY equals one of the term's own Mondo same_as
# source codes (SNOMED/ICD10CM/MeSH) — no OMOP `Maps to`. This removes the ICD->SNOMED
# decomposition that injects generic concepts (e.g. "pregnancy finding" inflating
# peripartum cardiomyopathy in 0105) and eliminates cross-term collisions (same_as is
# source-injective). Trade-off: coverage limited to the vocabularies Mondo lists.
cohort: population_rare_priority   # vestigial (validation only; driver ignores it)
source_table: condition_occurrence # REQUIRED for source space (era has no source id)
count_space: source
min_cell: 20
mondo_version: "2026-06-02"
mondo_cache_dir: "data/mondo"
---

# 0106 — Whole-Mondo EHR usage in SOURCE-concept space (A/B vs 0105)

Same report as 0105, but counts on `condition_source_concept_id` against each term's
own Mondo `same_as` codes instead of `condition_concept_id` via `same_as -> Maps to`.

## Why

0105 (standard space) inflates terms whose ICD10CM `same_as` code's OMOP `Maps to` is
one-to-many: the ICD code decomposes into several standard concepts, some generic
(e.g. `O90.3` peripartum cardiomyopathy → also "pregnancy finding" 444094, which is
common in the EHR), so a rare disease showed 24,629 patients and got collision-flagged
against preeclampsia (insight 0076). Source space never touches `Maps to`, so:

- no decomposition → counts reflect the term's own codes (peripartum cardiomyopathy
  should drop to a rare-disease-sized number);
- **no cross-term collisions** (`same_as` is source-injective — 0 shared codes);
- it is the route to **SNOMED-license-free** deployment (structure from Mondo, tokens
  from ICD), so this run also tells us how much coverage survives ICD-only.

## Read (vs 0105)

- **used %** and **rare-used** — expect lower than 0105 (source coverage < standard,
  since patients coded only in a vocabulary Mondo doesn't list are missed).
- **collision-flagged terms** — expect ~0.
- **codes per term** (`n_codes`, by vocab) — the new multiplicity; shown in the drawer
  + Table "Codes" column.
- Spot-check peripartum cardiomyopathy (MONDO:0018920): count should fall dramatically.

Disclosure: only per-term floored patient totals + exact code COUNTS are published;
never per-code patient counts, so nothing is differenceable.

## Run

```bash
make -C analysis/cloud exp ID=106
```

Load the written `mondo_usage.json` into `mondo-usage-dashboard/` (or via Open data)
and compare side-by-side with the 0105 export.

## Run log

_(pending first cluster run)_
