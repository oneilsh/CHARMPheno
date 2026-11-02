---
id: 122
slug: mondo-usage-aou-bin-midpoints
status: pending
model_class: mondo_usage
cohort: population_rare_priority
source_table: condition_occurrence
count_space: source_climb
min_cell: 20
mondo_version: "2026-06-02"
mondo_cache_dir: "data/mondo"
with_hpo: true
# pinned: the live purl moved to HPO 2026-09-01, which carries no SNOMED xrefs
hpo_obo_url: "http://purl.obolibrary.org/obo/hp/releases/2026-06-23/hp.obo"
---

# 0122 — Mondo/HPO usage re-export: cells at AoU bin midpoints

Same configuration and additive-cell structure as [0121](0121-mondo-usage-diagnosis-cells.md),
with cell values per the design's **Revision 3**
([spec](../superpowers/specs/2026-11-02-mondo-usage-mapping-provenance-design.md)): each
(term, source code) cell is published as the midpoint of its AoU bin (1–20 → 10, 21–40 → 30,
41–60 → 50, …). Term counts, the named / climbed-only split and roll-ups remain sums of
published cells.

## Run

```bash
make -C analysis/cloud exp ID=122
```

## Run log

_(pending)_
