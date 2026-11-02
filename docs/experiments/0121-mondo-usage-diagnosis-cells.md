---
id: 121
slug: mondo-usage-diagnosis-cells
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

# 0121 — Mondo/HPO usage re-export: additive diagnosis cells

Same configuration as [0120](0120-mondo-usage-provenance-split.md) (`source_climb`,
`--with-hpo`, HPO pinned to 2026-06-23), with the publication model replaced per the design's
**Revision 2** ([spec](../superpowers/specs/2026-11-02-mondo-usage-mapping-provenance-design.md)).

## What changed

- The unit is a **diagnosis**: a (patient, source code) pair, counted once.
- Each per-(term, source code) **cell** is published exactly above 20 and as **10** for 1–20.
- Every aggregate is a sum of published cells: the term count, the named / climbed-only split, and
  subtree roll-ups (each distinct source code counted once per subtree).
- Removed: per-code volume bands, banded totals, the ranged/withheld split, the roll-up ±.
- Mapping provenance (steps, `std`, paths, `concepts`) is unchanged.

## Output

`mondo_usage.json` + `hpo_usage.json` (+ `_nodes.tsv`) in the run dir. Per source code: `step`,
`n` (published cell), `std`/`path`/`path_ambiguous`/`multi_std`. Per node: `count` (Σ cells),
`imputed` (number of 10s), `split` `{named, climbed}`, `roll` `{all, named, climbed, imputed}`.
Top-level `concepts`.

## Run

```bash
make -C analysis/cloud exp ID=121
```

## Run log

_(pending)_
