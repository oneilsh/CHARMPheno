---
id: 120
slug: mondo-usage-provenance-split
status: pending
model_class: mondo_usage
cohort: population_rare_priority
source_table: condition_occurrence
count_space: source_climb
min_cell: 20
mondo_version: "2026-06-02"
mondo_cache_dir: "data/mondo"
with_hpo: true
---

# 0120 — Mondo/HPO usage re-export: two-sided disclosure fix + mapping provenance

Re-runs the 0109 configuration (`source_climb`, `--with-hpo`) with the driver changes from
the 2026-11-02 design ([spec](../superpowers/specs/2026-11-02-mondo-usage-mapping-provenance-design.md),
[plan](../superpowers/plans/2026-11-02-mondo-usage-mapping-provenance.md), Part A).

## Why

1. **Phase 0 — disclosure fix.** An audit of the published 0109 payloads found that the
   fractional (1/m) count could cap a masked (≤20) source code from *above* — down to exactly
   1 on some terms — and that the exact-vs-banded choice for a total leaked through the
   choice itself. Totals are now banded by band (selection-free), and `frac` is exact only on
   terms with no ≤20 code, otherwise a 10-wide bucket whose upper end never falls below the
   public protection threshold. `assert_diff_safe` is two-sided.
2. **Mapping provenance.** Each catalogued source code now records its ladder step
   (1 same-as · 2 via SNOMED · 3 HPO · 4 climbed; lowest wins per code and term), the
   standard concept it went through, and, for climbed codes, one shortest SNOMED `Is a`
   chain to the mapped ancestor. Every term gets a disclosure-safe split into **named**
   patients (≥1 step-1–3 code) and **rolled-up-only** patients (reached only by climbing),
   published as exact / ranged / mirror / withheld.

## Output

`mondo_usage.json` + `hpo_usage.json` (+ `_nodes.tsv`) in the run dir. New payload fields:
per source code `step`, `std`, `path`, `path_ambiguous`, `multi_std`; per node `split`
and `frac_range`; top-level `concepts` (identity table for `std`/`path` ids). The run log
gains a `[paths]` line (chain reconstruction coverage) and the safe summary gains the
rolled-up-only person count and split status counts.

## Run

```bash
make -C analysis/cloud exp ID=120
# writes <run-dir>/mondo_usage.json + <run-dir>/hpo_usage.json; fit-only (no eval/bundle).
```

## Run log

_(pending first cluster run)_
