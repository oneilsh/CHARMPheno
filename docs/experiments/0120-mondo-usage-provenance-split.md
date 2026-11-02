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
# pinned: the live purl moved to HPO 2026-09-01, which carries no SNOMED xrefs
hpo_obo_url: "http://purl.obolibrary.org/obo/hp/releases/2026-06-23/hp.obo"
---

# 0120 — Mondo/HPO usage re-export: two-sided disclosure fix + mapping provenance

Re-runs the 0109 configuration (`source_climb`, `--with-hpo`) with the driver changes from
the 2026-11-02 design ([spec](../superpowers/specs/2026-11-02-mondo-usage-mapping-provenance-design.md),
[plan](../superpowers/plans/2026-11-02-mondo-usage-mapping-provenance.md), Part A).

## Why

1. **Phase 0 — disclosure fix.** An audit of the published 0109 payloads found that the
   fractional (1/m) count could cap a masked (≤20) source code from *above* — down to exactly
   1 on some terms — and that the exact-vs-banded choice for a total leaked through the
   choice itself. Review then showed frac also bounds the patients holding 2+ of a term's
   codes (frac − N). **frac is no longer computed or published**; the dashboard's roll-up
   becomes the sum of published per-term counts (patient-term diagnoses, zero new
   disclosure). Totals are banded by band (selection-free) on a finer 1-2-5 ladder
   (21–50, 51–100, 101–200, …): on the 0109 aggregates 66 terms band instead of 587.
   `assert_diff_safe` checks both directions and asserts frac's absence.
2. **Mapping provenance.** Each catalogued source code now records its ladder step
   (1 same-as · 2 via SNOMED · 3 HPO · 4 climbed; lowest wins per code and term), the
   standard concept it went through, and, for climbed codes, one shortest SNOMED `Is a`
   chain to the mapped ancestor. Every term gets a disclosure-safe split into **named**
   patients (≥1 step-1–3 code) and **rolled-up-only** patients (reached only by climbing),
   published as exact / ranged / mirror / withheld.

## Output

`mondo_usage.json` + `hpo_usage.json` (+ `_nodes.tsv`) in the run dir. New payload fields:
per source code `step`, `std`, `path`, `path_ambiguous`, `multi_std`; per node `split`
(no `frac`); top-level `concepts` (identity table for `std`/`path` ids). The run log
gains a `[paths]` line (chain reconstruction coverage) and the safe summary gains the
rolled-up-only person count and split status counts.

## Run

```bash
make -C analysis/cloud exp ID=120
# writes <run-dir>/mondo_usage.json + <run-dir>/hpo_usage.json; fit-only (no eval/bundle).
```

## Run log

- **2026-11-02, run 1 — HPO axis empty; superseded.** The driver fetched the live
  `hp.obo` purl, which now serves HPO **2026-09-01**: that release carries **no SNOMED
  cross-references** (654 xrefs total, mostly Fyler/NCIT/MedDRA; v2026-06-23 has 3,439
  SNOMED-xref'd terms). The HPO axis came out with 0 terms and HPO claimed nothing, so
  Mondo's climb tier absorbed the phenotype codes (persons climbed 329,296 vs 322,454 in
  0109). The Mondo-side features otherwise validated: tripwire passes on the downloaded
  payload; 46,856 / 46,873 climbed catalog entries got a reconstructed SNOMED chain
  (7,071 one-of-several); split statuses exact 5,662 · mirror 2,589 · ranged 419 ·
  withheld 1,257; no frac. Fixes: pin `hpo_obo_url` (above); cache `hp.obo` by URL; fail
  loudly when `--with-hpo` loads 0 SNOMED xrefs; record `meta.hpo_version`. Also trimmed
  redundant catalog fields (`via`, `std` when `path` present): 19.0 -> ~17.1 MB.
- **run 2** — _(pending)_
