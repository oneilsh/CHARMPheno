---
id: 108
slug: mondo-ehr-usage-all
status: pending
model_class: mondo_usage
# ONE run, ALL three count spaces (analysis/cloud/mondo_usage_cloud.py with
# --count-space all). Shares the Mondo frames / DAG / rare structure across the three
# spaces in a single Spark session, and writes a payload per space plus a
# disclosure-safe, copy-pasteable summary:
#   mondo_usage_standard.json     (0105: condition_concept_id via same_as -> Maps to)
#   mondo_usage_source.json       (0106: condition_source_concept_id vs same_as, SNOMED-free)
#   mondo_usage_source_climb.json (0107: 3-tier partial roll-up + source-code catalog)
#   mondo_usage.json              (= source_climb, the dashboard's default fetch)
#   mondo_usage_<space>_nodes.tsv (spreadsheet-friendly, suppressed)
#   mondo_usage_summary.md        (SAFE: term counts + ≤-suppressed person figures, no CDR id)
cohort: population_rare_priority   # vestigial (validation only; driver ignores it)
source_table: condition_occurrence # REQUIRED for the source / source_climb spaces
count_space: all
min_cell: 20
mondo_version: "2026-06-02"
mondo_cache_dir: "data/mondo"
---

# 0108 — Whole-Mondo EHR usage, all count spaces in one run

Supersedes running 0105 / 0106 / 0107 separately. `--count-space all` computes the three
spaces in a single Spark session (the Mondo hierarchy, `same_as`, rare flags, and DAG
structure are built once and reused), and emits one payload per space plus a safe summary.

## Outputs

| file | what |
|---|---|
| `mondo_usage_standard.json` | standard space (0105) |
| `mondo_usage_source.json` | source space (0106) |
| `mondo_usage_source_climb.json` | partial-roll-up space (0107) |
| `mondo_usage.json` | copy of `source_climb` — the dashboard's default fetch |
| `mondo_usage_<space>_nodes.tsv` | per-space node table (suppressed) |
| `mondo_usage_summary.md` | **safe, copy-pasteable** cross-space summary |

## The safe summary

`mondo_usage_summary.md` (also echoed to the run log / `summary.md`) is built to be pasted
anywhere: it contains only **term counts** (terms are not patients) and aggregate
fractions, with **every** person-derived figure ≤-suppressed (`≤20`). It carries **no CDR /
workbench identifier** and **no per-term patient number**. It compares the three spaces
side by side (mapped / used / used % / reported / used-small / collisions / rare-used),
lists ≤-suppressed person coverage per space, and — for `source_climb` — the attribution
survey: persons by tier and the unmatched remainder by source vocabulary (the measured
SNOMED-only-ancestry gap).

## Read

- **used %** should rise standard ≲ source_climb, with source usually lowest (source-exact
  only). `source_climb` recovers coverage `source` drops via the climb.
- **collision-flagged terms**: ~0 for `source`; reappear for `standard` and `source_climb`.
- The survey's **unmatched-by-vocab** row is the honest answer to "what about codes the
  climb can't reach" (ICD9CM / local vocabularies with no SNOMED-standard ancestry).

## Run

```bash
cd ~/repos/CHARMPheno && git checkout main && git pull
make -C analysis/cloud exp ID=108
# fit-only (no NPMI eval / dashboard bundle); writes the files above into <run-dir>.
```

Load any `mondo_usage_<space>.json` into `mondo-usage-dashboard/` (or via Open data) to
browse a given space; drop `mondo_usage.json` for the default (source_climb) view.

## Run log

_(pending first cluster run)_
