---
id: 109
slug: mondo-hpo-dual-axis
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

# 0109 — Mondo + HPO dual-axis usage (source_climb, --with-hpo)

Runs the 0107 `source_climb` attribution ladder for the Mondo axis exactly as before, and
additionally turns on the driver's `--with-hpo` flag so a second, independent HPO
phenotype axis is built in the same job.

## Why

0107's HPO probe (`survey.hpo`) only *sizes* the phenotype gap — how many SNOMED concepts
Mondo climbs or drops that HPO could match exactly. This experiment goes one step further
and actually emits the HPO axis as a usable export, so the dashboard can offer it
alongside the Mondo axis rather than only reporting the gap.

## Routing rung

`--with-hpo` adds `t_hpo` as an independent exact-match rung over the HPO DAG. Attribution
priority across axes, most specific first:

1. **Mondo-exact** (source-exact or standard-exact — same as 0105/0106/0107)
2. **HPO-exact** (`t_hpo`; an EHR SNOMED concept that Mondo can't map exactly, but HPO has
   a term for)
3. **Mondo-climb** (nearest mapped SNOMED ancestor, the source_climb fallback)

Nothing about the Mondo attribution path is modified — this is purely a routing/reporting
knob. A run without `with_hpo: true` reproduces the prior single-axis 0107 output
byte-for-byte.

## Output

Writes both `mondo_usage.json` (the usual source_climb export) and `hpo_usage.json` (the
HPO-axis export) into the run dir. Building the HPO axis downloads `hp.obo` (default purl
latest, cached under `mondo_cache_dir`) on first use.

## Run

```bash
make -C analysis/cloud exp ID=109
# writes <run-dir>/mondo_usage.json + <run-dir>/hpo_usage.json; fit-only (no eval/bundle).
```

## Run log

_(pending first cluster run)_
