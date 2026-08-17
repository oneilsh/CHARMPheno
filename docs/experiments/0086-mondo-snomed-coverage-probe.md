---
id: 86
slug: mondo-snomed-coverage-probe
status: pending
model_class: mondo_probe
# NOT a model fit — a BQ-only placement-coverage probe (analysis/cloud/
# mondo_coverage_probe.py) that sizes the Mondo-backbone redesign: how many
# currently-"background" patients get placed on a real disease node as the target set
# expands from the 41 anchors -> the Mondo-mapped hierarchy (via the SNOMED-climb
# roll-up), and how big the truly-healthy residual is.
cohort: population_rare_priority   # vestigial (validation only; probe ignores it)
person_mod: 20                     # ~5% sample; drop to 1 for the full number
support_thresholds: "20,50,100,500"
# BYO Mondo mapping (CDR vocab has NO Mondo). EDIT this to your staged table (see
# "Stage the mapping" below). Table needs a `snomed_code` column (bare SNOMED CT SCTID
# string, from the Mondo SSSOM object_id) — the probe resolves it to OMOP concept_ids.
mondo_map_table: "REPLACE.with.your_mondo_snomed_map"
---

# 0086 — Mondo/SNOMED placement-coverage probe

Read-only. Answers, before we build anything: **does the Mondo-backbone redesign
actually rescue background patients onto real disease nodes, and what does insisting on
Mondo cost?** Produces a coverage ladder + a node-support histogram. See the use-case
discussion (detection = top-of-tree conditional; the synthetic-background problem).

## Stage the mapping (one-time, in your AoU workbench)

The CDR vocab has no Mondo, so bring the Mondo↔SNOMED mapping as a side table:

1. Grab **`mondo.sssom.tsv`** from the Mondo release
   (https://github.com/monarch-initiative/mondo/releases → `mondo.sssom.tsv`).
2. Keep rows whose `object_id` is SNOMED (e.g. `SNOMEDCT_US:xxxxxxxx`) and a real
   match predicate (`skos:exactMatch`, optionally `skos:closeMatch`). Extract the bare
   SCTID (the part after the colon) as **`snomed_code`** (string), keep `subject_id` as
   **`mondo_id`**.
3. Load that two-column table into BQ (e.g. `yourproj.yourds.mondo_snomed_map`), and set
   `mondo_map_table` in the frontmatter above to its fully-qualified name.

The probe joins `snomed_code` → `concept.concept_code` (vocabulary_id='SNOMED',
standard) to resolve OMOP concept_ids, then rolls each patient's condition codes UP via
`concept_ancestor` to any mapped ancestor (so an unmapped leaf still lands at a mapped
parent — the SNOMED-climb trick). `closeMatch` inflates coverage slightly; start with
`exactMatch` only if you want the conservative number.

> Verify once: `SNOMED_DISEASE = 4274025` (SNOMED "Disease") is the assumed
> disease-hierarchy root in the probe. Confirm with
> `SELECT concept_id, concept_name FROM concept WHERE concept_code='64572001' AND
> vocabulary_id='SNOMED'`; if your vocab pins a different concept, change the one
> constant in `mondo_coverage_probe.py`.

## Run

```bash
cd ~/repos/CHARMPheno && git pull origin claude/spectral-anchor-topic-k-200nqp && \
  make -C analysis/cloud exp ID=86
```

The report prints to the run log (teed to `summary.md`) — copy/paste the block back.

## What the ladder means

```
has >=1 standard condition code     complement = truly-healthy / no-code residual
rolls up to SNOMED Disease          PLACEABILITY CEILING (needs no Mondo at all)
under the 41 anchors                status-quo foreground (what we place today)
placed on a Mondo-mapped node       HEADLINE: proposed foreground
residual: has codes, NO Mondo map   the Mondo-incompleteness tax (your worry, quantified)
residual: truly healthy (no codes)  the irreducible background floor
node support (>=T patients)         sizes the tree / K at each min-patient threshold
```

Decision reads: **ceiling − anchors** = background patients the redesign can rescue;
**ceiling − Mondo landing** = the cost of insisting on Mondo (small ⇒ Mondo ~free;
large ⇒ lean on the SNOMED-climb or accept fall-through); **node support** ⇒ whether the
min-patient filter alone prunes to roughly your ~2,000-priority scale.

## Run log

_(pending — paste the `MONDO/SNOMED PLACEMENT COVERAGE PROBE` block)_
