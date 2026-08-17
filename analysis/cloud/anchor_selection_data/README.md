# Anchor-selection seed data

`priority_seed.tsv` is the frozen candidate universe for the expanded-SNOMED
anchor-selection pipeline (see
`docs/superpowers/specs/2026-07-31-expanded-snomed-anchor-selection-design.md`).
It reproduces the Monarch **dismech #1079** grouping directly from the
authoritative prioritised list, so the seed does not depend on transcribing a
rendered issue.

## Provenance

- Source: `prioritised-rare-disease-list.yml` from
  `monarch-initiative/rare-disease-identification` (branch `main`).
- Fetched: 2026-07-31.
- Source sha256: `12607c8bead03c7edc49249ffd6ee30905581c6fb0d97255cf30f66317ee8641`
  (14,208,813 bytes; 3,079 diseases). The 14 MB YAML itself is intentionally not
  vendored; this TSV is the reproducible artifact.
- Rules: the keyword categorization documented in dismech issue #1079
  ("Grouping methodology"), implemented verbatim in
  `analysis/cloud/anchor_selection.py:CATEGORY_KEYWORDS` + `categorize()`.

## Contents

One row per (disease, category). 793 rows, 760 distinct MONDO ids (32 diseases
match more than one category). Per-category counts reproduce #1079's methodology
header exactly:

| category | rows |
|---|---:|
| Neurodevelopmental | 311 |
| Cardiac | 306 |
| Neurodegenerative | 164 |
| Neuroimmune | 12 |

Columns: `mondo_id`, `label`, `category`, `prevalence_per_100k_us` (sparse — a
prior for the power filter, present for only ~100 diseases), and
`prioritization_category`.

This is the *candidate* universe. It is not yet mapped to OMOP, not yet
power-filtered, and not yet assembled into neighborhoods — those are the later
on-cluster stages.

## Regenerate

```bash
python analysis/cloud/anchor_selection.py from-yaml <prioritised-rare-disease-list.yml> \
  > analysis/cloud/anchor_selection_data/priority_seed.tsv
```
