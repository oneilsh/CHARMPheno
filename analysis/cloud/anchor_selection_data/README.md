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

---

# DisMech ribbon (`dismech_ribbon.tsv`)

The flat disease label set of spec
`docs/superpowers/specs/2026-10-07-dismech-ribbon-label-space-design.md`: one row
per DisMech disorder (`kb/disorders/*.yaml`) that carries a MONDO `disease_term`.
Loaded by `analysis/cloud/dismech_ribbon.py` and handed to the native-Mondo build
as the `label_set` the powered set is intersected with (`gated_pc_cloud --label-set`,
`label_set:` in an experiment's front matter). Subtypes are not rows; the hierarchy
lives only in the stacked readout.

## Provenance

- Source: `monarch-initiative/dismech`, `kb/disorders/`, at the commit named in the
  file's first line (`# dismech_commit: <sha>`). That sha is the pin; the bundle
  cache key folds `dismech:<sha12>:<n>:<digest12>` over the sorted Mondo ids
  (`dismech_ribbon.label_set_identity`), so re-cutting at a new commit is a new
  corpus and editing a descriptive column is not.
- Disorders without a MONDO term (poisonings, a few infections, ageing) are
  skipped and counted on stderr by the cutter. Two DisMech curations of one Mondo
  term both appear as rows and collapse to one label node.

## Columns

`mondo_id`, `disorder_name`, `category` (DisMech's coarse tag — descriptive only),
the nosology columns `harrisons_chapter`, `isds_skeletal_category`,
`icimd_category`, `iuis_category`, `icdo_morphology`, `mechanistic_category`,
`channelopathy_category`, `lysosomal_storage_category` (pipe-joined when a
disorder carries several values; §D4's candidate sets where a nosology is judged
phenotypically coherent — Harrison's and `category` are not), and `dismech_file`.

## Regenerate

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/monarch-initiative/dismech /tmp/dismech
git -C /tmp/dismech sparse-checkout set kb/disorders
python analysis/cloud/dismech_ribbon.py --kb /tmp/dismech/kb/disorders \
  --commit "$(git -C /tmp/dismech rev-parse HEAD)" \
  --out analysis/cloud/anchor_selection_data/dismech_ribbon.tsv
```
