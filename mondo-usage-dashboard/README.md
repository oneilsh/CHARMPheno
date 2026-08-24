# Mondo EHR-usage dashboard

A **standalone**, single-file dashboard for *"how much of the Mondo disease
ontology does an EHR (All of Us) actually touch?"* — the whole-Mondo, exact-map,
**no-roll-up** usage report. It is intentionally **not** connected to the topic-model
dashboard in [`../dashboard/`](../dashboard/): no shared build, no shared data.

```
mondo-usage-dashboard/
├── index.html               ← the whole app (self-contained; only Google Fonts is external)
└── mondo_usage.sample.json  ← real Mondo hierarchy + SYNTHETIC counts, for offline preview
```

## Viewing it

- **Quickest:** open `index.html` in a browser and click **Open data** to load a
  `mondo_usage.json`. (Opening the file directly also auto-loads the bundled
  `mondo_usage.sample.json` when the browser allows local fetches.)
- **Served (recommended):** `python3 -m http.server` in this directory, then visit
  `http://localhost:8000/`. It loads `mondo_usage.json` if present, else the sample.

Drop a real `mondo_usage.json` (see below) next to `index.html` as `mondo_usage.json`,
or load it at runtime with **Open data** — nothing to rebuild.

## Getting real data

The payload is produced on the AoU cluster (BigQuery + Spark) by
[`analysis/cloud/mondo_usage_cloud.py`](../analysis/cloud/mondo_usage_cloud.py),
run as experiment **0105**:

```bash
make -C analysis/cloud exp ID=105
# writes <run-dir>/mondo_usage.json  (+ mondo_usage_nodes.tsv)
```

Copy that `mondo_usage.json` here (or open it in the UI). The sample bundled in the
repo uses the **real Mondo 2026-06-02 hierarchy** for three body systems
(cardiovascular, respiratory, hematologic) with **synthetic, seeded** patient
counts — it is labeled `SAMPLE DATA` in the header and contains no patient data.

## What it shows

Four usage categories, each independently show/hide-able:

| category | meaning |
|---|---|
| **Reported (≥20)** | exact patient count, publishable |
| **Used <20** | genuinely used but below the AoU small-cell floor — kept & flagged, exact count withheld, never shown as 0 |
| **Used branch (0 count)** | a term with 0 *direct* usage that sits **above** a used term — a branch point on the "used skeleton" (kept because no-roll-up leaves abstract ancestors at 0) |
| **Rest of Mondo** | mapped terms with 0 usage and nothing used beneath them (off by default) |

The dashboard is a single **DAG browser** — a true node-link view of the Mondo
disease graph (the only view that shows Mondo's multi-parent edges — ~50% of terms
have >1 parent), horizontal and progressively expanded, nodes sized by prevalence.

**Interactions:** click to expand + select and pan-to-center; selecting a node reveals
and highlights its **ancestor lineage** (accent) and **descendants** (blue), dimming
the rest; **hover** peeks the same without committing; **shift-click** (or the ⌖
button) **focuses/re-roots** on a node; **✕** (on hover) hides a node/branch to
declutter (Reset restores); clicking empty canvas clears the selection. Wide fan-outs
show the top few and bottom few by prevalence with **＋N more** between (reveals 10 at
a time). Typing in the search box **marks the matching text** in labels. Nodes are
colored **by top-level Mondo class (≈ body system)** — a colorblind-safe Okabe-Ito
palette, toggle to color by usage — so drilling into one class reads as a coherent hue
family. A ✦ marks **rare-disease** terms (Mondo's GARD/Orphanet/NORD designations).
The detail drawer lists each term's source codes and its subtree roll-up. ("system"
here is the node's top-level ancestor in the mapped-term tree, not an ontology field.)

A **Per-term / Rolled up** toggle switches the count each node shows: per-term = the
term's own exact patient count; rolled up = the subtree's **upper number** (`≤`), the
sum of its used terms (each `<20` cell at 19). Because a patient with several of a
subtree's diagnoses is summed several times, the rolled number is best read as a
**count of diagnoses, not distinct patients** (the drawer states the floor — "at least
N distinct patients" — alongside it). Node size tracks whichever value is shown.

## Method (why no roll-up)

Each Mondo term is counted by the distinct persons coded **exactly** at its own
OMOP standard Condition concept(s) — no `concept_ancestor` climb. Real EHR diagnoses
are recorded at different granularities; rolling up would hide that a mid-level term
is used little while its descendants are common. A person is counted once per term;
comorbidities appear under each distinct term (summing across terms double-counts —
the export flags terms that share a standard concept via OMOP `Maps to`). Counts of
1–19 render as `<20` (AoU rule); only per-term floored counts are published, so
nothing can be differenced back to a suppressed cell. See `docs/insights/0075` and
`docs/experiments/0105-mondo-ehr-usage-export.md`.

**Count space.** The driver counts in `standard` space (`condition_concept_id` via
`same_as → Maps to`, default) or `source` space (`condition_source_concept_id` against
the term's own Mondo `same_as` codes, `--count-space source`, exp 0106). Source space
avoids OMOP's ICD→SNOMED `Maps to` decomposition — which otherwise injects generic
concepts (e.g. "pregnancy finding" inflating peripartum cardiomyopathy) and manufactures
cross-term collisions — at the cost of coverage limited to the vocabularies Mondo lists.
It's also the route to a SNOMED-license-free model (structure from Mondo, tokens from
ICD). See `docs/insights/0076`. The meta strip shows the active count space.

**Code multiplicity.** Each term reports how many distinct source/target ids roll into
it, by vocabulary (drawer "Codes" + the Table "Codes" column) — exact *code* counts,
never patient counts, so they're unsuppressed and can't be differenced against the
(suppressed) patient totals. Per-code patient counts are deliberately never published.
