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

Views: **Hierarchy** (collapsible Mondo is-a tree), **Treemap** (reported terms by
patient volume, grouped by body system), **Table** (sortable/searchable). Click any
term for its MONDO/OMOP links, parents/children, and multi-mapping (collision) detail.

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
