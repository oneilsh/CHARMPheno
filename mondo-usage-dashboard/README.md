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

- **Served (recommended):** `python3 -m http.server` in this directory, then visit
  `http://localhost:8000/`. It loads `mondo_usage.json` if present, else the sample.
- **Directly:** open `index.html` in a browser; it auto-loads the bundled
  `mondo_usage.sample.json` when the browser allows local fetches.

Drop a real `mondo_usage.json` (see below) next to `index.html` as `mondo_usage.json` —
nothing to rebuild. The page runs full-width; the graph fills the viewport height with
the detail panel docked beside it.

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

Every mapped Mondo term is browsable (all four usage kinds are always shown):

| kind | meaning |
|---|---|
| **Reported (>20)** | exact patient count, publishable |
| **Used ≤20** | genuinely used but at or below the AoU small-cell floor — kept & flagged, exact count withheld, never shown as 0 |
| **Used branch (0 count)** | a term with 0 *direct* usage that sits **above** a used term — a branch point on the "used skeleton" |
| **Rest of Mondo** | mapped terms with 0 usage and nothing used beneath them |

The dashboard is a single **DAG browser** — a true node-link view of the Mondo
disease graph (~50% of terms have >1 parent), horizontal and progressively expanded,
nodes sized by prevalence, with a **permanent detail panel** docked beside it (below it
on a phone) that fills in when you select a node. The run's provenance (Mondo version,
CDR, count space, floor, generated) sits in that panel's footer.

**Edges = reachability, not just adjacency.** Visibility (what's on screen) and edges
(how they connect) are separate concerns. Which nodes show is driven purely by your
expansion actions; the edges are then *derived* every render as the **transitive
reduction of full-DAG reachability restricted to the shown nodes**. So *if two shown
nodes are reachable in the full graph, there is always a path between them on screen* —
nothing ever floats, and collapsing an intermediate never disconnects anything. An edge
between a real parent/child renders **solid full-strength**; an edge that stands in for
a path through *hidden* nodes renders **solid but dimmer/thinner** ("a quiet shortcut").

**Interactions:** plain **click** = select + expand one level (the view stays put);
**shift-click** = add/remove from a **multi-selection** (highlight = the union of every
selected node's ancestors + descendants, rest dimmed); **hover** peeks a node's lineage
without committing. **Drag a node up or down** to reorder it within its column (overrides
the default size ordering; cleared by Reset or a mode switch). A **menu-dot** (on hover)
or **right-click** opens a per-node menu: *Expand children / parents*, *Expand all
descendants / ancestors* (opens the entire subtree; each fan-out still paginates with
**＋N more** per column), *Collapse descendants / ancestors* (fold the subtree / ancestor
chain out of view — this removes the targeted nodes **even if they're reachable via
another open parent**, and is fully reversible: re-expanding brings them back, no Reset
needed), and *Focus here* (re-root the view on this node). There is no "hide" — every
fold is undone by expanding. Toolbar: **Reset / Collapse all / Fit** (icon
buttons), **Node size** (Per-term / Rolled up), search, the **dim** slider, and a
**Hide unused** checkbox (**on by default**) that drops every fully-unused term (category *other* = 0
patients **and** no used descendant, so its whole subtree is dead), leaving just the used
skeleton — reported, used-small, and the *used_branch* nodes that connect them to root.
A **dim slider** in the graph's bottom-left corner sets how faint the non-highlighted
nodes go when a selection or hover is active (five steps, from nearly hidden to no
dimming). A **＋** badge marks nodes with children not yet shown (it fades out once
terminal or fully expanded); clicking empty canvas clears the selection; **Reset view**
clears hidden/reordered/focus state. Wide fan-outs show the top few
and bottom few by prevalence with **＋N more** between (reveals 10 at a time). Typing in
the search box **marks the matching text** in labels. Nodes are
colored **by rare-disease designation** — rare
terms in magenta, everything else neutral — so rare diagnoses pop as you drill in.
Rarity is taken from Mondo's dedicated rare-disease subsets (**GARD / Orphanet / NORD**)
only; Mondo also carries broader "rare" subsets from DOID/NCIt that over-include common
diseases (they flag e.g. prostate cancer), so those are deliberately excluded.
A ✦ also marks **rare-disease** terms.
The detail drawer lists each term's source codes and its subtree roll-up.

Each node shows **`direct of rolled (±unc)`** — e.g. `20k of 664k ±123`: the term's own
exact patient count, then its subtree roll-up. The roll-up spans the term's **full DAG
descendant closure** (every term reachable below it, each counted once even if it has
several parents — so a multi-parent child like *rheumatoid vasculitis* contributes to
every ancestor's roll-up). It is a **midpoint estimate**
(each `≤20` term put at ≈ 10) with a **±** that allows **±10 per suppressed term** → `±10·s`
(a round stand-in for the strict `±9.5` half-range of the 1–20 unknowns). It is deliberately
*not* a √s statistical band — that would assume independence and understate the uncertainty.
Because a patient with several of a subtree's diagnoses is summed several
times, the rolled number is a **count of diagnoses, not distinct patients** (the drawer
gives the honest bracket `floor–ceiling` and "at least N distinct patients"). The
**Per-term / Rolled up** toggle picks which of the two numbers **drives node size and is
bolded** (per-term → the direct count; rolled up → the roll-up); both are always shown.

## Method (why no roll-up)

Each Mondo term is counted by the distinct persons coded **exactly** at its own
OMOP standard Condition concept(s) — no `concept_ancestor` climb. Real EHR diagnoses
are recorded at different granularities; rolling up would hide that a mid-level term
is used little while its descendants are common. A person is counted once per term;
comorbidities appear under each distinct term (summing across terms double-counts —
the export flags terms that share a standard concept via OMOP `Maps to`). Counts of
1–20 render as `≤20` (AoU rule, floor inclusive); only per-term floored counts are published, so
nothing can be differenced back to a suppressed cell. See `docs/insights/0075` and
`docs/experiments/0105-mondo-ehr-usage-export.md`.

**Count space.** The driver counts in one of three spaces (`--count-space`, shown in the
meta strip):

- `standard` (default) — `condition_concept_id` via `same_as → Maps to`.
- `source` (exp 0106) — `condition_source_concept_id` against the term's own Mondo
  `same_as` codes. Avoids OMOP's ICD→SNOMED `Maps to` decomposition — which otherwise
  injects generic concepts (e.g. "pregnancy finding" inflating peripartum cardiomyopathy,
  `docs/insights/0076`) and manufactures cross-term collisions — at the cost of coverage
  limited to the vocabularies Mondo lists. Also the route to a SNOMED-license-free model.
- `source_climb` (exp 0107) — a **partial roll-up** ladder: credit each condition to the
  most specific mapped term reachable — source-exact, else standard-exact, else climb
  `concept_ancestor` (SNOMED-only) to the nearest mapped ancestor (ties → counted in each,
  flagged). Recovers coverage `source` drops. Each term catalogs the **originating source
  codes** that reached it, tagged exact vs climbed (`↑`), in a "Source codes" drawer
  section; the run also emits a per-vocabulary coverage survey.

**Code multiplicity.** Each term reports how many distinct source/target ids roll into
it, by vocabulary (drawer "Codes" + the Table "Codes" column) — exact *code* counts,
never patient counts, so they're unsuppressed and can't be differenced against the
(suppressed) patient totals. Per-code patient counts are deliberately never published.
