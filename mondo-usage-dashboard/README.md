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
currently run as experiment **0122** (`source_climb`, `--with-hpo`, HPO release pinned):

```bash
make -C analysis/cloud exp ID=122
# writes <run-dir>/mondo_usage.json + hpo_usage.json  (+ *_nodes.tsv)
```

Copy both files here (or open `mondo_usage.json` in the UI). The sample bundled in the
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

**Counting (exp 0122+).** Every count is of **diagnoses**: a participant with a given source
code, counted once however often it was recorded. Each (term, source code) pair is a *cell*,
published as the **middle of its 20-wide range**: 1–20 as 10 (shown `10*`), 21–40 as 30, 41–60
as 50, and so on. These are the same ranges the AllOfUs Data Browser uses (it shows the top of
each range), so nothing here is finer than AoU's own release, while midpoints avoid an upward
bias that would compound up the hierarchy. Everything else is a sum of published cells: a
term's count adds its codes' cells, its named / climbed-only split adds the cells of each kind,
and a subtree **roll-up** adds the cells of every term beneath it, counting each source code
once even when it reaches several of those terms. Each node shows **`direct of rolled`**; the
**Per-term / Rolled up** toggle picks which one drives node size.

**Mapping provenance (exp 0120+).** Each source code in a term's drawer is marked by the
ladder step that placed it: `≡` **same-as** (the source code is a Mondo cross-reference),
`→` **via SNOMED** (its Maps-to SNOMED concept is a mapped term), `◆` **HPO** (that SNOMED
concept is an HPO cross-reference), `↑` **climbed** (rolled up the SNOMED hierarchy to the
nearest mapped term). Colour is always paired with the glyph and a legend. Hovering a code
shows its name, step and volume; **clicking** opens its mapping path (from the payload's
`concepts` identity table) with every concept linked out (source + SNOMED to Athena, the
term to Monarch), e.g.
`ICD10CM J22 → SNOMED 195742007 → SNOMED 50417007 → SNOMED 128272009 → MONDO:0000270`
(one of several equally short chains is flagged as such). Concept names are never exported;
ICD/HCPCS names resolve from NIH Clinical Tables and SNOMED codes link to Athena.

**Count toggle (All / Exclude climbed / Climbed only).** Each term's diagnoses split into
**named** (codes that name the term: steps 1–3) and **climbed** (codes reached only by
climbing: step 4). A code reaching a term directly on some rows and by climbing on others
counts as named. The toggle recounts every term and drives sizes, colours, the hide-unused
filter, the code list and its band chips. Terms where the climbed-only share is large are
where a direct Mondo mapping is most needed.

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
  codes** that reached it, tagged with their ladder step (see *Mapping provenance*), in a
  "Source codes" drawer section; the run also emits a per-vocabulary coverage survey.
- `all` (exp 0108) — runs all three in **one job** and writes `mondo_usage_<space>.json`
  for each (plus a primary `mondo_usage.json` = source_climb), the per-space TSVs, and a
  **disclosure-safe `mondo_usage_summary.md`** — term counts + ≤-suppressed person figures,
  no CDR id — for a side-by-side comparison you can paste anywhere.

**Code multiplicity.** Each term reports how many distinct source/target ids roll into
it, by vocabulary (drawer "Codes" + the Table "Codes" column) — exact *code* counts,
never patient counts, so they're unsuppressed and can't be differenced against the
(suppressed) patient totals. Per-code patient counts are deliberately never published.
