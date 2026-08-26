---
id: 107
slug: mondo-ehr-usage-source-climb
status: pending
model_class: mondo_usage
# Partial-roll-up twin of 0105/0106 (analysis/cloud/mondo_usage_cloud.py with
# --count-space source_climb). A 3-tier attribution ladder that credits each condition
# to the MOST SPECIFIC mapped Mondo term it can reach, and catalogs the originating
# source code per term:
#   (1) source-exact  : condition_source_concept_id is a term's Mondo same_as code
#   (2) standard-exact : condition_concept_id (same_as -> Maps to) hits a term
#   (3) climb          : nearest mapped SNOMED ancestor of condition_concept_id via
#                        OMOP concept_ancestor; a SNOMED-distance tie is reduced to its
#                        most-specific Mondo term(s) (nested ancestors dropped), and a
#                        genuine orthogonal tie is counted in each + flagged as a collision
# concept_ancestor has subclass edges only for STANDARD (SNOMED) concepts — ICD is
# non-standard, so only the standard concept is climbed; the ICD source rides up through
# whatever standard concept OMOP assigned.
cohort: population_rare_priority   # vestigial (validation only; driver ignores it)
source_table: condition_occurrence # REQUIRED (needs condition_source_concept_id)
count_space: source_climb
min_cell: 20
mondo_version: "2026-06-02"
mondo_cache_dir: "data/mondo"
---

# 0107 — Whole-Mondo EHR usage with a partial roll-up (source_climb)

> **Tip:** exp **0108** runs this space together with `standard` (0105) and `source`
> (0106) in one job (`--count-space all`) and writes a safe cross-space summary. Run 0107
> alone only for the source_climb space in isolation.

Same report as 0105/0106, but instead of a single exact-match rule it walks an
attribution **ladder**, preferring the most specific mapped term and only rolling up
when nothing more specific is mapped.

## Why

- **0106 (source)** is the cleanest identification (ICD `same_as`, no `Maps to`
  decomposition, no collisions) but the narrowest coverage: any condition whose source
  code Mondo doesn't list is dropped.
- **0105 (standard)** recovers some of that via `Maps to`, but reintroduces the
  one-to-many decomposition that inflates terms with generic concepts (insight 0076).
- **0107 (source_climb)** keeps the source-exact match *first* (best case, SNOMED-free),
  falls back to the standard concept, and — only as a last resort — climbs the SNOMED
  hierarchy to the **nearest** mapped Mondo term. So a specific descendant code with no
  Mondo term of its own is credited to the closest ancestor that *does* have one (a
  genuine partial roll-up), rather than dropped.

The originating source code is always catalogued on the term it reached, tagged `exact`
vs `climbed`, so you can see exactly which real-world codes feed each Mondo term.

## Read (vs 0105 / 0106)

- **used % / rare-used** — expect **higher** than both 0106 (climb recovers coverage)
  and typically 0105 (source-exact preferred, plus climb).
- **`[source_climb survey]` stderr lines** — the key instrument:
  - distinct persons resolved at each tier (source-exact / standard-exact / climbed),
  - **unmatched persons by source vocabulary** — quantifies exactly what falls outside
    OMOP's SNOMED-only ancestry (ICD9CM, local vocabularies, etc.). This is the honest
    answer to "what about source codes not reachable by the climb."
- **collision-flagged terms** — reappear (unlike pure 0106): a source code with ≥2
  equally-near mapped ancestors, or a standard concept shared by several terms, counts
  in each and is flagged (orange ring + "don't sum" note).
- **Source-codes drawer section** — per term, the catalogued originating codes; `↑`
  marks the climbed ones.

Disclosure unchanged: only per-term floored patient totals + code identities (vocab +
code + exact/climbed) are published; never per-code patient counts, so nothing is
differenceable back to a suppressed cell.

## Run

```bash
make -C analysis/cloud exp ID=107
# writes <run-dir>/mondo_usage.json (+ mondo_usage_nodes.tsv); fit-only (no eval/bundle).
```

`model_class: mondo_usage` is a fit-only class in `run_experiment.py` — it dispatches
`mondo_usage_cloud.py` with the frontmatter's `count_space` / `source_table` / `min_cell`
and skips the NPMI eval + dashboard build (like `pc` / `dag_placement`). The `[source_climb
survey]` lines land in the run's `summary.md` + fit log.

Load the written `mondo_usage.json` into `mondo-usage-dashboard/` (or via Open data) and
compare side-by-side with the 0105 and 0106 exports.

## Run log

_(pending first cluster run)_
