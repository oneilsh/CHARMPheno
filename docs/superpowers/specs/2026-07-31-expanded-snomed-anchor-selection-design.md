# Expanded SNOMED anchor selection for a pooling-testable anchor set — design

**Date:** 2026-07-31
**Branch:** `hybrid-domain-reliability` (review branch
`claude/hybrid-domain-reliability-review-ckn2bq`)
**Status:** design for approval; first code (seed parser) landed alongside
**Follows:** insight 0075 and the 2026-07-31 dependence-aware design discussion —
the decision to expand anchors *before* adding measurement, so the model uses the
information it already has, and so a per-disease multidomain weighting mechanism
finally has enough anchors + ontology structure to learn from.

## Goal

Grow the case-finding anchor set from the six hand-picked, deliberately-distinct
rare6 diseases to ~20–30 rare-disease anchors arranged in **ontology
neighborhoods**, so we can test the open question from 0075: does per-disease
multidomain weighting benefit from **partial pooling across related diseases**?
Six distinct isolates cannot answer that — pooling needs relatives to borrow
from. rare6 is retained as a fixed reference cluster so the pooling effect can be
read against a known baseline.

This produces an *anchor set*, not the pooled estimator itself. The estimator is
a later arc; it is pointless to build until the anchors exist.

## Universe

The candidate universe is the Monarch **dismech #1079** prioritised subset (a few
hundred diseases), itself drawn from the authoritative
`prioritised-rare-disease-list.yml` (3,079 diseases). #1079 already groups
diseases into four keyword-derived categories that seed our neighborhoods:

- **Neurodevelopmental** (~309) — large; adult coding coverage is the risk.
- **Neurodegenerative** (~150) — rich sibling families (spinocerebellar ataxias,
  leukodystrophies, spinal muscular atrophies, neuronal ceroid lipofuscinoses,
  Parkinson variants); strong pooling testbed, adult-powered.
- **Neuroimmune** (~12) — compact, adult-powered, and contains rare6's myasthenia
  gravis; a natural bridge cluster.
- **Cardiac** (~139+) — very large sibling families (dilated / hypertrophic
  cardiomyopathy, familial atrial fibrillation, familial thoracic aortic
  aneurysm); another strong pooling testbed.

The #1079 grouping is keyword-based, not ontological (the issue says so). We use
the categories only as a **priority/relevance prior**; the actual pooling
neighborhoods are derived from SNOMED `is-a` structure via `concept_ancestor`,
because that is what the model's DAG pools over.

## Architecture: entirely on the cluster

No local OHDSI CSV downloads. The selection runs as a PySpark job in the existing
idiom (`charmpheno/omop/bigquery.py: load_omop_bigquery`, `concept_ancestor`
subtree expansion, Make-driven `make -C analysis/cloud …`, CDR resolved by
`setup_workspace`). It reuses the cohort machinery's positive semantics so
counts match how a fit will actually define positives.

### The one genuine external dependency: MONDO → OMOP mapping

OMOP's vocabulary tables do **not** contain the MONDO linkage, so the MONDO→SNOMED
(and →ICD10CM/OMIM/Orphanet) cross-references must be brought in. Two acceptable
ways, both keeping the heavy work on-cluster:

1. **BigQuery-native (recommended).** Stage only a small MONDO exact-xref table
   — MONDO SSSOM `skos:exactMatch` rows, or the `same_as` extract the user's
   `mondo_to_omop.py` already reads — as a table `mondo_xref(mondo_id, vocab,
   code, predicate)`. Reproduce the mapping in Spark/SQL against the CDR vocab:
   `mondo_xref.code+vocab → concept (concept_code, vocabulary_id) → concept_
   relationship 'Maps to' → standard concept`, filtered to `standard_concept='S'`
   AND `domain_id='Condition'`. Yields `mondo_id → omop_standard_condition_
   concept_id`.
2. **Reuse the user's tool.** Run `mondo_to_omop.py` once to emit `MONDO2OMOP.tsv`
   and load that as the mapping table. Simpler, but the script wants `CONCEPT.csv`
   + `CONCEPT_RELATIONSHIP.csv` locally, which cuts against the all-on-cluster
   choice.

Recommendation: option 1 — the only thing that must leave MONDO is the small
exact-xref table; everything else is CDR-native.

## Pipeline stages

1. **Seed parse** — #1079 markdown → `(category, mondo_id, label, curated)` rows.
   *(Implemented now: `analysis/cloud/anchor_selection.py`.)*
2. **Map** — seed MONDO ids → OMOP standard condition concept ids (above).
3. **Subtree expand** — each candidate anchor → its `concept_ancestor` descendant
   set (standard condition concepts), matching how rare6 anchors resolve.
4. **Power count** — distinct persons whose **first diagnosis under the anchor
   subtree** falls in-cohort with the ≥1yr lookback floor, i.e. the same positive
   definition `case_finding_assembly` uses, so counts are fit-faithful.
5. **Power filter** — keep anchors with ≥ ~100 positives (rare6's smallest usable
   was ~79–80; 100 buys stable 5×5 nested CV without sacrificing much rarity).
6. **Neighborhood assembly** — among powered candidates, group by SNOMED shared
   lowest-common-ancestor into 3–4 compact clusters (~4–6 each), plus a few
   isolates, **retaining rare6 as a fixed cluster**. Cross-check against #1079
   categories.
7. **Nesting rule (predeclared)** — no selected anchor may be an `is-a` ancestor
   of another; on a nested pair, keep the more specific one that still clears the
   floor. Prevents shared-subtree positive contamination.
8. **Node budget** — cap total DAG nodes (depth-cap a disease with a huge subtype
   taxonomy) so `K = n_bg + n_nodes × tpn` stays within ~2–3× the current fit.
9. **Freeze & attest** — persist the frozen anchor list (`concept_id`, `label`,
   `mondo_id`, `category`, `neighborhood`, `positive_count`, provenance) as a
   versioned artifact and generate a `_DISEASE_REGISTRY` entry (`"rareN"`), with
   a count-verification against the CDR (same discipline as the rare6 "VERIFY ON
   FIRST RUN" checks).

Then: fit on the expanded set and **re-run the 0075 weighting readout**, checking
specifically whether the rare6's own per-disease weights and headroom *change*
once pooled with relatives — the controlled pooling signal.

## Predeclared defaults (fix before viewing counts)

- positives floor: **100**
- target anchors: **~20–30**
- structure: **3–4 neighborhoods × ~4–6** + rare6 retained
- nesting: no anchor is an is-a ancestor of another
- node budget: cap so `K ≲ 3×` current

## What the user stages on the cluster

- the #1079 markdown (or we regenerate the seed from the YAML by the documented
  keyword rules);
- the MONDO exact-xref table (SSSOM `exactMatch`, or `mondo_to_omop.py` output);
- confirmation the CDR dataset (resolved by `setup_workspace`) carries the OMOP
  vocab tables `concept`, `concept_relationship`, `concept_ancestor`.

## Out of scope

- the pooled/hierarchical weighting estimator itself (separate arc);
- measurement / value-aware labs;
- full MONDO↔OMOP migration and multi-parent DAG semantics beyond what
  `condition_dag` already supports;
- any change to the topic-model fitting objective.

## Testing

- seed parser: unit-tested (both monarchinitiative and dismech-issue link forms,
  `[x]`/`[ ]` curation flags, multi-category diseases);
- neighborhood assembly + nesting rule: pure-logic unit tests over synthetic
  `concept_ancestor` edge maps (reusing `condition_dag` primitives);
- power-count job: validate against known rare6 counts (SLE ~6500, MG ~1100,
  amyloidosis ~1500, …) as a regression anchor before trusting new-anchor counts.
