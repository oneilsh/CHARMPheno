# 0075 — Mondo↔OMOP exact-map is source-injective; multi-mapping is an OMOP `Maps to` artifact

**Date:** 2026-08-21
**Topic:** diagnostics | ops
**Status:** Observed

## Observation

Measured directly on the Mondo `2026-06-02` release (the version the pipeline
pins), replicating the `mondo2omop` disease filters exactly. Universe: 22,918
disease terms under `human disease` (minus susceptibility / characteristic /
injury / obsolete).

The `same_as` exact-map is **injective on the source-code side**: across SNOMED,
MeSH, and ICD10CM, **zero** external codes are shared by more than one Mondo term
(max Mondo-terms-per-code = 1 in all three vocabularies). The only fanout is
benign and tiny on the Mondo→code direction — mean 1.01 codes/term, max 4 SNOMED;
just 97 terms carry >1 SNOMED code.

Coverage: 8,774 terms carry a SNOMED `same_as`, 7,737 MeSH, 1,992 ICD10CM, 12,239
in ≥1 of the three. **37% of SNOMED-mapped terms are internal (non-leaf)** Mondo
nodes (3,244 / 8,774), and **99.6% of mapped terms sit under another mapped
ancestor** — i.e. the ontology carries diagnosis codes at many granularities, and
a term and its mapped parent are independent measurements once roll-up is dropped.

## Interpretation

The multi-mapping worry — "one code lands in more than one Mondo bucket" — does
**not** originate at the exact-map layer. It can only enter via OMOP's `Maps to`
step (source SNOMED → standard concept), in two forms: (a) one Mondo term → several
standard concepts (benign: union the persons, count once per term); (b) several
Mondo terms whose distinct source codes normalize to the **same** standard concept
(the real cross-term collision — identical patients attributed to multiple terms).
Case (b) is measurable from the mapping frame alone
(`standard_concept_id → #distinct mondo_id`), so it can be quantified and flagged
without any patient data. Separately, the SNOMED "travel-up" multi-landing exists
**only** under a `concept_ancestor` roll-up; the no-roll-up usage count
(`mondo_usage_cloud.py`) removes it entirely.

## Implications

- A whole-Mondo EHR-usage report can count each term by its **own** exact standard
  concept(s) with no roll-up and no source-side collision; the only collisions to
  disclose are OMOP `Maps to` convergences, surfaced per-term as a flag (each term
  keeps its full count; consumers must not sum across flagged terms).
- Re-measure per release: this is a property of `2026-06-02`, not a Mondo
  invariant. `analysis/cloud/mondo_usage_cloud.py` recomputes the collision set on
  every run.

**Setting context:** No fit. Pure characterization of the Mondo `2026-06-02`
release TSVs (`mondo_nodes.tsv`, `mondo_edges.tsv`) under the `mondo2omop` disease
filters, plus the design of the exp 0105 usage export. The OMOP `Maps to`
collision rate itself is BQ-only and reported by the driver on the AoU CDR.
