# 0039 — Expanded anchor selection is a count-band + ontology filter, not hand-defined neighborhoods

**Status:** Accepted
**Date:** 2026-07-31

## Context

To test whether per-disease multidomain weighting benefits from partial pooling
(the follow-up to insight 0075), the case-finding anchor set must grow beyond the
six hand-picked rare6 diseases to a larger rare-disease set that contains
ontologically *related* diseases — pooling can only be exercised when anchors
have relatives to borrow strength from.

Candidates are drawn from Monarch's dismech #1079 prioritised subset
(`analysis/cloud/anchor_selection_data/priority_seed.tsv`, reproduced from the
authoritative prioritised list), mapped to OMOP standard Condition concepts by a
faithful port of `monarch-initiative/mondo2omop`, and counted against the CDR
(`analysis/cloud/anchor_selection_cloud.py`).

The first cluster run exposed three facts that shape the decision:

1. **The #1079 keyword categories are not coherent neighborhoods.** The "Cardiac"
   bucket swept in systemic vasculitides (GPA, MPA, Takayasu, Behçet, Churg-Strauss,
   temporal arteritis), pregnancy disorders, migraine, and cerebrovascular
   disease — the keyword-grouping limitation the issue itself flags.
2. **Exact MONDO→SNOMED mapping can land a rare disease on an over-general
   standard concept.** A rare MONDO term "Maps to" a generic SNOMED concept, so
   the OMOP anchor is not rare: e.g. "Disorder of pregnancy" (27,241 patients),
   "Migraine with aura" (13,457).
3. **The thin categories are CDR-coverage-limited, not floor-limited.** In an
   adult cohort, neurodevelopmental rare diseases are essentially absent, and
   neuroimmune / neurodegenerative plateau at 6 / 5 regardless of the floor.

## Decision

**The anchor set is selected by objective count-band + ontology criteria; it is
not organized into hand-defined neighborhoods.** Pooling structure is read from
the Mondo/SNOMED is-a DAG at model time, so a "neighborhood" categorical would be
redundant (and, from #1079's keywords, noisy). Related clusters are a property of
the selected set (the mapped set already contains a vasculitis cluster, a
neuroimmune cluster, a neurodegenerative cluster, and rare6's autoimmune cluster),
discovered from the ontology rather than tagged by hand. Any within- vs
across-cluster reporting is derived post-hoc from the DAG.

Selection criteria, all predeclared:

- **floor** — ≥ 50 distinct persons with an in-subtree `condition_occurrence`
  (the power proxy; the fit's exact positive set additionally applies the
  first-dx index + ≥1yr lookback, which only shrinks counts).
- **ceiling** — ≤ 10,000 persons, a rare-band upper bound that removes
  over-general mappings (fact 2). This is a mapping-quality / rarity guard, not a
  strict prevalence definition, and it operates on coded-patient counts, not true
  prevalence. On the current data 10,000 is equivalent to 5,000 (no anchor lies
  between 4,629 and 13,457); 10,000 is chosen as the more future-permissive
  default.
- **nesting** — keep the most-specific anchors: drop any anchor that is an is-a
  ancestor of another clearing anchor (`maximal_anchors`).
- **rare6 pinning** — the six rare6 anchors are always retained, so the nesting
  rule cannot discard a broad rare6 anchor in favor of a narrow relative, and the
  established reference isolates stay in the fit. Pinning is the *only* inclusion
  override; ontological relatives of a rare6 anchor (e.g. cardiac sarcoidosis ⊂
  sarcoidosis) are **not** pruned — both may remain, and the DAG represents their
  relationship for pooling to use.

**No hand exclusions.** Selection is fully objective: band + nesting + rare6
pinning. Over-broad concepts that clear the ceiling (e.g. "Congenital heart
disease", "Disorder characterized by eosinophilia") are kept — pruning them would
require exactly the per-disease judgment that does not scale, and as extra
anchors they do not compromise the pooling test (they act as isolates or
loose relatives; interesting signal, if any, is retained). The freeze tool
exposes an optional `--exclude` escape hatch, but it is empty for this set.
Neurodevelopmental is absent at the floor in an adult cohort and so contributes
nothing — an objective outcome, not a hand exclusion.

## Consequences

- Selection is reproducible and auditable from the frozen candidate table + the
  is-a edges under fully objective criteria (band + nesting + rare6 pin) — no
  per-disease exclusion policy, so the same rules apply unchanged at anchor scale.
- The set spans several ontology clusters (vasculitis+rare6 autoimmune,
  neuroimmune, neurodegenerative, cardiac-genetic) plus isolates, which is
  exactly the structure the pooling experiment needs; the DAG supplies the
  relatedness.
- The prevalence ceiling can drop a genuinely rare disease whose only SNOMED
  mapping is an over-general concept. That is accepted: such an anchor could not
  be scored cleanly anyway.
- Counts here are coded-patient power proxies; only counts ≥ floor are ever
  reported, keeping disclosure above small-cell thresholds.
- If a future domain or a larger anchor sweep makes explicit grouping useful, it
  should still be derived from the ontology, not reintroduced as hand labels.

## References

- `docs/superpowers/specs/2026-07-31-expanded-snomed-anchor-selection-design.md`
- `docs/insights/0075-hybrid-domain-weighting-headroom-is-small-and-not-identified-by-lambda-reliability.md`
- ADR 0038 (supervised-readout identity attestation) — the counts here feed the
  same nested-CV readout and inherit its one-row-per-person contract.
