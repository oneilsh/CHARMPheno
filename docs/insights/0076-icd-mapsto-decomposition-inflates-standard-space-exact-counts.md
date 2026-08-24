# 0076 — ICD `Maps to` decomposition inflates standard-space exact counts (source space fixes it)

**Date:** 2026-08-24
**Topic:** diagnostics | ops
**Status:** Observed

## Observation

In the whole-Mondo EHR-usage export (exp 0105, standard-concept space), the rare
disease **peripartum cardiomyopathy** (MONDO:0018920) showed **24,629** patients and
was collision-flagged against preeclampsia and severe pre-eclampsia. Tracing its three
standard concepts:

- `312383`, `4037495` — mapped only from peripartum cardiomyopathy (the real ones);
- `444094` "pregnancy finding" — mapped from peripartum cardiomyopathy **+ preeclampsia
  + severe pre-eclampsia**.

The term carries two `same_as` source codes: SNOMED `62377009` and ICD10CM `O90.3`.
SNOMED is already standard and maps 1:1 to itself; **ICD10CM `O90.3`'s OMOP `Maps to`
is one-to-many** and decomposes into several standard SNOMED concepts, one being the
generic context concept "pregnancy finding." Because the no-roll-up count unions all
of a term's standard concepts, the term inherits the (large) patient count of a generic
concept — inflating a rare disease and manufacturing collisions with other
pregnancy-chapter terms (e.g. `439658` is shared by HELLP / eclampsia / gestational
diabetes for the same reason).

## Interpretation

The cause is **`Maps to` cardinality, not precoordination per se**: both O90.3 and
62377009 are single precoordinated concepts, but OMOP's ICD→SNOMED standardization
splits the ICD one and attaches generic context. Generic signals to detect it:
Maps-to **fan-out** (source → ≥2 standard), **fan-in** (standard ← many unrelated Mondo
terms), and target **concept_class** (a "finding"/context vs a "disorder"). The robust
generic rule is the **anchor test** — keep the Maps-to targets corroborated by the
term's SNOMED `same_as` (or by ≥2 independent source codes); drop ICD-only singletons.

**Cleaner still: count in SOURCE space.** Matching `condition_source_concept_id` to the
term's own `same_as` source codes never invokes `Maps to`, so no decomposition, no
inflation, and **no cross-term collisions** (`same_as` is source-injective, insight
0075). Cost: coverage limited to the vocabularies Mondo lists (ICD9-only patients
missed). This is exp 0106 (`--count-space source`) and is also the path to a
SNOMED-license-free model (structure from Mondo, tokens from ICD).

## Implications

- Standard-space exact counts over-report any Mondo term whose ICD/combination source
  code decomposes; treat large counts on otherwise-rare terms with suspicion.
- Prefer source space for the *usage report*; keep standard space (and its SNOMED
  hierarchy) for the *ontology-gated modeling*, which depends on `concept_ancestor`.
- The dashboard now publishes per-term code multiplicity (exact code counts, never
  per-code patient counts) so the fan-out is visible without a differencing hazard.

**Setting context:** No fit. whole-Mondo `mondo_usage_cloud.py` on an AoU
condition_occurrence CDR (exp 0105 standard vs 0106 source), Mondo 2026-06-02.
