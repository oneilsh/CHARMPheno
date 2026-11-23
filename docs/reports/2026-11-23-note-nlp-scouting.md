# Scouting: the All of Us `note_nlp` release as a condition source

**Date:** 2026-11-23. **Data:** CDR `C2025Q4R6` (v9) and the off-cycle dataset
`C_V9_R6_offcycle_nlp` (`note`, `note_nlp`, two `_ext` tables). **Tool:**
`analysis/cloud/note_nlp_survey.py` (plain survey, then `--context`); full reports stay in
the workspace run dir (`note_nlp_survey_2026-11-23_v2.md`, `note_nlp_context_2026-11-23.md`).
Pooled figures only below; per-concept tables are workspace-internal.

**Recommendation: do not use `note_nlp` as a condition source in this release.** Only
negation can be filtered; family history, screening, hedged and past mentions cannot,
and the suspicious concepts show they are common.

## What the release holds

- One extractor (CLAMP 1.7.6), every extracted concept in the Condition domain: about
  97M rows over 9.6M notes and 99.8k patients (13% of the CDR's 747k; 99% of them also
  have coded conditions). About 87% of rows are from 2018-2024; likely a subset of sites.
- 72% of rows carry a standard concept, 26% map to one through `Maps to`, 1.7% do not
  map. After mapping, only 2.6% of affirmed rows carry a concept absent from
  `condition_occurrence`: the vocabularies are compatible.
- 25.8% of mentions are negated (`term_exists` is 'true' / 'false', not Y / N).
- Per patient, affirmed mentions give about 1.4x the distinct condition concepts of the
  same patients' `condition_occurrence` (mean 95 vs 69). Exact (patient, concept) overlap
  is low both ways (23% of NLP pairs are coded, 32% of coded pairs are in the notes): the
  sources are complementary. The gains are mostly signs and symptoms (the kind of
  feature HPO profiles are built from); chronic billable conditions are better coded.

## Why it cannot be cleaned

- `term_temporal` and `term_modifiers` are empty on every row, and `snippet` and
  `lexical_variant` are empty too (the matched text is not released), so no keyword
  filter is possible: every flag matched zero rows.
- `section_concept_id` holds small integers 1, 2, 3, ... whose counts fall steadily with
  the number: a section's ordinal position in the note, not its type. Joined to
  `concept` they read as the OMOP metadata concepts 1-33 ("Domain", "Gender", "Race"),
  which is a coincidence of ids. There is no Family History section to drop.
- The signal of contamination: colon carcinoma is affirmed in the notes of roughly a
  third of note patients but coded for a few dozen; influenza, "Disease", "Infectious
  disease", "Traumatic injury" and "Discomfort" are similar. Family history, screening,
  vaccination and templated review-of-systems text fit that pattern, and nothing in the
  release separates them.

## What would change the answer

A release with `term_modifiers` populated (CLAMP emits experiencer / certainty /
temporal attributes), or real section types, or the snippet. Worth asking the All of Us
support desk whether the off-cycle NLP data drops them by design. If a later release
has them, rerun `note_nlp_survey.py --context`: it reports what a family / screening /
hedge / history filter removes per concept.
