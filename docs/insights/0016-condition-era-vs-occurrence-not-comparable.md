# 0016 — condition_era and condition_occurrence runs are not directly comparable
**Date:** 2026-05-12
**Topic:** doc-units | npmi | diagnostics
**Status:** Confirmed

OMOP's `condition_era` collapses contiguous `condition_occurrence` rows
into era spans via a 30-day sliding window. A patient with 50 coded
encounters for Essential hypertension over five years contributes 50
rows in `condition_occurrence` but typically only 1–5 rows in
`condition_era`. The compression ratio is roughly 5–20× for chronic
conditions and ~1× for one-time events (fractures, single-episode
infections).

This means the two loaders produce different corpora even on the same
patients:

- **Vocabulary distribution** shifts. Chronic concepts that dominated
  occurrence-corpora drop in relative frequency; acute concepts
  (relatively uncompressed) appear comparatively more prominent.
  `min_df=10` admits a different vocabulary in each.
- **Doc-length distribution** shifts (more so for patient-lifetime
  docs than patient-year docs, but both are affected).
- **NPMI reference distribution** shifts. P(w_i), P(w_j), P(w_i, w_j)
  are all computed from the held-out doc-set, so a switch in loader
  changes the metric's denominators. Per
  [0010](0010-npmi-not-comparable-across-doc-units.md), NPMI numbers
  aren't transportable across the change.

**Implications.** When comparing runs, the doc unit *and* the source
table must match for absolute NPMI numbers to be meaningful. Relative
topic ranking within a single run is fine. The `corpus_manifest`
stamps `source_table` alongside `doc_spec`, so eval driver checks
both — humans need to as well in any write-up.

A safer path for cross-loader experiments: pick one loader for all
reported comparisons and call out the choice explicitly. The case for
`condition_era` is that it's noise-reduced and event-based (better
suited to phenotype discovery); the case for `condition_occurrence`
is more granular temporal information (better suited to anything
visit-anchored).

**Setting context.** Surfaced when comparing patient-year HDP/LDA runs
on condition_era against earlier patient-lifetime LDA runs on
condition_occurrence. Both used the same vocab_size=10000, min_df=10,
person_mod=10, but the resulting corpora are not the same and NPMI
deltas should not be over-interpreted across the switch.
