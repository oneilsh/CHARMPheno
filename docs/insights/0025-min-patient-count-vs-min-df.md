# 0025 — `min_patient_count` and `min_df` answer different questions

**Status:** implemented (2026-05-28)
**Related:** ADR 0018 (doc-unit abstraction), insight 0011 (min-doc-length tradeoffs)

## Observation

`CountVectorizer.minDF` is a *document-frequency* threshold. With `PatientDocSpec`
(one doc per patient), this coincides with a *per-patient* threshold. With
`PatientYearDocSpec` (one doc per patient-year), they diverge: a code seen in
5 distinct patients across 4 years each has document-frequency 20 but
patient-frequency 5.

This matters for two distinct concerns:

- **Modeling stability:** rare tokens (low total occurrence) produce noisy
  estimates. The natural threshold here is some count of *occurrences*.
- **Privacy:** disclosure risk scales with the number of *distinct patients*
  whose data must combine for a token to appear in artifacts. The natural
  threshold here is a count of *distinct person_ids*.

Conflating the two ("just bump min_df") gives a guarantee in
`PatientDocSpec` mode that silently degrades in `PatientYearDocSpec` mode.

## Resolution

`to_bow_dataframe` exposes both `min_df` and `min_patient_count` as
independent parameters. The fit-path vocab construction now computes both
counts in one Spark aggregation and AND-composes the filters. Defaults
in cloud fit drivers are `--min-df 20 --min-patient-count 20`, which gives:

- Equivalent behavior to the prior `--min-df 20` in `PatientDocSpec` mode.
- A genuine per-patient guarantee in `PatientYearDocSpec` mode.

## Why pre-fit, not post-filter

Two implementation shapes were considered:

- **Pre-fit (chosen):** compute per-token `(term_count, doc_count, patient_count)`
  in one Spark aggregation; filter; construct `CountVectorizerModel` directly
  from the eligible vocab list.
- **Post-filter:** let `CountVectorizer.fit()` build a vocab, then prune
  entries failing `min_patient_count`, then re-emit a `CountVectorizerModel`.

Pre-fit is one Spark pass over the data (just for stats) instead of two
(stats + fit) and puts both thresholds on the same footing instead of
treating one as "what PySpark provides" and the other as a CHARMPheno
patch.
