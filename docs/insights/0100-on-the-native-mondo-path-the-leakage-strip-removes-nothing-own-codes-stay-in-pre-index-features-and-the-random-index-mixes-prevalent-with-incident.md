# 0100 — On the native-Mondo path the leakage strip removes nothing: own codes stay in the pre-index features, and the random index mixes prevalent with incident cases

**Date:** 2026-10-13
**Topic:** labels, leakage, index date, case-finding evaluation, exps 0137, 0142
**Status:** Mechanism read from code; consequence measured in 0142.

## Observation

1. **Index date.** `population_mondo_all` uses `case_finding_population_index_table` →
   `_random_event_windows`: one random event-anchored, fully observed window per person.
   Features = the 5 years before the index; label = "X coded in the year after it". (The
   first-diagnosis index of `case_finding_index_table` is the single-disease path only.)
2. **The strip is a no-op here.** `assemble_multidomain_from_events` strips
   `before_dag.nodes()` from every feature vocabulary (`strip_mode: both`). In the SNOMED-DAG
   era nodes were OMOP condition concepts, so the disease codes left the features. On the
   native-Mondo path nodes are Mondo numbers (MONDO:0005021 → 5021) while the condition
   vocab is OMOP ids: 390 node ids, 0 collisions (0137 bundle) — nothing is stripped,
   nothing unrelated is either. Patients' own attestation codes stay in their history.
3. **Consequence (0142).** Top look-alikes for common chronic diseases carry X's own code
   in their pre-index history (leiomyoma 92.5%, SLE 63%, RA 56%): previously diagnosed,
   not re-coded in the random label year. Conversely, X in history predicts X re-coded in
   the label year, so the ribbon's case-finding AUCs partly reward reading a past code —
   most for prevalent chronic diseases, least for rare ones.

## Why it matters

- Every ribbon-era (0134–0142) case-finding figure mixes prevalent and incident cases; the
  within-cohort / de-novo AUCs are optimistic for chronic diseases by an unmeasured amount.
- The random index is worth keeping (no anchoring bias between cases and background, one
  index serves every disease); the repair is in the LABEL, per node: the incident arm
  (`preindex_closure`, E1/E2 machinery) masks node X for patients with a prior X
  attestation instead of calling them negatives. "Prior" should read the whole pre-index
  history, not only the 5-year lookback.
- The native-path strip should drop the code map's mapped codes (`std_cid`s), not node
  ids, wherever own codes must not be features (a prevalent "who has X" task).
- Hashed-module work (assembler): a deliberate cache drop and refit — the top item for the
  next rebuild, ahead of the [measured] filter and the code-map gaps (HCM 40565500,
  ancestor-group codes).
