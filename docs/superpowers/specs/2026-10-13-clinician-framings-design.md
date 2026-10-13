# Clinician framings for WP-D: differential, next question, undiagnosed look-alikes

**Date:** 2026-10-13
**Status:** draft for review (decisions D1–D5 taken in conversation 2026-10-13)
**Supersedes:** the sequential "20 questions" acquisition loop as WP-D's evaluation target
(0138 part 3, 0139). That loop stays in the record as the ordering result; it is not the
deliverable.
**Builds on:** the ribbon record 0137; VOI tables (0138 part 1); conditional heads (0138
part 2); insights 0098 (few-code θ fold-in, one-vs-rest heads) and 0099 (measurement
result states).

## Why

The acquisition loop answered "how fast does a posterior concentrate if you ask codes one
at a time from nothing", which is not a question a clinician asks. Three questions are:

1. **Differential with reasons** — "for this patient, which disease in group G, and what
   in their record points there?"
2. **Next question** — "given everything already in the record, which further fact —
   a condition, a medication, a test result — would most change that differential?"
3. **Undiagnosed look-alikes** — "who looks like disease X but carries no X code, and
   why?"

## Decisions taken

- **D1 — no measurement binning or typical-range imputation.** The data are too messy.
  Order-only `[measured]` tokens (insight 0099: 47% of measurement mass; All of Us
  physical measurements and site-missing ranges) are **pre-filtered out** of everything
  a framing shows or asks about. Result-bearing tokens (low/normal/high, pos/neg) stay.
  Removing them from the FIT is a vocabulary filter for the next deliberate rebuild
  (source-hashed path), not part of this work.
- **D2 — residual leakage from late diagnostic coding is accepted.** Attestation codes
  (the members' own codes and the group's own code) are removed from every input record;
  beyond that there is no way to tell a late-coded diagnosis from a missing one, so the
  framings are evaluated as-is and the limitation is stated.
- **D3 — framing 2 asks about any domain**: "does the patient also have condition X?",
  "have they been prescribed Y?", "was test Z abnormal?". Treatments are allowed; a
  candidate that is near-specific to one member (a likely treatment of it) is FLAGGED,
  not hidden, so the circularity is visible to the reader.
- **D4 — the engine is chosen per framing by evaluation**, not assumed. The multiclass
  head is a candidate, not the answer.
- **D5 — profiles are subtype mixtures** (2026-10-12 discussion): a disease with several
  signature topics is a mixture of subtypes, each with its own presence probabilities,
  not one averaged profile.

## Requirements

### Shared

- **R0.1** Input record = a held-out patient's full record minus attestation codes (D2)
  minus `[measured]` tokens (D1). Evaluation patients: test split, positive on exactly
  one member of the set (the multi-positive are reported apart).
- **R0.2** Candidate sets as in 0138: Mondo groups (`^cardiomyopathy$`, `^connective
  tissue disorder$`) and the ad hoc Marfan / EDS / hEDS set.
- **R0.3** Pooled numbers only; any cell under 20 patients suppressed. Per-patient
  output (a differential, a question list) is produced only for inspection inside the
  workspace, never committed.

### Engines (evaluated, D4)

- **E1 presence (subtype mixture)** — `voi_tables` presence model with D5; likelihood
  over present AND absent codes of the record, tempered (τ fit on half, scored on half).
- **E2 renormalized stacked heads** — the 0137 heads on whole-record θ (the 0139 baseline:
  balanced top-1 0.32 / 0.22 / 0.48).
- **E3 per-group multiclass head** — softmax over the set's members on whole-record θ,
  fit on the TRAIN split's patients positive on exactly one member (readout-side, no
  refit of the topic model).
- **E4 gated per-disease likelihood** — for each member d, the record's fold-in ELBO under
  the gate {background ∪ block d}; Bayes over members. Uses co-occurrence through θ and
  the gate exactly as trained; expected overconfident on long records (calibration
  measured, not assumed).

### Framing 1 — differential with reasons

- **R1.1** Output: member probabilities and, per member, the top supporting codes from
  the patient's own record. Attribution per engine: E1 per-code log-likelihood ratio; E2/E3
  each code's topic assignment (φ) times the head's topic weight; E4 the per-code
  log-likelihood difference between gates.
- **R1.2** Metrics: top-1, top-3, balanced top-1, ECE of the max posterior, log loss.
- **R1.3** Read: which engine wins on balanced top-1 at acceptable calibration, per set.

### Framing 2 — next question

- **R2.1** For each candidate fact v not in the record (conditions, drugs, result tokens;
  attestation and `[measured]` excluded), value = the expected reduction in the
  framing-1 differential's entropy if v's status were known, under the winning engine.
  Only engines with an absence likelihood (E1, E4) can value a NO answer; with E2/E3 the
  value is P(yes) × the change on yes, and this is stated.
- **R2.2** Flag (D3): a candidate whose presence is near-specific to one member
  (presence-model P(member | v present) ≥ 0.9) is marked as a likely treatment/diagnostic
  consequence.
- **R2.3** Evaluation (positive-unlabeled aware): hide a random 20% of each patient's
  non-attestation facts; rank all candidates on the remaining record. Metrics: **yield**
  — the share of the top-k recommended facts that are among the hidden ones (they are
  known YES); **gain** — the differential's improvement (balanced top-1, P(true)) when the
  top-k hidden-and-recommended facts are revealed, against revealing k random hidden
  facts. Absent candidates are unlabeled, never scored as NO.

### Framing 3 — undiagnosed look-alikes

- **R3.1** For a member disease X: test-split patients with no code of X (and none of
  its group's attestation codes), ranked by the engine's P(X); top features per R1.1.
- **R3.2** Evaluation: the existing de-novo readout (0137 stacked de-novo AUC 0.822) is
  the ranking metric; a temporal check (later-coded X) needs the incident arm
  (`preindex_closure`, off in 0137) and is deferred to the next rebuild.
- **R3.3** Output for inspection: per X, the pooled feature profile of the top-ranked
  look-alikes (codes over-represented vs the group), never patient lists in the tree.

## Revision 2026-10-13 (after 0143-0145): match each head to its framing

0143-0145 showed the readout heads answer "X coded in the next year", trained on mixed
prevalent + incident labels on the FULL record (own codes included). That is not
leakage (train split only; test θ is folded from the test record), but it is a
shortcut (past X codes -> X topic mass -> X re-coded) and it calibrates the heads for
re-coding, not new diagnosis. Rule: **train each head on the same kind of record it
is applied to, with the broadest positive set; evaluate on the population the
question is about.**

- **Framings 1 / 2 (E3)** already match: trained and scored on records with the set's
  attestation codes removed (`filter_record`). Widening: train positives = patients
  EVER coded with exactly one member (sidecar), not only the label year.
  "Exactly one member" (R0.1) is the multiclass head's requirement of a single true
  class; patients coded with two or more members are set aside and counted. A
  multi-label variant is the open alternative for comorbid patients.
- **Framing 3 splits.** 3a *missed known cases* (an X diagnosis anywhere in the
  history the current record does not reflect) is a lookup on the sidecar, not a
  model. 3b *undiagnosed look-alikes* gets its own head per X: X's attestation codes
  (and its label descendants') removed from the record at training AND scoring, θ
  folded from the stripped record, labels = ever-X (train split); evaluated on
  patients with no X before the index, against later X diagnoses (sidecar). Replaces
  0142's after-the-fact prior_dx / undiagnosed partition.
- **Scale.** Exact 3b costs one fold-in per disease per patient — fine for the
  framing sets now, not for every node. Long-term candidate: the leave-codes-out
  approximation γ_strip(X) ≈ γ − Σ over X's tokens of count × φ_token from ONE full
  fold-in per patient (X's codes are few, so one CAVI step's worth of removal should
  be close), validated against the exact refold on the framing sets first.
- The screening number (incident macro AUC on the readout heads, 0145 rule:
  stacked 0.694) stays as the population benchmark.

## Open

- Whether E4 is affordable at K = 3,145 for every member of a 19-member set (it is a
  fold-in per member per patient; the 0139 cost was ~2.4 s per patient for far more
  fold-ins, so likely yes).
- Whether framing 2 should also rank "ask the patient" facts (symptoms) — not in OMOP.
