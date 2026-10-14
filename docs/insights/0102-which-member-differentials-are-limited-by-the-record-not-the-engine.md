# 0102 — Which-member differentials are limited by the record, not the engine

**Setting:** 0137 fit; framings 1 and 2 (spec 2026-10-13) on the connective tissue (19
members), cardiomyopathy (8) and Marfan/EDS (3) sets; records with the members' own
codes, name-matched concepts and `[measured]` removed; 57–272 scored test patients,
482–1,852 train (exps 0140, 0141, 0148).

## Observation

- Six engines (presence subtypes, renormalized stacked heads, θ multiclass, gated
  likelihood, codes multiclass, θ + codes) span balanced top-1 0.14–0.18 on connective
  tissue and 0.14–0.30 on cardiomyopathy; the best (θ + codes) beats θ alone by ~0.02.
- Unlike case-finding (insight 0101: codes +0.04 on new diagnoses), θ and codes are about
  equal here; θ + codes is best everywhere on log loss.
- The FULL record (nothing hidden) gives P(true) 0.29–0.43 under every engine.
- Asking five questions recovers ~20% of the 20%-visible-to-full gap, and patient-specific
  VOI equals a population table under every engine.

## Why it matters

- More engine work on these sets buys ~0.02. The limit is how little a pre-index EHR
  record says about WHICH member a patient has once the diagnosis codes are removed.
- Framing 2's deliverable is the population VOI table per group; a per-patient
  question list adds nothing measurable at this signal level.
- Levers that could matter: more training patients per member (ever-coded, not only
  label-year), the multi-positive patients now set aside, and features outside the
  current record (procedures, notes, genetic testing results).
