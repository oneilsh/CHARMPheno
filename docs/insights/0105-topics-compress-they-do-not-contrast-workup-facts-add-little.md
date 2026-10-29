# 0105 — Topics compress, they do not contrast: workup facts add little to a within-group differential

**Date:** 2026-10-29
**Topic:** diagnostics
**Status:** Observed

**Observation.** Framing 2 recast as workup (exp 0167; spec 2026-10-28 workup-voi-design):
for incident patients of 9 same-question Mondo groups, every fact first recorded between
the index and the first member diagnosis (conditions, drugs, test results, "test done")
was ranked by expected information gain under the patient's framing-1 belief and scored
on its true answer.

- Most of a workup window repeats the pre-index record. Asking only about NEW facts
  leaves 2-188 candidates per group; the median patient has 0-3 non-default answers.
- Pooled per fact, EIG ranks with realized gain (Spearman +0.24 / +0.31 / +0.04 / +0.05
  in the four powered groups; top-5 facts realize 3-15x the average), but the gains are
  small: 0.02-0.13 nats per patient over ALL new facts, about 0.01-0.02 for the best one,
  and patient-specific ranking barely beats one list per group.
- Per-patient rank correlation is the wrong metric here: with sparse windows the realized
  answer is mostly "not recorded", which is correctly evidence against the truth on the
  most discriminating facts, so it can be strongly negative while the model is right on
  average (runs 2-3; a planted toy reproduces it).
- Likelihoods transfer train -> test (log-likelihood within 0.01-0.03): not a shift.

**Interpretation.** Topics (and the code heads built beside them) describe a disease's
neighbourhood: its care, comorbidity and consequences. Sibling diseases share most of
it; what separates them is a few specific facts (a serology, a finding), which a
compressive representation averages away and which the pre-diagnosis record rarely
holds. Framing 1 already showed raw codes beating θ for the differential (E5 > E3);
framing 2 shows the workup adds little that is new. Finding contrastive facts is a
different problem (feature selection, active learning) from topic modeling.

**Implications.** Use topics where the patient's STATE is the product: forecasting the
next year's record, retrieving similar patients (twins), new diagnoses in a disease
neighbourhood (framing 3). Not within-group differentials.

**Setting context:** exp 0167 runs 1-4 on 0163's fit (whole-Mondo leaves-first,
leaf-seeded, random init, bg200, tpn1), 9 same-question groups from 0152, incident test
patients, cross-fit, E6 / E3 beliefs, workup sidecar (year after the index, all domains).
