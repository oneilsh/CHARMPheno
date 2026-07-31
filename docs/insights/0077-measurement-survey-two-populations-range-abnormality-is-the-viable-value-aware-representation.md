# 0077 — OMOP measurement survey: the table is two populations (rangeless vitals vs range-carrying labs); range-derived low/normal/high is the viable value-aware representation for the diagnostically rich labs, coded values cover serologies, and the data is extremely bursty

**Date:** 2026-07-31
**Topic:** measurement | representation | labs | survey | burstiness | decision
**Status:** Confirmed on the CDR measurement table (survey, person_mod=100, floor=50)

Measurement arc, Step 1 (spec
`2026-07-31-measurement-arc-value-aware-representation-design.md`): a
privacy-safe survey of the OMOP `measurement` table, run to pick the value-aware
representation from data rather than a priori. It answered the question cleanly.

**Setting.** Sampled 1/100 persons: 4.76M measurement rows, 5,157 persons, 7,719
distinct measurement concepts. Table-wide coverage: `value_as_number` 0.90,
`value_as_concept_id` 0.09, `range_low`+`range_high` 0.41, `unit_concept_id`
0.69, `operator` 0.38. Range-derived abnormality is feasible on 0.40 of all rows;
of feasible rows the split is low 0.09 / normal 0.69 / high 0.23 (≈32% abnormal —
a healthy signal base rate).

## The table is two populations, not one

The top concepts by patient count split sharply:

1. **AoU program vitals / anthropometrics** (the most universal — Body height,
   weight, BMI, systolic/diastolic BP, heart rate, hip/waist circumference, plus
   the "Computed …" / "PhenX" / protocol variants). Numeric value ≈1.00, but
   reference range ≈0 and coded value ≈0.03. Units are clean (top-unit-share
   often 1.00). These are near-universal, so as *bare presence* tokens they are
   pure background; their information lives only in the numeric value → they need
   **binning**, and binning is unit-safe here because the units are clean.

2. **Standard chemistry / hematology / LFT panels** (Chloride, Creatinine, BUN,
   Glucose, Na/K/Ca, Hemoglobin, CBC indices, AST/ALT/ALP, bilirubin, …), each in
   ~half the cohort (~2,300–2,800 persons). Numeric ≈0.98, **range 0.69–0.79**,
   units clean (1–3 distinct, top-unit-share 0.85–0.92). These are tagged
   **range-abnormality** and are exactly the diagnostically rich labs.

Across the top 200 concepts the representation mix (concepts / summed
patient-count) is: range-abnormality 91 / 121,902; numeric-needs-binning 59 /
100,720; value-concept 35 / 29,245; presence-only 15 / 18,057.

## Findings that decide the representation

- **Unit harmonization is better than feared — for the labs that matter.** The
  standard chem/heme panel carries 1–3 units at 0.85–0.92 single-unit share.
  Worst offenders are Erythrocytes (9 units, 0.62) and BMI (7, 0.49). Crucially,
  **range-derived low/normal/high is unit-robust regardless**: the reference
  range travels in the value's own unit, so the abnormality call survives unit
  chaos with no conversion. This is the a-priori favorite (design spec option 3)
  and the data confirms it is both viable (0.70+ coverage on the rich labs) and
  the safest.
- **`value_as_concept_id` is thin (0.09) but covers a *different, useful* test
  class.** Its vocabulary is qualitative serology / urinalysis / micro results:
  Negative, Positive, Not detected, Reactive/Nonreactive, Trace, 1+, Clear,
  Yellow, Normal/Abnormal, High/Low. It is the natural (unit-free) home for
  autoantibody / infectious / urinalysis signals that never carry a numeric range
  — but it is polluted with junk codes (Null, "=", "0", bare numerics like "16")
  and near-constant codes ("Normal heart rate" on ~all persons), so it needs an
  **allowlist** of meaningful codes.
- **The data is extremely bursty.** ≈924 measurement rows per person (sampled);
  labs repeat (serial inpatient panels). Under the α→∞ LR readout
  (`s = Σ cnt·log[P(w|u)/bg(w)]`), raw repeat counts of "normal potassium" would
  swamp the sum. This makes the deferred dependence-aware concern
  (spec `2026-07-31-dependence-aware-evidence-design.md`) **live and concrete**
  for measurement: the token count must be damped — default to **per-patient,
  per-window binary presence** of each `(lab, state)`, not raw occurrence counts.

## Implication / decision

The representation is a **cascade**, chosen per row by what the row populates,
emitted as a per-patient-per-window **binary** token to kill burstiness:

1. **range present → `(concept, {low|normal|high})`** — range-derived
   abnormality. Covers the 91 diagnostically rich labs; unit-robust. **Build
   first.**
2. **else coded value in allowlist → `(concept, coded_value)`** — serologies /
   urinalysis. Unit-free. **Build with v1** (small allowlist).
3. **else numeric present → `(concept, quantile_bin)`** — binned vitals /
   rangeless numerics via per-`(concept,unit)` empirical population quantiles.
   **Defer to v2** (needs the quantile machinery; vitals are near-universal
   background, though BMI/HR/height may carry rare-disease signal — Marfan
   stature, POTS tachycardia).
4. **else → `(concept)`** presence fallback.

Long QT's QT-interval (the one place observation earned weight in 0076) is a
numeric lab and lands in layer 1 or 3 — the built-in positive control for Step 2.

**Next.** Confirm the v1 cascade depth + burstiness knob with the user, then Step
2: add a value-aware `measurement` source table to `load_omop_bigquery` +
`multidomain_cloud.DOMAIN_REGISTRY`, drop observation (user-approved), keep drug,
fit the 0076 `rare_priority` testbed, and run the α→∞ LR readout (judged by
PR/AP and precision-at-recall) against the 0076 cond+drug baseline.
