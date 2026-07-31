# Measurement arc — value-aware lab representation — design pass

**Date:** 2026-07-31
**Branch:** `hybrid-domain-reliability` (review branch
`claude/hybrid-domain-reliability-review-ckn2bq`)
**Status:** design options + survey plan for approval — **not** an implementation
plan
**Follows:** insight 0076 (domain-weighting/pooling is a flat lever at 41
anchors; case-finding is *information*-limited, not weighting-limited; the one
place the observation domain earned weight was Long QT ← the QT-interval
*measurement*) and insight 0062 (the binding constraint is information).
**Intended reader:** the user, deciding a representation before any plan.

---

## Why this pass exists

Every prior lever on domain *combination* is exhausted: condition scoring
plateaued (0062), adding drug/observation helped only a little (0071–0074), and
supervised/pooled per-disease domain weighting is not worth building (0075–0076).
The through-line of all of it is that the model keeps running out of
**information**, not cleverness. The next information to add is the
**measurement** domain — labs — and the decision that governs whether it helps
is *how the lab values are represented*, not whether the domain is "included."

The evidence that representation is the lever, not inclusion, is already in
hand. The current observation domain is a net drag (0071), yet the *single*
disease where it earned supervised weight in the 0076 readout was **Long QT
syndrome**, via its **QT-interval measurement** — a lab whose signal lives in the
*value*, not in the bare fact that the test was ordered. A "test was measured"
presence token throws that signal away. Labs are the domain where value-aware
representation should pay off most.

## The representation question, stated precisely

The deployed pipeline tokenizes every clinical event as a bare `concept_id` and
counts occurrences — the naive-Bayes independent log-LR readout (0076,
`dag_placement.lr_placement_scores`) then sums `cnt(i,w)·log[P(w|u)/bg(w)]` over
those tokens. For conditions and drugs a bare presence/count token is a
reasonable unit of evidence (you either have the diagnosis / the prescription or
you don't). For a **measurement** it is not: "serum potassium was measured" is
almost content-free; "serum potassium = 6.8 mmol/L (**high**)" is the evidence.
The question this arc must answer is **what token a measurement should emit** so
that the existing count-based engine sees the value, not just the visit.

Four candidate representations, cheapest/most-robust first:

1. **Bare presence (baseline / current observation shape).** Token =
   `measurement_concept_id`. What we already do for observation; keep it as the
   control the value-aware variants must beat.
2. **Coded qualitative value — `value_as_concept_id`.** Token =
   `(measurement_concept_id, value_as_concept_id)`. Uses OMOP's *already-coded*
   qualitative result ("Normal" / "High" / "Low" / "Positive" / "Detected" …).
   Unit-agnostic, so it sidesteps unit-harmonization entirely — **when it is
   populated.** Coverage is the open question (the survey answers it).
3. **Range-derived abnormality — low / normal / high.** Token =
   `(measurement_concept_id, {low,normal,high})`, computed from
   `value_as_number` vs the row's `range_low` / `range_high`. This is the
   representation the user hoped for. It is unit-*robust* (the reference range is
   in the same unit as the value, so the low/normal/high call survives most unit
   chaos without conversion), needs no external reference tables, and degrades
   gracefully (rows without a range fall back to presence). Feasibility = fraction
   of numeric rows that also carry a usable range; the survey measures it.
4. **Binned continuous value.** Token = `(measurement_concept_id, bin)` where
   `bin` is a coarse quantile/z-score bucket of `value_as_number` *within a
   harmonized unit*. Retains the most information ("cool," per the user) but is
   the most exposed to unit chaos and per-lab reference drift, and needs unit
   harmonization or per-(concept,unit) empirical quantiles. Highest ceiling,
   highest noise — a later escalation only if 2/3 leave signal on the table.

These are not exclusive: the natural production shape is a **cascade** —
prefer (3) where a range exists, fall back to (2) where a coded value exists,
fall back to (1) otherwise — but which layers are worth building is an empirical
call the survey should drive. We should **not** pick now; we should measure the
data and let coverage decide.

## Step 1 (this pass): survey the OMOP measurement table — readout before design

The user is (reasonably) less familiar with the measurement domain and flagged
the well-known hazard: **poor unit harmonization**. Consistent with this repo's
readout-first methodology (measure the ceiling before building the mechanism),
the first artifact is a **privacy-safe survey** of the CDR `measurement` table
that answers exactly the questions the representation choice turns on:

- **Volume & burstiness.** rows, distinct persons, rows-per-person distribution
  (labs are drawn in correlated bursts — this also scopes the dependence-aware
  question, spec `2026-07-31-dependence-aware-evidence-design.md`, which
  measurement makes more acute).
- **Candidate vocabulary.** top measurement concepts by distinct-patient count
  (what the measurement domain's vocabulary would actually be).
- **Value availability, per concept.** fraction of rows with `value_as_number`,
  with `value_as_concept_id`, with `range_low`+`range_high`, with
  `unit_concept_id`, with `operator_concept_id`. These fractions *are* the
  feasibility of representations 2/3/4, read straight off the data.
- **Unit harmonization, per concept.** distinct `unit_concept_id` count and the
  single-most-common-unit share — quantifies how bad the unit problem is for
  each lab (high share = clean; low share = messy), i.e. how exposed
  representation 4 is.
- **Coded-value vocabulary.** top `value_as_concept_id` values and names — shows
  whether representation 2's qualitative codes are the useful
  {Normal/High/Low/Positive/…} or junk.
- **Range-abnormality feasibility & mix.** of numeric rows that also carry a
  range, the low/normal/high split — scores representation 3 directly and shows
  how rare "abnormal" is (base-rate matters for evidence weight).

**Privacy.** Every figure is an aggregate group count or a ratio over a
large denominator. The survey applies the small-cell floor (≥ the configured
minimum, default 20; ≥50 for the candidate vocabulary) *before* anything is
written or printed, per the AoU small-cell rule — nothing sub-floor reaches disk
or the paste-back digest. Runs sampled by `--person-mod` for speed (fractions are
stable under whole-person sampling).

Deliverable: `analysis/cloud/measurement_survey_cloud.py` (+ pure
`measurement_survey.py` helpers, `make measurement-survey`). Output: per-concept
TSVs under `analysis/cloud/measurement_survey_data/` plus a compact stdout digest
the user pastes back.

## Step 2 (after the survey): choose the representation, then implement

With coverage in hand we pick the cascade layers worth building and implement the
**measurement** domain end-to-end, reusing the existing multi-domain machinery:

- add `measurement` to `load_omop_bigquery` (`_SUPPORTED_SOURCE_TABLES`) and to
  `multidomain_cloud.DOMAIN_REGISTRY`, emitting the chosen value-aware token
  instead of a bare `concept_id` (the token becomes the vocabulary unit; the
  count engine is unchanged);
- **drop the observation domain** from the default fit (user-approved) and add
  measurement; **keep drug**;
- fit an experiment (the 0076 `rare_priority` anchor set is the ready-made
  testbed — Long QT is the built-in positive control) and run the α→∞ LR readout,
  judged by PR/AP and precision-at-recall (not ROC alone), against the 0076
  cond+drug baseline.

The bar is concrete: value-aware measurement should recover Long QT (QT
interval), and lift the labs-dependent diseases that were near-useless from
cond+drug+obs in 0076 (vasculitides via inflammatory markers/autoantibodies,
etc.) — the diseases whose distinguishing evidence the survey confirms lives in
measurement values.

## Open questions for the user (before Step 2, not Step 1)

- **Cascade depth.** Once coverage is known: presence-only control + which of
  {value_as_concept, range-abnormality, binned-continuous}? (Recommendation will
  follow the survey; range-abnormality is the a-priori favorite for
  robustness.)
- **Burstiness.** If rows-per-person is heavy-tailed, do we damp repeated
  same-lab draws (e.g. count caps / per-window dedup) now, or defer to the
  dependence-aware spec? Measurement makes that question live.

**This pass asks approval only for Step 1 (the survey).** Steps 2's
representation choice is deferred to the survey results by design.
