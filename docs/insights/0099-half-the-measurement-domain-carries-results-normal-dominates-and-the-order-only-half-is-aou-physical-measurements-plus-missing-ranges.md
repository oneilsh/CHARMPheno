# 0099 — Half the measurement domain carries results; "normal" dominates, and the order-only half is All of Us physical measurements plus missing reference ranges

**Date:** 2026-10-13
**Topic:** measurement, value-aware tokens, data-quality, voi, exp 0137
**Status:** Observed on the ribbon record 0137's vocabulary (5,000 measurement tokens
over 2,916 tests; `inspect_topics --measurement-states`; mass = fitted token mass, λ
above its floor, summed over topics).

## Observation

| group | vocab share | mass share |
|---|--:|--:|
| order only (`[measured]`) | 47.4% | 47.0% |
| range result (low / normal / high) | 32.4% | 44.2% |
| coded result (pos / neg / ...) | 20.3% | 8.8% |

1. **Results are real and mostly "normal".** By mass: normal 32.5%, high 6.7%, low 5.0%,
   coded neg 6.7%, pos 1.2%. A `[normal]` or `[neg]` token is a tested-negative — the one
   kind of absence evidence that is trustworthy in this positive-unlabeled data.
2. **The order-only half has two sources.**
   * **All of Us enrollment physical measurements** lead it (blood pressure panels,
     "mean of 2nd and 3rd measures", waist/hip circumference, PhenX protocols, heart
     rate, BMI percentile; ~0.4% of mass each). They carry values but no reference
     range, so the cascade falls to presence — and nearly every participant has them,
     so the token says "enrolled", not anything clinical. The blood-pressure VALUE (the
     hypertension evidence) is lost.
   * **Missing ranges on common labs**: even the top chemistry/CBC tests are 19–26%
     order-only (glucose, hemoglobin, creatinine, sodium ...; RDW 36%; body height 85%).
     The same test splits into result tokens and a `[measured]` token depending on
     whether the site recorded a range — so `[measured]` partly encodes SITE.
3. **51% of tests have at least one result-bearing token; 33% appear both ways.**

## Why it matters

- Corrects a claim made in the 0138/0139 discussion ("labs encode orders, not results"):
  they encode results where a reference range or an allowlisted coded value exists. VOI
  table names truncated the `[state]` suffix until 62791e93, which hid this.
- "Next best workup" (a clinician-facing VOI) is feasible for range-bearing tests;
  `[measured]` tokens should be read as weak, partly site-driven evidence.
- Two cheap tokenizer fixes are visible (source-hashed modules — a deliberate cache
  drop): clinical-threshold bins for range-less vitals (BP stage, BMI class, resting
  heart rate), and a fallback to a test-specific population range when the row's range
  is missing, which would merge the site-split tokens.
