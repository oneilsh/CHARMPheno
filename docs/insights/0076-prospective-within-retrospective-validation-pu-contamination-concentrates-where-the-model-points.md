# 0076 — Prospective-within-retrospective case-finding validation: PU contamination concentrates exactly where the model points, and the decile gradient is what distinguishes it from background incidence

**Date:** 2026-09-02
**Topic:** evaluation, PU learning, case-finding, label noise, conversion analysis

**Status:** Confirmed on exp 0110's record run via the E4 first-attestation
sidecar + persisted readout heads (`diag-conversion-analysis --deciles on`, no
re-fit). Pooled, right-censored at each horizon.

**Setting.** "Negatives aren't really negatives" — every incident-negative
label mixes true never-cases with not-yet-diagnosed positives (PU
contamination). Channel 1 is measurable: the sidecar stores each person's
first attestation of every label node over their WHOLE record, so a document
scored incident-negative for c can be followed FORWARD past the label window.
Retrospective data, prospective question.

**Finding 1 — the pooled PU floor alone proves nothing.** Incident negatives
convert to the real diagnosis at 0.96% / 1.77% / 2.53% (1y/2y/3y horizons,
denominators observation-gated so they shrink with horizon — the censoring gate
working, not a bug). Low, reassuring for label quality — but by itself
indistinguishable from background incidence: healthy people also get diagnosed
eventually.

**Finding 2 — the score-decile gradient is the discriminating diagnostic.**
Stratify the same conversions by the model's own score decile:

| horizon | d0 | d9 | top−bottom |
|---|---|---|---|
| 365d | 0.006 | 0.020 | +0.0140 (3.3×) |
| 730d | 0.011 | 0.036 | +0.0241 (3.3×) |
| 1095d | 0.017 | 0.049 | +0.0323 (2.9×) |

Monotonic at every horizon. Background incidence would be FLAT across deciles;
instead the "false" negatives convert at a rate climbing with the score the
model gave them. The label-noise channel is concentrated exactly where the
model points — the contamination IS the model surfacing not-yet-diagnosed
cases. At 3y a top-decile "negative" carries ~1-in-20 future-diagnosis odds vs
~1-in-60 at the bottom.

**Why this is a validation pattern, not just a diagnostic.** This is
prospective-within-retrospective case-finding validation with NO chart review:
the chart itself, read forward, adjudicates the model's flags. The
prerequisites are exactly two artifacts, both cheap: a full-record
first-attestation sidecar (dates captured where the unwindowed frame still
exists — downstream, windowing discards them irrecoverably) and persisted
scoring heads (so stratification never re-fits). Any PU-flavored labeler over
longitudinal records can run this loop.

**Boundary.** It is a LOWER bound on contamination (channel 1 only — persons
diagnosed outside the observed record never convert on paper) and an
OPTIMISTIC validation read only if diagnosis dates are independent of the
features — here they are not fully (utilization drives both), so the gradient
validates ranking, not calibrated risk. The horizon-sweep companion finding
(insight 0077) bounds how much training-side juice the gradient implies: very
little.
