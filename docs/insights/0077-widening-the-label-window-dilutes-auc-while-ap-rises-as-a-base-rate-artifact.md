# 0077 — Widening the label window DILUTES discrimination (AUC 0.600→0.586) while AP rises as a pure base-rate artifact; a growing decile gradient and a falling AUC are the same phenomenon, not a contradiction

**Date:** 2026-09-02
**Topic:** evaluation, label windows, metrics, experiment scoping

**Status:** Confirmed by the 0111 scout (`diag-conversion-analysis
--horizon-eval on`, exp 0110 saved fit + E4 sidecar, zero re-fits). Report:
`docs/reports/2026-09-02-0111-scouting-window-depth.md`. Pre-registered
decision rule, not triggered.

**Setting.** Hypothesis: the 1y label window under-credits the model — cases
it correctly flags get diagnosed in year 2 or 3 and are scored as
false positives, so a wider training window would raise measured (and maybe
real) skill. Cheap eval-side test before any re-fit: score with the 1y-fit
heads, relabel from the sidecar at W ∈ {365, 730, 1095}, shared node set
(1,622 nodes scoreable at all horizons), right-censored denominators.
Pre-registered rule: RISING AUC with W ⇒ widen the training window.

**The numbers:**

| horizon | shared AUC | shared AP |
|---|---|---|
| 365d | 0.6003 | 0.0199 |
| 730d | 0.5899 | 0.0289 |
| 1095d | 0.5857 | 0.0369 |

**Finding 1 — AUC FALLS monotonically; the rule is not triggered.** Year-2/3
converters are HARDER to rank from year-earlier features than year-1
converters, so pooling them in dilutes discrimination. The re-fit upside is
bounded tiny: 1y features already rank 3y converters at 0.586, ~0.015 below
the 1y number. Window-widening is refuted as a training-target change; longer
horizons remain a free eval-side reporting axis (the sidecar supplies them
with no re-fit).

**Finding 2 — the AP rise is prevalence, not skill.** Widening W raises label
prevalence, and AP climbs mechanically with prevalence at a FIXED ranking. An
AP-only reader would have drawn the opposite (wrong) conclusion. Same trap as
insight 0075's AP halving, opposite direction.

**Finding 3 — reconciling the growing decile gradient (insight 0076) with the
falling AUC.** The conversion gradient grows with horizon (+0.014→+0.032)
while pooled AUC falls. Not a contradiction: the top-scored negatives keep
converting at longer horizons (ranking of eventual cases holds up — the
gradient), while the mass of added late converters is individually harder to
rank (the dilution). Case-finding validation survives; "train wider for
better AUC" does not.

**Practice.** Pre-register the decision rule before running the sweep — this
one was, and it converted an attractive idea into a cheap, clean negative in
one scoring pass. Never scope an experiment off AP movement without checking
whether prevalence moved with it.
