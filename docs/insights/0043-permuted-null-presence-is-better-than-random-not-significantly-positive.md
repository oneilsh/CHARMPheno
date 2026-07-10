# 0043 — Permuted-topic "presence" is a better-than-random test, NOT a significantly-positive one: because a random topic HURTS held-out prediction, "beats the null" is a LOWER bar than "beats zero," so it inflates weak/background topics (a zero-gain reference topic reads 70% "present")

**Date:** 2026-07-10
**Topic:** stm | diagnostics | predictive-gain | calibration
**Status:** Observed (exp 0044 population_glp1, 110 topics, one re-export diff)

The predictive-gain "presence" of a topic k is the fraction of its documents
where the per-doc held-out gain Delta_k = LL(allowed) − LL(allowed\\{k}) clears
some threshold. Two thresholds were compared on the real GLP-1 comparator corpus:
**beats-zero** (Delta_k > 0) and **beats-null** (Delta_k exceeds the document's
own permuted-topic null max over n_perm=4 permutations, `presence_vs_null`). The
second was intended as the "statistically greater than 0" version — a stricter,
noise-filtering bar. The re-export diff (against a pre-swap snapshot) showed it is
the **opposite**: more permissive, and it inflates exactly the topics we wanted to
filter.

## Observation

94 of 110 topics rose under beats-null. Median presence 0.222 → 0.350; background
topics 0.216 → 0.343 (+0.127), foreground drug arms 0.567 → 0.588 (+0.021). The
decisive case is the **reference topic** (topic 0): mean_gain = 0 (it adds nothing
to held-out prediction), beats-zero presence = 0.000 (correct), but beats-null
presence = **0.703**. A topic with zero predictive gain cannot be "statistically
present," so beats-null is not measuring "adds signal."

## Interpretation

A *permuted* topic (correct marginal, scrambled word associations) does not score
Delta ≈ 0 — forced into the allowed set it **actively hurts** held-out prediction,
stealing theta mass and predicting the wrong words, so its Delta is slightly
**negative** (the pooled null-band mean was −0.046). Therefore "beats the permuted
null" ≈ "Delta > slightly-negative," which is an **easier** bar than "Delta > 0."
For weak/background topics whose Delta hovers near 0, that sub-zero threshold flips
them from absent to present; for strong topics (SGLT2i arms, 50–90 nats) Delta is
clearly positive under either bar, so they barely move (a few even tick *down* by
~0.05, where an unusually high null-max among 4 permutations just edges out a
positive real Delta). Net: `presence_vs_null` is a **"better than a random topic"**
test. Because random topics are net-harmful, that bar sits *below* zero — the
reverse of the significance test intended.

## Implications

- Do not treat `presence_vs_null` as "statistically present." It answers "does the
  real topic beat a scrambled version of itself," a genuinely lower bar than "adds
  positive held-out signal." beats-zero is the more sensible "adds signal" reading
  on this corpus.
- A true significantly-greater-than-0 test needs a per-doc dispersion, e.g.
  Delta_k / SE_k > z with SE_k from the per-held-out-token log-ratio variance — a
  reference against **0 with a per-doc error bar**, not against a random topic.
- The null-band being negative is itself the useful diagnostic: it quantifies how
  much a random topic costs, and it is the reason the two presence definitions
  diverge in sign of leniency.
- Decision on which "presence" the dashboard ships is parked (revert headline to
  beats-zero vs build the per-doc t-test); the pre-swap beats-zero snapshot is
  retained for whichever path is taken.

**Setting context:** Local diff of two dashboard exports of the same fitted gated
STM (exp 0044 population_glp1: whole-population background + GLP-1/SGLT2i/
tirzepatide/combo drug arms, 110 display topics, 175,476 fit docs, eta_scale c =
7.89, held-out document-completion predictive gain at 30% masked tokens, n_perm=4
permuted-topic null, Jelinek-Mercer background smoothing). One re-export; direction
is mechanistically explained (negative permuted-null) so likely to generalize, but
marked Observed pending a second corpus.

**Related:** insight [0038](0038-heldout-ll-recovers-true-concentration-and-lda-alpha-opt-is-not-hot.md)
(held-out-LL concentration calibration), [0042](0042-cofit-beta-does-not-reproduce-stm-lda-peakiness-gap.md)
(the co-fit-beta negative result); the export-boundary presence swap is commit
53767c2; the beats-zero snapshot is memory `project_exp44_presence_snapshot`.
