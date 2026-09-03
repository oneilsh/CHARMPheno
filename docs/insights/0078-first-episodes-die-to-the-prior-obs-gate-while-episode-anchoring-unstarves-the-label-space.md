# 0078 — The 365d prior-obs gate kills 66% of FIRST diagnostic episodes vs ~15% of later ones (incidence-anticorrelated censoring), while episode anchoring un-starves 2,583 of 2,714 label nodes at ≥20 onset episodes

**Date:** 2026-09-03
**Topic:** corpus construction, observation gates, episodes, incident capture, exp 0111

**Status:** Measured by `diag-episode-probe` (WP8a) on the 0110 native-Mondo
E4 sidecar — 345k persons, gaps 60/90d, the assembler's own
`_window_observed_cohort` called for the gates. Pre-fit corpus facts,
independent of how exp 0111 turns out. Report:
`docs/reports/2026-09-02-0111-episode-probe-results.md`.

**Setting.** An episode = a gap-and-islands cluster of a person's
first-attestation dates (moments something NEW enters the record); candidate
index = episode start − 1 day. The audit (§5b) predicted the hardcoded
`_LOOKBACK_PRIOR_OBS_DAYS = 365` gate would drop the earliest, most
unambiguously incident episodes — a person's first episode sits at or near
record start by construction. Nothing had ever measured it.

**Finding 1 — the gate's censoring is strongly incidence-anticorrelated.**
Overall kill (both gates) is a mild ~19%, but it is not spread evenly: 66.2%
of first episodes die vs 14.1–14.7% of later ones (stable across gap). The
single most-onset-like anchor per person survives only ~1 time in 3. "100%
incident capture" was already superseded on paper (spec R5.10); this is its
magnitude. Any prior-obs-gated design over open-enrollment EHR inherits this:
the requirement "a year of history before onset" is precisely the thing new
patients presenting at onset don't have. Decision recorded: keep 365d for the
primary arm (the corpus is "incident among the year-plus-observed," stated),
with a relaxed-gate sensitivity as a named probe re-run.

**Finding 2 — episodes are abundant and heavy-tailed.** Gated: ~8.5–10.4
episodes/person (gap 90/60), p99 ≈ 36–45, max 63–87. Uncapped, the episode
corpus is ×8.6 — infeasible against the readout's O(N·C) driver collect and a
doc-multiplication hazard for chronic megapatients (insight 0009). Per-person
caps do the real sizing work: cap 3 → ×2.66, cap 5 → ×3.98. A larger gap (90d)
gives fewer, DENSER episodes (4.28 vs 3.49 new nodes each) with identical node
yield — under a cap, the retained docs carry more incident signal.

**Finding 3 — the headline: anchoring on presentations un-starves essentially
the whole label space.** 0110's incident evaluation scored ~1,791 nodes and
dropped 923 as <20 incident positives (insight 0075). The gated episode corpus
puts 2,583 of 2,714 Mondo nodes at ≥20 first-attestation episodes and 2,321 at
≥100 — a frontier-grain LOWER bound (the closure fold only adds). The
starvation was never about the diseases being rare in the data; it was the
one-random-year index looking away from almost every onset.

**Practice.** Measure doc-unit multipliers and gate kill rates with a cheap
probe BEFORE writing the plan that depends on them — this one moved the
distributed-eval decision from "contingency" to "precondition" and turned the
cap from a guess into a two-point choice. The probe cost two minutes per gap
because the sidecar already held the needed frame; artifacts persisted for one
analysis keep paying for the next.
