# 0111 episode-probe results — R5.9 multiplier, R5.10 kill, node yield — 2026-09-02

`make diag-episode-probe ID=110` (commit `4e9bb33`), off the 0110 native-Mondo
E4 sidecar. No fit, no rescan, no cache impact. Gaps 60d and 90d, W=365, prior
obs 365. 345,125 persons in the sidecar; pooled figures only (egress-safe).

## The three headline numbers

| | gap 60d | gap 90d |
|---|---|---|
| raw episodes / person | 11.03 | 9.10 |
| **gated** episodes / person (both obs gates) | **10.40** | **8.55** |
| gated persons with ≥1 episode | 295,058 | 294,426 |
| capped corpus multiplier — cap 3 | **×2.70** | ×2.66 |
| capped corpus multiplier — cap 5 | ×4.09 | ×3.98 |
| new nodes per gated episode (mean / p50 / p90) | 3.49 / 2 / 7 | 4.28 / 2 / 9 |
| **node yield ≥20 gated episodes** | **2,583** | 2,584 |
| node yield ≥100 | 2,321 | 2,324 |
| node yield ≥1 (any) | 2,659 | 2,659 |
| overall kill (both gates) | 19.3% | 19.8% |
| kill — prior-obs gate alone | 9.5% | 10.2% |
| kill — follow-up gate alone | 13.8% | 14.4% |
| **first-episode kill vs later-episode kill** | **66.2% vs 14.7%** | 66.2% vs 14.1% |

## Finding 1 — episodes un-starve essentially the whole label space (R7.3: PASS)

0110's incident readout scored ~1,791 nodes and dropped **923 as
incident-starved** (<20 incident positives). The episode corpus puts **2,583 of
the 2,714 Mondo nodes at ≥20 gated first-attestation episodes** — and 2,321 at
≥100. Only ~130 nodes fail to clear 20, at the *uncapped* frontier bound. This
is the entire justification for 0111, vindicated: a population-random index
catches one pre-onset year per person and starves a third of the DAG; anchoring
on presentations captures every case at onset and hands almost the whole label
space a scoreable incident cohort.

Two caveats kept honest. (a) This is the **uncapped gated** yield; the per-person
cap (D11) erodes it, so the ≥100 figure (2,321) is the more cap-robust floor —
a node with 100 uncapped episodes keeps ~33 after a cap that retains ~1/3, still
clear of 20. (b) It is a **frontier lower bound**; the closure fold only adds
episodes to ancestors. The real number is the episode-corpus incident census
after the build, on the existing GO/NO-GO tool. But the margin over 923 is so
large that R7.3's pre-registered anchor is cleared with room to spare.

## Finding 2 — the 66% first-episode kill is the R5.10 warning, quantified hard

The audit predicted (§5b) that the 365-day prior-observation gate would
"systematically drop the earliest, most unambiguously incident episodes." It
does, and the magnitude is stark: **two-thirds of every person's first episode
is lost** (66.2%, stable across gap), versus 14.7% of later episodes. The
overall kill is a modest 19%, but it is concentrated precisely on the
incident-richest slice — the single most-incident anchor per person survives
only ~34% of the time.

This is the death of "100% incident capture" made numeric, exactly as R5.10
said it would be. It is a **bias to document, not a dealbreaker** — 34% of first
episodes (~117k pure-onset anchors) plus ~2.9M later new-diagnosis episodes
still survive, an enormous incident corpus. But it forces one genuine design
question onto the plan (below): is 365-day prior-obs the right gate for an
experiment whose entire purpose is catching onset?

## Decisions this resolves (spec §4)

- **gap = 90d.** Node yield and kill are identical to 60d; 90d gives a smaller
  corpus (fewer, denser episodes) with *higher* new-nodes-per-episode (4.28 vs
  3.49), so under a per-person cap the retained docs carry more incident signal
  (~12.8 vs ~10.5 positives/person at cap 3) and overlap less (R5.11 / insight
  0009). The >10% multiplier gap that §4's rule flagged is real but points the
  same way the modeling argument does: take the smaller corpus.
- **cap = 3 (recommended), cap 5 held as fallback.** Uncapped is ×8.6 —
  infeasible (~6h × 2 solves, driver collect well past 16 GB). cap 3 → ×2.66,
  cap 5 → ×3.98. cap 3 is also the insight-0009 guard: it bounds how much a
  chronic megapatient (p99=36 episodes, max=63) can inflate the background. If
  the corpus census shows cap 3 erodes node yield below the ~1,800 bar, cap 5 is
  the fallback — one knob, decided by the census, not guessed.
- **R5.8 distributed eval + distributed calibration apply: WIRE IT.** Even at
  cap 3 / ×2.66 we sit at the ×3 boundary the spec called "no headroom," and
  `calibrate_per_node`'s driver float64 copy alone (~2.4 GB × ~2.7 ≈ 6.5 GB, on
  top of the lean bundle) is the wall regardless of cap. The ×3 trigger is
  effectively tripped; the distributed path is a plan precondition, not a
  contingency.

## The one fork left for Shawn (§4, capture-claim wording → but really a design choice)

The 66% first-episode kill is a **definitional** choice, not a default I should
make silently. `_LOOKBACK_PRIOR_OBS_DAYS = 365` is hardcoded and deliberately
un-overridable (`case_finding_assembly.py:43-53`). Two roads:

1. **Keep 365d, report the measured conditional-capture rate.** Honest, no code
   fork, admits the survivorship bias openly (the corpus is "incident *among the
   year-plus-observed*"). The onset-richest third of first episodes is simply
   out of reach — as it is for any prior-obs-gated design.
2. **Relax prior-obs for the episode arm (a sensitivity, e.g. 90d or 0d).**
   Rescues first episodes and the purest onset signal, at the cost of admitting
   prevalent contamination (less prior history to confirm the diagnosis is
   genuinely new). This touches the un-overridable gate — a real intervention,
   and a scientific statement about what "incident" means here.

My lean is **road 1 for the primary arm, road 2 as a named sensitivity** the
plan budgets one cheap probe pass for (the tool already decomposes the gate, so
measuring capture at relaxed prior-obs is a re-run with one flag, no fit). But
this is the incidence definition, which is yours to set before I write the plan.
