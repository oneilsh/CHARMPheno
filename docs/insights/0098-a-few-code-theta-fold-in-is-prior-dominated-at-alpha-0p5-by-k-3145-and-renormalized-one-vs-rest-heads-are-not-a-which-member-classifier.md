# 0098 — A few-code θ fold-in is prior-dominated at α 0.5 × K 3,145, and renormalized one-vs-rest heads are not a which-member classifier

**Date:** 2026-10-12
**Topic:** wp-d, acquisition, voi, fold-in, alpha, readout-heads, exp 0138, exp 0139
**Status:** Observed on the ribbon record 0137 (flat label space, K = 3,145, fixed
α = 0.5, stacked_mass99 heads), held-out patients of three candidate sets.

## Observation

Model-based acquisition (0139) re-inferred θ from the codes revealed so far with the
deployment fold-in and scored the saved readout heads. Over 20 questions the member
posterior hardly moved (Marfan/EDS: the empty-record call for every patient at every
step; cardiomyopathy: +0.01 balanced top-1), while 0138's per-code presence engine,
tempered to calibration, reached balanced top-1 0.61 (Marfan/EDS) and 0.24
(cardiomyopathy) on the same questions.

1. **Prior domination.** The fold-in's prior is α · K ≈ 1,570 pseudo-counts. A revealed
   code carrying a record count of a few shifts θ by well under 1%; θ from a sparse
   record sits at α / sum(α) for practical purposes. Whole records move it.
2. **Off-distribution.** The heads were fit on whole-record θ; the empty-record θ is
   outside that distribution, and they score it confidently wrong (connective tissue:
   the empty-record posterior picks a member right for 8% of patients; ECE 0.41–0.47).
3. **Wrong model class for "which child".** With the whole record — diagnosis codes
   included — renormalizing the stacked one-vs-rest heads across a set's members gives
   balanced top-1 0.32 (19 members), 0.22 (8), 0.48 (3). Per-edge cond AUC ~0.8 (0138
   part 2) ranks patients for ONE node; it does not make the heads comparable ACROSS
   nodes for one patient.

## Why it matters

- θ-based inference from a handful of codes needs a deployment α orders of magnitude
  smaller (with heads refit on that θ), not just a different question order. This is a
  concrete, measurable reason for the fixed-α ladder, independent of case-finding AUC.
- A clinical "which disease within this group" read wants a multiclass head per
  candidate set fit within the parent's cohort, not the marginal heads renormalized.
- Until then, the acquisition engine of record is the tempered presence model (0138),
  whose question ranking and calibration both held up.
