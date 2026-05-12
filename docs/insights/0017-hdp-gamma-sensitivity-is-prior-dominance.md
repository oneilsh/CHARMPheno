# 0017 — HDP γ-sensitivity reveals prior-dominated outcomes
**Date:** 2026-05-12
**Topic:** hdp | diagnostics
**Status:** Confirmed

The nominal promise of HDP is "let the data decide K" — replace the LDA
hyperparameter K with a nonparametric stick-breaking prior, and the
posterior tells you how many topics the data wants. The empirical
reality on our corpus is that we've replaced K with γ, and γ is *more*
sensitive than K in our regime.

Side-by-side outcomes from runs differing only in γ (or in optimize-γ
behavior) on the same patient-year condition-era corpus:

| run                       | γ (final) | catch-all E[β] | eff-K | active@0.5 |
|---------------------------|-----------|----------------|-------|-----------|
| γ₀=1, optimize on         | ~1 (collapsed) | n/a       | ~7    | ~3        |
| γ₀=50, optimize on, T=150 | ~68       | 0.22           | ~60   | ~19       |
| γ₀=50, optimize on, T=80  | ~74       | 0.55           | ~14   | 1         |
| γ clamped at ~25, T=80    | (smaller) | substantially  | (larger) | (more)  |
|                           |           | reduced        |       |           |

Comparable LDA runs (varying K=25 → K=50) produce qualitatively similar
mass distributions — gradual topic-count change, no regime shift.
HDP doesn't have this stability.

## Mechanism

GEM stick-breaking is non-exchangeable: the first stick has the largest
expected weight from Beta(1, γ), the second smaller, etc. Whichever stick
gets used first is determined by initialization + early-batch dynamics,
not by data structure. The "winner" stick then accretes mass
(rich-get-richer), and γ controls how steeply the geometric decay falls
off after that winner:

- Small γ → very steep decay → one winner takes everything (K-collapse,
  see [0001](0001-hdp-gamma-collapse-at-low-gamma0.md)).
- Large γ → shallow decay → many topics get nonzero mass but the
  leading topic still has structural advantage; catch-all hoarding
  results (see [0002](0002-hdp-catchall-hoarding-at-last-stick.md)).

There's no γ regime where the data's cluster structure dominates over
the prior's geometric-decay preference. The prior is dictating the
mass-distribution *shape*; the data is decorating it.

## Why our corpus makes this worse

The patient-year condition-era corpus has exactly one dominant
comorbidity pattern (HTN + HLD + T2DM + chronic-pain + GERD) plus a
long tail of rare phenotypes. GEM's geometric decay basically
*encodes* that shape as a prior — first stick big, rest small — so the
prior matches the worst feature of the data and amplifies it.

Pitman-Yor process (the two-parameter generalization of GEM) allows
power-law tail weights instead of geometric, which would fit medical
data's long-tailed phenotype distribution better in principle. We
don't have a PYP-HDP implementation; it'd be a nontrivial port.

## Implication

The marginal nonparametric benefit on this corpus — discovering 2-4
additional rare phenotypes (CF, Factor VIII deficiency, hypercoag) at
the cost of γ-sensitivity, catch-all dominance, and 50+ wasted topic
slots absorbing residual variance — may not be worth the modeling
pain. The defensible primary modeling story for this corpus is
**LDA at judicious K, treating HDP as a confirmation/exploration tool
rather than the headline**:

- LDA at K=25 produced a smooth mass-distribution decay (~36% spread
  across 4 background flavors, then 2-4% per phenotype, then a 1-2%
  tail) that matches the corpus structure (see
  [0005](0005-lda-decomposes-background-into-flavors.md)).
- LDA's K choice is at least honestly a hyperparameter; you pick it
  and the model's behavior is stable in a reasonable K-neighborhood.
- HDP-discovered rare phenotypes can still be reported as
  "exploratory; HDP at T=80 surfaced these clusters that LDA at K=25
  collapsed into broader topics."

## Setting context

Multiple HDP runs across γ ∈ {1, 25, 50}, T ∈ {80, 150}, all on AoU
OMOP condition_era patient-year docs (after [ADR 0018](../decisions/0018-document-unit-abstraction.md))
with `--doc-min-length` 20-30, vocab_size=10000, min_df=10, subsampling
0.1-0.2. The γ=1 collapse run was from earlier patient-lifetime
experimentation. The qualitative pattern — γ sensitivity dominates
modeling outcomes — held across all parameter settings tried.
