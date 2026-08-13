# 0040 — The VI-PC head's runaway `|w|` is fixed by a fixed L2 (`head_l2`), not by more Newton steps; the inner loop is fixed-point-equivalent

**Status:** Accepted
**Date:** 2026-08-13

> **Numbering note.** Authored on `claude/spectral-anchor-topic-k-200nqp`, which shares
> an ADR counter with sibling branches (see ADR 0038/0039's notes). Renumber if a sibling
> claimed 0040 first; the slug disambiguates.

## Context

ADR 0039 gave `OnlinePCLDA`'s logistic head an aggregatable one-step ridge-Newton
(`head_optimizer="newton"`) with a **relative** ridge (`head_newton_ridge · mean(diag H)`)
that "only conditions the solve." On a realistic Mondo-DAG case-finding benchmark planted
on real EHR β (`manual_pc_dag_case_finding_realistic.py`), that head exhibited the
signature 0039 flagged as a "known limit": it **shapes topics to be predictive**
(post-hoc LR on the shaped θ reaches AUC 0.965) yet **cannot predict itself** (head AUC
0.646), with `|w_CK|` measured at **3.4e11**.

The diagnosis (`manual_pc_head_shaping_diagnostic.py`): PC's own shaping makes the topics
**separable**, so `p(1−p) → 0`, so the relative ridge — a fraction of `mean(diag H)` —
**vanishes with the Fisher information it is scaled by**. The logistic MLE on separable
data is at infinity, `|w|` runs away, and the single damped Newton step then oscillates
and misaims (direction cos to the batch-LR solution 0.637). This is exactly the failure
Hughes (2018) forestalls with a *fixed* `lambda_w = 0.001` weight decay — a deliberately
weak but **non-vanishing** L2. The faithful full-batch L-BFGS reference (`analysis/pc`,
ADR 0038) confirms: with fixed `lambda_w=0.001` it keeps `|w|=5.97` and predicts 0.812 ≈
its own topics-LR ceiling 0.868.

## Decision

Add **`head_l2`** (default 0.0): a **fixed** per-doc L2 prior on `w_CK`, scaled by
`n_docs` to track the corpus-scaled Fisher, added to the Newton ridge *before* the
relative term. Any `head_l2 > 0` keeps the head finite on separable topics (the blowup
guard). This is the actual fix for the runaway `|w|`; the relative ridge stays as a pure
conditioner. Also add **`head_inner_iters`** (default 0): a driver-side inner Newton loop
on a bounded subsample (`head_sample_cap`) of the per-doc label-free design (θ, y, obs),
converging the head *within* an iteration.

## Alternatives considered

- **Inner loop as the head fix (rejected as redundant).** Empirically, with the same fixed
  L2 the one-step Newton **already converges**: one step per SVI iteration accumulates,
  over the ~60 iterations the topics take to settle, to the identical regularized fixed
  point. `INNER k=10 == one-step`, byte-for-byte (`|w|=4.76`); on a frozen design,
  1-step×60 lands ‖·‖=0 from a 60-step inner loop. The discriminating `INNER l2=0` config
  (internal 1e-3 fallback → `|w|=4.76`, unlike one-step `l2=0`'s 3.4e11) proves the branch
  fires — it just reaches the same place. So the lever is the **ridge type**, not step
  count. The inner loop is kept (off by default) only for the regime where topics never
  stabilize (aggressive minibatching / few iters), and it costs shipping raw θ to the
  driver, breaking 0039's aggregatable/scale-invariant property.
- **Raising `head_l2` to recover prediction outright (rejected).** A fixed L2 large enough
  to finitize `w` also **damps the shaping gradient** (∝ `|w_CK|`): topics-LR falls from
  0.965 (l2=0) to 0.525 (l2=1e-3). This shape-vs-regularize tension is intrinsic to the
  online/alternating scheme; no single `head_l2` reaches the reference's joint ceiling.

## Consequences

- **Positive.** `head_l2` eliminates the 3.4e11 blowup with a one-line, aggregatable,
  Hughes-faithful mechanism; 0039's default behavior is unchanged at `head_l2=0`.
- **Refines 0039.** 0039's "relative ridge only conditions the solve" is true only where
  the Fisher is bounded away from 0; under PC shaping it vanishes and must be backstopped
  by a fixed L2. 0039's "known limit" (head plateaus below the LR ceiling) is now
  attributed precisely: it is **not** head non-convergence (the head *does* converge with
  fixed L2) but **online/alternating vs joint optimization** — the reference's full-batch
  L-BFGS over (topics, head) jointly finds a basin (topics-LR 0.868) the alternating
  natural-grad-λ + head-Newton scheme does not (0.525), compounded by the
  shape-vs-regularize tension.
- **Costs.** `head_inner_iters > 0` ships raw θ (bounded by `head_sample_cap`) to the
  driver, unlike the fully-aggregatable one-step Newton. Left at 0 by default.
- **Open.** Closing the topics-quality gap likely needs a joint/second-order step over
  (topics, head) or a two-stage fit (shape with weak L2, then re-fit the head on frozen
  topics — the Hughes two-stage pattern), not more head iterations. Parked for the
  Gated-PC composition work.
