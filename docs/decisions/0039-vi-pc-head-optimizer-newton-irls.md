# 0039 — The VI-PC logistic head is optimized by an aggregatable Newton (IRLS) step, selectable via `head_optimizer`

**Status:** Accepted

> **Numbering note.** Authored on `claude/faithful-flat-pc`, which numbers ADRs from
> the same counter as sibling branches (see ADR 0038's note). If a sibling claimed
> 0039 first, renumber (`git mv` + retitle); the slug disambiguates. The number is
> cosmetic; the decision is the content.

## Context

`OnlinePCLDA` (ADR 0038's VI port) co-fits a logistic head `w_CK` jointly with the
topics under minibatch SVI. The head was originally updated with a single Robbins-Monro
gradient step per global iteration (`sgd`). On the AoU antidepressant task this head
read heldout AUC ≈ chance despite `|w_CK|` being non-zero. Insight 0065 established the
cause: **one gradient step per SVI iteration cannot converge a logistic head against a
continuously-moving topic representation** — it wanders and lands ~orthogonal to the
batch-LR direction. This was *not* a ridge, θ-mismatch, optimizer-family, or
learning-rate problem (all refuted); it is non-convergence. A converging head also
matters beyond its own AUC: the supervised topic correction's gradient flows through
`w_CK`, so a mis-directed head feeds the correction a garbage signal.

A driver-side SGD inner-loop (many head steps per iteration) is the obvious fix but is
blocked by two facts of the existing `VIRunner`: (1) it scales every sufficient-stat by
`corpus/batch`, which would corrupt raw per-doc θ shipped through the stats dict; (2)
the corpus is over-partitioned, so a per-partition inner-loop averages over ~3 docs.
Both would force a runner-level change to shared infrastructure.

## Decision

Add a `head_optimizer` parameter with two values; **default `sgd`** (prior behavior
byte-for-byte). The new **`newton`** takes one ridge-Newton (IRLS) step per global
iteration using only aggregatable sufficient statistics:

- per-label gradient `g_c = Σ_d (p−y)π_d` (already emitted as `grad_wCK_stat`) and
- per-label Fisher information `H_c = Σ_d p(1−p)·π_dπ_dᵀ` (new `head_hess_stat`),

both additive doc-sums that combine through the existing `treeReduce`. Because the
runner scales *both* `g` and `H` by the same `corpus/batch` factor, the solve `H⁻¹g` is
**scale-invariant** — the scaling cancels, no raw θ reaches the driver, and there is **no
runner change**. `head_lr` doubles as the Newton damping (an EMA over per-minibatch
optima; ~0.3 stabilizes the oscillation inherent to per-minibatch Newton), and
`head_newton_ridge` is a relative ridge (fraction of `mean(diag(H))`) that conditions
the per-label solve without biasing its direction (AUC is scale-invariant to head
magnitude).

An intermediate `adam` head (a per-parameter two-timescale step, decoupled from the
runner's ρ) was tried first; it landed at the *same* mis-directed `w_CK` as `sgd`
(cos ≈ 0.11), which is what ruled out the optimizer family and pointed at
non-convergence. Having served that diagnostic purpose and been superseded by `newton`,
`adam` was **removed** (2026-08, PC walkthrough review) to keep the head surface to the
two settled options — its moment buffers, `head_beta1/2`/`head_eps` params, and the
warm-start lazy-init branch were dead weight. The history lives in experiments 0072/0073
and the commit log.

## Consequences

- **Positive.** The head converges (AoU: 0.52 → 0.60, direction cos 0.09 → 0.35), the
  fix is entirely PC-model-local (no shared-infra risk), and it is the aggregatable/
  scale-invariant analogue of Hughes's per-step local re-solve. The topic correction now
  receives a valid head signal — the precondition for a fair test of whether PC's
  topic-shaping helps at all (insight 0066).
- **Costs.** `head_hess_stat` adds `C·K²` floats per partition (25k at C=10, K=50 —
  trivial) and one extra CAVI pass per doc to form `H`. Per-minibatch Newton oscillates
  without damping and can spike on a near-singular minibatch `H_c`; mitigated by
  `head_lr`/`head_newton_ridge`, not eliminated.
- **Known limit.** A converged head still plateaus below the batch-LR ceiling when the
  *topics* have not converged within the iteration budget (the head chases a moving
  target). Closable with more supervised iterations or Polyak-averaging the head; parked
  as a refinement, not required for the AoU conclusion.
- Persists through the mllib `OnlinePCLDAEstimator` (`headOptimizer`/`headNewtonRidge` params,
  save/resume round-trip) and the cloud driver (`--head-optimizer newton`), threaded by
  `run_experiment.build_pc_args`.
