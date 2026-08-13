# 0041 — The VI-PC head's `head_l2` is an ABSOLUTE ridge (= Hughes `lambda_w`); the topics-quality gap was miscalibration, NOT joint-vs-alternating

**Status:** Accepted
**Date:** 2026-08-13

> **Numbering note.** Same shared-counter caveat as ADR 0038–0040; renumber if a sibling
> branch claimed 0041 first. Supersedes the "residual gap" analysis of ADR 0040.

## Context

ADR 0040 added `head_l2` as a fixed-L2 blowup guard for the one-step Newton head and
attributed the *remaining* topics-quality gap (online co-fit topics-LR ≈ 0.53 vs the
faithful full-batch L-BFGS reference's ≈ 0.87) to **online/alternating vs joint
optimization** plus a **shape-vs-regularize tension**. Two follow-ups overturned that
attribution:

1. **The confound isolation** (`manual_pc_hughes_settings_experiment`, insight-recorded)
   refuted π-iters and weight_y: matching Hughes's `T≈100` and `λ ≈ tokens/doc` did not
   move topics-LR.
2. **The joint-vs-alternating de-risk** (`manual_pc_joint_vs_alternating`, this ADR):
   a new `fit_mode="alternating"` on the reference — block-coordinate L-BFGS (topics with
   head fixed, then head with topics fixed), holding objective / π-MAP / L2 / init /
   full-batch / solver identical to `fit_mode="joint"` — reached **topics-LR 0.874, head
   0.967, |w|≈105**, matching joint (0.862 / 0.802). **Alternation does not collapse.**

The tell was |w|: reference alternating grew |w| to ≈105; the online model was stuck at
≈4.76. Cause: `head_l2` was applied **per-doc, ×n_docs** (`ridge = head_l2·n_docs`), so
`head_l2 = 1e-3` acted like Hughes's `lambda_w ≈ 0.84` — **≈840× (= n_docs) too strong**.
That throttled |w| and collapsed shaping; it was never a jointness or tension problem.

## Decision

`head_l2` is a **fixed ABSOLUTE L2 ridge = Hughes's `lambda_w`**: the ridge on the
**corpus-summed** head gradient, so at the head fixed point `|w| ~ |g|/head_l2`. The
`×n_docs` factor is removed from both the one-step Newton and the inner-loop branches
(the `n_docs` in `|g|` and the ridge cancel — matching the reference, where `weight_y`
and `÷n_tokens` cancel to leave `lambda_w`). **Default `head_l2` 0.0 → 1e-3** (Hughes's
value); `0.0` blows up on the separable topics PC creates, so it is now opt-in only.

Recalibration sweep on the online model (`manual_pc_head_l2_recalibration`):

| head_l2 (absolute) | topics-LR | HEAD | \|w\|max |
|---|---|---|---|
| 0 | 0.948 | 0.788 | 3.4e11 (blowup) |
| 1e-4 | 0.947 | 0.699 | 1.4e5 |
| **1e-3** (= λ_w) | **0.957** | **0.849** | 1.3e4 |
| 1e-2 | 0.914 | 0.697 | 1581 |
| 1e-1 | 0.503 | 0.492 | 6.65 (over-regularized) |

The online Newton head now shapes to topics-LR ≈ 0.96 (≥ the reference's 0.87) with a
finite, readable head. The good basin is wide (≈1e-4…1e-2, topics-LR 0.91–0.96), centered
on Hughes's canonical 1e-3.

## Alternatives considered

- **A joint step in the online model** (Route B/C from the design discussion). Motivated
  by 0040's now-refuted attribution; **not needed** — alternating reaches the reference's
  quality once the head is regularized at the correct absolute scale. Keeps the
  natural-gradient + one-step Newton scheme (and avoids reintroducing Adam, per the
  standing preference for Newton). Parked unless a future task shows a genuine coupling
  gap the absolute ridge does not close.
- **Keep `×n_docs`, recommend `head_l2 ≈ lambda_w/n_docs`.** Rejected: data-dependent and
  a footgun; absolute semantics make the default Hughes's 1e-3, data-independent.

## Consequences

- **Supersedes ADR 0040's "residual gap" analysis.** 0040's blowup-guard mechanism stands;
  its claim that the gap is online/alternating-vs-joint + a shape-vs-regularize tension is
  **withdrawn** — the gap was `head_l2` miscalibration by the `n_docs` factor. The
  shape-vs-regularize "tension" was an artifact of over-regularization, not fundamental.
- **`fit_mode` on the reference** is retained as a standing isolation tool (`joint` default
  unchanged, byte-for-byte; `alternating` for coupling questions). 121 `analysis/pc` +
  28 PC pyspark tests pass.
- **Head still under-reads its own shaping** (HEAD 0.85 vs topics-LR 0.96): the one-step
  Newton direction at large |w| is slightly off the converged classifier. Small residual,
  separate from this decision; the downstream case-finding readout uses post-hoc LR on the
  shaped θ (≈ topics-LR), so it is not on the critical path.
- Recorded empirically in insight 0068.
