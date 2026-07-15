# 0027 — Lazy block updates for the gated SVI M-step

**Status:** Accepted
**Date:** 2026-06-25

## Context

Gated topic models (ADR 0026) estimate per-group parameters online by
ρ-blended stochastic-EM over minibatches. When a minibatch contains no
documents for a group G, G's foreground sufficient statistics are zero, so the
M-step formed *degenerate* targets for G's block — Γ columns solved from a zero
`XtX_groups` (rescued only by the tiny `sigma_ridge`, giving Γ≈0) and a zero
residual giving Σ→`SIGMA_FLOOR` — and then ρ-blended the block toward them.
Across many minibatches that miss a rare group, this systematically shrinks
that group's foreground Γ toward 0 and Σ toward the floor, in proportion to how
often minibatches lack the group. The bias falls hardest on exactly the
rare-disease foreground the gating exists to recover, and a near-floor Σ
further over-tightens the prior, suppressing the per-document deviations that
would express distinct sub-phenotypes. Diagnosed during the `stm`-branch
pre-merge review. The PLDA reference (Stanford TMT) never hit this because it
runs *batch* collapsed Gibbs / CVB0, where a rare group's tokens are present in
every sweep; the pathology is an artifact of our online/minibatch engine.

## Decision

**Lazy block updates.** A block (the background block, or a group's foreground
block) with zero documents in the current minibatch is left unchanged: its Γ
columns and Σ entries skip the ρ-blend, detected via `n_docs_per_topic > 0`
(already emitted by `local_update`). Implemented by defaulting each block's
target to the *current* parameter value, so the ρ-blend is a no-op for absent
blocks, and by guarding the per-block solve so a singular zero `XtX_groups` is
never factored. The present-block targets are self-normalizing ratios (a normal
-equations solve and a per-topic mean), so a present block is estimated
correctly from however few documents it has; the no-gating path and any
all-groups-present minibatch are numerically identical to before. The same
`n_docs_per_topic` gate ports verbatim to the forthcoming gated LDA (PLDA)
M-step on λ foreground rows.

## Alternatives considered

- **Full-batch EM (`batch_fraction = 1.0`).** Eliminates the pathology because
  no group is ever absent — and is the right thing to *test* with — but gives
  up SVI's scalability. Kept as a supported configuration, not the default.
- **Stratified / group-aware minibatch sampling.** Guarantee a quota of rare-
  group documents per batch. Keeps scalability but requires importance-
  reweighting to stay unbiased; lazy updates are simpler and unbiased without
  reweighting.
- **Memoized sufficient statistics** (cache per-group stats, update the global
  from the running cache; Hughes & Sudderth 2013). A more general version of
  the same idea; larger engine change. Revisit if a many-group cache is needed.

## Consequences

- Removes the rare-group downward bias and restores the invariant the batch
  PLDA reference gets for free. Also fixes two latent crashes: an empty
  minibatch (now a clean no-op) and a singular per-group solve when
  `sigma_ridge = 0`.
- Borrowing strength across rare groups (a hierarchical Γ/Σ prior) and seeded
  foreground β priors (PheCode/CCS) remain separate, larger options for the
  genuinely data-starved rare-disease regime; see insight 0028.
- Independent of the Hessian-positive-definiteness guard on the per-document
  Laplace inverse (ADR 0029), a sibling robustness fix from the same review.
