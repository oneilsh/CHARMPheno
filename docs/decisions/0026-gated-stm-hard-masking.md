# ADR 0026: Gated STM uses hard masking for background/foreground blocks

**Status:** Accepted
**Date:** 2026-06-23

## Context

Prevalence-only STM gives a rare group prevalence fidelity but not content
fidelity (docs/insights/0026). To give a rare group its own topic *content*
while borrowing the large cohort's strength for shared structure, we add an
opt-in background/foreground topic-block partition. Three mechanisms were
considered for enforcing that only a group's documents express its foreground
topics.

## Decision

Use **hard masking** (approach A): a document's allowed-topic set is
`background union (foreground blocks of its groups)`; per-document inference
optimizes eta only over the allowed set, so disallowed topics have theta exactly
0 and contribute zero sufficient statistics. The M-step is block-aware
(per-block Gamma normal equations, per-topic Sigma divisor). The canonical
no-gating path reduces to an implicit all-background partition, numerically
identical to prior STM.

## Alternatives considered

- **B — soft prevalence prior.** An informative Gaussian prior on Gamma drives
  majority foreground prevalence toward 0, so gating emerges rather than being
  imposed. Smallest engine change and a fully continuous/joint fit, which may
  have its own benefits (no hard structural zeros; the model can let a near-group
  document borrow a foreground topic when the data strongly support it). Rejected
  for v1 because isolation is soft (mild foreground contamination) and it adds a
  prior-strength tuning knob, but explicitly retained as a future option.
- **C — two-pass freeze-background.** Fit background on the full corpus, freeze
  beta, fit foreground on the group residual. Simplest math; rejected because it
  is not a joint fit (background cannot adapt) and adds two-stage checkpoint
  management.

## Consequences

- The group variable must NOT appear in the prevalence formula: within a
  foreground block's group-only document subset the group indicator is constant,
  so the foreground regression would be rank-deficient (only ridge-rescued,
  uninterpretable). Enforced by a guard.
- Group-shifted *background* prevalence (a legitimate separate effect) is NOT
  available via the gating variable; it would require a distinct covariate, a
  possible future extension.
- Foreground content fidelity comes from gating, not from covariate-dependent
  beta; content covariates (SAGE) remain a separate, unbuilt extension.
