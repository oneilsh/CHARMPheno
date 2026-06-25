# ADR 0028: Dashboard conditions its client-side samplers via a Dirichlet mean-match

**Status:** Accepted
**Date:** 2026-06-25

## Context

The dashboard's Simulator and Patient Atlas both generate synthetic samples
client-side from a document-topic prior: the Patient Atlas builds a synthetic
cohort (`generateCohort` in `dashboard/src/lib/cohort.ts`) and the Simulator
draws records (`runSimulator` in `dashboard/src/lib/simulator/runSamples.ts`).
Both use a **Dirichlet** prior with the corpus-average concentration vector
`model.alpha`, then a variational E-step against `beta`. Today both ignore STM
structure entirely — every bundle is sampled from the one corpus-average prior,
so covariate values and group gating have no effect on the simulated
subpopulation.

Rolling the conditioning context into these panels means the prior must depend
on the selected covariate values and group. The Phenotype Atlas already computes
the covariate-conditioned topic *mean* for display via
`covariatePrevalence(effects, x)` = softmax(Gamma^T x). The samplers need a
conditioned *prior*, not just a mean — and they are Dirichlet throughout, while
STM's true prevalence prior is logistic-normal (theta = softmax(eta),
eta ~ Normal(Gamma^T x, Sigma)).

## Decision

Condition the existing Dirichlet sampler by **moment-matching its mean** to the
conditioned topic proportions while **preserving the corpus prior's total
concentration**:

    alpha_cond[k] = alpha0 * p_cond[k],   alpha0 = sum_k(alpha_corpus[k])

A Dirichlet's mean is alpha / sum(alpha), so this keeps the corpus prior's total
concentration alpha0 (its peakiness / how sharply documents concentrate on a few
topics) and only rotates the mean to the conditioned proportions. `p_cond` is
chosen by the **same four quadrants the prevalence reader uses** — the sampler is
the generative twin of the display reader:

| Quadrant | p_cond | mask |
|---|---|---|
| plain (no covariate, no gating) | alpha_corpus normalized (= today's prior) | none |
| STM (covariate, no gating) | softmax(Gamma^T x) | none |
| gated LDA (gating, no covariate) | alpha_corpus normalized | hidden foreground floored |
| gated STM (covariate + gating) | covariatePrevalenceGated (mask-before-softmax) | built in |

Masked (out-of-group) foreground topics are **floored at a tiny epsilon**
(~1e-12, mirroring the engine's gated-block pin in ADR 0026) rather than set to
exactly zero, so the Dirichlet and the downstream E-step stay numerically defined
while those topics are effectively never drawn. The reader and the sampler share
one helper for the axis logic; `conditionedAlpha(...)` is the single source of the
conditioned prior.

The marginal sampler (per-patient draw from covariate marginals and group
proportions, for the Patient Atlas's realistic mixed cohort) draws a per-patient
`x` and `group`, then feeds each through the same `conditionedAlpha`.

## Alternatives considered

- **B — sample eta from the logistic-normal directly.** Faithful to STM's actual
  prior: draw eta ~ Normal(Gamma^T x, Sigma), theta = softmax(eta). Rejected for
  this increment because it requires exporting the per-topic Sigma (not in the
  bundle today) and replacing the Dirichlet sampling path that `generateCohort`,
  `runSimulator`, and the variational E-step are all built around — a large
  rework of forward-simulation code whose purpose is illustration, not fitting.
  Retained as a future option if the simulated cohorts need to match STM's
  interior-smoothing behavior.
- **C — sample only over allowed topics and renormalize (hard drop).** Instead of
  flooring masked topics, restrict the simplex to allowed topics. Rejected
  because it breaks the fixed-length theta/alpha contract that the E-step and the
  per-topic display code assume; the epsilon floor keeps vector lengths stable
  with negligible probability mass on masked topics.
- **D — preserve the per-topic alpha shape and scale it by the covariate effect.**
  Rejected because the covariate effect in this model is expressed as the softmax
  mean, so matching the Dirichlet mean to that softmax is the natural projection;
  there is no separate per-topic shape to preserve beyond the total concentration.

## Consequences

- This is a **forward-simulation approximation for visualization**, not a faithful
  reproduction of STM's generative process. Per insight
  [0028](../insights/0028-dirichlet-vs-logistic-normal-rare-phenotype-recovery-gated-lda-is-plda.md),
  the Dirichlet places density at the simplex vertices (documents concentrate on a
  few topics) while the logistic-normal smooths toward the interior; an STM bundle
  sampled through this Dirichlet will therefore look **more peaked / more sparse**
  per document than a true STM draw would. This is acceptable — the panels answer
  "what does this subpopulation's phenotype mix look like," an illustrative view,
  and they already made the Dirichlet approximation for every bundle. The ADR makes
  the approximation explicit rather than silent.
- Total concentration is preserved, so conditioned cohorts keep the same
  document-level peakiness as the corpus; only the topic emphasis shifts with the
  covariates/group.
- The four-quadrant `p_cond` is shared with the prevalence reader, so the axis
  decoupling (plain / STM / gated LDA / gated STM) has one definition for both
  display and generation; a bundle with one axis, the other, both, or neither is
  handled the same way in both places.
- Plain and gated-LDA-without-covariates bundles are unaffected in concentration
  (alpha_cond reduces to alpha_corpus, optionally masked), so non-STM dashboards
  generate exactly as before.
- Masked topics are floored, not removed, so a group selection never produces a
  zero-length or renormalized prior; out-of-group foreground simply carries
  negligible mass.
