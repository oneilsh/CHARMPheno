# 0035 — Dashboard logistic-normal forward sampler for STM bundles

**Status:** Accepted
**Date:** 2026-07-01

## Context

ADR [0028](0028-dashboard-conditioned-dirichlet-prior.md) chose the Dirichlet
mean-match to condition the dashboard's samplers (Simulator and Patient Atlas)
because the true STM prior — logistic-normal, θ = softmax(η), η ~ Normal(Γᵀx, Σ)
— required exporting the per-topic covariance Σ, which was not available in the
bundle at that time. The mean-match was a faithful approximation given the
constraint.

ADR [0034](0034-stm-blockwise-unit-diagonal-correlation-sigma.md) resolves that
constraint: Σ is now estimated as a unit-diagonal correlation matrix and exported
in `correlation.json` under the key `R`. The generative panels can now draw from
the true logistic-normal prior, not a Dirichlet proxy. This ADR records that
forward sampling path.

## Decision

**1. Logistic-normal forward sampling (reference to Blei & Lafferty 2007).**
For STM bundles, the samplers now draw θ directly from the conditioned
logistic-normal prior:

    η ~ Normal(Γᵀx, Σ_allowed)
    θ = softmax(η)

where Σ_allowed is the per-group correlation sub-block from `correlation.json` and
Γᵀx is the covariate effect on the mean (the same linear predictor the Phenotype
Atlas displays). The reference topic's η is pinned to 0 (standard practice to
anchor the softmax), so its row in Γ and its corresponding row and column in Σ
are excluded from the draw.

**2. Hand-rolled Cholesky + MVN sampler, no npm dependencies.**
The implementation uses a supply-chain-conscious approach: hand-rolled Cholesky
decomposition and multivariate normal draw (standard linear algebra, no external
package). This keeps the sampler self-contained and avoids adding npm-managed
dependencies to the dashboard frontend.

**3. Per-group PD sub-block selection.**
Each group's allowed topics (background ∪ foreground group) define a sub-block
of Σ. The sampler extracts and decomposes Σ[allowed, allowed] via Cholesky,
draws the per-topic latent variables, and applies the logit-softmax. The
sub-block is guaranteed PD by construction (ADR 0034, single-label gating).

**4. Forward-only scope; E-step deferred.**
This ADR records the forward generative sampling path. The prefix-posterior
Laplace E-step (variational conditioning of η given observed β) is deferred; STM
bundles currently use the same E-step as non-STM (fixed-point updates on the
mean, approximating the marginal posterior). Enhancing the E-step is a future arc
once forward sampling is validated.

**5. Non-STM (LDA/HDP) bundles keep the Dirichlet path.**
Bundles without STM structure have no exported Σ, so they continue to use ADR
0028's mean-match Dirichlet. The sampler routes STM bundles → logistic-normal,
non-STM → Dirichlet, on bundle detection (presence of `correlation.json` and
metadata flags).

## Alternatives considered

- **A — keep the Dirichlet mean-match for STM bundles too.** Rejected now that Σ
  is exported. The mean-match is a valid approximation when Σ is unavailable
  (ADR 0028), but the true prior is now accessible; using it improves fidelity
  to the model without extra complexity.
- **B — npm dependency for Cholesky/MVN (e.g., numeric.js, simple-statistics).**
  Rejected. Hand-rolled Cholesky is ~20 lines of straightforward code; MVN draw
  via QR is equally simple. The standard library additions are trivial and avoid
  supply-chain friction.
- **C — use the full (K×K) Σ and handle the reference topic's row/column via
  bookkeeping.** Rejected. Pinning the reference topic's η to 0 removes its row
  and column from the draw; including them would require either (i) adding dummy
  variance logic, or (ii) drawing a full vector and discarding the reference
  coordinate — both add complexity. Extracting the (K−1)×(K−1) sub-block is
  cleaner.
- **D — use Laplace E-step for the prefix posterior instead of fixed-point.**
  Deferred. Full Bayesian posterior inference via Laplace is valuable (it would
  capture smoothing from the prior), but the current fixed-point path is a
  working approximation and allows the forward sampler to roll out independently.
  The E-step enhancement is a future arc; see Consequences.

## Consequences

- **Sampled cohorts now reflect true topic co-occurrence structure.** The
  logistic-normal prior smooths toward interior-simplex points (topic mixtures),
  so sampled documents show realistic co-topic prevalence instead of the
  sparse-peaked pattern of the Dirichlet. This makes the Simulator and Patient
  Atlas more faithful illustrations of STM bundles' generative behavior (insight
  [0028](../insights/0028-dirichlet-vs-logistic-normal-rare-phenotype-recovery-gated-lda-is-plda.md)
  distinguishes the two).
- **Supersedes ADR 0028's mean-match for STM bundles.** For STM (and future
  DP-STM), the logistic-normal route is the reference; the Dirichlet mean-match
  remains the path for LDA, HDP, and other non-STM bundles without Σ export.
- **Σ-less models (LDA/HDP) are unaffected.** Non-STM samplers continue to use
  ADR 0028 unchanged; the routing is silent (bundle metadata).
- **No new npm dependency or external library.** Cholesky and MVN are hand-rolled
  (textbook linear algebra).
- **Forward sampling only; prefix-posterior E-step is future work.** The current
  E-step is fixed-point (per ADR 0028 / constant-prior regime); a Laplace or
  iterative E-step that uses the true posterior curvature from Σ would further
  tighten conditioning but is deferred.

## References

- Blei, D. M. & Lafferty, J. D. (2007). "A Correlated Topic Model of Science."
  *Annals of Applied Statistics*, 1(1), 17–35. — the logistic-normal prior
  θ = softmax(η), η ~ Normal(μ, Σ), which this ADR implements in the forward
  direction.
- ADR [0034](0034-stm-blockwise-unit-diagonal-correlation-sigma.md) — the
  block-wise unit-diagonal correlation Σ M-step and export to `correlation.json`.
- ADR [0028](0028-dashboard-conditioned-dirichlet-prior.md) — the Dirichlet
  mean-match approximation, now superseded for STM and retained for non-STM.
- ADR [0031](0031-stm-k1-reference-topic-parameterization.md) — the reference
  topic parameterization (η pinned to 0).
- ADR [0026](0026-gated-svi-variational-e-step-infrastructure.md) — gated SVI
  infrastructure and per-group prior selection.
- insight [0028](../insights/0028-dirichlet-vs-logistic-normal-rare-phenotype-recovery-gated-lda-is-plda.md)
  — the generative-process difference between Dirichlet and logistic-normal.
