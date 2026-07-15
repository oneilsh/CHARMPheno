# 0036 — In the gated setting a free Σ variance runs away at FIT time (even with reference + spectral) but a frozen-β pooled scale is bounded at EXPORT; the generative scale must be decoupled from the fitting prior

**Date:** 2026-07-03
**Topic:** stm | svi | conditioning | generation | gating | diagnostics
**Status:** Confirmed (exp 0032 real-cohort runaway; synthetic β-frozen bound; pooled export scale shipped ADR 0036 addendum)

Insight [0030](0030-spectral-init-closes-stm-sigma-blowup-on-real-data.md) established that
reference-topic + dense spectral init at σ_init=1 keeps Σ **bounded and proper** (~7.56) on
real data — but on a **non-gated** cohort (exp 0015). Insight
[0033](0033-gated-fullcov-variance-runaway-is-an-init-identifiability-failure.md) established
that the **gated** variance runaway is driven by rare-but-coherent, **document-scarce**
minority topics, which no initialization can fix ("no init adds documents to a rare
phenotype"). This insight resolves the tension between those two — and records the
consequence for the generative simulator.

## Finding 1 — estimating a free Σ diagonal at FIT time runs away in the gated setting, even with reference + spectral on

exp 0032 (population_cancer, K=60 gated, `estimate_sigma_diagonal: true`, all of exp 0028's
stabilizers — reference topic, dense spectral, σ_init=1) tested the hypothesis that
insight 0030's non-gated bound would transfer to the gated setting. **It did not.** Σ
estimated cleanly for ~45 iterations (`Σ_var[min=1 max=3.76]`, `Σ_eig[min=0.373 max=6.11]`,
|Γ| max 1.63 — bounded, PD, climbing from 1 toward the natural scale), then a
low-document-count topic ran away: iter 124 `Σ_var[max=1.82e8]`, `Σ_eig[max=1.83e8]`, |Γ|
max 7.95, ELBO −1.48e6 → −2.48e6. The runaway rode on a background topic with **effective
sample size ≈ 15** — exactly insight 0033's "weakly-identified, document-scarce" ingredient.

So the insight-0030 bound is a **non-gated** property. Gating (the project's core method
for rare-subgroup discovery) necessarily creates document-scarce minority topics, and a
**free** prior variance on those runs away regardless of init quality. This closes the
question exp 0032 posed: ADR 0034's unit-diagonal pin is not overcaution — a free per-topic
variance at fit time is not viable in the gated setting. It also completes the falsification
record alongside the variance-prior/anchor attempts (exps 0022/0024, insight 0032): **no
fit-time mechanism — free diagonal, inverse-Wishart anchor, or diagonal shrink — controls
the gated variance; only pinning (the ν→∞/scale-1 limit) does.**

## Finding 2 — a frozen-β, pooled, single scale is bounded at EXPORT — the runaway is a fit-time co-adaptation phenomenon

The fit-time runaway is a feedback loop between three co-adapting quantities: a topic's β
collapses → its η saturates → its free variance inflates → weaker regularization → β
collapses further. Freezing β (and R) breaks the loop. A synthetic β-frozen check confirms
it: iterating a per-doc E-step under a growing prior with β held fixed, the variance
converges (to ~3.4–4.6 depending on regime) instead of diverging — including with a planted
document-scarce topic, where the fit-time version blows up. The runaway is therefore
specifically a **fit-time (co-adaptation)** phenomenon, not a property of estimating a
variance per se.

This licenses **estimating the generative scale at EXPORT**, with the converged β and R
frozen, rather than at fit time. Two further guards make it robust: (a) a **single pooled**
scalar (not a per-topic free diagonal) means one document-scarce topic's noise is averaged
against every other topic's and cannot run away with the estimate; (b) it is a law-of-total-
variance EM, Var(η) = Var_d(E[η|d]) + E_d(Var[η|d]), pooled over free observed topics
(`corpus_eta_scale_gated`). On exp 0028 it converged to **c = 3.67** in ~5 iterations,
bounded, and generation with Σ_gen = c·R correctly identifies a 2-code type-2-diabetes
seed's phenotype (which unit variance did not). It under-corrects modestly (Laplace
posterior-variance bias — the converged c is a conservative estimate of the true generative
scale).

## Consequence — decouple the generation scale from the fitting prior

The fitting prior and the generation covariance are two different needs. The **gated fit**
requires a unit diagonal for stability (Findings above + ADR 0034). The **generative
simulator** requires a non-unit scale. Conflating them forces a choice between a stable fit
with over-diffuse patients (unit diagonal) and coherent patients with a runaway fit (free
diagonal). **Decoupling** — a unit-diagonal fit plus a separately-estimated, frozen-β,
export-time generative scale — satisfies both. This is the shipped architecture (ADR 0036
addendum, `eta_scale` field, Σ_gen = c·R). The remaining open question is whether the
export estimate can be made less conservative (closer to the true scale) data-drivenly; see
`docs/superpowers/specs/2026-07-03-stm-generative-variance-scale-open-problem.md`.

## What this does NOT claim

- It does not claim c = 3.67 is the true generative scale — it is a conservative (Laplace-
  biased-low) estimate; the true natural scale on comparable non-gated data was ~7.6 (exp
  0015).
- It does not reopen a fit-time free variance or variance-prior — those remain falsified in
  the gated setting (exps 0022/0024/0032).

**Related:** insight 0030 (non-gated bound), insight 0033 (gated runaway = document scarcity),
insight 0032 (variance-prior falsification), ADR 0034 (unit-diagonal pin), ADR 0036 +
addendum (record-completion + pooled eta_scale generative scale), exp 0032 (this runaway),
exp 0028 (production cohort).
