# Hand-off: faithful STM sampling in the dashboard client (ADR 0028 Alternative B)

**Date:** 2026-06-25
**Status:** Hand-off / spec stub — server prerequisite done; client work not started
**Owner:** dashboard front-end
**Motivation:** [ADR 0028](../../decisions/0028-dashboard-conditioned-dirichlet-prior.md) (Alternative B), insight [0028](../../insights/0028-dirichlet-vs-logistic-normal-rare-phenotype-recovery-gated-lda-is-plda.md)

## Goal

The dashboard's client-side samplers (`generateCohort`, `runSimulator`) draw the
document-topic vector θ from a **Dirichlet** prior moment-matched to the
conditioned mean. For **STM** bundles that is an approximation: STM's true prior
is **logistic-normal** (η ~ Normal(Γᵀx, Σ), θ = softmax(η)). Per insight 0028
the Dirichlet is more vertex-seeking, so STM panels currently render **more
peaked / sparser** per document than a true STM draw. Replace the Dirichlet draw
with a faithful logistic-normal draw **for STM bundles only**, keeping the
Dirichlet path unchanged for LDA / gated-LDA (PLDA), where it is exact.

## Already done (server side — no client coupling)

`model.json` now carries the per-topic diagonal Σ under **`"sigma"`** (length K,
topic-aligned with `alpha`/`beta` rows, already subset by k-anon `kept`).
Present **iff** the bundle is STM; absent for LDA/HDP. So `model.sigma`'s
presence is the "use the STM sampler" signal. (Commit `a0d10f6`;
`write_model_and_vocab_bundles(..., sigma=export.sigma)`, `DashboardExport.sigma`,
`model_adapter.adapt_stm`.)

## Client work (this hand-off)

1. **`types.ts`** — `Model.sigma?: number[]` (optional; length K).
2. **`bundle.ts`** loader — pass `sigma` through.
3. **`sampling.ts`** — two small primitives (a standard normal is *already*
   generated inside `sampleGamma` via the Marsaglia polar step — extract it):
   - `sampleStandardNormal(rng): number`
   - `sampleLogisticNormal(etaMean: number[], sigmaDiag: number[], rng): number[]`
     = for each k, `etaMean[k] + Math.sqrt(sigmaDiag[k]) * sampleStandardNormal(rng)`,
     then softmax over the result.
4. **`covariate.ts`** — a pre-softmax `covariateEta(effects, x) = Γᵀx` (factor it
   out of `covariatePrevalence`, which currently softmaxes), and a gated variant
   mirroring `covariatePrevalenceGated`'s mask-before-softmax (mask out-of-group
   foreground topics; mirror ADR 0028's ~1e-12 floor so vector length is stable).
5. **Conditioned prior** — a `conditionedEta(...)` parallel to `conditionedAlpha`,
   using the same four-quadrant axis logic. For STM the conditioned *mean* is the
   (gated-masked) `covariateEta`; Σ is the bundle's `model.sigma`.
6. **`cohort.ts` (the `sampleDirichlet(model.alpha, rng)` call) + `runSamples.ts`**
   — branch on `model.sigma`:
   - present (STM): `theta = sampleLogisticNormal(conditionedEta, model.sigma, rng)`
   - absent (LDA / PLDA): keep `sampleDirichlet(conditionedAlpha, rng)` as-is.
   Everything downstream (Poisson code count, categorical token draws, E-step,
   neighbors) is unchanged.
7. **Tests** (`sampling.test.ts`): logistic-normal mean recovery over many draws;
   the STM-vs-Dirichlet dispatch on a bundle with/without `sigma`.

## Math

η_k = (Γᵀx)_k + √(Σ_k)·N(0,1);  θ = softmax(η). Gated: mask η to the allowed
topic set (background ∪ the conditioned group's foreground block) before
softmax, so out-of-group foreground topics carry ~0 mass — mirroring
`covariatePrevalenceGated` and the engine's gated-block pin (ADR 0026).

## Coordination

`cohort.ts`, `store.ts`, `covariate.ts` are the **active conditioning surface**
(the ADR-0028 `conditionedAlpha` / four-quadrant reader is in flight). Sequence
this **after / alongside** that landing so `conditionedEta` mirrors
`conditionedAlpha` rather than racing it. The `sampleDirichlet` path must remain
intact — this is purely additive (a branch keyed on `model.sigma`), so PLDA
(Dirichlet, faithful) is unaffected.

## Scope

Small primitives (normals are nearly free) + moderate integration (the dispatch
+ `conditionedEta` + types/loader). Estimate ~half a day once the conditioning
helpers it mirrors have landed.
