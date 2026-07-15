# Conditioned-Coverage Bubble Plot — Design

**Date:** 2026-07-06
**Status:** Approved (design); ready for implementation plan
**Branch:** stm
**Supersedes on the atlas:** the Task-6b predictive-gain bubble encoding (commit 724235a)
**Realizes:** the atlas portion of the parked conditioning redesign (ADR 0028), now
unblocked by the generative scale/correlation work (ADR 0034 unit-diagonal Σ, ADR 0036
pooled eta_scale c*).

---

## Motivation

The Phenotype Atlas bubble size currently encodes the predictive-gain metric
(`mean_gain`, Task 6b). Analysis of the real exp-0028 bundle showed that metric collapses
onto a single axis — topic distinctiveness/rarity relative to the corpus background
(corr(mean_gain, prevalence) = −0.81; corr(presence, mean_gain) = +0.91) — and that its
signal for common topics sits inside the permutation null band (std ≈ 2.0 nats; common
topics score 0.6–2.7 nats). It is a specificity detector, not a general per-topic
importance, and it is not prevalence.

The intended readout is the **conditioned-coverage bubble plot** that predates the
Σ-blowup detour: bubble size = the fraction of patients who express a phenotype above a
threshold, with the demographic covariates (age, sex) resizing bubbles up and down, the
gating group (e.g. cancer) as a separate color, and rarer foreground phenotypes rendering
smaller as expected. That plot needs a faithful generative scale and correlation to sample
patients; those are now solved (c* = 4.6, block-wise correlation R), so the plot can be
restored on a correct footing.

A separate, long-standing terminology problem is corrected here: the field named
`corpus_prevalence` is `theta.mean(axis=0)` (mean topic mass), not epidemiological
prevalence. User-facing copy will stop calling it "prevalence."

---

## What already exists (reuse, do not rebuild)

The generative conditioning stack is built and tested:

- `conditioning/logisticNormal.ts` — `sampleConditionedTheta({...})`: the faithful STM
  forward draw θ = softmax(η), η ~ Normal(Γᵀx, Σ), restricted to the gated free rows
  (background ∪ selected group), with `buildGenerativeSigma` assembling Σ = eta_scale·R
  (c*·R) over those rows. Cholesky via `linalg.ts` (`choleskyPD`).
- `conditioning/marginalSampler.ts` — `sampleMarginalCovariates` / `sampleMarginalGroup`:
  per-patient marginal draws of covariates (triangular for continuous, Stein & Keblis 2009)
  and group (from `group_proportions` incl. background-only), for the marginal baseline.
- `cohort.ts` — `generateCohort(input)`: samples a whole cohort of patient θ via
  `sampleConditionedTheta`, in two modes that are exactly the two states this plot needs:
  - `'sample'`: each patient gets its own marginal covariate/group draw → the corpus's
    natural mix → **the baseline (no covariates active)**.
  - `'set'`: every patient shares one design vector + group → **the covariate-active state**.
- `covariate.ts` — `buildDesignVector`, `allowedMaskForGroup`.
- Bundle already ships everything: Γ (`covariate_effects.json`), R + c* (`correlation.json`:
  `R`, `eta_scale = 4.6`), gating (`gating.json`: `topic_blocks`, `group_proportions`,
  `background_only_proportion`), schema (`covariate_schema.json`).

The generative math is done. The gap is that the atlas bubble size does not use it: it
reads `prevalenceReader`, which is inconsistent (see below).

---

## The bug this fixes

`prevalenceReader` ([store.ts](../../../dashboard/src/lib/store.ts)) switches the *quantity*
bubble size encodes depending on covariate mode:

- **Covariates off:** `fractionAboveTau` — fraction of patients with θ_k > τ (a coverage).
- **Covariates on:** `covariatePrevalence` = `softmax(Γᵀx)` — a single predicted θ vector,
  ungated, on a different scale, and **not a fraction of patients at all**.

Turning covariates on silently changes the meaning and scale of bubble size (the band-aid
commit f98b140, "absolute bubble scale in covariate mode", is evidence of the resulting
jump). The fix is to make both states the *same* quantity, sampled from the generative model.

---

## Design

### The quantity

Bubble size = **patient coverage** for phenotype k:

    coverage_k = ( # sampled patients with θ_k > τ ) / ( # sampled patients )

Patients are drawn from the faithful generative model (η ~ Normal(Γᵀx, c*·R) over the gated
free rows, θ = softmax over those rows) via the existing `generateCohort`. τ defaults to
**0.01** (down from 0.02: the faithful, non-peaky prior gives longer-tailed θ than the
peaky Dirichlet, so fewer patients clear a 2% bar — 1% keeps meaningful expression visible).

The same computation runs in both states; covariates only change the sampling profile, so
bubbles **modulate smoothly** instead of switching units:

- **Baseline (covariates off):** cohort sampled in `'sample'` mode (per-patient marginal
  covariate *and* group draws — the natural corpus mix). Model-implied marginal coverage.
- **Covariate active:** cohort sampled with the **demographic design vector fixed** (from the
  sliders) but **group still drawn per-patient from its marginal**. This keeps foreground
  (cancer) phenotypes present — they get coverage from the sampled cancer patients, so cancer
  bubbles stay visible but smaller, and age/sex resize every bubble. **Group is never fixed by
  the atlas** (there is no group selector — out of scope), so cancer bubbles never vanish when
  you move the sliders.

  Note: `generateCohort` is the *wrong* primitive to reuse here — it also computes O(N²)
  cosine neighbors, code bags, and quality buckets, far too heavy to recompute for N≈1500 on
  a covariate change, and its `'set'`/`'sample'` modes fix or marginalize *both* covariates
  and group (never the mix we need). Instead, a focused `sampleThetaCohort` reuses the same
  tested sampler primitives directly — `sampleConditionedTheta` + `sampleMarginalCovariates`
  + `sampleMarginalGroup` + `buildDesignVector` — drawing only θ (no bags/neighbors) and
  controlling the fixed-covariate + per-patient-marginal-group draw in one loop.

**Bubble scale is absolute** in all states (bubble area ∝ actual coverage fraction, one
shared scale). Moving a slider makes a phenotype genuinely grow or shrink (prostate cancer
balloons for males; pregnancy vanishes for older males). Rare phenotypes stay small — that
is the honest signal, not a defect.

### Components

**New — `conditioning/coverage.ts` (TDD):** two focused functions.

    sampleThetaCohort(input: {bundle, active, values, n, seed}): number[][]
    cohortCoverage(thetas: number[][], tau: number, K: number): number[]

`sampleThetaCohort` draws n patient θ from the faithful conditional logistic-normal via the
existing sampler primitives (STM bundles only; caller guards). Group is always a per-patient
marginal draw (foreground phenotypes stay represented); covariates are fixed to `values` when
`active`, else drawn per-patient from the reported marginals. `cohortCoverage` is pure — it
counts θ_k > τ over a pre-sampled cohort. Both are easily tested (determinism under a fixed
seed; τ at a θ value, empty cohort → zeros; covariate-fixed draws shift coverage per Γ).

**New — `atlasThetaCohort` derived store (store.ts):**

Derived from `bundle` + `atlasConditioning`. When the bundle is STM (has `covariateEffects` +
`correlation`), calls `sampleThetaCohort` with **N ≈ 1500** and a **fixed seed** (stable
bubbles — no flicker), in the covariate-active (`active: true`) or marginal (`active: false`)
state. The derivation is **synchronous** (so consumer tests stay synchronous) and depends on
the covariate profile only — **not** on τ, so moving τ recounts the cached cohort without
resampling. Non-STM bundles (HDP/LDA: no Γ/R) yield `null` → the reader falls back to the
empirical histogram path. If slider *drag* proves janky, debounce the covariate-value writes
at the input layer (a follow-up, not a correctness concern — changes settle-commit already).

**Rewire — `coverageReader` (replaces `prevalenceReader`, store.ts):**

    derived([bundle, atlasCoverageCohort, tauThreshold], ...)
      STM bundle:     const cov = cohortCoverage(cohort, τ, K); return p => cov[p.id] ?? 0
      non-STM bundle: return p => fractionAboveTau(p, edges, τ)   // unchanged fallback

The `softmax(Γᵀx)` point-estimate branch is **deleted**. `fractionAboveTau` survives only as
the non-STM fallback. (Note: the θ-histogram bins are 0.02 wide, so the fallback rounds τ to
the histogram grid; the STM path uses continuous sampled θ and resolves τ = 0.01 exactly.)

**TopicMap.svelte:**

- `sizeReader = coverageReader` (revert the Task-6b `hasPredictiveGain ? meanGainReader : reader`
  swap; size is coverage again).
- Cohort color + "cancer smaller" behavior unchanged.
- Tooltip: honest coverage wording ("X% of patients express this above τ"); when covariates
  are active, note the profile. Remove the nats/gain line from the headline tooltip.

**Detail panel (advanced view):**

- Move `mean_gain` to advanced-only, relabeled **"distinctiveness (vs. corpus background)"**,
  shown alongside the null band so a value inside the noise floor is visibly untrustworthy.
- Drop `presence` and `depth` from the headline readout (they are the same rarity axis;
  keep in advanced only if cheap, otherwise remove from UI — estimator code and the Fable
  research thread are unaffected).

**Copy (copy.ts):**

- "prevalence" → **"patient coverage"** / **"% of patients expressing"** throughout the
  atlas and tooltips. `corpus_prevalence` stays as the internal bundle key (no bundle change).

### Data flow

    global conditioning store (age/sex sliders, group, covariateActive)
        └─> atlasCoverageCohort (generateCohort, N≈2000, fixed seed, debounced)
              └─> coverageReader (cohortCoverage · τ)
                    └─> TopicMap bubble size (absolute scale)  +  detail readout

### Performance

N ≈ 2000 patients × softmax(60) per resample is well under a frame; Cholesky depends only on
the free-set (group), not the covariate profile, so it is computed per distinct group (2–3),
not per patient. Debounced sampling + fixed seed keep interaction smooth and bubbles stable.

### Testing

- `coverage.ts`: pure unit tests — hand-built cohorts, τ at/above/below θ values, empty
  cohort, all-below-τ, K/θ-length agreement.
- `store` coverage reader: STM bundle → uses cohort coverage; non-STM bundle → falls back to
  `fractionAboveTau`; τ change recounts without resampling (assert cohort identity/seed
  stability); covariate toggle switches `'sample'` ↔ `'set'`.
- TopicMap: size reads coverage; back-compat — a bundle without predictive_gain still renders;
  a non-STM bundle still sizes by the histogram fallback.
- Reuse existing `logisticNormal.test.ts` / `cohort` coverage — do not re-test the sampler.

### Back-compat

- Non-STM bundles (HDP, plain LDA) keep the empirical `fractionAboveTau` sizing unchanged.
- The bundle format is unchanged — no re-export or cluster re-fit is required for this work
  (all inputs — Γ, R, c*, gating, schema — already ship). The existing
  `dashboard/public/data/population_cancer` bundle drives the new plot as-is.

---

## Out of scope (possible follow-ups)

- Reviving covariate response in the Simulator/Patient tabs beyond the atlas (the rest of
  ADR 0028 / the `conditionedAlpha` work).
- A directional differentiation encoding (hue = male/female/age association) as a *secondary*
  channel on top of coverage size — deferred; coverage + interactive resize already tells the
  covariate story.
- A group selector in the atlas conditioning bar (currently group is sampled from its
  marginal; `'set'`-mode group fix is supported by `generateCohort` if a selector is added).
- Re-exporting the bundle to drop/rename `corpus_prevalence` at the source (kept as internal
  key for now).

## Decisions baked in (from brainstorming)

1. Semantic target = **covariate differentiation**, realized as conditioned coverage (not a
   static per-topic scalar — the differentiation is the interactive resize).
2. Covariates = **demographics only** (age, sex). Group is the color axis, sampled from its
   marginal (its differentiation is structural via gating, already shown by color).
3. Coverage source = **fully generative** in both states (baseline and covariate-active
   sampled from c*·R), for one consistent quantity and smooth modulation.
4. Predictive-gain metric → **advanced-panel stat only**, honestly relabeled with the null
   band shown.
5. Cohort sourcing = **dedicated atlas cohort** driven by the global conditioning bar (not
   the shared Simulator cohort store).
6. Bubble scale = **absolute coverage** (no per-state renormalization).
7. τ default **0.01**.
