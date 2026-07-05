---
id: 33
slug: stm-population-cancer-global-scale
status: done
model_class: stm
cohort: population_cancer
cohort_def: population_cancer
prior_obs_days: 0
person_mod: 4
doc_unit: patient_cohort
doc_min_length: 10
covariate_formula: "~ C(sex) + age"
categorical_cols: [sex]
continuous_cols: [age]
known_sex_only: true
random_seed: 42
cache_uri: hdfs:///user/dataproc/charm/covariates_cache
K: 60
background_k: 40
foreground: "cancer:20"
group_var: source_cohort
max_iter: 200
subsampling_rate: 0.1
tau0: 128
kappa: 0.7
sigma_init: 1.0
reference_topic: true
spectral_init: true
spectral_method: dense
min_pair_support: 10
estimate_global_scale: true
global_scale_step_cap: 1.2
---

# Experiment 0033 — Population-cancer gated STM with a pooled global variance scale (A1)

## Goal

Re-fit exp 0028's config (population background + cancer foreground, gated STM)
with the new **`estimate_global_scale`** Σ parameterization: Σ = τ²·R, a single
POOLED variance scalar τ² estimated in-band at fit time on top of the block-wise
unit-diagonal correlation R (ADR 0034). This is the logistic-normal analog of
LDA's α-optimization — the "global softmax-temperature" (A1) from the
prior-family-vs-scale diagnostic. It tests whether the natural η-scale can be
recovered **at fit time**, stably (bounded, no runaway), so that generated
patients are concentrated without a post-hoc or export-time rescale.

## Why

Two prior fit-time attempts to recover the η-variance scale in the gated setting
FAILED: a free full Σ diagonal and the per-topic `estimate_sigma_diagonal`
(exp [0032](0032-stm-population-cancer-estimated-sigma-diagonal.md)) both ran
away — a document-scarce minority topic (effective sample size ≈ 15) inflated
its own free variance in a softmax-saturation feedback loop (insight
[0033](../insights/0033-gated-fullcov-variance-runaway-is-an-init-identifiability-failure.md),
insight [0036](../insights/0036-gated-free-variance-runs-away-at-fit-but-not-at-export.md)).
The shipped workaround decouples the scale to EXPORT time (a single pooled scalar
c with β frozen, `eta_scale`, ADR 0036 addendum) — which is runaway-safe but
under-corrects (Laplace posterior-variance bias, c=3.67 on exp 0028 vs the
natural ~7.6).

A1 asks whether the SAME runaway-safety argument that makes the export scalar
safe transfers **in-band**: a single pooled τ² (not a per-topic free diagonal)
averages one document-scarce topic's noise against every other topic's, so no
single topic can drag the estimate; and because it is estimated during the fit
(not with β frozen post-hoc), it does not incur the frozen-β export
under-correction. The one instability pooling alone would NOT break is a global
τ–β sharpening ratchet (τ↑ → documents inferred peakier → β sharper → η̂ modes
spread → τ↑); `global_scale_step_cap` (default 1.2) is a trust-region damping
guard that bounds the per-iteration multiplicative change of τ² so the ratchet
cannot run. See the diagnostic report
`docs/superpowers/specs/2026-07-04-prior-family-vs-scale-concentration-diagnostic.md`.

## Configuration

Identical to exp 0028 (population background, cancer foreground, K=60 = 40
background + 20 foreground, `~ C(sex) + age`, dense spectral init, reference
topic, `min_pair_support: 10`, σ_init=1.0), with two changes:

| Field | Value | Note |
|---|---|---|
| `estimate_global_scale` | true | pooled global variance scale Σ = τ²·R (A1) instead of the unit-diagonal pin |
| `global_scale_step_cap` | 1.2 | trust-region cap on the per-iteration multiplicative change of τ² (damping guard) |

`estimate_global_scale` is mutually exclusive with `estimate_sigma_diagonal`
(the engine raises if both are set); this run uses ONLY the pooled scale. All
other fields are unchanged from exp 0028.

## Success criteria

- **Bounded, stable τ².** The fit completes without the softmax-saturation
  runaway that killed exp 0032. Because Σ = τ²·R has a uniform diagonal, the
  existing per-iteration `Σ_var[min max]` log reads min ≈ max ≈ τ² — watch that
  it climbs from σ_init=1 toward a bounded value and STAYS there (the damping cap
  makes each step at most 1.2× the previous). A converged τ² near the natural
  η-scale (order ~5–8, cf. the non-gated ~7.6 of insight 0030 and the
  export-time c=3.67) is the target; a τ² that diverges or pins at 1 is a failure.
- **Correlation structure preserved.** Off-diagonal correlations
  (Σ_ij / sqrt(Σ_ii Σ_jj)) read comparably to exp 0028's unit-diagonal fit —
  cancer sub-phenotype correlations intact, no new spurious cross-block signal.
- **More concentrated patients than the unit baseline.** The new
  `concentration_readout` in the checkpoint metadata (per-document top_mass +
  eff_topics percentiles, mode-based) shifts toward higher top_mass / lower
  eff_topics than exp 0028 (unit-diagonal) — i.e. patients are less "rainbow",
  more coherent — WITHOUT a post-hoc rescale. Compare the readout against exp
  0028 (unit) and, if run, the LDA arm (see the diagnostic follow-up).

## Result — A1 STABLE, τ² = 2.36 (under-corrects)

Fit completed to iter 200 with NO runaway: `Σ_var[min=2.36 max=2.36]` (uniform
diagonal = τ² exactly, as designed), ELBO improved to −1.461e6,
`maxvar[topic=0 ess=19]` bounded, 40 background + 20 cancer topics all coherent
(cancer sub-phenotypes recovered: melanoma/skin, prostate, lymphoma, AML/CML,
lung, pancreas/liver, kidney, myeloproliferative; NPMI background mean +0.18,
cancer +0.20). The pooling + damping cap did their job — the primary stability
criterion is met, in sharp contrast to exp 0032's per-topic runaway (1.8e8).

τ² converged to **2.36** — bounded and data-driven, but LOWER than the frozen-β
export estimate (c = 3.67) and well below the diagnostic's faithful band (~5–7):
in-band, β co-adapts to the small Σ, pulling τ² to a low self-consistent fixed
point. The concentration readout (per-doc, mode-θ) was **top_mass p50 = 0.269,
eff_topics p50 = 8.5** (n = 47,838) — more concentrated than the unit baseline
but ~2× more diffuse than the LDA reference (exp 0034: top_mass 0.513). A1 is
the stable, runaway-safe fit-time parameterization but under-corrects for
generation. See insight
[0037](../insights/0037-in-band-pooled-scale-converges-below-export-scale-faithful-scale-not-fit-recoverable.md).

## Related

Builds on exp 0028 (population-cancer gated STM, config source), exp 0032 (the
per-topic `estimate_sigma_diagonal` variant that ran away — the failure A1's
pooling is designed to avoid), insight 0036 (fit-time free variance runs away but
a pooled scale is bounded), insight 0030 (non-gated natural η-scale ~7.6), ADR
0034 (block-wise unit-diagonal Σ — the pin this run's pooled scale sits on top
of), ADR 0036 addendum (export-time pooled `eta_scale`, the decoupled scale this
in-band variant is the alternative to). Diagnostic:
`docs/superpowers/specs/2026-07-04-prior-family-vs-scale-concentration-diagnostic.md`.
