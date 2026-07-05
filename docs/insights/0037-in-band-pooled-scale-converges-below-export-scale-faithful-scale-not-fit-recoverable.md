# 0037 — The in-band pooled generative scale (A1) converges BELOW the frozen-β export scale, and both under-concentrate vs the real per-patient distribution; the faithful generative scale is not fit-recoverable

**Date:** 2026-07-05
**Topic:** stm | svi | generation | gating | concentration | diagnostics
**Status:** Confirmed (exp 0033 STM+estimate_global_scale; exp 0034 LDA α-opt real-corpus reference; both on population_cancer)

Insight [0036](0036-gated-free-variance-runs-away-at-fit-but-not-at-export.md) established
that a free Σ variance runs away at FIT time in the gated setting but a frozen-β pooled scale
is bounded at EXPORT (c = 3.67 on exp 0028), and framed an open question: can a bounded scale
be recovered IN-BAND at fit time (the "A1 global softmax-temperature", Σ = τ²·R, one pooled
damped scalar)? This insight answers it, and — via a real-corpus prior-family comparison —
locates the true generative scale relative to every fit-anchored estimate.

## Finding 1 — A1 is stable in-band: pooling + damping hold, no runaway

exp 0033 (population_cancer gated STM, `estimate_global_scale: true`,
`global_scale_step_cap: 1.2`) fit cleanly to iter 200: `Σ_var[min=2.36 max=2.36]` (the exact
uniform-diagonal τ² signature), ELBO improved to −1.461e6, `maxvar[topic=0 ess=19]` bounded,
40 background + 20 cancer topics all coherent (cancer sub-phenotypes recovered: melanoma/skin,
prostate, lymphoma, AML/CML, lung, pancreas/liver, kidney, myeloproliferative; NPMI background
mean +0.18, cancer +0.20). No single topic self-amplified — the pooled scalar averages one
document-scarce topic's noise against every other topic's, exactly as designed. This is the
in-band counterpart to insight 0036's export-time bound: **a single pooled, damped scale is
runaway-safe at fit time**, unlike the per-topic free diagonal (exp 0032, insight 0036), which
blew up to 1.8e8.

## Finding 2 — but the in-band scale converges BELOW the frozen-β export scale

A1 converged to **τ² = 2.36**, which is LOWER than the export estimate (c = 3.67 on the same
cohort) — the opposite of the hope that estimating in-band (without freezing β) would avoid
the export's conservative bias. The scale ladder:

| scale | value | provenance |
|---|---|---|
| unit-diagonal (ADR 0034) | 1.0 | pinned for fitting stability; over-diffuse |
| **A1 in-band (exp 0033)** | **2.36** | pooled damped τ² estimated at fit time |
| frozen-β export (insight 0036) | 3.67 | law-of-total-variance EM, β/R frozen |
| plant-recover faithful band (diagnostic) | ~5–7 | scale at which STM inference is unbiased |
| non-gated natural scale (insight 0030) | ~7.6 | exp 0015, fully supported topics |

The mechanism: **in-band, β co-adapts to whatever Σ is.** A small Σ tightens the prior → the
per-doc MAP η̂ is shrunk toward Γx → the between-doc residuals stay small → τ²'s target stays
small → a low self-consistent fixed point. Freezing β (the export path) removes the downward
pull, so the export estimate is HIGHER. So the two data-driven estimators bracket the same
compression from different sides, and neither reaches the faithful band.

## Finding 3 — the real per-patient distribution is peakier than any fit-anchored scale

exp 0034 (plain LDA, `optimize_doc_concentration`, on the SAME population_cancer corpus,
non-gated) is the prior-family reference. α optimized to a small value (mean ≈ 0.018), and the
per-document concentration readout was **top_mass p50 = 0.513, eff_topics p50 = 2.8**, versus
STM-A1's **top_mass p50 = 0.269, eff_topics p50 = 8.5**. Leading with top_mass (the clean,
gating-independent comparison — eff_topics is confounded by the gated STM's allowed-set
support): **LDA patients are ~2× more peaked on their top topic than STM-A1's.**

LDA α-optimization on 44-token documents is a HOT (over-concentrated) reading — mean-field
variational Bayes under-estimates posterior spread, biasing α̂ downward — so 0.513 is an upper
bound on the true per-patient concentration. But even discounted for that bias, it sits well
above A1's 0.269. Two independent lines now agree the faithful scale is HIGHER than any
fit-anchored estimate: (a) the plant-and-recover diagnostic (STM recovers the true planted
concentration only at Σ ≈ 5–7); (b) this real-corpus LDA reference (patients are genuinely
peaky). A1's 2.36 (and the export's 3.67) leave patients too diffuse.

## Consequence — the faithful generative scale is not fit-recoverable; calibrate it externally

Every fit-anchored estimator — unit pin (1), in-band A1 (2.36), frozen-β export (3.67) —
inherits the MAP η-compression of the fit and lands below where a faithful generative scale
would sit. The compression is a property of the MAP/Laplace inference, not of any one
estimator, so no amount of in-band-vs-export cleverness escapes it. **The faithful generative
scale must be calibrated to an external target, NOT read off the fit.**

What we do NOT yet have is a gold standard for the TRUE per-document concentration. The two
pieces of evidence pointing "higher" are each limited: the plant-recover diagnostic measures
STM inference BIAS on SYNTHETIC data with a known planted concentration (STM is unbiased at
Σ ≈ 5–7 there) — that is an inference-self-consistency result, NOT a measurement that real
documents sit at 5–7; and the LDA reference (top_mass 0.513) is a HOT upper bound. So the
honest statement is a BRACKET on the real per-document top_mass — roughly [0.27 (STM,
known-diffuse-biased at low scale), 0.51 (LDA, known-peaky-biased)] — with no measurement
pinning the value inside it. The single non-gated fit at ~7.6 (insight 0030) is weak evidence
and should not be treated as the target.

The way to pin it (open, proposed): a prior-family-agnostic, ground-truth-free calibration —
held-out within-document token prediction. Hold out a random subset of each document's tokens,
infer θ from the visible subset with β fixed, score predictive likelihood on the held-out
tokens, and sweep the concentration knob (Σ scale for STM, α for LDA); the interior maximum is
the data's true effective concentration. Until that is run, any shipped generative scale is a
choice within the bracket, not a calibrated answer. This does NOT reopen a fit-time free
variance (still falsified, exps 0022/0024/0032); A1 remains valuable as the stable,
runaway-safe fit-time parameterization and as the lower bracket on the compression. See the
follow-up update `docs/superpowers/specs/2026-07-05-a1-results-and-concentration-gold-standard-update.md`.

## What this does NOT claim

- It does not claim A1 failed — A1 met its stability goal (Finding 1); it under-corrects for
  generation (Finding 2), which is informative, not a defect.
- It does not claim LDA's 0.513 is the true concentration — it is a hot upper bound (Finding 3).
- It does not claim a single scalar is the final answer — the diagnostic (finding 4 there)
  notes STM is faithful only in a band, so a corpus with a spread of true concentrations may
  need a richer (e.g. per-block) scale; that remains open.

**Related:** insight 0036 (export bound + the fit-vs-export decoupling), insight 0030 (non-gated
natural scale ~7.6), insight 0033 (gated runaway = document scarcity), ADR 0034 (unit-diagonal
pin), ADR 0036 addendum (export `eta_scale`), exps 0032 (per-topic runaway) / 0033 (A1) / 0034
(LDA reference), diagnostic report
`docs/superpowers/specs/2026-07-04-prior-family-vs-scale-concentration-diagnostic.md`.
The concentration metric (top_mass, effective number of topics = inverse-Simpson / Hill order
2) follows Hill 1973 and Jost 2006.
