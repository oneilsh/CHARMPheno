# 0045 — The per-document concentration heterogeneity is GENUINE cross-topic structure, not within-document burstiness: the dedup gate (spread survives deduplication, ordering preserved, peakiness uncorrelated with repeat-rate) rules out the likelihood-side confound, so the fix is prior-side (a per-document scale / multivariate-t), not a burstiness-aware emission

**Date:** 2026-07-10
**Topic:** stm | concentration | misspecification | diagnostics | burstiness
**Status:** Confirmed (exp 0047 population_cancer, `concentration_heterogeneity.json`; general diagnostic `spark_vi.eval.topic.concentration_heterogeneity` + `corpus_concentration_heterogeneity_rdd`)

Insight [0044](0044-marginalized-heldout-scale-is-a-lowdim-fix-realcorpus-drift-is-concentration-heterogeneity.md)
isolated a genuine residual holdout-fraction drift in the generative scale and attributed it to
**per-document concentration heterogeneity** (documents genuinely vary in how peaked their topic
mix is). But there is a confound that reads the same way under every model in this family:
**within-document token burstiness** — a document that repeats a few codes many times gives the
multinomial many pseudo-independent votes for one topic, so it *infers* as highly concentrated
when the true behavior is "repeated one token," not "used one topic." Genuine heterogeneity wants
a prior-side fix (a per-document scale, multivariate-t); burstiness wants a likelihood-side fix
(Dirichlet-compound-multinomial / Pólya-urn emission, Madsen et al. 2005). The dedup gate
distinguishes them; this insight records that it came back **genuine**.

## The diagnostic

For each document, infer θ under the fitted gated STM on RAW counts and on DEDUP'd counts (each
token capped at 1), and compare across the corpus. Emitted metrics (no thresholds/verdicts baked
in — general library): `spread_ratio_top_mass` = std(top_mass_dedup)/std(top_mass_raw);
`rank_corr_top_mass` = Spearman(top_mass_raw, top_mass_dedup); `burstiness_corr_top_mass` =
Pearson(top_mass_raw, repeat_fraction). The mean concentration necessarily drops under dedup
(shorter documents — a length confound), so the SPREAD, the RANK correlation, and the
concentration-vs-repeat-rate correlation are the signals, not the mean shift.

## Result (n = 2441, 5% sample, inference scale c = 4.6)

| metric | value | reading |
|---|---|---|
| spread_ratio_top_mass | **0.98** | the concentration spread is unchanged by dedup (std 0.176 → 0.173) |
| rank_corr_top_mass | **0.86** | documents keep their relative peakiness ordering after dedup |
| burstiness_corr_top_mass | **0.009** | per-document peakiness is uncorrelated with repeat-rate |

All three point the same way. For contrast, if apparent concentration were a burstiness artifact,
dedup would collapse the spread (spread_ratio → 0), scramble the ordering (rank_corr → 0), and the
raw peakiness would track repeat_fraction (burstiness_corr → high) — the opposite of all three.

Burstiness genuinely EXISTS in the corpus — `repeat_fraction` median 0.29, p90 0.52 (patients do
repeat codes) — and dedup does lower mean concentration (top_mass p50 0.30 → 0.25, eff_topics p50
6.8 → 8.9), exactly the length effect. But the repeats are **orthogonal** to the concentration
heterogeneity: they exist without explaining it.

## Consequence

- The residual scale drift (insight 0044) is real per-document concentration heterogeneity, now
  confirmed against the burstiness confound. **The fix is prior-side: a per-document scale
  s_d (η_d ~ Normal(μ_d, s_d·c·R), inverse-gamma s_d ⇒ multivariate-t), calibrated (c, ν) per
  corpus** — NOT a burstiness-aware emission for THIS purpose.
- The Dirichlet-compound-multinomial / Pólya-urn branch retires **with evidence** rather than by
  fiat — it is not the fix for the concentration heterogeneity. (It could still model the repeats
  themselves someday; that is a separate question the gate does not close.)
- This is the gate Fable's sequence was built to pass: dedup gate → (c, ν) t-prior fit → residual
  per-topic decomposition + adequacy check. The verdict routes to the t-prior.

## What this does NOT claim

- It does not claim burstiness is absent — it is present (repeat_fraction median 0.29); it claims
  burstiness does not DRIVE the concentration heterogeneity (burstiness_corr ≈ 0).
- It does not claim DCM is never worth modeling — only that it is not the fix for this drift.
- It is one corpus (population_cancer) at one sample/scale; the diagnostic is general
  (`concentration_heterogeneity.py`, model-agnostic, no baked-in verdict) so it is re-runnable to
  test whether the verdict holds elsewhere.

**Related:** insight 0044 (the heterogeneity reframe this gates), 0037/0038 (held-out-LL scale
lineage), 0033 (why per-document s_d is runaway-immune where per-topic variance was not — s_d is
the best-identified latent). Diagnostic code: `spark_vi/eval/topic/concentration_heterogeneity.py`,
`spark_vi/mllib/topic/stm.py::corpus_concentration_heterogeneity_{gated,rdd}`; driver flag
`BUILD_CONCENTRATION_HETEROGENEITY_DIAGNOSTIC` (branch stm, origin 0b79701). Burstiness emission
reference: Madsen, Kauchak & Elkan 2005 ("Modeling word burstiness", ICML); Doyle & Elkan (DCM-LDA).
Project memory: `project_concentration_scale_thread`.
