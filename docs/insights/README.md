# Modeling & Experimental Insights

This directory captures *learned things about the modeling regime or the data*
that aren't recoverable by reading the code. Insights complement ADRs:

- **ADRs** (`docs/decisions/`) record *why we chose X over Y* — architectural
  and organizational decisions. Forward-looking, prescriptive.
- **Insights** (`docs/insights/`) record *what we observed when we tried
  things* — empirical findings about how models behave on this data, which
  diagnostics actually discriminate, which failure modes are real, which
  hypotheses didn't survive contact with reality. Backward-looking,
  descriptive.

When a new modeling phenomenon, failure mode, or counterintuitive result
emerges from a run, add an insight here with the next four-digit number.
Existing insights can be marked **Refuted by NNNN** in their header when a
later run overturns them — don't delete; the trajectory of what we believed
and when is itself useful.

## Format skeleton

    # NNNN — Short Title
    **Date:** YYYY-MM-DD
    **Topic:** hdp | lda | doc-units | diagnostics | svi | ops | npmi
    **Status:** Observed | Confirmed | Tentative | Refuted by NNNN

    [Narrative body — typically 100–400 words. Subsections OK if useful
    (Observation / Interpretation / Implications), but optional.]

    **Setting context:** One short paragraph naming the run setup that
    revealed this — model, doc-unit, K/T, key hyperparameters that differ
    from recent defaults. Detail level: "trying with very large K values
    revealed X; other settings as in other recent patient-year LDA runs"
    is enough. The point is so a future reader knows whether the
    observation likely generalizes or was specific to that regime.

## Status meanings

- **Observed**: seen in one run; could be regime-specific. Default for new
  insights.
- **Confirmed**: reproduced across ≥2 distinct runs (different seeds, doc
  units, or hyperparameter neighborhoods).
- **Tentative**: noticed but with confounders that prevent ruling out
  alternative explanations.
- **Refuted by NNNN**: a later insight contradicts this one; keep the entry
  but link forward.

## What does NOT go here

- Facts about how the code works → code comments / docstrings.
- Architectural choices → ADRs in `docs/decisions/`.
- Coding conventions or agent-collaboration rules → AGENTS.md.
- Run logs and outputs → not committed; insights distill from runs but
  don't replicate their raw output.

## Index

(Append entries here as new insights are written, newest at top.)

- [0047](0047-the-sigma-eigmin-floor-is-a-mean-field-attenuation-artifact-gibbs-recovers-full-rank.md) — The at-scale Σ eigmin-floor is a MEAN-FIELD ATTENUATION ARTIFACT: on a full-rank well-conditioned synthetic truth (block PR 5.5, eigmin 2.8, cond 5), VI's fitted per-doc logit scatter collapses (PR 2.4, eigmin 2.8e-6, cond 2.5e6) while exact Gibbs with labels pinned recovers the truth (PR 6.7, eigmin 3.6, cond 7). Resolves 0046's open question; extends 0044 from correlation-sign to rank/conditioning/scale. Re-sequences the queue: low-rank Σ=ΛΛᵀ+D (item 3) would model the method not the data → demoted; the condition-on-VI-β Gibbs read-out (item 4) is the fix → promoted (pg-stm, sigma, mean-field, attenuation, rank, gibbs, condition-on-beta)
- [0046](0046-sigma-eigmin-floor-is-statistical-rank-deficiency-not-structural-to-the-gated-parameterization.md) — The at-scale Σ eigmin-at-floor (~1e-8) is NOT structural to the gated stick parameterization (Σ_true is well-conditioned, eigmin = eta_scale·(1−rho_grp), invariant to #groups) — it is statistical rank-deficiency in the fitted block scatter: rank-2-compressed scatter (an attenuation proxy) reproduces the exact 1e-8 MLE floor, while strong correlation alone can't (ρ=0.99 → 0.04). The IW ridge lifts the rank-deficient direction off the floor (2e-3 to 1e-1) — small-block shrinkage, a possible estimator contrast on conditioning the max|Σ| headline missed; low-rank Σ=ΛΛᵀ+D is the matched fix (pg-stm, sigma, eigmin, rank-deficiency, low-rank, inverse-wishart, null-space)
- [0045](0045-free-gibbs-sigma-readout-is-confounded-by-label-switching-not-weak-identification.md) — The unpinned free-Gibbs Σ read-out is confounded by topic label-switching (topics permute across chains, so per-stick R̂ is uninterpretable), NOT weak identification; conditioning on a fixed β pins labels (R̂→1, medians→truth) and the scarce-group scale is then wide-but-stationary under both weak and informative priors (no runaway resurfaces in exact Gibbs — refuting "the cure is only mean-field damping") (pg-stm, gibbs, sigma-readout, label-switching, mcmc-diagnostics, condition-on-beta)
- [0044](0044-meanfield-vi-fails-sigma-correlation-even-when-identified.md) — Mean-field VI reads the WRONG SIGN of the Σ correlation even on a stick-native corpus where Σ is identified and exact Gibbs recovers it; the comorbidity correlation read-out cannot use mean-field VI (needs exact Gibbs or a structured posterior); β/topic content is unaffected (pg-stm, vi, gibbs, sigma-correlation, mean-field)
- [0043](0043-permuted-null-presence-is-better-than-random-not-significantly-positive.md) — Permuted-topic "presence" (presence_vs_null) is a BETTER-THAN-RANDOM test, not a significantly-positive one: a random topic hurts held-out prediction (null-band mean < 0), so "beats the null" is a LOWER bar than "beats zero" — it inflates weak/background topics and rates a zero-gain reference topic 70% present (stm, diagnostics, predictive-gain, calibration)

- [0042](0042-cofit-beta-does-not-reproduce-stm-lda-peakiness-gap.md) — Co-fitting β does NOT reproduce the real STM-vs-LDA peakiness gap (the sign even flips: co-fit LDA is cooler than co-fit STM); LDA's Dirichlet pressure DOES carve a marginally sharper β, but a sharper β does not make patients peakier — so with 0038 (α ruled out) neither inference mechanism explains the real gap, which must be pinned on the real corpus (stm, lda, generation, concentration, beta-learning, calibration)
- [0041](0041-drug-anchored-active-comparator-gated-stm-recovers-drug-specific-structure.md) — A drug-anchored active-comparator gated STM (GLP-1 vs SGLT2i, exp 0044) separates drug-specific comorbidity (SGLT2i→cardiorenal, GLP-1→obesity+GI side effects) from the shared T2DM indication that lands in both arms; the drug-anchor track is a drop-in sibling of the disease track under the gate (stm, gating, cohorts, drug-anchor, pharmacoepi)
- [0040](0040-light-coder-years-are-mostly-routine-with-structured-pockets.md) — Light-coder (5–19-code) years are mostly routine/screening/acute-minor care with real MSK and metabolic pockets, justifying the doc_min_length floor without it being lossless; a foreground block's low NPMI (+0.10 vs +0.19) is the diffuse-short-doc signal, not a defect (stm, gating, cohorts, short-documents, npmi)
- [0039](0039-revisiting-0028-stabilized-gated-stm-recovers-rare-phenotypes.md) — Revisiting 0028: stabilized gated STM recovers rare/minority sub-phenotypes, so 0028's "prior family, not stabilization" verdict was confounded by missing stabilizers + fit-scale (stm, lda, gating, priors, rare-phenotypes, re-examination)
- [0038](0038-heldout-ll-recovers-true-concentration-and-lda-alpha-opt-is-not-hot.md) — Held-out predictive likelihood recovers the true per-document concentration (synthetic-validated, both prior families); LDA α-optimization does not read hot, so the STM-vs-LDA concentration gap is not an α-inference artifact (stm, lda, generation, concentration, calibration)
- [0037](0037-in-band-pooled-scale-converges-below-export-scale-faithful-scale-not-fit-recoverable.md) — The in-band pooled generative scale converges below the frozen-β export scale and both under-concentrate vs real per-patient θ; the faithful generative scale is not fit-recoverable (stm, generation, concentration, diagnostics)
- [0036](0036-gated-free-variance-runs-away-at-fit-but-not-at-export.md) — A free Σ variance runs away at FIT (even with reference + spectral) but a frozen-β pooled scale is bounded at EXPORT; the generative scale must be decoupled from the fitting prior (stm, generation, gating, diagnostics)
- [0035](0035-rare-disease-gated-foreground-recovers-eds-subphenotypes-on-full-population.md) — A rare-disease gated foreground (EDS, ~1k docs / ≈0.5%) recovers clinically faithful sub-phenotypes against a full-population background (stm, gating, cohorts, rare-disease)
- [0034](0034-blockwise-unit-diagonal-fixes-runaway-on-real-data-and-needs-a-correlation-clamp.md) — Block-wise unit-diagonal Σ fixes the variance runaway on real data (exp 27); per-cell standardization needs a −1..1 correlation clamp (stm, gating, conditioning, reporting)
- [0033](0033-gated-fullcov-variance-runaway-is-an-init-identifiability-failure.md) — The gated full-Σ variance runaway is an initialization / identifiability failure, not a thin-minority or model failure; unit-diagonal Σ is structural insurance (stm, conditioning, gating, diagnostics)
- [0032](0032-gated-fullcov-recovers-dementia-subphenotypes-and-exposes-spd-assembly-conditioning.md) — Gated full-covariance STM recovers Alzheimer's vs vascular dementia in the 19% minority arm; the predicted SPD-assembly ill-conditioning appears on real data and is gracefully handled (stm, full-covariance, gating, rare-phenotype, conditioning)
- [0031](0031-scalable-spectral-topic-quality-matches-dense-but-sigma-splits-one-runaway.md) — Scalable (random-projection) spectral init matches dense on topic quality but not Σ — a single dominant topic escapes into the blowup basin (stm, initialization, spectral, sigma)
- [0030](0030-spectral-init-closes-stm-sigma-blowup-on-real-data.md) — On real cancer data, spectral init brings STM's Σ from ~10^10 to ~7.6 and resolves all K topics at the default σ_init=1; the K−1 reference alone does not — the Σ blowup defeats the reference topic itself (stm, priors, svi, initialization, diagnostics, phenotyping)
- [0029](0029-stm-sigma-init-collapse-blowup-missing-stabilizers.md) — Our online STM's σ_init-selected collapse↔Σ-blowup (Σ→10^10) is a missing-stabilizer artifact, not a property of STM; published STM avoids it via spectral init + K−1 reference-topic identifiability + Σ shrinkage (stm, priors, svi, initialization, diagnostics, prior-art)
- [0028](0028-dirichlet-vs-logistic-normal-rare-phenotype-recovery-gated-lda-is-plda.md) — The prior family (Dirichlet vs logistic-normal), not gating, governs rare minority-phenotype recovery; STM collapses where LDA recovers (2x2 + repro); "gated LDA" is Partially Labeled Dirichlet Allocation (PLDA) (stm, lda, gating, priors, phenotyping, prior-art)
- [0027](0027-gated-stm-imbalanced-arms-majority-foreground-collapses.md) — Gated STM on imbalanced arms: the majority arm's foreground collapses into the shared background; the minority arm captures its anchor but not sub-phenotypes; per-block NPMI is non-comparable (stm, gating, covariates, npmi, doc-units, diagnostics)
- [0026](0026-stm-prevalence-gives-prevalence-not-content-fidelity.md) — Prevalence-only STM reproduces LDA's cohort concentration (prior-family-invariant) and gives rare covariate groups prevalence fidelity, not content fidelity (stm, lda, covariates, npmi, diagnostics)
- [0025](0025-min-patient-count-vs-min-df.md) — `min_patient_count` and `min_df` answer different questions (doc-units, diagnostics)
- [0024](0024-labeler-classifier-rules-have-regime-dependent-blind-spots.md) — LLM-classifier rubrics have regime-dependent blind spots: rules that look robust on the development corpus can mis-fire when the feature distribution shifts (ops, diagnostics, labeling)
- [0023](0023-producer-consumer-unit-mismatches-invisible-until-small-scale.md) — Dimensional unit mismatches in producer/consumer pairs are invisible at large input scale; small cohorts expose them (ops, diagnostics)
- [0022](0022-phenotype-vocabulary-refines-past-elbo-plateau.md) — Topic-word concentration on small-α tail topics continues refining well past mass-distribution convergence (lda, svi, diagnostics)
- [0021](0021-cohort-corpora-two-anchor-mass-concentration.md) — Cohort-filtered corpora concentrate 90%+ of mass into two universal-anchor topics regardless of K (lda, doc-units, diagnostics)
- [0020](0020-small-cohort-worker-count-reversal.md) — Below ~1k docs/worker/epoch, fewer Spark workers wins on wall time (ops, svi)
- [0019](0019-lda-large-k-with-full-convergence-gracefully-unused-slots.md) — LDA at large K with full convergence gracefully under-uses excess capacity, no micro-cluster artifacts (lda, doc-units, diagnostics, hdp)
- [0018](0018-full-corpus-plus-threshold-yields-unimodal-positive-npmi.md) — Full-corpus reference + min-pair-count threshold yields unimodal positive NPMI distribution (npmi, diagnostics, hdp)
- [0017](0017-hdp-gamma-sensitivity-is-prior-dominance.md) — HDP γ-sensitivity reveals prior-dominated outcomes (hdp, diagnostics)
- [0016](0016-condition-era-vs-occurrence-not-comparable.md) — condition_era and condition_occurrence runs are not directly comparable (doc-units, npmi, diagnostics)
- [0015](0015-crisp-topics-can-regress-when-k-undersized.md) — Crisp topics can regress in late iters when K is undersized (lda, diagnostics)
- [0014](0014-patient-year-npmi-bimodal-vs-lifetime-unimodal.md) — Patient-year LDA NPMI is bimodal; patient-lifetime is unimodal (lda, doc-units, npmi)
- [0013](0013-spark-scaling-driver-bottleneck.md) — Spark executor count past ~20 doesn't reduce iter time (ops, svi)
- [0012](0012-svi-batch-fraction-vs-iter-count-tradeoff.md) — SVI batch-fraction and iter count must be tuned together (svi)
- [0011](0011-min-doc-length-is-phenotype-vs-noise-tradeoff.md) — min_doc_length is a phenotype-vs-noise trade-off (doc-units, diagnostics)
- [0010](0010-npmi-not-comparable-across-doc-units.md) — NPMI absolute values are not comparable across doc units (npmi, doc-units)
- [0009](0009-year-binning-intensifies-chronic-bg-for-hdp.md) — Year-binning intensifies chronic-background dominance for HDP (doc-units, hdp)
- [0008](0008-patient-year-docs-surface-transient-phenotypes.md) — Patient-year docs surface transient phenotypes that lifetime docs smear (doc-units, lda)
- [0007](0007-npmi-zero-pair-floor-penalizes-rare-phenotypes.md) — NPMI floors at −1 for zero-pair counts, penalizing rare phenotypes (npmi, diagnostics)
- [0006](0006-spread-and-eff-k-interpretation.md) — λ-spread and effective-K must be read together (diagnostics)
- [0005](0005-lda-decomposes-background-into-flavors.md) — LDA decomposes "background" into multiple flavors on patient-year docs (lda, doc-units)
- [0004](0004-lda-asymmetric-alpha-settles-late.md) — LDA asymmetric α settles later than topic-word distributions (lda, diagnostics)
- [0003](0003-hdp-vs-lda-per-iter-cost-on-short-docs.md) — HDP per-iter cost grows faster than LDA on short documents (hdp, svi, ops)
- [0002](0002-hdp-catchall-hoarding-at-last-stick.md) — HDP catch-all hoarding at the last stick (hdp)
- [0001](0001-hdp-gamma-collapse-at-low-gamma0.md) — HDP γ-collapse at low γ₀ (hdp)
