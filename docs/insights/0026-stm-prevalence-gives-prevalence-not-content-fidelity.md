# 0026 — Prevalence-only STM reproduces LDA's cohort concentration and gives rare covariate groups prevalence fidelity, not content fidelity

**Date:** 2026-06-23
**Topic:** stm | lda | covariates | doc-units | npmi | diagnostics
**Status:** Observed

The first end-to-end prevalence-only STM fit on the combined
`cancer_or_dementia` cohort (K=40, ~13.3k cohort docs, 79% cancer /
21% dementia by document, 100 SVI iters) shows the *same* corpus-level
topic structure as α-optimized LDA on cohort corpora — despite a
different document–topic prior. And it makes precise what a prevalence
covariate does and does not buy a rare group.

## Same structure as LDA, across a model-class boundary

STM's document–topic prior is logistic-normal, η_d ~ N(Γᵀx_d, Σ), not
Dirichlet α. Yet the corpus decomposes exactly as [0021](0021-cohort-corpora-two-anchor-mass-concentration.md)
and [0019](0019-lda-large-k-with-full-convergence-gracefully-unused-slots.md)
describe for LDA:

- **3 anchor topics** carry 88% of corpus mass — universal-symptom
  (E[β]=0.536: Pain, Chest pain, Nausea, SoB, Vomiting), cardiometabolic
  (0.258: HTN, HLD, T2DM, GERD, OSA), generic-chronic (0.090). This
  sits *between* single-cohort dementia (2 anchors, 93% — 0021) and the
  full corpus (3 catch-alls, ~60% — 0019), confirming 0021's "anchor
  concentration scales inversely with cohort heterogeneity": unioning
  two index conditions adds heterogeneity and decomposes the baseline
  one step further.
- **~30 gracefully-unused slots** at E[β]≈0.0010, Σλ≈1.1e3, NPMI
  +0.02–0.05 with cov≈99% on baseline terms (HTN, HLD, Actinic
  keratosis, Obesity). This is exactly 0019's "NPMI mid, cov high, Σλ
  low = unused slot whose top-N is just common corpus terms" class —
  η-smoothing of frequent concepts, not real content. Filter on Σλ/E[β].

**The non-obvious part:** LDA's *asymmetric-α optimization actively
amplifies* the anchor concentration over iterations (0021). STM has no
α — and its Σ stayed near-uniform and small (diagonal ≈0.93, i.e.
similar residual prevalence variance across topics, no prior-driven
amplification). Yet it concentrates to the same ~88%. So the
concentration is a **corpus / pooled-β property, invariant to the
document–topic prior family**, not an artifact of either prior's
optimizer. This strengthens 0021's "this is the cohort filter, not the
model" claim by removing the prior as a possible cause.

## The crisp tail is majority-led; the minority arm has no tail

The small-E[β] tail resolved sharp phenotypes (0021's "what does
work"): breast cancer (topic 18, NPMI 0.227), skin/melanoma (21,
0.265), head & neck cancer (4, 0.310), pregnancy (17, 0.442), diabetic
eye (8, 0.141). **All cancer or general — no crisp dementia
phenotype.** The dementia arm (21% of docs, dominated by generic
post-dx comorbidity coding) feeds the anchors, not the tail.

## Why: a prevalence covariate touches prevalence, not content

This is the load-bearing observation. In prevalence-only STM:

- **β (topic–word) is SHARED across all documents** and estimated by
  pooled SVI, weighted by token mass. With cancer at 79% of tokens,
  the discovered topics are cancer-led; dementia's tokens reinforce the
  anchors rather than carving distinct topics.
- **The covariate enters ONLY the prevalence mean** (Γᵀx_d shifts the
  logit-topic mean η_d). It re-weights *how much of each shared topic*
  a dementia document expresses — it cannot create a dementia-specific
  *topic*, because β has no covariate dependence.

So a prevalence covariate gives a rare group **prevalence fidelity**
(here Γ is strongly estimated, |Γ|max=8.22 — dementia docs do get a
distinct topic-prevalence profile) but **not content fidelity** (no
dementia-specific vocabulary clusters). If the pooled β never surfaced
a dementia phenotype, the covariate has nothing dementia-specific to
up-weight.

## Are we optimizing the right priors? Yes — for this model

The OnlineSTM M-step *does* optimize the prevalence model: **Γ** (ridge
regression of η on X) and **Σ** (diagonal residual covariance), both
ρ-blended stochastic-EM; **λ/β** via SVI; **η_d** per-doc via L-BFGS.
The only fixed prior is the β-smoothing scalar `eta` (carried unchanged
through `update_global`). Optimizing `eta` is possible (LDA optionally
does) but it is a single global scalar — it would not give a minority
group its own content. So the missing fidelity is **not a missed
optimization**; it is a **model-scope property** of prevalence-only STM.

## Levers, if minority *content* fidelity is needed

None are implemented; prevalence-only was the deliberate design (the
Atlas story is "bubbles resize, don't move").

1. **Content covariates** (the proper STM extension; Roberts et al.
   content model / SAGE): make β itself covariate-dependent,
   log β_{k,c} = m + κ^topic_k + κ^cov_c + κ^{int}_{k,c} with a sparse
   prior on κ. Gives each group a flavored version of the shared
   topics. Biggest change; also changes the dashboard semantics
   (topics' *words* shift by group, not just their sizes).
2. **Class-balanced / stratified SVI sampling**: over-sample the
   minority arm in mini-batches so its tokens influence pooled β more.
   Small change, directly targets "rare group under-represented in β
   estimation" — at the cost of a corpus-β that no longer reflects the
   true document mix.
3. **Separate per-cohort fits**: fit dementia alone to discover its
   phenotypes; lose the single joint covariate comparison.

For the *validation* goal (does the prevalence-covariate machinery
work end-to-end), prevalence fidelity is the success criterion and it
is met. Content fidelity for both arms is a different goal that
prevalence-only STM does not serve.

## Setting context

Prevalence-only STM (Path A), K=40, `cancer_or_dementia` cohort (union
of `first_cancer_year` + `first_dementia_year`, prior_obs_days=0, 365d
fully-observed follow-up), `patient_cohort` doc-unit, person_mod=4
(~25% sample), formula `~ C(source_cohort) + C(sex) + age`, realized
vocab 4422, 100 SVI iters, batch_fraction=0.2, τ_0=64, κ=0.7.
NPMI eval: cohort-matched (fit-corpus) reference, reference_size=13295,
min_pair_count=3, top_n=20. NPMI distribution mean +0.091 / median
+0.040 / max +0.484, all positive (the 0018 unimodal-positive signature).
Absolute NPMI is not comparable to patient-year LDA runs (new doc-unit;
see [0010](0010-npmi-not-comparable-across-doc-units.md)).
