# 0028 — The prior family (Dirichlet vs logistic-normal), not gating, governs rare minority-phenotype recovery; "gated LDA" is Partially Labeled Dirichlet Allocation (PLDA)

**Date:** 2026-06-24
**Topic:** stm | lda | gating | priors | npmi | phenotyping | prior-art
**Status:** Observed (+ literature-confirmed)

The 0005/0027 thread asked whether gated STM's failure to surface dementia
sub-phenotypes was a gating defect. A controlled 2x2 plus a local reproduction
plus a literature search resolves it: **the differentiator is the document-topic
prior family — Dirichlet (LDA) vs logistic-normal (STM/CTM) — not gating.**
Dirichlet recovers rare minority phenotypes; logistic-normal washes them into the
anchors regardless of gating. And the gating design itself is not novel: it is
**Partially Labeled Dirichlet Allocation** (Ramage, Manning & Dumais, KDD 2011),
motivated by exactly this failure mode.

## The controlled comparison

Same `cancer_or_dementia` data, person_mod=4, condition_era, vocab built the same
way. The decisive same-corpus contrast is 0006 vs 0007 (only the model changes):

| Model | Setting | Dementia sub-phenotypes? |
|---|---|---|
| **LDA** (Dirichlet) | dementia-alone, mod=4 ([0006](../experiments/0006-lda-dementia-only-control.md)) | **✓** Alzheimer's (peak β 0.013), HIV-dementia, seizure, aphasia, BPH+amnesia by iter 10 |
| **STM** (logistic-normal) | dementia-alone, mod=4 ([0007](../experiments/0007-stm-dementia-only-control.md)) | **✗** 2 anchors hold ~98%; 38 tail clones, peak 0.004–0.008, all "HTN+Dementia+Postconcussion+baseline" |
| STM | combined, gated ([0005](../experiments/0005-gated-stm-dementia-only-foreground.md)) | ✗ 10 foreground clones, peak ~0.003 |
| STM | combined, prevalence-only ([0026](0026-stm-prevalence-gives-prevalence-not-content-fidelity.md)) | ✗ "minority arm has no tail" |
| LDA | dementia-alone, mod=1 ([0021](0021-cohort-corpora-two-anchor-mass-concentration.md)) | ✓ HIV-dementia 0.140, Lewy-body 0.090, epilepsy 0.060, … |

**STM collapses in every cell** (alone, gated-combined, prevalence-combined);
**LDA recovers in every cell** (alone at mod=1 and mod=4, and in the full corpus,
0019). The axis that flips the outcome is LDA↔STM. The gating/combined axis does
not flip it.

## Mechanism: Dirichlet vertex-sparsity vs logistic-normal interior-smoothing

- A **Dirichlet(α) with α < 1 places density at the simplex vertices**, so each
  document concentrates θ on a *few* topics. A doc carrying the epilepsy codes can
  dump nearly all its weight on one topic, which then accumulates epilepsy tokens
  and specializes. Asymmetric α (Wallach, Mimno & McCallum, NIPS 2009, "Rethinking
  LDA: Why Priors Matter") "substantially increases the robustness of topic models
  … to the highly skewed word frequency distributions" — exactly the rare tail.
  Our LDA already supports this (`optimize_doc_concentration`).
- The **logistic-normal** prior (θ = softmax(η), η ~ N(Γᵀx, Σ)) is unimodal and
  **never reaches the vertices**; it was introduced by the Correlated Topic Model
  (Blei & Lafferty) to model topic *correlation*, with no sparsity claim. Wang et
  al. (BAT, EMNLP 2020): the logistic-normal "does not exhibit multiple peaks at
  the vertices of the simplex as that in the Dirichlet … less capable to capture
  the multi-modality which is crucial in topic modeling." With a small learned Σ
  (0005's foreground Σ ≈ 0.92), η is pulled toward Γᵀx and θ stays in the interior,
  so rare-phenotype tokens get smeared across topics and no topic specializes.

This is why STM gives *prevalence* fidelity but not *content* fidelity (0026):
the logistic-normal can shift how much of a shared topic a group expresses, but
its non-sparsity resists carving a *new*, group-specific topic from weak signal.

## It is not a bug, and gating is sound

A local reproduction (real `OnlineSTM` code, in-process, sweeping signal strength
from clean→0005-like) found **no bug cliff** — smooth degradation — and **gated
STM ≥ non-gated STM** on the same rare docs. The block-aware masking, sufficient
statistics, Γ/Σ M-step, and `allowed_indices` are all correct. Gating *helps*
(the background absorbs common tokens, freeing the foreground); it was attached to
the wrong engine. (An earlier "gating starves the foreground" mechanism was
proposed and then **refuted** by this reproduction.)

## Prior art: gated LDA == PLDA

The gating design — each document may use a shared **background** topic block plus
**only its own group's** foreground block, by hard topic-masking — is
**Partially Labeled Dirichlet Allocation** (Ramage, Manning & Dumais, KDD 2011):
a generative restriction to a shared "latent class" (= background) plus the
document's labels' disjoint topic blocks, each label getting *multiple* sub-topics.
PLDA generalizes LDA (background-only) and Labeled-LDA (one topic/label, no
background); our single group variable is a special case of its per-document label
set. PLDA was motivated verbatim by our failure mode: plain LDA on imbalanced
labeled corpora "collapse[s] distinctions between small fields (folding them into a
single topic) and overly emphasize[s] the importance of larger ones, just based on
the amount of support in the data" (their corpus: 44,551 vs 1,041 instances — our
81/19). It scaled to ~1M docs on a small cluster.

**Our existing `allowed_indices` masking IS PLDA's generative restriction** — we
already built a working masked online-VI; we just need it on the Dirichlet (LDA)
engine. "Gated LDA" = a variational PLDA, and (like gated STM) the non-gated case
falls back to plain LDA.

EHR-specific state of the art is the **MixEHR lineage** (all Dirichlet-family):
MixEHR (Li et al. 2020, Nat Commun), MixEHR-Guided (surrogate-feature priors for
identifiability), MixEHR-Nest (seeded *hierarchical* sub-topics within a phenotype
topic — directly the rare-subphenotype goal). SAGE / PTM offer *word-level*
background/foreground (sparse log-deviation from a shared background), a different
axis from PLDA's topic-block masking; bcNMF is a contrastive-NMF alternative.

## Implications

1. **Adopt PLDA; build gated LDA as its variational implementation.** Graft the
   existing topic-masking onto the LDA E-step, keep asymmetric α on. Small lift —
   the masked online-VI scaffold already exists and is validated.
2. **Do not use logistic-normal STM for minority content *discovery*.** STM's
   strength is covariate *prevalence* comparison (0026); it is the wrong tool for
   carving rare group-specific topics. Prevalence-modeling and rare-content
   discovery are in genuine tension here (open question).
3. **Read order:** PLDA (Ramage 2011) → Wallach 2009 (asymmetric α) → Blei &
   Lafferty CTM → MixEHR / MixEHR-Nest → SAGE / bcNMF.

## Caveats

- **No published EHR head-to-head** proves logistic-normal under-recovers rare
  topics vs Dirichlet; the literature support is mechanistic (simplex geometry,
  asymmetric-Dirichlet robustness) and text-domain (PLDA on PhD abstracts, BAT on
  news). **This project's 2x2 is the most direct evidence and is plausibly
  publishable.**
- Wallach's asymmetric-prior benefit is **corpus-dependent** (stronger for
  short-text / small-vocab). OMOP condition_era docs are short bags of codes —
  favorable, but verify rather than assume.
- PLDA's demonstrations are non-EHR; whether its sub-topic recovery transfers to
  OMOP-coded data is what a gated-LDA run would establish (the next experiment).

## Setting context

Experiments 0004–0007 on the `cancer_or_dementia` cohort (union of
first_cancer_year + first_dementia_year, prior_obs_days=0, condition_era,
person_mod=4), and 0021 (first_dementia_year, mod=1). Online VI in Spark; NPMI
coherence eval (Röder et al. 2015), which per 0027 can reward degenerate uniform
topics — the load-bearing readout here is per-topic **peak β / Σλ spread /
top-term diversity** (now in the per-iter fit log), not the NPMI mean. Local
reproduction: `OnlineSTM`/`OnlineLDA` in-process, synthetic planted phenotypes.
Literature via a verified multi-source search (PLDA, Wallach 2009, CTM, BAT 2020,
MixEHR/-Guided/-Nest, SAGE, bcNMF).
