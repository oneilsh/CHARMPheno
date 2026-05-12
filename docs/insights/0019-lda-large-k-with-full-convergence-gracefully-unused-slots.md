# 0019 — LDA at large K with full convergence gracefully under-uses excess capacity, no micro-cluster artifacts
**Date:** 2026-05-12
**Topic:** lda | doc-units | diagnostics | hdp
**Status:** Observed

After ~80 iterations of K=80 LDA on patient-year condition-era docs
(an additional ~60 iters past the initial 20 we'd evaluated under
[0014](0014-patient-year-npmi-bimodal-vs-lifetime-unimodal.md)/[0018](0018-full-corpus-plus-threshold-yields-unimodal-positive-npmi.md)),
the topic structure resolved into a three-way decomposition that
contradicts the worry from earlier mid-iter snapshots about "many
undifferentiated chronic-comorbidity variants."

## The three classes

**1. Three distinct catch-alls** (Σλ in the 0.5M–1.1M range, ~14% to
29% of corpus mass each, NPMI +0.13 to +0.49 cov=100%):

- Acute-presentation signature (Pain, Chest pain, Nausea, SoB,
  Vomiting, Diarrhea) — 29% of mass
- Generic chronic-comorbidity background (HTN, GERD, HLD, Anxiety,
  T2DM, Obesity) — 20%
- Cardiometabolic (HTN, HLD, T2DM, Atherosclerosis, OSA) — 13%

**2. ~40 recognizable phenotype topics** (Σλ in the 1K–300K range,
peak 0.05–0.41, NPMI mostly +0.10 to +0.29). Includes back/spine
pain, foot/limb pain, PTSD-substance-use-MDD cluster, severe-mental-
illness-with-pulmonary, pregnancy, allergic rhinitis/sinusitis,
liver cirrhosis, T1DM-cancer, kidney transplant/ESRD/CKD, BPH-with-
prostate, mitral valve disease, RA-with-osteoporosis, SLE/Sjögren's,
ankylosing spondylitis+hepB, eye/glaucoma cluster, breast-cancer-
with-genetic-predisposition, postop-thyroid-cancer, neurogenic-
bladder, female urinary incontinence, dermatology aging cluster,
skin-aging, ulcerative colitis, AML/leukemia, herpes zoster,
schizoaffective, sickle cell, cystic fibrosis, dementia/Alzheimer's,
diabetic macular edema, peripheral venous insufficiency, essential
tremor, OSA/sleep-disorders, post-cardiac-event signature, sarcoidosis-
with-tear-film, psoriasis (peak 0.36!), myasthenia gravis,
neurofibromatosis, hypothyroidism (peak 0.21).

**3. ~25 gracefully-unused slots** (Σλ in the 500–3000 range, peak
< 0.05, α at the asymmetric-α floor ~0.011). These don't form
phenotypes; they don't form micro-clusters either; they just sit at
the prior-smoothing floor with a tiny amount of accumulated data.

## What "gracefully" means

"Gracefully unused" is doing real work as a phrase. Compare to
neighboring failure modes:

| failure mode                | LDA K=80 tail at iter 80         |
|-----------------------------|----------------------------------|
| **Collapsed model**         | No — the 40 phenotype topics work fine; the model didn't degrade. |
| **HDP micro-cluster artifacts** ([0002](0002-hdp-catchall-hoarding-at-last-stick.md)) | No — those have high peak, high NPMI, low coverage. LDA's tail has low peak, NPMI in line with what its sparse data supports, and high coverage when its top-N happens to be common terms. |
| **Noise** | Not really — the tail slots' top words are recognizable rare-or-weird concepts (Marfan's, Multiple personality, Contact dermatitis of eye, Post-acute COVID-19). They're just under-represented. |
| **Capacity competition** | No — low α means these slots rarely win in the per-doc inference, so they don't pull attention from active topics. |

The key property: **the tail doesn't pollute downstream analysis.**
A K=80 model with ~25 unused slots behaves like a K=55 model with 25
no-op decorations, not like a K=55 model with 25 noise generators.
You can ignore them by filtering on Σλ or E[β] < threshold and
nothing breaks.

This is essentially the BNP value HDP promised — "let the data pick
how many topics matter" — emerging from a parametric LDA model run
at generous K with sufficient iterations. The required ingredients
are (a) generous K, (b) full convergence (80+ iters here at
batch_fraction=0.2; would be more at smaller batches), (c) asymmetric
α optimization to let the unused slots drop to the floor.

## A subtler diagnostic interaction worth knowing

Within class 3 (gracefully-unused), there's a sub-population that
the new (NPMI, coverage, Σλ) joint reading [0018](0018-full-corpus-plus-threshold-yields-unimodal-positive-npmi.md)
makes visible: **topics with Σλ ~ 500–800 (basically unused) but
NPMI cov=100% and NPMI in the +0.07 to +0.19 range**. Examples:
topics 39, 25, 20 — peak 0.002–0.005 on HTN/HLD/T2DM/GERD/chest pain.

These score high NPMI because their top-N words are common terms
that always co-occur in the corpus, not because the topic itself
has crisp coherent content. The topic isn't actually used by any
doc; the top-N is determined by η-smoothing of common concepts.

**NPMI alone doesn't distinguish "this topic gets many docs" from
"this topic's top-N words happen to be common in the corpus."**
The (NPMI, Σλ, coverage) joint reading is the right diagnostic:

| pattern                         | reading                                    |
|---------------------------------|--------------------------------------------|
| NPMI high, cov high, Σλ high    | Real, important phenotype or catch-all.    |
| NPMI high, cov low, Σλ low      | HDP-style micro-cluster artifact.          |
| NPMI mid, cov high, Σλ low      | LDA's "gracefully-unused but common-words" — unused slot whose top-N is just baseline terms. |
| NPMI low, cov low, Σλ low       | True unused slot with rare-or-weird top-N. |

The middle two LDA cases are both unused but look different on the
metric. Worth checking Σλ before celebrating an apparent rare-pheno
discovery.

## Implications

For this corpus the production recipe is:

- **LDA K=80** on patient-year condition_era docs
- Asymmetric α optimization on
- batch_fraction=0.1 (drop from 0.2 for late-stage convergence
  stability)
- ~80 iterations or until α range plateaus
- The eval reports per-topic (NPMI, coverage, Σλ); filter or rank by
  any of these three depending on what you care about (sharpness,
  statistical support, doc-usage)

The HDP-vs-LDA comparison from [0017](0017-hdp-gamma-sensitivity-is-prior-dominance.md)
now resolves cleanly: the rare-phenotype recovery HDP was supposed to
provide is achievable with LDA at generous K + full convergence,
without γ-sensitivity, catch-all hoarding, or micro-cluster artifacts.
HDP retains exploratory value (its T=80 fit surfaced phenotypes
across one specific γ neighborhood), but the production fit for this
corpus is parametric.

## Setting context

K=80 LDA, online VI, patient-year condition_era docs, min_doc_length=20,
person_mod=10, vocab_size=10000, min_df=10, asymmetric α optimization
on, batch_fraction=0.2 (first 60 iters) and then 0.1 (continued past
60). τ_0=64, κ=0.7. Resumed from an existing checkpoint to reach
iter 80 cumulative. Eval as per ADR 0017 revision: full-corpus
reference, min_pair_count=3.
