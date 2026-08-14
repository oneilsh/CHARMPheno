# Deep dive: sureLDA and PhenoBrain (for CHARMPheno positioning)

**Date:** 2026-08-14 · Companion to `2026-08-14-rare-disease-diagnosis-lit-review.md`.
Two questions: (a) is **sureLDA** "already did it" prior art I need to know cold? (b) is
**PhenoBrain** a strong, well-established, non-genomic replacement for a Phenomizer comparison?

**Bottom line up front:**
- **sureLDA is close cousin prior art, NOT a scoop.** Same *family* (guided topic model for
  EHR phenotyping) but a *different task* (weakly-supervised labeling of **known** phenotypes,
  not discovery/case-finding of rare ones), a different guidance mechanism (per-patient
  surrogate prior, requires 2 surrogates per target), no hierarchy/DAG, no nonparametrics, and
  it is **not built for the rare regime** — a 2026 follow-up shows this whole weakly-supervised
  family degrades on rare outcomes. Cite it as the nearest guided-LDA precedent and distinguish
  cleanly.
- **PhenoBrain is a legitimate, well-established, non-genomic upgrade over Phenomizer.**
  npj Digital Medicine 2025, public code+data, dominates Phenomizer (top-3 0.49 vs 0.26,
  median rank 4 vs 15) and also beats LIRICAL, Phen2Disease, GPT-4-from-EHR, and 50
  physicians. Citing it buys you Phenomizer *transitively*. **But** it's a per-patient
  differential-diagnosis ranker that still needs HPO terms (it automates text→HPO, it doesn't
  remove the mapping problem), so wiring it to structured AoU OMOP takes a deliberate adapter
  (see §2.4).

---

## 1. sureLDA — Ahuja et al., *JAMIA* 2020 (celehs/sureLDA, on CRAN)

### 1.1 What task it actually solves
**Weakly-supervised, multi-disease *phenotyping*** — i.e. "for a pre-specified list of
phenotypes, assign each patient a probability of having each one, at phenome scale, without
chart-reviewed gold labels." It's a replacement for **rule-based / chart-review phenotype
algorithms** (the PheNorm / MAP / PheKB lineage from the Kohane/Cai CELEHS group). The target
phenotypes are **known and named up front**; the job is high-throughput *labeling*, not
*discovery* and not rare-disease *case-finding*. This is the crucial distinction from your
rare6 arc (which is closer to "find undiagnosed/hidden cases of a rare disease").

### 1.2 The method, mechanistically
Three stages (this is the part worth knowing precisely, because it's structurally adjacent to
your gate):

1. **PheNorm/MAP init (the "surrogate" step).** For each target phenotype you supply **2
   surrogate features** — typically the main ICD/phecode count and the NLP mention count for
   that disease — plus a healthcare-utilization proxy. PheNorm is an unsupervised denoising /
   normal-mixture procedure that turns those noisy surrogates into a **per-patient posterior
   probability** of the phenotype. So after stage 1 you already have a rough P(patient has
   phenotype _j_) for every _j_, with **no gold labels** — just the surrogate choices.
2. **Guided LDA (the "sure" step).** LDA is run with **one dedicated topic per target
   phenotype**. The PheNorm posteriors are injected as an **informative prior on the
   per-patient topic loadings** (the document–topic distribution θ): a patient PheNorm-scored
   high for phenotype _j_ gets a prior that pushes mass onto topic _j_. LDA then redistributes
   the high-dimensional feature counts (ICD, CPT, RxNorm, NLP concept counts) across these
   anchored topics. **This is the same shape as your gate** — an external per-document signal
   pins a topic block to the documents it belongs to — except the signal is a *surrogate-
   derived soft prior per patient*, not a *label / DAG-subtree membership*, and there is **no
   hierarchy** among topics.
3. **Ensemble readout.** The final per-phenotype probability combines the LDA loading for
   topic _j_ with the raw surrogate counts via a clustering ensemble (normal-mixture on the
   combined score). "Ensemble" here = fusing surrogate + LDA views, **not** an ensemble of
   models in the PhenoBrain sense.

### 1.3 Supervision, inputs, evaluation
- **Weakly-supervised / "label-free":** no gold-standard chart labels; the only human input is
  choosing 2 surrogates per phenotype. (You *do* need to know the disease and its main codes —
  so it can't discover an unnamed phenotype.)
- **Inputs:** high-dimensional structured + NLP counts per patient (ICD, procedures, meds, and
  NLP-extracted concept mentions) plus healthcare utilization.
- **Evaluation:** simulations + real phenotypes (the CELEHS Partners/VA-style validation sets,
  order ~10 common-to-moderate phenotypes). Headline: matches or beats PheNorm, MAP, and plain
  LDA; **robust to prevalence and to whether the surrogate or the non-surrogate features carry
  the signal.** Ships as an R package (`sureLDA`, CRAN + celehs GitHub).

### 1.4 Why it is NOT a CHARMPheno scoop (what to say in Related Work)
| Axis | sureLDA | CHARMPheno rare6 |
|---|---|---|
| Task | label **known** phenotypes at scale | find/rank patients for **rare** (often undiagnosed) disease |
| Guidance | per-patient PheNorm **surrogate prior** on θ; needs 2 surrogates/phenotype | **DAG-subtree gate** + optional PC label head; hierarchy-aware |
| Model | fixed-K parametric LDA, 1 topic/phenotype | **HDP** (nonparametric K) + background topics + gate |
| Hierarchy | none (flat topic per disease) | **is-a DAG placement**, ancestral closure |
| Uncertainty | mixture posterior on the score | **Bayesian generative profile**, uncertainty at every level |
| Rare regime | **not designed for it** — see §1.5 | the whole point |
| Output | per-phenotype probabilities | reusable interpretable **profile** (retrieval, trajectories, on-device) |

So the honest framing: *"Guided/anchored topic models for EHR phenotyping are established
(sureLDA, MixEHR-Guided). CHARMPheno differs by (i) targeting rare-disease discovery rather
than known-phenotype labeling, (ii) injecting an is-a hierarchy via the gate rather than a
per-patient surrogate prior, (iii) being Bayesian-nonparametric and uncertainty-aware, and
(iv) emitting a reusable generative profile."* That's a defensible, specific delta — and it
means you should **read the sureLDA paper once** so a reviewer can't say you missed it, but you
are not pre-empted.

### 1.5 The rare-regime caveat you can weaponize
There's a 2026 follow-up — *"Performance of weakly-supervised EHR phenotyping methods in
rare-outcome settings"* (arXiv 2604.09913, Hong/Nelson/Williamson) — that benchmarks exactly
this family (PheNorm/MAP/sureLDA-style) and finds the expected **degradation at low
prevalence**. This is useful external support for your own "AUC flatters low prevalence /
information-limited" findings AND a reason the sureLDA recipe doesn't trivially transfer to
rare6. Worth pulling in full if you write the related-work section.

Sources: [sureLDA (JAMIA 2020)](https://academic.oup.com/jamia/article/27/8/1235/5858306) ·
[celehs/sureLDA](https://github.com/celehs/sureLDA) ·
[rare-outcome benchmark (arXiv 2604.09913)](https://arxiv.org/pdf/2604.09913) ·
[PheNorm background](https://celehs.github.io/sureLDA/)

---

## 2. PhenoBrain — Mao et al., *npj Digital Medicine* 2025

### 2.1 Provenance / how established
- **npj Digital Medicine (Nature portfolio), published 28 Jan 2025** (submitted 2023-09, long
  review). Peer-reviewed, credible venue — a citable, defensible comparator, not a preprint.
- **Public and reproducible:** live tool (`phenobrain.cs.tsinghua.edu.cn`), source
  (`github.com/xiaohaomao/timgroup_disease_diagnosis`), de-identified data on Zenodo. You can
  actually run it.
- **Purely phenotypic (non-genomic)** — authors flag "no genetics" as a limitation, which is
  exactly why it fits your current no-genomics scope.

### 2.2 What it is, mechanistically
A two-module **differential-diagnosis ranker**: given a patient, output a ranked list of
candidate rare diseases.

1. **Phenotype extraction — PBTagger.** ALBERT (lite-BERT) + deep metric learning maps free
   **clinical text** → HPO terms (TopWORDS segmentation → twin-network match against a unified
   medical thesaurus). Trained primarily on **Chinese**; for English it leans on PhenoBERT /
   PhenoTagger. This module is what "reads the EHR."
2. **Differential diagnosis — 5-method ensemble** over the HPO term set against a knowledge
   base of **9,260 rare diseases / 168,780 disease–phenotype annotations** (OMIM + Orphanet +
   CCRD):
   - **ICTO** — information-content term-overlap semantic similarity (robust to unannotated
     phenotypes);
   - **PPO** — Bayesian probability propagation up the HPO DAG → P(disease | phenotypes);
   - **CNB** — complement naïve Bayes, few-shot, with Mixup / random-perturbation augmentation;
   - **MLP** — one-hidden-layer net, BCE + L2, same augmentation;
   - **Ensemble** — rank fusion via order statistics (Beta/Gamma approximation).

### 2.3 Evidence it beats Phenomizer (and the rest)
On the combined public rare-disease case sets (RAMEDIS/MME/LIRICAL/HMS, 362 diseases) and
PUMCH hospital data:

| Comparator | PhenoBrain top-3 | that tool top-3 | PhenoBrain top-10 | that tool top-10 | median rank |
|---|---|---|---|---|---|
| **Phenomizer** | **0.486** | 0.256 | **0.651** | 0.434 | 4.0 vs **15.0** |
| LIRICAL | 0.483 | 0.407 | 0.640 | 0.560 | — |
| Phen2Disease | — | — | 237/384 hits | 221/384 | — |
| GPT-4 (EHR in) | **0.613** | 0.507 | **0.813** | 0.667 | — |
| 50 physicians | **0.613** | 0.511 | **0.813** | 0.524 | — |

It **decisively dominates Phenomizer** and edges the modern phenotype rankers, LLMs-from-EHR,
and specialist physicians (physician+AI combined is best of all, top-3 0.768). So: **yes, it
is well-established as better than Phenomizer**, and citing PhenoBrain lets you claim the
current phenotype-only SOTA while getting the Phenomizer comparison transitively. You can
credibly **drop bare Phenomizer** and say so.

### 2.4 The catch for *your* use case (read this before committing)
PhenoBrain is a **per-patient differential-diagnosis ranker evaluated on richly-phenotyped
case reports / hospital admissions** — not a population case-finder over structured claims.
Three frictions for an AoU rare6 head-to-head:

1. **It still needs HPO terms.** PBTagger *automates* text→HPO but doesn't *remove* the mapping
   problem — and it needs **clinical text**. AoU is mostly **structured OMOP** with limited
   notes, so the PBTagger front-end may have little to chew on. The realistic path is to
   **skip PBTagger** and feed the diagnosis module HPO terms mapped from your OMOP condition
   codes (SNOMED/ICD → HPO), i.e. build a structured-code→HPO adapter. That's the same
   "map HPO into EHR" chore you were dreading — but you'd do it **once**, feed PhenoBrain's
   ensemble, and it's a legitimate strong baseline. (Bonus: that adapter is independently
   useful and is roughly what a Phenomizer port would need anyway.)
2. **Task transpose.** PhenoBrain ranks *diseases for a patient*; rare6 ranks *patients for a
   disease*. To compare on your metric, run PhenoBrain per patient, then for target disease _d_
   rank patients by the score PhenoBrain assigns _d_ — that transposes its output into a
   case-finder and yields per-disease detection-AP / P@R comparable to your arms. Worth a
   sentence in methods so reviewers see it's apples-to-apples.
3. **Language + KB shape.** PBTagger is Chinese-first (English via PhenoBERT/PhenoTagger), and
   the KB is annotation-driven — so diseases with sparse HPO annotations (the paper flags ~2000
   with <5 annotations) are weak spots, which may or may not overlap your rare6 set.

### 2.5 Recommendation
- **Adopt PhenoBrain as the phenotype-ranking comparator; drop bare Phenomizer** and state that
  PhenoBrain supersedes it (with the numbers above). Cleaner, stronger, still non-genomic,
  publicly runnable.
- Budget for a **OMOP-code → HPO adapter** to feed PhenoBrain's diagnosis module directly
  (bypassing PBTagger), and run it **transposed** into a per-disease case-finder for an
  apples-to-apples rare6 comparison.
- If the grad student's HPO-in-EHR pipeline already exists for the Phenomizer/PhEval effort,
  that adapter is largely the same artifact — reuse it.

Sources: [PhenoBrain (npj Digit Med 2025, PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11775211/) ·
[code](https://github.com/xiaohaomao/timgroup_disease_diagnosis) ·
[data (Zenodo)](https://zenodo.org/records/10774650) ·
[tool](http://www.phenobrain.cs.tsinghua.edu.cn/pc)

---

## 3. One-paragraph Related-Work draft (reusable)

> Guided and anchored topic models are established for EHR phenotyping: **sureLDA** [JAMIA
> 2020] anchors one LDA topic per target phenotype using PheNorm surrogate priors, and
> **MixEHR-Guided** anchors multimodal topics to PheCodes — but both address weakly-supervised
> labeling of *known* phenotypes at flat, fixed K, and weakly-supervised phenotyping is known
> to degrade in rare-outcome settings [arXiv 2604.09913, 2026]. Phenotype-driven differential
> diagnosis is likewise mature — **Phenomizer** [AJHG 2009] and its successors **LIRICAL**,
> **Phen2Disease**, and **PhenoBrain** [npj Digit Med 2025], the last of which dominates
> Phenomizer (median rank 4 vs 15) and 50 specialist physicians — but these rank candidate
> diseases for a *single richly-phenotyped patient*, not undiagnosed cases across a
> population. CHARMPheno differs on all three axes: it targets rare-disease case-finding rather
> than known-phenotype labeling or single-patient triage; it injects an is-a hierarchy through
> a subtree gate rather than a per-patient surrogate prior; and it is Bayesian-nonparametric
> and uncertainty-aware, emitting a reusable generative patient profile.
