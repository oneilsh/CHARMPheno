# Rare-disease diagnosis from EHR data — lit review for the CHARMPheno case-finding arc

**Date:** 2026-08-14 · **Scope:** approaches *directly comparable* to CHARMPheno's rare6
case-finding — i.e. **non-genomic, structured-EHR / phenotype-driven** identification of
rare-disease patients. Genomic variant-prioritization work (Exomiser proper, most of the
"diagnostic-odyssey sequencing" literature) is deliberately out of scope except where a
phenotype-only subcomponent is comparable. Regular searches, not deep research — treat this
as a map with entry points, not an exhaustive survey.

## TL;DR / orientation

The comparable field sorts into four tiers by how close each is to what CHARMPheno does
(learn an interpretable representation from OMOP codes, then rank patients for a rare
phenotype):

1. **Knowledge-based phenotype risk scores** — PheRS. No training; map disease→phecodes,
   weighted sum by inverse prevalence. The cheapest, most interpretable, most obviously
   AoU-runnable baseline. **This is the baseline you should actually run**, arguably before
   Phenomizer.
2. **Self-supervised / supervised ML case-finders over codes** — RarePT (transformer,
   masked-diagnosis modeling), PheNet (per-disease), and a large "single-disease ML"
   applied genre. RarePT is CHARMPheno's closest *learned-representation* cousin.
3. **Topic models / latent-phenotype models over EHR** — sureLDA, MixEHR / MixEHR-Guided,
   GETM. Same model family as CHARMPheno; sureLDA in particular is essentially the
   "guided LDA" idea you're doing with PC shaping. **Closest methodologically.**
4. **Phenotype-driven differential-diagnosis rankers** — Phenomizer and successors
   (LIRICAL, Phen2Disease, PhenoBrain). Comparable in *goal* but a different modality of
   input: they rank candidate diseases for *one* presented patient from a *curated* HPO
   term set, not mine longitudinal EHR at population scale.

**The Phenomizer framing needs care** (see §4): Phenomizer is not a population case-finder.
It ranks diseases given a clinician-curated HPO profile. The EHR-native analog is PhenoBrain
(extract HPO from notes → ensemble rank). For AoU **structured** OMOP data, the fair,
apples-to-apples comparators are **PheRS** and **RarePT** (both phecode-native), unless the
grad student's pipeline first extracts HPO terms from AoU notes.

---

## 1. Knowledge-based phenotype risk scores (PheRS) — the cheap, interpretable baseline

**Bastarache et al., *Science* 2018** — "Phenotype risk scores identify patients with
unrecognized Mendelian disease patterns." The foundational comparable method.
- **Method:** For each of 1,204 Mendelian diseases, map its OMIM clinical features → HPO →
  **phecodes** (consolidated ICD billing codes). A patient's PheRS for a disease = weighted
  sum of the disease's phecodes the patient has, each weighted by **log inverse prevalence**
  (rarer manifestations count more). No training, no labels — purely knowledge-driven.
- **Data / eval:** Vanderbilt BioVU, 21,701 genotyped adults. Validated by separating
  clinically-diagnosed cases from matched controls across 6 Mendelian diseases (5 at
  p<5×10⁻⁴²); then used to discover rare-variant↔phenotype associations and to flag
  undiagnosed carriers (16 with severe outcomes incl. transplants).
- **Tooling:** `phers` R package (Bioinformatics 2022) — free, on CRAN. Runs directly on
  phecodes; nothing genomic required for the scoring itself.
- **Why it matters for you:** This is the most **directly runnable** comparator on AoU OMOP.
  It's per-disease, interpretable, knowledge-driven, and needs only a phecode map. It is the
  natural "no-learning" floor against which CHARMPheno's learned representation must justify
  itself. Recent extensions show the genre is alive: a **TTR V142I / ATTR-CM PheRS**
  (medRxiv 2026) targets undiagnosed variant amyloidosis; a *Genetics in Medicine* /
  *Am J Hum Genet* line ("phenotypic presentation of Mendelian disease across the diagnostic
  trajectory," 2023) studies how these signatures build up over time in the EHR.
- **Limitation vs CHARMPheno:** PheRS is *not a representation* — it scores against a
  hand-mapped disease signature, one disease at a time, and inherits HPO→phecode mapping
  noise. It cannot discover a "hidden-low-mass" phenotype the ontology doesn't already name.
  CHARMPheno's pitch is exactly the learned, reusable, multi-purpose profile.

Sources: [Science 2018](https://www.science.org/doi/10.1126/science.aal4043) ·
[PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5959723/) ·
[phers R pkg (Bioinformatics 2022)](https://academic.oup.com/bioinformatics/article/38/21/4972/6694842) ·
[diagnostic-trajectory paper](https://www.sciencedirect.com/science/article/pii/S1098360023009346) ·
[TTR PheRS (medRxiv 2026)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12803306/)

## 2. ML case-finders over codes — the learned-representation cousins

**RarePT — "Rare disease Phenotype Transformer"** (medRxiv 2023 / 2024). **The closest
learned-representation analog to CHARMPheno's goal.**
- **Method:** Transformer decoder with self/cross-attention over **phecodes** (many-hot,
  set-of-codes not sequence) + age/sex. Trained by **masked diagnosis modeling** (BERT-style
  self-supervision): mask one phecode, reconstruct it. **Label-free** — needs only
  presence/absence of codes; reweights training to balance rare vs common codes.
- **Data / eval:** Trained on 436,407 UK Biobank; validated on 3.3M Mount Sinai patients.
  Predicts 155 rare diagnoses (<1/2,000). Validated indirectly — diagnostic OR (median 48 in
  UKB, 31 at Sinai), enrichment for known diagnostic biomarkers (72%), mortality (65%) and
  disease-burden (73%) associations. Headline claim: **≥50% of patients remain undiagnosed**
  for 20/32 rare diseases with a confirmatory test.
- **Contrast with CHARMPheno:** same spirit (learn a representation over codes, self-
  supervised, cohort-scale, rare-disease case-finding) but **opaque** (attention weights, not
  interpretable topics) and **discriminative** (not a generative profile you can reuse for
  patients-like-me / trajectories / on-device). This is the paper to position against as "the
  deep-learning incumbent"; CHARMPheno's differentiator is interpretability + a reusable
  generative profile + privacy-friendly compact model.

**PheNet** (Pasaniuc & Butte, UCLA) — undiagnosed **CVID** from EHR.
- Supervised phenotype-risk model that learns phenotypic patterns from verified cases and
  ranks patients. Blinded chart review of top-100: **74% highly probable CVID**;
  retrospectively **64% identifiable ~8 months earlier** than actual diagnosis; externally
  validated on **>6M records** across 5 UC Health systems + Tennessee. Single-disease, but
  the **gold-standard validation template** (top-ranked chart review + external + lead-time).

**Single-disease applied ML** — a large, practically important genre, usually gradient-
boosting/RF on claims+EHR features, single disease, chart-review validated:
- Acute hepatic porphyria (PLOS One 2020) · Gaucher (Optum claims) · APDS (claims) ·
  and many others (Fabry, aHUS, acromegaly…). See the JAMIA Open "lessons learned" write-up
  for the honest failure modes. CHARMPheno's rare6 is effectively a **multi-disease
  generalization** of this genre with a shared representation instead of one model per disease.

Sources: [RarePT (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10775679/) ·
[RarePT (medRxiv)](https://www.medrxiv.org/content/10.1101/2023.12.21.23300393v1.full) ·
[PheNet / CVID (medRxiv)](https://www.medrxiv.org/content/10.1101/2022.08.03.22278352v1) ·
[PheNet (PubMed)](https://pubmed.ncbi.nlm.nih.gov/38691621/) ·
[AHP case study (PLOS One)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0235574) ·
[Gaucher algorithm](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10492341/) ·
[ML-to-detect-rare-disease lessons (JAMIA Open)](https://academic.oup.com/jamiaopen/article/5/2/ooac053/6621904)

## 3. Topic models / latent-phenotype models over EHR — the same family as CHARMPheno

**sureLDA** (Ahuja et al., *JAMIA* 2020) — **the closest methodological neighbor.**
- Surrogate-guided ensemble LDA: PheNorm initializes per-phenotype probabilities from 2
  surrogate features, then **constrains LDA to produce phenotype-specific topics** — label-
  free, multi-disease. This is structurally what CHARMPheno's **gated / prediction-constrained
  shaping** is doing (welding topic blocks to a target). Worth a direct read + citation as
  prior art for "guided topic models for phenotyping"; your Bayesian-nonparametric HDP + PC
  head is a principled generalization.

**MixEHR / MixEHR-Guided** (Li et al., *Nature Communications* 2020; bioRxiv 2021) —
multimodal collapsed-Gibbs topic model over ICD/labs/meds/notes simultaneously, with
modality-specific distributions; the *Guided* variant anchors topics to PheCodes. Directly
relevant to your **"is multi-domain worth it"** question (insights 0071–0079): MixEHR is the
incumbent evidence that multimodal *can* help — but your per-domain ablation found condition
near-sufficient and observation net-negative, which is a legitimate, publishable contrast.

**GETM** (graph-embedded topic model, 2022) — injects biomedical graph structure into the
topic embedding (pain phenotypes, UK Biobank). Relevant to your DAG-placement work: it's a
different way to inject hierarchy than your gate.

Also: **A Systematic Review of Topic Modeling Techniques for EHR** (MDPI *Healthcare* 2026)
— use as a one-stop citation net for the family. And your own lab's LDA-at-scale work
([npj Digital Medicine 2024](https://www.nature.com/articles/s41746-024-01286-3)) is the
direct methodological ancestor.

Sources: [sureLDA (JAMIA)](https://academic.oup.com/jamia/article-abstract/27/8/1235/5858306) ·
[MixEHR-Guided (bioRxiv)](https://www.biorxiv.org/content/10.1101/2021.12.17.473215.full.pdf) ·
[GETM (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9142639/) ·
[topic-modeling-for-EHR review](https://www.mdpi.com/2227-9032/14/2/282)

## 4. Phenotype-driven differential-diagnosis rankers — the "Phenomizer" family

These match your *goal* (rank a rare disease for a patient) but differ in **input modality**
and **use case** — important for framing the comparison honestly.

**Phenomizer** (Köhler et al., *AJHG* 2009) — semantic-similarity ranking of a
**clinician-curated HPO term set** against OMIM/Orphanet disease→HPO annotations. Point-of-
care differential diagnosis for *one* patient. **It is not a population case-finder and does
not mine raw longitudinal EHR** — it presumes someone has already deep-phenotyped the patient
into HPO terms. So "compare to Phenomizer" implicitly requires an HPO-extraction front-end on
AoU data (from notes), which the structured-OMOP path doesn't naturally give you.

**Successors, all still phenotype-only:**
- **LIRICAL** — likelihood-ratio differential diagnosis over HPO (Monarch).
- **Phen2Disease** — phenotype similarity ranking.
- **PhenoBrain** (2025, *PMC11775211*) — **purely phenotype/EHR, non-genomic.** Extracts HPO
  from clinical text via a BERT tagger, ensembles 5 rankers; on 75 cases hits top-3 recall
  0.61 / top-10 0.81, **beating 50 specialist physicians** and outranking Phenomizer (median
  rank 4 vs 15), LIRICAL, and GPT-4-from-EHR. This is the modern, EHR-native, non-genomic
  version of "the Phenomizer comparison" — a better target than Phenomizer itself if you want
  a strong phenotype-ranking baseline.

**PhEval + Exomiser** (Monarch) — **mostly out of scope / genomic.** PhEval is a benchmarking
*harness* (phenopackets in → ranked genes out, with phenotype-scrambling to test noise
robustness). Exomiser is a **variant+gene prioritizer** — its phenotype score is only *one*
input combined with genomic variants (82% top-1 with variants+phenotype vs 55% phenotype-
alone on 4,877 cases). **Flag for the grad student's port:** getting Exomiser/PhEval running
in AoU only yields a *phenotype-only* comparator if you use its phenotype-scoring subcomponent
in isolation; the full pipeline needs genomes, which you've scoped out for now. PhEval's real
value to you is as an **evaluation harness / phenopacket standard**, not a case-finder.

Sources: [Phenomizer (AJHG 2009)](https://pubmed.ncbi.nlm.nih.gov/19800049/) ·
[PhenoBrain (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11775211/) ·
[PhEval (BMC Bioinformatics 2025)](https://link.springer.com/article/10.1186/s12859-025-06105-4) ·
[Exomiser benchmark](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7230372/)

## 5. Deep patient representation / "patients-like-me" retrieval

Relevant to CHARMPheno's *profile-based matching* framing, though several are genomic-guided:
- **SHEPHERD** (Zitnik Lab, *npj Digital Medicine* 2025) — few-shot, knowledge-graph-guided
  rare-disease diagnosis; produces patient representations for **"patients-like-me"** retrieval
  and causal-gene discovery. Patient rep is phenotype-based but the task is oriented to genetic
  diagnosis. Closest in spirit to your "profile-based patient matching."
- **RD-Embed** (medRxiv 2026), **PERADIGM** (bioRxiv 2025), **medical-concept-embedding for
  rare disease** (2021) — representation-learning for graded clinical similarity / retrieval.
  RD-Embed is EHR-native retrieval (up to >50% top-10 retrieval); the direct comparator for
  your cosine-over-profiles retrieval claim.

Sources: [SHEPHERD (npj Digit Med)](https://www.nature.com/articles/s41746-025-01749-1) ·
[RD-Embed (medRxiv)](https://www.medrxiv.org/content/10.64898/2026.04.02.26350083v1.full) ·
[PERADIGM (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12714201/)

## 6. Infrastructure + evaluation context (cross-cutting)

- **OMOP for rare disease:** RD-CDM / OMOP customization (*Orphanet J Rare Dis* 2024) —
  context for how rare-disease phenotypes are represented in OMOP; AoU already uses OMOP
  conventions.
- **Evaluation lessons that corroborate your own findings:** the *Deep Learning for Rare
  Disease* scoping review (*J Biomed Inform* 2022) and the imbalance literature converge on
  points you've already internalized — **use AUPRC/detection-AP over AUROC** in low-prevalence
  regimes (your insight 0064), chart-review validation, cross-institution generalization, and
  the difficulty of CV with few positives. Your "AUC flatters a low-prevalence problem"
  observation is the field consensus, not an idiosyncrasy — good to cite as external support.

Sources: [OMOP for rare disease (Orphanet JRD 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11325822/) ·
[DL for rare disease scoping review (JBI 2022)](https://www.sciencedirect.com/science/article/pii/S1532046422002325)

---

## Where CHARMPheno sits (positioning notes)

1. **The real incumbents to beat/cite are PheRS and RarePT**, not Phenomizer. PheRS is the
   knowledge-based, interpretable, zero-training floor; RarePT is the learned, self-supervised,
   opaque ceiling. CHARMPheno's distinctive claim is the **middle you'd actually want**: a
   *learned* representation that is *also interpretable and generative* (reusable for
   patients-like-me, trajectories, on-device, autoregressive what-ifs) — a slot neither PheRS
   nor RarePT fills.
2. **sureLDA is the closest prior art you should engage directly** — "guided topic models for
   phenotyping" is exactly your PC-shaping family. Positioning CHARMPheno as the Bayesian-
   nonparametric, uncertainty-aware, DAG-aware generalization is defensible and specific.
3. **Your "information-limited on condition codes, ~3× lift" result is consistent with the
   field.** Wins in this literature come from (a) multimodal features (MixEHR, PheNet's labs),
   (b) chart-review-refined labels, (c) per-disease tuning — all things your ablations already
   probed. The honest, publishable contribution may be less "we beat case-finding" and more
   "an interpretable generative profile that matches discriminative case-finders while
   *also* powering retrieval/trajectories," plus the negative result that condition codes are
   near-sufficient and PC-shaping helps only in proportion to what the unsupervised fit misses
   (your insight 0066, mechanistically demonstrated in exp 0076 run 6).
4. **Cheap next moves in AoU:**
   - Run the **`phers` R package** on AoU phecodes for the rare6 set — a genuinely
     apples-to-apples, near-zero-effort baseline (easier than the Phenomizer/Exomiser port).
   - If the grad student's Phenomizer/PhEval work extracts HPO from AoU notes, **PhenoBrain**
     is the stronger, non-genomic, EHR-native phenotype-ranking baseline to target instead of
     bare Phenomizer.
   - Clarify with the grad student that **Exomiser/PhEval are genomic**; in the no-genomics
     regime only the phenotype-scoring subcomponent is comparable, and PhEval is best used as
     an **evaluation harness / phenopacket standard**, not a case-finder.

## Open threads / not yet chased (say the word)

- Specific rare6 diseases (MG, SLE, EDS/POTS, Guillain-Barré, Marfan, Osler/HHT, sarcoid,
  CIDP, FH) each have their own single-disease case-finding papers — worth a targeted pull if
  you want per-disease comparators.
- AoU-specific rare-disease / phenotyping publications (beyond the OMOP-CDM infra work).
- Prediction-constrained topic models (Hughes et al.) as the methodological citation for your
  PC head — named in your experiments; not separately searched here.
