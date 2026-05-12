# 0009 — Year-binning intensifies chronic-background dominance for HDP
**Date:** 2026-05-12
**Topic:** doc-units | hdp
**Status:** Observed

The hypothesis going into the patient-year HDP run was: chronic
background will absorb into one topic, freeing the rest of the truncation
budget for phenotypes. The observed reality was the opposite.

A 60-year-old with HTN/HLD/T2DM contributes 20+ year-bins to the corpus,
*every one of which* is dominated by those three concepts. The chronic
background appears in nearly every doc with high intra-doc prevalence,
which is exactly the regime that drives HDP to grow a giant catch-all
topic — and what we got: topic 149 climbed monotonically to E[β]=0.224
by iter 12, with 14 vestigial "catch-all variants" alongside, while
only ~4 rare-phenotype topics emerged (cerebrovascular, substance use,
pregnancy, mental-health cluster).

Meanwhile LDA at K=25 on the **same docs** produced ~12 phenotype topics
plus a 3-flavor background decomposition (see
[0005](0005-lda-decomposes-background-into-flavors.md)). The doc unit
isn't the problem — the doc-unit-model interaction is.

**Implications.** Year-binning is not free for HDP. The GEM
stick-breaking rewards a small number of high-prevalence components,
and year-bins multiply the apparent prevalence of chronic conditions
relative to lifetime-doc baselines. If we want HDP on patient-year
docs to work, we likely need: (a) a higher min_doc_length to bias
toward acute year-bins, (b) warm-start from LDA λ to seed differentiated
topics, or (c) explicit modeling of "background vs phenotype" via a
hierarchical extension.

**Setting context.** Online HDP, T=150, γ₀=50, η=0.01, K_doc=15,
patient-year docs from condition_era with min_doc_length=30, era
replication on. ~12 iters of 20 (run killed early after diagnostic
trajectory was clear). Same corpus as [0005](0005-lda-decomposes-background-into-flavors.md).
