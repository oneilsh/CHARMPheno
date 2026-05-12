# 0008 — Patient-year docs surface transient phenotypes that lifetime docs smear
**Date:** 2026-05-12
**Topic:** doc-units | lda
**Status:** Observed

On patient-year LDA at K=25, we recovered ~12 recognizable phenotype
topics, several of them **transient** by nature:

- **Pregnancy** — Finding related to pregnancy, Complication occurring
  during pregnancy, High risk pregnancy, Third trimester, Second
  trimester, Disorder of pregnancy. Crisp.
- **Substance use + PTSD** — PTSD, Nicotine dep, Alcohol dep, Alcohol
  abuse, Bipolar disorder. Crisp.
- **Pericardial effusion / mediastinal signature** — Pericardial
  effusion, Chest pain, Lymphadenopathy. Sharp.

The chronic / non-transient phenotypes also showed up (sickle cell, SLE,
kidney transplant, CLL, sarcoidosis, AMD) but those would likely have
been recoverable from lifetime docs as well.

Mechanism: a transient phenotype occupies one year-bin nearly entirely
(pregnancy = 9–12 months of pregnancy-related events in one calendar
year). In a patient-lifetime bag, the same events are diluted by the
patient's other decades of unrelated events. Year-binning preserves
the within-doc co-occurrence structure that defines the phenotype.

**Implications.** This is empirical support for the
[ADR 0018](../decisions/0018-document-unit-abstraction.md) doc-unit
abstraction. The doc-unit isn't just a convenience knob; different
units expose different phenotype classes. Mixed reporting (some topics
recovered from lifetime, some from year-bins) may end up being the
useful presentation, not "pick the best single doc unit."

**Setting context.** K=25 LDA, online VI, patient-year docs from
condition_era with min_doc_length=30, era-replication on (each event
contributes to every year its era spans), η=0.04, batch-fraction 0.1,
~12 iters of 20. Reference comparison: prior patient-lifetime LDA on
the same condition-era loader.
