# 0005 — LDA decomposes "background" into multiple flavors on patient-year docs
**Date:** 2026-05-12
**Topic:** lda | doc-units
**Status:** Observed

On patient-year docs at K=25, LDA did not produce a single mega catch-all.
Instead, it carved the high-mass region into three distinguishable
"background flavors":

- **Chronic-comorbidity follow-up** — HTN, GERD, chronic pain, anxiety,
  HLD (peak 0.027 HTN).
- **Acute-presentation** — Pain, Chest pain, Nausea, SoB, Vomiting,
  Anxiety. Reads as ED-visit year-bins.
- **Metabolic syndrome** — HTN, HLD, T2DM, GERD, Obesity, Asthma
  (peak 0.039 HTN, sharper than the chronic-bg variant).

All three carried similarly high α (0.04–0.05) compared to the
rare-phenotype topics (~0.036).

This is itself a finding about the corpus, not a failure to converge.
Patient-year docs have at least two distinguishable "regime modes" —
chronic follow-up years versus acute-event years — and LDA separated
them. Patient-lifetime LDA cannot do this decomposition because the
two modes coexist within every patient's bag.

**Implications.** The "catch-all" pattern is regime-dependent. On
patient-lifetime docs we expect (and saw) one big chronic-comorbidity
topic; on patient-year docs there are multiple legitimate "background"
shapes. Eval and human interpretation should not assume a single
background topic.

**Setting context.** K=25 LDA, online VI, patient-year docs from
condition_era with min_doc_length=30, η=0.04, batch-fraction 0.1.
Observed clearly by iter 10, stable through iter 12. Other settings
as in recent fit-eval runs.
