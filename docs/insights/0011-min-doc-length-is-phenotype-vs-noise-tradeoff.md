# 0011 — min_doc_length is a phenotype-vs-noise trade-off, not just a noise floor
**Date:** 2026-05-12
**Topic:** doc-units | diagnostics
**Status:** Tentative

The default `min_doc_length=30` for patient-year docs was chosen from
the doc-length histogram knee (drops the bottom ~40% of bins which are
too sparse to be informative). The intuitive read was "filter noise."

A second-order effect surfaced when interpreting the patient-year LDA
results: heavy-utilizer years (30+ events) are dominated by chronic
comorbidity, while the **5–15 event range** likely contains the
phenotype-rich year-bins — a year where the patient had one acute
episode plus modest chronic comorbidity. Those are the bins where a
phenotype is most likely to *dominate* its doc, which is what enables
topic-level recovery.

By cutting at 30, we may be biasing the corpus *toward* chronic
background and *away* from the phenotype-discovery regime year-binning
was meant to enable.

This is currently a hypothesis, not a confirmed finding — we haven't
yet rerun with a lower cutoff to test it. The "phenotype topics did
emerge anyway" result of [0008](0008-patient-year-docs-surface-transient-phenotypes.md)
suggests the cutoff isn't catastrophic, but it might be leaving signal
on the table.

**Implications.** Worth a follow-up doc-size analysis with the
hypothesis explicit: plot doc-length distribution split by whether the
doc contains acute-condition concepts vs only chronic concepts. If the
acute-rich bins cluster below 30, lower the cutoff and rerun.

**Setting context.** Cutoff value originally chosen from
`analysis/cloud/doc_size_evals.ipynb` histogram. Implication noted
during interpretation of patient-year LDA at K=25 (see
[0005](0005-lda-decomposes-background-into-flavors.md),
[0008](0008-patient-year-docs-surface-transient-phenotypes.md)).
