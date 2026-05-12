# 0003 — HDP per-iter cost grows faster than LDA on short documents
**Date:** 2026-05-12
**Topic:** hdp | svi | ops
**Status:** Observed

On patient-year documents (5–50 tokens), HDP at T=150 ran at ~115s/iter
while LDA at K=25 ran at ~53s/iter on the same corpus and cluster — a
2.2× gap. On patient-lifetime documents (hundreds of tokens) we'd seen
HDP only ~1.3× slower than comparable LDA.

The likely cause: HDP's per-doc inner loop computes digamma /
E[log θ] terms over T topic slots for every doc. When docs are short,
the per-doc fixed overhead dominates the per-token work, and HDP's
T=150 vs LDA's K=25 makes that overhead 6× larger. On long docs, the
per-token work dominates and the ratio compresses.

**Implications.** When experimenting with short-document corpora
(year-binned, visit-binned), expect HDP iteration time to balloon
disproportionately. For exploratory iteration speed, LDA may be the
better choice on short docs even when you'd prefer HDP's nonparametric
properties. Save HDP for the runs where you're committed to the
modeling assumption, not for fast doc-unit experimentation.

**Setting context.** Dataproc cluster with 4 vcores × 6 GB executors,
~30 executors, online VI driver. Same corpus (AoU OMOP condition-era
patient-year docs after min_doc_length=30 cutoff) for both runs. Other
hyperparameters as in recent fit-eval runs.
