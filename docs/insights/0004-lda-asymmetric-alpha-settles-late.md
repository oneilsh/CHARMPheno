# 0004 — LDA asymmetric α settles later than topic-word distributions
**Date:** 2026-05-12
**Topic:** lda | diagnostics
**Status:** Observed

At iter 12 of a K=25 patient-year LDA run, topic-word distributions
already showed crisp, recognizable phenotypes (pregnancy, sickle cell,
kidney transplant, SLE, CLL, sarcoidosis, substance use + PTSD, AMD,
etc.) — but the asymmetric α range was still narrow:
`α[min=0.036 max=0.049 mean=0.038]`, only a 1.4× spread.

If α had separated, we'd expect background topics (chronic-comorbidity,
acute-presentation, metabolic-syndrome — see
[0005](0005-lda-decomposes-background-into-flavors.md)) to have α ~ 0.10+
while rare-phenotype topics stayed at the floor.

ELBO was still climbing ~60K per iter at that point, so the fit wasn't
converged. The α M-step depends on aggregated E[log θ] across docs;
with short docs (5–30 tokens), the signal per doc is weak and α updates
slowly even when the topics themselves have differentiated.

**Implications.** Narrow α range early in training is normal — don't
interpret it as a problem with the fit. Run enough iters (let ELBO
plateau) before judging whether α has actually learned. For doc-unit
experimentation with short docs, this means α may need notably more
iters than topic-word convergence would suggest.

**Setting context.** K=25 LDA, online VI, patient-year docs from
condition_era with min_doc_length=30, η=0.04, batch-fraction 0.1.
Observed through iter 12 of 20 planned.
