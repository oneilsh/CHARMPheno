# 0075 — Continuous disease-specific domain weighting has modest rare6 headroom; simple λ reliability does not identify it

**Date:** 2026-07-30
**Topic:** case-finding | multidomain | supervision | diagnostics
**Status:** Confirmed across two fitted artifacts

The nested-CV hybrid domain-weight readout was run on attested replications of
both recent condition+drug+observation rare6 fits. On the mini-batch artifact
(0073), macro median AP was 0.082 for the fixed condition+drug baseline and
0.087 for continuous nonnegative simplex weights. On the full-batch artifact
(0074), the corresponding values were 0.072 and 0.079. Thus continuous
disease-specific weighting exposes real but modest aggregate headroom over the
strong fixed baseline: about 6% and 10% relative, respectively.

The weights are not a universal domain policy. Five diseases selected a median
condition-only solution in both artifacts. Myasthenia gravis retained both
condition and drug (0.55/0.45 in 0073; 0.45/0.55 in 0074). Observation received
zero median weight for every disease. These are rare6 development-benchmark
findings, not evidence that condition should dominate other diseases or future
domains.

The discrete selector was unstable and underperformed the fixed baseline
(macro median AP 0.054/0.055). More importantly, three label-free weights
derived from fitted λ—distinctiveness, ownership, and their viability-weighted
product—underperformed both the fixed and continuous strategies. Ownership was
the strongest model-derived candidate (0.058 in 0073; 0.049 in 0074) and agreed
with myasthenia's condition/drug ordering, but did not provide a general
reliability estimator. Fitted structural specificity and token ownership are
therefore useful priors or inputs, not substitutes for task evidence.

**Implication:** do not turn rare6's condition-heavy solution into model
architecture. If supervised placement is pursued at scale, prefer one shared,
partially pooled mechanism across disease anchors and ontology structure over
independent per-disease classifiers. The present experiment is a diagnostic
ceiling for domain combination, not external validation.

**Setting context:** Experiments 0073 and 0074 are attested replications of
0072 mini-batch SVI and corrected 0071 full-batch VI: patient-level one-year
lookback documents, rare6 SNOMED DAG, condition V=5000, drug V=1291,
observation V=1500, K=180 (40 background plus 28 nodes × 5 topics), spectral
initialization, PPI observations excluded, defining markers stripped, seed 42.
The readout uses five repeats of nested stratified five-fold CV, α→∞ LR scores,
fold-local backgrounds/scales, a nonnegative three-domain simplex, and
tie-collapsing average precision.
