# Insight 0051 — Ridge-conditional DAG-offset intervals order uncertainty correctly but are severely overconfident absolutely (coverage ~0.13), because they are built from mean-field-biased ψ-means

**Date:** 2026-07-13
**Branch:** pg-stm
**Topic:** pg-stm | dag-offsets | uncertainty | mean-field | calibration | diagnostics
**Status:** Observed
**Relates to:** the DAG/ontology PG-STM model core v1 (Task 6); insight 0050 (anchor offsets un-identified; measure increments); the mean-field-bias thread (insights 0044 / 0047 / 0048) and the deferred read-out-honesty spec.

**Context:** v1 of the DAG additive-offset PG-STM adds a per-node offset-interval read-out —
`offset_cov_diag = σ²·diag((WᵀW+diag(penalty))⁻¹)` at the converged VI ψ-mean, with 90% intervals
`B ± 1.645·√var`. Task 6 set out to check this is calibrated (≈90% coverage) and that a
data-scarce node's intervals are wider than a well-populated node's. The identifiability fix from
insight 0050 (measure *identified subtype increments*, not dummy-trapped anchors) was applied
first.

**Finding:** A coverage probe with the **truth redrawn every rep** (the correct marginal-coverage
protocol; the initial test wrongly fixed one offset draw across reps) at near-real config
(K=14, V=250, 6 reps) gives:

- **Absolute coverage ≈ 0.13** (nominal 0.90) — an order of magnitude too low, and *systematic*
  (not an unlucky fixed seed).
- **Width ordering is correct**: scarce/populated interval-width ratio = 2.0. Robust — the ratio
  depends only on the fixed design moments `Ainv` (doc-counts), independent of σ² and n_iter.
- **Recovery is fragile across draws**: per-rep recovery correlation of the populated node's
  offset ranges −0.29 … 0.82; the scarce node is essentially never recovered (−0.41 … 0.48).
- **Decisive detail:** even reps where the point estimate recovers well (r = 0.73, 0.82) still
  cover only 0.11–0.17 — the ridge sd (~0.06) is ~10× smaller than the actual error against the
  planted truth.

**Interpretation:** The ridge-conditional variance is computed *from the mean-field VI ψ-means as
if they were noise-free data*. Its σ² captures within-model residual scatter but is blind to two
things: (i) the ψ posterior uncertainty it conditions away, and (ii) the mean-field **bias** in
the ψ-means themselves (the same attenuation/bias documented for Σ in insights 0044/0047 and for
the β-conditioned read-out in 0048). Coverage is measured against the *planted* offset, so
bias-induced error dominates — and no variance formula built from the biased ψ-means can see it.
Hence overconfidence that no re-tuning inside this formula can fix.

**Consequences:**
- The offset-interval read-out is shipped as a **relative** signal only: the cross-node *width
  ordering* (scarce wider) is trustworthy and is what Task 6 asserts (ratio > 1.5). Absolute
  coverage is **not** claimed.
- Calibrated absolute intervals require debiased inference (Gibbs, or ψ-uncertainty propagation)
  — the same wall the Σ read-out hit — and are deferred to the read-out-honesty spec, not v1.
- General pattern (now three-for-three: Σ correlation 0044, Σ rank/scale 0047, β-conditioned
  read-out 0048, offset intervals 0051): mean-field point estimates in this model family carry
  a bias their self-reported uncertainty cannot detect, so only *relative / identified*
  functionals are reportable from VI; absolute calibration needs a debiased engine.

**Does not claim:** anything about real-data coverage (unmeasurable — Task 7's real-data
spurious-edge shrinkage is the transfer-side guard), nor that the width ordering transfers to real
corpora (it is a design-moment property, demonstrated on synthetic).
