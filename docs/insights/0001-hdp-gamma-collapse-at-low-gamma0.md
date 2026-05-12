# 0001 — HDP γ-collapse at low γ₀
**Date:** 2026-05-12
**Topic:** hdp
**Status:** Confirmed

Initializing the HDP corpus concentration γ at O(1) (e.g. γ₀=1.0) causes the
optimizer to settle into a basin with very few active topics — we observed
~7 active topics at T=150 truncation. Initializing at γ₀ ~ O(T) (e.g. 50)
gives 70+ active topics on the same corpus.

The closed-form M-step is

    γ* = -(T-1) / Σ_t E[log(1 - W_t)]

where W_t are the GEM stick-breaking weights for corpus topics. The
iterates are self-consistent: shrinking γ makes the stick break early
(few-active regime), which makes the denominator small in magnitude,
which keeps γ small. There's no restoring force from low γ once you're
in that basin, so γ₀ must start above it.

**Implications.** Default γ₀ should scale with T. The robust prescription
is γ₀ ≈ T/3 to T/2; for T=150 we've been using γ₀=50 successfully. Don't
trust "low γ → fewer effective topics is good Occam pressure" — at the
γ₀=1 setting we observed, the model isn't choosing few topics, it's
trapped.

**Setting context.** Observed on AoU OMOP condition-era data with online
HDP, T=150, η=0.01, patient-document corpus and patient-year corpus both.
Compared γ₀=1.0 vs γ₀=50, all other settings as in recent HDP runs.
