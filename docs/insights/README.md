# Modeling & Experimental Insights

This directory captures *learned things about the modeling regime or the data*
that aren't recoverable by reading the code. Insights complement ADRs:

- **ADRs** (`docs/decisions/`) record *why we chose X over Y* — architectural
  and organizational decisions. Forward-looking, prescriptive.
- **Insights** (`docs/insights/`) record *what we observed when we tried
  things* — empirical findings about how models behave on this data, which
  diagnostics actually discriminate, which failure modes are real, which
  hypotheses didn't survive contact with reality. Backward-looking,
  descriptive.

When a new modeling phenomenon, failure mode, or counterintuitive result
emerges from a run, add an insight here with the next four-digit number.
Existing insights can be marked **Refuted by NNNN** in their header when a
later run overturns them — don't delete; the trajectory of what we believed
and when is itself useful.

## Format skeleton

    # NNNN — Short Title
    **Date:** YYYY-MM-DD
    **Topic:** hdp | lda | doc-units | diagnostics | svi | ops | npmi
    **Status:** Observed | Confirmed | Tentative | Refuted by NNNN

    [Narrative body — typically 100–400 words. Subsections OK if useful
    (Observation / Interpretation / Implications), but optional.]

    **Setting context:** One short paragraph naming the run setup that
    revealed this — model, doc-unit, K/T, key hyperparameters that differ
    from recent defaults. Detail level: "trying with very large K values
    revealed X; other settings as in other recent patient-year LDA runs"
    is enough. The point is so a future reader knows whether the
    observation likely generalizes or was specific to that regime.

## Status meanings

- **Observed**: seen in one run; could be regime-specific. Default for new
  insights.
- **Confirmed**: reproduced across ≥2 distinct runs (different seeds, doc
  units, or hyperparameter neighborhoods).
- **Tentative**: noticed but with confounders that prevent ruling out
  alternative explanations.
- **Refuted by NNNN**: a later insight contradicts this one; keep the entry
  but link forward.

## What does NOT go here

- Facts about how the code works → code comments / docstrings.
- Architectural choices → ADRs in `docs/decisions/`.
- Coding conventions or agent-collaboration rules → AGENTS.md.
- Run logs and outputs → not committed; insights distill from runs but
  don't replicate their raw output.

## Index

(Append entries here as new insights are written, newest at top.)

- [0019](0019-lda-large-k-with-full-convergence-gracefully-unused-slots.md) — LDA at large K with full convergence gracefully under-uses excess capacity, no micro-cluster artifacts (lda, doc-units, diagnostics, hdp)
- [0018](0018-full-corpus-plus-threshold-yields-unimodal-positive-npmi.md) — Full-corpus reference + min-pair-count threshold yields unimodal positive NPMI distribution (npmi, diagnostics, hdp)
- [0017](0017-hdp-gamma-sensitivity-is-prior-dominance.md) — HDP γ-sensitivity reveals prior-dominated outcomes (hdp, diagnostics)
- [0016](0016-condition-era-vs-occurrence-not-comparable.md) — condition_era and condition_occurrence runs are not directly comparable (doc-units, npmi, diagnostics)
- [0015](0015-crisp-topics-can-regress-when-k-undersized.md) — Crisp topics can regress in late iters when K is undersized (lda, diagnostics)
- [0014](0014-patient-year-npmi-bimodal-vs-lifetime-unimodal.md) — Patient-year LDA NPMI is bimodal; patient-lifetime is unimodal (lda, doc-units, npmi)
- [0013](0013-spark-scaling-driver-bottleneck.md) — Spark executor count past ~20 doesn't reduce iter time (ops, svi)
- [0012](0012-svi-batch-fraction-vs-iter-count-tradeoff.md) — SVI batch-fraction and iter count must be tuned together (svi)
- [0011](0011-min-doc-length-is-phenotype-vs-noise-tradeoff.md) — min_doc_length is a phenotype-vs-noise trade-off (doc-units, diagnostics)
- [0010](0010-npmi-not-comparable-across-doc-units.md) — NPMI absolute values are not comparable across doc units (npmi, doc-units)
- [0009](0009-year-binning-intensifies-chronic-bg-for-hdp.md) — Year-binning intensifies chronic-background dominance for HDP (doc-units, hdp)
- [0008](0008-patient-year-docs-surface-transient-phenotypes.md) — Patient-year docs surface transient phenotypes that lifetime docs smear (doc-units, lda)
- [0007](0007-npmi-zero-pair-floor-penalizes-rare-phenotypes.md) — NPMI floors at −1 for zero-pair counts, penalizing rare phenotypes (npmi, diagnostics)
- [0006](0006-spread-and-eff-k-interpretation.md) — λ-spread and effective-K must be read together (diagnostics)
- [0005](0005-lda-decomposes-background-into-flavors.md) — LDA decomposes "background" into multiple flavors on patient-year docs (lda, doc-units)
- [0004](0004-lda-asymmetric-alpha-settles-late.md) — LDA asymmetric α settles later than topic-word distributions (lda, diagnostics)
- [0003](0003-hdp-vs-lda-per-iter-cost-on-short-docs.md) — HDP per-iter cost grows faster than LDA on short documents (hdp, svi, ops)
- [0002](0002-hdp-catchall-hoarding-at-last-stick.md) — HDP catch-all hoarding at the last stick (hdp)
- [0001](0001-hdp-gamma-collapse-at-low-gamma0.md) — HDP γ-collapse at low γ₀ (hdp)
