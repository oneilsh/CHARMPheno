# 0006 — λ-spread and effective-K must be read together
**Date:** 2026-05-12
**Topic:** diagnostics
**Status:** Confirmed

λ-row-sum spread (max−min of Σ_w λ_tw across topics) is necessary but
not sufficient for "topics are differentiating." High spread can mean
either:

- **Healthy**: one or two background topics have large mass and the
  phenotype topics have smaller, well-sharpened mass. Spread tracks
  differentiation.
- **Pathological**: one topic is hoarding mass while the others stay
  flat at minimum. Spread climbs but the model is collapsing, not
  differentiating.

The discriminator is the **effective-topic count** trajectory. If eff-K
is climbing or stable while spread climbs, the model is healthy. If eff-K
is *dropping* while spread climbs, mass is concentrating into one slot —
the pathological case.

A third signal worth pairing in: the largest topic's E[β] trajectory. A
monotonic climb past ~0.20 alongside dropping eff-K is the hoarding
signature (see [0002](0002-hdp-catchall-hoarding-at-last-stick.md)).

**Implications.** When summarizing per-iter diagnostics, always report
the triple (eff-K, top E[β], λ-spread) together. Reading any one of
them alone misleads. Spreadsheet/plot the trajectories rather than just
the final values — direction of change is more diagnostic than the
endpoint.

**Setting context.** This pattern was clearest in the side-by-side of
the patient-year HDP run (hoarding) vs the patient-year LDA run (healthy
differentiation) — same corpus, different model, very different
diagnostic trajectories. T=150 HDP and K=25 LDA, conditions as in
the runs documented in [0002](0002-hdp-catchall-hoarding-at-last-stick.md)
and [0005](0005-lda-decomposes-background-into-flavors.md).
