# 0014 — Patient-year LDA NPMI was bimodal under the OLD metric (artifact); doc-unit-difference findings survive
**Date:** 2026-05-12 (revised same day after re-eval)
**Topic:** lda | doc-units | npmi | diagnostics
**Status:** Partially refuted — bimodal-distribution framing was metric-artifactual; the doc-unit-specific phenotype-recovery finding survives
**Refuted-by-data:** Same checkpoint re-evaluated under [ADR 0017 revision](../decisions/0017-topic-coherence-evaluation.md#revisions) gives a unimodal positive distribution; see [0018](0018-full-corpus-plus-threshold-yields-unimodal-positive-npmi.md)

> **Re-eval result (2026-05-12, same patient-year LDA K=25 checkpoint):**
>
> | stat   | OLD metric (holdout, floor=−1) | NEW metric (full, min_pair=3) |
> |--------|-------------------------------|-------------------------------|
> | mean   | −0.036                        | +0.138                         |
> | median | −0.037                        | +0.087                         |
> | stdev  | 0.265                         | 0.113                          |
> | min    | −0.549                        | +0.023                         |
> | max    | +0.509                        | +0.477                         |
>
> The "bimodal patient-year vs unimodal patient-lifetime" framing was
> the metric talking. Under the new metric, patient-year is also
> unimodal positive — and its stdev (0.113) is actually *lower* than
> the pre-revision patient-lifetime stdev (0.133), the opposite
> direction of the "2× spread" claim below. **Specifically refuted
> sub-claims:**
>
> - "bimodal distribution shape" — was the rare-phenotype floor; gone now.
> - "stdev 2× patient-lifetime's" — direction reverses with corrected metric.
> - "real-but-rare phenotypes get NPMI-penalty negative scores" — gone
>   now. SLE went from −0.53 to +0.087 (cov=50%); sarcoidosis from
>   −0.06 to +0.075 (cov=91%); kidney transplant from −0.14 to +0.10
>   (cov=84%); pregnancy from −0.09 (contaminated) to +0.24 (clean, cov=74%).
>
> **Surviving findings:**
>
> - Patient-year surfaces transient phenotypes (pregnancy, acute-event
>   signatures) that patient-lifetime *structurally cannot* produce as
>   distinct topics. Confirmed by their presence as high-NPMI topics
>   under both metrics (the metric change shifts their scores but
>   doesn't make them appear or disappear).
> - The K=25 patient-year LDA recovers more distinct rare phenotype
>   classes than patient-lifetime LDA at the same K, because rare-
>   phenotype topics get their own slot rather than being absorbed
>   into broader catch-alls.
> - The bottom-portion topic readouts in the old data weren't all
>   "junk drawers" — many were rare-phenotype topics being scored
>   negative by metric artifact. See e.g. topic 23 (SLE), now
>   recognizable as a clean phenotype under the corrected metric.

Side-by-side K=25 LDA NPMI evaluations on the same OMOP condition-era
corpus, varying only the doc unit:

| metric | patient-lifetime | patient-year | delta |
|--------|-----------------|--------------|-------|
| mean   | +0.142          | −0.036       | −0.18 |
| stdev  | 0.133           | 0.265        | 2.0×  |
| max    | +0.339          | +0.509       | +0.17 |
| min    | −0.345          | −0.549       | −0.20 |

Patient-lifetime gave a roughly unimodal moderate-coherence distribution
— every topic a chronic-comorbidity cluster of some flavor, scoring
+0.05 to +0.34. Patient-year gave a bimodal distribution with extreme
winners and extreme losers.

The winners are categories patient-lifetime *cannot* produce:

- topic 11 (+0.51): hospitalized-acute-event signature (Pericardial
  effusion, Chest pain, SoB, Bleeding, GI symptoms).
- topic 1 (+0.48): pure ED-visit signature (Pain, Nausea, Chest pain,
  SoB, Vomiting, Anxiety).

These genuinely co-occur in acute year-bins; in lifetime bags they smear
across decades of unrelated events.

The losers fall into two camps:

- **Rare-phenotype NPMI penalty** (real phenotypes, NPMI-floored): SLE +
  antiphospholipid + chemo-pancytopenia (−0.53), pulmonary sarcoidosis +
  allergic-inflammatory cluster (−0.41). Per
  [0007](0007-npmi-zero-pair-floor-penalizes-rare-phenotypes.md).
- **Genuine junk-drawer topics** from K-undersizing: somatic
  dysfunction + tongue swelling + nutritional anemia (−0.55) is real
  K-underbudget mixing.

**Implications.** When the doc unit changes from lifetime to year-binned
on this corpus, NPMI mean is a misleading summary statistic — its drop
reflects the metric punishing rarity, not the model getting worse. The
**NPMI distribution shape** (peakedness, mode count, tail length) is
the real diagnostic. A bimodal distribution with a few extreme winners
and a populated negative tail is the signature of a doc unit that
exposes new phenotype classes the metric can't fully appreciate.

This argues for either: a smoothed NPMI variant that floors rare-pair
contributions less aggressively, or supplementing mean-NPMI with a
"top-decile NPMI" metric that captures the high-coherence wins
year-binning produces.

**Setting context.** Both runs: K=25 LDA, online VI, condition_era
loader, person_mod=10, vocab_size=10000, min_df=10, holdout_fraction=0.2,
tau0=64, kappa=0.7, 20 iters. Doc unit only difference; patient-year
run included min_doc_length=30 cutoff and era replication.
