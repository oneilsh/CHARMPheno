# 0014 — Patient-year LDA NPMI is bimodal; patient-lifetime is unimodal
**Date:** 2026-05-12
**Topic:** lda | doc-units | npmi | diagnostics
**Status:** Observed

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
