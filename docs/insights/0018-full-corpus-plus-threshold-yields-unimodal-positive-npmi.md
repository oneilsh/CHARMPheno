# 0018 — Full-corpus reference + min-pair-count threshold yields unimodal positive NPMI distribution
**Date:** 2026-05-12
**Topic:** npmi | diagnostics | hdp | lda
**Status:** Confirmed — same pattern reproduced on HDP T=80 and LDA K=25, same corpus

The ADR 0017 revision (2026-05-12) replaced two NPMI defaults:
holdout-only reference → full corpus (train ∪ holdout); zero-pair −1
floor → min-pair-count threshold of 3 with coverage reporting. Two
runs under the new defaults — patient-year condition-era HDP at T=80
and LDA at K=25, both the same checkpoints that previously produced
the bimodal distributions documented in [0014](0014-patient-year-npmi-bimodal-vs-lifetime-unimodal.md) —
show a consistent qualitative shape change.

**HDP T=80, same patient-year condition-era checkpoint:**

| stat    | OLD (holdout, floor=−1) | NEW (full, min_pair=3) | delta  |
|---------|--------------------------|------------------------|--------|
| mean    | −0.036                   | +0.128                  | +0.16  |
| median  | −0.037                   | +0.107                  | +0.14  |
| stdev   | 0.265                    | 0.073                   | ÷3.6   |
| min     | −0.549                   | +0.040                  | +0.59  |
| max     | +0.509                   | +0.379                  | −0.13  |
| unrated | n/a                      | 0/80                    | —      |

**LDA K=25, same patient-year condition-era checkpoint:**

| stat    | OLD (holdout, floor=−1) | NEW (full, min_pair=3) | delta  |
|---------|--------------------------|------------------------|--------|
| mean    | −0.036                   | +0.138                  | +0.17  |
| median  | −0.037                   | +0.087                  | +0.12  |
| stdev   | 0.265                    | 0.113                   | ÷2.3   |
| min     | −0.549                   | +0.023                  | +0.57  |
| max     | +0.509                   | +0.477                  | −0.03  |
| unrated | n/a                      | 0/25                    | —      |

**The bimodality is gone in both runs.** No topic is below zero in
either; zero topics are unrated even at threshold=3. The mean rises
by +0.16 to +0.17 in both — a striking consistency suggesting the
full-corpus reference does the bulk of the structural lift, with the
threshold contributing the rare-phenotype rescue specifically.

The deep negative tails that previously contained genuine but rare
phenotypes (SLE: LDA topic 23 from −0.53 to +0.09; sarcoidosis: LDA
topic 6 from −0.06 to +0.07; kidney transplant: LDA topic 20 from
−0.14 to +0.10; pregnancy went from contaminated −0.09 to clean
+0.24) were the floor-at-−1 sparsity penalty, not coherence signal.

The max drops slightly because previously-best chronic-comorbidity
topics — whose top-N pairs were *always* well-represented in any
holdout — no longer get a free advantage over rare-phenotype topics
whose pairs were under-represented. The playing field is more level.

## Reading the new (NPMI, coverage) joint signal

Coverage = (pairs cleared threshold) / (total top-N pairs). It carries
information NPMI alone doesn't:

| NPMI | coverage  | interpretation                                                 |
|------|-----------|----------------------------------------------------------------|
| high | high      | Real, important phenotype (topic 1 ED-visit signature, +0.38 cov=100%) |
| high | low (<15%)| Micro-cluster artifact — a few rare pairs happen to co-occur, not a phenotype (topic 77 +0.33 cov=2%) |
| mid  | 100%      | Chronic-comorbidity catch-all variant (most of the +0.05–0.13 cov=100% topics) |
| mid  | moderate  | Rare-but-real phenotype (topic 7 T1DM +0.07 cov=91%; topic 16 eye +0.11 cov=50%) |

This is a meaningfully richer signal than the pre-revision metric.
"Look at the topic's NPMI rank" was the old workflow; "look at NPMI
*and* coverage together" is the new one.

## On HDP-specific implications

The eval surfaces something we already knew about HDP-on-this-corpus
from [0002](0002-hdp-catchall-hoarding-at-last-stick.md), [0009](0009-year-binning-intensifies-chronic-bg-for-hdp.md),
[0017](0017-hdp-gamma-sensitivity-is-prior-dominance.md): T=80 of capacity
goes to ~3–5 real phenotypes plus ~30 chronic-comorbidity catch-all
variants (cov=100%, NPMI +0.06 to +0.13) plus ~40 micro-cluster
artifact slots (high NPMI, low cov). Coverage makes this structure
legible. Previously these classes blurred together in the NPMI mean;
now they separate cleanly.

## On the (NPMI, coverage) joint reading for LDA

The LDA K=25 result also reveals the topic-class decomposition the
new metric makes legible — though differently from HDP, since LDA's
fixed K means no "wasted slot" tail:

- **High NPMI + high cov = real important phenotype**: topic 1
  (+0.48 cov=100% — acute ED-visit signature), topic 11 (+0.46
  cov=100% — hospitalized acute event, pericardial effusion +
  bleeding signature), topic 19 (+0.24 cov=74% — pregnancy).
- **High-to-mid NPMI + 100% cov = chronic-comorbidity background
  flavor**: topic 21 (+0.27 metabolic), topic 9 (+0.19 DM-CV),
  topic 18 (+0.18 skin/T1DM), topic 8 (+0.17 hospital course).
- **Mid NPMI + moderate cov = rare phenotype**: topic 23 SLE
  (+0.09 cov=50%), topic 20 kidney transplant (+0.10 cov=84%),
  topic 6 sarcoidosis (+0.07 cov=91%), topic 7 eye-cluster
  (+0.08 cov=89%).
- **Low NPMI + moderate-to-high cov = chronic-mixed undifferentiated**:
  topic 13 (+0.02 cov=75%), topic 4 (+0.07 cov=68%) — slots that
  ended up absorbing residual chronic-condition mixtures rather
  than crystallizing a phenotype.

LDA at K=25 doesn't produce the cov=2-15% high-NPMI micro-cluster
slots that HDP at T=80 does — those are an artifact of HDP's
T-budget exceeding the corpus's coherent-phenotype count. Coverage
cleanly separates that failure mode from LDA's K-budget allocation
behavior.

## Setting context

Two runs under the new metric, both online VI on the same patient-
year condition-era docs from AoU OMOP (min_doc_length=20 for HDP, 30
for LDA, person_mod=10, vocab_size=10000, min_df=10). HDP: T=80,
K=10, γ=50, η=0.01, subsampling=0.2, 20 iters. LDA: K=25,
subsampling=0.1, 20 iters, tau0=64 kappa=0.7. The same checkpoints
produced both the OLD and NEW metric numbers above; only the eval
metric changed between runs. Confirms the metric-revision effect is
consistent across model classes.
