# 0072 — PR-AUC reveals an observation-domain drag that LR-AUC almost entirely hides; the PPI strip cut observation's θ share but not its damage

**Date:** 2026-07-28
**Topic:** multidomain | case-finding | readout | metrics
**Status:** Confirmed
**Context:** exp 0071 (rare6, condition+drug+observation, lookback, full-batch) and 0072 (same, mini-batch) re-fit with the All of Us survey vocabulary stripped from the observation domain (`obs_exclude_vocab: PPI`), then read with the extended readout (LR-AUC + PR-AUC + precision@recall). 39,081 held-out docs, 6 rare6 anchors. Follows insight 0071, which measured ranking (AUC) only and found observation net-negative.

## Finding 1 — The strip fired and moved θ, but did not fix the drag

The fitted observation vocabulary contains **zero PPI** concepts (tally: SNOMED 811, LOINC 367, CPT4 182, HCPCS 102, OMOP Extension 14, NUCC 7, Visit 5, CMS Place of Service 5, Medicare Specialty 3, UCUM 2, Provider 1, APC 1), and the driver's filter-fired warning did not trigger. θ-contribution shifted materially: **observation 0.584 → 0.466, condition 0.291 → 0.373** (drug ~flat). So survey volume genuinely left the corpus.

Yet `drop:observation ≥ all` still holds for all six diseases in both fits. Removing the AoU survey bulk reduced observation's *volume dominance* without removing its *harm*.

(Incidental: the condition vocabulary still carries 10 PPI concepts — self-reported conditions. Not filtered; harmless.)

## Finding 2 (the headline) — PR and AUC disagree about the cost of a domain, by an order of magnitude

Comparing the **single-variable** contrast `all` vs `drop:observation` (identical except observation is in or out — NOT `all` vs `only:condition`, which also drops drug):

| disease | n+ | prev | AUC all → drop:obs | ΔAUC | PR all → drop:obs | ΔPR |
|---|---|---|---|---|---|---|
| Ehlers-Danlos | 145 | 0.0037 | 0.795 → 0.826 | **+3.9%** | 0.038 → 0.138 | **+263%** |
| Amyloidosis | 155 | 0.0040 | 0.786 → 0.805 | +2.4% | 0.017 → 0.038 | +124% |
| Myasthenia gravis | 80 | 0.0020 | 0.727 → 0.740 | +1.8% | 0.012 → 0.023 | +92% |
| Systemic lupus | 607 | 0.0155 | 0.731 → 0.755 | +3.3% | 0.104 → 0.146 | +40% |
| Sarcoidosis | 419 | 0.0107 | 0.755 → 0.777 | +2.9% | 0.050 → 0.066 | +32% |
| Scleroderma | 79 | 0.0020 | 0.958 → 0.963 | +0.5% | 0.090 → 0.088 | −2% |

(exp 0072 mini-batch reproduces the pattern: EDS 0.068→0.149, MG 0.028→0.052, SLE 0.099→0.148, amyloid 0.019→0.034.)

**AUC says observation costs a couple of points; PR says it can cost most of the precision.** The mechanism: ROC-AUC is dominated by the bulk of easy negatives, while average precision is sensitive to the *head* of the ranking — and that is where a high-volume, low-specificity domain does its damage, and where deployment operates. Insight 0071's AUC-only read therefore **understated** the observation drag.

Methodological implication for this project: **at rare-disease base rates, judge domain/feature decisions on PR, not ROC.** A domain can look free on AUC and be expensive on PR.

## Finding 3 — Drug is neutral-to-slightly-positive overall, decisive for one disease

`drop:observation` (cond+drug) vs `only:condition` is ~flat in PR: +0.002 (EDS), +0.001 (sarcoid), +0.005 (SLE), 0.000 (scleroderma), +0.005 (MG), 0.000 (amyloid) — exp 0072 the same shape. So adding drug to conditions costs nothing and occasionally helps.

The exception is **myasthenia gravis**, where drug carries independent signal: `only:drug` PR 0.030 vs `only:condition` 0.018 (exp 0071) — drug *alone* outperforms conditions alone, uniquely among the six, and cond+drug beats cond-alone by ~28% relative in both fits. Pyridostigmine is near-pathognomonic. Domain value is disease-specific, not global.

## Finding 4 — Deployability: real enrichment, not a classifier

Precision at recall for the best subset (`drop:observation`), exp 0071:

| disease | prev | prec@50% recall | lift | prec@80% recall |
|---|---|---|---|---|
| Scleroderma | 0.0020 | 0.055 | **27×** | 0.032 |
| Ehlers-Danlos | 0.0037 | 0.027 | 7.3× | 0.009 |
| Systemic lupus | 0.0155 | 0.079 | 5.1× | 0.024 |
| Amyloidosis | 0.0040 | 0.019 | 4.8× | 0.009 |
| Sarcoidosis | 0.0107 | 0.049 | 4.6× | 0.020 |
| Myasthenia gravis | 0.0020 | 0.009 | 4.5× | 0.003 |

5–27× enrichment over prevalence, but 1–8% absolute precision at 50% recall and 0.3–3% at 80%. This is a **triage ranker** — the same conclusion the single-domain LR work reached, now quantified on the multi-domain fit. Chasing high-recall operating points is not the right use; the top of the ranking is.

## Implication / next lever

The observation vocabulary's residual content is largely **administrative and billing**: ~306 of ~1500 concepts sit in CPT4 (182), HCPCS (102), NUCC (7), Visit (5), CMS Place of Service (5), Medicare Specialty (3), APC (1), Provider (1); the topic dumps show the rest is dominated by generic SNOMED status codes ("History of event", "Long-term current use of…", "Patient encounter procedure") and MIPS quality-measure codes ("Eligible clinician attestation", "Patient screened for…", "Prescription(s) generated"). That is billing exhaust, not clinical signal.

Testable with **frontmatter only** (the `exclude_vocabularies` mechanism generalizes; no code change):

```yaml
obs_exclude_vocab: PPI,CPT4,HCPCS,Visit,CMS Place of Service,NUCC,Medicare Specialty,APC,Provider
```

If that flips observation to neutral/positive, observation needs *curation*; if not, the generic SNOMED status codes are the culprit and a document-frequency cap (`max_df`) is the lever. Either way, judge the outcome on **PR**, per Finding 2.

Also confirmed again: `dead_node_report` says EMPTY while a mass-starved tail persists (deep rare6 sub-nodes at Σλ ≈ 43, several topics byte-identical at the prior). The check detects flatness, not mass-starvation, because the spectral seed plants a peak — a Σλ-vs-prior check would catch it. Not a fit bug; a blind spot in the sanity read.
