# Insight 0064 — The LR readout's aggregate-ranking edge (+0.12 ROC / 2.6x PR-AUC over theta-mass) yields ZERO FDR-controlled discoveries: applying the same Efron two-groups empirical-null per-node FDR to the LR and explain-away score matrices gives n=0 at every q (0.05/0.10/0.20), identical to theta-mass. ROC is average case-level ranking; FDR asks whether any individual (patient, node) score is extreme enough vs the length-matched background null to survive BH correction across ~31k patients — and at 4.5% prevalence with heavy foreground/background overlap (LR bg_fpr 0.43 @ 80% sens, precision 0.08), no threshold reaches q-controlled precision with any recall. So LR is a TRIAGE RANKER, not a discoverer/classifier; the buried-signal problem is NOT a lens artifact at the discovery level, it is information-limited. Resolves the LR-FDR un-chased lever (negative); reinforces multi-domain features (MixEHR) as the only path to FDR-defensible case-finding.

**Date:** 2026-07-24
**Branch:** case-finding
**Topic:** case-finding | fdr | lr-readout | explain-away | discovery-vs-ranking | null-result | information-constraint | decision

**Status:** Observed

**Relates to:** the collation retrospective (docs/reports/2026-07-23-case-finding-levers-retrospective.md,
un-chased lever #2). Built on the LR-FDR readout (spec 2026-07-23; SDD commits c2b49ab..ec3fe99):
the score-agnostic engine helper `fdr_discovery_report` now runs the identical FDR machinery
(`per_node_discoveries`, background docs = per-node/per-length-bin empirical null, BH per node) on the
LR + explain-away score matrices, printed beside the theta-mass FDR. Extends insights 0062/0063 (the
six model-side nulls) into the discovery readout.

## Setting

exp 0069 re-fit (rare6, 1yr lookback, learned alpha, symmetric deploy, n_bg 40, K=170, seed 42;
re-extracted bundle on a fresh cluster — reproduces the null within noise: LR ROC 0.774 vs the prior
0.779, mrr 0.577 vs 0.590). `make lr-readout ID=69` now prints a three-way FDR by_q table at
q in {0.05, 0.10, 0.20}, all scored through the SAME `fdr_discovery_report` (like-for-like verified):
theta-mass (from the manifest), LR @alpha=inf, explain-away @alpha=inf.

## Findings

1. **All three scorers: ZERO discoveries at every q.**
   - theta-mass: q=0.05 (n=0), q=0.10 (n=0), q=0.20 (n=0)
   - LR @alpha=inf: n=0 at every q
   - explain-away @alpha=inf: n=0 at every q
   LR's large aggregate advantage (ROC 0.774 vs theta 0.647 = +0.13; PR-AUC 0.215 vs 0.078 = 2.7x)
   does NOT convert into a single FDR-controlled per-node discovery.

2. **Not a bug — the machinery can discover, this data does not.** The engine helper's planted unit
   test (a node whose foreground scores sit clearly above the background null yields >=1 discovery at
   q=0.20 with precision 1.0) passes; the all-null test yields zero. So n=0 here is a property of the
   data, not the code. The theta-mass FDR (the older, reviewed evaluate path) is also zero and has been
   across 0067/0068/0069 — consistent.

3. **Why ROC wins but FDR is zero — discovery is a different, stricter question than ranking.** ROC-AUC
   is average case-level ranking (max-over-nodes case score separates true cases from background on
   average — LR does this better). The per-node FDR asks whether any individual (patient, node) score is
   so extreme vs the length-matched background null that it survives BH multiple-testing correction
   across ~31k patients. At 4.5% prevalence with the measured overlap (LR bg_fpr 0.43 @ 80% sens,
   precision 0.08), there is no threshold where the discovery set reaches q=0.20 precision (>= 80% true)
   with any recall — the top scores are still overwhelmingly background (~13000 FPs). Length-conditioning
   is correct (removes the length confound) and means raw magnitude does not help: a foreground patient
   must be extreme RELATIVE to same-length background, and none is. Hence zero.

## Decision / implication

- **LR is a TRIAGE RANKER, not a discoverer/classifier.** Its value is ordering candidates for review
  (the +0.12-ROC lift is real for that), not producing statistically-defensible individual case-finds.
  This makes rigorous what the low precision@sensitivity (0.05-0.08 across operating points) already
  implied.
- **The buried-signal problem is NOT lens-specific at the discovery level.** theta-mass buries signal
  that LR surfaces for RANKING (insight from the LR arc), but for FDR-controlled DISCOVERY both are
  zero — the limit is information, not the readout lens.
- **Resolves the LR-FDR un-chased lever (negative).** The retrospective's open question ("does LR's edge
  yield FDR-controlled discoveries where theta-mass got zero?") is answered: no. Every model-side lever
  AND the discovery readout on the winning lens are exhausted on the condition-code stream.
- **Only remaining path is information (meds/labs = MixEHR).** FDR-defensible rare-disease case-finding
  needs separating signal the condition codes do not carry.
- The LR-FDR readout stays as a validated, permanent capability (`make lr-readout ID=N` prints the
  three-way by_q table); it will surface discoveries the moment a richer feature set actually separates
  the classes.
