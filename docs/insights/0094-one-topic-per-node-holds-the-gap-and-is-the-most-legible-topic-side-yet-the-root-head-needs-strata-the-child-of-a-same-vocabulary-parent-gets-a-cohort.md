# 0094 — One topic per node (tpn=1, K=306) holds the gate's gap on ranking (+0.008 over tpn=5, +0.038 over flat), de-novo AUC 0.855, 0% starved, the most legible topic side of the series; what it loses is the root head (0.781 vs 0.807) and so de-novo AP — a width effect, the root reads strata; and a node whose vocabulary is its parent's gets a cohort stratum, not a signature

**Date:** 2026-10-07
**Topic:** gated-pc, tpn, stacked-closure, legibility, deflation, ancestor-capture, exp 0133, exp 0123, exp 0132
**Status:** Confirmed on the CV branch: 0133 (tpn=1, spectral, K=306) vs 0123 (tpn=5, K=1498) vs 0132 (flat, K=1498), ridge-100 flat head + stacked closure head, named digest.

## Observation

| read | tpn=5 (0123) | flat (0132) | tpn=1 (0133) |
|---|--:|--:|--:|
| within-cohort macro AUC | 0.795 | 0.761 | **0.801** (paired +0.008 / +0.038) |
| root head | 0.807 | 0.798 | 0.781 |
| stacked de-novo AUC / AP | 0.850 / 0.236 | 0.836 / 0.185 | **0.855** / 0.198 |
| starved | 1% | — | **0%** |

Digest: textbook topics wherever a node's vocabulary is distinct at its level (AF,
paroxysmal AF, cardiomyopathy, systolic/diastolic/congestive HF, hypertrophic, extrinsic,
alcoholic, rheumatic, Takotsubo). A cohort stratum wherever it is not (dilated CM and
intrinsic CM → the young-women/pregnancy cohort; persistent AF → diabetes; non-familial
restrictive CM → asthma; heart failure d3 → COPD/OSA/CKD comorbidity).

## Why

1. **The decoder no longer needs the strata inside the block.** Under the stacked head a
   block's job is one signature; five topics were four strata and one signature (insight
   0093's digests), and the signature was all the head used. One topic per node gives the
   head the same thing in a legible unit, and deeper nodes gain (d5–d7 +0.016 to +0.019).
2. **But the ROOT head is a strata reader.** Case-vs-background is decided on population
   strata (0132: 0.798 from flat strata alone), and K=306 offers eight. Every product
   inherits the root's factor, so de-novo AP and the within-cohort stacked read fall with
   it. Not a tpn effect: a width effect. The blend (flat background + one topic per node,
   K matched) is the direct test.
3. **Parent-wins-shared-vocabulary is the one failure mode left.** Forward deflation hands
   vocabulary shared along a closure to the ancestor that claims it first; the child's
   residual is then its seed documents' dominant cohort. The peripartum cohort inside
   dilated CM (0127–0131) is the same mechanism at tpn=1. It is a topic-side legibility
   problem only — the HEAD still decodes DCM (within-cohort ranking is up), because the
   head reads the parent's topic too.

## Consequence

tpn=1 is defensible on every axis but width. Exp 0134 = 0133 + 1,200 background topics
(K=1,498): if the root head returns to ~0.80 and de-novo AP to ~0.24 while the per-node
topics stay as legible, the program has its unit — one signature per node over shared
strata, with the closure product as the decoder — and the last open item is the
same-vocabulary child.
