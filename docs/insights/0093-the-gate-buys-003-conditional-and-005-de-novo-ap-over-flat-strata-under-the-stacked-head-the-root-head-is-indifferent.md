# 0093 — Under the stacked head, the gate buys +0.03 within-cohort AUC and +0.05 de-novo AP over flat strata (0.850/0.236 vs 0.836/0.185), uniformly by depth; the root head is indifferent to the topic side (0.806 vs 0.798); the stacking gain itself is the same size whatever the topics are

**Date:** 2026-10-06
**Topic:** gated-pc, hslda, stacked-closure, flat-lda, tpn, readout, exp 0132, exp 0123
**Status:** Confirmed on the CV branch: 0132 (flat K=1498 via a tpn=0 layout, random init) vs 0123 (gated tpn=5, spectral) vs 0124 (0123 + guided anchors), all three under the ridge-100 flat head and the stacked closure head (spec 2026-10-06).

## Observation

| read | gated 0123 | gated+guided 0124 | flat 0132 |
|---|--:|--:|--:|
| within-cohort macro AUC (flat head) | 0.795 | 0.792 | 0.761 |
| paired vs 0123, 193 nodes | — | −0.003 | −0.028 (33/160), uniform d2–d7 |
| root head detection | 0.806 | 0.804 | 0.798 |
| stacked de-novo macro AUC / AP (224) | 0.850 / 0.236 | 0.848 / 0.231 | 0.836 / 0.185 |
| de-novo stacking gain (paired) | +0.116 | +0.106 | +0.118 |

## Why it matters

1. **The gate's per-node blocks carry node signal flat strata do not.** Uniform −0.028
   within-cohort at every depth, −0.014 de novo, and −0.05 de-novo AP (−22% relative):
   the top of the ranked list is where the loss concentrates. Not "flat matches gated".
2. **But the gap is modest, and the root head does not care.** Case-vs-background is a
   strata question (0.806 vs 0.798). With no topic knowing the DAG the stacked decoder is
   within 0.014 of the gated one. Most of what the heads decode is in the strata; the
   gate adds a layer on top.
3. **The stacking gain is a property of the label side.** +0.106 to +0.118 de novo on all
   three topic sides; the closure product is worth the same whatever feeds it.
4. **Max-over-nodes detection is meaningless on every fit.** On flat strata it is 0.484,
   below chance: a within-cohort conditional transfers nothing outside its cohort.
5. **Guided anchors stay inert** under this decoder too (0124 ≈ 0123 on every read).

## Consequence

The gate stays, and its job is now sized: supply the +0.03 / +0.05 AP the strata lack,
as legibly as possible. The next pair is tpn=1 vs tpn=5 (0133 vs 0123) under the stacked
head — the first time tpn is asked with a decoder that needs one thing per block. If
tpn=1 holds the gap, the one-signature-per-node unit is back on the table; if it loses
it, the signal lives in the strata split and a blend (flat background + one residual
topic per node) is the arm after that.
