# 0092 — Detection was the READ, not the representation: one root head that saw the background scores 0.81 where max-over-conditionals scored 0.63; the closure product is the calibrated marginal decoder and the per-node head is the conditional one — stacking HURTS within-cohort ranking by construction

**Date:** 2026-10-06
**Topic:** gated-pc, readout, detection, hslda, stacked-closure, calibration, exp 0123
**Status:** Confirmed on the CV branch (0123's saved fit, ridge 100, 193 scored nodes, 54,753 test persons) via the `--readout-stacked` re-readout (spec 2026-10-06 Part A). 0124 and 0132 pending.

## Observation

Detection (case vs background, persons, prevalence 0.635), three reads of one test split:

| read | AUC | AP |
|---|--:|--:|
| flat: max_c σ(z_c) — every recorded "detection" number since 0104 | 0.6299 | 0.7600 |
| root head alone — ONE logistic on θ fit on every train row | **0.8065** | **0.8859** |
| stacked: max_c Π_{a ∈ closure(c)} σ(z_a) | 0.8065 | 0.8859 |

Within-cohort ranking, stacked vs flat, paired over 193 nodes: median −0.041, 33 up /
160 down; depth 3 −0.011, depth 4 −0.039, depth 5 −0.057, depth 6 −0.125. Marginal ECE
over all docs: flat 0.08–0.48 by depth, stacked 0.003–0.015.

**Per-node AUC over ALL docs (de novo, y_c vs everyone — HSLDA's own metric), 224
scoreable nodes:** flat 0.7118 / AP 0.131 → stacked **0.8497 / AP 0.236**; paired median
**+0.116**, 219 up / 4 down. Gain by depth: 2 +0.018, 3 +0.033, 4 +0.097, 5 +0.143,
6 +0.214, 7 +0.204 (8–9: +0.36/+0.40, n=3). The deeper the node, the more the chain of
ancestors' discriminations is worth.

## Why

1. **The 0.60–0.63 detection was an artifact of the read.** Under the closure mask every
   node head is trained inside its parent's cohort; no head ever sees a background
   document, and the root head is the degenerate constant (observed only on foreground
   rows, all positive). "max over nodes of σ(z_c)" therefore pools 298 conditionals that
   are out of distribution on the background. One ridge logistic on the same θ, trained
   on everyone, reaches 0.81. The representation was never the problem.
2. **Stacked = root-only is structural on a single-root branch.** The one depth-1 node
   (cardiovascular disorder) is in every case's closure, so its head is constant 1.0 and
   P_stack(depth-1) = σ(z_0). The max over nodes is the root head. Pooled detection
   cannot tell the product from the root head here; the per-node read over all docs
   (de-novo, HSLDA's own metric, added to the block after this run) can.
3. **Stacking hurts ranking inside a known cohort, by construction.** P_stack(c) =
   P_stack(parent)·σ(z_c). Inside the parent's cohort the parent is KNOWN, so the
   ancestor factors are noise with respect to the sibling contrast, and each extra
   factor adds more — the depth gradient. The per-node head is the right conditional
   decoder; the product is the right marginal decoder. HSLDA's product is for the
   marginal question. The two reads must not be collapsed into one headline.
4. **De novo, the product wins by as much as it loses within-cohort, and for the mirror
   reason.** A deep node's conditional was fit against siblings inside a small cohort and
   knows nothing about the other 99% of the population; the product multiplies in every
   ancestor's "is this person even in my cohort", each of which WAS fit on the right
   contrast at its level. The two paired reads (−0.041 within-cohort, +0.116 de novo)
   are not in tension: they measure the two decoders on their own questions.
5. **The product is a calibrated marginal.** ECE 0.003–0.015 over all docs at every
   depth, vs 0.08–0.48 for the conditionals read as marginals. For VOI / a diagnostic
   aid this is the number that had never been available.

## Consequence

Detection stops being a representation complaint. The readout now has two decoders with
two uses: σ(z_c) for "which child, given the parent" (ranking, unchanged) and P_stack
for "does this person have c, de novo" (0.85 macro AUC over 224 nodes, calibrated,
carries the root's background head) — the per-node case-finding score the program never
had. The open question Part B (0132, flat topics) answers is whether the TOPIC side
still matters once the label side carries the hierarchy: compare 0132-stacked to
0123-stacked on the per-node all-docs read and on within-cohort ranking.
