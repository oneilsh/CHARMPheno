# 0095 — Under the stacked head the topic side is second-order: four topic sides (tpn=5, flat, tpn=1, blend) decode within ~0.02 of each other on every read; the blend restores the root head (0.811, best) at a small within-cohort cost (−0.018 vs tpn=1); one topic per node is the legible choice at no decoding cost

**Date:** 2026-10-07
**Topic:** gated-pc, hslda, stacked-closure, tpn, blend, readout, exp 0123, exp 0132, exp 0133, exp 0134
**Status:** Confirmed on the CV branch, same corpus, same split, ridge-100 flat head + stacked closure head on all four fits.

## Observation

| read | tpn=5 (0123) | flat (0132) | tpn=1 (0133) | blend 1200 bg + 1 (0134) |
|---|--:|--:|--:|--:|
| within-cohort macro AUC | 0.795 | 0.761 | **0.801** | 0.782 |
| root head | 0.807 | 0.798 | 0.781 | **0.811** |
| stacked de-novo AUC / AP | 0.850 / **0.236** | 0.836 / 0.185 | **0.855** / 0.198 | 0.846 / 0.224 |
| starved | 1% | — | 0% | 1% |
| per-node legibility (digest) | strata splits, signature 1 of 5 | no node topics | textbook where vocabulary is own | same as tpn=1, sharper |

## Why it matters

1. **The decoder moved the numbers; the topic side moves the margins.** This week's
   decoder changes — a root head that saw the background (detection 0.63 → 0.81) and
   the closure product read de novo (+0.11 per node, 220 of 224 nodes, every topic
   side) — are five to ten times anything a change of topic side produced. Across four
   quite different topic sides every read sits within ~0.02.
2. **Width is the root head's lever, not tpn.** 0133's root loss (0.781) was the 306-wide
   θ; 1,200 random-init background strata brought it to 0.811, the best of the series,
   and de-novo AP with it (0.198 → 0.224).
3. **The per-node heads read a block slightly less well next to 1,200 strata** (−0.018
   within-cohort vs tpn=1, uniform by depth) while the digest shows the blocks intact and
   sharper. Candidate causes: halved node evidence per document (strata carry the generic
   mass) or the readout's top-256 θ truncation (84% of topics at K=306, 17% at K=1498).
   The second is one re-readout (`--readout-theta-topm 0`).
4. **A wide background is not anchorable and should not be.** Anchoring 1,200 topics
   blew up the greedy (O(n²Vd)) and then the per-word recovery; random-init strata were
   what 0132 proved works. `spectralBgAnchors` keeps 0133's eight deflation seeds.
5. **The same-vocabulary child is the one legibility failure left**, and no topic side
   fixes it: a child whose words are its parent's gets its seed cohort (DCM → young women
   / pregnancy) under tpn=5, tpn=1, and the blend alike. Forward deflation gives shared
   vocabulary to the ancestor; a background with the cohort in it does not change that.

## Consequence

With decoding a wash, the topic side is chosen on interpretability, and one legible
topic per node (0133/0134) is that choice. Two open items, in order: the top-m
re-readout on 0134 (decides whether the blend's −0.018 is truncation), and the
same-vocabulary child (a deflation-order question, not a background one).
