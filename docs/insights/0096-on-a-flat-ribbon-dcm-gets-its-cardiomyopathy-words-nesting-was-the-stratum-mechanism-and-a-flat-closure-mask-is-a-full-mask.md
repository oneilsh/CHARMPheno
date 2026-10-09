# 0096 — On a flat ribbon DCM gets its cardiomyopathy words (nesting WAS the stratum mechanism); and a flat forest turns the closure mask into a full mask, so the hierarchy must supply the readout's negatives

**Date:** 2026-10-08
**Topic:** dismech-ribbon, flat-label-space, gated-pc, same-vocabulary-child, closure-mask, readout, exp 0135
**Status:** Confirmed on exp 0135 launch 2 (digest + census; the readout did not land — see finding 3). **Finding 1's mechanism is REVISED by insight 0097**: the DCM stratum was pregnancy documents mislabelled peripartum cardiomyopathy by a code-map defect, which nesting delivered into DCM's closure; flattening relocated it rather than fixing it. Finding 3 (the full mask) stands.

## Observation

1. **Dilated cardiomyopathy's block is textbook on the flat ribbon** — `Dilated
   cardiomyopathy · Cardiomyopathy · chronic systolic HF · LBBB · VT // carvedilol ·
   furosemide · spironolactone` plus an acute-decompensation profile — where under
   every nested topic side of exps 0127–0134 (tpn=5, tpn=1, the 1,200-topic blend)
   it was its parent's young-women / pregnancy stratum. The pregnancy vocabulary now
   sits entirely in peripartum cardiomyopathy's own block. Two things changed at once:
   no ancestor deflates DCM's seed (flat DAG), and a peripartum patient attests
   peripartum CM only (R1a). Either way the "same-vocabulary child" problem of insights
   0094/0095 was the nesting, not the data.
2. **A disease carries several coherent profiles, and tpn=3 is binding for the well
   coded ones**: EDS (POTS-dysautonomia-GI / MCAS-immune / musculoskeletal-migraine),
   T2D (complicated-insulin / uncomplicated-metformin / deficiency-anaemia), heart
   failure (systolic / acute / CKD-comorbid), CVID (3), Marfan (3). The census's
   cosine-to-background criterion over-counts (generic-symptom strata pass at 0.3–0.77
   while true signatures of common diseases score the same); identity — own codes in
   the top words — is the discriminator.
3. **On a flat forest the `closure` label mask is a full mask.** "Siblings as
   negatives" with the root as every node's parent means every foreground document
   observes every node: 70.7M train cells (vs a few hundred thousand nested), each head
   a 180k-row logistic, ~70 s per L-BFGS iteration, 4 h per readout pass, and the
   within-cohort read collapses into "this disease vs every other". The ill-conditioned
   `topm 0` solve (81/389 converged at 150 iterations) was a separate, additive cost.

## Consequence

The ribbon design's "hierarchy in the read, not the fit" has to include the mask: the
readout needs Mondo-derived negatives (siblings under the nearest Mondo ancestor with
≥ 2 members) as well as ancestor heads for the stacked product — WP-B is on the
critical path, before any AUC is read on a ribbon fit. The fit side is settled:
flat ribbon, R1a attestation, tpn_max 5 for the record, mass-resolved top-m.
