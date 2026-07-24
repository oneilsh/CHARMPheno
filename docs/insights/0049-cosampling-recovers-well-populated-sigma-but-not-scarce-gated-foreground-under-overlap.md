# Insight 0049 — Warm-started co-sampling recovers well-populated Σ blocks (to oracle) but NOT the data-scarce gated foreground under topic overlap; the read-out failure lands exactly on the rare-subgroup case

**Date:** 2026-07-13
**Branch:** pg-stm
**Relates to:** insight 0048 (condition-on-VI-β *freeze* fails under overlap), Fable round 3 (the co-sampling pivot: VI warm-start + joint (β,η,Σ) Gibbs), insights 0044/0047 (mean-field unfit for Σ). Revised queue item (1): the co-sampling plant.

## Setup

Fable's pivot after the freeze died (0048): don't freeze VI's β — **warm-start** a Gibbs chain at VI's β (`beta_init`) and **co-sample (β, η, Σ) jointly**, letting β relax off VI's sharpened point toward the truth that oracle-β Gibbs proves recovers Σ, while the warm start holds one labeling basin. Plant-and-recover on a realistic overlapping-topic corpus (topic_overlap=0.6, adj_cos 0.58, peak word prob 0.036), K=12 (bg4, 2×fg4), V=400, D=1000, planted intra-block correlation 0.40. Score co-sampling vs the oracle-β ceiling on β-recovery and Σ-MAE.

## Result — co-sampling helps, but only where the data is plentiful

| arm (co-sample = warm-start β_init=β_VI) | adj_cos(β) | background MAE | foreground-A MAE / sign | eigmin |
|---|---|---|---|---|
| oracle-β (ceiling) | 0.576 | 0.123 | 0.034 / 1.00 | 1.35 |
| VI mean-field | 0.258 | — | — | — |
| co-sample, Gibbs beta_eta=0.1 (sparse) | 0.223 | 0.217 | 0.488 / 0.50 | 0.56 |
| co-sample, beta_eta=1.0 | 0.415 | **0.136 ≈ oracle** | 0.625 / 0.33 | 1.68 |
| co-sample, beta_eta=2.0 | 0.482 | 0.487 | 0.617 / 0.33 | 2.86 |

Two mechanisms, in sequence:

1. **Naive co-sampling (sparse beta_eta=0.1) does not relax β.** The warm-started β stays sharp (adj_cos 0.223 ≈ VI's 0.258; drift corr 0.80 — barely moved). VI's sharp β is a **metastable mode of the full Gibbs**, self-reinforced by the sparse Dirichlet draw (sharp β → sharp token assignments → sharp counts → sharp β). The warm start anchors not just the labels but the sharpness.

2. **A smoother Gibbs β-prior lets β move, and recovers the well-populated block.** At beta_eta=1.0, adj_cos climbs to 0.415 and the **background block reaches oracle** (MAE 0.136 vs 0.123, sign 1.00). beta_eta=2.0 over-smooths and breaks the background (MAE 0.49). So there is a narrow sweet spot (~1.0) that fixes the easy block.

3. **The gated foreground block fails across the board** (MAE ~0.62, sign 0.33) even though oracle nails it (0.034, sign 1.00). No beta_eta rescues it.

## The split is the signal — and it lands on the rare-subgroup case

Background topics are active in **all** documents; foreground topics are **gated** — active only in their group's ~500 documents — and they overlap the shared background. Co-sampling relaxes β correctly where data is plentiful (background → oracle), but **not for the data-scarce, overlapping foreground** topics. That is precisely the regime the gated model exists to serve (rare / minority subgroups), so the read-out failure is not a corner case — it is the target case.

## Two confirmed side-findings

- **Q3 caveat holds:** the Gibbs Σ eigmin is **interior everywhere** (0.56–2.86), never at the 1e-8 floor. Under Gibbs the near-singularity lifts on its own (it was a mean-field attenuation artifact, insight 0047), so the IW-vs-MLE *conditioning* contrast demotes — LKJ+half-t would be chosen on shrinkage grounds, not conditioning.
- **Co-sampling fixes the scale channel:** no scale blowup (the frozen β-prior fix's 2.5× explosion, insight 0048, is gone). "Letting β move" fixed scale and background; it did not fix the scarce foreground.

## What this leaves open (architecture question, not a hyperparameter)

The co-sampling pivot is **partially validated**: it recovers well-populated Σ blocks to oracle but not the data-scarce gated foreground under overlap. The remaining gap is where the model's value lives, and closing it is an architecture question — candidates: a structured / collapsed VI that does not attenuate β at the source (Fable's Q2 alternative, on its kill criterion); foreground-targeted β treatment (stratified subsample over-weighting the scarce group; a separate smoothing for gated blocks); annealed β init rather than the sharp VI point; or accepting a per-topic identity-fidelity flag (Fable Q4) that marks the foreground correlations as low-confidence. To be decided with Fable rather than by further beta_eta sweeps.

## Caveats

- D=1000 makes the gated foreground (~500 docs) genuinely data-scarce; a larger corpus would raise foreground identification, but oracle-β already recovers it here (MAE 0.034), so the co-sample failure is β accuracy / mixing, not raw identification.
- One seed / one corpus; the background-recovers / foreground-fails split is the robust qualitative finding, exact MAEs are config-specific.
- Co-sampling used 600 sweeps (burn 300); the scarce foreground may need longer mixing, untested.
