# Bias inversion (indirect inference) — closing §5 of the findings report

**Date:** 2026-07-10
**Script:** `scripts/marginalized_scale_bias_inversion.py` (reuses the exp 0046 harness).
**Regime:** production (K=60, V=5000, doc_len=44, D=1000, S=64), grid geomspace(0.5, 32, 13),
holdout f ∈ {0.5, 0.8, 0.95}. Plant scales c_true ∈ {2, 3.5, 5, 7, 10}. **Caveat:** the bias
map is measured on a synthetic β (`make_shared_beta`), a model of the real β; the inversion
assumes that bias transfers.

## Bias maps — recovered ĉ(c_true) per holdout

| c_true | MAP ĉ @f=0.5 / 0.8 / 0.95 | Marginalized ĉ @f=0.5 / 0.8 / 0.95 |
|---|---|---|
| 2.0 | 1.73 / 1.69 / 1.62 | 2.12 / 2.01 / 1.36 |
| 3.5 | 2.67 / 2.63 / 2.51 | 2.41 / 2.76 / 2.25 |
| 5.0 | 3.49 / 3.38 / 3.44 | 2.46 / 2.82 / 2.94 |
| 7.0 | 4.64 / 4.59 / 4.42 | 2.48 / 2.99 / 3.41 |
| 10.0 | 6.00 / 5.99 / 5.83 | 2.51 / 3.06 / 3.64 |

- **MAP is a clean instrument:** monotone, near-linear, f-stable, under-recovering by a
  roughly constant factor (ĉ/c_true ≈ 0.6–0.85). Invertible.
- **Marginalized is SATURATED:** at f=0.5, c_true sweeping 2→10 (5×) moves ĉ only 2.12→2.51.
  It is essentially flat, i.e. **nearly uninvertible** — a scale instrument that can't tell 3
  from 10. The mild slope only appears at f=0.95. So it cannot even be bias-corrected.

## Inversion — bias-corrected true scale implied by the real (exp 0047) readings

| f | MAP_full (4.61/3.75/3.65) → c_true | MAP_samp (5.30/3.90/3.80) → c_true | MARG_samp (2.36/2.65/3.76) → c_true |
|---|---|---|---|
| 0.5 | **6.95** | 8.42 | 3.21 |
| 0.8 | **5.60** | 5.85 | 3.25 |
| 0.95 | **5.41** | 5.71 | 10.0 (extrapolated — unreliable) |

## Answers to §5

1. **Do MAP and marginalized reconcile to a common corrected scale? NO.** Where the
   marginalized map is invertible at all they disagree by 1.8–2.6× (f=0.5: MAP→8.4 vs
   MARG→3.2). Fable's napkin "marginalized reads ~half of truth ⇒ 2.36/0.52≈4.5≈shipped 4.6"
   does **not** hold — it assumed a linear map, but the real marginalized map is saturated, so
   single-point ratio transfer is coincidental. The harness resolves it as Fable predicted it
   might: no reconciliation, because the marginalized instrument is too compressed to invert.

2. **Is the shipped scale (4.6) too high? NO — the opposite.** Bias-corrected through the clean
   MAP instrument, the true generative scale is **~5.4–7** (full corpus), *higher* than the
   shipped raw MAP 4.6 (consistent with the ~7.6 natural scale, insight 0030, and the
   plant-recover faithful band 5–7, insight 0037). The marginalized "low" reading (2.36) that
   suggested 4.6 might be too high was **Laplace saturation, not a genuine lower-scale signal**.
   §5's either/or resolves firmly to "Laplace bias pulling the objective down," not "proper
   uncertainty wants a lower scale."

3. **Does the corrected scale still drift across f? YES.** MAP_full bias-corrected: 6.95 →
   5.60 → 5.41 (drift 1.55). After removing the (f-stable) MAP instrument bias, the implied
   true scale still *decreases* with f — this is the **genuine per-document concentration
   heterogeneity, cleanly isolated and quantified** (~1.5 corrected-scale units), independent
   confirmation of the misspecification reading in insight 0044.

## Consequence

- **Marginalization is doubly dead** for this problem: not just biased but *uninvertible*
  (saturated) at the production regime — you cannot even bias-correct your way back to a scale
  through it.
- **The shipped MAP scale is if anything conservative (low), not high.** A live option this
  surfaces: ship a *bias-corrected* scale (raw MAP 4.6 → corrected ≈ 6) so generated units sit
  nearer the faithful band — at the cost of trusting the synthetic-β bias transfer. Not decided
  here.
- **The heterogeneity (residual f-drift ≈ 1.5) is real and quantified** — the case for the
  per-document scale (multivariate-t) fix, still gated on the burstiness/dedup check.
