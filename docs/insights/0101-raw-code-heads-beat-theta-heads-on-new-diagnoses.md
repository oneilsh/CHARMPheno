# 0101 — Raw-code heads beat θ heads on new diagnoses (same all-patient AUC)

**Setting:** ribbon record 0137 (gated LDA, K = 3,145, α = 0.5, label-year gate; 0146's
all-ever gate gives the same numbers), readout heads on the 698-node Mondo DAG,
distributed L-BFGS at l2 100, stacked arm, test split 54,753; new-diagnosis AUC under
the whole-history prior rule (0145). Raw arm (0147): the same heads on binary presence
of the 2,000 most frequent train tokens (680 condition, 978 measurement, 342 drug),
under-converged at 200 iterations.

## Observation

| macro AUC | raw codes | θ |
|---|--:|--:|
| all patients, stacked | 0.819 | 0.818 |
| new diagnoses, stacked (492 nodes) | **0.733** | 0.694 |
| new diagnoses, flat | **0.641** | 0.612 |

Equal on all patients, raw ahead by ~0.04 on new diagnoses in every prior-share band.

## Why it matters

- The all-patient AUC is the wrong yardstick for a representation: it cannot tell θ from
  2,000 codes, because both recognise prevalent cases equally well.
- θ, as folded in and read out today, loses new-diagnosis signal. Candidate causes,
  each a cheap readout-side test on the saved fit: (1) θ is LENGTH-NORMALIZED — record
  size (utilization) predicts new diagnoses and binary codes carry it, θ does not;
  (2) θ mass is dominated by frequent tokens (`[measured]` is 47% of measurement mass,
  insight 0099), so a rare predictive code moves θ little while a binary feature gives
  it full weight; (3) compression — K = 3,145 topics absorb specific codes into broad
  background topics.
- For the framings the topic model's value is interpretability (reasons, profiles,
  subtypes); for ranking, θ alone is not the best input. A hybrid (θ plus length, or θ
  plus codes) is the obvious candidate and is testable without a refit.
- Gating (0146) and head labels (0144) are ruled out as the lever; this is the first
  result that points at the representation.
- By disease size the raw advantage is LARGEST for the rarest nodes (+0.05 median for
  20–50 incident positives, +0.02 for ≥ 1000): θ does not share strength toward the
  rare end at this K; it blurs it.
