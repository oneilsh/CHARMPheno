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
- 0152: a whole-Mondo hierarchical gated fit (2,694 nodes, tpn 1, v6 code map, all-ever
  gate) leaves new-diagnosis AUC on the shared ribbon nodes where it was (stacked 0.694
  -> 0.700, flat 0.612 -> 0.604); raw codes still lead by ~0.03. Neither gating (0146)
  nor hierarchy closes the gap: it is the θ representation itself.
- 0163 (2026-10-27, leaves-first + balanced + leaf seed, same bundle for every arm):
  θ rises to 0.710 stacked / 0.606 flat; frequency codes 0.735 / 0.631 (median +0.019,
  696 of 827 nodes, largest at 20-300 new cases). The gap narrowed from ~0.035 to ~0.02,
  all of it on the θ side. A naive model-informed selector (the node topics' top 2,000
  words) does worse than both (0.683 / 0.583): those words are the nodes' own codes,
  which by the incident rule are absent before the index for the patients being scored,
  and it draws almost no measurement tokens.
- 0163 `rawr2000` (2026-10-27): a model-informed selector with both flaws fixed (LDAvis
  relevance λ = 0.6 over node topics, own and descendant codes excluded, frequency's
  domain mix) reaches 0.717 / 0.623: above θ (median +0.005 stacked, +0.016 flat) but
  below frequency codes (−0.012 / −0.003), with the deficit on nodes of 20-100 new cases.
  The model can pick predictors as good as θ, not better than raw frequency; what the
  frequency set adds is common, utilization-carrying codes, which the lift term drops.
  Next readout-only arms: λ = 1 (pure p(w|k), own codes still excluded), and θ plus
  frequency codes.
