# 0084 — Profile-eta at strength 1.0: the boost provably reaches the starved floor (legible profile tokens in E[log beta]) but cannot buy θ — a clean, wiring-validated null

**Date:** 2026-09-07
**Topic:** lda, gated-pc, eta-prior, hpoa, case-finding, readout
**Status:** Confirmed on exp 0116 vs 0113 (paired `readout-ab`: same bundle key, split,
readout code; 193 shared scored nodes, 91 credited / 102 uncredited). All figures are
model metrics / counts-of-nodes.

## Observation

Exp 0116 (= 0113 + the HPO-profile word-side eta prior, strength 1.0, one aligned
topic per credited node's block) is a **null on every readout axis, with the
internal control clean**:

- Macro 0.7804/0.5268 vs 0113's 0.7813/0.5255 (non-inferiority met); detection
  0.6362/0.7173 vs 0.6347/0.7156.
- **Paired per-node deltas: flat on both sides of the credit line.** Credited
  (n=91): median dAUC −0.0009, mean +0.0025, up/down 43/48. Uncredited internal
  control (n=102): median −0.0022, mean −0.0039, 47/55. No effect — and no
  leakage (the control moving ~0 validates the block-targeted wiring end-to-end).
- Starvation 72%, cliff at d4 — both unchanged from 0113.

**But the boost demonstrably landed exactly where designed.** The run log's
per-iter `η_boost[topics=132 nnz=28883 mass=440.1]` matches theory (440.6), and
the digest's STARVED deep cardiomyopathy topics — evidence ≈ 65 vs the flat
floor 60.7, i.e. sitting ON the boosted prior — now lead with their own
phenotype: hypertrophic CM d5 → "Hypertrophic cardiomyopathy · HOCM · Dyspnea";
restrictive CM d5 → "Amyloidosis · muscle weakness"; non-familial restrictive
d6 → "Amyloidosis · Fibrosis of lung · Interstitial lung disease". That is the
HPOA profile speaking through E[log beta] at the zero-count floor — in 0113
these rows were anonymous flat topics. Fed-but-misaligned topics are untouched
(dilated CM's ev 1.7e5 topic is still pregnancy-anchored): real counts dominate
the prior, by design.

## Reading

1. **Prior legibility ≠ θ movement.** Strength 1.0 doubles the topic's
   CONDITION-domain prior mass, but that is ~3.3 pseudo-mass against fed topics
   whose Σλ runs 10²–10⁶: in per-doc CAVI the tilted floor loses the θ
   competition to any topic with actual evidence, so the readout's features
   never change. The tilt is visible only where nothing competes (the starved
   rows' top-words).
2. **The faint depth gradient points the predicted way** (d6 +0.005, d7 +0.004
   median dAUC vs ~0/− above) — where starvation is total, the tilt helps a
   hair. Inside noise; recorded as a direction, not a result.
3. **This is the pre-registered failure read #2** ("everything flat → strength
   too low, try 3.0"), and the cleanest kind of null: mechanism verified at
   both ends (mass arithmetic + starved-row legibility + flat control), dose
   insufficient.

## Consequences

- The one-knob ladder continues: **0117 = 0116 with `profile_eta_strength: 3.0`**
  (pre-authorized by the plan's failure reads). If 3.0 is still flat, the next
  question is whether ANY eta-side dose can buy θ against fed competitors
  before distorting fed topics — i.e. whether the lever is eta at all.
- The credited-vs-uncredited CROSS-SECTIONAL gap (0.807 vs 0.773 median AUC) is
  selection (HPOA annotates different diseases), not effect; only the paired
  deltas carry causal weight. Keep using `readout-ab` for this read.
- Strength-1.0 profile-eta is a free legibility upgrade for starved topics
  (their top-words become their phenotype instead of noise) at zero readout
  cost — worth keeping ON for interpretability even where it buys no AUC.

**Setting context:** exp 0116 (CV branch MONDO:0004995, native-Mondo DAG,
tpn=5, K=1498, random init, fit-only 50 iters, multidomain, strip both,
holdout 0.2, seed 42) vs exp 0113, identical but flat eta; readout
pc_topics_lr, L-BFGS cap 200, theta top-m 256. Prior: rolled-up HPOA profiles
(132/299 credited), freq × IDF, one boosted topic per block.
