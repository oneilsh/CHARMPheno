# 0085 — The eta dose-response is answered: strength buys topic-profile ALIGNMENT monotonically (starved overlap 0.13 → 1.00) but zero discriminability, and starts costing globally at 3× — the profile prior is an interpretability lever, not an AUC lever

**Date:** 2026-09-07
**Topic:** lda, gated-pc, eta-prior, hpoa, case-finding, readout, dose-response
**Status:** Confirmed on the 0113 / 0116 (S=1) / 0117 (S=3) ladder — same bundle, split,
readout, credited set; paired `readout-ab` + the new `--profile-align` scorecard. All
figures are model metrics / counts-of-nodes-or-topics.

## Observation

Two rungs of `profile_eta_strength` give a clean, monotone dose-response — in
OPPOSITE directions on the two axes:

**Alignment (--profile-align: boosted topic vs its own positive profile):**

| group | 0113 flat | 0117 S=3 |
|---|--:|--:|
| starved credited (n=43): median E[β] mass / top-15 overlap | 0.039 / 0.13 | **0.303 / 1.00** |
| fed credited (n=89): median mass / overlap | 0.014 / 0.13 | **0.302 / 0.60** |

At S=3 a starved credited topic's top-15 IS its profile, verbatim (the digest's
restrictive/HCM rows are wall-to-wall `*`). The fed-topic shift is the mechanism
tell: ~10 added pseudo-mass cannot move a Σλ ≈ 10⁴ topic's β directly, so the
mass came through θ REALLOCATION — the tilted floor attracts profile-token
assignments in shared documents, and counts follow. The boost does buy θ.

**Discriminability (paired readout vs 0113):**

| paired median dAUC | S=1 (0116) | S=3 (0117) |
|---|--:|--:|
| credited (n=91) | −0.0009 | −0.0021 |
| uncredited control (n=102) | −0.0022 | −0.0046 |
| macro / detection AUC | 0.7804 / 0.6362 | 0.7778 / 0.6221 |
| deep-depth delta (d6/d7) | +0.005 / +0.004 | −0.009 / −0.005 |

Credited gain: zero at both doses. Everything else worsens monotonically with S;
detection gave back 0116's small gain, and the noise-level deep-depth hint
flipped sign. The control's drift is dose-scaled and directionally consistent
across both runs — NOT the wiring-bug read (0116's control was clean on
byte-identical wiring; only S changed): it is real spillover, θ drained from
honest topics into aligned-but-not-more-discriminative ones through the shared
documents each gate admits.

## Reading

1. **Alignment ≠ discriminability, quantified.** This is 0083's lesson from the
   mirror side: there, sharp-but-misaligned lost to weak-but-honest; here,
   sharp-AND-aligned still doesn't beat weak-but-honest, and crowds it out at
   dose. The readout head already extracts the label signal from ancestor/
   background θ; making node topics look like their HPOA profile does not add
   label information the head lacked — HPOA's rare-syndromic evidence and the
   corpus's common-acquired presentation are different distributions (the
   stage-2 coverage tail said so; the least-aligned nodes at S=3 are exactly
   the big common-disease nodes: hypertensive disorder, CAD, ov=0.00).
2. **The strength ladder STOPS.** Extrapolating to S≈30 ("one canonical
   patient", the doc-units unit) predicts more alignment and more global cost,
   not an AUC win. Do not run it as an AUC experiment.
3. **The operating point is S=1.0**: at that dose the prior is a free
   legibility upgrade (starved topics phenotype-named, 0084) at zero readout
   cost. It should be ON for interpretability in future record runs, and OFF
   or S=1 — never higher — when AUC is the metric.

## Consequences

- The θ-allocation lever for case-finding is NOT word-side eta. The remaining
  candidates from the record: label-side pull (PC revival, parked at co-fit ≳
  0.758 from 0103 — now with a genuinely new ingredient the 0096-0103 era
  lacked: an aligned target for the pull), or accepting that θ-feature
  discriminability is near its ceiling for this head/readout and moving the
  frontier elsewhere (index, cascade).
- The doc-units reparameterization (`profile_eta_docs`) remains worth building
  as an INTERPRETABILITY dose control (N patients of legibility), not an AUC
  knob.
- `--profile-align` + `readout-ab` are the standing verification pair for any
  future prior/init change: alignment and discriminability move independently
  and must both be read.

**Setting context:** exps 0116/0117 = 0113 (CV branch MONDO:0004995,
native-Mondo, tpn=5, K=1498, random init, fit-only 50 iters, multidomain,
holdout 0.2, seed 42) + rolled-up HPOA profile eta (132/299 credited, freq ×
IDF, one boosted topic per block) at strength 1.0 / 3.0. Readout pc_topics_lr,
193 shared scored nodes.
