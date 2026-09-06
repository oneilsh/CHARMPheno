# 0083 — The readout A/B inverts the spectral story: it fixes topic LEGIBILITY but COSTS case-finding at every depth; topic evidence is not discriminability

**Date:** 2026-09-06
**Topic:** lda, init, spectral, gated-pc, case-finding, readout
**Status:** Confirmed on the paired `gated-pc-readout`s of exp 0113 (random init) vs exp
0114 (spectral) — same bundle key, same split (`holdout_frac 0.2`), same readout code and
caps (L-BFGS max_iter 200, theta top-m 256), same 193 scored nodes. Only the fit's init
differs. All figures are model metrics / counts-of-nodes.

**Relates to:** 0082 (spectral cuts starvation 72%→1% — that finding STANDS, but this one
bounds what it buys), 0079 (the starvation it fixed), 0066 (supervision's payoff regime),
0080/0081 (unsupervised topics learn care archetypes, not labels — this is that lesson
biting the init).

## Observation

Random init BEATS spectral on downstream case-finding, uniformly:

| readout (pc_topics_lr) | 0113 random | 0114 spectral | delta |
|---|--:|--:|--:|
| macro ranking AUC / AP (193 nodes) | **0.7813 / 0.5255** | 0.7567 / 0.5031 | −0.025 / −0.022 |
| detection (case-vs-bg) AUC / AP | **0.6347 / 0.7156** | 0.5723 / 0.6725 | −0.062 / −0.043 |
| per-node median AUC | 0.791 | 0.761 | −0.030 |

The deficit **widens with depth** (median AUC by depth): d2 −0.023, d3 −0.015, d4 −0.019,
d5 **−0.061**, d6 −0.036, d7 **−0.081**. And the eyeballed misaligned nodes specifically
pay: intrinsic cardiomyopathy 0.890 → 0.797, dilated CM 0.909 → 0.772 (the two
pregnancy-anchored topics of 0114), cardiomyopathy-family matched-median 0.876 → 0.797.
The one gainer is Tako-tsubo (0.767 → 0.911) — a phenotype whose real signature IS
demographic/stress-shaped, i.e. the one node whose meaning coincides with the
most-separable stratum.

## Reading

1. **Sharp-but-misaligned beats nothing at legibility and loses to nothing at detection.**
   Under random init a starved deep topic is ~flat, so θ mass stays on ancestors and
   background; the LOCALIZED head (`head_support: path_cousins_kids`) reads those
   ancestor/sibling loadings, which carry honest, gate-correlated signal. Spectral
   replaces that honest diffuse signal with sharp node topics that soak θ toward whatever
   the anchor found — and when the anchor found a demographic stratum, the head inherits
   confidently wrong features. Wrong-sharp < weak-honest, for detection.
2. **The depth gradient fits forward-deflation compounding** (conjecture, untested):
   ancestors seed first and absorb their pool's dominant co-occurrence block; with
   imbalanced children (X.A at 90%, X.B at 10%), X's seed can gobble X.A's signature
   before X.A seeds, leaving X.A an off-target increment — and the error compounds down
   the path. Consistent with (not proven by) deeper-is-worse.
3. **The anchor criterion is variance, not discrimination.** "Maximally separable" here
   means: the vocabulary item whose residual projected co-occurrence row has the largest
   norm after deflating ancestor/already-chosen directions — the strongest REMAINING
   co-occurrence pattern in the node's pool. Nothing in it references the label. 0115
   showed binarization doesn't fix it; this readout shows it isn't harmless.

## Consequences

- **Spectral init is DEMOTED to a legibility/diagnostic tool** (it remains the honest way
  to SEE what deep pools contain — 0114's topics named the misalignment that this readout
  then priced). It is not the record init for case-finding runs, pending the eta-prior
  comparison below.
- **Never accept an init (or any fit change) on evidence/sharpness diagnostics alone.**
  0114 looked like a triumph on evidence-by-depth (1% starved); the readout priced it
  −0.025 macro AUC. Topic evidence and discriminability can anti-correlate. The
  `--readout-auc` slice is now part of the acceptance loop.
- **The live candidate is random init + HPO-profile word-side eta prior** (survey
  2026-09-06: feasible — 495 CV nodes with ≥5 SNOMED-realizable profile terms): knowledge
  alignment as PERSISTENT soft pressure, with no basin lock and no variance-seeking
  anchor. The A/B grid is {random, spectral} × {flat, profile-eta}; the record arm to run
  first is random × profile-eta vs 0113 (one knob again). Stage-2 survey (corpus-vocab
  intersect + strip-survival split) gates it.
