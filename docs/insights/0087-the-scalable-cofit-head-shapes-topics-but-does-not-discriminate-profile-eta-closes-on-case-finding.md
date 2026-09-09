# 0087 — The scalable strong co-fit head shapes topics but does not discriminate: profile-eta + PC closes on case-finding (credited AUC 0.734 < uncredited 0.767), and top-1% lift, though real (7.6×), lives only in the data-rich nodes

**Date:** 2026-09-09
**Topic:** gated-pc, prediction-constraints, co-fit-head, eta-prior, hpoa, case-finding, readout, top-1%, VOI
**Status:** Confirmed on exp 0120 (MONDO:0004995 branch, tpn=5, C=299, K=1498; profile-eta S=1.0 + the matrix-free L-BFGS co-fit head, weight_y=12, head_trust_move=0.03). Same bundle/split/readout as 0113/0116; credited set from the stage-2 `--emit-eta` probe. All figures are model metrics / counts-of-nodes.

## Observation

The PC-revival arc built a co-fit head that both **scales** (matrix-free
amortized batched L-BFGS, converges at K=1498 — insight 0086) and **shapes**
right-sized (trust cap pins `corr_relΔλ` at 0.03). Run on the strongest base
(random init + profile-eta S=1.0, the aligned target), it is the fair PC test the
2026-08-20 closeout demanded. The result is a clean negative on case-finding:

**1. AUC is DOWN, not flat.**

| arm | macro AUC | co-fit-head AUC |
|---|--:|--:|
| 0116 profile-eta S=1, no shaping | ~0.7804 | — |
| **0120 profile-eta S=1 + co-fit head** | **0.7555** | **0.6397** |

The shaping *lowered* linear decodability of θ by ~0.025 vs the no-shaping
baseline. The co-fit head's OWN decoder (0.64) is far weaker than a fresh LR on
the same shaped θ (0.7555): the head that shaped the topics reads them worse than
a plain logistic regression does.

**2. It hurt most where it was aimed.** Credited (the 36 profiled rare nodes; 14
scored) median AUC **0.734** < uncredited (179) **0.767**. The word-side prior
that made those nodes legible left them *below* the un-profiled majority on
discrimination. AUC also falls monotonically with depth (d2 0.816 → d7 0.729).

**3. `self-w ≈ 0` — legibility and decode-weight are different topics.** In the
readout decoder view, nearly every node's head puts ~0 weight on its own
(profile-aligned, legible) topic and routes the signal through ancestor / shared
"sick-patient" comorbidity topics. Alignment (own-topic top-15 overlap median
1.00) and discriminability (decode weight) are decoupled at the weight level —
the mechanical form of 0083/0084/0085.

**4. Top-1% is the one positive, and it is honest about its limits.** Macro
top-1% **lift 7.57, precision 0.605**: the top 1% by score is 60% true cases, 7.6×
the majority/random baseline — the representation is NOT useless for screening.
But the high-lift nodes all carry hundreds of positives (253: lift 51, n_pos 211;
15: 44, n_pos 282; 4: 30, n_pos 590); **the enrichment is concentrated in
data-rich nodes and absent from the rare credited tail.** Top-1% surfaces real
signal the macro AUC hides, and localizes it to where data already is.

## Why it matters

- **The PC / profile-eta line closes for case-finding.** profile-eta is an
  interpretability lever, not an AUC lever (0084/0085); the scalable strong co-fit
  head — the last mechanism that could have converted alignment into
  discriminability — does not, and mildly hurts. A right-sized, aligned-target
  label pull on static θ is not the missing ingredient. Do not spend more on
  word-side priors or head-strength tuning for case-finding AUC.
- **Engineering ≠ science, and both are now settled for this arc.** The head
  scales and couples cleanly (0086, and 0120's 412 clean passes) — that machinery
  is reusable. What it buys on this representation is the question that came back
  negative.
- **Add top-k lift to every readout verdict, not just AUC.** It is the
  deployment/VOI-honest operating point and it discriminated here: it said
  "not useless, but enriching only where data is," which macro AUC alone could
  not. (Landed this run: `evaluate._score_label`/`_macro`, per-node + macro,
  default top 1%.)
- **The VOI concern is sharpened, not relieved.** Feature-level VOI (spec
  `2026-09-01-incident-episode-eval-program.md` §3) needs per-feature
  discriminative structure. Here it lives in coarse shared ancestor topics and in
  the data-rich nodes — not the rare tail the prior was meant to rescue. Building
  VOI on static-θ gated topics would answer "what would knowing X buy" through
  generic-comorbidity signal, weakest exactly where a rare phenotype needs it.

## What this rules out / where next

- **Ruled out (for case-finding AUC):** eta strength (0084/0085), the co-fit head
  at any right-sized shaping (0120). Static-θ, word-side-prior shaping is done as
  an AUC lever.
- **Kept:** profile-eta as a *labeling / legibility* layer read off outputs
  (never as a fit-time prior on a predictive/VOI state); the L-BFGS co-fit head as
  reusable scalable machinery; top-k lift as a standing readout metric.
- **Next (the frontier the record now points at):** representation, not priors —
  the episode / temporal index (incident-episode eval program). The open question
  is whether *sequential* structure carries the conditional, per-feature signal
  that static topic mixtures do not — the necessary precondition VOI has been
  waiting on.

## Refinement (2026-09-09, `inspect_topics --collinearity` + `--strip-audit` on 0120) — two claims above are downgraded, and the mechanism is sharper

- **Point 3's `self-w ≈ 0` is UNVERIFIED as stated.** The decoder view it came from
  used the raw-θ heads `V = W_std/sd`, which explodes on low-variance (starved)
  topics; with ~1000 starved topics at K=1498 the |w| mass is dominated by
  ~constant features, so "own share 0.00" comes out for EVERY group (credited 28/28,
  uncredited fed 100/100, starved 131/131) and says nothing about what the head
  uses. The solver checkpoint (standardized W_std) was gone; the heads sidecar now
  persists W_std so the next readout gives an honest read. Until then, do not cite
  the routing claim.
- **The credited topics are NOT near-duplicates of each other or of shared topics.**
  Profile Jaccard among credited nodes: median-of-max 0.25, median-of-median 0.02, no
  token in ≥ half the profiles. Boosted-topic cosine: peers 0.09, background 0.05,
  ancestors 0.15, flat 0.06 (vs starved uncredited: 0.99/0.28/0.96/0.99). The
  profile prior makes each credited topic SHARP and DISTINCT.
- **They are sharp because of the prior, not the data — and that is the mechanism.**
  Boosted-topic evidence (λ mass): credited median **65.2 vs prior floor 60.7**
  (≈4.5 pseudo-counts of data above the prior; 0/36 flagged "at floor" only because
  the prior itself is above the flat floor), against **1318.7** for uncredited fed
  topics. Of the profiles' 3,135 mapped concepts only 834 are in the condition
  vocab at all, and (stage-2 probe) 53% of a node's positives carry none of those.
  So the aligned topic is a legible label with essentially no posterior evidence:
  θ on it cannot vary across documents, and no decoder — the co-fit head or a fresh
  LR — could ever load on it. "Legibility ≠ discriminability" resolves to
  "prior-shaped ≠ data-shaped."
- **The AUC numbers above are prevalent readouts under a no-op leakage strip
  (insight 0088)**: the disease's own pre-index codes were in the features for
  every node whose chart carried them before the random index. The credited <
  uncredited gap (0.734 < 0.767) and the depth slide are therefore a TRACKING
  gradient as much as a discriminability one. The honest verdict on PC needs the
  incident cohort, which this arc never ran.

## Reproduce

```bash
# fit + readout (bundle rebuilds per cluster; ~2h fit, ~1.7h readout)
CHARM_DRIVER_MEMORY=16g make -C analysis/cloud exp ID=120
make -C analysis/cloud gated-pc-readout ID=120 GPR_ARGS="--readout-mode distributed"
# reads (off-YARN; regen the eta TSV first on a fresh cluster)
make -C analysis/cloud inspect-topics ID=120 CREDITED=1 INSPECT_ARGS="--readout-auc"     # credited split
python3 -c "import json;r=json.load(open('<run>/results_readout.json'))['gated_pc']['ranking'];print(r['auc'], r['lift_at_k'], r['prec_at_k'])"
```
