# Insight 0062 — Explain-away (responsibility-weighted) routing AND doubled background capacity are BOTH null on case-finding, completing a three-lever sweep (learned alpha 0060/0061, explain-away routing, n_bg) that all fails. The decisive evidence is the ERROR-CLASS totals, not the third-decimal AUC: the false-negative class (rare called background) is IDENTICAL under routing (276 -> 276) and under 2x background (276 -> 276), and the dominant false-positive class (background called rare, ~13000 = 44% of background at 80% sensitivity) barely moves under either lever (13158 -> 12992 at n_bg 80). Both error classes reduce to the SAME diagnosis — INFORMATION, not scoring scheme or model capacity: FN = data starvation (thin profiles with nothing distinctive); FP = GENUINE code-level similarity (a background patient with anemia + CKD is indistinguishable from SLE from condition codes alone, because those codes really are SLE-associated). Plain LR at the alpha->inf lift limit remains the durable readout (crushes theta-mass by +0.11-0.13 ROC); scoring refinements have hit diminishing returns. Binding constraint = multi-domain features (meds/labs), i.e. the MixEHR direction.

**Date:** 2026-07-23
**Branch:** case-finding
**Topic:** case-finding | explain-away | routing | n-bg | background-capacity | false-positive | false-negative | LR-readout | null-result | information-constraint | decision

**Status:** Observed

**Relates to / completes:** insights 0060 (fixed asymmetric alpha null) and 0061 (learned
per-node alpha null; alpha-prior thread closed for detection). This is the third and final
"model-cleverness" lever in the same sweep and it closes the loop: after tuning the theta
prior (0060/0061), we tried re-weighting the evidence (explain-away routing) and adding model
capacity (n_bg), and both are null. Builds on the LR-readout arc
([[project_case_finding_lr_readout]]) and the alpha/explain-away build session
([[project_alpha_and_explainaway_arc]]).

## Setting

Post-hoc LR readouts (no re-fit for the scorer) on two fits:
- **exp 0067**: rare6, 1yr lookback, spectral init, frontier anchors, learned per-node alpha
  FIT + symmetric deploy, K=170 (40 bg + 26 nodes x 5 tpn), seed 42.
- **exp 0068**: identical to 0067 except n_bg 40 -> 80, K=210 (full re-extract; n_bg is in the
  bundle cache key). Different spectral init (80 anchors), so a genuine re-fit.

Three scorers compared at the alpha->inf lift limit, all max-over-nodes as the case score over
the SAME foreground/background split (like-for-like): theta-mass (manifest), plain LR, and the
new explain-away routing scorer s(u) = Σ_w cnt(w) · r(u|w) · log[P(w|u)/bg(w)], where r(u|w) is
code w's soft responsibility on node u's topic block (built + reviewed this session; Pearl 1988
explain-away / mixture E-step). Errors bucketed by the error-class viewer at the 80%-sensitivity
operating point into background_called_rare (FP), rare_called_background (FN),
rare_called_rare_wrong_disease, rare_called_rare_correct, background_called_background (TN).

## Findings

1. **Explain-away routing is null-to-slightly-negative on aggregate detection.** exp 0067,
   alpha-matched (both scorers at alpha=inf): explain-away ROC 0.7675 / PR-AUC 0.2122 vs plain LR
   0.7781 / 0.2219 — routing costs ~0.011 ROC and is below plain LR at every operating point.
   Mechanism: routing multiplies EVERY code's contribution by r(u|w) in [0,1], which suppresses
   comorbid drag AND attenuates genuine positives (a distinctive code rarely routes with r
   exactly 1); on the global rank order the second effect slightly wins. Both LR variants still
   beat theta-mass (0.6466) by +0.12 ROC, so the LR readout itself is the win — routing adds
   nothing.

2. **Explain-away does NOT rescue the false-negative class it was designed for.** The whole
   premise was the comorbidity-drag FN patient (strong positives sunk by a long tail of small
   negatives). Error-class totals, exp 0067: rare_called_background = 276 under BOTH plain LR and
   explain-away — the routing left the FN class untouched. Pixel-peeping those FNs shows thin
   profiles with few or non-distinctive codes: they are INFORMATION-limited, not drag-limited.
   Routing can suppress a negative that isn't there to begin with. Hypothesis refuted for the FN
   class.

3. **The dominant error is the false-POSITIVE class, not the FN class.** exp 0067 totals:
   background_called_rare = 13158 (= 44% of the 29595 background docs at 80% sensitivity), vs only
   276 FN. The binding error is background patients called rare, and pixel-peeping shows
   comorbidity clusters landing on the nearest rare node: lung/abdomen -> Sarcoidosis, anemia +
   CKD -> SLE, thyroid + injury -> Lichen amyloidosis.

4. **Doubling background capacity (n_bg 40 -> 80, exp 0068) is NULL on the FP class.**
   background_called_rare 13158 -> 12992 (-1.3%, noise); FN 276 -> 276 (identical); bg_fpr at 80%
   sensitivity 0.406 -> 0.428 (if anything worse); plain-LR ROC 0.778 -> 0.770; theta-mass
   0.647 -> 0.657. Every detection move is <= ~0.01, i.e. within single-seed re-fit noise
   (different spectral init). The capacity hypothesis — more background topics give the comorbid
   clusters a home so they stop landing on a node — does not hold.

5. **Why capacity was the wrong lever (the key mechanism).** The FP codes are GENUINELY
   disease-associated, not comorbid noise a background topic could claim away. Anemia (of chronic
   disease) and CKD (lupus nephritis) really are elevated in the SLE topic; they are true
   SLE-predictive features. So from condition codes alone, a background patient with anemia + CKD
   is indistinguishable from an SLE patient — no number of background topics separates them,
   because the separating information (autoantibodies, labs) is not in the ICD stream at all.
   Adding background capacity cannot manufacture a distinction the codes do not carry.

6. **Explain-away's one pre-registered structural advantage did show, but is immaterial.** The
   scorer's FP fix is explicitly n_bg-bounded (routing only removes a comorbid code's spurious
   node support if a BACKGROUND topic out-competes the node for it). As predicted, the
   explain-away-vs-LR gap narrowed from 0.0106 (n_bg 40) to 0.0044 (n_bg 80) — explain-away held
   up better than plain LR under added capacity. But since plain LR itself did not improve, this
   is a relative, immaterial gain.

## Decision / implication

- **Explain-away stays as a validated, characterized VIEWER MODE (--viewer-score-mode
  explain_away), NOT the default scorer.** It is null on detection and does not fix its target
  class. The build is correct (reviewed this session; math and routing-conservation verified) —
  a publishable negative, not a production lever. Same disposition as optimize_alpha (0059-0061).
- **n_bg = 80 is not adopted** (no benefit, ~2x the topics / compute). n_bg = 40 stays the
  default.
- **The unification (the real result): three model-cleverness levers, three nulls.** Tuning the
  theta prior (learned alpha, 0060/0061), re-weighting the evidence (explain-away routing), and
  adding model capacity (n_bg) all fail to move case-finding. Both error classes reduce to the
  SAME diagnosis — INFORMATION, not model: FN = data starvation (thin profiles), FP = genuine
  code-level similarity (comorbidities that are real disease features). The binding constraint is
  multi-domain features (meds / labs) — the MixEHR direction — not scoring scheme, prior, or
  capacity.
- **Plain LR at alpha->inf remains the durable readout** (crushes theta-mass by +0.11-0.13 ROC
  across 0067/0068). Scoring-scheme refinements on the condition-code stream alone have hit
  diminishing returns; further gains require richer features.
- **Caveat on the evidence.** The detection AUC deltas are all ~0.01 and single-seed; the
  conclusion does NOT rest on the third decimal. It rests on (a) the ERROR-CLASS totals — direct
  counts, stable across two independent fits (FN 276 = 276 = 276; FP ~13000 both ways) — and
  (b) the mechanism (genuine code-level similarity), which the pixel-peep supports directly.
