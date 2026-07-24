# Insight 0063 — Reverse-topological (leaves-first) spectral init is a null lever: it is indistinguishable from forward (ancestors-first) init on detection, placement, and the error classes, because the DAG gate already supplies identifiability and 200 SVI iterations move off the init geometry. This is the SIXTH consecutive model-side null (after fixed alpha 0060, learned alpha 0061, explain-away routing + n_bg 0062) and completes the case-finding lever sweep: the single durable contribution is the LR READOUT LENS, not any model/scoring/init/capacity change. The binding constraint is INFORMATION (condition codes alone cannot separate genuinely-similar comorbidity), i.e. the MixEHR multi-domain direction.

**Date:** 2026-07-23
**Branch:** case-finding
**Topic:** case-finding | spectral-init | topo-order | reverse-topo | placement | detection | null-result | information-constraint | collation | decision

**Status:** Observed

**Relates to / completes:** insights 0060/0061 (alpha nulls) and 0062 (explain-away + n_bg nulls).
This is the last of the six model-side A/Bs; see the full collation
docs/reports/2026-07-23-case-finding-levers-retrospective.md. The reverse-topo capability was built
this session (spec/plan 2026-07-23; SDD 5 tasks, final review READY-TO-MERGE) as the user's
leaves-first idea; this insight reports the A/B.

## Setting

exp 0069: clone of exp 0067 (rare6, 1yr lookback, learned per-node alpha FIT + symmetric deploy,
n_bg 40, frontier anchors, scalable spectral init, K=170, seed 42) with the single change
`spectral_topo_order: reverse` — the new knob that recovers DAG nodes leaves-first, deflating each
node's anchor-word spectral recovery against its already-recovered proper-DESCENDANTS instead of its
proper-ancestors. Hypothesis: since placement scores on the most-specific (leaf) nodes, letting them
claim their full defining signal at init (rather than only the increment beyond their ancestors) might
sharpen the discriminative leaf topics.

## Findings

1. **NULL on every axis.** 0069 (reverse) vs 0067 (forward): LR detection ROC 0.778 -> 0.779 (+0.001),
   explain-away 0.767 -> 0.773, theta-mass 0.647 -> 0.655, placement mrr 0.596 -> 0.590, top2 0.625,
   auc_by_depth deepest-level ~0.70. Every move is <= 0.01 = single-seed re-fit noise. NPMI mean 0.188
   ~ 0067; node topics coherent either way (up to 0.49 for cardiac amyloidosis/sarcoidosis).

2. **The error classes confirm the null and the diagnosis.** rare_called_background (FN) = 276,
   IDENTICAL across 0067/0068/0069 — the strongest single signal in the arc that the FN patients are
   information-limited (thin/non-distinctive profiles), untouched by any model change. FP class
   13158 -> 12608, but bg_fpr at matched 80% sensitivity 0.406 -> 0.413 (flat-to-worse): the FP
   "drop" is a threshold artifact (the error-class operating threshold moved +0.713 -> +0.734), not a
   real gain.

3. **Mechanism: the init geometry does not survive the fit.** The DAG gate already welds each node's
   topics to its subtree's documents (identifiability comes from the gate, not the seed), so the
   spectral init only sets the STARTING lambda; 200 SVI iterations then move off it. This is exactly
   the prototype finding that spectral init (either direction) does not beat random init on this
   engine. Reverse vs forward changes only where the iterations START, and they converge to
   effectively the same place.

4. **Learned-alpha ordering shifted again** (single-seed multimodality, insight 0059): a different
   init lands a different alpha basin; Spearman(alpha, coverage) = 0.18, still ~0 (tracks footprint
   diffuseness, not prevalence). Consistent with 0061.

## Decision / implication

- **Keep `spectral_topo_order` default = forward in production.** Reverse is null; it is a correct,
  validated, characterized capability (the knob + DagLayout.descendants + the identifiability-preserving
  descendants-first ordering), a publishable negative, not a lever.
- **The case-finding model-side lever sweep is COMPLETE and CONVERGED (six nulls).** fixed alpha
  (0060), learned alpha (0061), explain-away routing (0062), background capacity (0062), reverse-topo
  init (0063), count-mode log1p (0062) — every prior / scoring-scheme / init-geometry / capacity lever
  is null. The single durable contribution is the LR READOUT LENS (+0.11-0.13 ROC over theta-mass),
  a readout change, not a model change.
- **Binding constraint = INFORMATION, definitively.** Both error classes reduce to it (FP = genuine
  code-level similarity, FN = data starvation), and no model-side lever moves either. The un-chased
  levers are (a) richer features (meds/labs = the MixEHR multi-domain direction) and (b) the LR-FDR
  readout (does LR's detection edge yield FDR-controlled discoveries where theta-mass got zero?). We
  are not chasing our tail — we have systematically ruled out the model side.
