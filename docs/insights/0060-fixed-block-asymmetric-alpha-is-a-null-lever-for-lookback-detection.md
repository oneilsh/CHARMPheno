# Insight 0060 — A fixed block-asymmetric doc-topic prior (node_alpha_scale 0.1) is a NULL lever for case-finding detection on the honest lookback task: symmetric vs asymmetric are indistinguishable (≤0.003) across both lenses (theta-mass, LR) and both history depths (1yr, 5yr). The earlier forward-mode "asymmetry helps" was the init/epochs confound, not real. Mechanistically the LR readout reads lambda directly and bypasses theta, so the theta-prior alpha barely propagates to detection — which also tempers expectations for the LEARNED optimize_alpha on detection.

**Date:** 2026-07-22
**Branch:** case-finding
**Topic:** case-finding | node-alpha-scale | asymmetric-prior | detection | lookback | LR-readout | null-result | decision

**Status:** Observed

**Relates to:** the LR-readout arc ([[project_case_finding_lr_readout]]) — the finding that the
alpha->inf LR limit is the payoff detection lens (theta-mass buries the signal) and that the
binding constraint is information, not the model. Refines the alpha-asymmetry question that
insight 0059 (learned gated optimize_alpha) and the forward-mode rare6 runs (exp 0059's note:
old asym/test_only 0.709 vs symmetric cells 0.585-0.660, init/epochs confounded) left open.

## Setting

Four-cell A/B on the rare6 lookback corpus, gated placement engine, spectral init,
frontier-scoped anchors, K=170 (40 bg + 26 nodes x 5 tpn), max_iter 200, seed 42:
- 0061 (1yr) / 0062 (5yr): symmetric alpha (node_alpha_scale 1.0)
- 0063 (1yr) / 0064 (5yr): block-asymmetric alpha (node_alpha_scale 0.1 — per-node-topic Dirichlet
  alpha 10x smaller than background; a Wallach et al. 2009 asymmetric prior making each disease
  node a priori rarer)

0063/0064 reuse 0061/0062's cached corpora (node_alpha_scale is a fit param, not in the bundle
cache key), so this is a clean same-corpus fit-only contrast. Detection scored both by theta-mass
(node-block mass, from the run manifest) and by the LR alpha->inf lift limit (make lr-readout).
The alpha diagnostic min=0.000588 max=0.00588 confirms the asymmetry was actually applied.

## Findings

1. **Symmetric and asymmetric are indistinguishable on detection — a clean null.** LR alpha->inf:
   - 1yr: ROC 0.778 (sym 0061) vs 0.775 (asym 0063); PR-AUC 0.222 vs 0.219
   - 5yr: ROC 0.778 (sym 0062) vs 0.775 (asym 0064); PR-AUC 0.171 vs 0.172
   Every LR and theta-mass metric (ROC, PR-AUC, precision@{80,90,95}% sens) differs by <= 0.003
   between the arms — within run-to-run noise. The block-asymmetric prior does not help (nor hurt)
   detection at either history depth.

2. **The forward-mode "asymmetry helps" does not replicate on the lookback task.** The earlier
   rare6 forward runs suggested asym (0.709) beat symmetric (0.585-0.660), but that comparison was
   confounded by init (spectral vs random) and effective epochs (exp 0059's own caveat). On the
   honest lookback corpus, holding init/epochs fixed, the asymmetry effect vanishes -> the prior
   forward-mode signal was the confound, not the alpha.

3. **Mechanistic why: the LR detection readout bypasses theta.** LR alpha->inf reads the learned
   lambda (topic-word counts) as a per-node Naive-Bayes lift, never touching the doc-topic simplex
   theta. node_alpha_scale is a prior on theta; it shapes the E-step responsibilities, which
   propagate to lambda only weakly (the gating + spectral init + data dominate lambda). So moving
   alpha barely moves lambda -> barely moves LR detection. This is why even a 10x asymmetry is a
   null.

4. **Secondary (consistent with prior): more history sharpens topics but not detection.** 5yr NPMI
   mean 0.218 > 1yr 0.191, yet detection is flat 1yr-vs-5yr. Node topics stay coherent (Senile
   cardiac amyloidosis, Cardiac sarcoidosis, Scleroderma, SLE, Lichen amyloidosis at NPMI
   0.35-0.49).

## Decision / implication

- **Do not pursue fixed block-asymmetric alpha as a detection lever.** node_alpha_scale can stay
  at its symmetric default; tuning it does not buy detection.
- **Temper expectations for the LEARNED optimize_alpha (insight 0059) on DETECTION.** Since the LR
  detection lens bypasses theta and a 10x fixed asymmetry is null, a learned per-node alpha is
  unlikely to move detection either. It may still improve theta-mass placement *calibration* (an
  untested, different metric), but detection — the metric that matters — is LR-based. The learned
  feature is built and validated (ensemble recovery), but its expected payoff is not detection.
- **Confirms the LR arc's conclusion:** the binding constraint on case-finding detection is
  INFORMATION (richer meds/labs features), not the topic-model priors. This points at the
  multi-domain / MixEHR direction, not further prior tuning.
