# Insight 0061 — The LEARNED per-node alpha (optimizeDocConcentration) confirms insight 0060's prediction on real rare6 data: NULL on LR detection (0.777 vs symmetric 0.778), and it additionally HURTS theta-mass placement ranking (mrr 0.585->0.442, top2 0.607->0.405, disease_mass auc 0.68->0.63). The learned alpha is non-degenerate and interpretable — background learns UP 3.4x (uniform usage), disease nodes spread ~8x and mostly land BELOW background (peaked/sparse) — but the data-driven asymmetry does not help any task we care about. NPMI unchanged. Decision: do NOT use optimize_alpha in production case-finding.

**Date:** 2026-07-22
**Branch:** case-finding
**Topic:** case-finding | optimize-alpha | learned-alpha | detection | placement | LR-readout | null-result | decision

**Status:** Observed

**Relates to / confirms:** insight 0060 (fixed block-asymmetric alpha is a null lever for detection;
predicted the learned alpha would also be null because the LR readout bypasses theta). This is the
direct test of that prediction on real data, and it CONFIRMS it while adding the theta-mass
placement regression and the learned-alpha structure. Also relates to 0059 (the learned alpha is
correct but single-seed multimodal — hence read gross structure here, not per-node points) and the
LR arc ([[project_case_finding_lr_readout]]).

## Setting

exp 0065: rare6, 1yr lookback, spectral init, frontier-scoped anchors, K=170 (40 bg + 26 nodes x 5
tpn), max_iter 200, seed 42, node_alpha_scale 1.0 (symmetric INIT) + optimize_doc_concentration=true
(the learned per-node gated Newton alpha, wired through the cloud driver + per-node alpha log).
Reuses 0061's cached 1yr corpus (fit param, not in cache key), so 0061 (symmetric) / 0063 (fixed
asym 0.1) / 0065 (learned) are a same-corpus three-way contrast. Detection scored both by theta-mass
(manifest) and the LR alpha->inf lift limit (make lr-readout).

## Findings

1. **NULL on LR detection — insight 0060's prediction confirmed.** LR alpha->inf: ROC 0.777 /
   PR-AUC 0.223 (0065 learned) vs 0.778 / 0.222 (0061 symmetric) vs 0.775 / 0.219 (0063 fixed asym).
   Within <=0.002. The LR readout reads lambda directly and bypasses theta, so learning the theta
   prior alpha does not move detection — as predicted.

2. **Learned alpha HURTS theta-mass placement ranking (new).** Same 1yr corpus, so attributable to
   the learned alpha: placement mrr 0.442 (vs 0.585 at 0063), top2 0.405 (vs 0.607), disease_mass
   auc 0.626 (vs 0.683). The data-driven reshaping of theta de-optimizes the node-block-mass
   readout. So the learned alpha is not merely null — for theta-mass it is the WORST of the three
   arms (symmetric / fixed-asym / learned), while LR detection is identical across all three.
   (theta-mass DETECTION auc ticked 0.646->0.657, within noise; the clear signal is the placement
   /ranking degradation, not a detection change.)

3. **The learned alpha is non-degenerate and interpretable.** background=0.0200 vs init 1/K=0.0059
   (learned UP 3.4x — the 40 bg topics are used uniformly across every doc, so large alpha). Node
   blocks span ~8x: min 0.0058 (Sarcoidosis of lung with lymph nodes, node 23) to max 0.0467
   (Scleroderma, node 6); 19 of 26 nodes below background (disease-node topics are peaked/sparse ->
   small alpha). Gross ordering (single-seed; multimodal per 0059, so do not over-read per-node):
   the highest-alpha "spread" nodes are broad multi-system diseases (Scleroderma, Lichen amyloidosis,
   EDS, SLE); the lowest-alpha "peaked" nodes are narrower (lung+lymph sarcoid, Myasthenia gravis,
   Neurosarcoidosis). So the optimizer works on real data (the covered-subset fix f959daf matters:
   real rare6 has nodes with little/no coverage that stay near init) — the asymmetry it discovers is
   just not useful for the tasks.

4. **NPMI unchanged.** 0065 mean 0.183 (vs ~0.19 symmetric/asym); same coherent node topics (Cardiac
   sarcoidosis, Senile cardiac amyloidosis, Scleroderma, SLE, Lichen amyloidosis). The learned alpha
   does not reshape topic-word coherence.

## Decision / implication

- **Do NOT use optimize_alpha in production case-finding.** It is null on LR detection (the metric
  that matters), actively hurts theta-mass placement, and leaves coherence unchanged. Keep it off
  (the default); node_alpha_scale symmetric is fine.
- The feature stays in the codebase as a correct, validated (insight 0059), now empirically
  characterized capability — a publishable-methods negative, not a production lever.
- **Closes the alpha-prior thread.** Neither fixed (0060) nor learned (0061) doc-topic asymmetry
  helps case-finding detection. The binding constraint is INFORMATION (richer meds/labs features) ->
  the MixEHR multi-domain direction, not prior tuning.
