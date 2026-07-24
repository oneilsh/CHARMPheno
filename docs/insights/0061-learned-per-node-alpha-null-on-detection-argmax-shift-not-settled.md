# Insight 0061 — The LEARNED per-node alpha (optimizeDocConcentration) confirms insight 0060's prediction on real rare6 data: NULL on LR detection (0.777 vs symmetric 0.778). It also SHIFTS the cross-node argmax ranking (mrr 0.585->0.442, top2 0.607->0.405) — but per-node discrimination (auc_by_depth) and detection are UNCHANGED, the shift is the mechanical effect of a non-uniform node prior on fold-in argmax (confounded by single-seed multimodality), and it is measured on the wrong objective to judge a fit alpha (held-out likelihood, what optimize_alpha actually maximizes, was NOT measured). So placement is NOT settled; only "no demonstrated benefit + null on detection" is. The learned alpha is non-degenerate/interpretable (background up 3.4x = uniform usage; disease nodes spread ~8x, mostly below background = peaked/sparse). NPMI unchanged. Decision: keep optimize_alpha OFF in production (no demonstrated benefit); to actually adjudicate, measure held-out LL + ensemble over seeds.

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

2. **Learned alpha shifts the CROSS-NODE argmax ranking (mrr/top2), but NOT per-node discrimination
   or detection — and this is NOT established as "a fit alpha is worse" (corrected after review).**
   What moved: mrr 0.585->0.442, top2 0.607->0.405. What did NOT move: per-node discrimination
   auc_by_depth {0.62,0.48,0.68}->{0.61,0.47,0.68} (flat), LR detection (flat), theta-mass detection
   auc 0.646->0.657 (flat/within noise). The AUC-flat-but-mrr-down signature is the tell: the effect
   is on the CROSS-NODE argmax, not within-node discrimination. Mechanism: symmetric (0061) and
   fixed-asym (0063) give every node the SAME alpha, so there is no prior preference BETWEEN nodes
   and the per-patient argmax is evidence-driven; the learned alpha gives nodes DIFFERENT alphas
   (0.006-0.047), an informative prior over node membership that at deployment fold-in pulls theta
   toward high-alpha nodes and down-ranks low-alpha ones in the argmax. This is a property of ANY
   non-uniform node prior, and whether it helps or hurts depends on whether the prior is RIGHT --
   which at a single seed is confounded by the multimodality (insight 0059: this seed's alpha
   ordering could be a bad basin). CRUCIALLY, mrr/top2 are NOT what optimize_alpha maximizes (it
   maximizes the training ELBO / held-out likelihood), so a fit alpha scoring lower on a ranking
   metric it never targeted is not evidence it "fits worse." The fair test of whether the fit alpha
   is better -- held-out log-likelihood -- was NOT measured. So: cross-node argmax shifted (single
   seed, mechanistically expected, wrong objective for the "is it better" question); nothing about
   placement is settled.

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

- **Keep optimize_alpha OFF in production case-finding (the default).** It is null on LR detection
  (the metric that matters), shows no demonstrated benefit on any measured task, and leaves
  coherence unchanged. That is enough to not turn it on. It is NOT established that it "hurts" —
  see finding 2.
- **What would settle placement / "is the fit alpha better":** (a) held-out log-likelihood on the
  test split (the objective optimize_alpha actually maximizes — the fair "better?" test), and
  (b) a seed-ENSEMBLE fit (average out the insight-0059 multimodality) to see whether the mrr/top2
  shift persists or was a bad basin. Both are cluster runs; neither was done here.
- The feature stays in the codebase as a correct, validated (insight 0059), now partially
  characterized capability — a publishable-methods result, not a production lever.
- **Closes the alpha-prior thread for DETECTION.** Neither fixed (0060) nor learned (0061) doc-topic
  asymmetry moves case-finding detection (LR bypasses theta). The binding constraint is INFORMATION
  (richer meds/labs features) -> the MixEHR multi-domain direction, not prior tuning. (The placement
  /calibration question the learned alpha raises is separate and open, pending held-out LL.)
