# 0068 — Plants that split signature mass evenly over closure(v) do not identify their own label; the SVI-vs-Gibbs "divergence" was that artifact, but SVI's affinity over-concentration is real

**Date:** 2026-07-25
**Topic:** svi | lda | diagnostics
**Status:** Confirmed
**Context:** SP2 of the multi-domain gated DAG LDA arc (branch `multidomain-spectral-init`), building the multi-domain SVI≈Gibbs placement equivalence gate (plan Task 5). Three attempts; the first two produced a convincing false positive.

## Finding — part 1: the plant did not identify its own label

`two_domain_dag_corpus` (and `dag_placement_corpus` / `_multi`) distribute a document's signature draws **evenly across every node on `closure(v)`**, while labelling the document with its **deepest** attested node. So a node-5 document emits node-2 and node-5 signature tokens in equal measure: the evidence is symmetric between ancestor and descendant, but the label is not. **Ancestor-versus-descendant rank is unidentified by construction**, and no placement gate resting on that ranking can mean anything — for either inference engine.

This produced a reproducible, mechanistically-explained, and entirely spurious engine finding. On a widened DAG (`{1:0,2:0,3:1,4:1,5:2,6:2}`, 8 split seeds, 200 training documents), split-averaged mrr was **Gibbs 0.9971 vs SVI 0.8756** (gap 0.1215), with per-true-node SVI mrr `{4: 0.935, 5: 0.686, 6: 0.679}` against Gibbs `{4: 0.999, 5: 0.995, 6: 0.988}` — deep nodes pinned at **exactly 0.500**, i.e. the true node ranked second. It was invariant at SVI `n_iter` 50/200/400 (not under-convergence) and persisted at 800 training documents (not data starvation).

Weighting a node's own signature above its ancestors' (a new `ancestor_signature_decay` generator parameter, default 1.0 so every existing corpus is byte-identical; the gate uses 0.5) made the swap **vanish completely**: per-node mrr **1.000 vs 1.000 on all six nodes across all eight splits**. The single-domain pooled case moved 0.82 → 1.0 as well, confirming the per-domain factor was never implicated.

## Finding — part 2: what actually survives

The mechanism isolated during the false positive is real, and the plant fix did not remove it — it out-voted it:

- The multi-domain spectral seed leaves **~0.47 of every node topic's mass on the shared common pool**.
- Gated variational EM clears that only for **deep** nodes. Depth-1 node topics still carry ~0.40 after 400 iterations; the collapsed-Gibbs oracle gets below 0.06. Measured per node: SVI `{0.356, 0.423, 0.435, 0.505, 0.234, 0.168}` vs Gibbs `{0.027, 0.016, 0.226, 0.010, 0.023, 0.052}`.
- The inflated ancestor therefore over-leaks affinity. On node-2 documents the SVI affinity **level** is 0.834 where the oracle says 0.506 and ground truth is ~0.50.
- The bias is still visible through the identifiability margin: at `ancestor_signature_decay` 0.9/0.75 (the same corpus after rounding) the swap partially returns — split 0 gives SVI mrr 0.9133 with node 4 pinned at exactly 0.500 against Gibbs 1.0000.

So **placement ranking is validated; affinity calibration is not.** This is the same shape as the mean-field calibration failures recorded in [0044](0044-meanfield-vi-fails-sigma-correlation-even-when-identified.md), [0051](0051-ridge-conditional-offset-intervals-order-uncertainty-but-are-overconfident-absolutely.md) and [0057](0057-dag-offset-readout-recovers-ordering-but-fails-calibrated-coverage-under-exact-gibbs.md), now reproduced in the multi-domain gated LDA path with a concrete mechanism rather than an inference-theoretic argument.

## Why it matters

1. **A placement-ranking metric cannot certify a topic model's affinity levels.** The equivalence gate went green on ranking while the level disagreement was 0.834 vs 0.506. Anything downstream that consumes affinity *magnitude* — thresholds, calibrated scores, FDR-style read-outs — is not covered by an mrr/AUC gate.
2. **This is a candidate explanation for a precision wall already on record.** Ancestor topics carrying common-pool mass inflate ancestor affinity for every document, which is exactly the failure shape behind low-prevalence precision in the LR-readout arc ([0064](0064-lr-ranking-edge-yields-zero-fdr-discoveries-ranker-not-discoverer.md)). Reducing seed leakage is a concrete, testable lever there, and it is an engine fix rather than a read-out fix.
3. **The two existing single-domain equivalence gates rest on the same unidentified tie-break.** `test_svi_matches_gibbs_placement_single_parent` and `_multi_parent` still run on the even-split generators, which likely explains why both engines sit at mrr ~0.91 there rather than near 1.0. They should be ported to `ancestor_signature_decay`.
4. **Method note on how the false positive was caught.** It survived a widened DAG, 8 split seeds, an iteration sweep and a data-size sweep — every check aimed at the *engine*. What broke it was asking whether the **plant identifies the quantity being measured**. Recovery-style and ranking-style metrics both looked authoritative while the target was underdetermined.

**Bottom line:** verify a planted corpus identifies the label it claims before believing any engine comparison built on it. The 0.12 mrr gap was an artifact; the ~0.40-vs-0.06 common-pool leakage and the 0.834-vs-0.506 affinity over-concentration are real and unfixed.
