# 0089 — The readout LR was the bottleneck: ridge 100 on the SAME θ lifts 0120's macro AUC 0.7555 → 0.7927, and standardized heads put ~0 weight on a node's own topic even among the 546 fed topics — the arc's AUC deltas were read through an under-regularized, under-converged instrument

**Date:** 2026-09-09
**Topic:** readout, evaluation, regularization, case-finding, gated-pc, exp 0120, insight ladder 0082–0087
**Status:** Confirmed on exp 0120 (three re-readouts on the cached transform, same θ, same split, `gated-pc-readout --readout-l2`; and `inspect_topics --collinearity` on the standardized heads). Pooled figures only.

## Observation

**1. The ridge sweep.** Three re-readouts of the identical fitted model; only the per-node
readout head's ridge changed (`l2` on the SUMMED log-loss; 1.0 = sklearn `C=1.0`, the
default every experiment used):

| readout l2 | converged heads | macro AUC | AP |
|---:|---:|---:|---:|
| 1 (record) | 68/259 at 200 iters, max‖grad‖ 23.5 | **0.7555** | 0.4919 |
| 100 | 225/259 | **0.7927** | 0.5242 |
| 10 000 | 259/259 at 87 iters | 0.7909 | 0.5044 |

+0.037 macro AUC from the readout alone. At l2=1 the solve is both near-unregularized
(hundreds of thousands of cells, 1,498 standardized features, a few hundred positives
per node) and NOT converged in its 200-iteration budget; at 100 it converges and
generalizes better. The co-fit head's own numbers (0.6397) are invariant across the
sweep, as they must be — it is frozen in the fit.

**2. The standardized heads are diffuse and ignore own topics.** With `W_std` persisted
(the raw-θ `V` is inflated on starved topics and was unreadable — 0087 refinement),
median share of |w| per group:

| group | own block | background | ancestors | descendants | other nodes | top-10 share |
|---|--:|--:|--:|--:|--:|--:|
| credited (28) | 0.00 | 0.01 | 0.03 | 0.00 | 0.95 | 0.06 |
| uncredited fed (100) | 0.01 | 0.01 | 0.02 | 0.01 | 0.94 | 0.08 |
| uncredited starved (131) | 0.00 | 0.01 | 0.03 | 0.00 | 0.95 | 0.06 |

Restricting to the 546 FED topics (support_frac ≤ 0.5, background kept) — the control
for "a thousand standardized near-constant topics soak up |w| mass" — changes nothing:
own 0.00 / 0.02 / 0.00, other 0.95 / 0.89 / 0.94, top-10 0.07 / 0.14 / 0.08. The fed
nodes' own topics carry 20× the data evidence of the credited ones (λ mass 1319 vs 65
vs floor 61) and get the same ~zero weight. Of each head's five largest weights, 68–90%
sit on unrelated nodes' blocks.

## Why it matters

- **The insight ladder's AUC claims are within the instrument's error.** 0082/0083
  ("spectral init costs case-finding"), 0085 ("eta costs AUC by 3×"), 0087 ("the co-fit
  head lowers AUC 0.78 → 0.7555") are differences of 0.02–0.03 measured with a readout
  whose own regularization is worth 0.037 and whose solve did not converge. None of
  those deltas is established. The record for those experiments is "not
  distinguishable from flat" until they are re-read at a converged, regularized head
  (`make gated-pc-readout ID=<n> GPR_ARGS="--readout-l2 100"`; the transform is
  cached, ~20 min each).
- **"Legibility ≠ discriminability" was never tested against the own topic.** The head
  reads every node out of ~everything except its own block, fed or starved, prior-shaped
  or not. Whether a node's own block carries its signal is simply unmeasured: the
  decisive test is an ablation readout (own-block-only vs drop-own-block vs all), which
  the head solver can do with a per-node weight mask on the driver side (no closure
  payload, ADR 0047-clean). That test decides whether the gated per-node block is doing
  anything for case-finding at all — 0081's "the signal is in the shared topics" said
  so at whole-Mondo scale; this is its mechanism, at the decoder.
- **Diffuse heads are also the VOI worry made concrete.** A head that spreads weight
  over 1,400 unrelated topics is not a feature-level explanation of anything; a
  regularized (or hierarchy-masked) head is the precondition for any per-feature
  "what would knowing X buy" reading.
- **Together with 0088 (the strip is a no-op; prevalent numbers blend tracking) the
  0113–0120 arc's numbers need two corrections before any of them is compared:** a
  regularized head, and the incident cohort.

## What changed in the tree

`--readout-l2` on the fit driver (recorded in the manifest as `readout_l2`), the
`readout_l2` front-matter key, and on `gated_pc_readout` with manifest precedence
(CLI > manifest > 1.0), so a re-readout reproduces the fit's own ridge. The default
stays 1.0 — the record — so no existing number moves silently; new experiments should
set `readout_l2: 100` explicitly and say so. The heads sidecar carries `W_std`.

## Reproduce

```bash
make -C analysis/cloud gated-pc-readout ID=120 GPR_ARGS="--readout-mode distributed --readout-l2 100"
make -C analysis/cloud inspect-topics ID=120 CREDITED=1 INSPECT_ARGS="--collinearity"
```
