# 0089 — The readout LR was the bottleneck: ridge 100 on the SAME θ lifts 0120's macro AUC 0.7555 → 0.7927, and standardized heads put ~0 weight on a node's own topic even among the 546 fed topics — the arc's AUC deltas were read through an under-regularized, under-converged instrument

**Date:** 2026-09-09
**Topic:** readout, evaluation, regularization, case-finding, gated-pc, exp 0120, insight ladder 0082–0087
**Status:** Confirmed on exp 0120 (three re-readouts on the cached transform, same θ, same split, `gated-pc-readout --readout-l2`; `inspect_topics --collinearity` on the standardized heads), then on the PAIRED re-read of 0113 / 0116 / 0120 at ridge 100 (section below). Pooled figures only.

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

## The paired re-read at ridge 100 (2026-09-09, 0113 / 0116 / 0120 re-readouts + `readout-ab`)

All three arc runs re-read on the cached bundle at `--readout-l2 100` (converged: 234 / 234 /
226 of 259 heads), then paired per node on the 193 shared scored nodes:

| run | what it adds | macro AUC @ l2=1 (record) | macro AUC @ l2=100 | detection AUC @100 |
|---|---|--:|--:|--:|
| 0113 | random init, no prior, no head | 0.7813 | **0.8087** | 0.604 |
| 0116 | + profile-eta S=1.0 | 0.7804 | **0.8098** | 0.590 |
| 0120 | + L-BFGS co-fit head (w_y=12, trust 0.03) | 0.7555 | **0.7927** | 0.492 |

| paired (this − base) | all (n=193) median / mean | up/down | credited (n=14) | uncredited (n=179) |
|---|---|---|---|---|
| 0116 − 0113 | −0.0008 / +0.0012 | 91/102 | +0.0104 (9/5) | −0.0014 (82/97) |
| 0120 − 0116 | **−0.0151 / −0.0172** | **48/144** | −0.0138 (2/12) | −0.0151 (46/132) |

Three verdicts, now each on a converged, paired footing:

1. **The instrument was worth +0.027 to +0.037 on every run** — the same size as, or larger
   than, every delta the ladder reported. The record numbers (0.78 / 0.78 / 0.7555) were
   under-converged, under-regularized readouts of representations that support 0.81 / 0.81 /
   0.79.
2. **Profile-eta is a clean AUC null, paired.** −0.0008 median, 91 up / 102 down; the credited
   nodes +0.010 on n=14 (9 up / 5 down — not distinguishable from zero). 0084/0085's "the prior
   is an interpretability lever, not an AUC lever" SURVIVES the correction; 0085's "and starts
   costing by 3×" (0117) needs the same re-read before it is cited.
3. **The co-fit head's cost is real, and UNIFORM.** −0.015 median, 144 of 193 nodes down,
   every depth d2–d7 negative (−0.005 to −0.023), credited and uncredited alike. 0087's
   verdict ("shaping does not discriminate and mildly hurts") is RE-ESTABLISHED on a paired
   converged read, with a corrected magnitude (−0.017 mean, not −0.025) — but 0087's
   mechanism story is not: it did not "hurt most where aimed" (credited −0.014 ≈ uncredited
   −0.015), and `self-w ≈ 0` is universal, not the head's doing. A uniform cost across nodes
   the localized head never pulled on says the label term perturbs the SHARED representation
   the decoder actually reads (the PC θ-inference and sstats are global even when the head is
   block-local). Detection AUC 0.604 → 0.492 under the head says the same thing louder.
4. **"Credited < uncredited" flips.** At the converged head the profiled rare nodes decode at
   least as well as the rest (0116: credited median 0.847 vs uncredited 0.817; 0120: 0.799 vs
   0.796). 0087's "it hurt most where it was aimed" and the l2=1 depth slide (d2 0.816 → d7
   0.729) were instrument artefacts; the converged depth slide is d2 0.829 → d7 0.773.

So the arc's honest summary is: **profile-eta buys legibility at zero AUC cost; the co-fit
head costs ~0.015 AUC uniformly and buys nothing; the representation itself supports ~0.81
prevalent macro on this branch** — read through a decoder that ignores the per-node blocks,
on a corpus with no leakage strip (0088). The own-block ablation and the incident cohort
are the two remaining corrections before any number here is compared to anything outside
the arc.

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
