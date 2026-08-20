# Distributed per-node readout — driver-safe at whole-Mondo scale

**Date:** 2026-08-20 (v2 — batched multi-head fit replaces fit-side subsampling)
**Context:** the scale-back handoff (`docs/reports/2026-08-20-pc-arc-closeout-…`) makes the
unsup gate + post-hoc readout LR the model-of-use and flags the readout's driver-side
θ-collect as "the one thing to watch" at whole-Mondo. This plan promotes it from watch item
to **blocker**: collecting per-doc θ (and dense per-doc label/mask) to the driver breaks the
scalability-first position at K≈3,300. The readout must become a distributed fit.

**v2 note:** v1 proposed exploding (doc, node) cells and fitting each node's sklearn LR in
one `applyInPandas` task, with per-node case-control subsampling to bound task memory. That
subsamples where we don't have to. The v2 design fits ALL C heads in ONE distributed
optimization on full data — the same treeAggregate seam the engine's co-fit head already
uses, with batched L-BFGS instead of blockwise Newton. Subsampling survives only as a
debug/prototyping fallback.

## Why the current path breaks (sized)

The readout path (`gated_pc_cloud.py:_collect_theta_labels`, used at ~799/1205/1347)
collects to the driver, for train AND test:

- **θ (D,K) float64** — at K≈3,300 that is ~26 KB/doc; a 300k-doc cohort is **~8 GB**.
- **label (D,C) + mask (D,C) float64** — same shape again: **~16 GB** more. (At
  cardiovascular scale, C=K=444, the three arrays fit the 8g PC driver with
  `readout_sample_frac=0.3`; at whole-Mondo they are 10× over it.)
- Each scored arm then builds (N,C) proba arrays on the driver on top of that.

`readout_sample_frac` is the WRONG lever: uniform row sampling guts the rare tail (a node
with 100 positives at frac 0.05 keeps ~5) — exactly the Q1 slice the conditional/VOI
positioning cares about.

**A second, earlier wall (fit-side):** the Step-A adapter (`attach_labels`) materializes
DENSE per-doc `label`/`labelMask` C-vectors as DataFrame columns — ~53 KB/row at C≈3,300,
~16 GB of corpus weight at 300k docs — paid even by the unsup mainline (`weight_y=0`),
which never reads them during the fit. The sparse source of truth (`frontier`) is already
a column.

## Design (v2): one batched multi-head fit, exact eval, nothing D-sized on the driver

**Why not Spark ML's `LogisticRegression` directly:** it is the right *algorithm*
(executor-side treeAggregate of gradients, driver-side L-BFGS over coefficients only) at
the wrong *granularity* — one label per job. C≈3,300 nodes means 3,300 sequential Spark
jobs, each re-scanning θ several times per L-BFGS iteration, plus per-node observed-row
masking and per-node standardization that don't fit its API. That is the
"hundreds of unrelated classifiers" shape the project set out to avoid.

**The key structure:** on FROZEN θ the C per-node problems are independent convex K-dim
fits that can SHARE every data pass. One treeAggregate computes all C gradients at once —
doc d contributes `(σ(w_c·θ_d + b_c) − y_dc)·θ_d` to node c for each cell the mask
observes. This is exactly the engine's existing head-stats seam (`FlatLogisticHead`'s
per-node aggregation in `spark_vi/models/topic/pc.py`) with the blockwise-Newton solve
swapped for **batched L-BFGS** — i.e. the stash branch's "matrix-free full-K head"
(handoff §6), run on frozen θ. And frozen θ kills both things that sank the co-fit head:
no moving target (the +0.065 "solver" lever in the 0102 ladder was mostly the 1-step
Newton chasing a moving θ) and no O(C·K²) Fisher (L-BFGS is O(C·K)).

1. **Fit — batched distributed L-BFGS over all C heads.**
   - Parameters on the driver: W (C,K) + b (C) ≈ **87 MB** at 3,300² float64. L-BFGS
     history (m=5–10) is 2m such matrices — ~0.9–1.7 GB; use float32 history and/or fit
     nodes in batches of ~1,000 if the 8g driver objects.
   - Per pass: broadcast W (87 MB — same order as the existing λ broadcast), treeAggregate
     per-node gradient sums. `closure` mask mode: each doc touches only its observed cells
     (tens) → cheap. `full` mode: a (C,K)@(K,) matvec per doc — heavy but embarrassingly
     parallel, and one pass replaces 3,300 separate scans.
   - The objective is separable across nodes, so per-node step control vectorizes: one
     step-size vector, backtrack only where a node's Armijo check fails; freeze nodes as
     they converge so late passes touch only stragglers.
   - **Formulation must replicate the sklearn oracle** (`_lr_proba_per_label_masked`:
     L2 with sklearn's `C=1` scaling — summed log-loss + ½‖w_c‖², UNPENALIZED intercept —
     per-node standardization on that node's observed train rows). Standardization comes
     from one masked mean/var aggregate ((C,K)×2 ≈ 174 MB, one-time) folded into the
     objective as a fixed affine reparameterization — the fitted W is then mapped back to
     raw-θ coordinates, so scoring needs no scaler.
2. **Score (executors).** Broadcast the fitted (W, b); per doc emit `P(node c)` for the
   cells eval needs. No collect.
3. **Eval — exact, distributed, no subsampling.** Per-node metrics (AUC/AP, P@R, R@FDR,
   ECE, reliability bins) need only that node's `(y, p)` pairs — **16 bytes/cell**, not a
   K-wide feature row. Explode `(node, y, p)` and `groupBy(node)`: even a node positive
   for every doc (the root) is an O(D)-row group of ~5 MB — trivially fine. v1's skew
   guard existed to bound (rows × K) per-task design matrices; the batched fit never
   materializes those, so the guard is moot. Driver receives (C,)-sized metric tables
   only; `conditional_readout`, the quartile split and `format_arm_readout` consume them
   unchanged (the `P(child|parent)` cohorts = parent-positive cells, already carried by
   the closure explode).
4. **VOI untouched:** β never leaves the fitted model; LLR/EIG readouts are per-node and
   small.
5. **Fallback (debug only):** the v1 per-node `applyInPandas` sklearn path with
   case-control caps — useful for local prototyping and as an independent
   cross-implementation check at small C, not the production path.

**Strategic bonus — this readout IS the §6 revival-condition artifact.** The handoff's
revival condition wants a co-fit head that matches the gate (≥ ~0.74, now Q1 too). The
converged full-K frozen-θ multi-head built here is exactly that candidate's solver; the
co-fit variant is the same machinery warm-started and amortized (a few batched L-BFGS
steps per SVI iteration) against the moving θ. Building the readout this way makes the
one open PC question testable at a small marginal cost — one artifact, two uses.

## What must NOT change

`pc_topics_lr` semantics: same per-node masked LR, same formulation, same metrics. The
correctness gate is an **A/B equality run at cardiovascular scale** (C=444): batched
distributed readout vs the current driver readout on the same frozen fit — per-node AUC
equal to numerical tolerance before the driver path is retired.

## Steps (ordered)

1. **Batched L-BFGS multi-head on frozen θ** behind `readout_mode={driver,distributed}`
   (default `driver` at C≤500): the treeAggregate gradient seam (reuse/extend
   `FlatLogisticHead` stats), sklearn-equivalent objective incl. standardization
   reparameterization, vectorized step control + convergence freezing.
2. **Correctness gate:** cardiovascular A/B equality run vs the driver readout (same
   frozen fit, same seed).
3. **Distributed eval:** the `(node, y, p)` explode + per-node metric aggregation;
   equality-check metric tables against the driver `_bundle_masked` on the same scores.
4. **Whole-Mondo readout smoke** on the first K≈3,300 gate fit (watch: broadcast size,
   driver L-BFGS memory, stragglers under preemption — light per-pass state means cheap
   recompute, the same property that saved the localized head in 0102).
5. **Frontier-only corpus** for the unsup mainline: stop materializing dense
   `label`/`labelMask` columns at assembly; derive cells where needed. (Separate seam —
   touches `attach_labels` consumers; removes the fit-side ~16 GB.)
6. **(When ready to test PC revival)** wrap the same solver as the co-fit head
   (warm-start + amortized steps + `head_trust_move`) and re-run the 0102 A/B — the §6
   question answered with the §6-prescribed tool.

## Open questions

- L-BFGS memory policy at C·K ≈ 10.9M params (float32 history vs node batching) — decide
  on the 8g driver empirically at C=444 first.
- `full`-mask-mode pass cost at whole-Mondo (D × C·K flops/pass): if it binds, the
  closure-mode readout is the deliverable and `full` becomes the reported-but-sampled
  diagnostic — measure before optimizing.
- Degenerate nodes (single-class observed set) must yield the same constant-prediction
  fallback as `_lr_proba_per_label_masked`, so macro means stay comparable.
