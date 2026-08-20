# Distributed per-node readout — driver-safe at whole-Mondo scale

**Date:** 2026-08-20
**Context:** the scale-back handoff (`docs/reports/2026-08-20-pc-arc-closeout-…`) makes the
unsup gate + post-hoc readout LR the model-of-use and flags the readout's driver-side
θ-collect as "the one thing to watch" at whole-Mondo. This plan promotes it from watch item
to **blocker**: collecting per-doc θ (and dense per-doc label/mask) to the driver breaks the
scalability-first position at K≈3,300. The readout must become a distributed per-node fit.

## Why the current path breaks (sized)

The readout path (`gated_pc_cloud.py:_collect_theta_labels`, used at lines ~799/1205/1347)
collects to the driver, for train AND test:

- **θ (D,K) float64** — at K≈3,300 that is ~26 KB/doc; a 300k-doc cohort is **~8 GB**.
- **label (D,C) + mask (D,C) float64** — same shape again: **~16 GB** more. (At
  cardiovascular scale, C=K=444, the same three arrays are ~1 GB per split with
  `readout_sample_frac=0.3` — inside the 8g PC driver; at whole-Mondo they are 10× over it.)
- Each scored arm then builds (N,C) proba arrays on the driver on top of that.

`readout_sample_frac` is the WRONG lever to fix this: uniform row sampling guts the rare
tail (a node with 100 positives at frac 0.05 keeps ~5) — exactly the Q1 slice the
conditional/VOI positioning cares about (quartile split, `_print_headline`).

**A second, earlier wall (fit-side):** the Step-A adapter (`attach_labels`) materializes
DENSE per-doc `label`/`labelMask` C-vectors as DataFrame columns. At C≈3,300 that is
~53 KB/row → ~16 GB of extra corpus/shuffle weight at 300k docs — paid even by the unsup
mainline (`weight_y=0`), which never reads them during the fit. The sparse source of truth
(`frontier`) is already a column.

## Design

**Never move a (D,·) matrix to the driver. Explode sparse observed (doc, node) cells on
executors; fit each node's LR where its rows live; return only per-node results.**

The structural fact that makes this cheap: under `label_mask_mode=closure`, a doc's
observed set is its frontier closure + siblings-of-closure (`frontier_to_label`) —
O(depth·fan-out) nodes, tens not thousands. Positives are the closure; near-boundary
negatives are the siblings. The exploded row count is ~D × (avg observed set), not D×C.

1. **Observed-cell explode (executors).** From `frontier` (not the dense label cols),
   emit `(node c, y, θ_d)` rows per observed cell:
   - `closure` mode: exact — closure ∪ sibling cells, y=1 on closure.
   - `full` mode: positives exactly (closure cells); negatives are "every other doc" per
     node, so use **per-node case-control sampling**: include doc d as a negative for node
     c iff `hash(d, c) < q_c`, with `q_c = cap_c / n_neg_c`, `cap_c = min(r·n_pos_c,
     N_max)` (r≈20, N_max≈20k). `n_pos_c` comes from one cheap aggregate
     (`node_patient_counts`). Expected exploded rows: Σ_c cap_c — bounded and tunable.
     Rare-node positives are ALWAYS kept: strictly better for the rare tail than
     `readout_sample_frac`, which this replaces.
   - **Skew guard — cap positives too.** The explode bounds the TOTAL row count, but
     `groupBy(node)` lands each node's rows in ONE task, and a shallow/high-prevalence
     node is positive for nearly every foreground doc (the root for ALL of them) — an
     O(D) group. So positives are also hash-subsampled, at rate `s_c = P_max / n_pos_c`
     when `n_pos_c > P_max` (~20k). With both caps every group is ≤ `P_max + cap_c` rows
     regardless of D or depth; only shallow, data-rich nodes ever hit the positive cap,
     so the rare tail is untouched. Applies in `closure` mode too (its per-DOC sets are
     small, but a common node's per-NODE row count is not).

2. **Per-node fit (executors).** `groupBy("node").applyInPandas`: sklearn LR per node in
   the ladder-validated oracle formulation (intercept + standardized — exp 0099/0102's
   winning readout formulation), same masked semantics as
   `analysis.pc.evaluate._lr_proba_per_label_masked`. Each task also scores that node's
   TEST cells and computes the per-node metrics (AUC/AP, P@R, R@FDR, ECE, reliability
   bins) in place.

3. **Case-control calibration correction.** Subsampling positives at rate `s_c` and
   negatives at rate `q_c` biases the node's intercept by exactly `log(s_c/q_c)`
   (prior-correction / King–Zeng); correct `b_c -= log(s_c/q_c)` before emitting
   probabilities (s_c=1 for uncapped nodes reduces to the negatives-only correction). Required — calibrated `P` is a headline
   deliverable; do it even though per-node isotonic (`calibrate_per_node`) could absorb it,
   so raw probabilities stay meaningful. AUC/AP are invariant to negative subsampling in
   expectation; ECE is not — compute ECE only after the intercept correction, and in `full`
   mode report pooled ECE from inverse-probability-weighted negative cells.

4. **Driver receives (C,)-sized results only:** the per-node metric table (feeds
   `conditional_readout`, the quartile split, `format_arm_readout` unchanged — they are
   per-node reductions already; `P(child|parent)` cohorts = parent-positive cells, which
   the closure explode already carries) + the (C,K) coefficient matrix (~87 MB dense at
   3,300² — fine; VOI's β never left the driver's fitted model, unaffected).

5. **Task memory + the θ-width lever.** A node's design is rows_c × K: 20k × 3,300 × 8 B
   ≈ 530 MB — too fat; 5k rows or float32 is fine (~130 MB). The real lever is **top-m
   sparse θ**. NOTE: `GatedLDAModel._transform` folds held-out docs UNGATED full-K (label
   unknown at scoring time — deployment convention), so θ sparsity is empirical, not
   structural: **measure it first** (one dev-run stat: θ mass coverage at m=64/128/256).
   If ≥99.9% at m≈128, store θ as top-m sparse → shuffle D×m instead of D×K and ~20 MB
   sparse designs; if not, ship dense-float32 with a 5k cap and revisit.

## What must NOT change

`pc_topics_lr` semantics: same per-node masked LR, same formulation, same metrics. The
correctness gate is an **A/B equality run at cardiovascular scale** (C=444): distributed
readout with caps disabled vs the current driver readout on the same frozen fit — per-node
AUC must match to numerical tolerance before the driver path is retired.

## Steps (ordered)

1. **θ top-m mass measurement** — piggyback one stat on the next dev run (feeds step 5's
   dense-vs-sparse choice).
2. **Implement** the explode + `applyInPandas` per-node fit behind `readout_mode=
   {driver,distributed}` (default `driver` at C≤500, `distributed` above).
3. **Correctness gate:** cardiovascular A/B equality run (caps off, same seed).
4. **Rare-tail check:** per-node retained-positive counts vs the `readout_sample_frac`
   path — expect strict Q1 improvement (all positives kept by construction).
5. **Frontier-only corpus** for the unsup mainline: stop materializing dense
   `label`/`labelMask` columns at assembly; derive cells at readout time. (Separate seam —
   touches `attach_labels` consumers; do after the readout lands. Removes the fit-side
   16 GB.)
6. **Whole-Mondo readout smoke** on the first K≈3,300 gate fit.

## Open questions

- Negative-cap policy (r, N_max) — start r=20/N_max=20k and check per-node AUC stability
  vs caps-off at C=444 before trusting it at whole-Mondo.
- Degenerate nodes (single-class after explode) must yield NaN exactly as the current
  masked scorer does, so macro means stay comparable.
- `full`-mode pooled ECE weighting (item 3) needs a small unit test against an exact
  driver-side computation on synthetic data.
