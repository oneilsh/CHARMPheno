# ADR 0046 — The readout is a distributed batched-L-BFGS multi-head fit, not a driver collect

**Date:** 2026-08-20
**Status:** Accepted
**Context:** the PC-arc closeout (`docs/reports/2026-08-20-pc-arc-closeout-…`) makes the
unsup gate + post-hoc readout LR the model-of-use; the plan
`docs/superpowers/plans/2026-08-20-distributed-readout-plan.md` (v2/v2.1) sizes why the
existing readout breaks at whole-Mondo and designs the replacement. Exp 0103 is the
cardiovascular-scale equality gate.

## Decision

### 1. The per-node readout LRs are fit by ONE batched distributed L-BFGS, not per-node sklearn on a driver collect

The driver path (`_collect_theta_labels` → per-node sklearn) collects θ (D,K) float64 plus
DENSE (D,C) float64 label/mask for both splits — ~24 GB at whole-Mondo (C≈K≈3,300,
D≈300k) against the 8g PC driver, and its bounding lever (`readout_sample_frac`, uniform
row sampling) guts exactly the rare-tail positives the conditional/VOI positioning cares
about. Spark ML's own `LogisticRegression` is the right algorithm at the wrong granularity
(one label per job = C sequential re-scans). On frozen θ the C per-node problems are
independent convex fits that can share every data pass, so all C heads are fit together:
executors treeAggregate raw-space per-node stats (summed log-loss, `Σ(p−y)·θ`, `Σ(p−y)`)
once per L-BFGS pass; the driver holds only parameters (W is ~87 MB at 3,300²) and runs a
batched L-BFGS whose per-node Armijo step control and convergence freezing are EXACT
because the objective is separable (`analysis/pc/batched_lr.py`). No θ, label, or mask
matrix ever reaches the driver. L-BFGS, not blockwise Newton: O(m·C·K) history vs the
O(C·K²) Fisher wall that killed the dense co-fit head (exp 0101).

### 2. The objective replicates the sklearn oracle exactly; equality is gated, not assumed

Every formulation choice of `_lr_proba_per_label_masked` is preserved because each is
load-bearing for cross-path comparability: SUMMED log-loss + `0.5·‖w‖²` (sklearn `C=1.0`),
UNPENALIZED intercept, per-node standardization on the node's own observed train rows —
folded in as a fixed affine reparameterization (aggregate raw, optimize standardized,
fold back; scoring needs no scaler), with sklearn's `sd→1.0` zero-variance rule (an eps
floor divides cancellation residue by 1e-12 and manufactures phantom gradients).
Degenerate nodes (empty/single-class observed set) are masked out of the solve and get the
oracle's constant fallback. Production `gtol=1e-4` = sklearn's own tol; expected residual
disagreement vs the driver path is ~5e-4 per-cell probability (sklearn is the
less-converged party) with per-node AUC deltas ~0. `--readout-ab-check` prints the
equality report; exp 0103 is the C=444 gate that must pass before the driver path is
trusted less than the distributed one.

### 3. Eval stays on the driver via a LEAN collect; fully distributed metrics are the escape hatch

What breaks the driver is the K-wide float64 feature collect, not eval: once the fit is
distributed, eval needs only the TEST split's per-node probabilities + labels, collected
as float32 proba + uint8 y/mask (~6 bytes/cell; ~2 GB at D_te≈80k, C≈3,300). The entire
existing metric stack (`readout_from_proba`, mask-independent `conditional_readout`,
`detection_readout`, the quartile split) therefore runs UNCHANGED — zero semantic-drift
risk. `distributed_readout.score_cells_df` + `per_node_metric_rows` (per-node (y,p) pairs
are 16 bytes/cell, so even an O(D) node group is ~MB-scale) are implemented and tested as
the escape hatch for when D_te×C outgrows the driver.

### 4. Mode is a routed flag, defaulting by scale

`--readout-mode {driver,distributed,auto}`, `auto` = driver at C≤500 (byte-identical
legacy behavior, keeps the localized-oracle/ladder diagnostics, which fit per-node
feature-SUBSET LRs the batched solver deliberately does not express) else distributed
(`readout_sample_frac` is ignored — it bounded a collect that no longer happens).
`distributed_readout.py` ships to executors as a bare py-file because its kernels pickle
by module reference; `analysis.pc` (solver, sklearn) stays driver-side by construction.

## Consequences

- Whole-Mondo (K≈3,300) readout is unblocked without subsampling anywhere in the fit;
  rare-node positives are always all used.
- The converged full-K frozen-θ multi-head is the closeout §6 revival-condition candidate:
  the co-fit variant is the same solver warm-started against the moving θ, so the one open
  PC question is testable at small marginal cost.
- Open at cluster scale (plan "Steps" 4): driver L-BFGS history memory policy at C·K≈11M
  params (float32 history vs node batching), treeAggregate cost under preemption, and the
  `full`-mask-mode per-pass matvec cost — measure on exp 0103 and the first whole-Mondo
  smoke before optimizing.
