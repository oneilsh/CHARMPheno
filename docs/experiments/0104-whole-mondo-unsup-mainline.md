---
id: 104
slug: whole-mondo-unsup-mainline
status: pending
model_class: gated_pc
cohort: population_mondo_all
cohort_def: population_mondo_all
disease: rare_priority
# THE MAINLINE DELIVERABLE RUN (closeout handoff §7.2; gated on exp 0103's A/B gate
# PASSING). First whole-Mondo fit: the FULL powered DAG (no mondo_branch => all body
# systems), K≈3,827 / C≈3,820 (insight 0071: 2,513 powered anchors + 1,306 class nodes,
# n_bg=8, tpn=1). This is the SCALED-BACK MAINLINE, not a PC experiment: weight_y=0 makes
# the primary arm the unsupervised gated LDA itself (the PC apparatus is inert — closeout
# §5), and skip_unsup_gated drops the now-redundant twin, so ONE fit + the distributed
# readout (ADR 0046) is the whole run. Everything the driver collects is (C,)-sized or
# the lean float32/uint8 test bundle (~6 bytes/cell); the old driver readout is
# structurally impossible here (24+ GB of collects) — readout_mode pinned, not auto.
# Head params below are lineage carry-over from 0102/0103 and INERT at weight_y=0.
readout_mode: distributed
weight_y: 0.0
weight_y_warmup_iters: 0
skip_unsup_gated: true
dag_source: mondo
min_positives: 100
mondo_version: 2026-06-02
mondo_cache_dir: data/mondo
extra_domains: measurement,drug
label_mask_mode: closure
localize_head: true
head_support: path_cousins_kids
head_intercept: true
head_standardize: true
doc_concentration: 0.5
head_lr: 1.0
person_mod: 1
prior_obs_days: 0
doc_min_length: 10
min_n: 0
holdout_frac: 0.2
vocab_size: 5000
min_df: 20
min_patient_count: 20
window_mode: lookback
lookback_days: 1825
label_window_days: 365
strip_mode: both
n_bg: 8
tpn: 1
optimize_doc_concentration: true
head_optimizer: newton
head_newton_ridge: 0.05
head_l2: 0.01
grad_cavi_iters: 15
topic_trust: 0.05
max_iter: 100
subsampling_rate: 0.1
tau0: 64.0
kappa: 0.51
cavi_max_iter: 100
cavi_tol: 0.001
with_dag_head: false
baseline_max_iter: 100
min_label_count: 20
eval_every: 0
num_partitions: 96
seed: 42
cache_uri: hdfs:///user/dataproc/charm/case_finding_cache
---

# 0104 — Whole-Mondo unsupervised mainline: the first all-body-system gate + readout

**Why.** The scaled-back mainline (closeout 2026-08-20) is the unsup gated LDA + post-hoc
readout, and its whole-Mondo scale-up is next-steps item 2. Both former blockers are
cleared: the O(C·K²) dense-head wall belonged to the co-fit head (absent at weight_y=0 —
insight 0071's correction), and the readout's driver collect is replaced by the
distributed batched-L-BFGS fit + lean eval (ADR 0046, gated by exp 0103). This run
produces the first population-wide, all-condition calibrated per-node posteriors — the
substrate for exports, dashboards, and VOI (next-steps items 3+).

**Run only after exp 0103's A/B equality gate passes.**

## Scale expectations (watch these, they are the run's second deliverable)

- **Fit:** K grows 444 → ≈3,827 (~8.6×). Per-iter cost is roughly linear in K at fixed
  minibatch (gated E-step is O(|allowed|), but held-out CAVI and λ updates see full K);
  budget several × 0103's per-iter wall-clock and consider `num_partitions` 96 → 192 if
  executors are idle-skewed.
- **Readout fit:** L-BFGS driver state at C·K ≈ 14.6M params: W+b ~117 MB, m=6 history
  ~1.4 GB — inside the 8g PC driver but tight next to the eval bundle; set
  `CHARM_DRIVER_MEMORY=12g` if the solve OOMs (the plan's float32-history/node-batching
  fallbacks exist but measure first). Expect the heartbeat to show tens of iterations,
  each one treeAggregate over the train split.
- **Lean eval bundle:** ~6 bytes/cell → at D_te≈80k, C≈3,820 about 1.9 GB. The
  calibration diagnostic holds one extra float64 test-split copy while it runs.
- **Moments aggregate:** (C,K)×2 float64 ≈ 234 MB driver-side, one-time.

## What to read (make -C analysis/cloud report ID=104)

1. **The fit itself** — ELBO trajectory, per-node α behavior at K≈3,827 (ADR 0045's
   floor is load-bearing here), and wall-clock/iter.
2. **unsup readout macro + rarity quartiles** — the first whole-Mondo AUC/AP; compare the
   cardiovascular subset against 0103's unsup arm (expect the same ballpark on shared
   nodes; a big drop = the fit, not the readout).
3. **`batched L-BFGS` heartbeat + summary** — passes, converged/fittable, stalled count,
   wall-clock: the scale read that decides whether whole-population readouts need the
   float32-history/node-batching work.
4. **Per-node ECE / calibration** — the deliverable is calibrated posteriors; if per-node
   ECE degrades at depth, the isotonic layer (already in the driver) is the lever.
5. **[cost] driver RSS** at the readout and eval phases vs the estimates above.

## Run

```bash
cd ~/repos/CHARMPheno && git pull origin claude/gated-conditional-voi && \
  CHARM_DEV=1 CHARM_SPARK_CONF='spark.locality.wait=0s' make -C analysis/cloud exp ID=104  # smoke first
CHARM_SPARK_CONF='spark.locality.wait=0s' make -C analysis/cloud exp ID=104
make -C analysis/cloud report ID=104
```

## Run log

**2026-08-21 — smoke attempt 1: driver-JVM heap OOM at fit iteration 11 — a NEW scale
wall, now fixed.** The fit itself was healthy (48.6s/iter at K≈3,827, α floor holding,
ELBO rising, domain fracs sane) — then the driver JVM OOM'd in `task-result-getter` /
dispatcher threads; the executor "lost heartbeat" storm that followed was collateral of
the dead driver, not executor failure. Root cause: the SVI stats `treeReduce` (depth 2)
ships DENSE λ-shaped partials — **~355 MB each at K≈3,827 × V=11,601** (vs 41 MB at
C=444) — so the driver received ~sqrt(96)≈10 of them per iteration ≈ 3.5 GB of serialized
blocks through an 8g heap. Neither the exp doc's watch-list (which predicted the pinch at
the READOUT) nor 0103 could see this; it is K·V-driven and fit-side. Fix (same commit):
`spark_vi.core.runner._agg_depth` sizes the treeReduce depth from the params payload —
depth 3 above 128 MB/partial (driver burst ÷ ~P^(1/6)), byte-identical depth 2 below; the
readout aggregates got the same auto-rule (`_fit_readout_heads` depth=None → auto; its
(C,K) partials are ~117 MB at whole-Mondo). Belt to that suspenders: run the smoke with
`CHARM_DRIVER_MEMORY=16g` — depth 3 cuts the burst to ~4.6 partials ≈ 1.6 GB, comfortable
at 16g with the L-BFGS state beside it.

**2026-08-21 — smoke attempt 2 (depth-3 + 16g): FIT CLEAN, READOUT CRAWLS — the θ-width
lever comes due.** The fit is healthy end-to-end at whole-Mondo (30 dev iters, ~43s/iter
at K=3,827 — barely above cardiovascular's ~31s; depth-3 aggregation + 16g driver held,
zero drama). The readout is where the scale bites: **C=3,820, 3,057 fittable / 763
degenerate, 56.2M observed train cells** (5.7× cardiovascular — deeper DAG, bigger
closures), and each cell's dot is K=3,827 wide (8.6×) → ~49× more cell-work: measured
**~65s/data pass** vs 1.9s at C=444 (~1.7 TB memory traffic per pass), ~6-7h per 60-iter
dev solve, × main + calibration solves per arm. This is the plan §1 "θ-width lever" the
design deliberately deferred pending measurement: per-doc θ over 3,827 topics should be
mass-concentrated, so a top-m truncated-θ readout (m≈256) cuts pass cost ~K/m ≈ 15×.
In flight: always-on θ mass-coverage logging, a `readout_theta_topm` flag (default off,
sparse-exact kernels, by-node vectorization), and a dev-profile skip of the calibration
solve. Operational note: `results_partial.json` lands when the MAIN solve's readout
completes (at the `gated_pc (pc_topics_lr): macro AUC=` line) — the calibration solve
after it is safely interruptible in a smoke.

**2026-08-21 — UNBLOCKED: the 0103 A/B gate PASSED** (macro |Δ| ≤ 1.1e-4 both arms; see
0103's run log). Reference bar from 0103's full-row readout: unsup cardiovascular
**0.7584 AUC / 0.5428 AP over 241 nodes**, pooled conditional ECE 0.0028 (isotonic →
0.0010). Since staging, the readout also gained warm starts + a CHARM_DEV cap of 60
solver iterations (insight 0074), so the smoke's readout is ~3× cheaper than 0103's.
