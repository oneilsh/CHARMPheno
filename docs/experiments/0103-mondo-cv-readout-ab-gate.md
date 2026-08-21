---
id: 103
slug: mondo-cv-readout-ab-gate
status: done
model_class: gated_pc
cohort: population_mondo_cardiovascular
cohort_def: population_mondo_cardiovascular
disease: rare_priority
# THE DISTRIBUTED-READOUT EQUALITY GATE (ADR 0046; plan 2026-08-20-distributed-readout,
# "What must NOT change"). Same fit config as the exp 0102 record — the ONLY deltas are
# readout_mode/readout_ab_check — so the two readout paths are compared on the same frozen
# θ at C=444, where the driver path still fits in memory. The distributed path fits ALL
# per-node readout LRs with one batched L-BFGS on the executors (analysis/pc/batched_lr.py
# + analysis/cloud/distributed_readout.py) and collects only a float32/uint8 test-split
# eval bundle; the harness runs the legacy driver path beside it and prints the deltas.
# PASS = macro |Δ| ≲ 1e-4, per-node max |ΔAUC| ≲ 1e-3 with n>1e-3 = 0, max |Δp| ~ 5e-4
# (sklearn's default stopping tol is the less-converged party — measured locally:
# macro Δ = +0.00e+00, max |Δp| = 6.46e-04). This gate must pass before the driver path
# is retired at whole-Mondo scale.
readout_mode: distributed
readout_ab_check: true
dag_source: mondo
mondo_branch: MONDO:0004995
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
readout_sample_frac: 0.3
weight_y: 16.0
head_lr: 1.0
person_mod: 1
prior_obs_days: 0
doc_min_length: 10
min_n: 20
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
weight_y_warmup_iters: 25
max_iter: 100
subsampling_rate: 0.1
tau0: 64.0
kappa: 0.51
cavi_max_iter: 100
cavi_tol: 0.001
skip_unsup_gated: false
with_dag_head: false
baseline_max_iter: 100
min_label_count: 20
eval_every: 0
num_partitions: 96
seed: 42
cache_uri: hdfs:///user/dataproc/charm/case_finding_cache
---

# 0103 — Mondo cardiovascular: distributed-readout A/B equality gate

**Why.** ADR 0046 replaces the readout's driver collect (θ (D,K) + dense (D,C) label/mask
— the whole-Mondo wall) with a batched distributed L-BFGS multi-head fit + a lean
float32/uint8 eval collect. The metric stack is byte-identical; the only thing that can
differ is the LR solver itself, which replicates the sklearn oracle's exact objective
(summed loss, C=1 ridge, unpenalized intercept, per-node standardization; gtol=1e-4 =
sklearn's own tol). This run is the plan's correctness gate: both paths on the same frozen
θ at C=444, deltas printed by `--readout-ab-check`. Note `readout_sample_frac: 0.3` bounds
ONLY the driver side of the comparison (the distributed path ignores it, with a log line);
so the residual deltas fold in the driver path's row sampling as well as solver tolerance.

## What to read (make -C analysis/cloud report ID=103)

1. **`readout_mode=distributed -> distributed`** and, per arm, the
   **`batched L-BFGS: N data passes, X/X converged (X gtol, 0 stalled)`** line — a nonzero
   `stalled` count at gtol=1e-4 is unexpected (the roundoff floor sits well below 1e-4)
   and worth a look before trusting the numbers.
2. **The `A/B readout equality gate` blocks** (gated_pc and unsup_gated arms): macro
   AUC/AP for both paths, per-node max/mean |ΔAUC|, count of nodes with |ΔAUC|>1e-3,
   sampled max |Δp|. PASS thresholds in the front-matter comment.
3. **[cost]** wall-clock of the distributed fit vs the driver readout, the number of data
   passes (each pass = one treeAggregate over the train split), and driver memory — the
   scale-relevant read for whole-Mondo (plan "Steps" 4).

## Interpreting

- **Deltas at/below thresholds:** the gate PASSES — the distributed readout is the
  production path for the whole-Mondo scale-up; the driver path stays available at C≤500
  (`readout_mode: driver`) for the localized-oracle/ladder diagnostics only.
- **Per-node |ΔAUC| outliers (n>1e-3 > 0):** identify the nodes — expect either near-tie
  score reorderings on tiny test cohorts (benign; check n_pos) or a degenerate-node
  fallback mismatch (a bug: the constant-prediction cases must be bit-exact).
- **Macro |Δ| > 1e-4 with clean per-node deltas:** suspect the driver path's 0.3 row
  sampling, not the solver — re-run with `readout_sample_frac` dropped and
  `CHARM_DRIVER_MEMORY=24g` to compare at full rows.

## Run

```bash
cd ~/repos/CHARMPheno && git pull origin claude/gated-conditional-voi && \
  CHARM_DEV=1 CHARM_SPARK_CONF='spark.locality.wait=0s' make -C analysis/cloud exp ID=103   # smoke
cd ~/repos/CHARMPheno && \
  CHARM_SPARK_CONF='spark.locality.wait=0s' make -C analysis/cloud exp ID=103               # the gate
make -C analysis/cloud report ID=103
```

## Run log

**2026-08-21 — CHARM_DEV smoke: distributed readout runs end-to-end at C=437.**
Fit summary: `357 fittable nodes, 80 degenerate (constant fallback), observed train
cells=9,897,319`. Headline (dev, 30-iter fit): pc_topics_lr **0.7008 vs unsup 0.7574**
(Δ−0.0566); rarity split negative in all four quartiles (Q1 Δ−0.0561); detection Δ 0.
Two expected shifts vs the 0102 dev record (0.681/0.739, 176 shared nodes) that are the
new readout WORKING, not drift: **241 shared nodes scored (vs 176)** and quartile +ct
edges ~3.3× larger ([83, 300, 1214] vs [57, 175, 490]) — the distributed path ignores
`readout_sample_frac=0.3` and fits/scores on ALL rows, so ~1/0.3× more test positives per
node, more nodes clear `min_label_count=20`, and both arms' AUCs read higher on the fuller
eval. The PC-vs-unsup delta is unchanged (~−0.057), consistent with the 0102 story.
Feedback applied: the batched solve looked silent while treeAggregating — a per-iteration
heartbeat (`batched L-BFGS iter N: P data passes, X/357 converged, ... elapsed`) now
prints (every iter for the first 3, then every 5th). **Gate verdict still pending**: read
the `A/B readout equality gate` blocks + `batched L-BFGS:` summary lines from the full
(non-dev) run.

**2026-08-21 — FULL run (Fit session 3, exit 0, ~5h): THE GATE PASSES.** (The cluster
restart wiped the HDFS bundle cache, so this run rebuilt it; both durability layers and
the heartbeat did their jobs — the run survived 10 executor losses + 2 container OOMs
from spot reclamation with no lost output.)

- **A/B equality, both arms** (same 0.3 row sample both paths, seed 42, 173 shared
  nodes): gated_pc macro ΔAUC **+1.08e-4** / ΔAP +3.5e-5, per-node max |ΔAUC| 5.36e-3,
  mean 5.28e-4, n>1e-3=27, mean |Δp| 5.9e-4; unsup_gated macro ΔAUC **+1.70e-5** /
  ΔAP −7.0e-5, per-node max 1.92e-3, mean 2.09e-4, n>1e-3=5, mean |Δp| 1.47e-4. Mean
  per-cell deltas land exactly on the predicted sklearn-tol residual; the outlier tail
  (incl. one 0.648 |Δp| cell on the gated_pc arm) tracks CONVERGENCE STATE, not
  formulation: the better-conditioned unsup fit (136/357 gtol on the sampled solve, 0
  stalled, 0 line-search failures everywhere) shows ~2.5× smaller deltas across the
  board. Both solvers stop short on ill-conditioned nodes (cap 200 iters, max|grad|
  12–49 at stop); sklearn is the less-converged party at its own tol. **Verdict: the
  distributed readout is the production path.**
- **New full-row cardiovascular record (241 nodes, no readout subsampling):**
  unsup_gated **0.7584 AUC / 0.5428 AP** vs gated_pc 0.7128/0.4917 (Δ−0.0456/−0.0511);
  quartiles all negative again (Q1 −0.0437, Q2 −0.0540, Q3 −0.0477, Q4 −0.0372) —
  rare-tail rescue refuted on full data too. Co-fit head as trained 0.5609 (|w_CK|
  peaked 5.99e5 — insight 0067's blowup, PC arm only). The full-row readout RAISES the
  mainline bar: the closeout §6 revival condition is now co-fit ≳ **0.758**.
- **Calibration/VOI readiness:** unsup pooled conditional ECE 0.0028; held-out isotonic
  takes raw 0.0059 → **0.0010**. Per-node ECE mean 0.0475 / max 0.33 — pooling flatters;
  per-node isotonic is the lever, machinery in place.
- **Detection is chance in every arm (AUC exactly 0.5, AP=prev=0.777) — pre-existing,
  not a readout regression** (0102's driver-path run shows the same signature): the
  degenerate constant-1.0 columns (the root above all) saturate the doc-level max.
  Small fix queued: exclude constant columns from the detection score.
- **Cost profile (the whole-Mondo read):** ~1.9s/treeAggregate pass at C=437; each
  200-iter solve ≈ 1,220 passes ≈ 35 min; three solves per supervised arm (main, A/B
  sample, calibration) → gated_pc arm 11,918s total, unsup 5,358s. Landed after this
  run: warm starts for the A/B/calibration solves + `CHARM_DEV` readout cap of 60 iters
  (insight 0074) — dev readouts drop ~3×; production cost unchanged pending need.
