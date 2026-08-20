---
id: 103
slug: mondo-cv-readout-ab-gate
status: pending
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

(pending)
