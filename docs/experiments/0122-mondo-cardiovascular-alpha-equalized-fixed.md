---
id: 122
slug: mondo-cardiovascular-alpha-equalized-fixed
status: done
model_class: gated_pc
cohort: population_mondo_all
cohort_def: population_mondo_all
disease: rare_priority
# THE HELD-TILT ARM (re-specced 2026-09-11 after 0121's fit). 0121 = equalized init +
# learned alpha: the empirical-Bayes Newton step collapsed EVERY block to its 1e-3 floor
# within 50 iterations (alpha min 0.0010 / max 0.0026 / mean 0.0012 from an init of mean
# 0.5 and a ~1000x leaf/ancestor spread) — the ELBO wants alpha ~ 1/K for this corpus and
# walks there in a few steps from any start, so 'tilt then optimize' cannot test the tilt.
# This arm HOLDS it: equalized init, optimizer off. Still derived, not tuned — the only
# scale is doc_concentration's 0.5 mean carried from 0113. Pairs with 0113 (uniform 0.5,
# fixed) to price the children-first asymmetry alone, and with 0121 to price the
# alpha-collapse (0.5 -> ~0.001) that the optimizer produces.
dag_source: mondo_native
mondo_branch: MONDO:0004995
tpn: 5
max_iter: 50
diag_only: true
preindex_closure: false
readout_mode: distributed
readout_theta_topm: 256
readout_l2: 100
weight_y: 0.0
weight_y_warmup_iters: 0
skip_unsup_gated: true
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
optimize_doc_concentration: false
alpha_init: equalized
head_optimizer: newton
head_newton_ridge: 0.05
head_l2: 0.01
grad_cavi_iters: 15
topic_trust: 0.05
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
spark_conf:
  # 0113's geometry, K = 1,498 as there.
  spark.executor.cores: 2
  spark.executor.memory: 8g
  spark.executor.memoryOverhead: 3g
  spark.dynamicAllocation.enabled: "false"
  spark.executor.instances: 20
  spark.excludeOnFailure.timeout: 10m
  spark.cleaner.periodicGC.interval: "5min"
---

# 0122 — cardiovascular branch, children-first α HELD (equalized init, optimizer off)

0113 with `alpha_init: equalized` and the optimizer off. 0121's learned α collapsed to the
floor and erased its own init within the run, so this arm holds the asymmetry fixed: the
direct test of whether a children-first prior moves a node's identity into its own block.

## Run

Same as 0121 with `ID=122` (see that doc); same bundle key.

## Read

| head may load on | 0113 (α 0.5 fixed) | 0121 (equalized → learned ≈0.001) | 0122 (equalized, held) |
|---|--:|--:|--:|
| own block + background | | | |
| own + siblings + ancestors + background | | | |
| everything | 0.8087 | | |

0113's own-bg / family-closure ablations are not yet run at ridge 100; add them with
`make gated-pc-readout ID=113 GPR_ARGS="--readout-mode distributed --readout-l2 100 --readout-feature-mask own-bg"` (and family-closure) so all three columns pair.

## Results

**Digest (fit-only, 50 iters, held equalized α):** 77% starved, fed through depth 2, depth-3
median evidence 66.8 (floor 60.7), p90 1.51e3 — vs 0113 (uniform 0.5): 72% / depth 3 / 160 /
5.83e3, and vs 0121 (learned ≈0.001): 79% / depth 2 / 65.8 / 1.28e3. **The held
children-first tilt did not feed a single extra depth; it is indistinguishable from the
collapsed learned α and slightly worse than uniform.** α is closed as a lever in both
directions (insight 0091). Readout ladder not run — no reason to price a representation
that did not move.
