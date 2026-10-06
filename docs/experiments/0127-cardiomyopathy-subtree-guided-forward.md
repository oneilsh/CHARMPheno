---
id: 127
slug: cardiomyopathy-subtree-guided-forward
status: pending
model_class: gated_pc
cohort: population_mondo_all
cohort_def: population_mondo_all
disease: rare_priority
# PROBE PAIR, ARM A (forward). 0125's config on the CARDIOMYOPATHY subtree
# (MONDO:0004994) as its own branch; forward deflation (ancestors-first), as 0125.
# The control arm for 0128.
#
# WHY A SUBTREE. The 0126 question (does leaves-first deflation take the prenatal
# stratum out of dilated cardiomyopathy's block, and what does it do to the parents?)
# lives entirely inside the cardiomyopathy subtree. On the full CV branch one arm costs
# ~3h, ~1.5h of it the seed walking 298 nodes in batched passes over the corpus. The
# subtree as its own label DAG is ~30 nodes: seed in minutes, fit in minutes, readout in
# minutes — BOTH orders in about an hour, including one bundle rebuild for the new
# branch. What is given up: a different label DAG (no heart disorder / cardiovascular
# disorder ancestors to deflate against; cardiomyopathy becomes the branch's d1), so
# numbers do not pair with 0125's. For the mechanism under test — peripartum patients
# training DCM's block, and parents becoming residuals under reverse order — the
# subtree is a faithful probe. The full-branch run (0126) is for the winning order only.
# The CV profile table is reused as-is: nodes outside this DAG are skipped and counted.
dag_source: mondo_native
mondo_branch: MONDO:0004994   # cardiomyopathy SUBTREE (was MONDO:0004995, the whole CV branch): ~30 nodes, minutes not hours
tpn: 5
max_iter: 50
diag_only: true
# --- the ONLY change from 0113: spectral block-aligned seed for the gated engine ---
init: spectral
spectral_method: scalable   # concatenated V ~11.6k >= 8000 threshold; dense = driver wall
spectral_d: 768             # random-projection dim: smaller = faster + bigger safe batch (see COST)
anchor_scope: closure       # node trained from its whole closure; ancestors deflated by topo order
spectral_topo_order: forward  # ancestors-first: each node's seed = its increment over ancestors
count_transform: none       # raw per-visit counts, as 0123 (binary was a detection tax, 0123)
# --- THE ONLY CHANGE vs 0123: HPO-guided spectral anchors (spec 2026-10-05) ---
spectral_anchor_profile: ~/repos/CHARMPheno/data/ontology/profile_eta_MONDO_0004995.tsv  # REGENERATED with the prenatal exclusion (see COST)
# ------------------------------------------------------------------------------------
preindex_closure: false
readout_mode: distributed
readout_theta_topm: 256
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
# LANDMINE (2026-09-11): 0113-0120 all say `optimize_doc_concentration: true` but the flag
# was INERT on the PC path until 0121 wired it (pc.py `set_alpha_policy`); their alpha was
# FIXED at 0.5. Copying that line now turns the empirical-Bayes alpha ON, and learned alpha
# collapses to the floor and re-starves depth (insight 0091; 0121: 79% starved). The first
# 0123 launch did exactly that (alpha mean 0.5 -> 0.21 by iter 19, background block 0.28
# and falling ~2.5%/iter) and was killed. `false` here reproduces 0115's EFFECTIVE alpha.
optimize_doc_concentration: false
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
  # Same geometry as 0113 (from 0110, the whole-Mondo survivor). Over-provisioned and
  # safe at this branch's K; the scalable spectral seed's sequential passes are the new
  # cost, not the fit.
  spark.executor.cores: 2
  spark.executor.memory: 8g
  spark.executor.memoryOverhead: 3g
  spark.dynamicAllocation.enabled: "false"
  spark.executor.instances: 20
  spark.excludeOnFailure.timeout: 10m
  spark.cleaner.periodicGC.interval: "5min"
---


# 0127 — cardiomyopathy subtree, guided anchors, forward deflation (probe control)

See the front matter. Launched together with 0128 (the one chain in 0128's Run section).

## What it does

`make -C analysis/cloud exp ID=127` → the bundle for the cardiomyopathy branch is built
(new cache key: ~20 min BQ), then a fit-only spectral-init gated LDA with guided anchors
on ~30 nodes; then the ridge-100 readout and the named digest + anchors.

## Acceptance criterion

A control: its DCM block is expected to carry the prenatal stratum exactly as 0125's
does (same mechanism, same documents). If it does NOT, the mechanism story is wrong
again and 0128's result cannot be attributed to the order.

## Run log

## Results
