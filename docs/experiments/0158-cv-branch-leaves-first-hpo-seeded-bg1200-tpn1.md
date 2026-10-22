---
id: 158
slug: cv-branch-leaves-first-hpo-seeded-bg1200-tpn1
status: done
model_class: gated_pc
cohort: population_mondo_all
cohort_def: population_mondo_all
disease: rare_priority
dag_source: mondo_native
mondo_branch: MONDO:0004995   # cardiovascular branch (0132-0134): ~300 nodes
tpn: 1                      # as 0152; K = 1200 + 294
max_iter: 60   # 3 iters/level x height leaves >= 30 full-gate iters if height <= 10
diag_only: true
init: spectral
spectral_method: scalable
spectral_d: 768             # 0134's: only 8 bg anchors + 3 per node are anchored
anchor_scope: closure       # on a flat forest, closure(node) = {node, root}: own block
spectral_topo_order: forward
count_transform: none
preindex_closure: false
gate_attestation: ever      # 0146: neutral on AUC, the cleaner gate
readout_mode: distributed
readout_theta_mass: 0.99    # excess-over-prior-floor coverage (0135 launch 3 fix)
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
n_bg: 1200   # 0156 had 200; the leaf-round sink needs a background home (0152's width)
spectral_bg_anchors: 8
optimize_doc_concentration: false   # insight 0091; see 0134's LANDMINE note
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
  # 0134's geometry. K is ~4x 0134's; if executors OOM in the E-step, raise
  # spark.executor.memory before touching min_positives.
  spark.executor.cores: 2
  spark.executor.memory: 8g
  spark.executor.memoryOverhead: 3g
  spark.dynamicAllocation.enabled: "false"
  spark.executor.instances: 20
  spark.excludeOnFailure.timeout: 10m
  spark.cleaner.periodicGC.interval: "5min"
gate_warm: 3   # leaves-first: iters per height level, then the full gate
gate_open_seed: true   # seed each block when it opens (subtree word counts)
gate_seed_profile: /home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0152-whole-mondo-hierarchical-tpn1-bg1200-v6codemap/profile_marks.tsv   # HPO profile seed where a node has one; histogram elsewhere
---

# 0158 — cardiovascular branch, leaves-first, HPO profile seed (histogram fallback), 1,200 bg

## Why

The leaves-first schedule's second seed variant: a node with an HPO phenotype profile in
the vocabulary starts from it. Its condition-domain seed is the profile concepts weighted
by an IDF over this run's profiled nodes; leaves at iteration 0, later blocks at opening.
Unprofiled nodes keep the histogram seed; measurement and drug domains keep theirs.
Profile table: `hpo_profile_marks.py` over the whole-Mondo HPO codes, in 0152's run dir.
Everything else as 0157.

The run, read and comparison commands are in 0157 (one background job runs both).

## Run log

**2026-10-22 — done; 0157 → 0158 (HPO profile seed, at 1,200 background).**
- **Profile seed:** 35 of 294 branch nodes have an HPO profile in the vocabulary, all
  seeded (29 leaves, 6 later blocks); median 48 profile concepts per node; 5,553 profile
  rows had no concept in the vocabulary.
- **New-diagnosis:** stacked 0.749 → 0.754, median +0.003, 147 of 237 better (sign test
  p ≈ 1e-4). Flat +0.0025, 132 of 237 (p ≈ 0.05). The gain sits at 20–300 positives.
- **Census:** slightly worse. Signature 83% → 81%; own-code identity 125 → 120 of 172.
  Stratum share up at depths 4–7: 14 → 17, 13 → 17, 16 → 23, 18 → 21%. The root becomes a
  signature (n = 1).
- **Read:** the HPO start buys a small prediction gain and costs a little legibility.
  Profiled leaves start symptom-led and do not all move to their diagnosis codes. Only
  about 12% of nodes are profiled, so the effect is small in either direction. Measured
  at 1,200 background topics, whose baseline is itself 0.03 below 0156.
