---
id: 157
slug: cv-branch-leaves-first-seeded-bg1200-tpn1
status: planned
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
---

# 0157 — cardiovascular branch, leaves-first, histogram seed, 1,200 background topics

## Why

0156 (leaves-first with a seeded opening, 200 background topics) raised own-code identity
50% → 73% against the single round (insight 0104). A few leaves kept the
inpatient-encounter stratum, which had only leaves and the background to go to during the
leaf rounds. This run gives the background 0152's width, 1,200, with everything else as
0156. 0158 adds the HPO profile seed on top of this one.

## Run (both fits, both readouts, the comparisons, the census: one background job)

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUNS=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs
ls -la /home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0152-whole-mondo-hierarchical-tpn1-bg1200-v6codemap/profile_marks.tsv
mkdir -p $RUNS/0157-cv-branch-leaves-first-seeded-bg1200-tpn1 $RUNS/0158-cv-branch-leaves-first-hpo-seeded-bg1200-tpn1
L=$RUNS/0158-cv-branch-leaves-first-hpo-seeded-bg1200-tpn1/pair_log.md
nohup bash -c '
set -o pipefail
for ID in 157 158; do
  echo "### fit $ID"
  make -C analysis/cloud exp ID=$ID GPR_SPARK_CONF=spark.eventLog.enabled=false || { echo "FAILED: fit $ID"; exit 1; }
done
for ID in 157 158; do
  echo "### sidecar + readout $ID"
  make -C analysis/cloud build-conversion-sidecar ID=$ID || { echo "FAILED: sidecar $ID"; exit 1; }
  make -C analysis/cloud gated-pc-readout ID=$ID GPR_SPARK_CONF=spark.eventLog.enabled=false GPR_ARGS="--readout-mode distributed --readout-l2 100 --readout-theta-mass 0.99 --readout-stacked --readout-hierarchy mondo --prev-history" || { echo "FAILED: readout $ID"; exit 1; }
done
R156=$(ls -d '$RUNS'/0156-*); R157=$(ls -d '$RUNS'/0157-*); R158=$(ls -d '$RUNS'/0158-*)
echo "### compare 0156 vs 0157 (background width)"
python3 analysis/cloud/compare_incident_runs.py "$R156" prevalence_check_stacked_mass99.json "$R157" prevalence_check_stacked_mass99.json | cut -c1-220
echo "### compare 0157 vs 0158 (HPO profile seed)"
python3 analysis/cloud/compare_incident_runs.py "$R157" prevalence_check_stacked_mass99.json "$R158" prevalence_check_stacked_mass99.json | cut -c1-220
for ID in 157 158; do R=$(ls -d '$RUNS'/0$ID-*)
  echo "### census $ID"
  make -s -C analysis/cloud inspect-topics ID=$ID RESOLVE_NAMES=1 INSPECT_ARGS="--profile-census --own-codes --own-top 10 --out $R/census_own10.md" > /dev/null
  sed -n "/^block topics/p;/^own-code-decidable/p;/^depth · nodes/,/^\$/p" "$R"/census_own10.md
done
echo "### DONE"
' > "$L" 2>&1 &
tail -f "$L"
```

In 0158's log, check `[driver]   profile seed: {...}`: how many branch nodes have an HPO profile in the
vocabulary and got a seed. The `[pc] leaves-first gate` line adds the leaf and later topics
seeded from the profile.

## Read

- **0157 against 0156: background width only.** The census stratum share at the leaves
  (depth 5–8), and the digest on the 0156 leaf sinks (aortic valve stenosis, abdominal
  aortic aneurysm, benign essential hypertension, pregnancy-induced hypertension).
- **0158 against 0157: HPO profile seed only.** Census identity and stratum share, the
  new-diagnosis comparison, and whether profiled blocks keep HPO words (the digest's `*`
  marks) or drift to diagnosis codes. HPO phenotypes map mostly to symptom concepts, so a
  profiled leaf may start symptom-led.

## Run log
