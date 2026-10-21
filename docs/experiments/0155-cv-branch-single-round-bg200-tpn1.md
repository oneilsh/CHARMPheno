---
id: 155
slug: cv-branch-single-round-bg200-tpn1
status: done
model_class: gated_pc
cohort: population_mondo_all
cohort_def: population_mondo_all
disease: rare_priority
dag_source: mondo_native
mondo_branch: MONDO:0004995   # cardiovascular branch (0132-0134): ~300 nodes
tpn: 1                      # as 0152; K = 200 + subtree nodes
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
n_bg: 200   # 1200 would keep a background doc's CAVI at 0152's cost; same in both arms
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
---

# 0155 — cardiovascular branch, single-round gate (the baseline for 0156)

## Why

0153/0154 on the cardiomyopathy subtree (17 nodes): leaves-first with a seeded opening
gave the subtree root its own stratum back and rescued myocarditis, and moved the
parent-holds-child leak one level down (intrinsic cardiomyopathy took HCM words). On the
stacked new-diagnosis read it gained +0.016 (12 of 14 nodes); flat was unchanged. The
cardiovascular branch (about 300 nodes, deeper) tests how the schedule behaves further
up the tree, where most of the interpretability gain was at parent level.

0156 is identical except `gate_warm: 3`. Both arms use 60 iterations: the deeper branch
needs more schedule rounds.

## Run (both fits, then both readouts, then the comparison: one background job)

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUNS=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs
mkdir -p $RUNS/0155-cv-branch-single-round-bg200-tpn1 $RUNS/0156-cv-branch-leaves-first-seeded-bg200-tpn1
L=$RUNS/0156-cv-branch-leaves-first-seeded-bg200-tpn1/pair_log.md
nohup bash -c '
set -o pipefail
for ID in 155 156; do
  echo "### fit $ID"
  make -C analysis/cloud exp ID=$ID GPR_SPARK_CONF=spark.eventLog.enabled=false || { echo "FAILED: fit $ID"; exit 1; }
done
for ID in 155 156; do
  echo "### sidecar + readout $ID"
  make -C analysis/cloud build-conversion-sidecar ID=$ID || { echo "FAILED: sidecar $ID"; exit 1; }
  make -C analysis/cloud gated-pc-readout ID=$ID GPR_SPARK_CONF=spark.eventLog.enabled=false GPR_ARGS="--readout-mode distributed --readout-l2 100 --readout-theta-mass 0.99 --readout-stacked --readout-hierarchy mondo --prev-history" || { echo "FAILED: readout $ID"; exit 1; }
done
R155=$(ls -d '$RUNS'/0155-*); R156=$(ls -d '$RUNS'/0156-*)
echo "### compare 0155 vs 0156"
python3 analysis/cloud/compare_incident_runs.py "$R155" prevalence_check_stacked_mass99.json "$R156" prevalence_check_stacked_mass99.json | cut -c1-220
echo "### DONE"
' > "$L" 2>&1 &
tail -f "$L"
```

Watch for the `[pc] leaves-first gate` line in 0156's fit: it prints the branch height and the
iteration where the full gate starts (it must be well before 60).

## Read

1. **Legibility, both fits.** The parents are the point, so the tree tour as well as the grep:

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
for ID in 155 156; do make -s -C analysis/cloud inspect-topics ID=$ID RESOLVE_NAMES=1 INSPECT_ARGS="--digest --digest-exemplars 0 --top-words 10 --redundancy 30 --grep 'cardiomyopathy|heart failure|atrial fibrillation|valve|coronary|myocardial infarction|hypertensi|aneurysm|cardiac arrhythmia|heart disease'" | cut -c1-260; done
for ID in 155 156; do make -s -C analysis/cloud inspect-topics ID=$ID RESOLVE_NAMES=1 INSPECT_ARGS="--tour 2 --top-topics 0 --top-words 8 --out /dev/stdout" | sed -n '/Tree tour/,$p' | cut -c1-200; done
```

2. **New-diagnosis AUC:** the comparison at the end of the pair log, 0156 against 0155.

## Run log
