---
id: 162
slug: cv-branch-leaves-first-balanced-leafseed-random-init-bg200-tpn1
status: planned
model_class: gated_pc
cohort: population_mondo_all
cohort_def: population_mondo_all
disease: rare_priority
dag_source: mondo_native
mondo_branch: MONDO:0004995   # cardiovascular branch (0132-0134): ~300 nodes
tpn: 1                      # as 0152; K = 200 + subtree nodes
max_iter: 60   # 3 iters/level x height leaves >= 30 full-gate iters if height <= 10
diag_only: true
init: random   # every node block is data-seeded (leaves at iter 0, parents at opening): no spectral seed
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
gate_seed_mode: balanced   # each child's share equally + the parent's own docs; descendants' own codes removed
gate_seed_leaves: true   # leaf blocks start from their own documents' word counts
---

# 0162 — cardiovascular branch, 0160 with the spectral init replaced by a leaf seed

## Why

Under the leaves-first schedule every parent block is overwritten at opening by the
balanced seed, so the spectral init only ever reached the leaves. Its per-node batched
scans cost about 5–6 hours on whole Mondo (exp 0161: batches of 7 nodes at about 50 s
each over 2,694 nodes), and its anchors are rare outlier words the fit abandons (0152
read 3b). Here every leaf block starts at iteration 0 from the word counts of its own
documents (`gate_seed_leaves`), the same one pass that seeds the parents, and the
spectral init is off (`init: random`; the background starts random, as 1,192 of 0152's
1,200 background topics already did).

One parameter changed against 0160 (the central configuration): the leaves' starting
point, spectral → own-document counts. If it holds up here, whole-Mondo fits drop the
spectral init.

## Run (one background job)

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUNS=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs
mkdir -p $RUNS/0162-cv-branch-leaves-first-balanced-leafseed-random-init-bg200-tpn1
L=$RUNS/0162-cv-branch-leaves-first-balanced-leafseed-random-init-bg200-tpn1/job_log.md
nohup bash -c '
set -o pipefail
echo "### fit 162"
make -C analysis/cloud exp ID=162 GPR_SPARK_CONF=spark.eventLog.enabled=false || { echo "FAILED: fit 162"; exit 1; }
echo "### sidecar + readout 162"
make -C analysis/cloud build-conversion-sidecar ID=162 || { echo "FAILED: sidecar 162"; exit 1; }
make -C analysis/cloud gated-pc-readout ID=162 GPR_SPARK_CONF=spark.eventLog.enabled=false GPR_ARGS="--readout-mode distributed --readout-l2 100 --readout-theta-mass 0.99 --readout-stacked --readout-hierarchy mondo --prev-history" || { echo "FAILED: readout 162"; exit 1; }
echo "### npmi 162"
make -C analysis/cloud gated-pc-readout ID=162 GPR_SPARK_CONF=spark.eventLog.enabled=false GPR_ARGS="--npmi --npmi-top 10 --npmi-domains 0,2 --acq-tag top10" || { echo "FAILED: npmi 162"; exit 1; }
R() { ls -d '$RUNS'/0$1-*; }
echo "### census 162"
make -s -C analysis/cloud inspect-topics ID=162 RESOLVE_NAMES=1 INSPECT_ARGS="--profile-census --own-codes --own-top 10 --out $(R 162)/census_own10.md" > /dev/null
sed -n "/^block topics/p;/^own-code-decidable/p;/^depth · nodes/,/^\$/p" "$(R 162)"/census_own10.md
echo "### new diagnoses 0160 vs 0162"
python3 analysis/cloud/compare_incident_runs.py "$(R 160)" prevalence_check_stacked_mass99.json "$(R 162)" prevalence_check_stacked_mass99.json | cut -c1-220
echo "### npmi 0160 vs 0162"
python3 analysis/cloud/topic_npmi.py compare "$(R 160)" "$(R 162)" top10
echo "### DONE"
' > "$L" 2>&1 &
tail -f "$L"
```

In the fit log, the `[pc] leaves-first gate` line must end with `leaf blocks seeded at
iteration 0 from their own documents`, and no `scalable_block_aligned_lambda` lines
should appear.

## Read

Against 0160: own-code census by depth (the leaves, depths 5–8, are where the start
changed), new-diagnosis stacked and flat, gated NPMI. Fit wall time against 0160's too.

## Run log
