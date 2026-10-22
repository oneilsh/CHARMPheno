---
id: 160
slug: cv-branch-leaves-first-balanced-seed-bg200-tpn1
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
gate_warm: 3   # leaves-first: iters per height level, then the full gate
gate_open_seed: true   # seed each block when it opens (subtree word counts)
gate_seed_mode: balanced   # each child's share equally + the parent's own docs; descendants' own codes removed
---

# 0160 — cardiovascular branch, leaves-first, child-balanced seed, 200 background topics

## Why

0156 (leaves-first, subtree-mean seed, bg 200) is the best configuration so far
(insight 0104). The subtree-mean seed weights children by size, so a parent with one
dominant child opens close to that child (0154: intrinsic cardiomyopathy took HCM's
words). The balanced seed gives each child's subtree distribution equal weight, adds the
documents coded at the parent itself as one more share, and removes every descendant's
own attesting codes, so a parent opens on what its children share. One change from
0156: `gate_seed_mode: balanced`.

## Run (with 0159's NPMI: one background job)

Order: NPMI on 0153–0158 (readout-only, no transform; a fresh cluster rebuilds the two
bundles, subtree and branch, about 35 min each), then the 0160 fit, readout and NPMI,
then every comparison and the 0160 census.

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUNS=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs
mkdir -p $RUNS/0160-cv-branch-leaves-first-balanced-seed-bg200-tpn1
L=$RUNS/0160-cv-branch-leaves-first-balanced-seed-bg200-tpn1/job_log.md
nohup bash -c '
set -o pipefail
for ID in 153 154 155 156 157 158; do
  echo "### npmi $ID"
  make -C analysis/cloud gated-pc-readout ID=$ID GPR_SPARK_CONF=spark.eventLog.enabled=false GPR_ARGS="--npmi --npmi-top 10 --npmi-domains 0,2 --acq-tag top10" || { echo "FAILED: npmi $ID"; exit 1; }
done
echo "### fit 160"
make -C analysis/cloud exp ID=160 GPR_SPARK_CONF=spark.eventLog.enabled=false || { echo "FAILED: fit 160"; exit 1; }
echo "### sidecar + readout 160"
make -C analysis/cloud build-conversion-sidecar ID=160 || { echo "FAILED: sidecar 160"; exit 1; }
make -C analysis/cloud gated-pc-readout ID=160 GPR_SPARK_CONF=spark.eventLog.enabled=false GPR_ARGS="--readout-mode distributed --readout-l2 100 --readout-theta-mass 0.99 --readout-stacked --readout-hierarchy mondo --prev-history" || { echo "FAILED: readout 160"; exit 1; }
echo "### npmi 160"
make -C analysis/cloud gated-pc-readout ID=160 GPR_SPARK_CONF=spark.eventLog.enabled=false GPR_ARGS="--npmi --npmi-top 10 --npmi-domains 0,2 --acq-tag top10" || { echo "FAILED: npmi 160"; exit 1; }
R() { ls -d '$RUNS'/0$1-*; }
for ID in 153 154 155 156 157 158 160; do echo "### npmi summary $ID"; cat "$(R $ID)"/npmi_top10.md; done
for P in "153 154" "155 156" "156 157" "157 158" "156 160"; do set -- $P
  echo "### npmi compare $1 vs $2"; python3 analysis/cloud/topic_npmi.py compare "$(R $1)" "$(R $2)" top10
done
echo "### compare 0156 vs 0160 (new diagnoses)"
python3 analysis/cloud/compare_incident_runs.py "$(R 156)" prevalence_check_stacked_mass99.json "$(R 160)" prevalence_check_stacked_mass99.json | cut -c1-220
echo "### census 160"
make -s -C analysis/cloud inspect-topics ID=160 RESOLVE_NAMES=1 INSPECT_ARGS="--profile-census --own-codes --own-top 10 --out $(R 160)/census_own10.md" > /dev/null
sed -n "/^block topics/p;/^own-code-decidable/p;/^depth · nodes/,/^\$/p" "$(R 160)"/census_own10.md
echo "### DONE"
' > "$L" 2>&1 &
tail -f "$L"
```

In the fit log, check `[driver]   balanced seed: own codes ... for N node(s)` and the
`[pc] leaves-first gate` line, which should say `open-time seed ON (balanced)`.

## Read

- **Against 0156** (the one change): own-code census by depth, new-diagnosis stacked and
  flat, gated NPMI by depth; the digest on intrinsic cardiomyopathy, heart failure,
  hypertensive disorder and the other parents with a dominant child.
- **0159:** NPMI for every pair listed there.

## Run log
