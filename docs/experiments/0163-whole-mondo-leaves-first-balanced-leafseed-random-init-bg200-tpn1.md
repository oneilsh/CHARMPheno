---
id: 163
slug: whole-mondo-leaves-first-balanced-leafseed-random-init-bg200-tpn1
status: planned
model_class: gated_pc
cohort: population_mondo_all
cohort_def: population_mondo_all
disease: rare_priority
dag_source: mondo_native
mondo_branch: ""
tpn: 1   # as 0152; K = 200 + ~2,700
max_iter: 60   # as the branch runs; 2 iters/level keeps about a third of the fit in rounds
diag_only: true
init: random   # exp 0162: every block data-seeded (leaves at iter 0, parents at opening)
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
n_bg: 200   # the central configuration (0156/0160); 0152 had 1200
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
gate_warm: 2   # leaves-first; whole Mondo is deeper than the CV branch (heights 0..7 there)
gate_seed_mode: balanced   # exp 0160
gate_seed_leaves: true   # exp 0162: leaf blocks start from their own documents' word counts
---

# 0163 — whole Mondo, 0161 with the spectral init replaced by the leaf seed

## Why

On the cardiovascular branch, exp 0162 replaced the spectral init with a leaf seed (each
leaf block starts from its own documents' word counts; random background) and raised
own-code identity 77% → 90%, with new-diagnosis AUC and NPMI unchanged. On whole Mondo
the spectral seeding was about 6.5 h of 0161's ~24 h. This is the single-parameter
variation of the central configuration (0161): the leaves' starting point, spectral →
own-document counts. Everything else, including the bundle and corpus, is 0161's.

## Evaluations (all against 0161)

1. **Own-code census** by depth (0161's `census_own10.md` exists).
2. **Gated NPMI**, conditions and drugs.
3. **New-diagnosis AUC** on the 827 DisMech ribbon nodes, stacked and flat.
4. **Framings** on the 9 same-question groups (`--diff-members top`).

## Run (one background job)

Ordered so the cheap reads land first: fit, census, NPMI, then the sidecar and the
ribbon readout (the long step), the new-diagnosis comparison, and last the survey.

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUNS=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs
RIBBON=$HOME/repos/CHARMPheno/analysis/cloud/anchor_selection_data/dismech_ribbon.tsv
R152=$(ls -d $RUNS/0152-*); R161=$(ls -d $RUNS/0161-*); ls -la $RIBBON $R152/groups_4b.json $R161/census_own10.md
mkdir -p $RUNS/0163-whole-mondo-leaves-first-balanced-leafseed-random-init-bg200-tpn1
L=$RUNS/0163-whole-mondo-leaves-first-balanced-leafseed-random-init-bg200-tpn1/job_log.md
nohup bash -c '
set -o pipefail
RIBBON='$RIBBON'; R152='$R152'; R161='$R161'
echo "### fit 163"
make -C analysis/cloud exp ID=163 GPR_SPARK_CONF=spark.eventLog.enabled=false || { echo "FAILED: fit 163"; exit 1; }
R163=$(ls -d '$RUNS'/0163-*)
echo "### census 163 (then 0161 for reference)"
make -s -C analysis/cloud inspect-topics ID=163 RESOLVE_NAMES=1 INSPECT_ARGS="--profile-census --own-codes --own-top 10 --out $R163/census_own10.md" > /dev/null
for R in "$R163" "$R161"; do echo "-- $(basename $R)"; sed -n "/^block topics/p;/^own-code-decidable/p;/^depth · nodes/,/^\$/p" "$R"/census_own10.md; done
echo "### npmi 163"
make -C analysis/cloud gated-pc-readout ID=163 GPR_SPARK_CONF=spark.eventLog.enabled=false GPR_ARGS="--npmi --npmi-top 10 --npmi-domains 0,2 --acq-tag top10" || { echo "FAILED: npmi 163"; exit 1; }
echo "### npmi 0161 vs 0163"
python3 analysis/cloud/topic_npmi.py compare "$R161" "$R163" top10
echo "### sidecar + ribbon readout 163"
make -C analysis/cloud build-conversion-sidecar ID=163 || { echo "FAILED: sidecar 163"; exit 1; }
make -C analysis/cloud gated-pc-readout ID=163 GPR_SPARK_CONF=spark.eventLog.enabled=false GPR_ARGS="--readout-mode distributed --readout-l2 100 --readout-theta-mass 0.99 --readout-stacked --readout-hierarchy mondo --readout-nodes $RIBBON --prev-history" || { echo "FAILED: readout 163"; exit 1; }
echo "### new diagnoses 0161 vs 0163 (ribbon nodes)"
python3 analysis/cloud/compare_incident_runs.py "$R161" prevalence_check_stacked_nodes_mass99.json "$R163" prevalence_check_stacked_nodes_mass99.json | cut -c1-220
echo "### survey (9 same-question groups) 163"
make -C analysis/cloud gated-pc-readout ID=163 GPR_SPARK_CONF=spark.eventLog.enabled=false GPR_ARGS="--differential . --diff-survey 25 --diff-members top --diff-incident --diff-groups-like $R152/groups_4b.json --acq-heads stacked_nodes_mass99 --diff-max-patients 1500 --diff-crossfit --diff-next-question --diff-nq-hide 0.8 --acq-tag survey_inc_top" || { echo "FAILED: survey 163"; exit 1; }
echo "### framings 0163 vs 0161 (same-question groups)"
python3 analysis/cloud/survey_analysis.py "$R163" survey_inc_top "$R161" survey_inc_top | cut -c1-220
echo "### DONE"
' > "$L" 2>&1 &
tail -f "$L"
```

In the fit log, the `[pc] leaves-first gate` line must end with `leaf blocks seeded at
iteration 0 from their own documents`, and no `scalable_block_aligned_lambda` lines
should appear. Record the fit's wall time against 0161's (~1 h fit after ~6.5 h seed).

## Run log
