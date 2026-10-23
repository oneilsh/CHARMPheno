---
id: 161
slug: whole-mondo-leaves-first-balanced-bg200-tpn1
status: done
model_class: gated_pc
cohort: population_mondo_all
cohort_def: population_mondo_all
disease: rare_priority
dag_source: mondo_native
mondo_branch: ""
tpn: 1   # as 0152; K = 200 + ~2,700
max_iter: 60   # as the branch runs; 2 iters/level keeps about a third of the fit in rounds
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
n_bg: 200   # the central configuration (0156/0160); 0152 had 1200
spectral_bg_anchors: 8      # 0134's: anchor eight bg rows as deflation seeds, random-init the rest
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
---

# 0161 — whole Mondo, leaves-first + balanced seed, 200 background topics

## Why

The cardiovascular branch work (exps 0155–0160, insight 0104) found a configuration that
works: leaves-first gate schedule, child-balanced open-time seed, 200 background topics.
On the branch it took own-code identity 50% → 77%, raised parent drug coherence, and gave
a small new-diagnosis gain. This run asks whether that transfers to the whole Mondo DAG,
against 0152 (whole Mondo, single round, 1,200 background topics, same corpus and bundle).
It is the "central configuration" from which single parameters can later be varied.

Three things differ from 0152 at once (schedule, seed, background width), so this
answers "does the central configuration transfer", not which change is responsible:
the branch pairs already attributed those.

`gate_warm: 2`, not the branch's 3: whole Mondo is deeper, and 2 per level keeps about a
third of the 60 iterations in the leaf-first rounds, as on the branch. The fit log reports
the real height.

## Evaluations (all against 0152)

1. **Own-code census** by depth.
2. **Gated NPMI**, conditions and drugs (exp 0159's measure).
3. **New-diagnosis AUC** on the DisMech ribbon nodes (0152's readout scope), stacked and flat.
4. **Framings 1 and 2** on the 9 same-question groups of read 4b (`--diff-members top`).

## Run (one background job)

Order: fit, sidecar, ribbon readout (with the new-diagnosis check), NPMI on 0152 and
0161, the census on both, the 9-group survey, then every comparison. The fit and NPMI
share one bundle (same corpus as 0152), rebuilt once on a fresh cluster.

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUNS=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs
RIBBON=$HOME/repos/CHARMPheno/analysis/cloud/anchor_selection_data/dismech_ribbon.tsv
R152=$(ls -d $RUNS/0152-*); ls -la $RIBBON $R152/groups_4b.json
mkdir -p $RUNS/0161-whole-mondo-leaves-first-balanced-bg200-tpn1
L=$RUNS/0161-whole-mondo-leaves-first-balanced-bg200-tpn1/job_log.md
nohup bash -c '
set -o pipefail
RIBBON='$RIBBON'; R152='$R152'
echo "### fit 161"
make -C analysis/cloud exp ID=161 GPR_SPARK_CONF=spark.eventLog.enabled=false || { echo "FAILED: fit 161"; exit 1; }
R161=$(ls -d '$RUNS'/0161-*)
echo "### sidecar + ribbon readout 161"
make -C analysis/cloud build-conversion-sidecar ID=161 || { echo "FAILED: sidecar 161"; exit 1; }
make -C analysis/cloud gated-pc-readout ID=161 GPR_SPARK_CONF=spark.eventLog.enabled=false GPR_ARGS="--readout-mode distributed --readout-l2 100 --readout-theta-mass 0.99 --readout-stacked --readout-hierarchy mondo --readout-nodes $RIBBON --prev-history" || { echo "FAILED: readout 161"; exit 1; }
for ID in 152 161; do
  echo "### npmi $ID"
  make -C analysis/cloud gated-pc-readout ID=$ID GPR_SPARK_CONF=spark.eventLog.enabled=false GPR_ARGS="--npmi --npmi-top 10 --npmi-domains 0,2 --acq-tag top10" || { echo "FAILED: npmi $ID"; exit 1; }
done
echo "### survey (9 same-question groups) 161"
make -C analysis/cloud gated-pc-readout ID=161 GPR_SPARK_CONF=spark.eventLog.enabled=false GPR_ARGS="--differential . --diff-survey 25 --diff-members top --diff-incident --diff-groups-like $R152/groups_4b.json --acq-heads stacked_nodes_mass99 --diff-max-patients 1500 --diff-crossfit --diff-next-question --diff-nq-hide 0.8 --acq-tag survey_inc_top" || { echo "FAILED: survey 161"; exit 1; }
for ID in 152 161; do R=$(ls -d '$RUNS'/0$ID-*)
  echo "### census $ID"
  make -s -C analysis/cloud inspect-topics ID=$ID RESOLVE_NAMES=1 INSPECT_ARGS="--profile-census --own-codes --own-top 10 --out $R/census_own10.md" > /dev/null
  sed -n "/^block topics/p;/^own-code-decidable/p;/^depth · nodes/,/^\$/p" "$R"/census_own10.md
done
echo "### new diagnoses 0152 vs 0161 (ribbon nodes)"
python3 analysis/cloud/compare_incident_runs.py "$R152" prevalence_check_stacked_nodes_mass99.json "$R161" prevalence_check_stacked_nodes_mass99.json | cut -c1-220
echo "### npmi 0152 vs 0161"
cat "$R161"/npmi_top10.md
python3 analysis/cloud/topic_npmi.py compare "$R152" "$R161" top10
echo "### framings 0161 vs 0152 (same-question groups)"
python3 analysis/cloud/survey_analysis.py "$R161" survey_inc_top "$R152" survey_inc_top | cut -c1-220
echo "### DONE"
' > "$L" 2>&1 &
tail -f "$L"
```

In the fit log, check the `[pc] leaves-first gate` line (heights and the iteration the full
gate starts) and `[driver]   balanced seed: own codes ... for N node(s)`.

## Run log

**2026-10-23 — done (about 24 h: ~6.5 h spectral seed, ~1 h fit, the rest the ribbon
readout and the evaluations).** Whole Mondo is 15 levels high (heights 0..15); the full
gate opened at iteration 30 of 60. Against 0152 (single round, subtree-less init, 1,200
background topics). Three changes at once, so this reads as "does the central
configuration transfer", not as an attribution.

- **Own-code census:** signature blocks 75% → 88%; own-code identity 777 → 1,123 of
  1,442 decidable nodes (54% → 78%). Stratum share falls at every depth from 2 to 13:
  depths 3–7 53 → 41%, 40 → 14%, 32 → 14%, 26 → 12%, 22 → 10%. Starved blocks 3 → 15
  (1%).
- **New diagnoses, 827 DisMech ribbon nodes:** stacked 0.693 → 0.712, median +0.016,
  0161 better on 707 of 827; largest for the smallest nodes (+0.028 at 20–50 new
  cases). Flat 0.597 → 0.606, +0.009, 575 of 827. The largest new-diagnosis gain in
  this line of work.
- **NPMI:** conditions gated unchanged (+0.0002, 1,218 of 2,420); all-documents +0.012.
  Drugs gated +0.019 (1,485 of 2,453). The branch pattern.
- **Framings, 9 same-question groups:** unchanged within noise (median kappa change
  0.000; ≥ 0.3 in 4 against 5). Sjögren's (exocrine) fell 0.61 → 0.18 including its
  codes-only engine, which does not use θ, so the two survey runs scored different
  patients; likely the 1,500-patient sampling over a rebuilt bundle. The framing
  harness is noisy at these group sizes.
- **Topics up and down the DAG** (16 named nodes, 0152 → 0161):
  - *Parent-holds-child leaks fixed:* diabetes mellitus (T2DM words → its own
    "diabetes mellitus without complication"), heart failure (congestive heart failure
    first → heart failure†, hypertensive HF†), cardiomyopathy (HCM / HOCM words →
    cardiomyopathy†, primary cardiomyopathy†), inflammatory bowel disease (Crohn's
    first → IBD†, ulcerative colitis), breast carcinoma (the child's malignant-neoplasm
    codes → carcinoma of breast† with its presentation: palpable mass, nipple
    discharge).
  - *Junk fixed:* epilepsy (altered mental status, drug abuse → epilepsy†, seizure,
    status epilepticus†, levetiracetam); chronic kidney disease (osteoarthritis and
    pain → CKD stages 2 / 3†, CKD due to T2DM†).
  - *Unchanged and good:* asthma, type 1 diabetes, dilated cardiomyopathy, Crohn's,
    focal and generalized epilepsy, breast carcinoma in situ.
  - *Still generic:* human disease (now the inpatient-symptom stratum), nervous system
    disorder and cardiovascular disorder (comorbidity strata in both runs).
  - *HCM:* breast findings at evidence 108 → primary cardiomyopathy, arrhythmia and
    lipids at evidence 993; still no own code (the 40565500 code-map gap).
  - Redundancy: 5 collapsed parents against 3 (worst biliary tract neoplasm, 0.98).
- **Read:** the central configuration transfers to whole Mondo. Legibility gains match
  the branch's, the parent-holds-child failures from reads 3 / 3b are largely gone, and
  new-diagnosis AUC rises more than it did on the branch.
