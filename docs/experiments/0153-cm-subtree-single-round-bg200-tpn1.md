---
id: 153
slug: cm-subtree-single-round-bg200-tpn1
status: planned
model_class: gated_pc
cohort: population_mondo_all
cohort_def: population_mondo_all
disease: rare_priority
dag_source: mondo_native
mondo_branch: MONDO:0004994   # cardiomyopathy subtree (0130/0131): ~30 nodes
tpn: 1                      # as 0152; K = 200 + subtree nodes
max_iter: 50
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

# 0153 — cardiomyopathy subtree, single-round gate (the baseline for 0154)

## Why

0152 read 3b: on the nested whole-Mondo fit some parent blocks hold a child's
vocabulary (the "cardiomyopathy" block reads as hypertrophic cardiomyopathy; migraine
disorder reads as epilepsy) and some children lose their words to comorbidity. The
local Spark simulation (report 2026-10-19-leaves-first-gate-sim) reproduced both
failures and fixed them by fitting in rounds by node height with a seed when each block
opens. 0153/0154 are that test on real data, in a focused area: the cardiomyopathy
subtree, which holds the read-3 cases (HCM, DCM, restrictive, intrinsic, peripartum).

This run is the single-round baseline. 0154 is identical except `gate_warm: 5`.

## Config versus 0152

- `mondo_branch: MONDO:0004994`, about 30 nodes.
- `n_bg` 1200 → 200 in both arms, so a background document's CAVI is cheaper.
- Everything else as 0152: tpn 1, spectral init, gate-ever, bg anchors 8, 50 iterations.

## Run

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUN=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0153-cm-subtree-single-round-bg200-tpn1
mkdir -p "$RUN"
nohup make -C analysis/cloud exp ID=153 GPR_SPARK_CONF="spark.eventLog.enabled=false" > "$RUN"/sweep_log.md 2>&1 &
```

## Read (with 0154)

The topic digest on the subtree, same command for both runs:

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
for ID in 153 154; do make -s -C analysis/cloud inspect-topics ID=$ID RESOLVE_NAMES=1 INSPECT_ARGS="--digest --digest-exemplars 0 --top-words 10 --redundancy 30 --grep 'dilated cardiomyopathy|intrinsic cardiomyopathy|restrictive cardiomyopathy|^cardiomyopathy|hypertrophic cardiomyopathy|peripartum|arrhythmogenic|left ventricular noncompaction|takotsubo|myocarditis'" | cut -c1-260; done
```

What would count:
- **Parent stops claiming a child:** the "cardiomyopathy" block loses the HCM words.
  Caveat: HCM's own code is missing from the code map (40565500, R4.2), so its words have
  no attesting child to pull them down; watch DCM and intrinsic cardiomyopathy against
  their parent instead.
- **Children keep their own words:** a `†` word leads more child blocks.
- **Low-evidence children:** restrictive and non-familial restrictive (junk in 0152)
  either gain their words or stay junk. Low evidence is not what the schedule targets.

## Run log
