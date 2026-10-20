---
id: 154
slug: cm-subtree-leaves-first-seeded-bg200-tpn1
status: running
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
gate_warm: 5   # leaves-first: iters per height level, then the full gate
gate_open_seed: true   # seed each block when it opens (subtree word counts)
---

# 0154 — cardiomyopathy subtree, leaves-first gate with seeded opening

## Why

See 0153: the same config with one change, `gate_warm: 5`.
- **Rounds:** the first 5 iterations admit only leaf blocks (height 0), the next 5
  height ≤ 1, and so on, then the full gate. The driver logs the maximum height.
- **Seed:** each block, when it opens, is seeded with the word counts of the documents
  its gate admits (one pass before the fit, `gated_init.gate_open_seed_counts`), at a
  typical open topic's mass. Closed blocks are frozen.
- **Source:** report 2026-10-19-leaves-first-gate-sim. In simulation, with local Spark
  and the production minibatch regime, the plain schedule killed every parent block and
  the seeded one fixed both 0152 failures.

## Run (after 0153, same cluster, so the bundle is a HIT)

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUN=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0154-cm-subtree-leaves-first-seeded-bg200-tpn1
mkdir -p "$RUN"
nohup make -C analysis/cloud exp ID=154 GPR_SPARK_CONF="spark.eventLog.enabled=false" > "$RUN"/sweep_log.md 2>&1 &
grep -E "leaves-first gate|powering:|corpus: V=" "$RUN"/sweep_log.md | cut -c1-250
```

The `[pc] leaves-first gate` line must appear: it reports iterations per level, the
heights, the iteration where the full gate starts, and that the seed is ON.

## Read

The digest pair in 0153.

## Run log

**2026-10-20 — both fits done; digest pair (read 1).** K = 217 (200 bg + 17 nodes), 0%
starved in both. Per node, 0153 single round → 0154 leaves-first + seed:
- **Wins.**
  - cardiomyopathy (the subtree root): ischemic heart disease plus hypertrophic
    obstructive CM (a child's vocabulary), with no own code → Cardiomyopathy† and
    Primary cardiomyopathy† plus heart failure, its own stratum. Evidence 1.2e4 → 3.3e4.
  - myocarditis: fracture, osteoporosis, DVT (junk) → myocarditis†, pericarditis,
    pericardial effusion.
- **Losses.**
  - intrinsic cardiomyopathy: generic symptoms plus HF drugs → HOCM and HCM words. The
    parent-holds-child leak moved down from the root to this node.
  - restrictive CM: sarcoidosis, primary CM and VT (clinically plausible: cardiac
    sarcoid) → generic junk, evidence 1.5e3 → 297.
  - acute myocarditis (evidence about 150 in both): myocarditis first → junk.
- **Same in both:** DCM (own code, HF-led), HCM (clean in both, so the HCM fix over 0152
  comes from the narrower fit, not the schedule), peripartum (clean), non-familial
  restrictive (junk in both; in 0153 it absorbed asthma at evidence 7.6e3, in 0154 it sits
  near the floor at 151).
- **Read:** the targeted failure is fixed at the top node, and myocarditis is rescued.
  The leak reappears one level down, and two low-evidence nodes lose their words. One
  seed per arm. Readouts next.
