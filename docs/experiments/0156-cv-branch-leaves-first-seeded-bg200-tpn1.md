---
id: 156
slug: cv-branch-leaves-first-seeded-bg200-tpn1
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
gate_warm: 3   # leaves-first: iters per height level, then the full gate
gate_open_seed: true   # seed each block when it opens (subtree word counts)
---

# 0156 — cardiovascular branch, leaves-first gate with seeded opening

## Why

See 0155. One change: `gate_warm: 3` (3 iterations per height level, then the full
gate), with the subtree-mean seed at opening (ADR 0049). Known weakness from 0154: the
seed weights children by size, so a parent with one dominant child opens close to that
child; the child-balanced, own-code-excluded seed is the candidate fix if the leak
recurs here.

The run, read and comparison commands are in 0155 (one background job runs both).

## Run log

**2026-10-21 — pair done; new-diagnosis comparison (read 2).** 0156 log: `[pc] leaves-first
gate: 3 iters/level over heights 0..7 (full gate from iter 21); open-time seed ON`, so 39
full-gate iterations followed the schedule. On the whole-history prior, 237 nodes shared:
- **Stacked:** 0.7805 → 0.7863, median +0.0067, 0156 better on 172 of 237 (one-sided sign
  test p ≈ 1e-12). The gain is flat across size bands, +0.004 to +0.008 from 20–50 to
  1000+ incident positives.
- **Flat:** 0.6644 → 0.6640, median −0.0006, 113 of 237 better. No change.
- **Read:** this is the 0153/0154 pattern at about 14× the nodes (0154: +0.016, 12 of 14
  stacked; flat unchanged). The schedule gives a small, robust gain to the stacked
  product and nothing to the per-node heads. Topic legibility (read 1) still to come.

**2026-10-21 — digest pair (read 1), top 25 matched nodes by evidence.**
- **Wins (0155 → 0156):**
  - atrial fibrillation: knee osteoarthritis and actinic keratosis → AF†, chronic AF†,
    paroxysmal AF.
  - congestive heart failure: distress, rales, thyromegaly → CHF†, systolic HF.
  - coronary artery disorder (d4): generic inpatient symptoms → coronary
    arteriosclerosis†, atherosclerosis, grafts.
  - essential hypertension: T2DM words → essential hypertension†.
  - heart failure (d3) no longer leads with its child CHF's words; it now leads with
    heart failure†, left heart failure† and hypertensive heart failure†.
  - hypertensive disorder (d4) stops leading with its child's code (essential
    hypertension) and reads as an obesity / sleep-apnea stratum.
- **Losses:**
  - abdominal aortic aneurysm: AAA† first → generic inpatient symptoms.
  - benign essential hypertension and rheumatic mitral disease: own code pushed down
    by symptoms.
  - aortic valve stenosis and pregnancy-induced hypertension carry the generic
    inpatient stratum in 0156 (pain, nausea, vomiting, shortness of breath; sodium
    chloride, fentanyl): aortic valve stenosis at evidence 1.8e5.
- **Same in both:** MI, persistent AF, intermediate coronary syndrome, ocular
  hypertension, hypertensive urgency, DCM and mitral valve disorder are legible.
  Diastolic HF and secondary hypertension read as T2DM. Coronary atherosclerosis reads
  as a COPD / smoking stratum.
- **Mechanism, from the depth medians:** evidence moves down the tree. 0155 → 0156 by
  depth: d1 3.7e5 → 4.0e4, d2 1.2e5 → 1.9e4, d3 1.6e4 → 5.1e3; d7 1.3e3 → 2.0e3, d8 163
  → 743. Parents end up thinner and own-code-led. During the leaf rounds, though, the
  documents' shared generic content (the inpatient-encounter stratum) has only leaves
  and the 200 background topics to go to, so some leaves keep it.
- **Next:** a quantitative legibility read (profile census with the own-code rule:
  signature against stratum share by depth) instead of eyeballing the top 25.

**2026-10-21 — profile census with the own-code rule (read 1b, all 294 nodes).**
Signature blocks 206 → 245 (70% → 83%). Own-code-decidable nodes with an own-code
identity 86/172 → 126/172 (50% → 73%). Stratum share by depth, 0155 → 0156: d2 75 → 25%,
d3 36 → 21%, d4 34 → 14%, d5 23 → 14%, d6 26 → 16%, d7 29 → 21%, d8 25 → 0% (the root is a
stratum in both). A clear legibility win at every depth; the leaf sinks in read 1 are a
minority. Insight 0104.
