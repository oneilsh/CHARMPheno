---
id: 152
slug: whole-mondo-hierarchical-tpn1-bg1200-v6codemap
status: running
model_class: gated_pc
cohort: population_mondo_all
cohort_def: population_mondo_all
disease: rare_priority
# WHOLE-MONDO, HIERARCHICAL (DAG-gated) — 0110 redone with everything learned since.
# 0110 (2026-09) fit the whole native Mondo DAG (C = 2,714 nodes at min_positives 100;
# 1,855 evaluable) at tpn 1 with 8 background topics on native-mondo-v1, and read it
# with flat heads only: within-cohort macro AUC 0.735. Since then: the code map is v6
# (insight 0097: O90.3 / multi-map ancestor targets — the "same-vocabulary child gets a
# cohort" failure of insight 0094 was partly a code-map defect); a 1,200-topic shared
# background restores the root head (0134, insight 0095); the stacked closure product
# is the de-novo read; the gate can read all-ever attestation (0146); and evaluation is
# new-diagnosis-only on the whole history (0145) with a raw-code anchor (0147).
# CHANGES vs 0137 (the flat ribbon record): no label set (every Mondo term with closure
# support >= 100, branch ALL, nested induced-Hasse DAG, so the gate is hierarchical:
# a document may use the blocks of every ancestor of its attested terms); tpn 5 -> 1;
# gate_attestation ever. Everything else as 0137 (bg 1200, multi-domain, lookback 5 y,
# label window 1 y, spectral closure / forward init, alpha 0.5, mass-0.99 readout).
# NOT in this run: the [measured] filter and code-map additions (hashed rebuild, R4 of
# the label-redesign spec) — they would confound the hierarchy comparison.
# AUDIT FIRST (bundle build): the powering line (expect C ~2,700, K ~3,900), the
# multi-map guard, the attestation audit, and the [gate-ever] receipt.
# PRE-REGISTERED READS (all on the whole-history new-diagnosis rule, 0145):
# (1) on the ribbon nodes shared with 0137, stacked incident macro AUC vs 0137's 0.694
#     and the raw-code anchor 0.733 (0147) — does the hierarchical gate close the gap
#     to raw codes (> ~0.02 over 0.694 counts);
# (2) the same by depth and by node size (0147's table: raw wins most on the rarest);
# (3) block legibility at the same-vocabulary children of 0094 (DCM, intrinsic CM,
#     persistent AF, non-familial restrictive CM): do they get their own words now;
# (4) framings: the incident survey (0151's protocol) on this fit vs 0151.
# COST: bundle MISS (no label set; gate-ever); fit at K ~3,900 over the whole
# population (longer than 0137); readout C ~2,700 fit heads + ancestors — several
# times 0137's stacked solve.
dag_source: mondo_native
mondo_branch: ""
tpn: 1                      # 0095/0134: the legible nested choice; K = 1200 + ~2,700
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
n_bg: 1200                  # 0134's (insight 0095: width is the root head's lever)
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
---

# 0152 — whole-Mondo hierarchical gated fit, tpn 1, bg 1200, v6 code map

## Run

Step 1 — fit (bundle MISS, then the fit). Audit the bundle lines as soon as they print.

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUN=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0152-whole-mondo-hierarchical-tpn1-bg1200-v6codemap
mkdir -p "$RUN"
nohup make -C analysis/cloud exp ID=152 GPR_SPARK_CONF="spark.eventLog.enabled=false" > "$RUN"/sweep_log.md 2>&1 &
grep -E "powering:|multi-map guard|attestation audit|gate-ever|corpus: V=" "$RUN"/sweep_log.md | cut -c1-250
```

Step 2 — readout on the RIBBON nodes (and their readout ancestors) only. A full
readout of ~2,700+ heads at K = 3,894 is several times 0137's solve (a day-plus); the
pre-registered reads are on the ribbon nodes shared with 0137, so every other head is
masked to a constant (`--readout-nodes`, it drops out of the solve at iteration 1). The
sidecar is rebuilt for this run's code map first. A solve that dies with the cluster
resumes from its checkpoint when the same command is rerun.

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUN152=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0152-whole-mondo-hierarchical-tpn1-bg1200-v6codemap
RIBBON=$HOME/repos/CHARMPheno/analysis/cloud/anchor_selection_data/dismech_ribbon.tsv
nohup bash -c "
  make -C analysis/cloud build-conversion-sidecar ID=152 || exit 1
  make -C analysis/cloud gated-pc-readout ID=152 GPR_SPARK_CONF=spark.eventLog.enabled=false GPR_ARGS='--readout-mode distributed --readout-l2 100 --readout-theta-mass 0.99 --readout-stacked --readout-hierarchy mondo --readout-nodes $RIBBON --prev-history'
" > "$RUN152"/readout_sweep_log.md 2>&1 &
grep -E "node subset|fittable nodes|batched L-BFGS iter" "$RUN152"/readout_sweep_log.md | tail -3 | cut -c1-200
```

Step 3 — compare with 0137 (θ) and 0147 (raw codes), nodes matched by Mondo id:

```bash
R137=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0137-dismech-ribbon-record-tpn5-v2codemap
python3 analysis/cloud/compare_incident_runs.py "$R137" prevalence_check_history.json "$RUN152" prevalence_check_stacked_nodes_mass99.json prevalence_check_stacked_raw2000.json
```

## Run log

**2026-10-15 — launch 0 never ran** (the queued wait loop's `pgrep` pattern matched its
own command line; nothing submitted).

**2026-10-16 — launch 1, bundle audit.** Powering: 30,044 standard codes -> 8,883 Mondo
terms; 6,024 with closure support; **2,962 clear min_positives 100**. Induced Hasse
2,963 nodes (1,369 multi-parent); splice removed 253 thin-chain + 15 childless ->
**2,695 nodes** (1,269 multi-parent), so K = 1,200 + 2,695 = 3,895 at tpn 1, as planned.
Multi-map guard: 3 source codes, 0 ancestor targets dropped. Hierarchy test: 53
over-broad non-SNOMED targets dropped over 52 terms (Finding of head and neck region,
Primary malignant neoplasm, Disorder of GI tract, Enzymopathy, …); 12 suspect SNOMED
xrefs kept for review (eye degenerative disorder, hearing loss, HMSN, …). Attestation:
30,044 codes -> 2,651 label nodes; 7,352 codes attest >= 2 nodes (4,204 exact-shared,
3,148 climb-tie) across 1,640 nodes — far more sharing than the ribbon (220 / 4,871),
as expected on a multi-parent DAG where unpowered terms' codes climb to every nearest
kept ancestor. Most-attested nodes are broad groupings that collect unpowered rare
terms' codes (bone disorder 1,128, hereditary disease 974, skin disorder 950, syndromic
disease 923, arthropathy 817). The ancestor-group code gap of 0141 ("Cardiomyopathy"
321319) is closed by construction: the group term is a label node here.

## Results
