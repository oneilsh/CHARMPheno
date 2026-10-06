---
id: 131
slug: cardiomyopathy-subtree-frontier-scope
status: pending
model_class: gated_pc
cohort: population_mondo_all
cohort_def: population_mondo_all
disease: rare_priority
# FRONTIER ANCHOR SCOPE. 0127's config (subtree, forward, guided) with ONE line changed:
# anchor_scope: frontier. WHY (0127-0130): a node's block is a five-way split of its seed
# documents' strata; anchors only label them, deflation only removes what anchors span,
# and the frequency floor only relabels. DCM carries the pregnancy stratum because
# peripartum patients are in its CLOSURE seed documents. Frontier scope trains each
# node's seed only on documents where it is the most specific attested node, so DCM's
# seed never sees a peripartum patient. (The fit itself is unchanged: the gate still
# lets closure docs use the block.) READ: DCM's five topics; parent legibility; anchors;
# paired AB vs 0127. Bundle HIT; ~25 min.
dag_source: mondo_native
mondo_branch: MONDO:0004994   # cardiomyopathy SUBTREE (was MONDO:0004995, the whole CV branch): ~30 nodes, minutes not hours
tpn: 5
max_iter: 50
diag_only: true
# --- the ONLY change from 0113: spectral block-aligned seed for the gated engine ---
init: spectral
spectral_method: scalable   # concatenated V ~11.6k >= 8000 threshold; dense = driver wall
spectral_d: 768             # random-projection dim: smaller = faster + bigger safe batch (see COST)
anchor_scope: frontier      # THE ONLY CHANGE vs 0127: a node's seed docs = those where it is the MOST SPECIFIC attested node (peripartum patients no longer train DCM's block)
spectral_topo_order: forward  # ancestors-first: each node's seed = its increment over ancestors
count_transform: none       # raw per-visit counts, as 0123 (binary was a detection tax, 0123)
# --- THE ONLY CHANGE vs 0123: HPO-guided spectral anchors (spec 2026-10-05) ---
spectral_anchor_profile: ~/repos/CHARMPheno/data/ontology/profile_eta_MONDO_0004995.tsv  # REGENERATED with the prenatal exclusion (see COST)
# ------------------------------------------------------------------------------------
preindex_closure: false
readout_mode: distributed
readout_theta_topm: 256
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
n_bg: 8
# LANDMINE (2026-09-11): 0113-0120 all say `optimize_doc_concentration: true` but the flag
# was INERT on the PC path until 0121 wired it (pc.py `set_alpha_policy`); their alpha was
# FIXED at 0.5. Copying that line now turns the empirical-Bayes alpha ON, and learned alpha
# collapses to the floor and re-starves depth (insight 0091; 0121: 79% starved). The first
# 0123 launch did exactly that (alpha mean 0.5 -> 0.21 by iter 19, background block 0.28
# and falling ~2.5%/iter) and was killed. `false` here reproduces 0115's EFFECTIVE alpha.
optimize_doc_concentration: false
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
  # Same geometry as 0113 (from 0110, the whole-Mondo survivor). Over-provisioned and
  # safe at this branch's K; the scalable spectral seed's sequential passes are the new
  # cost, not the fit.
  spark.executor.cores: 2
  spark.executor.memory: 8g
  spark.executor.memoryOverhead: 3g
  spark.dynamicAllocation.enabled: "false"
  spark.executor.instances: 20
  spark.excludeOnFailure.timeout: 10m
  spark.cleaner.periodicGC.interval: "5min"
---


# 0131 — subtree, forward, guided, `anchor_scope: frontier`

## Run

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/gated-conditional-voi && git checkout claude/gated-conditional-voi && git pull --ff-only
RUNS=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs
mkdir -p "$RUNS"/0131-cardiomyopathy-subtree-frontier-scope
nohup bash -c '
  RUN='"$RUNS"'/0131-cardiomyopathy-subtree-frontier-scope
  echo "=== 131 fit START $(date)"
  make -C analysis/cloud exp ID=131 || exit 1
  make -C analysis/cloud gated-pc-readout ID=131 GPR_ARGS="--readout-mode distributed --readout-l2 100"
  make -C analysis/cloud inspect-topics ID=131 COMPARE=127 INSPECT_ARGS="--readout-auc"
  make -C analysis/cloud inspect-topics ID=131 RESOLVE_NAMES=1 INSPECT_ARGS="--digest --redundancy 30 --grep \"cardiomyopathy|myocarditis|pregnan|gestation|heart failure\""
  python3 analysis/cloud/name_spectral_anchors.py "$RUN" --bundle-meta /tmp/inspect_meta_131.json --concept-names /tmp/concept_names_131.csv
  echo "=== 131 DONE $(date)"
' > "$RUNS"/0131-cardiomyopathy-subtree-frontier-scope/sweep_log.md 2>&1 &
```

Compact read:

```bash
L=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0131-cardiomyopathy-subtree-frontier-scope/sweep_log.md
grep -nE "^=== |spectral anchor guide|gated_pc \(pc_topics_lr\): (macro|detection)|^all: n=|^by depth|starved|zero training docs" "$L" | cut -c1-160
grep -nE "^  (dilated|intrinsic|peripartum)? ?cardiomyopathy +d[0-9]|^(dilated |intrinsic |peripartum )?cardiomyopathy \|" "$L" | cut -c1-230 | awk '!seen[$0]++'
```

## Run log

## Results
