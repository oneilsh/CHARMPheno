---
id: 140
slug: dismech-ribbon-alpha0p02
status: planned
model_class: gated_pc
cohort: population_mondo_all
cohort_def: population_mondo_all
disease: rare_priority
# THE FIXED-α LADDER on the ribbon record (handoff 2026-10-12 next action 3): 0137 with
# ONE change, doc_concentration 0.5 -> 0.02 (held; optimize_doc_concentration stays false).
# Sibling: 0139 (α 0.1). Bundle HIT on the cluster that read 0137/0138 (α is
# not in the cache key).
# WHY AGAIN, given insight 0091 ("do not spend another fit on α"): 0091 measured LEARNED
# and HELD-ASYMMETRIC α on the nested CV DAG, with depth starvation as the outcome. The
# ribbon is a FLAT label space with spectral init, and the outcome here is different: θ
# at α 0.5 is diffuse (0137: top-m mass truncation resolved to nearly full K — the prior
# floor dominates), because every document carries 1,200 background topics plus its own
# blocks, i.e. ≥ 600 pseudo-counts of uniform prior competing with the record's own
# tokens. At α 0.02 that prior is 24 pseudo-counts. Question: does a θ that
# can be spiky ("are patients spiky?", the user's open question) read out better, read
# out the same, or re-starve the label blocks as 0121 did?
# PRE-REGISTERED READS (vs 0137): (1) θ concentration — the theta-mass top-m line (0137:
# ~full K); (2) within-cohort and stacked de-novo macro AUC/AP and the root head (0137:
# 0.797 / 0.822 / 0.836), paired by node; (3) starvation and the own-code census
# (decidable n_profiles median 2, p90 4; starved 4%) — a re-starve is the 0091 failure;
# (4) part-2 conditional AUC by depth (0138: 0.77–0.81, flat).
# The gated driver's 1/K warning (Jacobian underflow kills PC shaping) does not bite:
# weight_y is 0 on the ribbon record, there is no shaping to kill.
# COST: as 0137 (fit ~1.5 h, readouts ~1.5–3 h each); the two fits run sequentially.
dag_source: mondo_native
mondo_branch: ""
label_set: analysis/cloud/anchor_selection_data/dismech_ribbon.tsv
tpn: 5                      # tpn_max from 0136 (ceiling binding for ~1/3 of decidable nodes)
max_iter: 50
diag_only: true
init: spectral
spectral_method: scalable
spectral_d: 768             # 0134's: only 8 bg anchors + 3 per node are anchored
anchor_scope: closure       # on a flat forest, closure(node) = {node, root}: own block
spectral_topo_order: forward
count_transform: none
preindex_closure: false
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
doc_concentration: 0.02              # 0137: 0.5 — THE one change
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

# 0140 — fixed α 0.02 on the ribbon record (the α ladder vs 0137)

## Run

Both arms of the ladder in one sequential job (fit, flat readout, stacked readout, then
the census bundle meta), launched from the 0139 doc; the same script serves both.

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUNS=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs
mkdir -p "$RUNS"/0139-dismech-ribbon-alpha0p1 "$RUNS"/0140-dismech-ribbon-alpha0p02
cat > "$RUNS"/0139-dismech-ribbon-alpha0p1/ladder.sh <<'EOF'
for ID in 139 140; do
  echo "=== $ID fit START $(date)"
  make -C analysis/cloud exp ID=$ID || continue
  echo "=== $ID readout START $(date)"
  make -C analysis/cloud gated-pc-readout ID=$ID GPR_ARGS="--readout-mode distributed --readout-l2 100 --readout-theta-mass 0.99"
  echo "=== $ID stacked START $(date)"
  make -C analysis/cloud gated-pc-readout ID=$ID GPR_ARGS="--readout-mode distributed --readout-l2 100 --readout-theta-mass 0.99 --readout-stacked"
  echo "=== $ID conditional START $(date)"
  make -C analysis/cloud gated-pc-readout ID=$ID GPR_ARGS="--conditional-from-heads stacked_mass99"
  echo "=== $ID DONE $(date)"
done
EOF
nohup bash "$RUNS"/0139-dismech-ribbon-alpha0p1/ladder.sh > "$RUNS"/0139-dismech-ribbon-alpha0p1/ladder_log.md 2>&1 &
```

Progress: `grep "===" "$RUNS"/0139-dismech-ribbon-alpha0p1/ladder_log.md`. The α line and
the θ-mass line, as soon as each readout prints them:

```bash
grep -hE "theta.*mass|top-m|alpha" "$RUNS"/01{39,40}-*/readout_log.md | cut -c1-170
```

Reads after both finish (paired against 0137 by node; census on each):

```bash
make -C analysis/cloud inspect-topics ID=140 COMPARE=137 INSPECT_ARGS="--readout-auc --results-file results_readout_stacked_mass99.json --compare-results-file results_readout_stacked_mass99.json" | cut -c1-170 | head -60
make -C analysis/cloud inspect-topics ID=140 RESOLVE_NAMES=1 INSPECT_ARGS="--profile-census --own-codes --relevance-lambda 0.6 --own-top 5 --top-words 4 --digest-exemplars 0" | grep -vE "^\[inspect-topics\]|^make" | cut -c1-170 | head -40
```

## Run log

## Results
