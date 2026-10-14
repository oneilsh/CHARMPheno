---
id: 146
slug: gate-ever-on-0137-config
status: done
model_class: gated_pc
cohort: population_mondo_all
cohort_def: population_mondo_all
disease: rare_priority
# LABEL REDESIGN R2 (spec 2026-10-13): 0137's configuration with ONE change — the
# topic GATE of each TRAIN document is its person's ALL-EVER label attestations
# (full-history condition_era through the same code map and provider), not the label
# year. label / labelMask, the test split, K, alpha, tpn, seed, readout: as 0137.
# Bundle MISS by design (gate_attestation folds into the key); every other cache and
# every hashed module untouched.
# AUDIT FIRST: the "[gate-ever]" line at bundle build — gated (doc, node) cells and
# gated docs, label year -> ever, and the median per-node ratio (expect >> 1: 0143
# found about half of label-year positives prevalent, and every never-re-coded prior
# case and every later diagnosis now joins).
# PRE-REGISTERED READS (vs 0145, the whole-history incident baseline on 0137): incident
# macro AUC, stacked and flat, under the same rule (--prevalence-check --prev-history
# on the new run). A clear rise (> ~0.02) = label-year gating starved the topics;
# flat = labels are not the lever at this K. Secondary: all-patient AUC (record
# readout), own-code census (n_profiles), root head.
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
gate_attestation: ever     # label redesign R2 (the only change vs 0137)
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

# 0146 — all-ever topic gating on 0137's configuration (label redesign R2)

## Run

Bundle MISS (~20 min + one full-history condition_era read for the gate), fit (as
0137), readout + stacked arm, then the whole-history prevalence check. The sidecar
must be on this cluster (built 2026-10-13 for 0137; the new run reuses 0137's code
map, so copy the witness).

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
R137=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0137-dismech-ribbon-record-tpn5-v2codemap
RUN=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0146-gate-ever-on-0137-config
mkdir -p "$RUN"
nohup bash -c '
  make -C analysis/cloud exp ID=146 || exit 1
  cp "'"$R137"'"/conversion_sidecar.json "'"$RUN"'"/
  echo "=== readout START $(date)"
  make -C analysis/cloud gated-pc-readout ID=146 GPR_ARGS="--readout-mode distributed --readout-l2 100 --readout-theta-mass 0.99 --readout-stacked"
  echo "=== prevalence START $(date)"
  make -C analysis/cloud gated-pc-readout ID=146 GPR_ARGS="--prevalence-check --prev-history --acq-tag history"
  echo "=== DONE $(date)"
' > "$RUN"/sweep_log.md 2>&1 &
grep -E "gate-ever" "$RUN"/sweep_log.md
```

## Run log

**2026-10-13/14 — run.** Bundle MISS as designed (gate post-pass 1,052 s — the
provider over the full-history condition_era of the train persons), fit 50 iterations
(ELBO rising, -2.17e8 -> -8.55e7), readout + stacked, whole-history prevalence check.
The report's `pc-no-op` / STARVED flags are expected: weight_y = 0 (unsupervised gated
fit, post-hoc heads), as 0137.

## Results

**Gate receipt.** Train gated (doc, node) cells 722,718 -> 1,857,140 (2.6×); docs with
any gate 176,479 -> 209,026; median per-node ratio of gated docs **3.51** (387 nodes).
The gate change was large.

**The read: no change in new-diagnosis AUC.**

| | 0137 (label-year gate) | 0146 (all-ever gate) |
|---|--:|--:|
| incident macro AUC, whole-history rule, stacked (492 nodes) | 0.694 | **0.696** |
| same, flat | 0.612 | **0.612** |
| incident, lookback rule, stacked (535 nodes) | 0.702 | 0.705 |
| all-patient macro AUC, stacked (whole-history node set) | 0.822 | 0.825 |
| within-parent ranking macro AUC (554 nodes) | — | 0.733 |
| root head AUC | — | 0.836 |

Pre-registered read (> ~0.02 = gating starved the topics): **not met** (+0.002).
Gating each disease's topics on 3.5× as many training patients leaves the
new-diagnosis signal where it was. Label-year gating was not what limited 0137 —
labels are not the lever at this K. Remaining candidates: what the 5-year pre-index
record carries at all (the raw-code baseline answers this), and representation
(K, α, the background/private split) — not the gate.
