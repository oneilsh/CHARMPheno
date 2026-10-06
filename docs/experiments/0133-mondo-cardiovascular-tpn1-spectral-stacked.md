---
id: 133
slug: mondo-cardiovascular-tpn1-spectral-stacked
status: planned
model_class: gated_pc
cohort: population_mondo_all
cohort_def: population_mondo_all
disease: rare_priority
# THE tpn=1 ARM UNDER THE STACKED HEAD (insight 0093). 0123's config VERBATIM with ONE
# knob: tpn 5 -> 1 (K = 8 + 298 = 306). The user's lean since the handoff of 2026-10-06:
# tpn=1 is the only non-arbitrary budget, and nothing found the right number per node.
#
# WHY NOW. 0132 (flat K=1498) vs 0123 (gated tpn=5) under the same two decoders sized what
# the gate buys: +0.03 within-cohort, +0.014 de-novo AUC, +0.05 de-novo AP, uniform by
# depth; the root head does not care (0.806 vs 0.798). Ten digests say a tpn=5 block is a
# five-way strata split with the signature as one topic in five. The stacked head needs
# exactly one thing per block — a signature — so this is the first time tpn is asked with
# a decoder that does not need the strata. Does one topic per node hold the gap?
#
# READ (ridge 100, both readouts; AB vs 0123 and 0132):
#   - flat head within-cohort macro + paired vs 0123 (0.7946) and 0132 (0.7612).
#   - stacked de-novo macro AUC / AP vs 0123 (0.850 / 0.236) and 0132 (0.836 / 0.185).
#   - digest: is the one topic the signature, or the largest stratum?
# OUTCOMES:
#   - 0133 ≈ 0123 on both: tpn=1 holds the gap — one signature per node, the unit the
#     program wanted, with the stacked head decoding. tpn is closed at 1.
#   - 0133 ≈ 0132 (or below): the gate's +0.03 lives in the strata split, not in a
#     signature; next is the blend — flat background (n_bg large) + one residual topic
#     per node, 0134.
#   - 0133 between: read the digest before choosing; the signature may be there but
#     starved (spectral seed at tpn=1 anchors ONE word per node).
# COST: K=306 — the spectral seed and the fit are both far cheaper than 0123's (fewer
# anchors to find, a gated doc sees ~8 + depth topics). Bundle HIT on a warm cluster.
dag_source: mondo_native
mondo_branch: MONDO:0004995
tpn: 1                      # THE ONLY CHANGE vs 0123
max_iter: 50
diag_only: true
# --- the ONLY change from 0113: spectral block-aligned seed for the gated engine ---
init: spectral
spectral_method: scalable   # concatenated V ~11.6k >= 8000 threshold; dense = driver wall
spectral_d: 768             # random-projection dim: smaller = faster + bigger safe batch (see COST)
anchor_scope: closure       # node trained from its whole closure; ancestors deflated by topo order
spectral_topo_order: forward  # ancestors-first: each node's seed = its increment over ancestors
count_transform: none       # THE ONLY CHANGE vs 0115: raw per-visit counts (0114's representation)
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

# 0133 — cardiovascular branch at tpn=1 under the stacked head (the budget question, finally askable)

0123's config with `tpn: 1` (K=306). Insight 0093 sized what the gate buys over flat strata
under the stacked closure head (+0.03 within-cohort, +0.05 de-novo AP); every digest from
0127–0131 says a tpn=5 block is a strata split with the signature as one topic in five.
A decoder that needs one thing per block is the right place to ask whether one topic per
node holds that gap.

## What it does

`make -C analysis/cloud exp ID=133` → fit-only spectral-init gated LDA at tpn=1, then the
ridge-100 readout (record), the stacked readout, the paired ABs vs 0123 and 0132, and a
digest of the cardiomyopathy neighbourhood (one topic per node: is it the signature?).

## Acceptance criterion

The three-way split in the front matter: ≈0123 (tpn closed at 1), ≈0132 (signal is the
strata split → blend arm 0134), or between (read the digest).

## Run

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/gated-conditional-voi && git checkout claude/gated-conditional-voi && git pull --ff-only
RUN=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0133-mondo-cardiovascular-tpn1-spectral-stacked
mkdir -p "$RUN"
nohup bash -c '
  make -C analysis/cloud exp ID=133 || exit 1
  echo "=== readout START $(date)"
  make -C analysis/cloud gated-pc-readout ID=133 GPR_ARGS="--readout-mode distributed --readout-l2 100"
  echo "=== stacked START $(date)"
  make -C analysis/cloud gated-pc-readout ID=133 GPR_ARGS="--readout-mode distributed --readout-l2 100 --readout-stacked"
  make -C analysis/cloud inspect-topics ID=133 COMPARE=123 INSPECT_ARGS="--readout-auc"
  make -C analysis/cloud inspect-topics ID=133 COMPARE=132 INSPECT_ARGS="--readout-auc"
  make -C analysis/cloud inspect-topics ID=133 INSPECT_ARGS="--digest --redundancy 30 --grep '"'"'cardiomyopathy|atrial fibrillation|heart failure|pregnan'"'"'"
  echo "=== DONE $(date)"
' > "$RUN"/sweep_log.md 2>&1 &
```

Early check: the corpus line must read `K=306 gated topics (8 bg + 298 nodes x 1 tpn)`.

Pull the numbers with:

```bash
grep -E "^=== |corpus: V=|starved|gated_pc(_stacked)? \(pc_topics_lr\): (macro|detection)|root head alone|stacked: max|ranking \(within|paired per-node|paired delta|depth [0-9]+:|flat sigma|stacked P_stack|^all: n=|^by depth" "$RUN"/sweep_log.md
```

## Run log

(none yet)

## Results

(pending)
