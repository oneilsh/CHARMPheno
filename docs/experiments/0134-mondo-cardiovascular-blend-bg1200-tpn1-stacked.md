---
id: 134
slug: mondo-cardiovascular-blend-bg1200-tpn1-stacked
status: planned
model_class: gated_pc
cohort: population_mondo_all
cohort_def: population_mondo_all
disease: rare_priority
# THE BLEND: shared strata + one signature per node, under the stacked head (insight 0094).
# 0133's config with ONE knob: n_bg 8 -> 1200, so K = 1200 + 298 = 1498 — the SAME width as
# 0123 (tpn=5) and 0132 (flat), the non-arbitrary match. Every document sees the 1,200
# background topics plus its closure's one-topic blocks.
#
# WHY. 0133 (tpn=1, K=306) holds the gate's gap on ranking (+0.008 vs tpn=5, +0.038 vs
# flat), has the best de-novo AUC (0.855), 0% starved, and the most legible per-node topics
# of the series — and loses the ROOT head (0.781 vs 0.807/0.798) and with it de-novo AP
# (0.198 vs 0.236). The root head reads population strata and K=306 offers eight; 0132's
# 1,250 live flat strata gave it 0.798. This run adds the strata back WITHOUT touching the
# per-node unit. HSLDA's shared profiles + the gate's one signature per node.
#
# SEED. The background anchors are found by the projected greedy on a spectral_d-dim
# sketch; with n_bg > spectral_d the greedy past d anchors is degenerate (the residual
# span is full). spectral_d 768 -> 2048 (> 1200 + the deepest closure's anchors). The seed
# is slower than 0133's (sketch and projections scale with d) but there is no tpn=5
# anchor hunt; expect it between 0133's and 0123's.
#
# READ (ridge 100, both readouts; ABs vs 0133, 0123, 0132; named digest):
#   - root head and de-novo AP: back to ~0.80 / ~0.24 (0123's) or not.
#   - within-cohort macro + paired vs 0133 (0.8006): do the per-node topics keep their
#     signal next to 1,200 strata, or does the background absorb it (the head can read
#     either; the digest says which it is).
#   - digest: are the per-node topics still the 0133 textbook ones? Does the background
#     carry the cohort strata (pregnancy, diabetes, asthma) that 0133's same-vocabulary
#     children fell into — i.e. does DCM's one topic get its own words back once the
#     young-women stratum has a background home?
# OUTCOMES:
#   - root ~0.80, per-node ≈ 0133, digest as legible or better: the unit is found — one
#     signature per node over shared strata, closure product as decoder. tpn closed at 1.
#   - root ~0.80 but per-node topics emptied into the background (paired vs 0133 down,
#     digest generic): the background competes with the blocks; n_bg is then a real knob
#     and the answer is somewhere between 8 and 1200 — stop and think, do not sweep.
#   - root unchanged (~0.78): width was not it; the root head wants something else
#     (e.g. the readout's top-256 truncation at K=1498 — check theta top-m mass).
# COST: fit between 0133's and 0132's (a gated doc sees 1,200 + depth topics); seed as
# above; bundle HIT on a warm cluster (same key as 0123/0132/0133).
dag_source: mondo_native
mondo_branch: MONDO:0004995
tpn: 1
max_iter: 50
diag_only: true
# --- the ONLY change from 0113: spectral block-aligned seed for the gated engine ---
init: spectral
spectral_method: scalable   # concatenated V ~11.6k >= 8000 threshold; dense = driver wall
spectral_d: 2048            # must exceed n_bg + the deepest closure's anchors (see SEED)
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
n_bg: 1200                  # THE ONLY CHANGE vs 0133: K = 1200 + 298 = 1498
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

# 0134 — shared strata + one signature per node (n_bg=1200, tpn=1, K=1498) under the stacked head

0133 with 1,200 background topics. Insight 0094: one topic per node holds the gate's gap
and is the most legible topic side of the series; what it lacks is strata for the root
head to read. This adds them at matched width.

## What it does

`make -C analysis/cloud exp ID=134` → spectral-init gated fit with a 1,200-topic shared
background and one topic per node; the ridge-100 readout (record); the stacked readout;
paired ABs vs 0133, 0123 and 0132; a named digest of the cardiomyopathy neighbourhood.

## Acceptance criterion

The three-way split in the front matter: root head back to ~0.80 with per-node topics
intact (the unit), root back but blocks emptied (n_bg is a knob — stop), or root
unchanged (width was not it).

## Run

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/gated-conditional-voi && git checkout claude/gated-conditional-voi && git pull --ff-only
RUN=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0134-mondo-cardiovascular-blend-bg1200-tpn1-stacked
mkdir -p "$RUN"
nohup bash -c '
  make -C analysis/cloud exp ID=134 || exit 1
  echo "=== readout START $(date)"
  make -C analysis/cloud gated-pc-readout ID=134 GPR_ARGS="--readout-mode distributed --readout-l2 100"
  echo "=== stacked START $(date)"
  make -C analysis/cloud gated-pc-readout ID=134 GPR_ARGS="--readout-mode distributed --readout-l2 100 --readout-stacked"
  for b in 133 123 132; do make -C analysis/cloud inspect-topics ID=134 COMPARE=$b INSPECT_ARGS="--readout-auc"; done
  make -C analysis/cloud inspect-topics ID=134 RESOLVE_NAMES=1 INSPECT_ARGS="--digest --redundancy 30 --grep '"'"'cardiomyopathy|atrial fibrillation|heart failure|pregnan'"'"'"
  echo "=== DONE $(date)"
' > "$RUN"/sweep_log.md 2>&1 &
```

Early check: the corpus line must read `K=1498 gated topics (1200 bg + 298 nodes x 1 tpn)`.

Pull the numbers with:

```bash
grep -E "^=== |corpus: V=|starved|top-m mass|gated_pc(_stacked)? \(pc_topics_lr\): (macro|detection)|root head alone|stacked: max|ranking \(within|paired per-node|paired delta|depth [0-9]+:|flat sigma|stacked P_stack|^## paired|^all: n=|^by depth" "$RUN"/sweep_log.md
```

## Run log

**2026-10-07 — first launch killed in the seed.** 54 minutes of silence after
`>>> gated_pc fit` with the driver at ~10% CPU: the background anchor greedy
(`find_anchors_projected`) re-projected every vocabulary row against the whole basis at
every step — O(n² · V · d) in Python-level dots — fine for 8 background anchors, days for
1,200. Rewrote both greedies (dense and projected) onto one basis-form search
(`spectral_init.greedy_anchors`: basis as state, one GEMV per chosen anchor, incremental
residual norms, seeds spanned in one SVD batch); same anchors up to floating-point ties,
pinned against the old loop by `spark-vi/tests/test_anchor_greedy_residual_form.py`.
Relaunched on the fixed code.

## Results

(pending)
