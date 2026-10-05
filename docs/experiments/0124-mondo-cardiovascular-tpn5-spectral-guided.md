---
id: 124
slug: mondo-cardiovascular-tpn5-spectral-guided
status: pending
model_class: gated_pc
cohort: population_mondo_all
cohort_def: population_mondo_all
disease: rare_priority
# THE GUIDED-ANCHOR ARM. 0123's config VERBATIM (CV branch MONDO:0004995, tpn=5,
# spectral init, raw counts, alpha fixed 0.5, fit-only) with ONE line added:
# spectral_anchor_profile -> the profile-eta TSV. So 0124-vs-0123 is a clean A/B on
# WHICH WORDS spectral may anchor each node's block on, and nothing else.
#
# WHY. The 0113/0115/0123 ladder settled three things: spectral FILLS the deep blocks
# (starvation 72% -> 1%); a filled block CARRIES its own node's signal (own-bg 0.73 vs
# 0.67 unfed); and what spectral still costs vs random init (-0.016 median, uniform
# across depth, 0123 AB) is a head-side cost of the anchor words it picks by
# co-occurrence geometry alone — dilated/intrinsic cardiomyopathy anchored on
# gestation-week codes (0114, insight 0082) because the pregnant-young-female stratum
# is a sharp, pure direction in Q. Binary counts were tried as the fix (0115) and cost
# detection instead. The lever is the ANCHOR SEARCH.
#
# MECHANISM (spec docs/superpowers/specs/2026-10-05-hpo-guided-spectral-anchors.md).
# For each profiled node the greedy farthest-point search runs in two stages over ONE
# basis: stage 1 picks only from the node's PREFERRED set — its rolled-up HPO-profile
# terms, realized to OMOP condition concepts, positive, weight>0 (freq x IDF, so a
# term every credited node shares is excluded), in the fitted vocabulary, and above
# the node's OWN df floor; stage 2 fills whatever stage 1 could not from the open
# pool. Deflation against background + ancestors is unchanged; beta recovery for
# every word is unchanged. "HPO decides WHICH words may anchor; the data decides WHAT
# the topic is." A set, not a weight: no strength knob. 132 of 299 label nodes have a
# set (report 2026-09-06, --rollup); the other 167 run today's open search exactly.
#
# READ (priority order — this build is FOR legibility; the readout is the guardrail):
#   1. --digest --grep on the cardiomyopathies: do intrinsic/dilated CM shed the
#      gestation anchors; does peripartum CM KEEP them; do AF/HF/valve stay coherent.
#      The seed log line `[pc] spectral anchor guide:` says how much of the anchoring
#      the profiles decided (nodes guided / anchors from profile / fallback-only).
#   2. starvation % from the fit log: must stay ~1%.
#   3. ridge-100 readout: full vs 0123 (0.7946) and 0113 (0.8087); detection vs
#      0.630; own-bg vs 0.730; paired AB vs 0123 by depth.
# ACCEPTANCE: anchors re-align (1) AND macro AUC within ~0.005 of 0123 (3). Recovery
# toward 0113 is the hypothesis, not the bar.
# OUTCOMES: re-align + hold -> spectral + guided anchors is the base init. re-align
# but AUC drops -> profile terms legible but not discriminative here; read WHICH
# profile anchors the heads don't load (digest + loadings) before touching the rule.
# No re-align (pregnancy persists) -> the preferred set is being exhausted (floor or
# ancestor deflation); the seed counts say which.
#
# COST: identical to 0123 (same bundle, K=1498, spectral seed ~1.5h + ~35 min fit on
# the n2-standard-8 master shape). The eta TSV is cluster-ephemeral (~ wiped on
# restart): regenerate with `hpoa-profile-survey --emit-codes ... --rollup` then
# `hpoa-stage2-probe ID=123 --emit-eta ...` (bundle HIT needed) before `make exp`.
dag_source: mondo_native
mondo_branch: MONDO:0004995
tpn: 5
max_iter: 50
diag_only: true
# --- the ONLY change from 0113: spectral block-aligned seed for the gated engine ---
init: spectral
spectral_method: scalable   # concatenated V ~11.6k >= 8000 threshold; dense = driver wall
spectral_d: 768             # random-projection dim: smaller = faster + bigger safe batch (see COST)
anchor_scope: closure       # node trained from its whole closure; ancestors deflated by topo order
spectral_topo_order: forward  # ancestors-first: each node's seed = its increment over ancestors
count_transform: none       # raw per-visit counts, as 0123 (binary was a detection tax, 0123)
# --- THE ONLY CHANGE vs 0123: HPO-guided spectral anchors (spec 2026-10-05) ---
spectral_anchor_profile: ~/repos/CHARMPheno/data/ontology/profile_eta_MONDO_0004995.tsv
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


# 0124 — CV branch, spectral init, raw counts, HPO-GUIDED anchors

0123's exact config with **`spectral_anchor_profile`** set: each profiled node's spectral
anchor search prefers its own HPO-profile terms (positive, IDF-weight > 0, in vocab,
above the node's own df floor) and falls back to the open search for the rest. No
strength knob. Nodes without a profile are untouched. See the spec for the rules and
the handoff 2026-09-11 §4/§5.1 for why this is the build.

## What it does

`make -C analysis/cloud exp ID=124` → fit-only spectral-init gated LDA on the CV branch
with guided anchors; `diag_only` saves the fitted globals. Then the ridge-100 readout
sweep, the paired ABs vs 0123 and 0113, and the cardiomyopathy digest.

## Acceptance criterion

Legibility first: the gestation anchors leave the non-peripartum cardiomyopathies and
the seed log shows the profiles decided a material share of the anchoring. Then the
guardrail: ridge-100 macro AUC within ~0.005 of 0123's 0.7946, detection ≥ 0.63,
starvation ≈ 1%.

## Run

Pre-flight: the eta TSV must exist at the path in the front matter (see COST above).

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/gated-conditional-voi && git checkout claude/gated-conditional-voi && git pull --ff-only
ls -l ~/repos/CHARMPheno/data/ontology/profile_eta_MONDO_0004995.tsv
RUN=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0124-mondo-cardiovascular-tpn5-spectral-guided
mkdir -p "$RUN"
nohup bash -c '
  make -C analysis/cloud exp ID=124 || exit 1
  echo "=== readout START $(date)"
  make -C analysis/cloud gated-pc-readout ID=124 GPR_ARGS="--readout-mode distributed --readout-l2 100"
  for m in own-bg family-closure; do
    make -C analysis/cloud gated-pc-readout ID=124 GPR_ARGS="--readout-mode distributed --readout-l2 100 --readout-feature-mask $m"
  done
  make -C analysis/cloud inspect-topics ID=124 COMPARE=123 INSPECT_ARGS="--readout-auc"
  make -C analysis/cloud inspect-topics ID=124 COMPARE=113 INSPECT_ARGS="--readout-auc"
  make -C analysis/cloud inspect-topics ID=124 INSPECT_ARGS="--digest --redundancy 30 --grep '"'"'cardiomyopathy|atrial fibrillation|heart valve|mitral|aortic|myocardial infarction|heart failure|pregnan|gestation'"'"'"
  echo "=== DONE $(date)"
' > "$RUN"/sweep_log.md 2>&1 &
```

Early checks: `grep -E "anchor guide|spectral anchor guide" "$RUN"/sweep_log.md` after the
seed (the driver's builder counts, then the estimator's pooled seed counts); the fit's
starvation line at the end of the fit.

Pull the numbers with:

```bash
grep -E "anchor guide|^=== |gated_pc(_own_bg|_family_closure)? \(pc_topics_lr\): (macro|detection)|^## paired|^all: n=|^by depth|starved" "$RUN"/sweep_log.md
```

## Run log

## Results
