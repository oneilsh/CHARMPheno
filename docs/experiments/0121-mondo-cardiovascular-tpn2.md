---
id: 121
slug: mondo-cardiovascular-tpn2
status: planned
model_class: gated_pc
cohort: population_mondo_all
cohort_def: population_mondo_all
disease: rare_priority
# THE SPARE-CAPACITY TEST (insight 0090). 0113's config with ONE knob changed: tpn 5 -> 2.
#
# WHY. 0116's feature ablation at a converged head (0089/0090) found a node's identity
# spread over its CLOSURE's spare topics — own block + bg 0.67, + siblings 0.72,
# + ancestors 0.78, everything 0.81 — and the profile-support read found 29/36 credited
# nodes with no node-specific residual at all: deflation lets an ancestor's five topics
# split the parent cohort into sub-presentations that line up with its children, so the
# children's words land THERE. At tpn=2 an ancestor has one spare topic; at tpn=1 none.
# The question: does removing spare capacity CONCENTRATE identity in the own block (own+bg
# ablation rises toward the full decoder; boosted/own topics gain evidence) or push it to
# the background (bg ablation rises, own+bg does not)? tpn=2 keeps one free topic per node
# so a distinctive non-profile signature still has somewhere to go (the --profile-support
# pair read stays available); the tpn=1 companion is the same doc with tpn: 1 (exp 0122).
#
# NO profile-eta here: a clean 0113 twin, so the comparison is tpn alone. readout_l2: 100
# is the converged-head instrument (0089); the record's l2=1 is not used again.
#
# READ (after `make exp` + the readouts below): own-bg vs family-closure vs all AUC, and
# `inspect-topics --digest` evidence-by-depth vs 0113. Acceptance: own+bg within ~0.05 of
# all (identity concentrated) => tpn=1/2 is the structural choice and heads aimed at own
# blocks become meaningful; own+bg flat at ~0.67 while all drops => spare capacity was
# doing real work and the cascade (SAGE regime b) is the next move.
dag_source: mondo_native
mondo_branch: MONDO:0004995
tpn: 2
max_iter: 50
diag_only: true
preindex_closure: false
readout_mode: distributed
readout_theta_topm: 256
readout_l2: 100
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
optimize_doc_concentration: true
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
  # 0113's geometry; K ≈ 8 + 299×2 ≈ 606, smaller than 0113's 1,498.
  spark.executor.cores: 2
  spark.executor.memory: 8g
  spark.executor.memoryOverhead: 3g
  spark.dynamicAllocation.enabled: "false"
  spark.executor.instances: 20
  spark.excludeOnFailure.timeout: 10m
  spark.cleaner.periodicGC.interval: "5min"
---

# 0121 — cardiovascular branch at tpn=2 (the spare-capacity test)

0113 with `tpn` 5 → 2 and nothing else. Insight
[0090](../insights/0090-a-nodes-identity-is-spread-over-its-closures-spare-topics-own-block-067-family-072-closure-078-all-081.md)
found that at tpn=5 a node's case-finding identity lives in its ancestors' spare topics
(own+bg 0.67 vs everything 0.81 on 0116, at a converged head) and that 29/36 credited
nodes keep no node-specific residual. This run removes most of that spare capacity and
asks where the identity goes.

## Run

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/gated-conditional-voi && git checkout claude/gated-conditional-voi && git pull --ff-only
make -C analysis/cloud exp ID=121
# readout (l2=100 comes from the manifest), then the three ablation reads
RUN=$(ls -d /home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0121-*)
nohup bash -c '
make -C analysis/cloud gated-pc-readout ID=121 GPR_ARGS="--readout-mode distributed"
for m in own-bg bg family-closure; do
  make -C analysis/cloud gated-pc-readout ID=121 GPR_ARGS="--readout-mode distributed --readout-feature-mask $m"
done
make -C analysis/cloud inspect-topics ID=121 INSPECT_ARGS="--digest"
' > "$RUN"/sweep_log.md 2>&1 &
```

Same bundle key as 0113/0116/0120 (tpn is a fit parameter, not a corpus one) — a cache
HIT on any cluster where one of those ran today.

## Read

| head may load on | 0116 (tpn=5) | 0121 (tpn=2) |
|---|--:|--:|
| background only | 0.6002 | |
| own block + background | 0.6688 | |
| own + siblings + ancestors + background | 0.7768 | |
| everything | 0.8098 | |

Plus `--digest` evidence-by-depth against 0113's, and 0121−0113 paired per-node AUC via
`make readout-ab ID=121 BASE=113` (0113 is scored at l2=100 on the current cluster).

## Results

*(pending)*
