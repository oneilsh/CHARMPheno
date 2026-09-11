---
id: 121
slug: mondo-cardiovascular-alpha-equalized
status: planned
model_class: gated_pc
cohort: population_mondo_all
cohort_def: population_mondo_all
disease: rare_priority
# THE PRIOR-ASYMMETRY TEST (insight 0090). 0113's config with the gated engine's alpha
# policy changed — and, for the first time on this path, alpha actually LEARNED.
#
# WHY. 0116's feature ablation at a converged head (0089/0090) found a node's identity
# spread over its closure's spare topics (own block + bg 0.67 vs everything 0.81) and
# 29/36 credited nodes with no node-specific residual: ancestor capture. The asymmetry
# that seeds it is in the PRIOR: an ancestor's block is visible to every document under
# it (N_anc), a leaf's to few (N_leaf), so a uniform alpha hands the ancestor N_anc*alpha
# of prior mass and the leaf N_leaf*alpha. `alpha_init: equalized` inverts that with a
# DERIVED init — equal TOTAL prior pseudo-count per block, alpha_b ∝ 1/N_b, rescaled to
# doc_concentration's mean (gated_lda.equalized_alpha) — and then the per-node tied
# empirical-Bayes alpha (insight 0059) takes over: an init that picks the basin (single
# fits are multimodal in alpha), not a strength knob.
#
# CORRECTION IT CARRIES. `optimize_doc_concentration: true` in 0113-0120 was INERT on the
# Gated-PC path (the injected engine never received it; alpha fixed at 0.5). This build
# wires it (`set_alpha_policy`, histogram from the document RDD), so BOTH the init and the
# optimizer are new here. Exp 0122 (uniform init + optimizer) is the control that isolates
# the init; 0113 (fixed alpha) is the baseline both pair against.
#
# READ: own-bg vs family-closure vs all AUC at ridge 100 (the 0116 ladder is the reference:
# 0.669 / 0.777 / 0.810), the fit log's `[pc] gated alpha policy` line and the learned
# alpha by depth, `--digest` evidence-by-depth vs 0113, and --profile-support is N/A (no
# eta). Acceptance: own+bg rises materially toward all => the ordering was the lever and
# own blocks now carry identity; own+bg flat while all holds => the competition is not
# prior-driven and the cascade (SAGE regime b) is the structural move.
dag_source: mondo_native
mondo_branch: MONDO:0004995
tpn: 5
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
alpha_init: equalized
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
  # 0113's geometry, K = 1,498 as there.
  spark.executor.cores: 2
  spark.executor.memory: 8g
  spark.executor.memoryOverhead: 3g
  spark.dynamicAllocation.enabled: "false"
  spark.executor.instances: 20
  spark.excludeOnFailure.timeout: 10m
  spark.cleaner.periodicGC.interval: "5min"
---

# 0121 — cardiovascular branch, children-first α init + learned α (the prior-asymmetry test)

0113 with the gated engine's α policy changed and nothing else: `alpha_init: equalized`
(every topic block gets the same total prior pseudo-count across the corpus, α ∝ 1/N_b,
a derived init with no strength parameter) followed by the per-node empirical-Bayes α,
which this build wires for the first time on the Gated-PC path (insight
[0090](../insights/0090-a-nodes-identity-is-spread-over-its-closures-spare-topics-own-block-067-family-072-closure-078-all-081.md)
point 4: the flag was inert on 0113–0120, α was fixed at 0.5).

## Run

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/gated-conditional-voi && git checkout claude/gated-conditional-voi && git pull --ff-only
make -C analysis/cloud exp ID=121
RUN=$(ls -d /home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0121-*)
nohup bash -c '
make -C analysis/cloud gated-pc-readout ID=121 GPR_ARGS="--readout-mode distributed"
for m in own-bg family-closure; do
  make -C analysis/cloud gated-pc-readout ID=121 GPR_ARGS="--readout-mode distributed --readout-feature-mask $m"
done
make -C analysis/cloud inspect-topics ID=121 INSPECT_ARGS="--digest"
make -C analysis/cloud readout-ab ID=121 BASE=113 CREDITED=1
' > "$RUN"/sweep_log.md 2>&1 &
```

Same bundle key as 0113/0116/0120 (α policy is a fit parameter) — a cache HIT wherever
one of those ran today. The readout ridge (100) comes from the manifest.

## Read

| head may load on | 0116 (fixed α=0.5, tpn=5) | 0121 (equalized init + EB α) | 0122 (uniform init + EB α) |
|---|--:|--:|--:|
| own block + background | 0.6688 | | |
| own + siblings + ancestors + background | 0.7768 | | |
| everything | 0.8098 | | |

Plus the learned α by depth (fit log), `--digest` evidence-by-depth vs 0113, and the
paired per-node deltas vs 0113 from `readout-ab`.

## Results

*(pending)*
