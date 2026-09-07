---
id: 117
slug: mondo-cardiovascular-tpn5-profile-eta-s3
status: planned
model_class: gated_pc
cohort: population_mondo_all
cohort_def: population_mondo_all
disease: rare_priority
# THE STRENGTH RUNG (insight 0084, plan 2026-09-06 failure read #2). 0116 was a
# clean, wiring-validated NULL at profile_eta_strength 1.0: paired dAUC flat on
# both sides of the credit line, starvation unchanged — but the boost provably
# reached the starved floor (mass 440.1 vs 440.6 theory; starved deep CM topics'
# top-words became their HPO phenotype). Diagnosis: ~3.3 pseudo-mass per boosted
# topic cannot buy theta against fed topics in per-doc CAVI. This run turns the
# ONE knob the failure read pre-registered: strength 1.0 -> 3.0 (the boosted
# topic's condition-domain prior mass goes from 2x to 4x flat). Everything else
# is 0116 = 0113 verbatim; judged by `readout-ab ID=117 BASE=113` under the SAME
# acceptance as 0116. If still flat: question whether ANY eta dose can buy theta
# before distorting fed topics — i.e. whether the lever is eta at all (0084).
dag_source: mondo_native
mondo_branch: MONDO:0004995
tpn: 5
max_iter: 50
diag_only: true
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
# The one knob vs 0116 (which was itself one knob vs 0113): strength 1.0 -> 3.0.
# Same ~-anchored TSV as 0116 (the probe's --emit-eta output; regenerate if the
# cluster restarted since).
profile_eta: ~/repos/CHARMPheno/data/ontology/profile_eta_MONDO_0004995.tsv
profile_eta_strength: 3.0
profile_eta_topics: 1
profile_eta_min_coverage: 0.0
spark_conf:
  # Copied from 0110 (the only geometry that survived a whole-Mondo solve). At this
  # branch's much smaller K (~1,400 vs 3,827) it is over-provisioned and safe; a wider
  # shape is fine if the cluster is bigger. See 0110 for the kill-swarm forensics.
  spark.executor.cores: 2
  spark.executor.memory: 8g
  spark.executor.memoryOverhead: 3g
  spark.dynamicAllocation.enabled: "false"
  # 20 executors (whole cluster) per Shawn (2026-09-04). A tpn=5 branch fit is small;
  # YARN grants what it needs and holds the rest pending.
  spark.executor.instances: 20
  spark.excludeOnFailure.timeout: 10m
  spark.cleaner.periodicGC.interval: "5min"
---

# 0117 — profile-eta at strength 3.0 (the dose rung)

The pre-registered response to 0116's null
([0116](0116-mondo-cardiovascular-tpn5-profile-eta.md), insight
[0084](../insights/0084-profile-eta-strength-1-tilts-the-floor-legibly-but-cannot-buy-theta.md)):
same fit, same prior, same credited/uncredited split — `profile_eta_strength`
1.0 → **3.0** (≈10 pseudo-mass per boosted topic; condition-domain prior mass
4× flat). Acceptance is 0116's verbatim, judged by
`make -C analysis/cloud readout-ab ID=117 BASE=113`:

1. **Primary:** macro ≥ 0.7813 − noise; WIN = credited paired dAUC up,
   uncredited ~0.
2. **Secondary:** credited-deep starvation vs 72%; cardiomyopathy digest.
3. **Failure reads:** credited DOWN → the dose distorts before it helps (stop
   raising strength; question the eta lever per 0084); still flat → same
   question; uncredited moved → wiring bug (would be new — 0116's control was
   clean), stop.

## Run

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/gated-conditional-voi && git checkout claude/gated-conditional-voi && git pull --ff-only
make -C analysis/cloud exp ID=117
make -C analysis/cloud gated-pc-readout ID=117 GPR_ARGS="--readout-mode distributed"
make -C analysis/cloud readout-ab ID=117 BASE=113
make -C analysis/cloud inspect-topics ID=117 RESOLVE_NAMES=1 \
    INSPECT_ARGS="--digest --grep 'cardiomyopath'"
```

(The fit log should show `η_boost[topics=132 nnz=28883 mass≈1320]` — 3× 0116's
440.)

## Run log

(pending)

## Results

(pending)
