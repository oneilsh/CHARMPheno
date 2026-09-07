---
id: 116
slug: mondo-cardiovascular-tpn5-profile-eta
status: planned
model_class: gated_pc
cohort: population_mondo_all
cohort_def: population_mondo_all
disease: rare_priority
# THE PROFILE-ETA A/B (insight 0083, plan docs/superpowers/plans/
# 2026-09-06-profile-eta-prior-plan.md). 0113's config VERBATIM — random init,
# fit-only, 50 iters, CV branch at tpn=5 — plus ONE effective knob: the HPO-profile
# word-side eta prior (profile_eta below). Each of the 132 credited label nodes'
# blocks gets its FIRST topic's Dirichlet eta boosted on the OMOP condition tokens
# its rolled-up HPO phenotype profile maps to (freq x IDF weights, D3-normalized to
# double the topic's condition-domain prior mass, NOT terms multiplicatively
# downweighted); the 167 uncredited label nodes keep flat eta and are the INTERNAL
# CONTROL. The prior re-enters every lambda update (lambda_target = eta_vec +
# counts), so a starved zero-count topic still holds a sharp E[log beta] on its
# profile tokens — persistent anti-starvation AND alignment — while real counts
# trivially dominate it.
#
# WHY eta and not the init. The 0113/0114 readout A/B (insight 0083) showed random
# init BEATS spectral on every case-finding axis: spectral's variance-seeking
# anchors produce sharp-but-misaligned topics that soak theta away from the honest
# ancestor/background signal the localized head reads. An init can wash out or lock
# a wrong basin; a prior is soft, persistent, and label-aligned by construction
# (knowledge, not variance). Spectral stays demoted to a legibility/diagnostic tool.
#
# JUDGED BY THE READOUT, not evidence/sharpness diagnostics (the standing lesson of
# 0083: topic evidence and discriminability can anti-correlate). Baseline of record:
# 0113's gated-pc-readout — macro ranking AUC 0.7813 / AP 0.5255 over 193 scored
# nodes, detection 0.6347/0.7156, per-node median 0.791. Pre-registered acceptance
# below in the body.
#
# INPUT ARTIFACT: profile_eta points at hpoa_stage2_probe's --emit-eta TSV
# (mondo_id, concept_id, weight, neg, coverage; freq x IDF folded; concept ids so
# the file is bundle-agnostic). WORKSPACE-INTERNAL (the coverage column is
# train-derived) — regenerate it on the cluster before the fit (Run below); it is
# never committed. No bundle/corpus cache-key change: profile-eta is a fit
# parameter recorded in the run manifest.
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
# The one effective knob vs 0113 (defaults per plan D1-D4; the flags emit only
# because profile_eta is set, so a config without it stays byte-identical).
# PATH NOTE: `make -C analysis/cloud exp` leaves the driver's cwd at
# analysis/cloud, so a repo-root-relative path would miss; this ~-anchored path
# matches the probe's --emit-eta target exactly (pandas expands ~ on read).
profile_eta: ~/repos/CHARMPheno/data/ontology/profile_eta_MONDO_0004995.tsv
profile_eta_strength: 1.0
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

# 0116 — CV branch at tpn=5 + HPO-profile eta prior (the knowledge-prior A/B)

The record arm of insight
[0083](../insights/0083-spectral-init-costs-case-finding-topic-evidence-is-not-discriminability.md)'s
live candidate: **0113's exact config (random init) + the profile-eta word-side
prior**, one effective knob, judged by `gated-pc-readout` against 0113's recorded
numbers. Design and work packages:
[the plan](../superpowers/plans/2026-09-06-profile-eta-prior-plan.md); feasibility:
the 2026-09-06 HPOA scouting report (rolled-up profiles reach 132/299 label nodes
at median 0.87 positive-doc coverage).

## What it does

`make -C analysis/cloud exp ID=116` → the 0113 fit-only gated LDA (CV-restricted native-Mondo DAG,
`tpn=5`, K≈1,498, random init), with each credited node's block carrying one
profile-aligned topic: its eta boosted on the node's rolled-up HPO-profile
condition tokens (freq × IDF, normalized to `1.0 × eta_base × V_condition` added
pseudo-mass; NOT concepts multiplicatively downweighted with a positivity floor).
The other four block topics, all uncredited nodes' blocks, and the background
topics keep flat eta. The driver builds the boost from the TSV + bundle meta
(concept ids → condition vocab, curie → engine id) and logs a counts-only stats
line; the manifest records a `profile_eta` block.

## Pre-registered acceptance (decided before the run; plan §acceptance)

1. **Primary:** readout macro AUC over the shared scored nodes ≥ 0.7813 − noise
   (non-inferiority); a WIN is credited-node improvement — per-node AUC on the
   132 credited nodes UP vs 0113 while the 167 uncredited nodes (internal
   control) move ~0. Read via `inspect_topics.py --readout-auc` on both runs.
2. **Secondary (legibility):** starvation fraction vs 0113's 72% on credited
   deep nodes (the tilted-floor hypothesis predicts credited nodes feed under
   random init); digest `--grep 'cardiomyopath'` topics anchor on phenotype, not
   pregnancy/demographics.
3. **Failure reads:** credited nodes DOWN → prior misweighted (suspect IDF/scale
   before abandoning); everything flat → strength too low (try 3.0); uncredited
   nodes moved → a wiring bug (the boost leaked past its blocks), stop and fix.

## Run

Regenerate the prior TSV first (the probe requires a bundle-cache HIT — run
`gated-pc-readout ID=114` first on a fresh cluster to rebuild the bundle):

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/gated-conditional-voi && git checkout claude/gated-conditional-voi && git pull --ff-only
make -C analysis/cloud hpoa-stage2-probe ID=114 GPR_ARGS="--emit-eta ~/repos/CHARMPheno/data/ontology/profile_eta_MONDO_0004995.tsv"
```

Then the fit and readout:

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/gated-conditional-voi && git checkout claude/gated-conditional-voi && git pull --ff-only
make -C analysis/cloud exp ID=116
make -C analysis/cloud gated-pc-readout ID=116 GPR_ARGS="--readout-mode distributed"
```

Read (off-cluster safe):

```bash
make -C analysis/cloud inspect-topics ID=116 INSPECT_KEY=<key> RESOLVE_NAMES=1 \
    INSPECT_ARGS="--readout-auc"
make -C analysis/cloud inspect-topics ID=116 INSPECT_KEY=<key> RESOLVE_NAMES=1 \
    INSPECT_ARGS="--digest --grep 'cardiomyopath'"
```

## Run log

(pending)

## Results

(pending)
