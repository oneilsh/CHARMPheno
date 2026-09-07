---
id: 118
slug: mondo-cardiovascular-tpn5-profile-eta-pc
status: planned
model_class: gated_pc
cohort: population_mondo_all
cohort_def: population_mondo_all
disease: rare_priority
# THE PC REVIVAL PROBE (closeout 2026-08-20 §6's "one open empirical question",
# now with insight 0085's new ingredient). The eta ladder (0113/0116/0117) proved
# the profile prior buys topic-profile ALIGNMENT but zero discriminability: the
# word-side prior can reallocate theta (0117's fed-mass shift) yet has no term
# preferring PREDICTIVE allocation. PC is exactly that term — a label-side pull on
# lambda. The 0096-0103 arc refuted PC at weight_y=16 (0103: PC arm -0.046 vs
# unsup; shaping toward a weak head hurts by construction) and parked it behind
# "co-fit >= ~0.758". But the closeout's open question — does CONTROLLED shaping
# (low weight_y, corr_relDlambda in the healthy 2-5% band, trust cap on) reach
# neutral-or-help? — was never answered: exp 0100 (weight_y=2) has no logged
# results (stashed). This run answers it, on today's stronger base: random init +
# profile-eta S=1.0 (the 0085 operating point), so the pull has an ALIGNED target
# the 0096-0103 era never had.
#
# CONFIG = 0116 VERBATIM + the PC block below (weight_y 0 -> 2.0, warmup 0 -> 10
# of 50 iters — one conceptual knob: PC on, controlled regime). Everything the
# era's post-mortems demanded is already inherited: EG mass-preserving lambda
# correction (0098 run-2: stable to weight_y 1000), topic_trust 0.05 (the
# trust-region cap), head_standardize true + localized path_cousins_kids head,
# grad_cavi_iters 15 (the CAVI-Jacobian peak, 0098 run-3).
#
# WATCH IN THE FIT LOG (0100's read, verbatim): corr_relDlambda must sit ~0.02-
# 0.05 (NOT >1); |w_CK|max WILL be large under standardization (cosmetic — judge
# by corr + readout); ELBO must not detonate (EG fix makes that an engine
# regression, not a tuning issue). eta_boost[...] sentinel must still show
# topics=132 mass~440.
dag_source: mondo_native
mondo_branch: MONDO:0004995
tpn: 5
max_iter: 50
diag_only: true
preindex_closure: false
readout_mode: distributed
readout_theta_topm: 256
weight_y: 2.0
weight_y_warmup_iters: 10
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
# The 0085 operating point, unchanged from 0116 (strength 1.0 — free legibility,
# zero readout cost; regenerate the TSV via the ID=114 probe if the cluster
# restarted).
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

# 0118 — profile-eta + controlled PC (the label-side pull, revived on an aligned base)

Answers the PC closeout's open question
([2026-08-20 closeout §6](../reports/2026-08-20-pc-arc-closeout-and-scaled-back-mainline-handoff.md)):
does **controlled** shaping — `weight_y: 2.0`, 10-iter warmup, `topic_trust`
cap, corr_relΔλ held in the healthy 2–5% band — reach **neutral-or-help**,
where weight_y=16 hurt (exp 0103: −0.046)? Run on the strongest current base
(0116 = random init + profile-eta S=1.0), so the label pull has a
phenotype-aligned target: the eta ladder proved word-side pressure alone
reallocates θ without adding label information (insight
[0085](../insights/0085-eta-dose-buys-alignment-monotonically-but-no-auc-the-prior-is-an-interpretability-lever.md));
PC is the term that prefers PREDICTIVE allocation, which the unsupervised
objective simply lacks.

Expect a slower fit than 0116's ~27s/iter (the co-fit head work returns);
warmup means iters 1–10 are unsupervised, so early iteration lines should
match 0116's shape.

## Pre-registered acceptance (decided before the run)

1. **Primary (neutral-or-help):** `readout-ab ID=118 BASE=113` — macro ≥
   0.7813 − noise is the floor (non-inferiority); a WIN is paired per-node
   dAUC up vs BOTH 0113 and 0116 (run `readout-ab ID=118 BASE=116` too).
   Independently, the co-fit head arm's AUC (now readable — weight_y > 0)
   vs the closeout bar: **≥ 0.758 = full revival**; below it, PC can still
   pass on the readout arms.
2. **Secondary:** `--profile-align` vs 0116 (does the label pull reinforce
   the profile alignment or fight it? Either answer is informative);
   starvation fraction; Q1-rarity flavor via the by-depth deltas (the
   closeout named rare-tail rescue as PC's fallback value).
3. **Failure reads:** corr_relΔλ > ~0.1 sustained → over-driving even at
   2.0 → stop and re-run ONCE at `weight_y: 1.0`; macro down with corr in
   the healthy band → controlled shaping hurts even with an aligned target
   → PC is closed pending the stash-branch head-quality lift (matrix-free
   full-K head), write the closing insight; ELBO detonation → engine
   regression (EG fix should make this impossible), stop and debug; the
   η_boost sentinel absent → wiring bug, stop.

## Run

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/gated-conditional-voi && git checkout claude/gated-conditional-voi && git pull --ff-only
make -C analysis/cloud exp ID=118
make -C analysis/cloud gated-pc-readout ID=118 GPR_ARGS="--readout-mode distributed"
```

Reads (off-YARN; bundle key auto-discovered):

```bash
make -C analysis/cloud readout-ab ID=118 BASE=113
make -C analysis/cloud readout-ab ID=118 BASE=116
make -C analysis/cloud inspect-topics ID=118 CREDITED=1 COMPARE=116 INSPECT_ARGS="--profile-align"
make -C analysis/cloud inspect-topics ID=118 CREDITED=1 RESOLVE_NAMES=1 INSPECT_ARGS="--digest --grep 'cardiomyopath'"
```

## Run log

(pending)

## Results

(pending)
