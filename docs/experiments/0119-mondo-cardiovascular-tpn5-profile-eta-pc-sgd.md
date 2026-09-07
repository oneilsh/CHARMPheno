---
id: 119
slug: mondo-cardiovascular-tpn5-profile-eta-pc-sgd
status: planned
model_class: gated_pc
cohort: population_mondo_all
cohort_def: population_mondo_all
disease: rare_priority
# THE CHEAP PC PROBE (continues 0118, which abandoned at iter 7: the Newton head's
# O(C·K²) Hessian collect OOM'd the driver at K=1498, steady 1450s/iter — the
# model was stable, corr_relDlambda 0.05, the infra wasn't). Rather than build the
# matrix-free head speculatively, ask the cheap question first: does controlled
# shaping help the READOUT at all? Swap the ONE knob that caused the wall —
# head_optimizer newton -> sgd — which drops the head to an O(C·K) RM-damped
# gradient step (no Hessian, no per-node Hessian collect, no driver OOM, fast
# iters). Everything else is 0118 verbatim.
#
# READING (asymmetric by design, stated up front): the SGD head shapes WEAKER
# than Newton (closeout's ladder: the solver was worth +0.065 in head quality), so
# only a CLEAR result is conclusive — a positive readout delta greenlights the
# matrix-free head build (0118's deferred fix); a clear negative closes PC without
# it; a mild null stays ambiguous (weak-head confound, not a PC verdict).
#
# HEAD KNOBS UNDER SGD: head_optimizer=sgd's step is rho*headLrScale*weightY*g, so
# the inherited head_lr: 1.0 is IGNORED (it only damps the Newton step) — left in
# the front matter inert so the diff vs 0118 is exactly one line. The SGD shaping
# strength is weight_y (2.0) × head_lr_scale (default 1.0); corr_relDlambda on the
# first post-warmup iter self-diagnoses it. WATCH: corr_relDlambda should land
# ~0.02-0.05 (same healthy band). If it comes out tiny (<0.01, shaping too weak to
# test) raise head_lr_scale (2-3) and re-run; if >0.1 sustained, drop weight_y to
# 1.0. The η_boost[topics=132 mass~440] sentinel must still show.
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
head_lr: 1.0                 # INERT under head_optimizer: sgd (Newton-step damping only)
head_optimizer: sgd         # the one knob vs 0118 — O(C·K) gradient, no Hessian collect
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
# The 0085 operating point (strength 1.0), unchanged from 0116/0118.
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

# 0119 — profile-eta + controlled PC, SGD head (the cheap PC decider)

The runnable continuation of [0118](0118-mondo-cardiovascular-tpn5-profile-eta-pc.md),
which abandoned at iter 7 when the Newton head's O(C·K²) Hessian collect OOM'd
the driver at K=1498 (steady 1450s/iter). One knob changes —
`head_optimizer: newton → sgd` — dropping the head to an O(C·K) RM-damped
gradient step: no Hessian, no per-node Hessian collect, no driver OOM, fast
iters. Everything else is 0118 (= 0116 + weight_y 2.0 / 10-iter warmup).

**What this can and cannot decide** (stated before the run, since the read is
asymmetric): the SGD head shapes weaker than Newton — the 2026-08-20 closeout's
ladder put the solver at +0.065 of head-quality lever. So a **clear positive**
(readout up vs 0113 AND 0116) greenlights building the matrix-free L-BFGS head
for a fair full test; a **clear negative** (readout down even here) closes PC
without the engine spend; a **mild null** stays ambiguous — it could be the weak
head, not PC, and would itself argue for the head build if PC is still wanted.

## Pre-registered acceptance

1. **Primary:** `readout-ab ID=119 BASE=113` and `BASE=116` — is the paired
   per-node dAUC up vs both? (0116 is the no-shaping profile-eta base; 0113 the
   original flat baseline.) Macro ≥ 0.7813 − noise is the non-inferiority floor.
2. **Head sanity (not acceptance):** the co-fit head arm's AUC is now readable
   (weight_y > 0) but SGD-limited — read it for direction only, NOT against the
   0.758 revival bar (that bar is for a Newton-quality head).
3. **Secondary:** `--profile-align` vs 0116 (does the label pull move alignment?);
   starvation fraction; by-depth deltas (rare-tail flavor).
4. **Fit-health gates (first post-warmup iter):** corr_relΔλ in ~0.02–0.05;
   if <0.01 the SGD step is too weak to be a fair test → raise `head_lr_scale`
   to 2–3 and re-run; if >0.1 sustained → drop `weight_y` to 1.0. ELBO must not
   detonate; `η_boost` sentinel present.

## Run

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/gated-conditional-voi && git checkout claude/gated-conditional-voi && git pull --ff-only
make -C analysis/cloud exp ID=119
make -C analysis/cloud gated-pc-readout ID=119 GPR_ARGS="--readout-mode distributed"
```

Reads (off-YARN; bundle key auto-discovered):

```bash
make -C analysis/cloud readout-ab ID=119 BASE=113
make -C analysis/cloud readout-ab ID=119 BASE=116
make -C analysis/cloud inspect-topics ID=119 CREDITED=1 COMPARE=116 INSPECT_ARGS="--profile-align"
make -C analysis/cloud inspect-topics ID=119 CREDITED=1 RESOLVE_NAMES=1 INSPECT_ARGS="--digest --grep 'cardiomyopath'"
```

## Run log

(pending)

## Results

(pending)
