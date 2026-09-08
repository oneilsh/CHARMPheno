---
id: 120
slug: mondo-cardiovascular-tpn5-profile-eta-pc-lbfgs
status: planned
model_class: gated_pc
cohort: population_mondo_all
cohort_def: population_mondo_all
disease: rare_priority
# THE FAIR PC TEST (the arc's goal). 0118 (Newton head) OOM'd on the O(C·K²)
# Hessian collect at K=1498; 0119 (SGD head) was too weak (corr_relDlambda ~6e-6,
# non-converging). Both stock solvers fail in opposite ways. This run uses the new
# scalable STRONG head — head_optimizer: lbfgs, the matrix-free amortized batched
# L-BFGS co-fit head (plan 2026-09-07-lbfgs-cofit-head-plan.md; solver lifted from
# the readout, distributed re-scoring provider, EG + trust cap unchanged) — which
# PASSED the local-simulator coupling gate by wide margins (insight 0086: corr
# climbs to the cap then DECAYS, grad_y SHRINKS 5.2->0.12 = converges, tracks the
# sklearn oracle within 0.005). This is the first head that both scales AND shapes.
#
# CONFIG = 0118 (= 0116 + PC) with the head swapped to lbfgs and the scale-free
# trust cap ON. Two deliberate changes from 0118's newton block:
#   1. head_optimizer newton -> lbfgs (the whole point).
#   2. weight_y 2.0 -> 12.0, head_trust_move 0.03. The trust cap makes the shaping
#      scale-free: it clips corr_relDlambda to 0.03 regardless of weight_y/K (the
#      sim proved corr peaks at exactly the cap, never exceeds). weight_y is set
#      generously (12, the validated-sim value) so the lbfgs head's natural move
#      EXCEEDS the cap and the cap GOVERNS — the failure mode to avoid is
#      under-shooting (0119's corr 6e-6), which a too-small weight_y invites; the
#      cap makes a too-large weight_y safe (clipped). head_inner_iters 3 /
#      head_history_reset true are the sim-decided defaults (insight 0086).
#
# WATCH IN THE FIT LOG (the coupling signals, from insight 0086 — this is the
# cluster analog of the sim gate):
#   * corr_relDlambda should PEG at ~0.03 (cap active) or sit in 0.02-0.05. If it
#     stalls <0.02 the lbfgs natural move is under the cap -> raise weight_y. If it
#     tries to exceed 0.03 the cap holds it (that is correct).
#   * ||grad_y|| must SHRINK across outer iters (the head converging on moving θ).
#     A GROWING grad_y is the 0119 chase = the coupling broke at this K (θ moving
#     faster than the sim); then revisit head_inner_iters (up) / head_history_reset.
#   * NO O(C·K²) collect / driver OOM (structurally absent — lbfgs emits grad-only
#     stats, no Hessian). eta_boost[topics=132 mass~440] sentinel present.
#   * |w_CK|max large under standardization is cosmetic (judge by corr + readout).
#
# COST: the co-fit head RE-SCORES θ every outer iter (one persisted scoring pass +
# the inner L-BFGS treeAggregates), so per-iter wall will exceed 0116's ~27s
# materially — this is inherent to co-fit, and 0120 is the scale test. Right-size
# the cluster and watch per-iter wall; if a pass OOMs an executor, lower
# num_partitions granularity is NOT the lever (it is the driver-collect that was
# the Newton problem, now gone) — the lbfgs passes are O(C·K) executor-side.
dag_source: mondo_native
mondo_branch: MONDO:0004995
tpn: 5
max_iter: 50
diag_only: true
preindex_closure: false
readout_mode: distributed
readout_theta_topm: 256
weight_y: 12.0
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
head_optimizer: lbfgs        # the scalable strong head (WP0-3, insight 0086)
head_inner_iters: 1          # was 3 (0086 default); dropped to 1 on the driver-OOM
                             # hardening re-run (2026-09-08). 0086: inner_iters is
                             # NON-DISCRIMINATIVE under the trust cap — corr pegs at
                             # 0.03 regardless — so 1 preserves the verdict AND cuts
                             # the treeAggregate/stage count ~3x (driver-heap + wall).
head_history_reset: true     # sim-decided (0086); fresh curvature = no stale-curvature risk
head_trust_move: 0.03        # scale-free cap — pins corr_relDlambda <= 0.03 regardless of K
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
fit_save_interval: 5         # crash insurance (2026-09-08): dump fit-only lambda
                             # every 5 iters, atomic, re-scoreable with
                             # gated-pc-readout. profile-eta can't resume (D5), so a
                             # death mid-run is salvaged by reading the last dump.
seed: 42
cache_uri: hdfs:///user/dataproc/charm/case_finding_cache
# The 0085 operating point (strength 1.0), unchanged from 0116/0118 — the aligned
# target the label pull shapes toward. Regenerate the TSV via the ID=114 probe if
# the cluster restarted.
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
  spark.executor.memoryOverhead: 4g   # was 3g; the co-fit θ-scoring UDF (numpy CAVI)
                                      # is Python OFF-HEAP, charged to overhead — the
                                      # first executor exit-143s were node memory
                                      # pressure. +1g/container, YARN-safe.
  spark.dynamicAllocation.enabled: "false"
  # 20 executors (whole cluster) per Shawn (2026-09-04). A tpn=5 branch fit is small;
  # YARN grants what it needs and holds the rest pending.
  spark.executor.instances: 20
  spark.excludeOnFailure.timeout: 10m
  spark.cleaner.periodicGC.interval: "5min"
  # DRIVER-OOM hardening (2026-09-08): the app-killer at iter 32 was the DRIVER
  # heap (SparkUI-*/ContextCleaner/task-result-getter all threw OOM), not an
  # executor. The co-fit head runs many stages (head_inner_iters x Armijo passes x
  # partitions x 50 iters) and broadcasts expElogbeta each outer iter; retained UI
  # task metadata + a broadcast-GC backlog fill the 8g driver over the run.
  # Disable the UI (nobody watches a 4h batch UI; all diagnostics are in driver_log)
  # to drop the single largest driver-heap sink, and force GC often. Driver HEAP
  # itself is raised out-of-band with CHARM_DRIVER_MEMORY=16g at launch (client-mode
  # --driver-memory can't be set from spark_conf — see the run command).
  spark.ui.enabled: "false"
  spark.cleaner.referenceTracking.blocking: "true"
---

# 0120 — profile-eta + controlled PC, matrix-free L-BFGS head (the fair PC test)

The run the whole PC-revival arc has been building toward. 0118 proved the Newton
head OOMs at K=1498 (O(C·K²) Hessian collect); 0119 proved the SGD head is too
weak (corr 6e-6, non-converging). The new `head_optimizer: lbfgs` — the
matrix-free amortized batched-L-BFGS co-fit head (plan
[2026-09-07-lbfgs-cofit-head-plan.md](../superpowers/plans/2026-09-07-lbfgs-cofit-head-plan.md),
insight [0086](../insights/0086-lbfgs-cofit-head-couples-cleanly-on-the-simulator-trust-cap-makes-the-inner-knobs-non-discriminative.md))
— is the first head that BOTH scales (O(C·K) shuffle, no Hessian collect) AND
shapes right-sized (it passed the local-simulator coupling gate: corr to the cap
then decaying, grad_y shrinking, tracks the sklearn oracle). Run on the strongest
base (random init + profile-eta S=1.0, the aligned target), this is finally the
neutral-or-help PC test the 2026-08-20 closeout's open question demanded.

## Pre-registered acceptance (decided before the run)

1. **Fit-health (the cluster coupling gate — watch live):** corr_relΔλ pinned at
   ~0.03 / in 0.02–0.05; **grad_y SHRINKING** across outer iters (the head
   converging, NOT 0119's chase); ELBO stable; no O(C·K²) collect / driver OOM;
   per-iter time O(C·K) (higher than 0116's 27s, but bounded, not 0118's 1450s);
   η_boost sentinel present. A growing grad_y or corr that won't reach the band =
   the coupling broke at this K → adjust head_inner_iters / weight_y per the front
   matter, do not force the readout.
2. **Primary (the fair PC test):** `readout-ab ID=120 BASE=113` AND `BASE=116` —
   credited paired dAUC UP vs both (0113 flat, 0116 profile-eta-no-shaping);
   macro ≥ 0.7813 − noise floor. The co-fit head arm's AUC is now a FAIR read
   against the closeout's 0.758 revival bar (a right-sized head, unlike SGD).
3. **Secondary:** `--profile-align` vs 0116 (does the label pull reinforce or
   fight the profile alignment?); starvation fraction; by-depth deltas (rare-tail).
4. **Failure reads (decided now):** corr healthy + readout FLAT/DOWN → controlled
   shaping does not help even with a strong, aligned-target head → **PC closes on
   the merits** (write the closing insight; the record then points at
   representation — episode index / cascade — as the frontier). corr won't reach
   the band → coupling/knob issue, not a PC verdict (revisit inner_iters). Over-drive
   past the cap → the EG radius needs lowering (should be impossible — the cap is a
   hard clip). grad_y grows → θ moves faster than the sim at this K → raise
   head_inner_iters or set head_history_reset false.

## Run

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/gated-conditional-voi && git checkout claude/gated-conditional-voi && git pull --ff-only
CHARM_DRIVER_MEMORY=16g make -C analysis/cloud exp ID=120
make -C analysis/cloud gated-pc-readout ID=120 GPR_ARGS="--readout-mode distributed"
```

`CHARM_DRIVER_MEMORY=16g` raises the driver heap (client-mode `--driver-memory`
can't come from `spark_conf`); it pairs with `spark.ui.enabled: false` +
`head_inner_iters: 1` in the front matter — see the 2026-09-08 run-log entry.

Reads (off-YARN; bundle key auto-discovered):

```bash
make -C analysis/cloud readout-ab ID=120 BASE=113
make -C analysis/cloud readout-ab ID=120 BASE=116
make -C analysis/cloud inspect-topics ID=120 CREDITED=1 COMPARE=116 INSPECT_ARGS="--profile-align"
make -C analysis/cloud inspect-topics ID=120 CREDITED=1 RESOLVE_NAMES=1 INSPECT_ARGS="--digest --grep 'cardiomyopath'"
```

## Run log

**2026-09-07 — first attempt CRASHED at ~22s (ADR-0047 closure violation), fixed.**
The fit died before iter 1 with `[CONTEXT_ONLY_VALID_ON_DRIVER]` (SPARK-5063):
VIRunner ships the model into every E-step task closure (`_model=model`), and the
injected distributed lbfgs provider closes over the SparkContext, so cloudpickling
the model with it attached raised. `OnlinePCLDA.set_head_stats_provider`'s docstring
already CLAIMED the provider "stays off the closure per ADR 0047" — but the
`__getstate__` that enforces it was never added (the invariant was documented, not
implemented). Fixed by adding `OnlinePCLDA.__getstate__` stripping
`_head_stats_provider` from pickled copies (mirrors `GatedOnlineLDA.__getstate__`'s
eta-boost exclusion): the driver instance keeps the provider — update_global consumes
it there — every executor-bound copy carries None. The local coupling test used the
IN-MEMORY provider (captures a picklable `rows` list, not a SparkContext), so nothing
had pickled the model with a Spark-capturing provider — the gap. Added a regression
test that cloudpickle (the production serializer) round-trips the provider-bearing
model dropping the provider. Re-run below.

**2026-09-08 — second attempt: DRIVER OOM at iter 32 (fit healthy throughout), hardened.**
The fit ran cleanly to iter 32 on 20 executors — ELBO climbing (−125.7M→−84.1M),
`corr_relΔλ` pegged at the 0.03 cap, `grad_y` flat (~5.8e4), η_boost sentinel
present — i.e. the L-BFGS co-fit head COUPLES and CONVERGES at K=1498 (the whole
engineering question of the arc). It died on the **driver**, not an executor: the
OOM stacks are all driver-side JVM threads (`SparkUI-*`, `Spark Context Cleaner`,
`task-result-getter-1`, `YARN application state monitor`, `driver-heartbeater`),
and the executor exit-143s were the downstream cascade after the AM stopped
heartbeating. Root cause = driver-heap accumulation over 30+ iters: retained
SparkUI task metadata (head_inner_iters × Armijo passes × 96 partitions × iters =
tens of thousands of tasks) + an expElogbeta broadcast-GC backlog, on an 8g driver.
Not a code leak in the head — the scored-df persist/unpersist lifecycle is bounded
(verified) — a Spark-bookkeeping resourcing wall. Hardening (verdict-preserving):
`head_inner_iters 3→1` (0086: non-discriminative under the cap; ~3x fewer stages),
`spark.ui.enabled=false` (drops the largest driver-heap sink; diagnostics live in
driver_log), `CHARM_DRIVER_MEMORY=16g`, `memoryOverhead 3g→4g` (the executor-side
Python-UDF pressure behind the first 143s). Also added `fit_save_interval: 5` —
the fit now atomically dumps the fit-only λ every 5 iters (new `--fit-save-interval`
on the `on_iteration` seam), so any future death mid-run leaves a readout-able
checkpoint (`gated-pc-readout ID=120` scores the last dump). profile-eta fits
can't resume (D5), so a periodic dump is the right insurance, not a resume.
Re-run per the command above.

## Results

(pending)
