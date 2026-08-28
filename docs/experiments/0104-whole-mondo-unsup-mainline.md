---
id: 104
slug: whole-mondo-unsup-mainline
status: pending
model_class: gated_pc
cohort: population_mondo_all
cohort_def: population_mondo_all
disease: rare_priority
# THE MAINLINE DELIVERABLE RUN (closeout handoff §7.2; gated on exp 0103's A/B gate
# PASSING). First whole-Mondo fit: the FULL powered DAG (no mondo_branch => all body
# systems), K≈3,827 / C≈3,820 (insight 0071: 2,513 powered anchors + 1,306 class nodes,
# n_bg=8, tpn=1). This is the SCALED-BACK MAINLINE, not a PC experiment: weight_y=0 makes
# the primary arm the unsupervised gated LDA itself (the PC apparatus is inert — closeout
# §5), and skip_unsup_gated drops the now-redundant twin, so ONE fit + the distributed
# readout (ADR 0046) is the whole run. Everything the driver collects is (C,)-sized or
# the lean float32/uint8 test bundle (~6 bytes/cell); the old driver readout is
# structurally impossible here (24+ GB of collects) — readout_mode pinned, not auto.
# Head params below are lineage carry-over from 0102/0103 and INERT at weight_y=0.
readout_mode: distributed
# θ-width lever (plan v2.2): per-doc top-m truncated θ — measured 3.7× per pass on the
# cluster (17.6s vs 65s at m=256). MEASURED COVERAGE (smoke attempt 3): m=256 keeps only
# 0.132 mean / 0.077 p10 of θ mass — BELOW the raw Dirichlet(0.5) prior (~0.34), because
# the α=0.5 floor × K=3,827 = 1,913 pseudo-counts DOMINATES per-doc evidence (~10² tokens):
# θ ≈ 90% uniform prior haze + 10% signal (run-log arithmetic). The mass rule is therefore
# the WRONG test here: the truncated 87% is near-non-discriminative (per-doc-constant-ish
# haze, largely absorbed by intercept + per-node standardization), while top-m SELECTION
# (= ordering by evidence counts) keeps the signal. ENABLEMENT NOW RESTS ON THE PRICING
# TEST, not coverage: `make gated-pc-readout ID=103 GPR_ARGS="--readout-mode distributed
# --readout-theta-topm 256"` vs 0103's recorded full-K numbers — ΔAUC ≲ 1e-3 keeps this
# key; larger drops it (full-K readout, ~6-7h/solve).
readout_theta_topm: 256
weight_y: 0.0
weight_y_warmup_iters: 0
skip_unsup_gated: true
dag_source: mondo
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
tpn: 1
optimize_doc_concentration: true
head_optimizer: newton
head_newton_ridge: 0.05
head_l2: 0.01
grad_cavi_iters: 15
topic_trust: 0.05
max_iter: 100
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
# Per-run Spark tuning, RECORDED HERE rather than retyped into CHARM_SPARK_CONF on
# every launch (run log 08-26/08-27/08-28: this run's operational settings were
# rediscovered three times, and the recovery relaunch that dropped one of them ran
# at a third of the cores). Layered after the fixed submit flags and before
# CHARM_SPARK_CONF, which stays the last-wins per-invocation escape hatch; the same
# block is injected into `make gated-pc-readout ID=104` so a recovery re-readout
# runs under the same settings as the fit it is rescuing.
spark_conf:
  # Spot-VM kill waves under memory pressure: at K≈3,827 the fit broadcasts ~355 MB
  # per iteration and the readout's (C,K) partials are ~117 MB, so the 2g default
  # overhead is what the container gets killed for exceeding, not the heap.
  spark.executor.memoryOverhead: 4g
  # Dynamic allocation is CONFIRMED enabled on the cluster (08-28: checked in the
  # Spark config directly). Separately, `yarn node -list` showed all nodes RUNNING
  # while containers were "killed on request", ruling preemption OUT for those
  # waves; DA idle-release is the remaining suspect, not a proven cause. The floor
  # is cheap insurance either way: it stops the app shrinking below the
  # non-preemptible primaries during the driver-side L-BFGS gaps.
  spark.dynamicAllocation.minExecutors: 8
---

# 0104 — Whole-Mondo unsupervised mainline: the first all-body-system gate + readout

**Why.** The scaled-back mainline (closeout 2026-08-20) is the unsup gated LDA + post-hoc
readout, and its whole-Mondo scale-up is next-steps item 2. Both former blockers are
cleared: the O(C·K²) dense-head wall belonged to the co-fit head (absent at weight_y=0 —
insight 0071's correction), and the readout's driver collect is replaced by the
distributed batched-L-BFGS fit + lean eval (ADR 0046, gated by exp 0103). This run
produces the first population-wide, all-condition calibrated per-node posteriors — the
substrate for exports, dashboards, and VOI (next-steps items 3+).

**Run only after exp 0103's A/B equality gate passes.**

## Scale expectations (watch these, they are the run's second deliverable)

- **Fit:** K grows 444 → ≈3,827 (~8.6×). Per-iter cost is roughly linear in K at fixed
  minibatch (gated E-step is O(|allowed|), but held-out CAVI and λ updates see full K);
  budget several × 0103's per-iter wall-clock and consider `num_partitions` 96 → 192 if
  executors are idle-skewed.
- **Readout fit:** L-BFGS driver state at C·K ≈ 14.6M params: W+b ~117 MB, m=6 history
  ~1.4 GB — inside the 8g PC driver but tight next to the eval bundle; set
  `CHARM_DRIVER_MEMORY=12g` if the solve OOMs (the plan's float32-history/node-batching
  fallbacks exist but measure first). Expect the heartbeat to show tens of iterations,
  each one treeAggregate over the train split.
- **Lean eval bundle:** ~6 bytes/cell → at D_te≈80k, C≈3,820 about 1.9 GB. The
  calibration diagnostic holds one extra float64 test-split copy while it runs.
- **Moments aggregate:** (C,K)×2 float64 ≈ 234 MB driver-side, one-time.

## What to read (make -C analysis/cloud report ID=104)

1. **The fit itself** — ELBO trajectory, per-node α behavior at K≈3,827 (ADR 0045's
   floor is load-bearing here), and wall-clock/iter.
2. **unsup readout macro + rarity quartiles** — the first whole-Mondo AUC/AP; compare the
   cardiovascular subset against 0103's unsup arm (expect the same ballpark on shared
   nodes; a big drop = the fit, not the readout).
3. **`batched L-BFGS` heartbeat + summary** — passes, converged/fittable, stalled count,
   wall-clock: the scale read that decides whether whole-population readouts need the
   float32-history/node-batching work.
4. **Per-node ECE / calibration** — the deliverable is calibrated posteriors; if per-node
   ECE degrades at depth, the isotonic layer (already in the driver) is the lever.
5. **[cost] driver RSS** at the readout and eval phases vs the estimates above.

## Run

```bash
cd ~/repos/CHARMPheno && git pull origin claude/gated-conditional-voi && \
  CHARM_DEV=1 CHARM_SPARK_CONF='spark.locality.wait=0s' make -C analysis/cloud exp ID=104  # smoke first
CHARM_SPARK_CONF='spark.locality.wait=0s' make -C analysis/cloud exp ID=104
make -C analysis/cloud report ID=104

# Recovery re-readout on a SAVED fit (the early save lands npz + fit-only manifest
# before the readout, so a readout death never costs the fit):
make -C analysis/cloud gated-pc-readout ID=104 GPR_ARGS="--readout-max-iter 60"
```

The front matter's `spark_conf:` (memoryOverhead 4g + a minExecutors floor) is
injected into BOTH commands automatically — do not retype it into
`CHARM_SPARK_CONF` / `GPR_SPARK_CONF`; those remain for genuinely one-off probes
(`spark.locality.wait=0s` after a mid-run scale-up) and win over the doc if passed.
The explicit `--readout-max-iter 60` above is needed for THIS run only: its manifest
was written before fits recorded `readout_max_iter`, so the tool cannot tell a
60-iter `CHARM_DEV` smoke from a 200-iter record run and falls back to 200. Fits
written from now on record it, and the flag can be dropped.

## Run log

**2026-08-21 — smoke attempt 1: driver-JVM heap OOM at fit iteration 11 — a NEW scale
wall, now fixed.** The fit itself was healthy (48.6s/iter at K≈3,827, α floor holding,
ELBO rising, domain fracs sane) — then the driver JVM OOM'd in `task-result-getter` /
dispatcher threads; the executor "lost heartbeat" storm that followed was collateral of
the dead driver, not executor failure. Root cause: the SVI stats `treeReduce` (depth 2)
ships DENSE λ-shaped partials — **~355 MB each at K≈3,827 × V=11,601** (vs 41 MB at
C=444) — so the driver received ~sqrt(96)≈10 of them per iteration ≈ 3.5 GB of serialized
blocks through an 8g heap. Neither the exp doc's watch-list (which predicted the pinch at
the READOUT) nor 0103 could see this; it is K·V-driven and fit-side. Fix (same commit):
`spark_vi.core.runner._agg_depth` sizes the treeReduce depth from the params payload —
depth 3 above 128 MB/partial (driver burst ÷ ~P^(1/6)), byte-identical depth 2 below; the
readout aggregates got the same auto-rule (`_fit_readout_heads` depth=None → auto; its
(C,K) partials are ~117 MB at whole-Mondo). Belt to that suspenders: run the smoke with
`CHARM_DRIVER_MEMORY=16g` — depth 3 cuts the burst to ~4.6 partials ≈ 1.6 GB, comfortable
at 16g with the L-BFGS state beside it.

**2026-08-21 — smoke attempt 2 (depth-3 + 16g): FIT CLEAN, READOUT CRAWLS — the θ-width
lever comes due.** The fit is healthy end-to-end at whole-Mondo (30 dev iters, ~43s/iter
at K=3,827 — barely above cardiovascular's ~31s; depth-3 aggregation + 16g driver held,
zero drama). The readout is where the scale bites: **C=3,820, 3,057 fittable / 763
degenerate, 56.2M observed train cells** (5.7× cardiovascular — deeper DAG, bigger
closures), and each cell's dot is K=3,827 wide (8.6×) → ~49× more cell-work: measured
**~65s/data pass** vs 1.9s at C=444 (~1.7 TB memory traffic per pass), ~6-7h per 60-iter
dev solve, × main + calibration solves per arm. This is the plan §1 "θ-width lever" the
design deliberately deferred pending measurement: per-doc θ over 3,827 topics should be
mass-concentrated, so a top-m truncated-θ readout (m≈256) cuts pass cost ~K/m ≈ 15×.
In flight: always-on θ mass-coverage logging, a `readout_theta_topm` flag (default off,
sparse-exact kernels, by-node vectorization), and a dev-profile skip of the calibration
solve. Operational note: `results_partial.json` lands when the MAIN solve's readout
completes (at the `gated_pc (pc_topics_lr): macro AUC=` line) — the calibration solve
after it is safely interruptible in a smoke.

**2026-08-21 — smoke attempt 3 (top-m=256 + dev calibration-skip): the coverage line
lands a REAL finding, then the master's disk fills.** Fit clean again (~43s/iter).
- **`theta top-m mass: m=64:0.079/0.025 m=128:0.098/0.043 m=256:0.132/0.077
  m=512:0.196/0.144 (mean/p10)`** — coverage BELOW the raw Dirichlet(0.5) prior (~0.34
  at m=256). Arithmetic: the ADR 0045 α floor (0.5) × K=3,827 = **1,913 uniform
  pseudo-counts vs ~10² tokens of per-doc evidence**, so posterior-mean θ is ~90% flat
  prior haze + ~10% signal; top-256 of that ≈ 0.13 — matches the measurement. The
  diffuseness is STRUCTURAL (α·K vs doc length), not a property of patients. Two
  consequences: (1) mass coverage is the wrong enablement test here — the truncated haze
  is near-non-discriminative (per-doc-constant-ish, absorbed by intercept + per-node
  standardization) while top-m selection = evidence-count ordering keeps the signal;
  enablement now rests on the PRICING test (front-matter comment; `gated-pc-readout
  ID=103 --readout-theta-topm 256` vs the recorded full-K numbers, ~30 min). (2) A
  broader flag: EVERYTHING reading posterior-mean θ mass at whole-Mondo (node_affinity,
  dashboards) sees the same haze; the fit itself is fine (CAVI works in the far sparser
  E[log θ] geometry), and α=0.5's original justification (the PC shaping Jacobian, ADR
  0045) is moot at weight_y=0 — a lower/K-scaled α at whole-Mondo is a legitimate future
  experiment, deliberately NOT bundled into this one.
- Sparse-path cluster speed: **17.6s/pass** (vs 65 full-K) — 3.7× on a preempting
  cluster vs 5.4-7.3× local.
- **Died at ~readout iter 5-10: `OSError: Errno 28 No space left on device`** on the
  MASTER's local disk (the wrapper's summary.md append; the runs dir lives on the
  separate workspace mount). Prime suspect: stale Spark scratch from the day's killed/
  OOM'd runs (`/tmp/spark-*`, block-manager dirs — attempt 1's OOM killed the
  ContextCleaner mid-flight; Ctrl-C'd runs never clean; each whole-Mondo fit iteration
  broadcasts ~355 MB). Remedy before the next attempt, with no job running:
  `df -h; sudo du -x -d 2 /tmp /hadoop | sort -h | tail; rm -rf /tmp/spark-*`.

**2026-08-22 — smoke attempt 4 (FRESH cluster): ENOSPC again at readout iter 15 — with
the master's local disk 80% FREE. Root cause found: it was gcsfuse all along.** The runs
dir is a GCS-FUSE mount. Two of its semantics explain every log-file incident this week:
(1) a long-lived append handle uploads to GCS only on CLOSE — so the ORIGINAL
empty-summary.md (0103 smoke, "4h of output vanished") was not truncation: the cluster
died holding the handle and GCS kept the last-closed content, the header. (2) The per-line
open-append-close that "fixed" it makes every committed line a FULL-OBJECT rewrite against
GCS's ~1 mutation/s/object cap; bursty phases (readout heartbeats + Spark executor-loss
stack traces, on summary.md AND driver_log.md simultaneously) back gcsfuse's staged temp
files up behind the throttle until ENOSPC kills the wrapper — twice, on two different
clusters, with local disk healthy. Fix (same commit): both writers now BATCH lines and
close once per ~20s per file — far under the mutation cap, bounded staging, and a crash
loses at most one batch instead of causing the crash. Also noted from this attempt: the
main-fit L-BFGS burned ~26 passes/iter in iters 5-15 (heavy Armijo backtracking at
C=3,057) — if that persists it strengthens the Wolfe-line-search case; watch the next
smoke's passes/iter.

**2026-08-22 — smoke attempt 5 (gcsfuse fix in): sparse pass cost CONFIRMED (~18s/pass,
was 65); the remaining wall is PASS COUNT.** Iters 3→10: 164 passes / 7 iterations ≈ 26
passes/iter (vs ~1.5-6 at C=437) — the shared-pass Armijo backtracker pays a full pass
whenever ANY of 3,057 nodes is still halving, and the per-iteration straggler depth grows
with C. Two independent runs show the same profile: structural, not cluster noise. At this
rate the 60-iter dev solve is ~7h despite the sparse kernels. Fix package in flight:
(1) masked trial passes — trials 2+ (and every pass, for frozen nodes) evaluate only
still-searching nodes' cells (exact bookkeeping; composes with the by-node kernel so late
trials are nearly free); (2) safeguarded quadratic-interpolation backtracking (typical
depth 25 → 2-4; Armijo acceptance rule unchanged, so converged solutions are unchanged);
(3) the fit now SAVES (npz + fit-only manifest) immediately after the fit phase, before
any readout — a readout death stops costing the fit, and gated-pc-readout can always
resume. Run left grinding overnight as the first attempt with the ENOSPC cause fixed.

**2026-08-27 — smoke attempt 6: fit SAVED (early-save's first real rescue); readout
killed by spot-VM churn; recovery then exposed that the MONDO PATH NEVER CACHES THE
BUNDLE.** The fit completed (4,198s) and the early save landed npz + fit-only manifest
before the readout — which died when a reclaimed worker ate all 4 attempts of one task
(fixed same day: `spark.task.maxFailures=8` + `spark.excludeOnFailure.enabled=true` in
every submit). The recovery re-readout then MISSed the bundle cache — root cause:
`gated_pc_cloud`'s mondo branch calls the multidomain assembler DIRECTLY, never
`load_or_build`. No mondo fit has ever written the bundle cache; every mondo fit
silently re-assembles from BigQuery; `cache_uri` has been inert for mondo runs; and
BOTH re-readout failures this week (0103's — previously mis-attributed to the cluster
restart wiping HDFS — and today's) were this gap. The "cache hit" lines in mondo logs
are the ontology-FILE cache, not the bundle. Fix in flight: multidomain-aware cache
format + mondo-aware key (SNOMED keys byte-identical), mondo fit path routed through
load_or_build (DAG-climb skipped on a hit), manifest records the mondo key fields, and
the readout tool gains load-OR-REBUILD with a vocab/λ-dimension safety check so a saved
fit is never scored against a drifted corpus.

**2026-08-28 — recovery attempt (fresh cluster, 2 primary + 10 spot): the load-or-rebuild
path WORKS in production, and the "kill swarms" get a better explanation.**
- Cache MISS → **bundle rebuilt from manifest params in 1,086s and written through**
  (`.../case_finding_cache/d0f1fba8c73d7860`); multi-domain model reconstruct + transform
  17s; **coverage line bit-identical to the fit-era run (assembly is deterministic; the
  drift gate passed silently)**. The whole recovery seam is validated end-to-end.
- Solve progressing on the saved fit: identical gradient trajectory to prior attempts
  (5.56e4 → 9.9e3 by iter 5), and the line-search economy is visibly working at scale —
  **~4.5 passes/iter (vs ~26 pre-fix)**, avg nodes/pass falling 2,384 → 1,096 by iter 5.
- Pace is core-starved, though: ~85-100s/pass. The kill waves left 8 registered nodes ×
  1 container × the GPR_SPARK_CONF cores=2 override = **16 active cores (vs ~48 in fit
  runs)** — ETA ~5-7h at this pace. The bundle being cached now makes a restart cheap:
  drop the cores=2 override (keep memoryOverhead=4g if desired), lose ~5 solve
  iterations, run ~2×+ faster.
- **Kill-swarm reframe:** `yarn node -list -all` mid-run shows ALL 8 nodes RUNNING (none
  UNHEALTHY) — including w-0/w-1 (primaries), whose containers were among those "killed
  by external signal". Non-preemptible primaries can't be reclaimed and their nodes never
  went bad, so a large share of the week's "kill swarms" are most plausibly **Spark
  dynamic allocation releasing idle executors at phase boundaries** (kills cluster right
  AFTER heavy phases = when executors go idle; "Container killed on request" = the AM
  asked) — cosmetic churn, not failures — interleaved with genuine spot preemptions
  (which DID abort a stage once, pre-maxFailures-8). Candidate hardening for long solves:
  pin `spark.dynamicAllocation.enabled=false` (or a floor via minExecutors) in the
  readout/fit submits so executors stop flapping across the driver-side L-BFGS gaps.
  Disk-health theory: not supported by this listing (no UNHEALTHY nodes).

**2026-08-28 — the operational settings move INTO this doc (they were being
rediscovered every launch).** Three of this week's launches differed only in which
`CHARM_SPARK_CONF` / `GPR_SPARK_CONF` string got retyped, and the cost was real: the
recovery attempt above ran at **16 active cores instead of ~48** because a `cores=2`
override was carried over while the settings that mattered were not. Fix (same
commit): a `spark_conf:` front-matter key on the experiment doc, layered
fixed-flags → doc → `CHARM_SPARK_CONF` (spark-submit last-wins), validated at parse
time — a key not starting with `spark.` now fails the launch instead of being
silently dropped by spark-submit, which is how a typo'd conf would otherwise
"apply" for hours and then not have. This doc now records the two settings the
run's own history argues for: `spark.executor.memoryOverhead=4g` (the kill waves of
08-26/08-27 are container-overhead kills at ~355 MB broadcasts, not heap) and
`spark.dynamicAllocation.minExecutors=8` (DA confirmed on above; the floor spans the
non-preemptible primaries so idle-release cannot shrink the app across the
driver-side L-BFGS gaps). `make gated-pc-readout ID=104` reads the same block via
`run_experiment.py --print-spark-conf`, so the recovery path and the fit path can no
longer drift apart. Companion fix for the OTHER thing that had to be remembered: the
fit manifest now records `readout_max_iter` (the post-`CHARM_DEV` effective value)
and the readout tool defaults to it, so the relaunch is just
`make -C analysis/cloud gated-pc-readout ID=104 GPR_ARGS="--readout-max-iter 60"` —
and the explicit 60 is needed for THIS run only, whose manifest predates the field.

**2026-08-21 — UNBLOCKED: the 0103 A/B gate PASSED** (macro |Δ| ≤ 1.1e-4 both arms; see
0103's run log). Reference bar from 0103's full-row readout: unsup cardiovascular
**0.7584 AUC / 0.5428 AP over 241 nodes**, pooled conditional ECE 0.0028 (isotonic →
0.0010). Since staging, the readout also gained warm starts + a CHARM_DEV cap of 60
solver iterations (insight 0074), so the smoke's readout is ~3× cheaper than 0103's.
