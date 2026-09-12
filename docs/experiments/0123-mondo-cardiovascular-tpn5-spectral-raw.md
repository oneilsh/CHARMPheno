---
id: 123
slug: mondo-cardiovascular-tpn5-spectral-raw
status: pending
model_class: gated_pc
cohort: population_mondo_all
cohort_def: population_mondo_all
disease: rare_priority
# THE COUNTS ARM. 0115's config VERBATIM with ONE knob flipped back: count_transform: none
# (raw per-visit counts, 0114's representation). This is 0114's config re-fit so it can be
# read at a CONVERGED head — 0114's own run dir lost its spectral λ to a MAX_ITER=2
# bootstrap re-run, so it cannot be re-read at ridge 100 and the raw-count arm has no
# converged-head number.
#
# WHY. 0115 landed: spectral + binary costs a uniform −0.019 macro vs 0113 at ridge 100
# (median paired dAUC −0.020, 151/193 nodes down, same at every depth), while own-bg
# (0.731 vs 0116's unfed 0.669) says the fed blocks DO carry their node's signal. The tax
# is head-side and spread across the whole tree. Two things changed between 0113 and 0115
# and this run cannot tell them apart: WHICH words spectral anchors on (volume-driven,
# insight 0082/0083) and the BINARY count representation. 0123 splits them.
#
# The user's prior on binary: it "rubs the wrong way" — a lot of repetition is noise, but
# some counts carry meaning (a code recorded at every visit for years is not the same
# patient as one recorded once), and log1p is an equally arbitrary squash. Binary was a
# cheap fix for the pregnancy-by-volume anchors; if it costs case-finding, the anchors
# should be fixed at the anchor search (HPO-guided anchors, handoff 2026-09-11 §5.1), not
# by flattening the data.
#
# READ (all at ridge 100, `--readout-mode distributed --readout-l2 100`):
#   - full macro AUC vs 0115 (0.7898) and 0113 (0.8087). Paired AB via inspect-topics
#     COMPARE=115 and COMPARE=113 (no credited split: no profile is fed).
#   - own-bg and family-closure masks, to pair with 0115's ladder (0.731 / 0.785 / 0.790).
#   - starvation % from the fit log (0114: 1%) — raw counts must not re-starve depth.
# OUTCOMES:
#   - 0123 ≈ 0113 (tax gone): binary counts WERE the tax; spectral is free at a converged
#     head and 0115's −0.019 is the count representation. Drop binary; anchors on raw.
#   - 0123 ≈ 0115 (tax stays): the tax is the anchors, not the counts. Binary is a neutral
#     choice for case-finding and the pregnancy fix is a topic-legibility win; the anchor
#     search is the lever either way.
#   - 0123 < 0115: binary was protecting the head from burst volume; keep it (or think
#     about a per-domain choice) and still fix the anchors.
#
# COST: identical to 0114/0115 (same bundle, K=1498, spectral seed ~1.5h + ~35 min fit on
# the lean cluster). Bundle is a HIT if the cluster has not restarted since 0115's; else
# rebuilds.
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


# 0123 — CV branch, spectral init, RAW counts (the counts arm)

0115's exact config with **`count_transform: none`** — raw per-visit counts, as 0114 and
0113 used them. 0114 is this config, but its run dir lost its λ and it was only ever read
at ridge 1 (0.7567, unconverged). This re-fit gives the raw-count spectral arm a
converged-head number so 0115's −0.019 can be split between the count representation and
spectral's anchor choice.

## What it does

`make -C analysis/cloud exp ID=123` → fit-only spectral-init gated LDA on the CV branch,
`diag_only` saves the fitted globals. Then the standard ridge-100 readout sweep.

## Acceptance criterion

A converged-head (ridge 100) macro AUC and a paired AB against BOTH 0115 (binary) and
0113 (random init, raw counts), plus the own-bg and family-closure ladder rungs. The read
is the three-way split in the front matter: tax gone / tax stays / worse.

## Run

One launch does everything: the fit-only fit, then the ridge-100 readout sweep, then the
two paired ABs. The run dir name is fixed per experiment and the fit tolerates a
pre-created dir, so the wrapper log can live there from the start. The readout driver
hard-exits after each step (`5205d0d`), so no `timeout` wrappers.

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/gated-conditional-voi && git checkout claude/gated-conditional-voi && git pull --ff-only
RUN=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0123-mondo-cardiovascular-tpn5-spectral-raw
mkdir -p "$RUN"
nohup bash -c '
  make -C analysis/cloud exp ID=123 || exit 1
  make -C analysis/cloud gated-pc-readout ID=123 GPR_ARGS="--readout-mode distributed --readout-l2 100"
  for m in own-bg family-closure; do
    make -C analysis/cloud gated-pc-readout ID=123 GPR_ARGS="--readout-mode distributed --readout-l2 100 --readout-feature-mask $m"
  done
  make -C analysis/cloud inspect-topics ID=123 COMPARE=115 INSPECT_ARGS="--readout-auc"
  make -C analysis/cloud inspect-topics ID=123 COMPARE=113 INSPECT_ARGS="--readout-auc"
' > "$RUN"/sweep_log.md 2>&1 &
```

Progress: `tail -3 "$RUN"/sweep_log.md` (the fit also tees to `"$RUN"/driver_log.md`).
Expect ~2h for seed + fit, then ~15 min per readout step.

Pull the numbers with:

```bash
grep -E "gated_pc(_own_bg|_family_closure)? \(pc_topics_lr\): (macro|detection)|^all: n=|^by depth|starved" "$RUN"/sweep_log.md
```

## Run log

**2026-09-11 — first launch killed at iter ~20.** Copied 0115's `optimize_doc_concentration:
true`, which was inert for 0115 but live since 0121's wiring: learned tied alpha was
collapsing (mean 0.5 → 0.21 by iter 19, background 0.28 → 0.27 → 0.27 per iter). Not a
clean A/B; killed and relaunched with `optimize_doc_concentration: false` (alpha fixed
0.5, 0115's effective setting).

**2026-09-12 — relaunch (alpha fixed 0.5) fit + full readout at ridge 100.** The chain
then wedged on the full readout's SparkSubmit JVM (Dataproc metrics publisher, non-daemon
thread in a socket write; see the handoff's cluster facts) — the readout numbers were
already on disk. Remaining steps relaunched `timeout`-wrapped into `sweep2_log.md`
(own-bg START 04:04); the driver-side fix (`hard_exit`) landed after that relaunch. 258/259 heads
converged (186 gtol, 73 stalled; same shape as 0115's 257/259).

## Results

**Full head, ridge 100 (pc_topics_lr, 193 nodes):**

| | macro AUC | AP | detection AUC | det AP |
|---|--:|--:|--:|--:|
| 0113 random init, raw counts | 0.8087 | 0.5451 | 0.6043 | — |
| 0115 spectral, binary | 0.7898 | 0.5263 | 0.6042 | 0.7272 |
| **0123 spectral, raw** | **0.7946** | **0.5327** | **0.6299** | **0.7600** |

Node-macro P@R0.5/0.8/0.9 = 0.517/0.395/0.357; R@FDR0.1/0.25/0.5 = 0.254/0.395/0.551
(0115: 0.513/0.390/0.351; 0.247/0.373/0.528).

**Ablation ladder at ridge 100 (own-bg landed; family-closure pending):**

| head may load on | 0123 (spectral, raw) | 0115 (spectral, binary) | 0116 (random, unfed) |
|---|--:|--:|--:|
| own block + background (`own-bg`) | **0.7298** (AP 0.482; det 0.543) | 0.7309 (det 0.497) | 0.669 |
| + ancestors (`family-closure`) | pending | 0.7851 | 0.777 |
| everything | 0.7946 (det 0.630) | 0.7898 (det 0.604) | 0.810 |

own-bg is unchanged by the count representation (0.730 vs 0.731): a fed block carries
its own node's signal, and how the counts are represented has nothing to do with
whether it does. The gap own-bg → full is 0.065 here vs 0.059 on binary — the raw counts'
+0.005 on the full head comes from OUTSIDE the own block (ancestors / the rest), and the
detection gain shows up even at own-bg (0.543 vs 0.497). Nothing about the block-unit
conclusion from 0115 changes.

**Read (full head + own-bg; family-closure + paired ABs pending).** Putting the raw counts back
recovers about a quarter of 0115's cost: macro +0.005 vs binary, and the spectral arm now
sits −0.014 under 0113 instead of −0.019. So the split is roughly **0.005 to the count
representation, 0.014 to spectral's anchor choice** — the "tax stays" outcome, mostly.
Binary counts were not free, but the anchors are the larger piece and the lever.

The bigger move is **detection: 0.604 → 0.630** (AP 0.727 → 0.760). Case-vs-background
separation is where repetition carries meaning — a code recorded at every visit for years
IS a different patient from one recorded once — and binarizing threw that away. This is
the user's prior on binary ("some counts do have meaning") landing in the number. The
per-node ranking, which contrasts a node against its siblings inside the parent's
closure, cares much less (+0.005): siblings share the utilization stratum.

