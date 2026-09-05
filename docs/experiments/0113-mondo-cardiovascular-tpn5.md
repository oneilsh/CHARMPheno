---
id: 113
slug: mondo-cardiovascular-tpn5
status: pending
model_class: gated_pc
cohort: population_mondo_all
cohort_def: population_mondo_all
disease: rare_priority
# THE TOPIC-BUDGET TEST (insight 0079 tpn-confound). 0110's native-Mondo config with
# THREE knobs changed: restrict the label DAG to ONE body-system branch
# (cardiovascular disorder, MONDO:0004995), raise the per-node topic budget tpn 1 -> 5,
# and run FIT-ONLY (diag_only: no readout/eval — we only need the fitted lambda to
# inspect topic evidence by depth). max_iter 100 -> 50 (enough for topic-word mass to
# resolve; we are not chasing a converged head).
#
# WHY. 0111 (whole-Mondo, tpn=1) starves deep node topics (insight 0079); but the
# starvation is CONFOUNDED with topic budget. Every prior deep-recovery SUCCESS used a
# generous budget — rare6/diabetes tpn=5 (exps 0052-0069), EDS a 20-topic foreground
# block (insight 0035, which recovered POTS/MCAS/vascular-EDS/GI subphenotypes) — while
# whole-Mondo dropped to tpn=1 for compute (tpn=5 at 2714 nodes is K~13,600). A one-topic
# node cannot out-compete its sharp ancestor stack to express its residual; with five it
# has room. This run gives one body-system branch (~273 CV nodes -> K ~ 8 + 273*5 ~ 1,400,
# cheap) a tpn=5 budget and asks whether the deep CV nodes that starve in 0111 (arrhythmia
# / cardiomyopathy / valve subtypes) get FED. It doubles as the insight-0071 cascade
# prototype (a per-body-system branch fit).
#
# READ: inspect_topics.py on the saved fit (off-cluster) — deep-node evidence-by-depth,
# --tour, --redundancy, and --grep the CV subtypes that were STARVED in 0111 (ev~5,
# frac~1.0). Acceptance: does the depth->=5 CV evidence floor LIFT under tpn=5 vs 0111's
# tpn=1? A lift => topic budget is the lever (=> the cascade at tpn=5). No lift => budget
# is not it, look to deflation/strip-scope/init next.
#
# CAVEAT (index): 0110/0113 use the STANDARD lookback index; 0111 (the tpn=1 reference we
# can still inspect) is EPISODE-indexed. Both feed the topics, so the index is a
# second-order confound for the topic-FEEDING question. For an airtight tpn isolation, a
# tpn=1 companion on this same branch/index (a second fit-only 50-iter run, K~280) is the
# clean A/B — recommended as a follow-on, not required for the first read.
#
# NATIVE-MONDO NOTES (from 0110): dag_source=mondo_native + mondo_branch restricts the
# kept label set to the CV subtree (mondo_native_dag.build branch_root); the CORPUS stays
# all-patients (non-CV patients are background-only docs), so the documents match 0110's.
# preindex_closure OFF: fit-only needs no incident census / R_d, and skipping it saves a
# full-history scan at build time. dag_collapse is intrinsic to the native build.
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

# 0113 — cardiovascular branch at tpn=5 (the topic-budget test)

Isolates the `tpn`-budget confound behind the whole-Mondo depth-starvation (insight
[0079](../insights/0079-gated-pc-topics-starve-at-a-hard-depth-5-cliff-label-coverage-is-not-topic-learnability.md)):
fit one body-system branch (cardiovascular disorder, MONDO:0004995) from the current
native-Mondo + multidomain corpus at **`tpn=5`** (five topics per node, vs 0111's one),
**fit-only** (`diag_only`, no readout), 50 iters, and inspect whether the deep CV node
topics that starve in 0111 get fed when the budget is generous.

**Grounding.** Every prior deep-recovery success used a generous per-node budget — rare6 /
diabetes `tpn=5` (exps 0052–0069), EDS a 20-topic foreground block (insight
[0035](../insights/0035-rare-disease-gated-foreground-recovers-eds-subphenotypes-on-full-population.md),
which recovered POTS / MCAS / vascular-EDS / GI subphenotypes) — while whole-Mondo dropped
to `tpn=1` for compute. So "recovers vs starves" is confounded with budget. This run
de-confounds it on one branch and doubles as the insight-0071 per-body-system **cascade
prototype**.

## What it does

`make -C analysis/cloud exp ID=113` → fit-only gated LDA on the CV-restricted native-Mondo
label DAG at `tpn=5`. K is emergent (`n_bg + kept_CV_nodes × 5`; the build prints C and K —
expect ~273 nodes → K ≈ 1,400). `diag_only` saves the fitted globals
(`gated_pc_result.npz`, the fit-only λ) and the per-node head-distribution report, then
returns before any θ-collect / readout / baseline.

## What to read (off-cluster, `inspect_topics.py`)

Pull the fit + the branch's bundle meta (off-YARN `hdfs dfs -cat …/meta/part-*`) and run:

- `--tour 2` — the CV tree top-to-bottom, all domains: do arrhythmia / cardiomyopathy /
  valve **subtypes** carry sharp, distinct signal at depth, or still flatten?
- `--redundancy 30` — do the fed CV siblings differentiate (they did whole-Mondo)?
- `--grep 'cardiomyopathy|arrhythmia|heart valve|myocardial infarction|…'` — look up the
  specific CV subtypes and compare their **evidence / frac** to the SAME nodes in 0111
  (`--grep` on 0111, `tpn=1`).

**Acceptance criterion.** Does the depth-≥5 CV node **evidence floor LIFT** under `tpn=5`
vs 0111's `tpn=1`?
- **Lift** (deep CV subtypes go from ev≈5 / frac≈1.0 to sharp, fed) → topic budget is the
  lever → the cascade should run at `tpn≥5` per branch, and the whole-Mondo monolith
  under-fed depth purely for compute.
- **No lift** → budget is not the binding constraint; return to deflation / strip-scope /
  init as the candidate mechanisms (0079).

## Run

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/gated-conditional-voi && git checkout claude/gated-conditional-voi && git pull --ff-only
make -C analysis/cloud exp ID=113
```

Then, off-cluster (safe any time):

```bash
hdfs dfs -ls hdfs:///user/dataproc/charm/case_finding_cache          # find the new CV bundle key
make -C analysis/cloud inspect-topics ID=113 INSPECT_KEY=<key> RESOLVE_NAMES=1 \
    INSPECT_ARGS="--tour 2 --redundancy 30"
```

## Run log

*(pending)*

## Results

*(pending; model params / counts-of-nodes only, egress floor)*
