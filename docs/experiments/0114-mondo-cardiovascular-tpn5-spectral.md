---
id: 114
slug: mondo-cardiovascular-tpn5-spectral
status: pending
model_class: gated_pc
cohort: population_mondo_all
cohort_def: population_mondo_all
disease: rare_priority
# THE INIT TEST (insight 0079 flat-start trap). 0113's config VERBATIM with ONE knob
# flipped: init random -> spectral. Everything else (CV branch MONDO:0004995, tpn=5,
# fit-only, 50 iters, standard lookback index, same corpus/bundle) is identical, so
# 0114-vs-0113 is a CLEAN A/B on the seed alone.
#
# WHY. 0113 falsified topic BUDGET as the deep-node rescue lever: at tpn=5, 72% of node
# topics still starved (frac>0.5), the depth-4 evidence floor unmoved (~60.6 = prior
# floor). The remaining pre-registered candidate is the FLAT-START TRAP: a leaf topic
# initialized UNIFORM has no concentration anywhere, so on the first pass its own rare
# distinctive tokens — where its ancestors are WEAK (a rare token is rare in the
# ancestor's whole doc pool) — are still not won, and, being rare, the leaf's block is
# barely sampled (ADR-0027 lazy blocks). The opportunity is in the leaf's own docs; the
# model just can't bootstrap it from flat. SPECTRAL init hands each block initial
# concentration from block-aligned anchor-word recovery (gated_init.py; Arora et al.
# 2013), deflated down the DAG so a node's seed is its INCREMENT over its ancestors —
# exactly the concentration the flat start denies it.
#
# NOT budget (0113), NOT prevalence/redundancy/n_bg/PC (0079/0080/0081). This isolates
# init at whole-branch depth, the regime 0063's null (a shallow 170-block node-affinity
# DAG) did not reach — 0079's standing note that "init is genuinely untested at depth."
#
# CAVEAT the spectral machinery itself carries (gated_init.py docstring; insight 0067):
# on synthetic plants block-aligned spectral did NOT beat random ("the gate already
# breaks symmetry") and could regress shallow nodes. But that null is a DIFFERENT
# symmetry — the gate differentiates UNRELATED blocks; within a leaf's own docs the leaf
# still competes with its ANCESTORS, the deflation competition the gate leaves unbroken
# and the one that bites with depth. So a shallow-plant null does not transfer; this is
# the real-DAG-at-depth A/B the machinery was kept as the extension point for.
#
# READ: inspect_topics.py --digest on the saved fit (off-cluster), same as 0113:
#   - depth rollup + cliff marker: does the depth>=4 evidence floor LIFT vs 0113/0111?
#   - --grep the CV subtypes STARVED in 0113 (alcoholic/Tako-tsubo cardiomyopathy,
#     congenital valve defects, acute anterolateral MI): do they go from ev~61/frac~1.0
#     to sharp+fed?
#   - --redundancy: do the newly-fed siblings differentiate (not collapse)?
# Acceptance: a LIFT of the depth>=4 floor => the flat-start trap was the lever, spectral
# init is the deep-node fix. NO lift => init is not it either; the residual is strip-scope
# (0113's own-label variant leakage) + minibatch rarity, and the plain-K-LDA-on-one-subDAG
# test (0079) becomes the discriminator for whether uncoded signal exists at all.
#
# COST NOTE (scalable seed). Concatenated multi-domain V ~11.6k >= spectral_max_vocab
# (8000) -> SCALABLE routing (dense = driver V×V collect, the 0110 disk wall). The seed
# is BATCHED (gated_init.scalable_block_aligned_lambda): B nodes per projected-cooccurrence
# pass, batched within a depth level, ~n_nodes passes -> ~n_nodes/B, with per-batch
# progress logging. Two costs bound B, both now handled:
#   - DRIVER COLLECT. Each pass treeReduces ~sqrt(numPartitions) partials to the driver,
#     each holding all B+1 dense (V,d) float32 sketches; peak ~ sqrt(P)·(B+1)·V·d·4. B
#     AUTO-SIZES to spark.driver.maxResultSize (0.7 budget) so it cannot OOM — the first
#     batched attempt DID OOM (4.1 GiB > 4 GiB maxResultSize at B=6, V·d=11601·1498)
#     because the old sizing ignored the sqrt(P) fan-out. Override via
#     CHARM_SPECTRAL_BATCH=<n> (warns past the safe cap).
#   - PROJECTION DIM d. The scalable path places only ~tpn+|seed_rows| anchors PER NODE,
#     never K, so d need not clear K (the dense floor). spectral_d: 768 here (JL "safe
#     margin" ~1000; EM refines the seed, so a slightly smaller d is a speed lever, not a
#     quality cut) shrinks each (V,d) sketch ~2x vs the old K=1498 floor -> faster passes
#     AND a bigger safe B. Expected B≈6 at 768 / 4 GiB; ~50 passes, ~40-50 min on the lean
#     cluster. FASTER: spectral_d: 512 -> B≈10, ~30 passes; or raise maxResultSize if the
#     master has RAM. The seed is batch-size-INVARIANT (tuning B changes speed, not the
#     fitted seed); d changes the seed slightly but within EM's refinement.
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

# 0114 — cardiovascular branch at tpn=5 with SPECTRAL init (the flat-start test)

0113's exact config with **one knob flipped — `init: random` → `init: spectral`** — so
that a lift of the deep-node evidence floor is attributable to the seed alone. This is
the pre-registered next lever after 0113 falsified topic budget: the **flat-start /
deflation trap** (insight
[0079](../insights/0079-gated-pc-topics-starve-at-a-hard-depth-5-cliff-label-coverage-is-not-topic-learnability.md)).

**The mechanism being tested.** A uniformly-initialized leaf topic can win no token on
the first CAVI pass — including its own rare distinctive tokens, where its ancestors are
actually weak (a rare token is rare in the ancestor's whole document pool). With no
concentration to amplify, it never bootstraps, and being rare its block is barely
sampled (ADR-0027 lazy updates). The winning tokens sit in the leaf's own documents; the
model just can't start from flat. **Spectral** init seeds each block from block-aligned
anchor-word recovery (`gated_init.py`; Arora et al. 2013), deflated down the DAG so a
node's seed is its *increment over its ancestors* — exactly the concentration a flat
start denies it.

## What it does

`make -C analysis/cloud exp ID=114` → fit-only gated LDA on the CV-restricted
native-Mondo DAG at `tpn=5`, **seeded by the scalable block-aligned spectral init**
instead of random Gamma. K is emergent (`n_bg + kept_CV_nodes × 5 ≈ 1498`, same as
0113). `diag_only` saves the fitted globals and returns before readout/eval.

## What to read (off-cluster, `inspect_topics.py --digest`)

Same reads as 0113, compared node-for-node against it:

- depth rollup + auto cliff-marker — **does the depth-≥4 evidence floor LIFT** vs 0113's
  ~60.6 prior floor?
- `--grep` the CV subtypes that STARVED in 0113 (alcoholic / Tako-tsubo cardiomyopathy,
  congenital mitral/aortic valve defects, acute anterolateral MI): ev≈61 / frac≈1.0 → sharp?
- `--redundancy` — do newly-fed siblings differentiate, or collapse to one topic?

**Acceptance criterion.** Does spectral init LIFT the depth-≥4 CV evidence floor that
`tpn=5` (0113) did not?
- **Lift** → the flat-start trap was the binding constraint; spectral init is the
  deep-node fix, and it should carry into the whole-Mondo cascade (once the scalable
  seed's O(n_nodes)-passes cost is addressed).
- **No lift** → init is not the lever either. The residual candidates are strip-scope
  (0113's own-label OMOP-variant leakage — fed nodes lean on surviving variant codes)
  and minibatch rarity; the plain K≈80 LDA on one subDAG (0079) becomes the discriminator
  for whether recoverable uncoded signal exists at all.

## Run

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/gated-conditional-voi && git checkout claude/gated-conditional-voi && git pull --ff-only
make -C analysis/cloud exp ID=114
```

Then, off-cluster (reuse 0113's cached bundle meta — same branch/bundle):

```bash
make -C analysis/cloud inspect-topics ID=114 \
    INSPECT_META=/tmp/inspect_meta_113.json INSPECT_NAMES=/tmp/concept_names_113.csv \
    INSPECT_ARGS="--digest --redundancy 30 \
      --grep 'cardiomyopathy|arrhythmia|atrial fibrillation|heart valve|mitral|aortic|myocardial infarction|heart failure'"
```

## Run log

*(pending)*

## Results

*(pending; model params / counts-of-nodes only, egress floor)*
