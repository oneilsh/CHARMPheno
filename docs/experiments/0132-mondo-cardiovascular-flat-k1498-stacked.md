---
id: 132
slug: mondo-cardiovascular-flat-k1498-stacked
status: done
model_class: gated_pc
cohort: population_mondo_all
cohort_def: population_mondo_all
disease: rare_priority
# THE HSLDA-LIKE ARM (spec 2026-10-06 Part B). 0113's config with the GATE OFF: one flat
# unsupervised LDA at K=1498 — the gated run's own K (8 + 298 x 5), the non-arbitrary
# choice — random init, same corpus, same split, same ridge-100 readout, then the STACKED
# readout (closure product of the per-node heads with a root head fit on every row).
# This is what Perotte et al. 2011 actually did: flat shared topics, one logistic per
# label, label-side gating (a child fires only if its parent fires). Nothing on the
# topic side knows about Mondo; the heads do all the mapping.
#
# WHY. Ten gated runs (0123-0131) say a per-node block is a five-way split of its seed
# docs into co-occurrence strata, and nothing done to anchors or deflation changes that.
# The hierarchy-on-the-topic-side design has been pushed as far as it goes; the question
# is whether the hierarchy-on-the-LABEL-side design (which we built once as DagClosureHead
# and never read on real data) does the job with topics that are strata BY CONSTRUCTION.
#
# WHAT TO EXPECT ON THE TOPIC SIDE. Flat topics at K=1498 will be patient strata — there
# is no block to be "the dilated cardiomyopathy topic". The user's concern ("on full-Mondo
# we'd have a lot of LDA topics to wrangle") is answered by the READ, not the fit: the
# digest / tour are not the deliverable here; the heads are. A node is interpretable
# through its head's top loadings (inspect-topics --top-loadings), i.e. which strata a
# node's conditional reads, rather than through "its" topic. If that read is legible the
# wrangling problem is solved by the head; if not, that is the finding.
#
# READ (ridge 100, `--readout-mode distributed --readout-l2 100`):
#   - flat readout macro AUC / detection vs 0123 (0.7946 / det 0.630) and 0113 (0.8087 /
#     0.604): paired AB via inspect-topics COMPARE=123 and COMPARE=113 (--readout-auc only;
#     the digest has no node blocks to report on: every topic is background).
#   - the STACKED readout (--readout-stacked): detection three ways (flat max / root head
#     alone / stacked max), paired per-node delta by depth, marginal ECE by depth — read
#     NEXT TO 0123's stacked readout (same tool on the gated fit).
#   - cost: wall of the fit (no spectral seed; E-step over all K per doc).
# OUTCOMES:
#   - flat+stacked ≈ gated+stacked on ranking AND detection: the gate's per-node blocks
#     buy nothing a flat LDA + closure heads does not; the representation question moves
#     to the heads (tpn is moot). Interpretability then lives in the head loadings.
#   - gated+stacked > flat+stacked: the blocks carry node signal the heads cannot recover
#     from strata; keep the gate, ask tpn=1 vs tpn=5 under the stacked head next.
#   - flat > gated on ranking but not detection (or vice versa): the two heads disagree
#     about what the topic side owes them; read the per-depth split before deciding.
#
# LANDMINE: 0113 carried optimize_doc_concentration: true when it was inert; it has
# been LIVE since 0121 and collapses alpha. OFF here (alpha fixed 0.5, 0123's setting).
# COST: bundle HIT on a warm cluster (same key as 0113/0123); fit ~1h (50 iters, no
# seed); each readout ~15 min.
dag_source: mondo_native
mondo_branch: MONDO:0004995
# THE CHANGE — a FLAT layout through the gated engine: every node's block is EMPTY
# (tpn: 0) and the background is the whole topic range (n_bg: 1498 = 0123's K, 8 +
# 298 x 5). Every document's allowed set is then all K topics = plain online LDA, on the
# same multi-domain corpus, saved in the same format, re-readable by the same tools.
# (The driver's only other ungated path, --with-dag-head, never saves its globals and
# the estimator refuses multi-domain feature columns without a gate; tpn=0 needs no
# driver change — see spark-vi/tests/test_flat_layout_tpn0.py.)
tpn: 0
n_bg: 1498
max_iter: 50
diag_only: true
init: random                # the spectral seed is per-block; there are no blocks
alpha_init: uniform         # the equalized init has no blocks to equalize (returns uniform)
count_transform: none
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
optimize_doc_concentration: false   # LIVE since 0121 — see LANDMINE
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
  # 0113/0123's geometry. With empty blocks the E-step touches all K per doc (a gated
  # doc sees only its closure's blocks), so expect a slower iteration than 0123's fit —
  # but no 1.5h spectral seed.
  spark.executor.cores: 2
  spark.executor.memory: 8g
  spark.executor.memoryOverhead: 3g
  spark.dynamicAllocation.enabled: "false"
  spark.executor.instances: 20
  spark.excludeOnFailure.timeout: 10m
  spark.cleaner.periodicGC.interval: "5min"
---

# 0132 — cardiovascular branch, FLAT topics at K=1498 + stacked closure heads (HSLDA-like)

The output-side-hierarchy arm (spec
[2026-10-06](../superpowers/specs/2026-10-06-stacked-closure-readout-and-hslda-arm.md)
Part B): 0113's corpus and split, one flat LDA at the gated run's K, random init, and the
same two readouts 0123 gets — the flat ridge-100 heads and the stacked closure product
with a root head fit on every row. Part A (the stacked readout on the saved 0123/0124
gated fits) is the paired control: same tool, same corpus, gated vs flat topic side.

## What it does

`make -C analysis/cloud exp ID=132` → fit-only flat LDA through the gated engine with
empty node blocks (`gated_pc_result.npz` at K=1498, manifest `n_bg: 1498, tpn: 0`), then
the ridge-100 readout (record + the two ABs), then the stacked readout. `inspect-topics`
on this fit labels every topic as background — correct, there are no node blocks.

## Acceptance criterion

Two numbers side by side with 0123's: the flat-head macro AUC / detection, and the stacked
block (detection three ways, paired per-depth ranking delta vs its own flat heads, marginal
ECE). The read is the three-way split in the front matter.

## Run

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/gated-conditional-voi && git checkout claude/gated-conditional-voi && git pull --ff-only
RUN=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0132-mondo-cardiovascular-flat-k1498-stacked
mkdir -p "$RUN"
nohup bash -c '
  make -C analysis/cloud exp ID=132 || exit 1
  echo "=== readout START $(date)"
  make -C analysis/cloud gated-pc-readout ID=132 GPR_ARGS="--readout-mode distributed --readout-l2 100"
  echo "=== stacked START $(date)"
  make -C analysis/cloud gated-pc-readout ID=132 GPR_ARGS="--readout-mode distributed --readout-l2 100 --readout-stacked"
  make -C analysis/cloud inspect-topics ID=132 COMPARE=123 INSPECT_ARGS="--readout-auc"
  make -C analysis/cloud inspect-topics ID=132 COMPARE=113 INSPECT_ARGS="--readout-auc"
  echo "=== DONE $(date)"
' > "$RUN"/sweep_log.md 2>&1 &
```

Early check: `grep -E "corpus: V=" "$RUN"/sweep_log.md` — the driver must print
`K=1498 gated topics (1498 bg + 298 nodes x 0 tpn)` before the fit starts.

Pull the numbers with:

```bash
grep -E "^=== |corpus: V=|gated_pc(_stacked)? \(pc_topics_lr\): (macro|detection)|STACKED|three reads|flat: max|root head alone|stacked: max|ranking \(within|paired per-node|depth [0-9]+:|marginal ECE|^all: n=|^by depth" "$RUN"/sweep_log.md
```

## Run log

**2026-10-06 — fit + both readouts + AB, first try (bundle HIT).** The corpus line read
`K=1498 gated topics (1498 bg + 298 nodes x 0 tpn)` as required. Fit ~2 min/iter (the
E-step touches all 1498 topics per document where a gated document sees a few dozen),
~100 min for 50 iters; λ mass concentrated fast from a random start (iter 5: heaviest
condition topic 5.4e5 vs lightest 6.7e3). Stacked readout 260/260 heads converged.

**Alive topics (off the saved λ, data mass per topic above the η prior):** all 1498 alive,
none under 10× prior; 50 / 90 / 99 % of the mass in 263 / 1,248 / 1,473 topics. The early
concentration did not persist: the flat fit used ~1,250 strata, not a few hundred. A
blend arm's background cannot be cut to a few hundred topics without losing mass.

## Results

**Three fits under the same two decoders (ridge 100; 193 within-cohort / 224 de-novo
nodes; 54,753 test persons). 0123 = gated tpn=5 spectral-raw, 0124 = 0123 + guided
anchors, 0132 = flat K=1498 (this run):**

| read | 0123 gated | 0124 gated+guided | **0132 flat** |
|---|--:|--:|--:|
| flat head, within-cohort macro AUC / AP | 0.7946 / 0.533 | 0.7920 / 0.526 | **0.7612 / 0.476** |
| flat head, detection (max over nodes) | 0.630 | 0.633 | 0.484 |
| root head alone (= stacked detection) | 0.8065 / AP 0.886 | 0.8040 / 0.884 | 0.7983 / 0.881 |
| stacked, within-cohort macro AUC | 0.7209 | 0.7201 | 0.7005 |
| stacked, de-novo per-node macro AUC / AP | 0.8497 / 0.236 | 0.8483 / 0.231 | **0.8362 / 0.185** |
| de-novo paired (stacked − own flat) | +0.116, 219/4 | +0.106, 221/2 | +0.118, 220/3 |

Paired per-node, 0132 minus 0123 on the within-cohort flat heads (193 shared): median
**−0.028**, 33 up / 160 down, and the same at every depth (d2 −0.012, d3 −0.020, d4
−0.028, d5 −0.027, d6 −0.049, d7 −0.030). Within-cohort AUC by depth on 0132: 0.81 /
0.78 / 0.76 / 0.77 / 0.76 / 0.69 (d2–d7).

**Read.** The gate buys a modest, uniform amount of node signal that flat strata do not
carry: +0.03 within-cohort, +0.014 de-novo AUC, and the largest gap is de-novo AP, 0.236
vs 0.185 (+28% relative) — precision at the top of the list, the case-finding operating
point. It is not the "flat matches gated" outcome; it is also not a large gap: with no
topic knowing the DAG, the stacked head still reaches 0.836 de novo and the root head
0.798, within a hundredth of the gated fits. The root head is nearly indifferent to the
topic side (0.806 / 0.804 / 0.798): case-vs-background lives in the strata.

Two things the flat fit says on its own: (1) the conditionals read as marginals are
BELOW chance (0.484 — 39 constant columns, and flat strata give a within-cohort head
nothing that transfers outside its cohort), so the max-over-nodes detection number is
meaningless on any fit, not just a bad read on the gated ones; (2) the de-novo gain from
stacking is the same size on every topic side (+0.106 to +0.118) — the label-side
hierarchy is worth the same whatever the topics are.

**Outcome among the three in the front matter:** the second one, softened. The blocks
carry node signal the heads cannot fully recover from strata; the gate stays; tpn=1 vs
tpn=5 is the next pair under the stacked head (exp 0133). The topic side's job is now
clear: supply what the strata lack (the +0.03 / +0.05 AP), as legibly as possible.
