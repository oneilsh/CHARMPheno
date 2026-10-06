---
id: 128
slug: cardiomyopathy-subtree-guided-reverse
status: done
model_class: gated_pc
cohort: population_mondo_all
cohort_def: population_mondo_all
disease: rare_priority
# PROBE PAIR, ARM B (reverse). 0127's config with ONE line changed:
# spectral_topo_order: reverse — leaves first, each node deflated against its
# descendants' anchors (spec 2026-07-23). Peripartum CM anchors on pregnancy before DCM
# searches; DCM's block = what is left after its children claim theirs.
#
# WHY A SUBTREE. The 0126 question (does leaves-first deflation take the prenatal
# stratum out of dilated cardiomyopathy's block, and what does it do to the parents?)
# lives entirely inside the cardiomyopathy subtree. On the full CV branch one arm costs
# ~3h, ~1.5h of it the seed walking 298 nodes in batched passes over the corpus. The
# subtree as its own label DAG is ~30 nodes: seed in minutes, fit in minutes, readout in
# minutes — BOTH orders in about an hour, including one bundle rebuild for the new
# branch. What is given up: a different label DAG (no heart disorder / cardiovascular
# disorder ancestors to deflate against; cardiomyopathy becomes the branch's d1), so
# numbers do not pair with 0125's. For the mechanism under test — peripartum patients
# training DCM's block, and parents becoming residuals under reverse order — the
# subtree is a faithful probe. The full-branch run (0126) is for the winning order only.
# The CV profile table is reused as-is: nodes outside this DAG are skipped and counted.
#
# READ: 1. DCM's block (digest) — prenatal stratum gone? in peripartum CM's block now?
#       2. cardiomyopathy (d1 here) and intrinsic CM: legible, or residual junk?
#       3. anchors for the same nodes. 4. paired AB vs 0127 (same DAG, like for like).
# ACCEPTANCE: (1) yes, (2) not worse than 0127, (4) within ~0.005 → run 0126 (full
# branch, reverse). (1) yes but (2) degrades → spec the two-pass seed (forward anchors,
# then re-recover each node deflated against ancestors AND powered descendants).
dag_source: mondo_native
mondo_branch: MONDO:0004994   # cardiomyopathy SUBTREE (was MONDO:0004995, the whole CV branch): ~30 nodes, minutes not hours
tpn: 5
max_iter: 50
diag_only: true
# --- the ONLY change from 0113: spectral block-aligned seed for the gated engine ---
init: spectral
spectral_method: scalable   # concatenated V ~11.6k >= 8000 threshold; dense = driver wall
spectral_d: 768             # random-projection dim: smaller = faster + bigger safe batch (see COST)
anchor_scope: closure       # node trained from its whole closure; ancestors deflated by topo order
spectral_topo_order: reverse  # THE ONLY CHANGE vs 0127: leaves-first; each node deflated against its DESCENDANTS' anchors
count_transform: none       # raw per-visit counts, as 0123 (binary was a detection tax, 0123)
# --- THE ONLY CHANGE vs 0123: HPO-guided spectral anchors (spec 2026-10-05) ---
spectral_anchor_profile: ~/repos/CHARMPheno/data/ontology/profile_eta_MONDO_0004995.tsv  # REGENERATED with the prenatal exclusion (see COST)
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


# 0128 — cardiomyopathy subtree, guided anchors, LEAVES-FIRST deflation (probe arm)

See the front matter.

## What it does

Same as 0127 with `spectral_topo_order: reverse`. Bundle is a HIT after 0127.

## Acceptance criterion

DCM's block without the prenatal stratum; cardiomyopathy / intrinsic CM no worse than
0127; paired macro AUC vs 0127 within ~0.005.

## Run (BOTH arms, one chain; ~1h)

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/gated-conditional-voi && git checkout claude/gated-conditional-voi && git pull --ff-only
RUNS=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs
mkdir -p "$RUNS"/0127-cardiomyopathy-subtree-guided-forward "$RUNS"/0128-cardiomyopathy-subtree-guided-reverse
G='cardiomyopathy|myocarditis|pregnan|gestation|heart failure'
nohup bash -c '
  echo "=== tables START $(date)"
  make -C analysis/cloud hpoa-profile-survey HPOA_ARGS="--emit-codes $HOME/repos/CHARMPheno/data/ontology/profile_codes_MONDO_0004995.tsv --rollup" || exit 1
  make -C analysis/cloud hpoa-stage2-probe ID=125 GPR_ARGS="--emit-eta $HOME/repos/CHARMPheno/data/ontology/profile_eta_MONDO_0004995.tsv" || exit 1
  for ID in 127 128; do
    RUN=$(ls -d '"$RUNS"'/0${ID}-*)
    echo "=== $ID fit START $(date)"
    make -C analysis/cloud exp ID=$ID || exit 1
    echo "=== $ID readout START $(date)"
    make -C analysis/cloud gated-pc-readout ID=$ID GPR_ARGS="--readout-mode distributed --readout-l2 100"
    [ $ID = 128 ] && make -C analysis/cloud inspect-topics ID=128 COMPARE=127 INSPECT_ARGS="--readout-auc"
    make -C analysis/cloud inspect-topics ID=$ID RESOLVE_NAMES=1 INSPECT_ARGS="--digest --redundancy 30 --grep \"'"$G"'\""
    python3 analysis/cloud/name_spectral_anchors.py "$RUN" --bundle-meta /tmp/inspect_meta_$ID.json --concept-names /tmp/concept_names_$ID.csv
    echo "=== $ID DONE $(date)"
  done
' > "$RUNS"/0128-cardiomyopathy-subtree-guided-reverse/pair_log.md 2>&1 &
```

Pull the reads with:

```bash
L=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0128-cardiomyopathy-subtree-guided-reverse/pair_log.md
grep -E "^=== |anchor guide|gated_pc \(pc_topics_lr\): (macro|detection)|^all: n=|^by depth|starved|K=" "$L"
grep -nE "cardiomyopathy|pregnan|toxemia|myocarditis" "$L" | grep -E "fed \|| \| " | cut -c1-300
```

## Run log

## Results (2026-10-06; 17 nodes, K=93, 7 scored; 12 guided, 56/85 anchors from profile)

| arm | macro AUC (7) | detection | DCM block | parent (cardiomyopathy d1) |
|---|--:|--:|---|---|
| 0127 forward | 0.9123 | 0.713 | 4 of 5 topics pregnancy strata | textbook topic ev 3.2e5 |
| 0128 reverse | 0.9069 | 0.709 | 2 of 5 pregnancy; ONE real DCM topic appears (ev 3.4e4) | textbook topic ev 9.1e4; pregnancy now intrinsic CM's TOP topic |

Paired 0128−0127: median −0.0004 (2/5 up); d2 −0.023 (n=3), d3 +0.008 (n=4).

**Verdict: leaves-first deflation does NOT remove the stratum; it spreads it** (intrinsic
CM's top topic, still DCM, less in peripartum) and thins the parents as predicted.
Mechanism: deflation removes only the directions the child's ANCHORS span, and
peripartum's anchors (toxic goiter, T1DM, hypothyroidism, RF positive) do not span the
pregnancy direction — rare-code anchors defeat deflation as surely as they defeat
legibility. **The lever is the candidate floor** (df ≥ 5 admits rare syndromic codes as
vertices): exps 0129/0130 apply the dense path's rule — at least as common as the node's
average word — to the guided pool / to every candidate. 0126 (full branch, reverse) is
NOT run.
