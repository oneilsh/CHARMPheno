---
id: 126
slug: mondo-cardiovascular-tpn5-spectral-guided-reverse
status: pending
model_class: gated_pc
cohort: population_mondo_all
cohort_def: population_mondo_all
disease: rare_priority
# LEAVES-FIRST DEFLATION. 0125's config VERBATIM with ONE line changed:
# spectral_topo_order: reverse (spec 2026-07-23, built July, never run on this branch).
#
# WHY. 0124/0125 showed the prenatal stratum in dilated cardiomyopathy's block is NOT
# brought in by its anchor (0125 changed the anchors, the block did not move). It is in
# DCM's TRAINING DOCUMENTS: peripartum cardiomyopathy is DCM's child, so every peripartum
# patient trains DCM's block under anchor_scope: closure, and FORWARD deflation removes
# only ancestors' directions — a child's signal survives into the parent's search and
# whichever vertex is nearest collects it in recovery. In 0123 the same stratum sat in
# intrinsic CM, one level up, for the same reason. REVERSE order processes leaves first:
# peripartum CM anchors on pregnancy, and DCM is deflated against its descendants'
# anchors before its own search; the parent's block becomes "what is left after the
# children claim theirs".
#
# THE A/B. Ancestors become residuals. The d3 cardiomyopathy block the guide fixed in
# 0124 (textbook: Cardiomyopathy · Dilated CM · Heart failure · LBBB · Cardiomegaly) is
# deflated here against intrinsic/extrinsic/familial/Tako-tsubo CM — it may thin out or
# turn generic. That is the cost to weigh against DCM (and every parent of a pregnancy-
# heavy or otherwise stratum-heavy child) coming clean. Guided anchors are unchanged.
#
# READ (priority order):
#   1. dilated cardiomyopathy's block (digest): is the prenatal stratum gone? where did
#      it go (peripartum CM's block, where it belongs)?
#   2. cardiomyopathy d3 and intrinsic CM d4: still legible, or residual junk?
#   3. anchors (spectral_anchors.json) for the same nodes.
#   4. guardrail: ridge-100 macro AUC vs 0125 (0.7914) / 0123 (0.7946); detection;
#      own-bg vs 0.723; paired AB vs 0125 by depth — with ancestors deflated against
#      children, the shallow levels are where a cost would show.
# ACCEPTANCE: (1) yes AND (2) not worse AND (4) within ~0.005. If (1) yes but (2)
# degrades, the structural answer is a two-pass seed (forward anchors, then re-recover
# each node deflated against ancestors AND powered descendants) — spec before build.
#
# COST: identical to 0125 (~1.5h seed + 35 min fit). Tables must be regenerated on a
# fresh cluster (survey offline; probe needs the bundle HIT).
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
spectral_topo_order: reverse  # THE ONLY CHANGE vs 0125: leaves-first; each node deflated against its DESCENDANTS' anchors
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


# 0126 — 0125 with leaves-first deflation (`spectral_topo_order: reverse`)

0125's exact config and table, one line changed. See the front matter for why.

## What it does

`make -C analysis/cloud exp ID=126` → fit-only spectral-init gated LDA with guided
anchors, nodes recovered leaves-first and each deflated against its descendants' anchors;
then the ridge-100 readout sweep, paired ABs vs 0125 / 0123 / 0113, the named digest,
and the named anchors.

## Acceptance criterion

DCM's block without the prenatal stratum (which should now sit in peripartum CM's block),
the d3/d4 cardiomyopathy blocks no worse than 0125, macro AUC within ~0.005 of 0125.

## Run

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/gated-conditional-voi && git checkout claude/gated-conditional-voi && git pull --ff-only
RUN=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0126-mondo-cardiovascular-tpn5-spectral-guided-reverse
mkdir -p "$RUN"
nohup bash -c '
  echo "=== tables START $(date)"
  make -C analysis/cloud hpoa-profile-survey HPOA_ARGS="--emit-codes $HOME/repos/CHARMPheno/data/ontology/profile_codes_MONDO_0004995.tsv --rollup" || exit 1
  make -C analysis/cloud hpoa-stage2-probe ID=125 GPR_ARGS="--emit-eta $HOME/repos/CHARMPheno/data/ontology/profile_eta_MONDO_0004995.tsv" || exit 1
  echo "=== fit START $(date)"
  make -C analysis/cloud exp ID=126 || exit 1
  echo "=== readout START $(date)"
  make -C analysis/cloud gated-pc-readout ID=126 GPR_ARGS="--readout-mode distributed --readout-l2 100"
  for m in own-bg family-closure; do
    make -C analysis/cloud gated-pc-readout ID=126 GPR_ARGS="--readout-mode distributed --readout-l2 100 --readout-feature-mask $m"
  done
  for b in 125 123 113; do make -C analysis/cloud inspect-topics ID=126 COMPARE=$b INSPECT_ARGS="--readout-auc"; done
  make -C analysis/cloud inspect-topics ID=126 RESOLVE_NAMES=1 INSPECT_ARGS="--digest --redundancy 30 --grep \"cardiomyopathy|atrial fibrillation|heart valve|mitral|aortic|myocardial infarction|heart failure|pregnan|gestation\""
  python3 analysis/cloud/name_spectral_anchors.py "'"$RUN"'" --bundle-meta /tmp/inspect_meta_126.json --concept-names /tmp/concept_names_126.csv --grep "dilated cardiomyopathy|intrinsic cardiomyopathy|^cardiomyopathy$|pregnancy-induced|peripartum|toxemia"
  echo "=== DONE $(date)"
' > "$RUN"/sweep_log.md 2>&1 &
```

Pull the numbers with:

```bash
grep -E "anchor guide|^=== |gated_pc(_own_bg|_family_closure)? \(pc_topics_lr\): (macro|detection)|^## paired|^all: n=|^by depth|starved" "$RUN"/sweep_log.md
grep -nE "cardiomyopathy|pregnan|toxemia|heart failure" "$RUN"/sweep_log.md | grep -E "fed \||\| " | cut -c1-330 | head -40
```

## Run log

## Results
