---
id: 125
slug: mondo-cardiovascular-tpn5-spectral-guided-noprenatal
status: pending
model_class: gated_pc
cohort: population_mondo_all
cohort_def: population_mondo_all
disease: rare_priority
# 0124 WITH THE PRENATAL EXCLUSION. 0124's config VERBATIM; the ONLY change is the profile
# table it reads: re-emitted with `hpoa_profile_survey --rollup --emit-codes` which now
# DROPS every profile term under HPO HP:0001197 "Abnormality of prenatal development or
# birth" (265 terms) before realization. No fit-side knob changes.
#
# WHY. 0124 (guided anchors) fixed the headline misalignment — cardiomyopathy d3 went
# from headache/backache/migraine to a textbook cardiomyopathy block; intrinsic CM left
# the gestation-week stratum — at zero readout cost (paired −0.001 vs 0123), with 594 of
# 1490 anchors drawn from profiles. Its ONE residual: dilated cardiomyopathy anchored on
# `Reduced fetal movement` (HP:0001558 → a SNOMED finding recorded in the MOTHER's chart),
# inherited from a fetal-onset subtype; once its parent (intrinsic CM) no longer claimed
# the pregnancy direction, that term was DCM's purest remaining candidate and β recovery
# filled the block with prenatal-visit codes. A fetal phenotype's code is never the
# patient's own in this cohort, so terms under HP:0001197 must not guide a block — an
# ontology fact, not a threshold. Maternal phenotypes (pre-eclampsia HP:0100602, maternal
# hypertension HP:0008071) are NOT under it (verified on the hp.obo): pregnancy label
# nodes keep their guides.
#
# ALSO NEW IN THIS RUN (instrumentation, no model change): every spectral fit now writes
# `<run>/spectral_anchors.json` — per node, the chosen anchor vocab ids with a
# from-profile flag and the node's Mondo id/name (no patient data) — so the anchor read
# is DIRECT (0124's DCM anchor had to be inferred from its recovered topic).
#
# READ (priority order):
#   1. dilated cardiomyopathy's anchors (spectral_anchors.json, named via the names
#      CSV) and its STARRED digest line (`CREDITED=1` marks profile tokens with `*`):
#      is the prenatal stratum gone; what did DCM anchor on instead.
#   2. the starred digest over the cardiomyopathy/pregnancy grep vs 0124: cardiomyopathy
#      d3 and intrinsic CM must stay fixed; count topics with ZERO starred tokens
#      (stratum capture whatever the anchor) — the misplacement screen.
#   3. `--profile-align` paired vs 0124 and vs 0123 (all 132 guided nodes, not the
#      eyeballed few).
#   4. guardrail: ridge-100 macro AUC within ~0.005 of 0124 (0.7920) / 0123 (0.7946);
#      detection ≥ 0.63; starvation ≈ 1%; paired AB vs 0124 by depth.
# ACCEPTANCE: (1) DCM off the prenatal stratum, (2) no regression on the fixed nodes,
# (4) within the bar. (3) is the systematic read that decides whether the generic-
# symptom pattern (PIH's vaginitis/UTI/migraine topic: legitimate HPO terms — headache,
# abdominal pain — anchoring the young-woman primary-care stratum) needs its own rule.
#
# COST: identical to 0124. The TABLE must be regenerated on the cluster first (the
# survey's exclusion is default-on; `--keep-prenatal` reproduces 0124's table):
#   make hpoa-profile-survey HPOA_ARGS="--emit-codes <codes.tsv> --rollup"
#   make hpoa-stage2-probe ID=123 GPR_ARGS="--emit-eta <eta.tsv>"      (bundle HIT)
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


# 0125 — 0124 + the prenatal exclusion (and the anchor dump)

0124's exact config, reading a profile table re-emitted WITHOUT HPO's prenatal/birth
subtree. The engine, estimator, driver flags and front matter are unchanged from 0124;
the one new artefact is `spectral_anchors.json` in the run dir.

## What it does

Regenerate the codes + eta tables (survey now drops HP:0001197 and its 265 descendants
before realization; stderr reports rows/terms/nodes dropped), then
`make -C analysis/cloud exp ID=125` → fit-only spectral-init gated LDA with guided
anchors, then the ridge-100 readout sweep, paired ABs vs 0124 / 0123 / 0113, the STARRED
digest, and `--profile-align` vs 0124.

## Acceptance criterion

Dilated cardiomyopathy's block no longer carries the prenatal stratum (anchors + starred
digest); cardiomyopathy d3 and intrinsic CM stay as in 0124; macro AUC within ~0.005 of
0124. The profile-align scorecard and the zero-star count are the systematic reads.

## Run

One launch, after pulling. The chain regenerates the tables first (survey is offline;
the probe needs the bundle — a HIT on this cluster), then runs everything.

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/gated-conditional-voi && git checkout claude/gated-conditional-voi && git pull --ff-only
RUN=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0125-mondo-cardiovascular-tpn5-spectral-guided-noprenatal
mkdir -p "$RUN"
nohup bash -c '
  echo "=== tables START $(date)"
  make -C analysis/cloud hpoa-profile-survey HPOA_ARGS="--emit-codes $HOME/repos/CHARMPheno/data/ontology/profile_codes_MONDO_0004995.tsv --rollup" || exit 1
  make -C analysis/cloud hpoa-stage2-probe ID=123 GPR_ARGS="--emit-eta $HOME/repos/CHARMPheno/data/ontology/profile_eta_MONDO_0004995.tsv" || exit 1
  echo "=== fit START $(date)"
  make -C analysis/cloud exp ID=125 || exit 1
  echo "=== readout START $(date)"
  make -C analysis/cloud gated-pc-readout ID=125 GPR_ARGS="--readout-mode distributed --readout-l2 100"
  for m in own-bg family-closure; do
    make -C analysis/cloud gated-pc-readout ID=125 GPR_ARGS="--readout-mode distributed --readout-l2 100 --readout-feature-mask $m"
  done
  for b in 124 123 113; do make -C analysis/cloud inspect-topics ID=125 COMPARE=$b INSPECT_ARGS="--readout-auc"; done
  make -C analysis/cloud inspect-topics ID=125 CREDITED=1 RESOLVE_NAMES=1 INSPECT_ARGS="--digest --redundancy 30 --grep cardiomyopathy|atrial.fibrillation|heart.valve|mitral|aortic|myocardial.infarction|heart.failure|pregnan|gestation"
  make -C analysis/cloud inspect-topics ID=125 CREDITED=1 COMPARE=124 INSPECT_ARGS="--profile-align"
  echo "=== DONE $(date)"
' > "$RUN"/sweep_log.md 2>&1 &
```

(The grep pattern uses `.` for spaces so it needs no quoting inside the wrapper.)

Early checks: `grep -E "prenatal exclusion|anchor guide" "$RUN"/sweep_log.md` (the
survey's drop counts, then the builder's and the seed's guide counts).

Pull the numbers with:

```bash
grep -E "prenatal exclusion|anchor guide|^=== |gated_pc(_own_bg|_family_closure)? \(pc_topics_lr\): (macro|detection)|^## paired|^all: n=|^by depth|starved" "$RUN"/sweep_log.md
```

Name the anchors of the nodes that matter (after the digest step has produced the
names CSV); the script is `analysis/cloud/name_spectral_anchors.py`:

```bash
python3 ~/repos/CHARMPheno/analysis/cloud/name_spectral_anchors.py "$RUN" \
    --bundle-meta /tmp/inspect_meta_125.json --concept-names /tmp/concept_names_125.csv \
    --grep "dilated cardiomyopathy|intrinsic cardiomyopathy|^cardiomyopathy|pregnancy-induced"
```

## Run log

**2026-10-05 — launched (n2-standard-8 cluster, bundle HIT).** Table regeneration:
`prenatal exclusion (HP:0001197, 266 subtree terms): dropped 502 rows / 38 terms across
145 nodes` (branch nodes; 132 are label nodes). Eta table 99,348 rows (0124's: 99,517).
Driver builder: 132/132 nodes guided, **28,438 candidates (0124: 28,441)** — the
exclusion removed THREE in-vocab candidate concepts across the whole branch. The fetal
terms realize to almost nothing in this adult vocabulary; `Reduced fetal movement` was
essentially the only one, which is why one term could do what it did. The seed should
therefore differ from 0124 only at the nodes where such a candidate was actually chosen.

## Results
