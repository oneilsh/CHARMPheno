---
id: 124
slug: mondo-cardiovascular-tpn5-spectral-guided
status: pending
model_class: gated_pc
cohort: population_mondo_all
cohort_def: population_mondo_all
disease: rare_priority
# THE GUIDED-ANCHOR ARM. 0123's config VERBATIM (CV branch MONDO:0004995, tpn=5,
# spectral init, raw counts, alpha fixed 0.5, fit-only) with ONE line added:
# spectral_anchor_profile -> the profile-eta TSV. So 0124-vs-0123 is a clean A/B on
# WHICH WORDS spectral may anchor each node's block on, and nothing else.
#
# WHY. The 0113/0115/0123 ladder settled three things: spectral FILLS the deep blocks
# (starvation 72% -> 1%); a filled block CARRIES its own node's signal (own-bg 0.73 vs
# 0.67 unfed); and what spectral still costs vs random init (-0.016 median, uniform
# across depth, 0123 AB) is a head-side cost of the anchor words it picks by
# co-occurrence geometry alone — dilated/intrinsic cardiomyopathy anchored on
# gestation-week codes (0114, insight 0082) because the pregnant-young-female stratum
# is a sharp, pure direction in Q. Binary counts were tried as the fix (0115) and cost
# detection instead. The lever is the ANCHOR SEARCH.
#
# MECHANISM (spec docs/superpowers/specs/2026-10-05-hpo-guided-spectral-anchors.md).
# For each profiled node the greedy farthest-point search runs in two stages over ONE
# basis: stage 1 picks only from the node's PREFERRED set — its rolled-up HPO-profile
# terms, realized to OMOP condition concepts, positive, weight>0 (freq x IDF, so a
# term every credited node shares is excluded), in the fitted vocabulary, and above
# the node's OWN df floor; stage 2 fills whatever stage 1 could not from the open
# pool. Deflation against background + ancestors is unchanged; beta recovery for
# every word is unchanged. "HPO decides WHICH words may anchor; the data decides WHAT
# the topic is." A set, not a weight: no strength knob. 132 of 299 label nodes have a
# set (report 2026-09-06, --rollup); the other 167 run today's open search exactly.
#
# READ (priority order — this build is FOR legibility; the readout is the guardrail):
#   1. --digest --grep on the cardiomyopathies: do intrinsic/dilated CM shed the
#      gestation anchors; does peripartum CM KEEP them; do AF/HF/valve stay coherent.
#      The seed log line `[pc] spectral anchor guide:` says how much of the anchoring
#      the profiles decided (nodes guided / anchors from profile / fallback-only).
#   2. starvation % from the fit log: must stay ~1%.
#   3. ridge-100 readout: full vs 0123 (0.7946) and 0113 (0.8087); detection vs
#      0.630; own-bg vs 0.730; paired AB vs 0123 by depth.
# ACCEPTANCE: anchors re-align (1) AND macro AUC within ~0.005 of 0123 (3). Recovery
# toward 0113 is the hypothesis, not the bar.
# OUTCOMES: re-align + hold -> spectral + guided anchors is the base init. re-align
# but AUC drops -> profile terms legible but not discriminative here; read WHICH
# profile anchors the heads don't load (digest + loadings) before touching the rule.
# No re-align (pregnancy persists) -> the preferred set is being exhausted (floor or
# ancestor deflation); the seed counts say which.
#
# COST: identical to 0123 (same bundle, K=1498, spectral seed ~1.5h + ~35 min fit on
# the n2-standard-8 master shape). The eta TSV is cluster-ephemeral (~ wiped on
# restart): regenerate with `hpoa-profile-survey --emit-codes ... --rollup` then
# `hpoa-stage2-probe ID=123 --emit-eta ...` (bundle HIT needed) before `make exp`.
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
spectral_anchor_profile: ~/repos/CHARMPheno/data/ontology/profile_eta_MONDO_0004995.tsv
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


# 0124 — CV branch, spectral init, raw counts, HPO-GUIDED anchors

0123's exact config with **`spectral_anchor_profile`** set: each profiled node's spectral
anchor search prefers its own HPO-profile terms (positive, IDF-weight > 0, in vocab,
above the node's own df floor) and falls back to the open search for the rest. No
strength knob. Nodes without a profile are untouched. See the spec for the rules and
the handoff 2026-09-11 §4/§5.1 for why this is the build.

## What it does

`make -C analysis/cloud exp ID=124` → fit-only spectral-init gated LDA on the CV branch
with guided anchors; `diag_only` saves the fitted globals. Then the ridge-100 readout
sweep, the paired ABs vs 0123 and 0113, and the cardiomyopathy digest.

## Acceptance criterion

Legibility first: the gestation anchors leave the non-peripartum cardiomyopathies and
the seed log shows the profiles decided a material share of the anchoring. Then the
guardrail: ridge-100 macro AUC within ~0.005 of 0123's 0.7946, detection ≥ 0.63,
starvation ≈ 1%.

## Run

Pre-flight: the eta TSV must exist at the path in the front matter (see COST above).

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/gated-conditional-voi && git checkout claude/gated-conditional-voi && git pull --ff-only
ls -l ~/repos/CHARMPheno/data/ontology/profile_eta_MONDO_0004995.tsv
RUN=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0124-mondo-cardiovascular-tpn5-spectral-guided
mkdir -p "$RUN"
nohup bash -c '
  make -C analysis/cloud exp ID=124 || exit 1
  echo "=== readout START $(date)"
  make -C analysis/cloud gated-pc-readout ID=124 GPR_ARGS="--readout-mode distributed --readout-l2 100"
  for m in own-bg family-closure; do
    make -C analysis/cloud gated-pc-readout ID=124 GPR_ARGS="--readout-mode distributed --readout-l2 100 --readout-feature-mask $m"
  done
  make -C analysis/cloud inspect-topics ID=124 COMPARE=123 INSPECT_ARGS="--readout-auc"
  make -C analysis/cloud inspect-topics ID=124 COMPARE=113 INSPECT_ARGS="--readout-auc"
  make -C analysis/cloud inspect-topics ID=124 INSPECT_ARGS="--digest --redundancy 30 --grep '"'"'cardiomyopathy|atrial fibrillation|heart valve|mitral|aortic|myocardial infarction|heart failure|pregnan|gestation'"'"'"
  echo "=== DONE $(date)"
' > "$RUN"/sweep_log.md 2>&1 &
```

Early checks: `grep -E "anchor guide|spectral anchor guide" "$RUN"/sweep_log.md` after the
seed (the driver's builder counts, then the estimator's pooled seed counts); the fit's
starvation line at the end of the fit.

Pull the numbers with:

```bash
grep -E "anchor guide|^=== |gated_pc(_own_bg|_family_closure)? \(pc_topics_lr\): (macro|detection)|^## paired|^all: n=|^by depth|starved" "$RUN"/sweep_log.md
```

### Stacked readout (spec 2026-10-06 Part A) — re-readout of the saved fit, no refit

The closure-product arm: root head fit on every train row (case vs background), each
node scored by Π over its closure of the per-node heads. Tagged outputs
(`results_readout_stacked.json`, `readout_heads_gated_pc_stacked.npz`); the record is
untouched. ~15 min on a warm bundle (+~20 min rebuild on a fresh cluster).

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/gated-conditional-voi && git checkout claude/gated-conditional-voi && git pull --ff-only
RUN=$(ls -d /home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0124-*)
nohup make -C analysis/cloud gated-pc-readout ID=124 GPR_ARGS="--readout-mode distributed --readout-l2 100 --readout-stacked" > "$RUN"/stacked_log.md 2>&1 &
```

Pull the numbers with:

```bash
grep -E "STACKED|stacked readout|three reads|flat: max|root head alone|stacked: max|ranking \(within|paired per-node|depth [0-9]+:|marginal ECE" "$RUN"/stacked_log.md
```

## Run log

**2026-10-05 — fit + full chain, first try (n2-standard-8 cluster, bundle HIT, eta TSV
from the 0123 probe).** Guide reached the seed:
`anchor guide` (driver builder): 132/132 profiled nodes in the DAG and guided, 28,441
candidates total (972 concepts dropped as NOT-annotated, 0 as weight-0, 68,197 rows
dropped as not in the fitted vocabulary).
`spectral anchor guide` (seed): nodes guided 132/298; **anchors from profile 594/1490**;
nodes fully guided 112, partially 17, fallback-only 3. Starvation 1% (unchanged).

## Results

**Full head, ridge 100 (193 nodes):**

| | macro AUC | AP | detection AUC | det AP |
|---|--:|--:|--:|--:|
| 0113 random init, raw | 0.8087 | 0.5451 | 0.6043 | — |
| 0123 spectral, raw | 0.7946 | 0.5327 | 0.6299 | 0.7600 |
| **0124 spectral, raw, guided** | **0.7920** | 0.5261 | **0.6325** | 0.7595 |

Paired vs 0123: median dAUC **−0.0013** mean −0.0026 (p25 −0.011, p75 +0.009), up/down
87/106; by depth d2 −0.001, d3 +0.001, d4 −0.002, d5 −0.001, d6 −0.006, d7 +0.004. Paired
vs 0113: median −0.0177, up/down 53/140; by depth d2 −0.013, d3 −0.010, d4 −0.015,
**d5 −0.031**, d6 −0.005, d7 +0.006.

**Guardrail: MET.** −0.003 macro vs 0123 is inside the ~0.005 bar and the paired median
is −0.001 with a near-even split; detection is unchanged (+0.003). Replacing 40% of all
anchors (594 of 1490) with profile terms cost the heads nothing. The remaining tax vs
random init is still at d5 (−0.031, n=54): guided anchors did not recover it either.

**Legibility (named digests, 0124 vs 0123, same grep, top-25 by evidence):**

- **WIN — `cardiomyopathy` (d3).** 0123: *Headache · Abdominal pain · Backache · Migraine
  · Cough · Nausea · Acute pharyngitis · URI · …* — the generic primary-care topic that
  insight 0082 named as the misalignment. 0124: *Cardiomyopathy · Primary cardiomyopathy
  · Dilated cardiomyopathy · Heart failure · Chronic systolic HF · CHF · Left bundle
  branch block · Cardiomegaly · Hypertensive HF* // carvedilol · spironolactone ·
  lisinopril. A textbook cardiomyopathy block where there was none.
- **WIN (to confirm) — `intrinsic cardiomyopathy` (d4).** 0123 had TWO intrinsic-CM
  topics in the top 25, both pure gestation-week lists (*Gestation period, 32/34/37/36…
  weeks*). Neither appears in 0124's top 25: the gestation stratum no longer carries
  intrinsic CM's evidence. (Its 0124 topics still need a direct look — `grep intrinsic`.)
- **REMAINING — `dilated cardiomyopathy` (d5).** 0124: *Carrier of cystic fibrosis gene
  mutation · Unplanned pregnancy · First trimester pregnancy · Abnormal cytological
  finding · Gestation period, 11 weeks · Pregnancy test negative/positive · ASCUS ·
  Obesity · High risk pregnancy* — the prenatal-screening stratum, one level DOWN from
  where it sat in 0123 (intrinsic CM, DCM's parent). **Mechanism, confirmed from the
  tables (2026-10-05):** NOT peripartum CM's profile (peripartum IS DCM's child in this
  DAG, but DCM's 1,744 code rows / 331 HPO terms are all inherited and contain only four
  pregnancy-related terms, all FETAL: decreased fetal movement, hydrops fetalis, fetal
  ascites). Of DCM's 355 in-vocab candidates, exactly one is an obstetric code:
  **77619 "Reduced fetal movement"** — HP:0001558 realized to a SNOMED finding that is
  recorded in the MOTHER's chart. In an adult EHR that code marks a pregnancy and nothing
  about the patient's own phenotype. Once the guide moved intrinsic CM (DCM's parent) onto
  proper cardiomyopathy terms, ancestor deflation no longer removed the pregnancy
  direction from DCM's documents, "Reduced fetal movement" became DCM's purest remaining
  candidate, the search anchored on it, and β recovery filled the block with the
  prenatal-visit codes that co-occur with it. (Intrinsic CM and the d3 cardiomyopathy
  node carry the SAME 3,798-concept rolled-up set — the whole subtree's union — and
  contain the same code; they anchored elsewhere first because at their level stronger
  cardiomyopathy directions remained.)
- Heart failure / CHF / systolic HF / valve topics: clean in both runs.
- Pregnancy label nodes (`toxemia of pregnancy`, `hypertension, pregnancy-induced`)
  carry the pregnancy stratum in both — correct; those ARE cardiovascular branch nodes.

**Verdict.** Guided anchors fixed the headline misalignment (cardiomyopathy d3,
intrinsic CM d4) at zero readout cost. The one residual is a realization artefact, not an
anchor-search failure: **a fetal phenotype term must not guide a block**, because its
realized code lives in the mother's record. The derived fix (no knob): drop profile terms
under HPO's `Abnormality of prenatal development or birth` (HP:0001197) subtree from the
GUIDE — the obo parse the survey already does gives the closure — and re-emit the table.
Pregnancy label nodes are unaffected (their own terms — pre-eclampsia, gestational
hypertension — are maternal phenotypes, not under HP:0001197). Second change for the next
run: dump each node's chosen anchors (vocab ids, from-profile flag) to the run dir so the
anchor read is direct instead of inferred from the recovered topic. That is exp 0125.
