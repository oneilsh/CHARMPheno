---
id: 115
slug: mondo-cardiovascular-tpn5-spectral-binary
status: done
model_class: gated_pc
cohort: population_mondo_all
cohort_def: population_mondo_all
disease: rare_priority
# THE BURST-BIAS TEST. 0114's config VERBATIM (CV branch MONDO:0004995, tpn=5, spectral
# init, fit-only, spectral_d 768) with ONE knob added: count_transform: binary. So
# 0115-vs-0114 is a CLEAN A/B on the count representation alone.
#
# WHY. 0114 (spectral init) lifted the deep-node starvation floor decisively (72% -> 1%
# starved; depth-4 median evidence 62 -> 1000) — the flat-start trap was the binding
# constraint, not budget (0113). But a MINORITY of the newly-fed deep topics anchor on the
# WRONG signal: intrinsic/dilated cardiomyopathy came back dominated by PREGNANCY codes
# (trimesters, gestation weeks), a cardiomyopathy topic by generic primary-care symptoms —
# sharp, but on a demographic/utilization stratum, not the phenotype. (peripartum
# cardiomyopathy -> pregnancy is CORRECT; DCM/intrinsic-CM is the anchor grabbing a
# young-female subpopulation.)
#
# MECHANISM (multi_domain.py:456: domain_binary = [False] + [d=="measurement"]). Only the
# MEASUREMENT domain is per-doc binary (insight 0077, "bursty"); CONDITIONS and DRUGS are
# RAW OCCURRENCE COUNTS. A gestation-week code is recorded every prenatal visit — one
# pregnant patient contributes ~20+ pregnancy tokens vs one diagnosis token. The anchor
# search maximizes co-occurrence residual norm and the topic evidence maximizes token
# mass; BOTH are dominated by what REPEATS most per document. So pregnancy won the
# intrinsic-CM anchor by VOLUME, not meaning — the same bias behind 0113's "fed but
# generic lab panel" topics. Reframes the misalignment: spectral's criterion is aligned
# with TOKEN MASS, and raw-count BOWs make token mass a proxy for utilization, not
# phenotype.
#
# FIX (cheap, no supervision, no cache rebuild). count_transform: binary collapses every
# token to per-doc PRESENCE (min(count,1)) before the seed AND the fit — insight 0077's
# measurement fix extended to all domains. Applied in-memory in the PC shim
# (pc.py _transform_counts); the cached bundle stays raw-count, so NOT a cache-key change
# and it reuses 0114's bundle. A code recorded every visit now counts once, like a dx.
#
# FALSIFIABLE PREDICTION. The pregnancy-anchored intrinsic/dilated-CM topics RECEDE
# (pregnancy -> 1 token/patient); the phenotype-coherent topics (AF, systolic/diastolic
# HF, valve) HOLD. Whatever misalignment SURVIVES binarization is the genuine residual —
# the "unsupervised separability != meaning" part that needs supervision (parked PC). So
# this DECONFOUNDS burst-bias from true objective misalignment; today they are tangled.
#
# READ: inspect_topics.py --digest node-for-node vs 0114 (reuse 0113/0114 cached meta):
#   - starved% + depth rollup: must stay LIFTED (binarization must not re-starve depth).
#   - --grep the pregnancy-anchored CMs: do intrinsic/dilated cardiomyopathy shed the
#     gestation terms for cardiac/management signal? does peripartum CM KEEP pregnancy?
#   - do the AF/HF/valve topics stay coherent?
# Acceptance: burst-anchored topics re-align toward phenotype (pregnancy recedes where a
# confound, survives where it IS the disease) WITHOUT re-starving the floor. A null
# (topics unchanged) => misalignment is NOT burst-bias but genuine objective misalignment
# => supervision is the lever, not counts. A floor DROP => presence discarded needed
# repetition => fall back to log1p.
#
# COST: identical to 0114 (same bundle, K=1498, spectral seed; the transform is a free
# in-memory map). ~1.5h seed + ~35 min fit on the lean cluster.
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
count_transform: binary     # THE NEW KNOB: per-doc PRESENCE not raw per-visit counts (burst-bias test)
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


# 0115 — CV branch, spectral init, BINARY counts (the burst-bias test)

0114's exact config with **one knob added — `count_transform: binary`** — so any change is
attributable to the count representation alone. This deconfounds 0114's misaligned deep
topics into **burst / utilization-volume bias** (fixable here, cheaply) vs **genuine
unsupervised-objective misalignment** (supervision's job).

**Grounding.** 0114 lifted the starvation floor (0079's flat-start trap confirmed) but a
minority of fed deep topics anchored on demographic substructure — intrinsic/dilated
cardiomyopathy on pregnancy codes — because conditions/drugs are raw per-visit counts
(only measurements were binarized, 0077), so the anchor and evidence criteria track
utilization volume, not phenotype. Binary presence removes that lever.

## What it does

`make -C analysis/cloud exp ID=115` → 0114's fit with every document's BOW collapsed to
per-token presence (`min(count,1)`) before the spectral seed and the fit. Same cached
bundle, same K, same seed; the transform is an in-memory `.map` in the PC shim.

## Acceptance criterion

Do the burst-anchored deep topics **re-align toward phenotype** — pregnancy receding from
intrinsic/dilated CM (a confound) while surviving in peripartum CM (correct) — **without
re-starving** the depth floor 0114 lifted?
- **Re-align** → the misalignment was largely burst-bias; binary presence is the fix, no
  supervision needed, and it should become the default count representation.
- **Null** (topics unchanged) → genuine objective misalignment (separability ≠ phenotype);
  supervision (parked PC) is the lever, not counts.
- **Re-starves** (floor drops) → presence discarded discriminative repetition some nodes
  needed; fall back to `log1p` (softer damping).

## Run

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/gated-conditional-voi && git checkout claude/gated-conditional-voi && git pull --ff-only
make -C analysis/cloud exp ID=115
```

Then (reuse 0113/0114 cached meta/names):

```bash
make -C analysis/cloud inspect-topics ID=115 \\
    INSPECT_META=/tmp/inspect_meta_113.json INSPECT_NAMES=/tmp/concept_names_113.csv \\
    INSPECT_ARGS="--digest --redundancy 30 \\
      --grep 'cardiomyopathy|atrial fibrillation|heart valve|mitral|aortic|myocardial infarction|heart failure|pregnan|gestation'"
```

## Run log

**2026-09-05 — fit.** 0114's config + `count_transform: binary`, fit-only, K=1498, same
bundle/seed. Saved fit-only λ.

## Results

**Acceptance — did binarization re-align the burst-anchored topics? NO. Burst-bias is
FALSIFIED as the cause. The floor stayed lifted, but the pregnancy anchoring persisted —
so the misalignment is GENUINE objective misalignment (separability ≠ phenotype), not a
token-mass artifact.**

- **Floor held (no re-starvation):** 1% starved, all depths sharp (frac ~0.01–0.02).
  Median evidence is uniformly lower than 0114 (depth-4 383 vs 1000; depth-5 279 vs 719)
  — expected and benign: binarization removes repeat mass, so λ-sum drops while sharpness
  is unchanged. Redundancy is even cleaner than 0114 (0/81 collapsed, worst fed-cosine
  **0.17** vs 0.36). So per-doc presence is a viable representation.
- **The pregnancy anchoring PERSISTED** on the cardiomyopathy family, essentially
  unchanged from 0114: **intrinsic cardiomyopathy** (d4) still *Gestation period · 1st/2nd/
  3rd trimester · high-risk pregnancy · miscarriage* (and one topic a genetic grab-bag —
  cystic fibrosis / trisomy 21 / hemophilia); **dilated cardiomyopathy** (d5) still
  *trimesters · gestation weeks · disorder of pregnancy // oxytocin*; **cardiomyopathy**
  (d3) still generic-symptom + pregnancy. The coherent nodes stayed coherent (AF →
  *Paroxysmal/Chronic/Persistent AF · atrial flutter · VT // rivaroxaban·metoprolol·
  diltiazem*; systolic/diastolic HF; valve).

**Interpretation.** At presence level, "was pregnant" is *still* the single most-separable
feature of the intrinsic/dilated-CM patient population — because these nodes' patients ARE
disproportionately young women diagnosed peri-pregnancy, and (with the CM code stripped)
the residual most-distinctive PRESENCE direction is the demographic/etiologic stratum
(peripartum, genetic), not a unified cardiac phenotype. So the anchor is surfacing REAL
population structure of a heterogeneous node, not a counting artifact. This is exactly the
"unsupervised separability ≠ node meaning" residual: the deflated CAVI objective has no
term preferring the cardiac-discriminating direction over the most-separable substratum,
and binarization (which only removes repetition) cannot supply one.

**Verdict — the burst-bias / objective-misalignment fork (insight 0082) resolves toward
OBJECTIVE.** The cheap fix (presence) is ruled out for this residual. The principled lever
is a LABEL-AWARE objective (supervision) — pull each node's topic toward its
case-vs-control discriminating direction — and this is plausibly insight 0066's payoff
regime (a lower-separability cardiac signal buried under a more-separable demographic one),
unlike the AoU antidepressant task where PC was marginal. Cheaper-than-PC alternative:
NUISANCE DEFLATION (deflate each node against a corpus-wide demographic basis). BUT whether
this residual is worth paying for is a DETECTION question, not a topic-legibility one: the
next move is `gated-pc-readout` on 0114/0115 — does the demographic anchor cost case-finding
AUC, or does the localized head simply not read it? Only if it costs detection is
supervision / nuisance-deflation worth the compute.

Secondary: binary presence did not hurt (cleaner redundancy, floor held), so it is a
reasonable default representation independent of the alignment question.

## Readout (2026-09-11, first ever; ridge 100, 257/259 heads converged)

macro AUC **0.7898** / AP 0.5263 over 193 nodes; detection 0.6042. vs 0113 (random init,
same ridge): 0.8087 / 0.5451 / 0.6043. **Spectral's case-finding cost at a converged head is
−0.019 macro** (0083's −0.025 was the ridge-1 instrument), detection unchanged.

**Paired per-node AB vs 0113 (`inspect-topics ID=115 COMPARE=113 --readout-auc`, no
credited split — 0115 feeds no profile, so the credit line is not a contrast here):**
193 shared scored nodes, median dAUC **−0.0202** mean −0.0189 (p25 −0.039, p75 −0.004),
up/down **42/151**. By depth (median dAUC): d2 −0.018 (n=4), d3 −0.016 (34), d4 −0.020
(51), d5 −0.027 (54), d6 −0.018 (37), d7 +0.001 (13). Per-depth medians 0.81/0.80/0.80/
0.80/0.79/0.69 (d2..d7). The cost is **broad and uniform**, not a depth story: 78% of
nodes are down, and the deep levels where spectral fed the blocks (d5–d6; starvation 1%
vs 0113's 72%) lose as much as the shallow ones — feeding the block did not buy the head
any case-finding at depth. The one flat level, d7, is 13 nodes.

**Ablation ladder at ridge 100 (same masks as 0116 / insight 0090):**

| head may load on | 0115 (spectral, fed) | 0116 (random, unfed) |
|---|--:|--:|
| own block + background (`own-bg`) | pending (in `<run>/sweep2_log.md`, `results_readout_own_bg.json`) | 0.669 |
| + ancestors (`family-closure`) | **0.7851** (AP 0.516; det 0.536 / AP 0.689) | 0.777 |
| everything | 0.7898 (det 0.604) | 0.810 |

family-closure sits **0.0047 under the full head** — closer to full than 0116's
(0.777 vs 0.810, −0.033): with fed blocks, ancestors+own carry nearly everything the full
head reads, and the rest of the topic space adds little. Detection at 0.536 for a
family-restricted head is expected (near chance: the family mask throws away the
background topics that separate case from non-case). The own-bg number is the decisive
one (does a FED leaf block carry its own node's signal?) — see the 2026-09-11 handoff §4.

Both masked readouts ran under `timeout` on pre-`5205d0d` code and ended rc=124: the
client-mode driver wedged in teardown AFTER its results were written (the sweep-chain
hang fixed in `5205d0d`); the numbers are valid. `readout-ab` with its default
`CREDITED=1` crashed on the profile-eta TSV (dies with the cluster); the plain
`inspect-topics` AB above is the read.

