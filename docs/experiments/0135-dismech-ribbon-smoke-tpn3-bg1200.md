---
id: 135
slug: dismech-ribbon-smoke-tpn3-bg1200
status: planned
model_class: gated_pc
cohort: population_mondo_all
cohort_def: population_mondo_all
disease: rare_priority
# THE RIBBON SMOKE (spec docs/superpowers/specs/2026-10-07-dismech-ribbon-label-space-design.md).
# First fit on a FLAT label set: the DisMech disorders (3,239 Mondo ids at DisMech 71cd0452;
# analysis/cloud/anchor_selection_data/dismech_ribbon.tsv), whole Mondo, no branch. The
# powered set (closure support >= 100, whole population) is intersected with the ribbon;
# codes under a member roll up to it; codes under no member are background-only. The
# FIT sees a forest of disorders under the root; the stacked READ sees root x node (the
# HSLDA two-level product). Ancestor heads (WP-B) are a later, additive change.
#
# WHY. Every topic-side problem of 0123-0131 was nesting (parent/child competing for the
# same docs and words). The ribbon removes nesting from the fit and keeps the hierarchy
# where it earned its keep: the stacked read (insight 0092).
#
# ONE VARIABLE vs 0134: the label space. Everything else is 0134's config except
#   tpn 1 -> 3   (spec R2: tpn is a CEILING; n_profiles(d) is estimated post-fit in 0136 —
#                 rare6/EDS at 20 topics per block is the precedent that a disease can
#                 carry several coherent profiles; 3 is the cheapest ceiling that can show
#                 >= 2 signatures)
#   mondo_branch MONDO:0004995 -> '' (the ribbon spans Mondo)
#   readout_theta_topm 256 -> 0 (see TOP-M below)
# n_bg 1200 CONFIRMED by 0134 (insight 0095): 1,200 random-init strata restored the root
# head (0.811, best of the series) and de-novo AP (0.224) with the per-node blocks intact;
# `spectral_bg_anchors: 8` because a wide background is not anchorable (0134's two seed
# blow-ups) — the eight anchored rows are the deflation seeds, the rest start random.
#
# TOP-M. 0134's one open decoding number is whether its -0.018 within-cohort vs 0133 is the
# readout's top-256 theta truncation (84% of topics kept at K=306, 17% at K=1498). At this
# run's K (~5-6k) top-256 would keep ~5%, so the truncation is OFF here (readout_theta_topm
# 0): the ribbon's heads read the whole theta. The cost is readout time, not a refit.
#
# EXPECTED SIZE. 0110 (whole Mondo, native) kept 2,713 terms at min_positives 100; the
# ribbon is rare-disease-heavy, so expect roughly 1,200-1,800 kept members -> K about
# 1200 + 3 x (1,200..1,800) = 4,800..6,600. Heavier than 0132's K=1498 and 0104's 3,827.
# If the fit wall or the top-m readout truncation bites, min_positives 200 is the knob
# (recorded as a NEW doc, not an edit here). The `[mondo-native] label set` receipt
# prints |R| -> kept, unknown-to-release, unpowered, and nested-pairs-within-kept.
#
# READ (ridge 100, flat + stacked; the stacked product here is root x node):
#   - the label-set receipt: how many members are powered; how many nested pairs the
#     ribbon has under Mondo (treated as flat; this is the flatness number of record).
#   - starvation rate at tpn=3 (0133/0134 at tpn=1: 0-1%; 0123 at tpn=5 had strata).
#   - the same-vocabulary-child failure (insight 0094/0095: DCM -> its parent's young-women
#     cohort under every nested topic side): on the ribbon DCM has NO parent in the fit, so
#     its seed is its own closure documents with nothing deflated away. If DCM's block is
#     now cardiomyopathy words, nesting WAS the mechanism; this is the single most
#     diagnostic digest line of the run.
#   - within-cohort macro AUC / AP and de-novo AUC / AP, overall and on the nodes SHARED
#     with the cardiovascular branch runs (0133 / 0134): the pre-registered reads.
#   - the digest on a few named neighbourhoods (Ehlers-Danlos; dilated cardiomyopathy;
#     a metabolic and an immunodeficiency member): are the three topics per block
#     signatures, strata, or one signature + two unused?
# OUTCOMES (pre-registered, on the shared nodes):
#   (a) within +/-0.01 of 0134 -> nesting on the topic side was buying nothing; proceed
#       to 0136 (profile census) on this fit.
#   (b) clearly below -> the gate's +0.03 needed nested negatives; re-examine the closure
#       mask on a flat forest (siblings are now the whole ribbon) before anything else.
#   (c) clearly above -> nesting was a tax; same next step as (a), note it as a finding.
# COST: whole-population fit at K ~5-6k, 50 iterations, diag_only; bundle MISS (new key)
# with the sidecar a HIT; the seed is the basis-form greedy (0134's fix). Expect the fit
# between 0104's and 0134's; the distributed readout over ~1,500 heads is 0104-sized.
dag_source: mondo_native
mondo_branch: ""
label_set: analysis/cloud/anchor_selection_data/dismech_ribbon.tsv   # THE CHANGE vs 0134
tpn: 3                      # a CEILING (spec R2); n_profiles(d) is read in 0136
max_iter: 50
diag_only: true
init: spectral
spectral_method: scalable
spectral_d: 768             # 0134's: only 8 bg anchors + 3 per node are anchored
anchor_scope: closure       # on a flat forest, closure(node) = {node, root}: own block
spectral_topo_order: forward
count_transform: none
preindex_closure: false
readout_mode: distributed
readout_theta_topm: 0       # no top-m truncation at K ~5-6k (see TOP-M)
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
n_bg: 1200                  # 0134's (insight 0095: width is the root head's lever)
spectral_bg_anchors: 8      # 0134's: anchor eight bg rows as deflation seeds, random-init the rest
optimize_doc_concentration: false   # insight 0091; see 0134's LANDMINE note
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
  # 0134's geometry. K is ~4x 0134's; if executors OOM in the E-step, raise
  # spark.executor.memory before touching min_positives.
  spark.executor.cores: 2
  spark.executor.memory: 8g
  spark.executor.memoryOverhead: 3g
  spark.dynamicAllocation.enabled: "false"
  spark.executor.instances: 20
  spark.excludeOnFailure.timeout: 10m
  spark.cleaner.periodicGC.interval: "5min"
---

# 0135 — the DisMech ribbon as the label space (flat fit, tpn=3 ceiling, 1,200 shared strata)

First run of the 2026-10-07 spec. Same engine, same readouts, same background as 0134;
the label set is DisMech's flat disorder list instead of a nested Mondo branch.

## What it does

`make -C analysis/cloud exp ID=135` → native-Mondo build with `--label-set` (the ribbon
receipt prints before the fit), spectral-init gated fit with three topics per disorder
over a 1,200-topic background, then the ridge-100 flat readout and the stacked readout
(root × node), and the digest.

## Acceptance criterion

The (a)/(b)/(c) split in the front matter, read on the nodes shared with 0133/0134.
Whatever the split, the label-set receipt and the starvation rate are results in their
own right (how flat the ribbon is under Mondo; whether tpn=3 starves).

## Before launching

- WP-A on the cluster: `claude/dismech-ribbon` checked out (the preamble below).
- A warm cluster has no bundle under this key (new label set) — expect the MISS and the
  sidecar HIT.

## Run

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUN=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0135-dismech-ribbon-smoke-tpn3-bg1200
mkdir -p "$RUN"
nohup bash -c '
  make -C analysis/cloud exp ID=135 || exit 1
  echo "=== readout START $(date)"
  make -C analysis/cloud gated-pc-readout ID=135 GPR_ARGS="--readout-mode distributed --readout-l2 100 --readout-theta-topm 0"
  echo "=== stacked START $(date)"
  make -C analysis/cloud gated-pc-readout ID=135 GPR_ARGS="--readout-mode distributed --readout-l2 100 --readout-theta-topm 0 --readout-stacked"
  for b in 134 133; do make -C analysis/cloud inspect-topics ID=135 COMPARE=$b INSPECT_ARGS="--readout-auc"; done
  make -C analysis/cloud inspect-topics ID=135 RESOLVE_NAMES=1 INSPECT_ARGS="--digest --redundancy 30 --grep '"'"'Ehlers|cardiomyopathy|atrial fibrillation|heart failure|pregnan|immunodeficiency|Gaucher|Marfan'"'"'"
  echo "=== DONE $(date)"
' > "$RUN"/sweep_log.md 2>&1 &
```

Early checks: the `[mondo-native] label set dismech:71cd0452358b:3226:f27a6b36ca62` receipt line (…`label DAG built FLAT`),
then the corpus line `K=<1200 + 3*kept> gated topics (1200 bg + <kept> nodes x 3 tpn)`.

Pull the numbers with:

```bash
grep -E "^=== |label set|powering:|label DAG|corpus: V=|starved|top-m mass|gated_pc(_stacked)? \(pc_topics_lr\): (macro|detection)|root head alone|stacked: max|ranking \(within|paired per-node|paired delta|depth [0-9]+:|^all: n=|^by depth" "$RUN"/sweep_log.md
```

Note on the comparison: `inspect-topics COMPARE=` pairs nodes on the intersection of the
two runs' scored node ids (`inspect_topics.py:1299`), and native-path ids are stable Mondo
cids independent of the kept set (`mondo_native_dag.mondo_cid`), so pairing across label
spaces works as is. 0133/0134 are cardiovascular-branch runs, so the paired set is the
ribbon members inside that branch — a few hundred nodes, enough for the (a)/(b)/(c) read.

## Run log

**2026-10-07 — launch 1 ran to the digest on a NESTED label DAG (a WP-A bug).** The
ribbon filter intersected the powered set correctly, but `build_native_label_dag` still
built the induced Hasse over the kept members, so Mondo's nesting of ribbon members
(peripartum cardiomyopathy under dilated cardiomyopathy; hundreds of members under three
generic "disease" members) came through as a depth-4 fit DAG: digest depth table d1 3 ·
d2 327 · d3 65 · d4 8 nodes, K=2403 (1200 bg + 401 nodes × 3), C=402. The gate, the
closure mask and the closure-scope anchors all saw a hierarchy. What the digest still
says (recorded because it is informative on its own): 2% starved; 0 collapsed parents;
textbook blocks for AF (2 distinct profiles + a CM/VT one), heart failure (systolic /
diastolic-ischemic / acute valvular-effusion — three real profiles), HCM, Marfan,
Takotsubo, CVID (IgG-replacement / sinus-asthma / IgG-subclass), EDS (POTS-autonomic /
MCAS-immune / migraine-musculoskeletal — the insight 0035 sub-phenotypes, back); DCM is
STILL the pregnancy stratum in all three topics (peripartum patients rode into DCM's
closure under the nested DAG, exactly 0134's mechanism) and its cardiomyopathy words
sit in the sibling heart-failure block; "insomnia" and "ectopic pregnancy" carry
generic-ED-symptom strata inside their blocks despite 1,200 background topics. C=402
is far below the expected 1,200–1,800 powered members — the `[mondo-native] label set`
receipt (unknown-to-release vs unpowered) decides whether that is rarity or a Mondo
version mismatch. Fixed in `mondo_native_dag.build_native_label_dag(flat=True)` on a
ribbon run (every member under the root; attestation unchanged).

**2026-10-08 — the receipt, computed offline against Mondo 2026-06-02** (the cluster's
`[mondo-native] label set` line was not pulled before the thread moved; this is the same
arithmetic, `apply_label_set_filter` on the committed TSV):
- 3,239 members; **4 unknown** to the 2026-06-02 release (MONDO:1060229–32, newer than
  the pin) — so the pin is NOT the cause of C=402. **Only ~401 members clear
  `min_positives` 100** in the whole population: the ribbon is rare-disease-heavy
  (1,925 Mendelian), and the 400 are the diseases All of Us can see at that floor. This
  is the scale truth of the ribbon; `min_positives` 50 (or 20, the egress floor itself)
  is the lever if the record run wants more of the tail, at the cost of thinner heads.
- **The ribbon is not flat under Mondo: 312 members are ancestors of other members,
  4,855 (ancestor, descendant) pairs.** Top: MONDO:0000001 "disease" (3,223 descendants
  — the ONTOLOGY ROOT, carried by DisMech's `Dorsalgia.yaml` as its disease_term, a
  curation error; the three d1 lines of the digest are that one node's three topics),
  then neurodevelopmental disorder (197), epilepsy (109), congenital nervous system
  disorder (104), nonsyndromic hearing loss (55), DEE (55), inherited retinal dystrophy
  (50), **dilated cardiomyopathy (49)**, hypertrophic CM (39), lymphoma (32), …,
  Ehlers-Danlos (13). DisMech lists a disease AND its subtypes as separate disorders,
  so "flat" has to be a modelling choice, not a property of the list — which is what
  spec R1 says and what `flat=True` now enforces: every member a root child; a patient
  attests the most specific member only (a generic-DCM code → DCM; a subtype code → the
  subtype, not DCM). The receipt line prints the pair count so the choice is visible.
- The cutter now EXCLUDES the root (`dismech_ribbon.EXCLUDED_MONDO_IDS`); the TSV is
  re-cut at the same DisMech commit (3,238 ids; then 3,226 after the 2026-10-08 mis-mapping exclusions — the launch-2 identity).

**2026-10-08 — two more changes before launch 2, both from the launch-1 digest.**
(i) R1a: a patient coded with both members of a nested pair (EDS + hEDS; the digest had
"Ehlers-Danlos syndrome" in hEDS's block and "EDS type 3" in EDS's) now attests the
DESCENDANT only — `member_ancestor_pairs` on the final nodes + an anti-join in the
provider; umbrellas become "not otherwise specified" nodes. (ii) The cutter excludes 15
curated mis-mappings (a specific disorder file attached to a broad Mondo term:
PGM2L1 deficiency → neurodevelopmental disorder, Mediator-complex NDD → congenital
nervous system disorder, MYO6 hearing loss → nonsyndromic hearing loss, …;
`dismech_ribbon.EXCLUDED_DISMECH_FILES`), re-cut at the same DisMech commit (new
identity). The 296 legitimate umbrella+subtype pairs stay.

**Launch 2 = the flat DAG on the re-cut ribbon**, same front matter otherwise; the
comparison column stays 0133/0134, and launch 1's digest is the nested-ribbon control
for the DCM line (under nesting DCM's block was its 49 subtypes' and peripartum's
patients; flat, DCM's block is generic-DCM-coded patients only).

## Results

(pending)
