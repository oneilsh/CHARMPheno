---
id: 135
slug: dismech-ribbon-smoke-tpn3-bg1200
status: done
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

**2026-10-08 — launch 3 (planned): hierarchical re-readout of the SAVED launch-2 fit
(WP-B, ADR 0048; no refit).** The readout tool now reads a label-set fit through the
Mondo readout DAG by default (`--readout-hierarchy auto`): sibling negatives under the
nearest grouping ancestor instead of the full mask, ancestor heads for the stacked
product. On a fresh cluster the bundle rebuilds (MISS, ~20 min) and the run dir gains
`code_map.tsv` for 0136's `--own-codes` census.

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUN=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0135-dismech-ribbon-smoke-tpn3-bg1200
nohup bash -c '
  echo "=== hierarchical readout START $(date)"
  make -C analysis/cloud gated-pc-readout ID=135 GPR_ARGS="--readout-mode distributed --readout-l2 100 --readout-theta-mass 0.99"
  echo "=== hierarchical stacked START $(date)"
  make -C analysis/cloud gated-pc-readout ID=135 GPR_ARGS="--readout-mode distributed --readout-l2 100 --readout-theta-mass 0.99 --readout-stacked"
  echo "=== DONE $(date)"
' > "$RUN"/sweep3_log.md 2>&1 &
```

Receipts to pull: the `[readout dag]` lines (label nodes, ancestor heads, root aliases,
rungs dropped, depth histogram, sibling-group sizes), `frames widened ... C=`, the
solver's `observed train cells` (the number to compare with launch 2's 70.7M), the
`theta top-m by mass` resolved m, and then the usual macro/detection/stacked lines:

```bash
grep -E "^=== |readout dag|frames widened|observed train cells|converged|theta top-m by mass|gated_pc(_stacked)? \(pc_topics_lr\): (macro|detection)|root head alone|stacked: max|ranking \(within|paired per-node|paired delta|depth [0-9]+:|^all: n=|^by depth" "$RUN"/sweep3_log.md
```

Pre-registered reads for launch 3: (i) observed cells drop by an order of magnitude or
more and the solve converges for ≥ 95% of nodes; (ii) the within-cohort macro AUC is now
a sibling contrast — read it beside 0134's with that in mind, and read the stacked
marginal (de-novo) AUC as the comparable number; (iii) the ancestor heads' own AUCs by
depth (ids ≥ C_fit in `readout_dag.json`).

**2026-10-08 — launch 3 running; early receipts.** Readout DAG: 391 label nodes + 310
ancestor heads = 702 (2 root aliases folded, 73 rungs dropped), depth 0–7, 356 sibling
groups (mean 3.4, max 40), root children 4. Solver: 687 fittable, 15 degenerate,
**observed train cells 25.5M (launch 2: 70.7M; 2.8× fewer)**, ~30–35 s/iter (launch 2:
~70 s), max|grad| 8.3e4 → 1.2e4 by iter 10. The reduction is smaller than the hoped
10×: multimorbid foreground documents activate many grouping ancestors, so a typical
document observes ~20% of the 702 heads.

**θ is not spiky in mass, and that is α's floor, not the topics.** The mass resolver
read p10 coverage 0.117 at m=256 and 0.866 at m=2048, and fell back to full K. With
α=0.5 per topic over a 1,200-topic background, every allowed topic carries a floor of
α/(Σα+N) — about 600 prior pseudo-counts against documents of tens to hundreds of tokens
— so the floor is most of θ's mass whatever the token assignments do. Truncation drops
only the floor on the dropped topics (a function of document length and allowed set),
so the resolver now measures coverage of the EXCESS over the per-document floor
(`excess=True`, commit after `76e9212e`); the top-m set is unchanged. Launch 3 was
already running and stays dense; the excess rule applies from the next readout on.

## Results (launch 3, hierarchical readout of the launch-2 fit; 2026-10-09)

Readout DAG 702 heads (391 label + 310 ancestor; one pass, then the stacked pass);
ridge 100, dense θ (the mass rule fell back to full K before the excess fix); 25.5M
observed cells; stacked solve 372/688 converged at the 200-iteration cap (256 gtol, 117
stalled, max|grad| 106). 54,753 test persons, foreground prevalence 0.819 (the ribbon
covers most of the population, so detection AP is not comparable with the CV branch's).

| read | 0135 ribbon, hierarchical | 0134 CV blend (reference) |
|---|--:|--:|
| flat head, within-cohort macro AUC / AP | 0.8012 / 0.517 (559 heads) | 0.7823 / 0.512 (193) |
| root head alone (= stacked detection) | **0.8206** / 0.954 | 0.8113 / 0.890 |
| flat detection, max over heads | 0.7038 | — |
| stacked within-cohort, paired vs flat | −0.042 (95/464) | −0.036 |
| stacked de-novo, paired vs flat | **+0.074 (572/14)**, d1 +0.009 → d7 +0.150 | +0.117 |
| marginal ECE, stacked / flat, by depth | 0.002–0.010 / 0.010–0.474 | 0.003–0.016 (stacked) |

**Read.**

- **The ribbon decodes at least as well as the best nested branch.** Within-cohort
  0.801 over 559 heads (now a sibling contrast under the nearest grouping ancestor, so
  not the same question as 0134's nested cohorts) and the best root head of the series,
  0.821. Nothing was lost by taking the hierarchy out of the fit (pre-registered read
  (a)/(c), pending the shared-node de-novo pairing below).
- **The stacked product behaves exactly as on the CV branch**: it pays a within-cohort
  tax that grows with depth (−0.013 at d2 to −0.15 at d7: the ancestor factors are
  near-constant inside a cohort and only add noise) and wins de novo by a margin that
  grows with depth (+0.074 overall, 572 of 587 heads up), with marginal calibration two
  orders of magnitude better than the flat heads at depth ≥ 4. The flat heads answer
  "which sibling?", the product answers "does this patient have it?"; both are kept.
- **The de-novo gain is smaller than 0134's (+0.074 vs +0.117).** Expected: under the
  Mondo readout DAG a flat head's cohort is already a fair fraction of the foreground
  (multimorbidity; ~20% of heads observed per document), so the flat head is closer to
  a marginal than a nested CV head was.
- **Detection is the root head.** Max-over-heads (0.704) is not a detector; the stacked
  max equals the root head alone (0.8210 vs 0.8206), as on every run since 0131.

**De novo, and the pre-registered comparison with 0134 (2026-10-09).** Stacked de-novo
macro AUC 0.831 / AP 0.291 over 587 heads (label nodes median 0.849 over 286; ancestor
heads 0.834 over 301). Paired by Mondo id against 0134's stacked de novo on the **74
shared nodes**: median −0.014, mean −0.017, 23 up / 51 down. Named nodes (de novo): heart
failure 0.902, EDS 0.906, hEDS 0.948, peripartum CM 0.893, DCM 0.878, HCM 0.858,
Tako-tsubo 0.755, restrictive CM 0.738.

**Verdict: read (b), mildly.** The flat ribbon decodes the shared cardiovascular nodes
about 0.015 below the nested CV branch, just outside the ±0.01 band. Confounded in the
ribbon's disfavour in three ways the comparison cannot separate: 0134 spent its whole
node budget on ~300 CV nodes (the ribbon spreads 2,373 topics over 391 diseases from
every system), 0134's closure product multiplies CV-specific ancestor heads, and this
read is dense θ at `tpn` 3. Its size is that of every topic-side difference in
0123–0134 (insight 0095: within ~0.02) and a fifth of the decoder-side gain the stacked
product delivers on either run. Against it the ribbon buys the legibility the nested
branches never delivered (DCM's own block; insight 0096) and whole-ribbon coverage in
one fit. **Proceed to 0137 on the ribbon**; `tpn_max` 5 and the excess-mass top-m are
the two changes there that could close part of the gap, and the shared-node pairing is
re-read on 0137.

## Results (launch 2, flat DAG, re-cut ribbon; digest + census 2026-10-08)

**The fit: flat, healthy.** `K=2373 (1200 bg + 1173 node, tpn=3) · C=392` — **391 powered
members of 3,226** at `min_positives` 100; digest depth table has ONE row (d1 · 1173
topics); 3% starved; evidence min 7.6 / med 634 / p90 2.0e4 / max 4.6e5; redundancy
has nothing to score (one parent = the root).

**DCM is rescued — nesting WAS the mechanism (insight 0096).** Under every nested topic
side 0127–0134 dilated cardiomyopathy's block was its parent's young-women / pregnancy
stratum. Flat, with no parent to deflate it and peripartum patients attesting peripartum
CM only (R1a), DCM's block is textbook: `Dilated cardiomyopathy · Cardiomyopathy ·
Chronic systolic HF · … · LBBB · VT // carvedilol · furosemide · spironolactone`
(ev 1.4e4), plus an acute-decompensation profile (cardiogenic shock, effusions,
pulmonary oedema, AV regurgitation) and one small transplant/muscular-dystrophy topic.
The pregnancy vocabulary now lives entirely in peripartum cardiomyopathy's block.

**Legible blocks across the grep:** AF (2 profiles + a CM/VT one), heart failure
(systolic / acute-decompensation / CKD-anaemia comorbidity), HCM, Takotsubo, Marfan
(aortic / ocular-valvular / skeletal), CVID (IgG-replacement / subclass / sinus-asthma),
EDS (POTS-dysautonomia-GI / MCAS-immune / musculoskeletal-migraine — insight 0035's
sub-phenotypes), hEDS (pain-hypermobility / MCAS-immune / thyroid-tremor).

**R1a caveat (membership vs words).** hEDS's top word is still "Ehlers-Danlos syndrome"
and EDS's block carries "EDS, type 3" at rank 12. The first is expected: hEDS patients
CARRY the generic code as a token; R1a decides which block a patient feeds, not which
codes they have. The second needs the receipt: either the hEDS SNOMED code resolves to
the EDS term in the ladder (then R1a never sees a pair), or EDS-NOS patients carry it.
The `label set` receipt line (pair count among final nodes) and the attestation counts
decide; not pulled before the thread moved.

**Oddity to chase:** peripartum cardiomyopathy has evidence 6.0e4 — four times DCM's —
and all three topics are generic pregnancy; some common pregnancy code is climbing to
it through `source_climb`. Check the code map for MONDO peripartum CM.

**The readout did not land, and the reason is a design finding.** Flat, the `closure`
mask's "siblings as negatives" is EVERY other member: `observed train cells =
70,730,912` (every foreground doc × every node), each head a 180k-row logistic. At
`topm 0` 81/389 heads converged in 150 iterations (ill-conditioned near-constant
standardized columns, max|grad| ~300 flat); at `--readout-theta-mass 0.99` the solve
converged (max|grad| 8.8e4 → 2.1e3 in 30 iterations) but at ~70 s/iteration toward the
200 cap, ~4 h per pass; killed. The flat forest turned the within-cohort read into
"this disease vs every other ribbon disease" — pre-registered outcome (b). The
hierarchy has to supply the NEGATIVES as well as the stacked product: a readout-side
Mondo mask (siblings under the nearest Mondo ancestor with ≥ 2 members) — WP-B, now on
the critical path. No AUC column for this run; the comparison with 0133/0134 moves to
0137 under WP-B.

**Census (0136):** 96% of block topics classed signature, 91% of nodes at the tpn=3
ceiling — but the digest shows the census OVER-counts: generic-symptom strata (insomnia,
OSA, aortic stenosis, obesity: nausea · SOB · pain · vomiting // saline · ondansetron)
have background cosine 0.3–0.77 and pass the 0.8 threshold, and common diseases'
TRUE signatures (hyperlipidemia 0.71, DCM 0.50) resemble a background topic just as
much. Cosine-to-background cannot separate "a stratum" from "a common disease whose
profile is also a population stratum". The honest criterion is identity: does the topic
carry the node's OWN codes (the code-map concepts that attest it) in its top words?
Insomnia's stratum has none; its 4.5e4 topic (Insomnia · zolpidem · trazodone) does.
WP-C′: the fit writes the (std_cid, node_cid) code map to the run dir; the census marks
own-code tokens and classes a topic with none as `stratum`. Eyeballed under that rule:
EDS 3, DCM 2, HCM 2, CVID 3, Marfan 3, T2D 3, asthma 2, insomnia 2 — the ceiling is
binding for the common and the well-coded rare diseases, so **`tpn_max` 5 for the
record**, with the honest census deciding per node.
