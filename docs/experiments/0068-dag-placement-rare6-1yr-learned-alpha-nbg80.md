---
id: 68
slug: dag-placement-rare6-1yr-learned-alpha-nbg80
status: done
model_class: dag_placement
cohort: population_rare6
cohort_def: population_rare6
disease: rare6
person_mod: 1
prior_obs_days: 0
doc_min_length: 10
min_n: 20
n_bg: 80
tpn: 5
print_topics_every: 10
holdout_frac: 0.2
init: spectral
node_alpha_scale: 1.0
optimize_doc_concentration: true
transform_alpha_mode: symmetric
spectral_max_vocab: 12000
spectral_method: scalable
anchor_scope: frontier
strip_mode: both
window_mode: lookback
lookback_days: 365
label_window_days: 365
max_iter: 200
seed: 42
cache_uri: gs://dataproc-staging-getting-started-with-registered-tier-data-copy/charm/case_finding_cache
---

# exp 0068 — double the background capacity (n_bg 40 -> 80)

Identical to exp 0067 (learned per-node alpha FIT, symmetric deploy, 1yr lookback)
**except `n_bg: 40 -> 80`** — twice the background topics, everything else held
(same corpus/cache, same fit recipe, same seed 42). K grows 170 -> 210
(80 bg + 26 nodes x 5 tpn = 80 + 130).

## What this isolates

The 0067 explain-away readout exposed that the binding error is NOT the
comorbidity-drag false-negative class (routing left it unmoved; those patients
are information-limited, not drag-limited) but the false-POSITIVE class:
~44% of background patients were called rare at 80% sensitivity
(background_called_rare = 13158 / 29595). Pixel-peeping those FPs showed generic
COMORBIDITY CLUSTERS with no background home leaking into the nearest rare node:
lung/abdomen -> Sarcoidosis, anemia + CKD -> SLE, thyroid + injury -> Lichen
amyloidosis.

Both the plain-LR base rate and the explain-away routing can only send a comorbid
code "to background" if a BACKGROUND topic actually claims it. With n_bg=40 the
background repertoire is too small to absorb these clusters, so they land on a
node instead. This run tests the direct lever: does doubling background capacity
give those comorbidity clusters a home and shrink the FP class?

- **0068 vs 0067** (n_bg 80 vs 40, all else equal): the headline is the FP class
  and bg_fpr at fixed sensitivity. Expect background_called_rare to drop and
  precision@{80,90}% sens to rise IF the FPs are capacity-limited (a background
  topic now out-competes the node for the comorbid cluster). If the FPs are
  instead "genuinely disease-like background" patients, more bg topics will NOT
  help and the FP class holds -> that distinguishes the two FP sub-causes.
- **explain-away @ n_bg=80**: the explain-away detection block should benefit MORE
  than plain LR from added capacity (its FP fix is explicitly n_bg-bounded per the
  scorer spec), so watch whether the explain-away-vs-LR gap narrows or flips at
  n_bg=80. This is the pre-registered capacity test for the routing scorer.

## Cache: full re-extract (NOT a reuse of 0067)

`n_bg` is part of the CaseFindingBundle cache key
(analysis/cloud/_case_finding_cache.py compute_bundle_cache_key), because the
assembled bundle stores a DagLayout built with n_bg
(charmpheno.omop.case_finding_assembly). So n_bg 40 -> 80 is a cache MISS: this run
re-extracts from BigQuery, refits CountVectorizer, and rebuilds/prunes the DAG,
then fits at K=210 (deterministic, seed 42) with spectral init seeding 80
background anchors + the same 26 node frontier. Expect a full pipeline run, longer
than 0067 (both re-assembly and more topics).

(Aside: the document BOW / vocab / split are actually n_bg-INDEPENDENT — n_bg only
sizes the topic bookkeeping, not the corpus content — so the key could in principle
exclude n_bg to allow corpus reuse across n_bg sweeps. Not changed here; noted as a
possible future cache-key refinement if we sweep n_bg often.)

## What to read

- `detection` + `make lr-readout ID=68 LR_ARGS="--viewer-score-mode explain_away
  --viewer-per-class 8"`: LR and explain-away ROC/PR-AUC, and especially bg_fpr /
  precision at 80-95% sensitivity vs 0067.
- error-class viewer totals: background_called_rare (the FP class) vs 0067's 13158
  is the headline number. rare_called_background (FN, 276 in 0067) expected ~flat
  (info-limited, unaffected by bg capacity).
- The three FP signatures above (lung/abdomen, anemia+CKD, thyroid+injury): do
  they now route to a background topic (drop out of the rare calls)?
- NPMI coherence: more/smaller background topics may raise or fragment coherence;
  watch mean/median vs 0067 (0.183 / 0.156).

## Result — NULL on the FP class (see insight 0062)

Doubling background capacity did NOT reduce the false positives. Error-class totals
vs 0067: background_called_rare 13158 -> 12992 (-1.3%, noise); rare_called_background
276 -> 276 (identical); bg_fpr @80% sens 0.406 -> 0.439 (if anything worse). Detection
flat-to-slightly-worse: LR ROC 0.778 -> 0.770, explain-away 0.767 -> 0.766, theta-mass
0.647 -> 0.657; all moves <= ~0.01 = single-seed re-fit noise. Placement mrr 0.596 ->
0.632 (re-fit side effect, not the FP target). Learned alpha again tracks footprint
diffuseness not prevalence (Spearman(alpha,coverage)=0.09).

Mechanism (insight 0062): the FP codes are GENUINELY disease-associated (anemia of
chronic disease, CKD = lupus nephritis are real SLE features), not comorbid noise a
background topic could claim away, so from condition codes alone a background patient
with anemia+CKD is indistinguishable from SLE. Capacity is the wrong lever; the binding
constraint is INFORMATION (meds/labs = MixEHR direction). explain-away's gap to LR did
narrow (0.0106 -> 0.0044) as pre-registered (its FP fix is n_bg-bounded) but immaterial
since LR itself did not improve. n_bg=80 NOT adopted.
