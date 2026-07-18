---
id: 62
slug: dag-placement-rare6-lookback-5yr
status: pending
model_class: dag_placement
cohort: population_rare6
cohort_def: population_rare6
disease: rare6
person_mod: 1
prior_obs_days: 0
doc_min_length: 10
min_n: 20
n_bg: 40
tpn: 5
print_topics_every: 10
holdout_frac: 0.2
init: spectral
node_alpha_scale: 1.0
spectral_max_vocab: 12000
spectral_method: scalable
anchor_scope: frontier
strip_mode: both
window_mode: lookback
lookback_days: 1825
label_window_days: 365
max_iter: 200
seed: 42
cache_uri: hdfs:///user/dataproc/charm/case_finding_cache
---

# exp 0062 — DAG-placement rare6 forest, LOOKBACK window (5yr pre-index)

Same corpus/schedule/config as exp 0061 (rare6 forest lookback A/B) except
`lookback_days: 1825` (5 years, vs 0061's 1 year); `label_window_days` stays
365.

## The A/B

0061 vs 0060 established forward-prediction as a genuinely harder task than
same-window recognition (see 0061's writeup). This experiment holds the
lookback mechanism fixed and varies only the depth of pre-index history
available to the feature bag: 0061 gives the model 1 year of pre-index
conditions to place a patient a year before diagnosis; 0062 gives it 5 years.

The question is whether more pre-index history recovers some of the ground
lost going from 0060's same-window setup to 0061's honest forward setup — a
longer lookback should surface more of a patient's early/prodromal signal
(comorbidities, misdiagnoses, referral chains) that a 1-year window may
truncate, at the cost of noisier/more diffuse per-node topic estimates from a
longer, more heterogeneous feature window.

As in 0061, `strip_mode` and `prior_obs_days` are moot: the lookback path's
feature/label frames are disjoint by construction (leakage-free), and the
≥1yr-prior-observation gate is intrinsic to `case_finding_index_table`, not a
separate knob to set here.

## What to read

- Compare `metrics.detection` and `metrics.auc_by_depth` against 0061 (same
  metrics, 1yr lookback). An improvement would support "more pre-index
  history helps"; a flat or worse result would suggest 1 year already
  captures most of the near-index signal, or that the longer window's extra
  heterogeneity outweighs the extra signal.
- `bundle.ledger` / `corpus_stats` feature-bag size vs 0061 — a much larger
  per-patient vocabulary is the expected mechanical effect of the longer
  window, independent of whether it helps placement.
