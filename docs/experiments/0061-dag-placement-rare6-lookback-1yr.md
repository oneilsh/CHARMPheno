---
id: 61
slug: dag-placement-rare6-lookback-1yr
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
lookback_days: 365
label_window_days: 365
max_iter: 200
seed: 42
cache_uri: hdfs:///user/dataproc/charm/case_finding_cache
---

# exp 0061 — DAG-placement rare6 forest, LOOKBACK window (1yr pre-index)

Same corpus/schedule as exp 0060 (rare6 forest, spectral init, frontier-scoped
anchors, symmetric alpha, strip_both) except `max_iter: 200` (0060's sealed
200-iter A/B record, not the 1000-iter long-run overwrite) and the windowing
mode: `window_mode: lookback` with `lookback_days: 365` and
`label_window_days: 365`.

## The A/B

0060 assembles a single forward window: features and labels are drawn from
the same observation window, so a patient's index/anchor conditions can
appear in both the feature bag and the frontier label — the model is scored
on same-window recognition, not prediction.

0061 switches to the pre-index/lookback path (Task 2's
`case_finding_index_table` + `lookback_feature_label_events`, Task 3's
`assemble_case_finding_corpus(window_mode="lookback")`): features come ONLY
from the `lookback_days` (365) window strictly BEFORE each patient's index
date, and labels (the frontier) come from the `label_window_days` (365)
window strictly AFTER the index date. This is a genuine forward-prediction
setup — the model must place a patient using only what was known a year
before diagnosis.

`strip_mode` and `prior_obs_days` are moot here: in `forward` mode they exist
to prevent the label-defining conditions from leaking into the feature bag
within a single shared window (a post-hoc strip). In `lookback` mode the
feature and label frames are disjoint by construction (pre-index vs
post-index), so there is nothing to strip — leakage-free by construction.
(2026-09-01 NOTE: true for THIS experiment's `index_mode="disease"` — the
index precedes the first disease code by definition. It does NOT carry to the
whole-Mondo mainline's `index_mode="population"` random index, where prior
codes of labeled chronic conditions sit in the lookback; see the 2026-09-01
incident-episode eval program spec.)
Likewise the "at least a year of prior observation" gate that `prior_obs_days`
implements in forward mode is intrinsic to the lookback index table itself
(`case_finding_index_table` only emits an index date once the ≥1yr-prior
window is satisfied), so `prior_obs_days` has no separate effect to set.

## What to read

- Compare `metrics.detection` and `metrics.auc_by_depth` against 0060 (sealed
  200-iter numbers). Expect a real drop from same-window recognition down to
  a harder, honest forward-prediction task — that drop is the point, not a
  regression.
- `bundle.ledger` / `corpus_stats` should reflect the smaller, disjoint
  pre-index feature corpus vs 0060's single shared window.
- Paired with exp 0062 (5yr lookback), this is a two-point A/B on how much
  pre-index history is needed for useful placement.
