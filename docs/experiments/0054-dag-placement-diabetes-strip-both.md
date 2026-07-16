---
id: 54
slug: dag-placement-diabetes-strip-both
status: pending
model_class: dag_placement
cohort: population_diabetes
cohort_def: population_diabetes
person_mod: 10
prior_obs_days: 365
anchor: 201820
min_n: 50
n_bg: 2
tpn: 1
holdout_frac: 0.2
init: spectral
strip_mode: both
max_iter: 100
seed: 42
cache_uri: hdfs:///user/dataproc/charm/case_finding_cache
---

# exp 0054 — DAG-placement diabetes case-finding (strip-both leakage ablation)

Leakage-strip ablation arm against exp 0053: same corpus, same spectral init
(the dense block-aligned anchor-word seed, Arora et al. 2013), same engine —
but with `strip_mode: both` instead of the default `test_only`. Where 0053
only strips the DAG-node type codes from held-out (test) documents (the
minimum needed so evaluation can't read a patient's own label off its
features), this arm ALSO strips those codes from the TRAIN documents.

The question this arm answers: does letting the model see the DAG-node type
codes during training inflate placement performance via shortcut learning
(the model just memorizing "code X co-occurs with node X's frontier" rather
than learning from the surrounding clinical vocabulary)? If 0054's placement
metrics (ap_macro, ap_prevalence_weighted, recall_at_k, node_auc) are
substantially lower than 0053's under an otherwise-identical config, that is
evidence the train-side codes are doing more than incidental co-occurrence
work. A third arm — codes supervise the gate but are excluded from the topic
(β) sufficient statistics — would isolate this more precisely; it requires a
deeper GatedOnlineLDA change and is deferred (see
docs/superpowers/plans/2026-07-16-placement-eval-rigor.md, Task 4).

Shares the case_finding_cache with 0052/0053 for the corpus (`strip_mode` is
folded into the cache key, so the bundle itself is rebuilt/cached separately
under its own key — only the raw assembly inputs are shared work).
