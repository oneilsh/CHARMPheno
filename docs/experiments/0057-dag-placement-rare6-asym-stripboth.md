---
id: 57
slug: dag-placement-rare6-asym-stripboth
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
init: random
node_alpha_scale: 0.1
spectral_max_vocab: 12000
strip_mode: both
max_iter: 200
seed: 42
cache_uri: hdfs:///user/dataproc/charm/case_finding_cache
---

# exp 0057 — DAG-placement rare6 forest, asymmetric α + strip_both

The strip_both, asymmetric-α cell of the rare6 2×2 (strip_mode × α). Identical to
exp 0055 (asymmetric, test_only) except `strip_mode: both`, and identical to
exp 0058 (symmetric, strip_both) except `node_alpha_scale: 0.1`.

## The 2×2

| | strip test_only | strip both |
| --- | --- | --- |
| asymmetric α (0.1) | 0055 | **0057** |
| symmetric α (1.0) | 0056 | 0058 |

Diffing across a row isolates the strip effect; down a column isolates the α
effect. Motivation: on diabetes, strip_both beat test_only (0054 detection 0.729
vs 0053 0.690) because stripping the DAG-node type codes from TRAINING forces the
node topics to learn the surrounding footprint rather than memorizing a code that
is absent at test time — which is exactly the rare-disease case-finding goal
(find *uncoded* patients from their footprint). Asymmetric α was a no-op on
diabetes but is designed for the low-prevalence (rare) regime, so its effect on
rare6 is the open question. This cell tests both levers together; the cleanest
expected winner across the grid is strip_both, with α as the second-order knob.

Shares the cached rare6 corpus with 0055/0056/0058 (strip_mode is applied at
bundle assembly, so 0057/0058 build/cache their own strip_both bundle; the raw
BigQuery extraction + DAG prune are shared work).
