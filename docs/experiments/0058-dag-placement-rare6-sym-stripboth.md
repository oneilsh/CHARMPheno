---
id: 58
slug: dag-placement-rare6-sym-stripboth
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
strip_mode: both
max_iter: 200
seed: 42
cache_uri: hdfs:///user/dataproc/charm/case_finding_cache
---

# exp 0058 — DAG-placement rare6 forest, symmetric α + strip_both

The strip_both, symmetric-α cell of the rare6 2×2 (strip_mode × α). Identical to
exp 0056 (symmetric, test_only) except `strip_mode: both`, and identical to
exp 0057 (asymmetric, strip_both) except `node_alpha_scale: 1.0`.

## The 2×2

| | strip test_only | strip both |
| --- | --- | --- |
| asymmetric α (0.1) | 0055 | 0057 |
| symmetric α (1.0) | 0056 | **0058** |

Diffing across a row isolates the strip effect; down a column isolates the α
effect. This is the symmetric-α baseline at the (expected-better) strip_both
setting: if the diabetes result transfers, this cell should beat 0056 (symmetric,
test_only) on detection purely from the strip change, independent of α. The
0057-vs-0058 column then reports whether asymmetric α adds anything on top of
strip_both in the rare regime.

Shares the cached rare6 corpus extraction with 0055/0056/0057 (strip_both bundle
built/cached jointly with 0057).
