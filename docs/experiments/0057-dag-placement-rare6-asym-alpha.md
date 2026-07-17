---
id: 57
slug: dag-placement-rare6-asym-alpha
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
spectral_max_vocab: 12000
node_alpha_scale: 0.1
max_iter: 100
seed: 42
cache_uri: hdfs:///user/dataproc/charm/case_finding_cache
---

# exp 0057 — DAG-placement rare6 forest, block-asymmetric alpha

A/B arm against exp 0055 (identical rare-disease-forest corpus, engine, spectral
init) varying ONE thing: `node_alpha_scale: 0.1` — the per-node-topic Dirichlet
prior is 10× smaller than the background block's (Wallach et al. 2008/2009
asymmetric-α). See exp 0056 for the full rationale.

## Why this is the decisive arm

The rare6 forest is where an asymmetric prior should matter most: prevalence is
~5% (n_fg ≪ n_bg), so a symmetric α gives the ~64% of topics that are disease
nodes no prior penalty even though almost no patient belongs to any of them. A
smaller node-α encodes "any single rare disease is rare" and, at ungated
transform time, keeps the overwhelming background majority parked on the
background block instead of drifting onto disease nodes.

This runs on top of the gating fix (background docs now train the background
block only). If exp 0055 (symmetric) already separates cases from background well
after that fix, this arm tests whether the asymmetric prior sharpens it further;
if 0055 detection is still soft, this is the cheap next lever before the heavier
seeded-β Monarch-profile layer. Diff `metrics.detection` + placement metrics
against 0055.
