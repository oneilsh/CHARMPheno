---
id: 52
slug: dag-placement-diabetes-random
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
init: random
max_iter: 100
seed: 42
cache_uri: hdfs:///user/dataproc/charm/case_finding_cache
---

# exp 0052 — DAG-placement diabetes case-finding (random init)

Baseline arm of the pre-registered init A/B: fit the gated-SVI hierarchical
case-finding engine on the diabetes type taxonomy (anchor 201820) + background
population, with random lambda init. Reports held-out placement AUC-by-depth,
MRR, top2 (see manifest.json). Pair: exp 0053 (spectral init).
