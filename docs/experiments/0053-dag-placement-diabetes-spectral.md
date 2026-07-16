---
id: 53
slug: dag-placement-diabetes-spectral
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
max_iter: 100
seed: 42
cache_uri: hdfs:///user/dataproc/charm/case_finding_cache
---

# exp 0053 — DAG-placement diabetes case-finding (spectral init)

Spectral arm of the pre-registered init A/B: same corpus + engine as exp 0052
but with the dense block-aligned anchor-word spectral seed (Arora et al. 2013).
Tests the user's gated-STM observation that spectral init helps on real data —
on the synthetic plants it was validated-negative (the gate already breaks
symmetry). Shares the case_finding_cache with exp 0052 (identical corpus).
