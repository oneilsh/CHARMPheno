---
id: 52
slug: dag-placement-diabetes-random
status: pending
model_class: dag_placement
cohort: population_diabetes
cohort_def: population_diabetes
person_mod: 10
prior_obs_days: 365
disease: diabetes
min_n: 50
n_bg: 20
tpn: 5
print_topics_every: 10
holdout_frac: 0.2
init: random
node_alpha_scale: 0.1
max_iter: 200
seed: 42
cache_uri: hdfs:///user/dataproc/charm/case_finding_cache
---

# exp 0052 — DAG-placement diabetes case-finding (random init)

Baseline arm of the pre-registered init A/B: fit the gated-SVI hierarchical
case-finding engine on the diabetes type taxonomy (anchor 201820) + background
population, with random lambda init. Reports held-out placement AUC-by-depth,
MRR, top2 (see manifest.json). Pair: exp 0053 (spectral init).

## Prior: block-asymmetric α (node_alpha_scale: 0.1)

This run uses a block-asymmetric Dirichlet prior over topics — the per-node-topic
blocks get α_node = 0.1/K vs α_background = 1/K (Wallach, Mimno & McCallum
2008/2009). A disease node costs ~10× more evidence to invoke, which suppresses
spurious node loading by background docs at ungated transform time and reflects
the low prevalence of any single node. Applied across the 0052+ dag_placement
batch (the engine default stays symmetric, 1.0). Compare `metrics.detection` and
placement metrics against the symmetric post-fix diabetes numbers already on
record (0053 detection 0.690 / 0054 0.729); rare6 (0055) has no symmetric
post-fix baseline, so read it against the qualitative expectation + the bug-era
0.532 floor.
