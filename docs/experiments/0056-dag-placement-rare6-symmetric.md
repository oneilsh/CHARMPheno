---
id: 56
slug: dag-placement-rare6-symmetric
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
strip_mode: test_only
max_iter: 100
seed: 42
cache_uri: hdfs:///user/dataproc/charm/case_finding_cache
---

# exp 0056 — DAG-placement rare6 forest, SYMMETRIC alpha (A/B baseline for 0055)

The symmetric-α baseline for the rare-disease forest, identical to exp 0055 in
every way except `node_alpha_scale: 1.0` (vs 0055's 0.1). This is the clean
symmetric-vs-asymmetric α comparison at the rare-disease regime (~5% prevalence)
— the one place the diabetes A/B could not answer, because diabetes sits at ~31%
prevalence where the "nodes are rare" prior is barely justified and was a no-op
(0053 sym detection 0.690 vs asym 0.684, essentially identical).

## Why this arm exists

The gating fix (background docs train the background block only) was the real
win across 0052–0054; block-asymmetric α on diabetes earned nothing on top of it.
But the asymmetric prior is *designed* for the low-prevalence regime, where the
overwhelming background majority should be pulled onto the background block and
any single disease node is genuinely rare. rare6 is that regime. Diffing
0055 (asymmetric) against this arm on `metrics.detection` (auc, ap, operating
points, bg_mass) answers whether the prior earns its keep when nodes really are
rare — or whether, as on diabetes, the gating fix already captured the signal and
the ceiling is phenotype non-specificity.

Shares the cached rare6 corpus with 0055 (only the fit re-runs). `strip_mode:
test_only` matches 0055 so the two differ in α alone; see the note below on the
strip_both option.
