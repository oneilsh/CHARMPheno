---
id: 63
slug: dag-placement-rare6-lookback-1yr-asym
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
node_alpha_scale: 0.1
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

# exp 0063 — DAG-placement rare6, 1yr lookback, BLOCK-ASYMMETRIC alpha (0.1)

Byte-for-byte the same config as exp 0061 (rare6 forest, spectral init,
frontier-scoped anchors, 1yr lookback, strip_both, max_iter 200, seed 42)
**except `node_alpha_scale: 0.1`** (0061 is `1.0` = symmetric). This is the
asymmetric arm of a fixed-alpha A/B.

## The A/B

`node_alpha_scale` multiplies the per-node-topic Dirichlet alpha relative to
the background alpha (1/K); `1.0` is symmetric, `< 1` down-weights the node
blocks so a document needs stronger evidence to place mass on a disease node —
a block-asymmetric prior (Wallach et al. 2009) reflecting the low prevalence
of any single rare-disease node.

0061 (symmetric) vs 0063 (asymmetric 0.1) isolates whether making the disease
nodes a priori rarer sharpens detection under the honest forward-prediction
(lookback) setup. Motivation: the earlier forward-mode rare6 runs favored
asymmetry (exp 0059 notes the old asym/test_only run scored detection AUC
0.709 vs 0.585-0.660 for symmetric cells, though init/epochs were confounded),
but the sealed lookback arms 0061/0062 reverted to symmetric — this pins the
asymmetry axis directly on the lookback corpus.

## Corpus reuse (fast)

`node_alpha_scale` is a FIT parameter, not a corpus-assembly parameter, so the
case-finding bundle cache key is identical to 0061's. With `cache_uri` pointing
at the shared `case_finding_cache`, this run REUSES 0061's already-assembled
1yr-lookback corpus and only re-fits — no BigQuery re-assembly. Run 0061 first
(or confirm its bundle is cached) so 0063 is a cheap re-fit.

## What to read

- Compare `metrics.detection` and `metrics.auc_by_depth` against 0061 (same
  corpus, symmetric alpha). A gain would say the fixed block-asymmetric prior
  helps on the lookback task; a wash would say symmetric is fine here and the
  learned optimizeDocConcentration (branch case-finding) is the lever to try
  next.
- Pair with exp 0064 (5yr lookback, asym 0.1) for the asymmetry axis at both
  history depths.
