---
id: 64
slug: dag-placement-rare6-lookback-5yr-asym
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
lookback_days: 1825
label_window_days: 365
max_iter: 200
seed: 42
cache_uri: hdfs:///user/dataproc/charm/case_finding_cache
---

# exp 0064 — DAG-placement rare6, 5yr lookback, BLOCK-ASYMMETRIC alpha (0.1)

Byte-for-byte the same config as exp 0062 (rare6 forest, spectral init,
frontier-scoped anchors, 5yr lookback `lookback_days: 1825`, strip_both,
max_iter 200, seed 42) **except `node_alpha_scale: 0.1`** (0062 is `1.0` =
symmetric). This is the asymmetric arm at the deeper (5yr) history depth.

## The A/B

See exp 0063 for the full rationale. `node_alpha_scale: 0.1` imposes a fixed
block-asymmetric Dirichlet prior (Wallach et al. 2009): disease-node topics are
a priori rarer than background, so a document needs stronger evidence to place
mass on a node.

0062 (symmetric, 5yr) vs 0064 (asymmetric 0.1, 5yr) isolates the asymmetry axis
at the deeper history depth. Cross-read against 0061-vs-0063 (1yr): does more
pre-index history change whether asymmetry helps?

## Corpus reuse (fast)

`node_alpha_scale` is a FIT parameter, not a corpus-assembly parameter, so this
run REUSES 0062's already-assembled 5yr-lookback corpus (same bundle cache key,
shared `case_finding_cache`) and only re-fits — no BigQuery re-assembly. Run
0062 first (or confirm its bundle is cached) so 0064 is a cheap re-fit.

## What to read

- Compare `metrics.detection` / `metrics.auc_by_depth` against 0062 (same
  corpus, symmetric alpha).
- The four-cell picture (0061/0062 symmetric x 0063/0064 asymmetric, 1yr/5yr)
  answers: does a fixed block-asymmetric prior help on the honest lookback task,
  and does the answer depend on history depth. If asymmetry helps but is
  history-sensitive, that motivates the learned per-node optimizeDocConcentration
  (branch case-finding) over a single hand-set scale.
