---
id: 66
slug: dag-placement-rare6-lookback-5yr-learned-alpha
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
optimize_doc_concentration: true
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

# exp 0066 — DAG-placement rare6, 5yr lookback, LEARNED per-node alpha

Same config as exp 0062 (symmetric, 5yr lookback) **except
`optimize_doc_concentration: true`** — the learned per-node asymmetric Dirichlet
alpha at the deeper (5yr) history depth. `node_alpha_scale: 1.0` is the initial
alpha; the gated Newton step refines it. See exp 0065 for the full rationale.

## What this tests

The learned-alpha counterpart to 0062, and the deeper-history arm of the
0065/0066 pair. Same two questions as 0065: (1) what do the learned per-node
alpha values range over (driver logs them by node + name), and (2) does the
learned alpha reshape topic composition / NPMI vs 0062. With 5yr of history the
node topics are better-attested than at 1yr (0062 NPMI 0.218 > 0061 0.191), so
the learned alpha may move more nodes off the 1/K init here than at 1yr.

## Corpus reuse (fast)

Reuses 0062's cached 5yr-lookback bundle (`optimize_doc_concentration` is a fit
param, not in the cache key) — cheap re-fit. Run 0062 first (or confirm cached).

## What to read

- The `learned alpha (optimizeDocConcentration)` driver log: background vs
  per-node learned alpha, range, which nodes moved. Compare the range/spread vs
  0065 (1yr) — does more history let the learned alpha separate nodes more?
- `metrics.detection` + `make lr-readout ID=66`: expect ~0062 (null on detection).
- NPMI + per-iter topics vs 0062.
