---
id: 65
slug: dag-placement-rare6-lookback-1yr-learned-alpha
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
lookback_days: 365
label_window_days: 365
max_iter: 200
seed: 42
cache_uri: hdfs:///user/dataproc/charm/case_finding_cache
---

# exp 0065 — DAG-placement rare6, 1yr lookback, LEARNED per-node alpha

Same config as exp 0061 (symmetric, 1yr lookback) **except
`optimize_doc_concentration: true`** — turns on the learned per-node asymmetric
Dirichlet alpha (branch case-finding: `optimizeDocConcentration`, engine
`GatedOnlineLDA` + the gated Newton step). `node_alpha_scale: 1.0` is the
INITIAL alpha (symmetric); the gated per-node Newton step refines it from there.

## What this tests

Insight 0060 found the FIXED block-asymmetric alpha (0063/0064) is a null lever
for detection because the LR readout reads lambda directly and bypasses theta.
The learned per-node alpha adapts each node's concentration to the data rather
than applying one global scale — the expectation (per 0060) is that detection is
STILL null, but the interesting questions are:

1. **What do the learned alpha values range over?** The driver logs the learned
   alpha per node (with condition names) after the fit — background alpha vs each
   disease node's learned alpha, sorted. On real mass-starved node topics
   (Sigma-lambda ~55-77) the gated Newton self-regularizes: a node with little
   attested data should stay near its 1/K init, while well-attested nodes may
   move. Read the RANGE and gross ordering, not per-node point values (single-seed
   fits are multimodal — insight 0059).
2. **Does it change topic composition / NPMI?** Compare the per-iter topic log and
   the `make lr-readout ID=65` NPMI table against 0061 (symmetric). A learned
   alpha reshapes the doc-topic responsibilities, which can shift which codes load
   on which node topic.

## Corpus reuse (fast)

`optimize_doc_concentration` is a FIT parameter, not a corpus-assembly parameter,
so this run REUSES 0061's cached 1yr-lookback bundle (same cache key) and only
re-fits. Run 0061 first (or confirm its bundle is cached).

## What to read

- The `learned alpha (optimizeDocConcentration)` driver log lines: background vs
  per-node learned alpha, range, and which nodes moved off the 1/K init.
- `metrics.detection` + `make lr-readout ID=65` (LR alpha->inf ROC/PR-AUC): expect
  ~0061 (null on detection, per insight 0060) — confirm or refute.
- NPMI table + per-iter topics vs 0061: did learned alpha reshape composition?
- Pair with exp 0066 (5yr, learned alpha).
