---
id: 67
slug: dag-placement-rare6-1yr-learned-alpha-deploy-symmetric
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
transform_alpha_mode: symmetric
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

# exp 0067 — learned alpha FIT, but SYMMETRIC deploy (decoupled transform prior)

Identical to exp 0065 (learned per-node alpha, 1yr lookback) **except
`transform_alpha_mode: symmetric`** — the fold-in / deployment uses a flat 1/K
prior instead of the fitted asymmetric alpha. Same fit (deterministic, seed 42),
different deployment prior.

## What this isolates

Insight 0061 saw the learned-alpha run (0065) drop cross-node argmax ranking
(mrr 0.585->0.442) while per-node discrimination and detection were unchanged.
The hypothesis: that drop is the mechanical effect of the fitted asymmetric alpha
acting as an informative prior at fold-in (up-ranking high-alpha / diffuse-footprint
nodes), NOT a worse model. This run tests it directly:

- **0067 vs 0065** (same learned fit; symmetric vs fitted DEPLOY): does mrr/top2
  RECOVER toward the symmetric baseline when the deployment prior is neutralized?
  If yes -> the 0065 drop was the deploy-prior bias, not the fit.
- **0067 vs 0061** (both deploy symmetric; learned vs symmetric FIT): does learning
  alpha during the fit HELP anything once the deployment prior is neutralized (via
  better beta / cleaner topics)? Expectation per insight 0060 (LR detection is
  alpha-invariant): little to no change -> the learned alpha is a fitting aid with
  no downstream benefit for this task.

## Corpus reuse (fast fit) / caveat

Reuses 0065's cached 1yr corpus. NOTE: transform_alpha_mode only changes the
DEPLOYMENT fold-in, which happens inline after the fit; the fit itself is
re-run (deterministic, identical to 0065). A saved-model re-transform readout would
avoid the re-fit; not built (kept minimal).

## What to read

- `placement metrics` mrr / top2 / auc_by_depth vs 0065 (fitted deploy) and 0061
  (symmetric fit+deploy). The mrr/top2 recovery (or not) is the headline.
- `detection` + `make lr-readout ID=67`: expect ~unchanged (LR reads lambda, and
  the fit is the same as 0065).
- Optional follow: transform_alpha_mode: block_balanced with a high
  transform_bg_weight (e.g. 0.9) to test the background-vs-node baseline.
