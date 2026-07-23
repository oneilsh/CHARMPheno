---
id: 69
slug: dag-placement-rare6-1yr-reverse-topo
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
spectral_topo_order: reverse
max_iter: 200
seed: 42
cache_uri: hdfs:///user/dataproc/charm/case_finding_cache
---

# exp 0069 — reverse-topological spectral init (leaves-first) A/B vs 0067

Identical to exp 0067 (learned per-node alpha FIT, symmetric deploy, 1yr lookback,
n_bg 40, frontier anchors, scalable spectral init) **except
`spectral_topo_order: reverse`** — the spectral init recovers nodes LEAVES-FIRST,
deflating each node against its already-recovered proper-DESCENDANTS instead of its
proper-ancestors. Same corpus recipe, same fit knobs, same seed 42; only the init
seed geometry changes.

## What this isolates

The forward init (0067) makes each node's topic its INCREMENT over its ancestors:
the ancestor claims the family-generic signal, the leaf gets the subtype-specific
residual. The reverse init flips the decomposition: the leaf (most-specific, most
discriminative) is recovered first and claims its FULL defining signal, and an
ancestor's topic becomes the residual after its descendants take their share.

Hypothesis: since placement scores on the leaf/most-specific nodes, letting leaves
claim their full signal at init could sharpen exactly the topics that matter for
case-finding. This is the A/B for the leaves-first init idea.

- **0069 vs 0067** (reverse vs forward init, all else equal): placement mrr /
  auc_by_depth (does leaves-first sharpen the deep-node discrimination?), LR +
  explain-away detection (`make lr-readout ID=69`), and the error-class totals
  (background_called_rare / rare_called_background) vs 0067's 13158 / 276.
- The init only changes the STARTING lambda; 200 SVI iterations then run from it.
  If the gate + iterations wash out the init difference (as the prototype findings
  suggested — spectral did not beat random on synthetic plants because the gate
  already breaks symmetry), expect ~no change. That is a real possible outcome and
  a legitimate result: it would say the init geometry does not survive the fit for
  this data.

## Caveat / expectation

Reverse-topo is a HYPOTHESIS, not a presumed win. Spectral init (either direction)
has not beaten random init on this engine in prior tests; the DAG gate supplies most
of the identifiability. A null here (0069 ~ 0067) is informative — it would confirm
the init geometry is not the binding lever, consistent with the arc's broader finding
(insight 0062) that the binding constraint is information, not the model/init. Read
alongside the collation of all levers tried.

## What to read

- `placement metrics` mrr / top2 / auc_by_depth vs 0067 (forward). Leaf-depth AUC
  (auc_by_depth at the deepest level) is the most direct test of the hypothesis.
- `detection` + `make lr-readout ID=69 LR_ARGS="--viewer-score-mode explain_away
  --viewer-per-class 8"`: LR + explain-away ROC/PR-AUC and the error-class totals vs
  0067.
- NPMI coherence vs 0067 (0.190 / 0.172 at n_bg 40... note 0067 was 0.183/0.156;
  compare like-for-like against 0067's own run).
