---
id: 56
slug: dag-placement-diabetes-asym-alpha
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
init: spectral
node_alpha_scale: 0.1
max_iter: 100
seed: 42
cache_uri: hdfs:///user/dataproc/charm/case_finding_cache
---

# exp 0056 — DAG-placement diabetes, block-asymmetric alpha

A/B arm against exp 0053 (identical corpus, engine, spectral init, strip mode)
that varies ONE thing: `node_alpha_scale: 0.1`. This gives the per-disease-node
topic blocks a Dirichlet prior 10× smaller than the background block's (α_node =
0.1/K vs α_background = 1/K) — a block-asymmetric prior in the spirit of Wallach,
Mimno & McCallum (2008/2009, "Rethinking LDA: Why Priors Matter"), where an
asymmetric α over topics + a symmetric β is the robust default and lets a few
"common" topics (here, the 20 background topics) carry most mass.

## Why

Post-gating-fix diabetes runs (0053/0054) place well and separate cases from
background at detection AUC ~0.69–0.73, but several node topics still fill with
the diabetic population's generic comorbidity (depression, anxiety, back pain),
and the transform is ungated (full-K) at scoring time — so with a symmetric α a
background patient can freely put mass on those residual node topics. A smaller
node-α makes a disease node cost more evidence to invoke, which should (a)
suppress spurious node loading by background docs (raising `detection.auc` and
`bg_mass_background_mean`) and (b) reflect the low prevalence of any single node.

## What to compare

Diff `metrics.detection` (auc, ap, operating points, bg_mass) and the placement
metrics (mrr, top2, ap_macro) against 0053. Expected direction: higher detection
AUC + a larger bg_mass gap (background parks harder on the background block), with
a precision/sensitivity trade at the operating points. If it also lifts placement
mrr/top2, the asymmetric prior is a clean win; if detection rises but sensitivity
drops sharply, 0.1 is too aggressive and the scale should be tuned up (e.g. 0.3).

Note (honest caveat): lowering node-α also lowers the total Dirichlet
concentration, so θ is globally slightly sparser — the knob mixes asymmetry with
overall sparsity. Diabetes may also be near a phenotype ceiling (diabetics
resemble the comorbid general population), so the rare-disease forest (exp 0057)
is where this prior has the most leverage.
