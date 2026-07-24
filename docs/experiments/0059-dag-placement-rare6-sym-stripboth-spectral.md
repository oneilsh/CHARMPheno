---
id: 59
slug: dag-placement-rare6-sym-stripboth-spectral
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
spectral_method: scalable
strip_mode: both
max_iter: 200
seed: 42
cache_uri: hdfs:///user/dataproc/charm/case_finding_cache
---

# exp 0059 — DAG-placement rare6 forest, sym α + strip_both, SPECTRAL init (init-axis diagnostic)

Run A of the under-training decomposition. Identical to exp 0058 (the best cell
of the rare6 2×2: symmetric α + strip_both, detection auc 0.660) in every field
EXCEPT `init: spectral` (0058 is `random`). Same cached strip_both corpus
(0057/0058 bundle), same minibatch schedule, same 200 iterations — so the only
moving part is the initialization.

## Why

The old asym/test_only rare6 run scored detection auc 0.709 at **spectral init +
full-batch**; the new-settings matched cell (0055, random + minibatch 0.1 × 200 =
~20 epochs) scored 0.585 — a 0.124 drop with TWO factors confounded (init and
effective epochs). This run isolates the **init** axis: hold minibatch/epochs at
the new-settings value, flip only init random→spectral, and read the delta
against 0058's 0.660.

The prior that spectral matters here is stronger than the diabetes arms suggested.
The only evidence for "random ≥ spectral" was diabetes (exp 0052 0.719 ≥ 0053
0.684), and diabetes is a degenerate guide for the rare regime — the α effect
flipped sign between them (symmetric wins for rare, was inert for diabetes). Rare
node topics are data-starved (Ehlers-Danlos ~160 patients, each seen ~20× total
under a 10% minibatch over 200 iters), and data-starvation is exactly where a good
anchor-word basin (Arora et al. 2013, block-aligned per gated_init.py) should help
that random + few epochs cannot reach. The persistent dead-node-topic tail at the
Σλ ≈ η·V prior floor in 0055–0058 is consistent with a bad random start as much as
with under-training.

## Reading the result

- Detection auc climbs materially toward/past 0.709 ⟹ spectral init matters for
  the rare regime; 0.660 is not the ceiling; the seeded-β baseline is higher than
  the 2×2 implied — and it motivates finally wiring the SCALABLE projected init
  (spectral_init_scalable.py) so spectral is affordable at K=180 without the
  driver-side V×V bottleneck.
- Detection auc stays ~0.66 ⟹ init is not the factor; the epoch axis (Run B,
  max_iter ~1000) is the remaining suspect, and if that is flat too the
  phenotype-non-specificity wall is real → seeded-β Monarch layer
  (project_kg_rare_disease_casefinding).

## Cost note

The DENSE block-aligned spectral init is slow at K=180 / V=10000 / person_mod=1 —
it collects the training corpus to the driver and builds ~29 V×V co-occurrence
matrices (pooled + one per surviving node), single-threaded, before iteration 1.
This is a one-off diagnostic; the slow init is expected and tolerated here.
`spectral_max_vocab: 12000` keeps the dense-path guard above V ≈ 10000.
