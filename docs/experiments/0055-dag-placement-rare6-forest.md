---
id: 55
slug: dag-placement-rare6-forest
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
spectral_max_vocab: 12000
strip_mode: test_only
max_iter: 100
seed: 42
cache_uri: hdfs:///user/dataproc/charm/case_finding_cache
---

# exp 0055 — DAG-placement rare-disease forest case-finding

The strategic pivot away from the diabetes type taxonomy (exps 0052–0054),
which failed on its own terms: diabetes types are near-identical phenotypes,
and with roughly a third of the corpus in the foreground the "background" was
itself heavily diabetic. This experiment replaces the single-disease taxonomy
with a **six-disease rare-disease forest** against a large, genuinely clean
whole-population background — the rare-disease case-finding thesis
(project_kg_rare_disease_casefinding) run in practice.

## Foreground: six phenotypically distinct rare diseases

`disease: rare6` unions the descendants of six OMOP anchors into one foreground
arm and reuses those same six anchors as the label-DAG roots (each becomes a
subtree under a synthetic forest root):

| disease | anchor | approx. AoU count |
| --- | --- | --- |
| Ehlers-Danlos syndrome | 79145 | ~160 |
| Sarcoidosis | 438688 | few 1000s (AoU-over-represented) |
| Systemic lupus erythematosus | 257628 | ~6500 (best-powered) |
| Scleroderma / systemic sclerosis | 40352976 | few 1000s |
| Myasthenia gravis | 76685 | ~1100 |
| Amyloidosis | 432595 | ~1500 |

Because the six diseases are clinically distinct (unlike diabetes subtypes),
placement is a genuine two-level task: (1) find cases against the background,
and (2) route each to the right disease subtree. A patient coded for more than
one of the six gets a set-valued (multi-disease) frontier.

## Settings and why they differ from the diabetes arms

- `person_mod: 1` — the WHOLE population, not the 1/10 sample. The rare
  foreground is a tiny fraction of the corpus, so a clean, large background is
  the point (mirrors exp 0030's EDS-on-population settings).
- `min_n: 20` — a node needs ≥ 20 attesting TRAIN patients to survive pruning.
  Lower than the diabetes arms' 50 because rare-disease subtypes are sparse;
  several disease anchors may collapse to a single node (their subtypes pruned),
  which is the correct behavior for case-finding-vs-background.
- `prior_obs_days: 0` — admit prevalent cases (no prior-coverage lookback), so
  rare patients diagnosed near their record start are not silently dropped.
- `doc_min_length: 10` — drop near-empty windows.
- `n_bg: 40` — a larger shared background block for the whole-population
  reference (vs 20 in the diabetes arms).
- `tpn: 5` — five topics per surviving DAG node (matching the diabetes arms for
  comparability; a knob to revisit if per-node topics look redundant).
- `init: spectral` — the dense block-aligned anchor-word seed (Arora et al.
  2013), which won every placement metric over random init on the diabetes arms.
  `strip_mode: test_only` (the default) — the leakage strip removes DAG-node
  type codes from held-out documents only.
- `spectral_max_vocab: 12000` — raises the gated shim's dense-spectral guard
  above the shared `vocab_size: 10000`. The gated engine currently has only the
  DENSE block-aligned init (a V×V driver co-occurrence, ~800 MB at V=10000);
  its default cap is a conservative 8000. STM ran dense spectral at this same
  V=10000 (exp 0030), so the driver handles it. The scalable projected init
  (which sidesteps the V×V matrix) exists in spark_vi but is not yet wired into
  the gated shim; building it is the clean fix for larger vocabularies. If the
  driver OOMs on the co-occurrence, fall back to `init: random`.

K is emergent (`n_bg` + surviving-DAG-nodes × `tpn`), so there is no `K` field.

## What success looks like

Distinct phenotypes should make the per-disease AUC high (routing a held-out
case to its disease subtree is easy when the diseases don't overlap), a sharp
contrast to the diabetes arms where near-identical types capped `ap_macro`
around 0.19.

The headline for deployment is the new `metrics.detection` block — the
foreground-vs-background question, since in practice the vast majority of
patients scored will NOT be in any rare-disease group. It reports case-vs-
background ROC/PR-AUC and, at 80/90/95% sensitivity, the background false-
positive rate and precision (with ~5% prevalence, precision is the demanding
number: catching the true cases without flooding on false flags). The
`bg_mass_*` fields confirm background patients park their topic mass on the
background block rather than on disease nodes. Also watch `test_coarsening_rate`
(how much the test foreground was rolled up by a train-pruned DAG) and the
corpus stats' rare6-vs-general split to confirm the background is clean. Shares the `case_finding_cache` with the
other dag_placement runs; `disease` is folded into the cache key, so this
corpus is built and cached under its own key.
