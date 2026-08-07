---
id: 70
slug: pc-mdd-antidepressant-90d-stability-multitask
status: planned
model_class: pc
cohort: mdd_antidepressant
# --- feature (BOW) config ---
feature_domains: [condition, drug, procedure]   # fused into ONE vocab
lookback_days: 365          # pre-index feature window [index-lookback, index)
vocab_size: 2000
min_df: 20
min_patient_count: 20
person_mod: 1
# --- cohort / outcome config ---
window_days: 365            # fully-observed follow-up (must be >= stability_days)
stability_days: 90          # "the drug worked" = >=90d continuation
grace_gap_days: 30          # permissible refill gap in the coverage stitch
prior_obs_days: 365         # required prior coverage (new-user gate)
# --- model / eval config ---
pc_multitask: true          # ONE shared PC, per-drug heads, per-cell missing labels
K: 25                       # topics (TUNE)
weight_y: 100.0             # PC prediction-constraint weight (TUNE)
alpha: 1.1
tau: 1.1
pi_iters: 100
max_iter: 500
test_frac: 0.25
seed: 0
# NOTE: this experiment does NOT run via scripts/run_experiment.py (that runner
# supports model_class lda|stm|dag_placement with an NPMI/dashboard lifecycle).
# PC has a different, in-memory eval lifecycle. Run it with:  make pc-antidepressant
run_via: make pc-antidepressant
---

<!-- NUMBERING NOTE: authored on branch claude/faithful-flat-pc. `main` was at
exp 0069 at authoring time, but sibling branches have claimed 0070+ under
different slugs. If one merges first, renumber this file (git mv + retitle) to the
next free experiment id — the slug disambiguates. The number is cosmetic. -->

# Experiment 0070 — PC antidepressant-stability replication (fused vocab, multi-task)

## Status: PLANNED — pre-registration. Results filled in AFTER the AoU run.

Per our convention, an experiment doc is written when it is designed; the
**Results** section stays empty until the driver has actually run on All-of-Us.
Everything above the Results heading is the pre-registered design.

## Goal

Replicate Hughes, Hope, Weiner, McCoy, Perlis, Sudderth & Doshi-Velez (2017/2018)
— can a Prediction-Constrained topic model predict, from a patient's pre-treatment
code history, whether a specific antidepressant will "work" (be sustained ≥90
days) — **on All-of-Us OMOP**. The machine is trusted independently of the data
(Phase A: our objective reproduces the authors' published PC trade-off at their
own optima; see ADR 0038), so a null here is a *data* finding about AoU, not a
bug in the model. Target shape from the paper: **per-drug AUC ~0.67–0.71 for PC
vs ~0.55–0.64 for logistic regression**.

## Cohort — `mdd_antidepressant`

Major-depression patients who are **incident new-users** of an antidepressant
(index = first antidepressant `drug_era`), MDD-restricted, with `prior_obs_days`
of prior coverage and a fully-observed `window_days` follow-up
(`apply_mdd_antidepressant_cohort`). 15 antidepressants across SSRI / SNRI / TCA /
atypical (`_DRUG_REGISTRY`). Record N and per-drug positive rate at run time.

## Features — fused multi-domain BOW

One bag of OMOP concept "words" fused across **condition + drug + procedure**
(one flat vocabulary; `load_omop_bigquery(concept_types=(condition,drug,procedure))`),
restricted to the pre-index window `[index − lookback_days, index)` via
`lookback_feature_label_events`, vectorized by `to_bow_dataframe`.

## Outcome — per-drug ≥90-day stability

`antidepressant_stability_label`: the index ingredient has continuous coverage
≥ `stability_days` from index (consecutive same-ingredient eras stitched across
gaps ≤ `grace_gap_days`); a switch to a *different* antidepressant within the
window is a failure. Uncensored under the follow-up gate.

## Model — joint multi-task PC

ONE shared PC topic model with a per-drug outcome head. Each patient is labeled
only for the drug they initiated → a D×C label matrix with one observed cell per
row (`evaluate_pc_multitask`, per-cell `label_mask`). Rare drugs borrow strength
from the shared representation.

## Baselines (same masking)

Two-stage (unsupervised topics `weight_y=0` → per-drug logistic regression on the
frozen representation) and logistic-regression-on-codes. Per-drug heldout ROC AUC
+ AP, macro-averaged over non-degenerate drugs.

## How to run (AoU Dataproc master)

```
cd analysis/cloud && make pc-antidepressant \
  PC_AD_ARGS='--K 25 --weight-y 100 --out runs/exp0070_results.json'
```
Requires `WORKSPACE_CDR` / `GOOGLE_CLOUD_PROJECT` (via `make setup`). The in-memory
PC eval runs on the **driver** (collect-to-memory) — hence `--driver-memory 8g`
and `autograd` on the cluster overlay (`cluster-requirements.txt`). `analysis.pc`
is put on `PYTHONPATH` by the target (it is not in any `--py-files` zip).

## To tune before trusting a number

- **`K` and `weight_y` are placeholders** — Hughes swept these; sweep with small
  `PC_AD_ARGS` variations and read the macro-AUC.
- **Verify concept ids** — the `_DRUG_REGISTRY` / MDD `_DISEASE_REGISTRY` entries
  carry `VERIFY ON FIRST RUN` comments; check the `concept_ancestor` counts against
  the live CDR before trusting cohort N.
- **Driver memory** — dense `D×V` at collect time; raise `--driver-memory` or lower
  `person_mod` / `vocab_size` if the driver OOMs on a large cohort.

## Results

_TBD — awaiting the All-of-Us run. Record: N, per-drug positive rate, the per-drug
AUC table (PC vs two-stage vs LR), macro-AUC, and the effective merged config.
Then write the interpretation as an insight (Phase C3): with Phase A passed, a
PC-beats-baselines result replicates Hughes on AoU; a null is an AoU
med-completeness / cross-system-leakage finding, not a model failure._
