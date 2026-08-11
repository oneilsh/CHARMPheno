---
id: 70
slug: pc-mdd-antidepressant-90d-stability-multitask
status: done
model_class: pc
cohort: mdd_antidepressant
# Real knobs map 1:1 to pc_antidepressant_cloud.py flags via build_pc_args
# (key -> --key). Cohort defaults live in experiments/defaults/mdd_antidepressant.yaml;
# only per-experiment overrides need appear here.
# --- feature (fused-vocab BOW) config ---
lookback_days: 365          # pre-index feature window [index-lookback, index)
vocab_size: 2000
min_df: 20
min_patient_count: 20
person_mod: 1
# --- cohort / outcome config ---
window_days: 365            # fully-observed follow-up (must be >= stability_days)
stability_days: 90          # "the drug worked" = >=90d continuation
# (prior coverage for the new-user gate is derived from lookback_days by the
# PC driver; there is no separate --prior-obs-days flag)
grace_gap_days: 30          # permissible refill gap in the coverage stitch
# --- model / eval config ---
K: 25                       # topics (TUNE)
weight_y: 100.0             # PC prediction-constraint weight (TUNE)
alpha: 1.1
tau: 1.1
pi_iters: 100
max_iter: 500
test_frac: 0.25
seed: 0
# Documentary-only (NOT driver flags; ignored by build_pc_args): the vocab is
# fused across condition+drug+procedure and the model is joint multi-task PC
# (one shared model, per-drug heads, per-cell missing labels).
feature_domains: [condition, drug, procedure]
pc_multitask: true
run_via: make exp ID=70   # (or `make pc-antidepressant` for free-form sweeps)
---

<!-- NUMBERING NOTE: authored on branch claude/faithful-flat-pc. `main` was at
exp 0069 at authoring time, but sibling branches have claimed 0070+ under
different slugs. If one merges first, renumber this file (git mv + retitle) to the
next free experiment id — the slug disambiguates. The number is cosmetic. -->

# Experiment 0070 — PC antidepressant-stability replication (fused vocab, multi-task)

## Status: PENDING — pre-registration. Results filled in AFTER the AoU run.

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
bug in the model. Target shape from the paper (Hughes et al. AISTATS 2018,
Fig. 3 — the Antidepressant task): **avg heldout AUC across the meds ~0.60–0.65
for PC-sLDA, beating logistic regression slightly and improving reliably on the
Gibbs-LDA init**. (Their run: 11 meds, 29774/3721/3722 patients, V=5126 fused
ICD-9 dx + CPT procedure + medication codewords, PC-sLDA initialized from
Gibbs-LDA.) Do not confuse this with the group's later clinical follow-up
(*Assessment of a Prediction Model for Antidepressant Treatment Stability*, JAMA
Network Open 2020) — same method, a much larger two-site cohort, whose PC topic
model lands at AUC 0.627/0.619 (Sites A/B), i.e. the same ~0.60–0.66 band. See
`docs/hughes-comparison.md`.

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

Tracked / reproducible (params from this frontmatter + `experiments/defaults/mdd_antidepressant.yaml`):
```
cd analysis/cloud && make exp ID=70
```
Free-form sweeps (K / weight_y grids) via the standalone target:
```
cd analysis/cloud && make pc-antidepressant \
  PC_AD_ARGS='--K 25 --weight-y 100 --out runs/exp0070_results.json'
```
Both require `WORKSPACE_CDR` / `GOOGLE_CLOUD_PROJECT` (via `make setup`). Either
way the in-memory PC eval runs on the **driver** (collect-to-memory): the runner
gives PC an 8g driver (`_driver_memory_for`, overridable via `CHARM_DRIVER_MEMORY`),
puts `analysis.pc` on `PYTHONPATH` (it is not in any `--py-files` zip), and ships
`autograd` on the cluster overlay. `make exp ID=70` fits, writes `pc_results.json`
+ `summary.md` under the run dir, and skips the NPMI eval (PC has its own metrics).

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
