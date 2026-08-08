---
id: 71
slug: pc-vi-mdd-antidepressant-stability
status: pending
model_class: pc
cohort: mdd_antidepressant
backend: vi                  # distributed VI-native PCEstimator (SVI), not in-mem L-BFGS
# Real knobs map 1:1 to pc_antidepressant_cloud.py flags via build_pc_args
# (key -> --key). Cohort defaults live in experiments/defaults/mdd_antidepressant.yaml;
# only per-experiment overrides need appear here. build_pc_args passes the SVI
# schedule knobs (subsampling_rate/tau0/kappa) ONLY because backend==vi.
# --- feature (fused-vocab BOW) config ---
lookback_days: 365          # pre-index feature window [index-lookback, index)
vocab_size: 2000
min_df: 20
min_patient_count: 20
person_mod: 1
# --- cohort / outcome config ---
window_days: 365            # fully-observed follow-up (must be >= stability_days)
stability_days: 90          # "the drug worked" = >=90d continuation
grace_gap_days: 30          # permissible refill gap in the coverage stitch
# --- model / eval config ---
K: 25                       # topics (SWEEP)
weight_y: 100.0             # PC prediction-constraint weight (SWEEP)
alpha: 1.1                  # theta Dirichlet concentration (-> PCEstimator docConcentration)
tau: 1.1                    # baseline (in-mem two-stage) topic Dirichlet; VI eta = 1/K
max_iter: 200               # SVI global iterations (~10 passes at subsampling 0.05;
                            # 500 was ~5h wall-clock, 200 lands a trained head in ~2h.
                            # Raise later if the fit still looks under-converged.)
test_frac: 0.25
seed: 0
# --- distributed-SVI schedule (backend: vi only) ---
subsampling_rate: 0.05      # mini-batch fraction per SVI iteration (-> --subsampling-rate)
tau0: 64.0                  # Robbins-Monro learning offset (-> --tau0)
kappa: 0.6                  # Robbins-Monro learning decay in (0.5, 1.0] (-> --kappa)
# warm_start_unsup_iters: 0 # unsup warm-start (Hughes): 0=cold start; N>0 = phase-1
                            # weight_y=0 topic warm-up then fresh-RM supervised phase 2
                            # (-> --warm-start-unsup-iters). See "Warm-start A/B" below.
# Documentary-only (NOT driver flags; ignored by build_pc_args): the vocab is
# fused across condition+drug+procedure and the model is joint multi-task PC
# (one shared model, per-drug heads, per-cell missing labels).
feature_domains: [condition, drug, procedure]
pc_multitask: true
run_via: make exp ID=71
---

<!-- NUMBERING NOTE: authored on branch claude/faithful-flat-pc, as the VI-backend
sibling of 0070 (pc-mdd-antidepressant-90d-stability-multitask). `main` was below
0070 at authoring time, but sibling branches have claimed 0071+ under different
slugs. If one merges first, renumber this file (git mv + retitle) to the next free
experiment id — the slug (pc-vi-...) disambiguates. The number is cosmetic. -->

# Experiment 0071 — VI-native PC antidepressant-stability replication (distributed SVI)

## Status: PENDING — pre-registration. Results filled in AFTER the AoU run.

Per our convention, an experiment doc is written when it is designed; the
**Results** section stays empty until the driver has actually run on All-of-Us.
Everything above the Results heading is the pre-registered design.

## Goal

Same clinical question as **0070** — can a Prediction-Constrained topic model
predict, from a patient's pre-treatment code history, whether a specific
antidepressant will "work" (be sustained ≥90 days) — but fit with the
**distributed VI-native PC** (`--backend vi`) instead of the in-memory L-BFGS PC.

Why: on the full cohort the in-memory L-BFGS fit **under-converged** — the
logistic head stayed pinned at its zero init (`|w_CK|max ≈ 0`), so every drug's
heldout AUC came out at exactly 0.5. The VI-PC (`spark_vi.mllib.topic.pc.PCEstimator`,
SVI, distributed) fits the head with a Robbins-Monro schedule and never collects
the corpus to the driver, so it is the path meant to actually train the head at
full-cohort scale. This experiment is the first full-cohort VI-PC run; the
`vi_convergence` block in `pc_results.json` (final ELBO, `n_iter`, `|w_CK|max`) is
the primary thing to read before trusting any AUC.

Target shape from Hughes et al. (unchanged from 0070; AISTATS 2018 Fig. 3):
**avg heldout AUC across the meds ~0.60–0.65 for PC-sLDA, beating logistic
regression slightly and improving reliably on the Gibbs-LDA init** — NOT the
0.67–0.71 of the later JAMA-Psychiatry 2020 follow-up.

## Cohort / features / outcome — identical to 0070

The VI backend REUSES the entire cohort/outcome/feature pipeline of 0070
(`apply_mdd_antidepressant_cohort` → `antidepressant_stability_label` → fused
condition+drug+procedure BOW over the pre-index window). See 0070 for the cohort,
feature, and outcome definitions — only the PC **fit backend** changes here.

## Model — joint multi-task VI-PC (distributed)

ONE shared `PCEstimator` with `numLabels = C` per-drug outcome heads, `weightY > 0`.
The per-patient label + mask are attached to the BOW DataFrame as `ArrayType`
columns (one observed cell per row = the index drug), the corpus is split by person
at the DataFrame level, and the estimator fits distributed SVI (no collect). The
head's per-label `sigmoid(w_CK·θ)` (`probabilityCol`) is scored per-drug on the
heldout split.

## Baselines (same masking, collected to memory)

Identical to 0070 and computed with the SAME code
(`analysis.pc.evaluate.multitask_baseline_probas`), so the VI-PC number is
comparable to the same baselines: two-stage (unsupervised topics `weight_y=0` →
per-drug masked logistic regression on the frozen representation) and
logistic-regression-on-codes. Per-drug heldout ROC AUC + AP, macro-averaged over
non-degenerate drugs.

## Knobs to sweep before trusting a number

- **`K` and `weight_y`** — placeholders, as in 0070. Sweep and read macro-AUC.
- **SVI schedule (`subsampling_rate`, `tau0`, `kappa`)** — the head's ability to
  leave zero depends on the Robbins-Monro step. The MLlib default `tau0=1024` is
  glacial on a cohort this size (ρ₀ ≈ 0.03); this doc starts at `tau0=64`,
  `kappa=0.6`. If `vi_convergence.w_CK_absmax ≈ 0` after a run, lower `tau0`
  further (≈10–32) and/or raise `weight_y`; if the ELBO/head diverges, raise
  `tau0` or reach for the estimator's `head_lr_scale` / `weight_y_warmup_iters`.
- **Unsupervised warm-start (`warm_start_unsup_iters`)** — Hughes et al. seed the
  supervised PC fit from the topics of an *unsupervised* fit (they used Gibbs-LDA;
  our analogue is a `weight_y=0` SVI phase). `0` (default) = cold start. `N > 0`
  runs PHASE 1 (`weight_y=0`, N iters — learns topics, head stays at zero) then
  warm-starts PHASE 2 (the real supervised fit, `--max-iter` iters) from those
  topics with a **fresh** Robbins-Monro schedule so the head trains against an
  undecayed ρ (a `--resume`-style decayed ρ would leave the head barely moving —
  this knob is deliberately distinct from `--resume-from`). If a cold fit's
  `w_CK_absmax` is low or its topics look label-agnostic, try `warm_start_unsup_iters:
  50` and compare. **A/B (warm vs cold):** run two experiments differing ONLY in
  this knob (`0` vs e.g. `50`); `warm_start_unsup_iters=0` is byte-for-byte the
  single-phase fit, so the comparison isolates the warm-start's effect on the
  head trajectory and per-drug AUC. Phase 1 does NOT checkpoint to the run dir
  (it is a warm-up); `--save-dir` still checkpoints the real phase-2 fit, and
  resume skips phase 1 (it continues an existing phase-2 fit).

- **Verify concept ids** — the `_DRUG_REGISTRY` / MDD `_DISEASE_REGISTRY` entries
  carry `VERIFY ON FIRST RUN` comments; check `concept_ancestor` counts against
  the live CDR before trusting cohort N.

## How to run (AoU Dataproc master)

Tracked / reproducible (params from this frontmatter +
`experiments/defaults/mdd_antidepressant.yaml`):
```
cd analysis/cloud && make exp ID=71
```
Free-form sweeps via the standalone target (note `--backend vi` + the SVI knobs):
```
cd analysis/cloud && make pc-antidepressant \
  PC_AD_ARGS='--backend vi --K 25 --weight-y 100 --subsampling-rate 0.05 \
              --tau0 64 --kappa 0.6 --out runs/exp0071_results.json'
```
Both require `WORKSPACE_CDR` / `GOOGLE_CLOUD_PROJECT` (via `make setup`). The VI
fit runs distributed on executors (the head's autograd is contained to the model
core and never crosses the partition boundary); only the two baselines collect the
dense `D×V` matrix to the driver, so the 8g driver (`_driver_memory_for`,
overridable via `CHARM_DRIVER_MEMORY`) still applies. `make exp ID=71` fits, writes
`pc_results.json` + `summary.md` under the run dir, and skips the NPMI eval (PC has
its own metrics).

### Resume + eval-from-checkpoint (VI backend only)

The VI-native `PCEstimator` checkpoints its `VIResult` every `save_interval` SVI
iters into the run dir (`manifest.json` + `params/`), so the fit is resumable with
the SAME UX as the LDA/HDP models (`run_experiment.py` detects the checkpoint by
`manifest.json` and threads `--resume-from`):

- **Resume (continue training):** re-run `make exp ID=71`. It detects the existing
  checkpoint and continues from the last saved iteration — `--max-iter` is then
  **additional** iters on top of the loaded count (e.g. a killed 200-iter run
  resumed with `max_iter: 200` reaches ~400 total). A corpus-config change between
  runs (person_mod / lookback / window / stability / grace-gap / min_df /
  min_patient_count) is refused by `check_resume_compat` — revert the config or
  `rm -rf` the run dir to start fresh.
- **Eval from a checkpoint (no training):** peek the per-drug AUC without more fit:
  ```
  cd analysis/cloud && make pc-antidepressant \
    PC_AD_ARGS='--backend vi --eval-only --save-dir <run_dir> --out <run_dir>/pc_results.json'
  ```
  `--eval-only` loads the checkpoint into a `PCModel`, reads the drug→column order
  from the checkpoint metadata, and runs the existing transform + per-drug scoring.

Checkpoint/resume/`--eval-only` are **VI-only**: the inmem (L-BFGS) backend has no
interim state, so `--save-dir` there is rejected and resume is a no-op.

## Results

_TBD — awaiting the All-of-Us run. Record: N, per-drug positive rate, the
`vi_convergence` block (final ELBO, n_iter, `|w_CK|max` — the untrained-head tell),
the per-drug AUC table (VI-PC vs two-stage vs LR), macro-AUC, and the effective
merged config. Then write the interpretation as an insight (Phase C3): with Phase A
passed, a VI-PC-beats-baselines result replicates Hughes on AoU; a converged fit
(`|w_CK|max` ≫ 0) that still nulls is an AoU med-completeness / cross-system-leakage
finding, not a model failure._
