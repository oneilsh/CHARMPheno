---
id: 72
slug: pc-vi-stable-treatment-hughes
status: pending
model_class: pc
cohort: mdd_stable_treatment
backend: vi                  # distributed VI-native PCEstimator (SVI), not in-mem L-BFGS
# Real knobs map 1:1 to pc_antidepressant_cloud.py flags via build_pc_args
# (key -> --key). Cohort defaults live in
# experiments/defaults/mdd_stable_treatment.yaml; only per-experiment overrides
# need appear here. build_pc_args threads --cohort mdd_stable_treatment + the
# stable knobs (min_days/max_gap_days/min_history_events/age_min/age_max) and,
# because backend==vi, the SVI schedule (subsampling_rate/tau0/kappa).
# --- feature (fused-vocab BOW, ALL-HISTORY) config ---
# NOTE: no lookback_days — this cohort uses the patient's ENTIRE pre-index
# history (all events before the stable-interval start).
vocab_size: 5000
min_df: 20
min_patient_count: 20
person_mod: 1
# --- cohort (stable-treatment knobs) config ---
min_days: 90                # minimum stable-interval length (days)
max_gap_days: 395           # max visit gap (an encounter at least every ~13 months)
min_history_events: 2       # min events strictly before the first antidepressant era
age_min: 18                 # inclusive age gate at the stable-interval start
age_max: 80
# --- model / eval config ---
K: 25                       # topics (SWEEP)
weight_y: 100.0             # PC prediction-constraint weight (SWEEP)
alpha: 1.1                  # theta Dirichlet concentration (-> PCEstimator docConcentration)
tau: 1.1                    # baseline (in-mem two-stage) topic Dirichlet; VI eta = 1/K
max_iter: 200               # SVI global iterations. Kept SHORT (~1.9h at ~34s/iter)
                            # and paired with a HOT HEAD (head_lr_scale below) instead
                            # of the slow 500-iter crawl: 0071 showed the topics/lambda
                            # already stable while only |w_CK| under-moved, so the fix
                            # is to accelerate the head, not run everything longer.
test_frac: 0.25
seed: 0
save_interval: 25           # checkpoint the VIResult every 25 SVI iters into the run
                            # dir (-> --save-interval), so the fit is resumable and
                            # peekable via --eval-only (~8 checkpoints over 200 iters).
# --- distributed-SVI schedule (backend: vi only) ---
subsampling_rate: 0.05      # mini-batch fraction per SVI iteration (-> --subsampling-rate)
tau0: 32.0                  # Robbins-Monro learning offset (-> --tau0). The GLOBAL
                            # (topic + lambda) schedule is left moderate; the head is
                            # sped up on its own via head_lr_scale, so tau0 need not be
                            # pushed to the unstable extreme.
kappa: 0.6                  # Robbins-Monro learning decay in (0.5, 1.0] (-> --kappa)
head_lr_scale: 3.0          # HEAD-ONLY step multiplier (-> --head-lr-scale). Scales the
                            # logistic-head SGD ~3x so the head reaches (and exceeds) its
                            # 500-iter scale-1 movement within 200 iters, WITHOUT touching
                            # the topic/lambda step. Lower toward 1.5 if |w_CK| runs away
                            # or the ELBO destabilizes; raise toward 5 if still under-moved.
weight_y_warmup_iters: 20   # ramp weight_y 0->100 over the first 20 SVI steps
                            # (-> --weight-y-warmup-iters), so the 3x-hot head does not
                            # spike on the early, high-variance minibatches.
# --- baseline controls -------------------------------------------------------
baseline_max_iter: 100      # cap the two-stage baseline's SECOND (distributed SVI,
                            # weight_y=0) topic fit at 100 iters (-> --baseline-max-iter).
                            # Unsupervised topics converge faster than the head, so 100
                            # keeps the extra fit ~1h rather than a full 200. Set
                            # skip_two_stage: true (or --eval-only --skip-two-stage) to
                            # drop it entirely for a fast PC-vs-LR-on-codes readout.
# skip_two_stage: false     # true => report only PC + LR-on-codes (no two-stage fit)
# warm_start_unsup_iters: 50 # unsup warm-start (Hughes): 0=cold start; N>0 = phase-1
                            # weight_y=0 topic warm-up then fresh-RM supervised phase 2
                            # (-> --warm-start-unsup-iters). See "Warm-start" below.
# Documentary-only (NOT driver flags; ignored by build_pc_args): the vocab is
# fused across condition+drug+procedure over all pre-index history, and the model
# is joint multi-task PC with a FULLY-OBSERVED length-10 label (all-ones mask).
feature_domains: [condition, drug, procedure]
feature_window: all_history
pc_multitask: true
label_shape: fully_observed_10
run_via: make exp ID=72
---

<!-- NUMBERING NOTE: authored on branch claude/faithful-flat-pc, as the
stable-treatment sibling of the antidepressant PC runs (0070 inmem, 0071 VI).
`main` was below 0072 at authoring time, but sibling branches have claimed 0071+
under different slugs. If one merges first, renumber this file (git mv + retitle)
to the next free experiment id — the slug (pc-vi-stable-treatment-...)
disambiguates. The number is cosmetic. -->

# Experiment 0072 — VI-native PC Hughes stable-treatment replication (distributed SVI)

## Status: PENDING — pre-registration. Results filled in AFTER the AoU run.

Per our convention, an experiment doc is written when it is designed; the
**Results** section stays empty until the driver has actually run on All-of-Us.
Everything above the Results heading is the pre-registered design.

## Goal

Reproduce the Hughes et al. (AISTATS 2018) **stable-treatment antidepressant**
result: from a patient's entire pre-treatment code history, can a
Prediction-Constrained topic model recover WHICH antidepressant(s) that patient
is on stable, sustained treatment with — as a **fully-observed** length-10
multi-label over the 10 Hughes-aligned antidepressants — better than logistic
regression on the raw codes?

This is the FAITHFUL Hughes cohort (supplement B.4), distinct from the
per-index-drug ">=90-day the drug worked" framing of 0070 / 0071:

- **Cohort** (`mdd_stable_treatment`, `apply_mdd_stable_treatment_cohort`):
  age 18–80 at the stable-interval start; ≥1 major-depression dx (SNOMED 440383
  and descendants, excluding bipolar 439254 / schizoaffective 4224940); ≥2 events
  before the first antidepressant era; and a qualifying STABLE INTERVAL — a
  maximal interval whose active antidepressant SUBSET is constant, lasting ≥90
  days, with encounters at least every ~13 months (max visit gap ≤395 days,
  bounding both endpoints). The FIRST such interval defines the label; its
  `stable_start` is the index / feature anchor. An add-on or switch splits the
  interval.
- **Label** (`stable_treatment_label`, fully observed): a length-10 indicator of
  which of the 10 Hughes drugs (fluoxetine, sertraline, paroxetine, citalopram,
  escitalopram, venlafaxine, duloxetine, amitriptyline, nortriptyline, bupropion)
  are in that interval's stable subset — usually one positive, occasionally a held
  combination (two). The mask is **all-ones**: every drug head trains on every
  patient (a 0 means "not part of the stable regimen", a true negative, not an
  unobserved cell), so `evaluate_pc_multitask` treats this as standard 10-way
  multi-label.
- **Features** (`all_history_feature_events` -> fused BOW): the patient's ENTIRE
  history of condition+drug+procedure codes BEFORE `stable_start` (unbounded
  lookback — no `lookback_days`), vectorized to a per-patient count BOW over a
  fused vocab (V≈5000).

## Model — joint multi-task VI-PC (distributed), fully-observed

ONE shared `PCEstimator` with `numLabels = 10` per-drug heads, `weightY > 0`. The
per-patient label + all-ones mask are attached to the BOW DataFrame as
`ArrayType` columns (`attach_fullyobserved_label_columns`, over the FIXED
`_HUGHES_ANTIDEPRESSANTS` column order — column c is the same Hughes drug every
run, NOT a `stable_drug_order` over present drugs), the corpus is split by person
at the DataFrame level (`person_hash_split`), and the estimator fits distributed
SVI (no collect). The head's per-label `sigmoid(w_CK·θ)` (`probabilityCol`) is
scored per-drug on the heldout split. On `--backend inmem` the same labels are
assembled in memory (`assemble_fullyobserved_labels`) and the in-memory L-BFGS PC
runs via `evaluate_pc_multitask` — byte-for-byte the same eval, mask all-ones.

## Baselines (SVI-consistent two-stage + LR-on-codes)

Two comparators, both scored per-drug (heldout ROC AUC + AP, macro-averaged over
non-degenerate drugs):

- **Two-stage (unsupervised topics → per-drug LR)** — now a **second distributed
  SVI fit**: a `PCEstimator` at `weight_y=0` (the SAME machinery as the model and
  the warm-start phase 1, `driver::_vi_two_stage_bundle`), whose K-dim
  `topicDistribution` (θ) is collected (NOT the dense D×V matrix) and fed to a
  per-drug masked logistic regression. This replaces the old in-memory
  `PCTopicModel(weight_y=0)` fit, which collected the dense matrix to the driver
  and **OOM-killed the post-fit step** (SIGTERM 143) at full-cohort scale — and it
  makes the baseline's π-estimator (SVI-CAVI θ) match the model's rather than the
  reference's NEF-MAP π. It is a *second* SVI fit, so `baseline_max_iter` caps it
  (default 100 here) and `--skip-two-stage` / `skip_two_stage: true` drops it
  entirely for a fast readout.
- **LR-on-codes** — per-drug masked logistic regression straight on the raw fused
  counts. Inherently code-space, so it still collects the dense D×V matrix to the
  driver (the irreducible cost); this alone fits the 8g driver once the in-memory
  two-stage fit is gone.

## Target

**Per-drug AUC ≈ 0.60–0.65, with PC slightly ABOVE LR-on-codes** — the Hughes
AISTATS-2018 antidepressant stable-treatment result (Fig. 3; avg AUC across meds,
PC-sLDA "beating LR slightly" and improving on its Gibbs-LDA init). This is the
FAITHFUL target: 0072 runs the paper's own stable-treatment / fully-observed
multi-label task, whereas the sibling 0070/0071 use a different (per-index-drug,
"did the drug work") label the paper never ran — so ~0.60–0.65 is the directly
comparable number here and only a borrowed expectation there. The
stable-treatment label is close to "which drug is this patient on", a modest but
real signal from history, where PC's shared representation buys a small edge over
raw-code LR rather than a large one. A macro-AUC well above ~0.65, or PC ≫ LR,
would be a red flag to check for label leakage (e.g. the anchoring drug era
bleeding into the "history" features) before believing it. See
`docs/hughes-comparison.md` for the full setup crosswalk (and how this relates to
the group's JAMA Network Open 2020 clinical follow-up — same method, larger
two-site cohort, whose ≥90-day stability definition this cohort actually mirrors).

## Knobs to sweep before trusting a number

- **`K` and `weight_y`** — placeholders. Sweep and read macro-AUC. The
  fully-observed all-ones mask means every head sees every patient, so `weight_y`
  trades topic quality against 10-way predictive fit differently than the sparse
  index-drug mask of 0070/0071 — re-tune, don't inherit.
- **Head learning rate (`head_lr_scale`, `weight_y_warmup_iters`)** — the primary
  convergence lever here. 0071 showed the topics/Σλ already stable while only
  `|w_CK|` under-moved (crawling ~linearly to ~2.3 at iter 500, `converged=False`),
  so 0072 keeps `max_iter` short (200) and instead runs a **hot head**:
  `head_lr_scale=3` scales ONLY the logistic-head SGD step (topic/λ schedule
  untouched), with `weight_y_warmup_iters=20` softening the first, high-variance
  steps. Watch `vi_convergence.w_CK_absmax` across the `save_interval=25`
  checkpoints: if it plateaus by ~150 iters you're converged; if it's still
  climbing, resume for more iters or raise `head_lr_scale` toward 5; if `|w_CK|`
  runs away or the ELBO destabilizes (Σλ blows up), lower it toward 1.5.
- **SVI schedule (`subsampling_rate`, `tau0`, `kappa`)** — the GLOBAL (topic + λ)
  Robbins-Monro step, left moderate at `tau0=32`, `kappa=0.6`. Prefer moving the
  head via `head_lr_scale` above; only reach for a lower `tau0` (≈16) if you want
  the *topics* to move faster too (and accept the higher instability risk).
- **Unsupervised warm-start (`warm_start_unsup_iters`)** — Hughes seed the
  supervised fit from unsupervised topics. `0` = cold start; `N > 0` runs a
  `weight_y=0` SVI phase-1 topic warm-up, then warm-starts the supervised phase-2
  with a fresh Robbins-Monro schedule (distinct from `--resume-from`). For the
  faithful replication, try `warm_start_unsup_iters: 50` (uncomment in the
  frontmatter) and A/B against `0`.
- **Cohort knobs (`min_days`, `max_gap_days`, `min_history_events`, `age_min`,
  `age_max`)** — the stable-interval definition. `max_gap_days=395` encodes
  "an encounter at least every ~13 months". Tightening `min_days` or `max_gap_days`
  shrinks the cohort toward more-certain stable treatment.
- **Verify concept ids** — the `_DRUG_REGISTRY` (10 Hughes drugs) / MDD
  `_DISEASE_REGISTRY` entries carry `VERIFY ON FIRST RUN` comments; check
  `concept_ancestor` counts against the live CDR before trusting cohort N.

## How to run (AoU Dataproc master)

Tracked / reproducible (params from this frontmatter +
`experiments/defaults/mdd_stable_treatment.yaml`):
```
cd analysis/cloud && make exp ID=72
```
Free-form sweeps via the standalone target (note `--cohort mdd_stable_treatment`
+ the stable knobs + `--backend vi`):
```
cd analysis/cloud && make pc-antidepressant \
  PC_AD_ARGS='--cohort mdd_stable_treatment --backend vi --K 25 --weight-y 100 \
              --min-days 90 --max-gap-days 395 --min-history-events 2 \
              --age-min 18 --age-max 80 --vocab-size 5000 \
              --subsampling-rate 0.05 --tau0 64 --kappa 0.6 \
              --out runs/exp0072_results.json'
```
Both require `WORKSPACE_CDR` / `GOOGLE_CLOUD_PROJECT` (via `make setup`). The VI
fit runs distributed on executors; only the two baselines collect the dense `D×V`
matrix to the driver, so the 8g driver (`_driver_memory_for`, overridable via
`CHARM_DRIVER_MEMORY`) applies. `make exp ID=72` fits, writes `pc_results.json` +
`summary.md` under the run dir, and skips the NPMI eval (PC has its own metrics).

Resume + eval-from-checkpoint are VI-only and identical to 0071
(`run_experiment.py` detects `manifest.json` and threads `--resume-from`;
`check_resume_compat` refuses a resume if the corpus config — person_mod / the
stable knobs / min_df / min_patient_count — changed between runs). The augmented
re-save records `cohort='mdd_stable_treatment'` + the stable knobs in the
checkpoint's `corpus_manifest`.

### Fast readout off an existing checkpoint (skip the two-stage)

The fit checkpoints before scoring, so a completed-fit run whose post-fit step
died (or any saved run) can be scored without re-fitting — and `--skip-two-stage`
drops the second SVI baseline for the quickest possible read (rebuilds the BOW +
loads the head + PC transform + LR-on-codes only):
```
cd analysis/cloud && make pc-antidepressant \
  PC_AD_ARGS='--cohort mdd_stable_treatment --backend vi --eval-only \
              --skip-two-stage --save-dir <run_dir> \
              --vocab-size 5000 --min-df 20 --min-patient-count 20 \
              --age-min 18 --age-max 80 --min-days 90 --max-gap-days 395 \
              --min-history-events 2 --min-label-count 20 \
              --out <run_dir>/pc_results.json'
```
Add `CHARM_DRIVER_MEMORY=12g` if the dense LR-on-codes collect is tight on the
8g default. Drop `--skip-two-stage` to also get the distributed two-stage number
(capped by `--baseline-max-iter`).

## Results

_TBD — awaiting the All-of-Us run. Record: N, per-drug positive rate (the
`stable_treatment_label` positives), the `vi_convergence` block (final ELBO,
n_iter, `|w_CK|max` — the untrained-head tell), the per-drug AUC table (VI-PC vs
two-stage vs LR), macro-AUC, and the effective merged config. Then write the
interpretation as an insight: a PC-slightly-beats-LR result at ~0.60–0.65 macro
replicates Hughes' stable-treatment finding on AoU; a converged fit
(`|w_CK|max` ≫ 0) that nulls, or one that overshoots ~0.65, is an
AoU-completeness / label-leakage finding to chase down, not a headline._
