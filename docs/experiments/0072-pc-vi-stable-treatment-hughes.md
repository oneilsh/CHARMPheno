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
max_iter: 500               # SVI global iterations. Tuned up from 200 after the
                            # 0071 trace showed |w_CK| still climbing (linearly,
                            # not saturating) at iter 500 with converged=False. The
                            # fully-observed all-ones mask gives every head ~10x the
                            # per-iter gradient of 0071's one-cell-per-patient mask,
                            # so 500 should get further here; save_interval below
                            # makes it kill/resume/eval-able if it plateaus early.
test_frac: 0.25
seed: 0
save_interval: 25           # checkpoint the VIResult every 25 SVI iters into the run
                            # dir (-> --save-interval), so a long fit is resumable and
                            # peekable via --eval-only. ~20 checkpoints over 500 iters.
# --- distributed-SVI schedule (backend: vi only) ---
subsampling_rate: 0.05      # mini-batch fraction per SVI iteration (-> --subsampling-rate)
tau0: 32.0                  # Robbins-Monro learning offset (-> --tau0). Lowered from
                            # 64 after 0071: at tau0=64 the tail rho flattened near
                            # 0.022 and the head crawled up ~linearly without settling.
                            # tau0=32 gives a larger sustained step so the head can
                            # actually converge (raise back toward 64 if the ELBO/head
                            # destabilizes; drop toward 16 if it is still under-moved).
kappa: 0.6                  # Robbins-Monro learning decay in (0.5, 1.0] (-> --kappa)
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

## Baselines (same masking, collected to memory)

Computed with the SAME code as the antidepressant runs
(`analysis.pc.evaluate.multitask_baseline_probas`), so the VI-PC number is
comparable to the same baselines: two-stage (unsupervised topics `weight_y=0` →
per-drug logistic regression on the frozen representation) and
logistic-regression-on-codes. Per-drug heldout ROC AUC + AP, macro-averaged over
non-degenerate drugs.

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
- **SVI schedule (`subsampling_rate`, `tau0`, `kappa`)** — the head's ability to
  leave zero depends on the Robbins-Monro step. **Starts at `tau0=32`** (lowered
  from 0071's 64: there the tail ρ flattened near 0.022 and `|w_CK|` crawled up
  ~linearly to ~2.3 without settling, `converged=False`). `max_iter=500` +
  `save_interval=25` mean you can watch `vi_convergence.w_CK_absmax` across
  checkpoints and stop when it plateaus. If it is still `≈0` or under-moved after a
  run, drop `tau0` toward 16 and/or raise `weight_y`; if the ELBO/head destabilizes
  (Σλ blows up, |w_CK| runs away), raise `tau0` back toward 64.
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

## Results

_TBD — awaiting the All-of-Us run. Record: N, per-drug positive rate (the
`stable_treatment_label` positives), the `vi_convergence` block (final ELBO,
n_iter, `|w_CK|max` — the untrained-head tell), the per-drug AUC table (VI-PC vs
two-stage vs LR), macro-AUC, and the effective merged config. Then write the
interpretation as an insight: a PC-slightly-beats-LR result at ~0.60–0.65 macro
replicates Hughes' stable-treatment finding on AoU; a converged fit
(`|w_CK|max` ≫ 0) that nulls, or one that overshoots ~0.65, is an
AoU-completeness / label-leakage finding to chase down, not a headline._
