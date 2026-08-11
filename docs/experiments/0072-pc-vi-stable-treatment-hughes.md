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
K: 50                       # topics. Raised from 25 after Run 1: the cold K=25 fit's
                            # topic representation under-informed the head (PC macro
                            # 0.545 << LR-on-codes 0.622); Hughes swept to 100, so more
                            # capacity is a prime lever. (SWEEP: 25/50/100.)
weight_y: 1000.0            # PC prediction-constraint weight. Raised 100 -> 1000 after
                            # the digamma-Jacobian gradient fix: the corrected supervised
                            # topic gradient is ~65x smaller (true λ-space), so weight_y
                            # must be ~an order of magnitude larger to shape topics — and
                            # the fix makes that SAFE (toy sweep monotone 0.57@50 ->
                            # 0.91@1000 with Σλ pinned, no runaway at any weight_y). SWEEP
                            # up/down from 1000; watch heldout PC vs pc_topics_lr for
                            # over-supervision (heldout drop) on noisy real data.
alpha: 1.1                  # theta Dirichlet concentration (-> PCEstimator docConcentration)
tau: 1.1                    # baseline (in-mem two-stage) topic Dirichlet; VI eta = 1/K
max_iter: 300               # SVI global iterations. Raised 200 -> 300 for Run 5 (converge
                            # the joint head): Run 4's head under-converged (VI-PC 0.518 vs
                            # pc_topics_lr 0.612 on the SAME topics) because 200 iters at
                            # subsampling 0.05 was ~10 noisy passes vs a batch LR. With
                            # subsampling now 0.2 (4x data/iter -> ~60 passes at 300) the
                            # head has the passes + lower noise to reach its optimum on the
                            # (now stable) topics. Watch VI-PC head climb toward 0.612.
test_frac: 0.25
seed: 0
save_interval: 25           # checkpoint the VIResult every 25 SVI iters into the run
                            # dir (-> --save-interval), so the fit is resumable and
                            # peekable via --eval-only (~8 checkpoints over 200 iters).
# --- distributed-SVI schedule (backend: vi only) ---
subsampling_rate: 0.2       # mini-batch fraction per SVI iteration (-> --subsampling-rate).
                            # 0.2 not 0.05: RDD.sample is partition-wise, so a 0.05 batch on
                            # this ~34.5k-doc / ~1000-partition corpus is ~1.7 docs/partition
                            # -> tiny tasks + idle executors. 0.2 fills them (~7 docs/part).
tau0: 32.0                  # Robbins-Monro learning offset (-> --tau0). The GLOBAL
                            # (topic + lambda) schedule is left moderate; the head is
                            # sped up on its own via head_lr_scale, so tau0 need not be
                            # pushed to the unstable extreme.
kappa: 0.6                  # Robbins-Monro learning decay in (0.5, 1.0] (-> --kappa)
grad_cavi_iters: 50         # Run 6 — THE head fix (-> --grad-cavi-iters). Run 5 proved
                            # the co-fit head was stuck (0.522, invariant to iters/data/
                            # head_lr) because it trains on a 20-step CAVI theta but is
                            # SCORED on the converged (cavi_max_iter=100/tol=1e-3) theta —
                            # a train/test theta mismatch. On avg-322-token AoU docs CAVI
                            # needs ~50 iters to converge (20-step theta is cos 0.987 off);
                            # 50 aligns training theta with scoring theta (cos 0.99995), so
                            # the head can finally reach the batch-LR optimum (~0.61). This
                            # is exactly why the toy (small docs, CAVI converges by 20)
                            # passed while AoU (large docs) did not. Cost: ~2.5x the per-doc
                            # supervised-gradient tape. Default 20 (fine for small docs).
head_lr_scale: 2.0          # HEAD-ONLY step multiplier (-> --head-lr-scale). Run 5: 1.0
                            # -> 2.0 to help the joint head reach its optimum before ρ
                            # decays (Run 4 settled short at 0.518 vs the batch-LR 0.612 on
                            # the same θ). Safe to bump now: the topics are stable (no drift
                            # to chase) and subsampling 0.2 cut the gradient noise 4x. Only
                            # scales the head step, not the topic/λ update. If |w_CK| starts
                            # oscillating, ease back toward 1.5; if the head still under-
                            # moves, this + more iters is the lever (not weight_y).
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
skip_two_stage: true        # two-stage established at 0.614 (~Hughes) and stable, so
                            # skip its second ~1.4h SVI fit on iteration runs; the
                            # always-on pc_topics_lr diagnostic + LR-on-codes remain.
                            # Set false for a final confirmation run.
topic_trust: 0.1            # per-iter trust-region on the supervised topic correction
                            # (-> --topic-trust). 0.1 caps one step but COMPOUNDS -> the
                            # Run-2 Σλ blow-up; lower to 0.02 (or 0 = freeze topics =
                            # head-only) once localization says the topics are degraded.
warm_start_unsup_iters: 50  # unsup warm-start (Hughes): 0=cold start; N>0 = phase-1
                            # weight_y=0 topic warm-up (50 iters) then fresh-RM supervised
                            # phase 2 (-> --warm-start-unsup-iters). ON for Run 2: Run 1
                            # was COLD and PC under-informed — Hughes initialize from an
                            # unsupervised fit, and cold PC-sLDA is prone to generatively-
                            # good but discriminatively-weak topics. This is THE deferred
                            # warm-vs-cold test (Run 1 is the cold arm).
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
loads the head + PC transform + LR-on-codes only).

**Run dir location.** Artifacts live under `RUNS_DIR`, defined in
`analysis/cloud/Makefile` — on the AoU cluster this resolves to
`/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs`,
so this experiment's checkpoint is `…/runs/0072-pc-vi-stable-treatment-hughes`
(note the `-pc-vi-…` slug — a sibling `0072-multidomain-…` dir from the other
branch shares the number, so use the full slug, not a `0072-*` glob). Copy-paste
as-is (the `RUN_DIR` var means no path editing):
```
RUN_DIR=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0072-pc-vi-stable-treatment-hughes
cd analysis/cloud && CHARM_DRIVER_MEMORY=12g make pc-antidepressant \
  PC_AD_ARGS="--cohort mdd_stable_treatment --backend vi --eval-only \
              --skip-two-stage --save-dir $RUN_DIR \
              --vocab-size 5000 --min-df 20 --min-patient-count 20 \
              --age-min 18 --age-max 80 --min-days 90 --max-gap-days 395 \
              --min-history-events 2 --min-label-count 20 \
              --out $RUN_DIR/pc_results.json"
```
`CHARM_DRIVER_MEMORY=12g` guards the dense LR-on-codes collect on the 8g default.
Drop `--skip-two-stage` to also get the distributed two-stage number (capped by
`--baseline-max-iter`).

## Results

This experiment is a **run log** — each re-fit of ID 72 appends a dated entry
(the run dir is overwritten, but the findings are kept here).

### Run 1 — COLD, K=25, weight_y=100, 200 iters (2026-08-10)

N=45,991 MDD stable-treatment persons (34,515 train / 11,476 test); per-drug
positives 1,503–9,246 (all ≥ 20, nothing masked). Fit: `n_iter=200`,
`converged=False`, `|w_CK|max=4.15` (head **trained**, not the 0.5-AUC untrained
mode — still climbing at 200).

| model | macro AUC | macro AP |
|---|---|---|
| **LR-on-codes** (scaled) | **0.622** | 0.176 |
| **VI-PC** (K=25, cold) | **0.545** | 0.122 |

**Read:** LR-on-codes lands squarely in Hughes' raw-code band (~0.60–0.66) — the
cohort, features and outcome **replicate**. But the PC topic model **underperforms**
both LR-on-codes and Hughes' own PC (~0.627): the K=25 representation under-informs
the head. It is *not* uniformly broken — duloxetine PC 0.636 ≥ LR 0.609, but
bupropion PC 0.498 (chance) — i.e. 25 topics capture some drugs' signal and wash
out others. Diagnosis (ranked): (1) **cold start** — Hughes initialize from an
unsupervised fit, we didn't; (2) **weight_y vs avg tokens** — the driver now
reports avg tokens/doc; if ≫ 100 the supervised term is swamped; (3) **K too
coarse**; (4) not fully converged. Run 2 addresses (1) + (3).

### Run 2 — WARM-start (50) + K=50, weight_y=100, 200 iters (2026-08-10)

Fit `n_iter=200`, `converged=False`, `|w_CK|max=4.81` — but oscillating (rose to
~4, fell to 3.8, rose to 4.8), and `Σλ_max` **blew up 6e5 → 1.58e8** with
`α_max → 1.2` over the supervised phase.

| model | macro AUC | macro AP |
|---|---|---|
| **two-stage (unsup topics → LR)** | **0.614** | 0.149 |
| LR-on-codes | 0.613 | 0.169 |
| **VI-PC (supervised, K=50 warm)** | **0.521** | 0.115 |

**Read — supervision is HURTING.** The distributed two-stage baseline (our own
*unsupervised* SVI topics + a clean LR) hits **0.614 — a Hughes replication** (≈
their 0.627). But the *supervised* PC drops to 0.521, *below* both its own
unsupervised topics and Run 1's cold 0.545. Warm-start + K=50 made PC WORSE, not
better. The `Σλ` blow-up + oscillating head are the tell: the **supervised topic
correction** (a β-probability-space gradient applied to count-space `λ`, capped
per-iter at `topic_trust·λ` but **compounding** over 200 iters — the design's
flagged "biggest open question") is drifting the topics into a degenerate,
*less*-predictive state, so the jointly-trained head chases a moving target while
the two-stage's *frozen* topics score cleanly. The unsupervised half of the
pipeline replicates Hughes; the bug is isolated to the supervised global step.

### Run 3 — LOCALIZE (eval-only on the Run-2 checkpoint, no refit)

The `pc_topics_lr` diagnostic on the buggy Run-2 checkpoint:

| model | macro AUC |
|---|---|
| PC-topics + external LR (`pc_topics_lr`) | **0.604** |
| LR-on-codes | 0.613 |
| VI-PC head | 0.521 |

Both failure modes, head the larger: PC's supervised topics are only *mildly*
degraded (`pc_topics_lr` 0.604 vs unsupervised 0.614, −0.01 — the trust region
kept the corruption small), but the PC **head** extracts only 0.521 from those
topics where a clean LR gets 0.604 (−0.083) — the head co-adapted to the drifting
topics during the `Σλ` runaway.

### ROOT CAUSE — a mis-transformed supervised gradient (fixed)

The supervised topic correction subtracted `∂loss/∂expElogbeta` (topic-PROBABILITY
space) directly from `λ` (Dirichlet-COUNT space), skipping the chain-rule step
through `expElogbeta = exp(ψ(λ) − ψ(Σλ))`. Finite-difference vs the true `∂loss/∂λ`:
the applied gradient was **~65× too large and ~33° mis-directed** (cosine 0.84).
That one defect produced every symptom — the trust region existed only to cap the
65×, a mis-directed push still ratcheted `Σλ` (as STM's damping-cap test predicts),
and the wrong direction degraded the topics. Fix (commit, `_grad_topics_to_lambda`):
the exact digamma-Jacobian completes the chain rule to λ-space, finite-difference-
exact. Toy `weight_y` sweep with the fix: PC AUC **0.57→0.91→0.92** (50→1000→3000),
approaching raw-code LR (0.949) where the unsupervised two-stage is stuck at ~0.56,
with **Σλ pinned at ~7.6e4 (no runaway at any weight_y)**. Unlike STM's structural
pin, this is a plain gradient correction — λ now moves in the true descent direction.

### Run 4 — the corrected fit: weight_y=1000, K=50, warm, fixed gradient (2026-08-10)

`Σλ_max` **flat 5e4–7e5 through all 200 iters** (at `weight_y=1000`, 10× what blew
up in Run 2), `|w_CK|` settled ~4.2 (no oscillation), avg tokens/doc = 322.6.

| model | macro AUC | macro AP |
|---|---|---|
| **PC-topics + external LR** (`pc_topics_lr`) | **0.612** | 0.149 |
| LR-on-codes | 0.613 | 0.169 |
| (unsupervised two-stage, Run 2) | 0.614 | — |
| **VI-PC head** | **0.518** | 0.114 |

**Fix confirmed + Hughes replicated, with one loose end.** The runaway is dead
(`Σλ` flat at 10× the old blow-up weight). The **topics fully recovered**:
`pc_topics_lr` 0.612 ≈ raw codes (0.613) ≈ unsupervised (0.614) ≈ Hughes (~0.62) —
a 50-topic representation captures essentially all the raw-code predictive signal,
the Hughes result on AoU. Two conclusions: (1) **supervision ≈ unsupervised here**
(0.612 vs 0.614) — a *data* finding, not a bug: unlike the toy's hidden low-mass
signal, AoU's predictable signal is already in the unsupervised topics (0.614 ≈
0.613 raw codes), so there is no headroom for supervision to add; Hughes-consistent.
(2) **The headline VI-PC head (0.518) is an under-converged SGD head** — NOT a topic
problem (topics = 0.612). 200 minibatch iters ≈ ~10 noisy passes vs a converged
`lbfgs` LR on 50-dim θ; the head is the last mile. "PC done right" = `pc_topics_lr`
= 0.612. Next: converge the head (0.2 subsampling + more iters, or a final exact
per-drug LR head refit on the trained θ — which by construction lands at 0.612).

### Run 4 config (recorded)

Same cohort as Runs 2/3 but with the gradient fix in code and the config it
implies: **`weight_y` 100 → 1000** (the corrected gradient is ~65× smaller, so the
prediction weight must be ~an order larger to shape topics — now safe at any
strength), **`head_lr_scale` 3 → 1.0** (the "hot head" was compensating for a head
chasing drift; stable topics + `weight_y=1000` drive it enough). `K=50`,
`warm_start_unsup_iters=50`, `weight_y_warmup_iters=20`, `tau0=32`,
`topic_trust=0.1` (now a light floor-guard, not load-bearing), `skip_two_stage:
true`, `max_iter=200`. **Requires `rm -rf` of the run dir** (the Run-2 checkpoint
carries the degenerate topics; a fresh warm-start is needed). Expect: `Σλ` bounded
(no blow-up), late iters no longer crawling, and PC rising off 0.521 toward — and
ideally past — the ~0.614 unsupervised/LR band (supervision finally helping). Read
`PC head` vs `pc_topics_lr`: if they converge, the head closed its gap on stable
topics; sweep `weight_y` from 1000 if PC under- or over-shoots.

### Run 5 — option B (converge the head): FAILED, and told us why (2026-08-10)

subsampling 0.05→0.2, max_iter 200→300, head_lr_scale 1→2. Result: **VI-PC head
0.522** (≈ Run 4's 0.518), `pc_topics_lr` 0.608. **The head did not move** — 4× the
data, 1.5× the iters, 2× the step, same fixed point (`|w_CK|` 4.05). So it is NOT
under-convergence; the head **converged to a fixed point that is not the batch-LR
optimum** on the same θ. For a convex logistic problem that means it is optimizing
on a *different* θ than it is scored on.

### ROOT CAUSE #2 — train/test θ-depth mismatch (the head's real bug)

The head's supervised gradient uses `_cavi_theta_anp`, a **fixed 20-step** CAVI
unroll (`grad_cavi_iters=20`, kept short to bound the autograd tape). Scoring
(`infer_local`) runs CAVI **to convergence** (`cavi_max_iter=100`, `cavi_tol=1e-3`).
On avg-322-token AoU docs CAVI needs **~50 iters** to converge — 20-step θ is cosine
0.987 / L1 0.13 off, 50-step is 0.99995. So the head trains on under-converged θ and
is scored on converged θ: a train/test **representation** mismatch that pins it below
a batch LR (which fits+scores on the *same* converged θ). This is exactly why the
**toy passed** (small docs → CAVI converges within 20 → θ match → head reached 0.907)
while **AoU failed** (large docs → 20-step θ ≠ converged θ). The head machinery
(ridge, optimizer) is identical toy-vs-AoU and the toy converged, so it is not the
head math — it is θ-depth. The `_cavi_theta_anp` docstring's "faithfulness invariant
(train π = test π)" silently breaks for large docs.

### Run 6 — the θ fix — PENDING

Single change from Run 5: **`grad_cavi_iters` 20 → 50** (new `--grad-cavi-iters`),
so the differentiable training θ converges like the scorer's. Everything else held
(K=50, warm 50, weight_y 1000, head_lr_scale 2, subsampling 0.2, max_iter 300,
topic_trust 0.1, skip_two_stage). Cost: ~2.5× the per-doc supervised-gradient tape.
`rm -rf` the run dir first (fresh warm-start). Expect the VI-PC head to finally climb
off 0.52 toward `pc_topics_lr`'s ~0.61 — the methods-faithful joint head reaching the
same optimum the dedicated LR already shows. If it still lags, the residual is the
head ridge/objective, and Option A (`pc_topics_lr` = 0.61 = Hughes) remains in hand.

### Local head-optimizer diagnosis — the θ-depth theory is FALSIFIED (2026-08-11)

Rather than spend another ~3 h cluster run confirming Run 6, the head question was
settled **locally, cluster-free**, by faithfully reproducing the EXACT
`OnlinePCLDA.update_global` head update (per-doc-MEAN logistic gradient + ridge,
Robbins-Monro `rho_t=(tau0+t+1)^-kappa`) on synthetic θ and comparing to a batch
`LogisticRegression` on the same θ. Script + full output:
`analysis/pc/diagnostics/head_optimizer_diagnosis.py`.

Six independent stressors were applied to the exact head update; in **every** one it
reaches the batch-LR ceiling (within SGD noise), never the 0.52 collapse:

| # | stressor | head vs LR ceiling |
|---|---|---|
| 1 | ridge `lambda_w` 0.001→0 (eff. 60×→0× LR's ridge) | 0.623 vs 0.623 — **AUC is scale-invariant**; ridge shrinks ‖w‖, not ranking |
| 2 | iteration budget 100/300/1000, `head_lr_scale` 1/2/5, minibatch=512 | 0.614–0.623 vs 0.622 — converged by ~iter 100; **not** budget-starved |
| 3 | **static θ mismatch**: fit head on `_cavi_theta_anp` 50-unroll, score on converged `_cavi_doc_inference` | routines agree cosine **0.9988**; matched vs mismatch AUC **delta +0.000** |
| 4 | class imbalance (drug prevalence) 0.5→0.05, head has no intercept | 0.57–0.62 vs LR; no-intercept LR == with-intercept (simplex absorbs bias) |
| — | online **topic drift** (β_init→β_final during head SGD, separate sim) | drift penalty **+0.006** vs fixed-topic control — head recovers |

**This falsifies ROOT CAUSE #2 (θ-depth).** Experiment 3 is the direct test: fitting
the head on the short-unroll θ and scoring on the converged θ costs **zero** AUC (the
two CAVI routines are the same fixed point; 50-step is cosine 0.9988 to converged).
So `grad_cavi_iters` 20→50 was never going to move the head — consistent with Run 6
reading 0.522 again. The earlier θ-depth story mistook a real-but-negligible
representation difference for the cause.

**Reframed conclusion.** With `proba_DC = sigmoid(w_CK·θ)` on the *same* converged θ
that `pc_topics_lr` reads (verified in `PCModel._transform`), and the exact head
provably reaching the LR ceiling under every stressor, there is **no reproducible
mechanism** for a 0.52 co-fit head given topics that yield `pc_topics_lr` 0.61. The
head **optimizer is sound**; a persistent 0.52 is therefore a **run-specific
artifact** — a stale/misconfigured fit, or a genuinely near-untrained / mis-directed
`w_CK` in that particular run — not an inherent PC deficiency. **A head-L2 knob would
be a dead end.**

**Cheap next checks (on the saved artifacts, not another optimizer knob):**
1. `grep -o '"grad_cavi_iters": [0-9]*' pc_results.json` — was it the intended run?
2. `w_CK_absmax` in `pc_results.json` / the driver log — `~0` ⇒ head never trained.
3. cosine( co-fit `w_CK[c]`, a fresh `LR.coef_` on the same train θ ) — `~0` ⇒ the
   head trained to the wrong direction (then debug the actual fit, not a synthetic);
   aligned but AUC still 0.52 ⇒ a scoring/eval-wiring bug in that run.

The banked science is unchanged: `pc_topics_lr` ≈ 0.61 ≈ LR-on-codes 0.613 ≈
unsupervised two-stage 0.614 — **Hughes replicated**; supervision ≈ unsupervised on
AoU because the predictable signal is already in the unsupervised topics.

### Run 7 — the direction cosine LOCALIZES it: head TRAINED but MIS-DIRECTED (2026-08-11)

Eval-only re-score of the saved checkpoint (post-fix, K=50, weight_y=1000, n_iter=300,
`|w_CK|max=3.61` — the head is TRAINED, not zero), now emitting the new per-label
`head_vs_lr_cosine` (co-fit `w_CK[c]` vs a fresh raw-θ LR on the same **train** θ):

```
PC (head)  macro AUC = 0.5246
pc_topics_lr         = 0.6150
lr_codes             = 0.6127
head vs raw-θ LR direction: mean cosine = +0.081
  per-label = [-0.03,+0.12,-0.09,+0.16,+0.27,+0.02,+0.20,+0.07,+0.04,+0.05]
```

**This overturns the "scoring artifact" reading.** The head is trained (`|w_CK|`=3.6)
but points **~orthogonal** (mean cos +0.08) to the label direction — and the cosine is
measured on **train** θ, so `w_CK` does not even fit the *training* labels. It never
received a correct label-descent gradient. Yet `pc_topics_lr`=0.61 on the *same* final
θ ⇒ the **topics carry generalizable signal**; only the co-fit head misses it.

This is genuinely new: the faithful *serial* head-SGD reproductions
(`analysis/pc/diagnostics/head_optimizer_diagnosis.py`) ALWAYS reach the LR direction
(cos≈1). So the fault is something the **joint distributed fit** does beyond
head-SGD-on-final-topics — the supervised **topic-correction coupling** or the
**distributed per-doc label/θ plumbing** — neither exercised by those serial sims. The
passing toy test (`test_pc_supervised_beats_two_stage_on_heldout_auc`, head 0.907) uses
the SAME distributed VIRunner path but at C=1, FULL-BATCH, K_FIT<K_DOM — so the plumbing
is not broken in general; the failure is config/scale-dependent (minibatch subsampling,
C=10, K=50, 300 iters of the correction). Bisection in progress via
`spark-vi/tests/manual_pc_head_direction_repro.py` (local SparkContext, real
OnlinePCLDA/VIRunner, toggling full-batch↔minibatch and C=1↔C=10).

**Note on the inspector's `[1]` line:** for an `--eval-only` read, `params.grad_cavi_iters`
etc. reflect the eval command's argparse DEFAULTS, not the checkpoint's training config
(eval-only only restores K + weight_y from metadata). Only `weight_y`, K, `n_iter`, and
`|w_CK|max` in that readout describe the actual fit. (Inspector caveat noted; fix TODO.)
