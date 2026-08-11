---
id: 73
slug: pc-vi-adam-hothead-fast
status: pending
model_class: pc
cohort: mdd_stable_treatment
backend: vi
# --- feature (fused-vocab BOW, ALL-HISTORY) config (same as 0072) ---
vocab_size: 5000
min_df: 20
min_patient_count: 20
person_mod: 1
# --- cohort (stable-treatment knobs, same as 0072) ---
min_days: 90
max_gap_days: 395
min_history_events: 2
age_min: 18
age_max: 80
# --- model / eval config ---
K: 50
weight_y: 1000.0
alpha: 1.1
tau: 1.1
subsampling_rate: 0.1        # scalable minibatch (~3.45k docs), unchanged from 0072
tau0: 32.0
kappa: 0.6
grad_cavi_iters: 50
# --- the 0073 changes vs 0072: a HOT head at FEWER outer iters -------------
# Fair-comparison baseline for the "converge the head more aggressively" test:
# CURRENT code (single head step / SVI iter, Adam) but with an aggressive head_lr
# and a short schedule, so we can tell whether just pushing the head harder at
# fewer iters moves it off 0.52 — BEFORE spending on a true per-iteration head
# inner-loop (which needs a runner-level change: the runner scales every stat by
# corpus/batch, and the corpus is over-partitioned, so a driver-side inner-loop or
# FedAvg both need infra work). If this hot-lr baseline already lifts the head, the
# inner-loop may be unnecessary; if not, it isolates the inner-loop's effect.
warm_start_unsup_iters: 25   # 50 -> 25 (unsupervised warm-up; topics converge fast)
max_iter: 100                # 300 -> 100 supervised SVI iters (post-warmup)
head_optimizer: adam
head_lr: 0.3                 # 0.05 -> 0.3 (aggressive; Adam self-scales vs weight_y,
                             # so this is the head's only step dial). Bump to 0.5 if
                             # |w_CK| is still climbing at max_iter.
topic_trust: 0.1
weight_y_warmup_iters: 0     # inert under adam anyway
# --- baseline controls ---
baseline_max_iter: 100
skip_two_stage: true         # pc_topics_lr + LR-on-codes are the comparators we need
min_label_count: 20
---

# 0073 — VI-PC Adam hot-head, fast schedule (fair baseline for the inner-loop test)

Clone of 0072 with a HOTTER head (`head_lr` 0.05 → 0.3) on a SHORTER schedule
(`warm_start_unsup_iters` 50 → 25, `max_iter` 300 → 100). Everything else identical.

**Purpose.** The eval-only localizer (0072 Run 7–8) showed the co-fit head is TRAINED
but MIS-DIRECTED (cos +0.09 to the batch-LR direction on the same topics), and that this
is a head **non-convergence** in online co-adaptive SVI — the head takes one gradient
step per iteration against a θ from continuously-moving topics and never catches the
final geometry. `pc_topics_lr` (= a converged LR on the final topics) is 0.6185, the
ceiling a converged head would reach. Neither θ-depth, Adam, nor sgd moved the head,
because all take **one head step per SVI iteration**.

This run is the **fair-comparison baseline** for the "converge the head more aggressively
within each SVI iteration" idea: it pushes the *current* single-step head as hard as
config allows (hot `head_lr`, short schedule) so a subsequent true head inner-loop can
be attributed to the inner-loop, not to the param change.

**Also note (important):** the topic correction's gradient flows through `w_CK`, so a
mis-directed head means the supervised topic-shaping has *never* had a valid signal —
i.e. "supervision doesn't help topics on AoU" (pc_topics_lr ≈ two-stage ≈ codes ≈ 0.61)
has not yet been tested with a converged head. That is the real prize the inner-loop
unlocks, and the reason to carry it into the Mondo rare-disease work.

## Run log

_(pending)_
