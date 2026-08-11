---
id: 74
slug: pc-vi-newton-head-fast
status: pending
model_class: pc
cohort: mdd_stable_treatment
backend: vi
# --- feature (fused-vocab BOW, ALL-HISTORY) config (same as 0072/0073) ---
vocab_size: 5000
min_df: 20
min_patient_count: 20
person_mod: 1
# --- cohort (stable-treatment knobs, same as 0072/0073) ---
min_days: 90
max_gap_days: 395
min_history_events: 2
age_min: 18
age_max: 80
# --- model / eval config (identical schedule to 0073 for a fair comparison) ---
K: 50
weight_y: 1000.0
alpha: 1.1
tau: 1.1
subsampling_rate: 0.1
tau0: 32.0
kappa: 0.6
grad_cavi_iters: 50
warm_start_unsup_iters: 25
max_iter: 100
topic_trust: 0.1
weight_y_warmup_iters: 0
# --- THE 0074 change vs 0073: newton head (converges the head each iteration) ----
# 0073 (adam, hot lr) confirmed the head never converges: one noisy gradient step per
# SVI iteration -> w_CK oscillates, stays ~orthogonal to the batch-LR direction
# (cos +0.11), head AUC 0.53 while pc_topics_lr=0.61. 'newton' takes ONE aggregatable
# ridge-Newton (IRLS) step per iteration instead: g_c = Σ_d (p-y)π and
# H_c = Σ_d p(1-p)ππ' are additive corpus-scaled doc-sums, so H⁻¹g is scale-invariant
# and converges the logistic head on the current θ (Newton converges logistic in a few
# steps). This ALSO feeds the topic correction a VALID head signal each iter (the
# correction's ∂loss_y/∂θ flows through w_CK) — so it is the FIRST fair test of whether
# PC's supervised topic-shaping helps on AoU (every prior run drove the correction with
# a mis-directed head). Read at the end:
#   * head AUC and `head vs raw-theta LR direction` cosine: does the head now converge
#     (cos -> ~1, AUC -> pc_topics_lr ~0.61)?
#   * pc_topics_lr vs 0073's 0.6115 and LR-on-codes 0.6127: with a valid head signal,
#     do the supervised topics finally beat unsupervised? (If yes, PC helps on AoU after
#     all; if not, banked closure holds and the machinery carries to the Mondo rare-
#     disease cohorts — same AoU data, Mondo-mapping-based identification.)
head_optimizer: newton
head_lr: 0.7                 # newton step-damping fraction (~0.5-1.0). One damped
                             # Newton step/iter already converges the head; damping < 1
                             # smooths minibatch noise in g/H while topics move.
head_newton_ridge: 0.01      # relative ridge (fraction of mean(diag(H))) conditioning
                             # the per-label IRLS solve; only stabilizes it (AUC is
                             # scale-invariant to head magnitude), does not bias direction.
# --- baseline controls ---
baseline_max_iter: 100
skip_two_stage: true
min_label_count: 20
---

# 0074 — VI-PC Newton (IRLS) head, fast schedule

Identical to 0073 except `head_optimizer: sgd/adam → newton`. The direct test of "converge
the head aggressively within each SVI iteration," done the aggregatable way: one ridge-
Newton step per iteration on the per-label Fisher information, which needs no raw per-doc
θ on the driver and is scale-invariant under the runner's corpus/batch stat scaling.

**Two questions this answers:**
1. Does a *converged* co-fit head reach the `pc_topics_lr` ceiling (~0.61)? (Head AUC +
   `head vs raw-theta LR direction` cosine → ~1.)
2. With a valid head signal driving the supervised topic correction (for the first time),
   do the topics become more predictive than unsupervised — i.e. does PC *actually* help
   on AoU? (`pc_topics_lr` vs 0.6115 / LR-on-codes 0.6127.)

## Run log

_(pending)_
