---
id: 75
slug: pc-vi-newton-head-damped
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
head_lr: 0.3                  # damped: 0.7 -> 0.3. With damping d the head is an EMA
                             # Newton step/iter already converges the head; damping < 1
                             # smooths minibatch noise in g/H while topics move.
head_newton_ridge: 0.05      # 0.01 -> 0.05: regularize near-singular per-minibatch H_c
                             # the per-label IRLS solve; only stabilizes it (AUC is
                             # scale-invariant to head magnitude), does not bias direction.
# --- baseline controls ---
baseline_max_iter: 100
skip_two_stage: true
min_label_count: 20
---

# 0075 — VI-PC Newton (IRLS) head, DAMPED (stabilize the oscillation)

Identical to 0073 except `head_optimizer: sgd/adam → newton`. The direct test of "converge
the head aggressively within each SVI iteration," done the aggregatable way: one ridge-
Newton step per iteration on the per-label Fisher information, which needs no raw per-doc
θ on the driver and is scale-invariant under the runner's corpus/batch stat scaling.

0074 showed newton WORKS (head AUC 0.52->0.60, cos 0.09->0.35) but OSCILLATES: |w_CK|
bounced 8.8->13->26.7->8, because per-minibatch Newton chases each minibatch's own
logistic optimum and head_lr=0.7 tracked it too closely (+ a near-singular minibatch H_c
spiked to 26.7). 0075 damps harder (head_lr 0.3: the head becomes an EMA of the per-
minibatch optima -> converges to their mean = the corpus optimum, ~3-4x less oscillation)
and ridges more (0.05: regularizes the near-singular H_c, no spikes). Expect the head to
climb from 0.60 toward the pc_topics_lr ceiling (~0.62) and |w_CK| to stay bounded/steady.

**Two questions this answers:**
1. Does a *converged* co-fit head reach the `pc_topics_lr` ceiling (~0.61)? (Head AUC +
   `head vs raw-theta LR direction` cosine → ~1.)
2. With a valid head signal driving the supervised topic correction (for the first time),
   do the topics become more predictive than unsupervised — i.e. does PC *actually* help
   on AoU? (`pc_topics_lr` vs 0.6115 / LR-on-codes 0.6127.)

## Run log

### Run 1 (2026-08-11) — damping stabilized |w_CK|, but the head plateaus at ~0.60

| run | head AUC | cos(head, LR-dir) | pc_topics_lr | lr_codes | \|w_CK\| |
|---|---|---|---|---|---|
| newton hot (0074) | 0.599 | +0.347 | 0.6191 | 0.6127 | oscillating (spike 26.7) |
| **newton damped (this)** | 0.606 | +0.331 | 0.6192 | 0.6127 | **steady 6–8** ✓ |

- **Damping fixed stability:** `|w_CK|max` steady 6.3–7.8, no spikes (head_lr 0.3 + ridge 0.05).
- **But the head plateaus (~0.60, cos 0.33), NOT the 0.62 ceiling** — and NOT because of
  oscillation. The topics never converge in 100 supervised iters (`converged=False`; α + Σλ
  still drifting at iter 100), so the head chases a MOVING representation the whole run and
  never gets a stable target. `cos(head, LR-on-final-topics)=0.35` = the head calibrated to
  the topic TRAJECTORY, not the final topics.
- **PC benefit on AoU = marginal but consistent:** `pc_topics_lr` 0.619 across BOTH newton
  runs, vs unsupervised two-stage 0.614 (0072) and lr_codes 0.613. With a valid-ish head
  signal finally driving the correction, supervision edges unsupervised by ~+0.005 — a real
  but tiny gain, as expected where the signal is already in the topics.

**Closure (AoU PC arc, 0070–0075):** the digamma-Jacobian gradient bug (fixed), the head
non-convergence (diagnosed via the eval-only localizer, fixed by newton/IRLS), and the fair
PC test (marginal on AoU) are all resolved. Remaining head gap 0.60→0.62 is a convergence-
budget/topic-drift polish (more supervised iters and/or Polyak-average the head) that does
NOT change the AoU conclusion. Newton machinery + finding carry to the Mondo rare-disease
cohorts (same AoU data, Mondo-mapping identification), the hidden-low-mass-signal regime
where topic-shaping should give a real pc_topics_lr gain.
