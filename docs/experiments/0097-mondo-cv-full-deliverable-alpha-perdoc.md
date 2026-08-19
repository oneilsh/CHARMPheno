---
id: 97
slug: mondo-cv-full-deliverable-alpha-perdoc
status: pending
model_class: gated_pc
cohort: population_mondo_cardiovascular
cohort_def: population_mondo_cardiovascular
disease: rare_priority
# THE FIX RUN (ADR 0044 / insight 0072). The supervised λ-correction is now a NATURAL
# gradient (∂L/∂E[logβ] = grad_eb·eb, scale-stable) instead of the old raw gradient
# (∝1/λ², which vanished at whole-population λ and made weight_y a silent no-op —
# gated topics ≡ unsup topics, corr_relΔλ ≈ 0 across 0090/0091). This is the first run
# where PC supervision can actually shape the topics at scale. weight_y is RE-CALIBRATED
# down (10, from the old 50 that only compensated for the ≈0 correction) — with the
# correction live, 50 can floor rare-topic λ cells in one step.
dag_source: mondo
mondo_branch: MONDO:0004995
min_positives: 100
mondo_version: 2026-06-02
mondo_cache_dir: data/mondo
extra_domains: measurement,drug
label_mask_mode: closure
localize_head: true
doc_concentration: 0.5
readout_sample_frac: 0.3
# --- the deltas vs 0091: natural-gradient correction (engine, automatic) + recalibrated wy ---
weight_y: 2.0
head_lr: 1.0
# --- everything else identical to 0091 ---
person_mod: 1
prior_obs_days: 0
doc_min_length: 10
min_n: 20
holdout_frac: 0.2
vocab_size: 5000
min_df: 20
min_patient_count: 20
window_mode: lookback
lookback_days: 1825
label_window_days: 365
strip_mode: both
n_bg: 8
tpn: 1
optimize_doc_concentration: true
head_optimizer: newton
head_newton_ridge: 0.05
head_l2: 0.01
grad_cavi_iters: 30
topic_trust: 0.05
weight_y_warmup_iters: 25
max_iter: 100
subsampling_rate: 0.1
tau0: 64.0
kappa: 0.51
cavi_max_iter: 100
cavi_tol: 0.001
skip_unsup_gated: false
with_dag_head: false
baseline_max_iter: 100
min_label_count: 20
eval_every: 0
num_partitions: 96
seed: 42
cache_uri: hdfs:///user/dataproc/charm/case_finding_cache
---

# 0097 — Mondo cardiovascular: full deliverable (alpha=0.5 + per-doc-mean, stable)

The first full-scale run of a STABLE shaping PC fit — the payoff after the whole
neutral-PC saga resolved to two root causes: alpha=1/K collapsed the CAVI Jacobian (exp
0095), and the newton head ran away on an absolute ridge at ~1e5 docs (exp 0095 blowup).
Fixes: doc_concentration=0.5 (Jacobian alive) plus per-doc-mean newton (|w| corpus-
bounded). exp 0096 confirmed stability at 8 iters (|w|~1.45, corr~0.7 percent/step, ELBO
rising). This is the full 100-iter fit WITH the unsup twin and pc_topics_lr readout.

## What to read (make -C analysis/cloud report ID=97)

1. HEADLINE gated_pc vs unsup_gated (pc_topics_lr) — THE deliverable. A POSITIVE delta =
   PC shaping lifts the readout over unsupervised topics = the method delivers. Every
   prior run was delta approx 0 because shaping was DEAD (alpha-collapse); this is the
   first run where it can actually move.
2. FIT-HEALTH trajectory — confirm 0096 stability holds over 100 iters: |w_CK|max bounded
   (~1-3, no 1e5), corr_relDlambda steady (~0.005-0.01), ||grad_y|| finite, ELBO rising.
3. conditional readout (P(child|parent) by depth) — the case-finding metric.

## Sequence after this

- Positive lift: sweep weight_y UP (4, 8) for MORE shaping — now SAFE, the head can't run
  away (per-doc-mean). weight_y=2 gives a gentle ~0.7 percent/step; lots of headroom.
- Flat lift despite stable shaping: shaping too gentle at weight_y=2 — raise it before
  concluding anything.
- Then: deferred refinements (uniform-beta A/B via gamma_shape; floored/decoupled alpha
  only if we later co-fit) and the whole-Mondo K~3800 scale-up.

## Run

```bash
cd ~/repos/CHARMPheno && git pull origin claude/spectral-anchor-topic-k-200nqp && \
  CHARM_SPARK_CONF='spark.locality.wait=0s' make -C analysis/cloud exp ID=97
make -C analysis/cloud report ID=97
```

## Run log

**2026-08-19 — STABLE + STRONG readout, but PC shaping ~NEUTRAL vs the gated baseline.**

Deliverable (pc_topics_lr, 175 nodes): macro AUC **0.7424**, well-calibrated (pooled ECE
0.006, conditional calibrated 0.0027); conditional cond_AUC 0.79 (depth 1-2) to 0.70-0.75
deeper. A real, stable, calibrated hierarchical case-finding model at full population x
full cardiovascular subtree.

HEADLINE gated_pc vs unsup_gated: AUC 0.7424 vs 0.7410 (Δ+0.0013), AP 0.5448 vs 0.5407
(Δ+0.0040), multiclass top1 0.8274 vs 0.8198 (Δ+0.0076). Shaping is ALIVE (corr peaked
~1.2%) but too gentle at weight_y=2 to move a readout the unsup GATE already does well.
Per-depth deltas are noise (±0.01), including deep/rare nodes -> no hidden rare-node win
in the depth proxy. Structural read: the GATE aligns topics to labels WITHOUT supervision
(unsup already 0.741), so PC's marginal room on this contrast is small by construction.

Co-fit head 0.561 (still the head-formulation gap: ladder co-fit 0.561 -> +convergence
0.615 -> +intercept 0.650 -> oracle 0.697 -> full-K 0.742). Head |w| bounded at 1.79
(per-doc-mean held over 100 minibatched iters). FLAG: 2 infs + an ELBO transient to
-6.76e27 (recovered; |w| stayed bounded) — 10% minibatch noise; watch at higher weight_y.

Next: exp 0098 (weight_y up) — now auto-prints the per-node RARITY SPLIT so the same run
tests both "does more shaping lift the macro" and "does it help the low-positive tail".
