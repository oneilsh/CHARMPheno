---
id: 92
slug: mondo-cv-natgrad-correction
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
readout_sample_frac: 0.3
# --- the deltas vs 0091: natural-gradient correction (engine, automatic) + recalibrated wy ---
weight_y: 10.0
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

# 0092 — Mondo cardiovascular: natural-gradient supervised correction (the fix)

The make-or-break run for the PC thesis at scale. insight 0072 root-caused the
"PC is neutral" result as a SCALING BUG: the supervised λ-correction was a raw-gradient
step (`∝ 1/λ²`) that vanished at whole-population λ (~1e6). ADR 0044 replaces it with
the natural gradient (`∂L/∂E[logβ] = grad_eb·eb`), scale-stable. `topic_trust` is
retired. `weight_y` re-calibrated to 10 (was 50 compensating for the dead correction).

## What to read (use `make -C analysis/cloud report ID=92`)

1. **`corr_relΔλ`** (fit-health trajectory) — MUST now be >> 0 (0090/0091 were ~0). This
   is the direct signal that supervision is finally moving the topics. If it's still ~0,
   the fix didn't land; if it's huge / λ floors (watch `Σλ_k` min, ELBO trend), `weight_y`
   is too strong — lower it.
2. **HEADLINE `gated_pc vs unsup_gated`** — the payoff. 0090/0091 were Δ−0.0000
   (identical). A POSITIVE readout delta = PC shaping recovered = the neutral-PC
   conclusion was a bug, not a data truth. This is the result we're after.
3. **ELBO trend** — should still rise; a fall means the (now live) correction is
   destabilizing λ → lower `weight_y`.
4. **co-fit head + ladder** — secondary; the head's own 0.52 needs the intercept +
   convergence (separate axis, next). But better-shaped topics may lift it.

## Sequence after this

- If shaping is positive: sweep `weight_y` for the best shaping vs stability, then add
  the head **intercept** (engine change) for the co-fit head's own prediction + a
  sharper shaping direction. Re-confirm on the 41-anchor 0089 setup (the neutral-PC
  finding was consistent across runs — overturning it deserves a second scale).
- If shaping is still flat despite `corr_relΔλ >> 0`: the topics move but not usefully —
  investigate the shaping DIRECTION (the head's `∂L/∂eb`), which points back to the
  intercept/convergence head fixes.

## Run

```bash
cd ~/repos/CHARMPheno && git pull origin claude/spectral-anchor-topic-k-200nqp && \
  CHARM_SPARK_CONF='spark.locality.wait=0s' make -C analysis/cloud exp ID=92
make -C analysis/cloud report ID=92
```

## Run log

_(pending first run)_
