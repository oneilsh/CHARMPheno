---
id: 93
slug: mondo-cv-head-starvation-probe
status: pending
model_class: gated_pc
cohort: population_mondo_cardiovascular
cohort_def: population_mondo_cardiovascular
disease: rare_priority
# FAST head-starvation probe (minutes, not hours). Same Mondo-cardiovascular corpus as
# 0092, but: max_iter=8, skip the unsup twin, and --diag-only — fit the gated_pc arm
# ONLY, then print the per-node head-magnitude histogram (|w_c| on each node's support
# vs its positive count) and EXIT before the slow θ-collect readouts/baselines/ladder.
# Directly tests the exp 0092 hypothesis: is the whole-Mondo neutral-PC / corr≈0 caused
# by head STARVATION (the low-positive nodes' localized heads never train, |w_c|≈0)?
# The per-iter ||grad_y|| / |w_CK|max trajectory (b67c3c3) also prints during the fit.
dag_source: mondo
mondo_branch: MONDO:0004995
min_positives: 100
mondo_version: 2026-06-02
mondo_cache_dir: data/mondo
extra_domains: measurement,drug
label_mask_mode: closure
localize_head: true
diag_only: true
skip_unsup_gated: true
# --- the ONLY deltas vs 0092: fast probe (few iters, no readout, per-node head dump) ---
max_iter: 8
weight_y: 10.0
head_lr: 1.0
# --- everything else identical to 0092 ---
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
subsampling_rate: 0.1
tau0: 64.0
kappa: 0.51
cavi_max_iter: 100
cavi_tol: 0.001
with_dag_head: false
baseline_max_iter: 100
min_label_count: 20
eval_every: 0
num_partitions: 96
seed: 42
cache_uri: hdfs:///user/dataproc/charm/case_finding_cache
---

# 0093 — Mondo cardiovascular: fast head-starvation probe

The cheap answer to the exp 0092 question, without paying for a full fit + readout.
Local reproduction PROVED the natural-gradient correction shapes topics hard given
signal (Δ+0.42) and could not make corr≈0 at K=20/D=2500 — so the standing hypothesis
for the cluster's neutral-PC is **head starvation**: with ~3,800 nodes and ~100
positives each, the low-positive nodes' localized Fisher is degenerate → the Newton
solve returns `w_c≈0` → those heads never shape their topics → `grad_topics≈0` →
`corr≈0`.

## What to read

The driver prints, right after the fit and before exiting:

```
[head-starvation probe] per-node |w_c| on localized support:
    dead  (≈0)     NNNN nodes   median +ct=..
    ...
    -> D/C heads DEAD (|w_c|<1e-06), T trained
    -> terminal +count: trained median=.. vs DEAD median=..  (starvation ⇔ dead≪trained)
```

- **Most nodes DEAD + DEAD median +count ≪ trained median** → CONFIRMED starvation. The
  fix is on the sparsity axis (node selection ≥N positives, or cross-node hierarchical
  shrinkage so rare nodes borrow strength), NOT head convergence.
- **Few dead, |w_c| broadly nonzero** → NOT starvation; the neutral-PC is something else
  (revisit the correction-application path or the shaping direction).

Also watch the in-fit trajectory: `||grad_y||` and `|w_CK|max` should be ≈0 at every
iter under the starvation hypothesis (the aggregate |w_CK|max is dominated by the few
well-populated nodes, so the histogram is the real evidence).

## Run

```bash
cd ~/repos/CHARMPheno && git pull origin claude/spectral-anchor-topic-k-200nqp && \
  CHARM_SPARK_CONF='spark.locality.wait=0s' make -C analysis/cloud exp ID=93
```

(No `make report` needed — the probe output is a few lines, printed inline. Paste them.)

## Run log

**2026-08-18 — starvation REFUTED, SATURATION found.** 437-node cardiovascular branch
(not 3,800 — that's whole-Mondo). `0/437 heads DEAD`, `|w_c|` median ~150 (max 770) —
every head trained. But the in-fit trajectory showed the real mechanism:

```
|w_CK|max: 110 → 170 → 219 → 254 → 270 → 272 → 273 → 273
||grad_y||: 0 → 1.3e-84 → 6.2e-85 → 3.0e-85 → 2.1e-85 → ...   (≈ e^{-z} underflow)
corr_relΔλ = 0.00e+00 every iter
```

The head trains to `|w|~273` and, with θ razor-peaked (`α=0.0022` from
`optimize_doc_concentration`), the logits `z=w·θ ≈ 200+` saturate the sigmoid so the
shaping gradient `∂loss/∂θ ∝ σ(−z) ≈ e^{−200}` underflows to 0 → `corr=0` bit-exact.
The newton head hits `|w|=110` in the FIRST step (before `weight_y` warmup engages:
`eff_wy=3.2` at iter 8), so the correction never sees a live gradient. `head_l2=0.01` is
ABSOLUTE and negligible vs the ~110k-doc corpus-summed gradient → no cap on `|w|`.
→ confirming test = exp 0094 (head_l2 0.01→5.0, head_lr→0.3).
