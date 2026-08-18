---
id: 94
slug: mondo-cv-head-saturation-ridge-probe
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
head_lr: 0.3
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
head_l2: 5.0
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

# 0094 — Mondo cardiovascular: head-SATURATION ridge probe (the confirming test)

exp 0093 REFUTED starvation: 0/437 heads dead, `|w_c|` median ~150. But the in-fit
trajectory revealed the true cause — head **SATURATION**:

```
0093:  |w_CK|max: 110 → 170 → 219 → 254 → 270 → 273    (trained HARD)
       ||grad_y||: 0 → 1.3e-84 → 6e-85 → ... ≈ e^{-z} underflow
       corr_relΔλ = 0.00e+00 every iter
```

The shaping gradient is `∂loss/∂θ ∝ (p−y) = σ(−z)`, `z = w·θ`. With `|w|~273` AND a
razor-peaked θ (`α=0.0022`, driven there by `optimize_doc_concentration`), `z≈200+` so
`σ(−z)≈e^{−200}≈1e-87` — the head is so confident the residual (and thus the topic-
shaping gradient) underflows to zero. This is the separable-data / logistic-MLE-at-
infinity failure the `head_l2` ridge was built to guard (`pc.py:787`), but `head_l2=0.01`
is ABSOLUTE, so at ~110k docs/batch it's negligible against the corpus-summed gradient
and fails to cap `|w|` — the newton head jumps to `|w|=110` in the FIRST step, already
saturated, LONG before `weight_y` warmup turns the correction on (`eff_wy=3.2` at iter 8).
The local repro missed this because `α=1.05` kept θ un-peaked and `z` moderate.

**This probe's one change: a much stronger head ridge (`head_l2` 0.01→5.0) + gentler
`head_lr` (1.0→0.3)** to hold `|w|` in the UNSATURATED band. We are PAST the saturation
peak, so bounding `|w|` should REVIVE the shaping gradient (opposite of the near-side
"ridge weakens shaping" tension the local ladder showed).

## What to read (the in-fit `iter N/8:` trajectory — paste those lines)

- **PREDICTION if saturation is the cause**: `|w_CK|max` stays modest (tens, not ~273),
  `||grad_y||` comes ALIVE (≫1e-80 — e.g. 1e2…1e6), and `corr_relΔλ` rises off 0. That
  CONFIRMS saturation and the fix = a corpus-scale-aware, non-vanishing head ridge (so
  `|w|` can't run to the sigmoid's tail at any corpus size), likely paired with an α
  floor. The head histogram should also show a lower `|w_c|` median.
- **If `||grad_y||` is STILL ~1e-80 and corr still 0** despite bounded `|w|`: the
  saturation is θ-driven (peaked α), not `|w|`-driven — next lever is disabling
  `optimize_doc_concentration` / flooring α, not the ridge.

## Run

```bash
cd ~/repos/CHARMPheno && git pull origin claude/spectral-anchor-topic-k-200nqp && \
  CHARM_SPARK_CONF='spark.locality.wait=0s' make -C analysis/cloud exp ID=94
```

(No `make report` — paste the 8 `iter N/8:` lines + the `[head-starvation probe]` block.)

## Run log

**2026-08-18 — saturation REFUTED; the dead gradient is `∂θ/∂eb` (alpha-collapse).**
With `head_l2=5.0, head_lr=0.3` the ridge worked — `|w_CK|max` bounded to **27** (was 273),
median `|w_c|=1.8`, max 90. But `||grad_y||` stayed **~1e-84** and `corr_relΔλ=0` every iter:

```
|w_CK|max:  8.6 → 14.5 → 18.7 → 21.7 → 23.8 → 25.3 → 26.3 → 27
||grad_y||: 0 → 4.9e-85 → 7.6e-85 → ... → 1.1e-84   (unchanged by the |w| drop)
```

Bounding `|w|` 10× did NOT revive the gradient ⇒ it is not logit-saturation (`z=w·θ`).
`grad_topics = (∂loss/∂θ)·(∂θ/∂eb)`; with `|w|~27` the first factor is fine, so the dead
factor is the CAVI Jacobian `∂θ/∂eb`. Autograd confirms it collapses with alpha:
`||dθ/deb||` = 3.1e-1 (α=0.5) → 2.9e-10 (α=0.05) → **2.7e-90 (α=0.0022=1/K, the cluster)** —
matching the cluster `||grad_y||~1e-84`. The tiny 1/K doc-concentration collapses θ so the
shaping gradient cannot flow. → real fix probed in exp 0095 (`doc_concentration=0.5`).
