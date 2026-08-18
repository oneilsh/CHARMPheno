---
id: 95
slug: mondo-cv-alpha-collapse-probe
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
doc_concentration: 0.5
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

# 0095 — Mondo cardiovascular: alpha-collapse probe (the real cause)

exps 0093/0094 killed the head theories: heads TRAIN (|w|~27-273, 0 dead), and bounding
|w| via a strong ridge (0094: 273->27) did NOT revive shaping — `||grad_y||` stayed ~1e-84.
So the dead gradient is not `∂loss/∂θ` (fine at |w|~27) but the OTHER factor,
`∂θ/∂eb` — the differentiable-CAVI Jacobian. Autograd proves it collapses with alpha:

```
alpha:    2.0     0.5     0.1      0.05      0.0022 (=1/K, the cluster)
||dθ/deb||(ni=30): 3.3e-2  3.1e-1  6.7e-6   2.9e-10   2.7e-90   <- ~ the cluster ||grad_y||
```

At `alpha = 1/K = 0.0022` the doc-topic posterior collapses (ψ(0.0022)≈-455 underflows the
unroll) so `∂θ/∂eb ≈ 2.7e-90` — the supervised shaping gradient cannot flow back to the
topics AT ALL, upstream of the head. (Second killer: `grad_cavi_iters=30` also kills the
Jacobian for alpha≤0.1; the SAFE zone is `alpha ≳ 0.5` at any ni. Local runs stayed alive
only because they used alpha=0.05 + ni=8.) The gated engine hard-coded `alpha=1/K` and
IGNORED `docConcentration` (now fixed); `optimize_doc_concentration` was also a silent
no-op on the gated path (alpha pinned at the 1/K init) — which is why the log shows uniform
alpha=0.002252 for the whole fit.

**This probe's one change: `doc_concentration=0.5`** (out of the collapse regime), head
params back to 0093's (head_l2=0.01, head_lr=1.0).

## What to read (the in-fit `iter N/8:` trajectory — paste those lines)

- **PREDICTION**: `||grad_y||` comes ALIVE (≫1e-80 — e.g. 1e3…1e7) and `corr_relΔλ` rises
  off 0 for the first time. That CONFIRMS alpha-collapse as the neutral-PC cause and the
  fix = keep the gated alpha out of the collapse regime (a floored/scalar
  docConcentration, NOT the 1/K default; optionally decouple a moderate shaping-only alpha
  to preserve a sparse inference prior if the readout wants it).
- **If still dead**: the collapse is `grad_cavi_iters`-driven too — next lever is a smaller
  shaping-unroll depth (the map shows ni=8 keeps alpha=0.5 alive with margin).

Watch `α[...]` in the log now reads 0.5 (not 0.002252), and the head |w| may shift since
the topics it reads are no longer collapsed.

## Run

```bash
cd ~/repos/CHARMPheno && git pull origin claude/spectral-anchor-topic-k-200nqp && \
  CHARM_SPARK_CONF='spark.locality.wait=0s' make -C analysis/cloud exp ID=95
```

(No `make report` — paste the 8 `iter N/8:` lines + the `[head-starvation probe]` block.)

## Run log

_(pending first run)_
