---
id: 80
slug: multidomain-gated-pc-firth
status: pending
model_class: gated_pc
cohort: population_rare6
cohort_def: population_rare6
disease: rare6
# 0079 (closure-mask) blew the CO-FIT HEAD to |w_CK|~1.3e4 — the per-node problems
# become separable when trained vs siblings only, and the plain logistic head
# diverges (its detection AUC fell BELOW chance). This is the canonical Firth case.
# 0080 = 0079 + head_penalty=firth (the resurrected cheap Firth: gradient-norm
# backtracking, no slogdet). Hypothesis: Firth bounds |w_CK| to a finite fixed
# point AND yields CALIBRATED P(child|parent) from the co-fit head (low ECE) — the
# usable, VOI-ready conditional-diagnosis model.
extra_domains: measurement
label_mask_mode: closure
head_penalty: firth        # <-- the delta vs 0079: parameter-free separation cure
head_inner_iters: 25       # Firth needs the inner-loop IRLS path (Path B)
head_l2: 0.0               # Firth is parameter-free — no ridge (the firth path ignores it)
# --- everything else identical to 0079 ---
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
weight_y: 50.0
head_optimizer: newton
head_lr: 0.3
head_newton_ridge: 0.05
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
eval_every: 20
num_partitions: 96
seed: 42
cache_uri: hdfs:///user/dataproc/charm/case_finding_cache
---

# 0080 — Multi-domain Gated-PC with the Firth head (closure-mask)

0079 (closure-mask) trains each node against its siblings — the conditional
objective — but the per-node problems become separable and the plain logistic
co-fit head DIVERGED (|w_CK| ~1.3e4, detection AUC below chance). The
head-independent `pc_topics_lr` metric was unaffected, but the co-fit head — the
thing that would emit a calibrated `P(child|parent)` for a diagnostic aid / VOI —
was unusable.

Firth (Jeffreys prior, +½·log det I) is the parameter-free separation cure: as
|w|→∞ the leverage term pulls p back toward ½, bounding |w| with no ridge to tune.
Resurrected cheaply (gradient-norm backtracking IRLS, no slogdet — commit prior).

## The test

- **Does Firth tame the blow-up?** `co-fit head |w_CK|max` should drop from ~1.3e4
  to O(1–10).
- **Is the co-fit head now CALIBRATED?** The new co-fit-head conditional readout
  reports ECE of `P(child|parent)`. Firth should give a low ECE where the blown-up
  plain head could not — the VOI prerequisite (H(p) must be real).
- **Does the co-fit head's conditional discrimination recover?** Compare its
  `cond_AUC` / top-1-vs-majority to the pc_topics_lr headline (which is
  head-independent and unchanged by Firth).

The `pc_topics_lr` numbers should match 0079 (Firth touches only the head). The
win, if any, is entirely in the co-fit head becoming a usable, calibrated
conditional-probability model.

## Run

```bash
cd ~/repos/CHARMPheno && \
  git fetch origin claude/spectral-anchor-topic-k-200nqp && \
  git checkout claude/spectral-anchor-topic-k-200nqp && \
  git pull origin claude/spectral-anchor-topic-k-200nqp && \
  CHARM_SPARK_CONF='spark.locality.wait=0s' make -C analysis/cloud exp ID=80
```

Watch `|w_CK|max` in the iteration log — under Firth it should stay bounded (vs
0079's climb to 1.3e4). If Firth is driver-slow again, that's the cost signal.

## Run log

_(pending first run)_
