---
id: 101
slug: mondo-cv-full-k-head-diagnostic
status: superseded
# SUPERSEDED. Dense full-K co-fit head is not shuffleable — hit the O(C*K^2) Hessian-collect
# wall (4.3 GiB > driver cap) and thrashed on preemptible executors (~10 min/iter, near-zero
# corr: weak AND infeasible). Full-K is the best ESTIMATOR but not a viable co-fit SHAPING
# head. Replaced by bounded-support + exact Newton (0102 path-cousins; then MI-selection).
model_class: gated_pc
cohort: population_mondo_cardiovascular
cohort_def: population_mondo_cardiovascular
disease: rare_priority
# HEAD-QUALITY DIAGNOSTIC. Clean A/B vs 0099: the ONLY change is localize_head true->false
# (full-K co-fit head instead of the localized one). Tests whether the cluster's PC harm
# (0099: Δ-0.04) came from a WEAK SUPERVISOR, not from shaping over-drive. The sparse-many-
# topic harness could NOT reproduce the harm: across corr up to 2.0, |w| up to 1e10, and
# 14/120 starved nodes, shaping ALWAYS helped (readout +0.30). The one thing the harness
# does NOT have is the cluster's lossy LOCALIZED head (0.585, capped 0.635 by localization
# vs 0.706 full-K). Hypothesis (consistent with 0098's "a blind supervisor shapes topics
# the wrong way"): the localized head is a bad compass, so shaping drags the readout toward
# its bad predictions. If the full-K head (0.706 ceiling) makes PC HELP, head quality is the
# lever and DAG-heterogeneous localization loss is the cluster-specific poison.
dag_source: mondo
mondo_branch: MONDO:0004995
min_positives: 100
mondo_version: 2026-06-02
mondo_cache_dir: data/mondo
extra_domains: measurement,drug
label_mask_mode: closure
# THE SINGLE DELTA vs 0099: localized head OFF (full-K co-fit head, the stronger supervisor).
localize_head: false
head_intercept: true
head_standardize: true
doc_concentration: 0.5
readout_sample_frac: 0.3
# match 0099 exactly so head localization is the ONLY variable (NOT 0100's lowered wy).
weight_y: 16.0
head_lr: 1.0
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
grad_cavi_iters: 15
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

# 0101 — Mondo cardiovascular: full-K co-fit head (head-quality diagnostic)

**Why this run.** exp 0099 (standardized localized head, weight_y=16) HURT the readout at
K=444 (Δ−0.04). I originally diagnosed this as SHAPING OVER-DRIVE (corr_relΔλ=1.83). The
sparse-many-topic harness was built to reproduce that harm and **refuted it**: at C=40 and
C=121, pushed to **corr 2.0** (above the cluster's 1.83), **|w| up to 1e10**, and **14/120
nodes <1% prevalence**, PC supervision **always helped** the readout (up to +0.30). The
trust-region cap (which pins corr to a target) is a pure *tax* in the harness — it never
rescues, because there is no harm to rescue. So over-drive is NOT the cluster's problem.

The discriminating variable between "harness helps" and "cluster hurts" is the **co-fit head
quality**:

| | co-fit head AUC | ceiling | readout Δ |
|---|---|---|---|
| harness (localized, shallow balanced DAG) | 0.678 | — | **+0.30** |
| cluster 0099 (localized, real Mondo DAG) | 0.585 | 0.635 (localized) vs 0.706 (full-K) | **−0.04** |

The localized head is capped at 0.635 by its restricted support and only reached 0.585 — a
weak supervisor. Real Mondo's heterogeneous, overlapping closures make localization far more
lossy than the harness's clean balanced tree (the harness never drops below the full-K head,
which is why it can't reproduce the harm). Per 0098: *a blind supervisor shapes topics the
wrong way.*

**This run flips the single knob** `localize_head: true → false` so the co-fit head reads all
K topics (0.706 ceiling — a good supervisor), keeping everything else identical to 0099
(standardize, intercept, weight_y=16). It is a clean A/B on head localization.

## What to read (make -C analysis/cloud report ID=101)

1. **HEADLINE gated_pc vs unsup_gated (pc_topics_lr)** — does PC now LIFT the readout (Δ>0),
   reversing 0099's −0.04? If YES → **head quality is the lever**; the fix is the head
   (de-localize, or supervise from a stronger head), NOT shaping magnitude.
2. **co-fit head macro AUC** — should rise toward ~0.70 (full-K ceiling) from 0099's 0.585.
   This is the direct check that the supervisor got stronger.
3. **corr_relΔλ** — will likely still be large (weight_y=16, standardized). If PC helps
   *despite* high corr, that is positive confirmation that corr/over-drive was a red herring.
4. **RARITY SPLIT** — where does the flip come from (rare vs common nodes)?
5. **|w_CK|max** — will be large (full-K standardization); judge by readout Δ, not |w|.

## Interpreting the outcomes

- **PC helps (Δ>0):** head quality confirmed. Localization was the poison. Next: either run
  the co-fit head full-K (accepting the unified-head story shifts from "localized" to
  "full-K") or find a middle localization that keeps most of the 0.706 without the harm. The
  trust region stays gated-off (not the fix).
- **PC still hurts (Δ<0) with a 0.70 head:** head quality is NOT sufficient — something else
  the harness lacks is in play (multi-domain λ interaction, real-data confounding, DAG
  overlap directly). Next diagnostic: single-domain cluster run, or per-node Δ vs DAG
  structural features (closure size, sibling overlap) to localize the harm in DAG-space.

## Run

```bash
cd ~/repos/CHARMPheno && git pull origin claude/spectral-anchor-topic-k-200nqp && \
  CHARM_SPARK_CONF='spark.locality.wait=0s' make -C analysis/cloud exp ID=101
make -C analysis/cloud report ID=101
```

## Run log

**2026-08-20 — HIT the O(C·K²) Hessian-collect wall (the predicted distributed bottleneck).**
The gated_pc fit itself finished (421s), but the run aborted collecting the head statistics:
`Total size of serialized results of 6 tasks (4.3 GiB) is bigger than spark.driver.maxResultSize
(4.0 GiB)`. This is exactly the full-K exact-Newton cost: the per-node head Hessian sufficient
statistic is `(C, K+1, K+1)` = 437×445²×8B ≈ 692 MB, treeAggregated across 6 partial-aggregate
tasks ≈ 4.3 GiB > the 4 GiB driver cap.

- **CV-scale band-aid (diagnostic only):**
  `CHARM_DRIVER_MEMORY=16g CHARM_SPARK_CONF='spark.locality.wait=0s spark.driver.maxResultSize=12g' make -C analysis/cloud exp ID=101`
  This does NOT scale: at whole-Mondo (K≈3,800) the same stat is ~400 GB.

- **Conclusion / pivot:** full-K is the best *estimator* (readout 0.706 / head 0.742 per 0097/0099),
  but a literal full-K *co-fit shaping head* is not shuffleable and never will be at scale. The
  production path is **MI-selected bounded support**: per node, the top-m most-predictive topics
  (incl. the out-of-closure comorbidity topics the localized head misses — the ~0.07 loss), giving
  full-K-quality shaping at an O(C·m²) stat that fits. Superseded as a production candidate by that;
  keep 0101 (with the band-aid) only as a diagnostic that full-K *quality* helps shaping.
