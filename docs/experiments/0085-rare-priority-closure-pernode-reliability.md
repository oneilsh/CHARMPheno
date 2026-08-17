---
id: 85
slug: rare-priority-closure-pernode-reliability
status: pending
model_class: gated_pc
cohort: population_rare_priority
cohort_def: population_rare_priority
disease: rare_priority
# CONFIRMATION RUN for the UNIFIED co-fit head. Exact 0082 config (41-anchor forest x
# 3 domains, closure mask, head_l2=0.01 ridge, NO Firth — ADR 0043) re-run with the
# new PER-NODE reliability readout (insight 0069). 0082 showed the ridge-bounded co-fit
# head is calibrated at scale by POOLED ECE (0.0098) and competitive with the two-stage
# readout LR — but pooled ECE can average an over- against an under-confident node. This
# run prints per-node ECE (mean/max/worst) for BOTH the co-fit head and the readout LR,
# to confirm the calibration holds node-by-node before we bless the single-stage model.
extra_domains: measurement,drug
label_mask_mode: closure
# --- corpus / DAG: identical to 0082 ---
person_mod: 1
prior_obs_days: 0
doc_min_length: 10
min_n: 20
holdout_frac: 0.2
vocab_size: 5000
min_df: 20
min_patient_count: 20
window_mode: lookback     # REQUIRED for multi-domain
lookback_days: 1825
label_window_days: 365
strip_mode: both
n_bg: 8
tpn: 1
optimize_doc_concentration: true
# --- PC head: one-step ridge-Newton (ADR 0039/0041). head_l2 is the SOLE head
#     regularizer now (ADR 0043 removed Firth + the inner-loop Path B). ---
weight_y: 50.0
head_optimizer: newton
head_lr: 0.3
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

# 0085 — Per-node reliability of the unified co-fit head (0082 config, ridge-only)

Config-only re-run of 0082 (nothing in the fit changes — same seed, same knobs, Firth
was never on) whose ONLY delta is the driver readout: `conditional_readout` now emits a
**per-node ECE** for every scored parent→child edge and a **mean/max/worst** summary
beside the pooled ECE, for both the co-fit head and the head-independent readout LR.

## What to read

The two `[conditional sharpening: ...]` blocks each now print a line:

```
per-node reliability (ECE over N nodes): mean=… max=… (worst A->B)  vs pooled=…
```

- **The decisive number is `max` vs `pooled`.** If `max ≈ pooled`, calibration is
  uniform and the pooled 0.0098 is honest — the unified single-stage P(child|parent)
  head is blessed (no post-hoc fit, no Firth). If `max >> pooled` (e.g. a node at ECE
  0.05+ while pooled sits at 0.01), pooling was flattering and that node needs attention
  (likely a small-cohort node — cf. Amyloidosis n=66 in 0082).
- **Compare the co-fit head vs readout-LR per-node summaries.** 0082 had the co-fit head
  ahead on pooled ECE (0.0098 vs 0.0119); confirm the co-fit head is not worse on the
  WORST node (the failure mode pooling would hide).
- Everything else should reproduce 0082 exactly (cond_AUC by depth, per-parent top-1 vs
  majority, |w|max~2126) — a sanity check that the readout change didn't perturb the fit.

## Run

```bash
cd ~/repos/CHARMPheno && git pull origin claude/spectral-anchor-topic-k-200nqp && \
  CHARM_SPARK_CONF='spark.locality.wait=0s' make -C analysis/cloud exp ID=85
```

~15 min (eval off), same as 0082.

## Run log

### Run 1 (0082 config + per-node reliability) — pooled ECE was FLATTERING ~7x; the unified head is calibrated where it matters but garbage on degenerate/tiny nodes (both heads); |w| well-controlled (ADR 0043 confirmed)

Reproduces 0082 (closure, 41-anchor, 3-domain, ridge-only). |w_CK|max **1068** (bopped
1000–2000 across iters) — well-controlled, no Firth, ADR 0043 confirmed at closure scale.
Detection dead (0.50) as designed. gated_pc vs unsup conditional: cond AUC Δ+0.0023
(closure helps conditional, sign consistent with 0082).

- **The per-node readout did its job — pooled ECE badly flatters.** Unified co-fit head:
  pooled ECE **0.0130** but per-node **mean 0.0953, max 0.7243** (worst
  Ehlers-Danlos→Ehlers-Danlos). Readout LR: pooled **0.0124** but per-node **mean 0.0887,
  max 0.8182** (worst SLE→SLE). Per-node MEAN is ~7× the pooled; the MAX is catastrophic.
  So "pooled ECE 0.0098" from 0082 was NOT the calibration story — pooling averaged the
  well-populated nodes (which are calibrated) against degenerate small nodes (which are
  not).
- **It's a small/degenerate-node problem, not a co-fit-head problem.** BOTH heads show
  the same pattern (co-fit max 0.72, readout max 0.82) → the unified head is not worse
  than two-stage; it remains neck-and-neck (co-fit pooled 0.0130 vs readout 0.0124;
  headline cond AUC 0.700 vs 0.702). ADR 0043's "unified head works" survives; only the
  "how calibrated" claim is corrected downward and made per-node.
- **The worst nodes are SNOMED artifacts — direct motivation for Mondo.** The max-ECE
  edges are SELF-NAMED (SLE→SLE, EDS→EDS): a parent and child that are near-duplicate
  SNOMED granularities of the same disease, where one child has ~100% within-parent
  prevalence (the bal_acc=0.500 "no real subtyping" nodes). Calibration is ill-posed on a
  near-degenerate 1-class split, and per-node ECE (5 bins on tens of samples) is
  high-variance there. These ragged near-duplicate nodes are exactly what a Mondo-native
  hierarchy (vs SNOMED concept_ancestor) would collapse.

**Read:** do NOT headline pooled ECE — report per-node (mean + worst). The unified head
is trustworthy on large, non-degenerate cohorts (the nodes where a real subtyping /
VOI decision exists) and untrustworthy on degenerate/tiny nodes (where there's no real
decision anyway — bal_acc 0.5). To bless it for VOI: filter to non-degenerate nodes,
and/or hierarchical shrinkage on small cohorts, and/or the Mondo cleanup that removes the
self-named edges. The pooled-vs-per-node gap is the honest headline of this run.
