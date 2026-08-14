---
id: 79
slug: multidomain-gated-pc-closure-mask
status: pending
model_class: gated_pc
cohort: population_rare6
cohort_def: population_rare6
disease: rare6
# Conditions + value-aware measurement (as 0078) but with the CONDITIONAL training
# objective: label_mask_mode=closure trains each node's head against its DAG
# SIBLINGS (the parent cohort's other children), not against all background.
# Motivated by 0078 run 2: full-mask PC supervision HELPS detection but NOT
# conditional sharpening (P(child|parent)) — because full-mask IS a detection
# objective. Hypothesis: closure-mask aligns the objective with the sharpening
# task, so it should improve the conditional metric where full-mask didn't.
extra_domains: measurement
label_mask_mode: closure   # <-- the one delta vs 0078: conditional (vs-siblings) objective
# --- everything else identical to 0078 ---
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
head_penalty: none
head_inner_iters: 0
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

# 0079 — Multi-domain Gated-PC with the CONDITIONAL (closure-mask) objective

0078 run 2 delivered a clean dichotomy: PC supervision (with `label_mask_mode=full`)
**helps de-novo detection** (det AP Δ+0.025 over the unsupervised twin) **but not
conditional sharpening** — P(child|parent) was marginally *worse* than the
unsupervised fit. The mechanism: full-mask trains every node against *background*,
so it optimizes marginal detection and can blur within-parent subtype distinctions.

`label_mask_mode=closure` observes only each active node's DAG **closure + its
siblings** (the near-boundary negatives = the parent cohort's other children),
leaving distant nodes unobserved. That is exactly the **conditional (vs-siblings)
objective** — training the head to discriminate *within a parent*, which is the
clinician's "which subtype?" task.

## The test

Same readouts as 0078; the number that matters is the **conditional sharpening**
headline (gated_pc vs unsup_gated): cond AP / cond AUC / multiclass top-1 for
P(child|parent). 

- **Hypothesis:** closure-mask flips the sign — conditional sharpening now beats the
  unsupervised twin (where full-mask lost, Δ−0.013).
- **Watch the trade:** closure-mask observes far fewer cells (background docs
  contribute nothing), so de-novo detection AP may *drop* vs 0078's full-mask. The
  interesting outcome is a clean trade — closure BUYS sharpening, full BUYS detection
  — which would say the mask mode is the knob that targets the clinical task.

## Run

```bash
cd ~/repos/CHARMPheno && \
  git fetch origin claude/spectral-anchor-topic-k-200nqp && \
  git checkout claude/spectral-anchor-topic-k-200nqp && \
  git pull origin claude/spectral-anchor-topic-k-200nqp && \
  CHARM_SPARK_CONF='spark.locality.wait=0s' make -C analysis/cloud exp ID=79
```

## Run log

### Run 1 (measurement + closure-mask) — objective-alignment confirmed, but pure closure-mask is NOT deployable; the "AUC lift" is an eval artifact

**Sign-flip confirmed (the prediction).** Supervised vs unsupervised, same closure
mask: cond AUC Δ**+0.0037**, cond AP +0.0054, top1 +0.0036 — all positive, where
full-mask (0078) was negative (cond AP Δ−0.0127). So training the conditional
objective makes supervision help the conditional metric. But the magnitude is tiny.

**Trap 1 — detection collapses to chance.** det AP 0.034 = prevalence, AUC 0.500
(both arms). Closure-mask leaves background docs ENTIRELY unobserved, so the
case-vs-background contrast is gone from training AND scoring. The full↔closure
trade is not partial — pure closure loses detection completely.

**Trap 2 — the head blew up.** |w_CK|max = 1.36e4 (vs ~13 under full-mask).
Closure observes only siblings → few cells → separable per-node problems → the
logistic head diverges. The co-fit head is unusable (its detection AUC 0.47,
below chance); only the head-independent pc_topics_lr metric survives. This is the
separation the parked Firth/ridge work (task #29) targets.

**Trap 3 — the cond_AUC jump (0.62→0.75) is an EVALUATION artifact, not a model
win.** The unsupervised arm jumped identically (0.6232→0.7510), but its θ is
weightY=0 with the same seed/features/frontier as 0078-unsup — i.e. LITERALLY the
same θ. Identical θ, different AUC ⇒ the change is the eval definition: under
closure-mask the conditional cohort's negatives are just SIBLINGS (the mask hides
distant nodes), a cleaner/easier contrast. Honest within-parent discrimination is
still ~0.62; closure just pointed the metric at an easier slice. **Do not headline
"closure lifted AUC to 0.75."**

**Side confirmation.** The unsup λ-mass table (now printed) shows the anchors
condition-heavy in the UNSUPERVISED arm too (Amyloidosis 0.87, Sarcoidosis 0.80,
Cardiac sarcoid 0.94) — the hierarchy-aligned specialization is a property of the
gated multi-domain representation, not PC. Settled.

**Read + next.** "Mask = task selector" is validated but sharpened: pure
closure-mask is not deployable (no detection + head divergence), and its apparent
lift is confounded. The clean design is the two-stage FACTORIZATION —
`P(d|x) = P(d | x, x∈C) · P(x∈C | x)`: **full-mask for the detection factor,
closure-mask for the sharpening factor**, composed at inference (this also keeps
the detection head in the well-conditioned full-mask regime). And the conditional
readout needs a mandatory fix: **evaluate against a FIXED (full-closure) label
definition regardless of the training mask**, or cross-mask numbers are meaningless
(as Trap 3 shows). See the VOI/metrics report
(claude/rare-disease-diagnosis-lit-review-ojs4ms) §2–3.
