---
id: 98
slug: mondo-cv-weighty-sweep-rarity
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
weight_y: 16.0
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

# 0098 — Mondo cardiovascular: weight_y up (16) + rarity split

exp 0097 gave a stable, calibrated 0.74 readout but PC shaping was ~neutral vs the gated
baseline at weight_y=2 (Δ+0.0013) — shaping alive (corr ~1.2%) but too gentle, and the
GATE already aligns unsup topics to labels (unsup 0.741). Two open questions, both
answered by ONE run now:

1. Does STRONGER shaping (weight_y 2 -> 16, ~8x; corr should rise to ~5%/step) lift the
   macro readout?
2. Does it help the LOW-POSITIVE tail specifically (insight 0066)? The HEADLINE now prints
   a RARITY SPLIT (per-node AUC/AP at the median test-positive count, rare vs common,
   gated_pc vs unsup) — a positive rare delta = PC rescues the tail; flat = the gate
   already serves it.

Safe on the head (per-doc-mean bounds |w|), but weight_y=16 raises the ELBO-transient
risk 0097 flagged — watch the first ~10 iters; if ELBO explodes past ~-1e10 and does not
recover, kill and drop to weight_y=8.

## What to read (make -C analysis/cloud report ID=98)

1. HEADLINE gated_pc vs unsup_gated (pc_topics_lr) macro delta — bigger than 0097's
   +0.0013?
2. RARITY SPLIT — rare-node AUC/AP delta (the insight-0066 test).
3. FIT-HEALTH — |w_CK|max bounded, corr steady (~0.03-0.06), ELBO rising (no -1e27 spike).
4. conditional sharpening delta by depth.

## Sequence after this

- Rare delta positive: PC's value IS the low-mass tail — sweep weight_y for the tail, keep
  the gate for the head. Report tail-focused metrics.
- Everything flat even at weight_y=16: PC cannot beat the gate on this branch — the
  deliverable is the gated model itself (0.74, calibrated). Pivot to whole-Mondo K~3800
  scale-up and/or a weak-gate contrast (does the gate NEED supervision at all).

## Run

```bash
cd ~/repos/CHARMPheno && git pull origin claude/spectral-anchor-topic-k-200nqp && \
  CHARM_SPARK_CONF='spark.locality.wait=0s' make -C analysis/cloud exp ID=98
make -C analysis/cloud report ID=98
```

## Run log

**2026-08-19 run 1 — weight_y=16 DETONATED λ (additive correction).** Head stayed
bounded (|w|max 0.09->1.87->...->4.22, per-doc-mean held), but corr_relΔλ hit 9.3% by
iter 3 and ELBO went to -4.5e27 at iter 4 and never recovered; by iter 80 some topics
were starved (Σλ_k min=12.9) while others bloated (1e7). Root: the supervised correction
was an ADDITIVE subtraction on λ with no simplex constraint, so a large weight_y drains
topic mass -> empty topic -> E[logβ] detonates the ELBO. Readout on wrecked topics is
useless (no clean rarity answer from this run).

**FIX (engine): exponentiated-gradient, MASS-PRESERVING correction** (the reference's
simplex-safe update). λ now moves MULTIPLICATIVELY and each topic-row is renormalized to
its unsupervised total mass Σλ_k — no starvation/bloat, λ>0 by construction, weight_y
bounded (numerically verified: Σλ drift 2e-16 at weight_y up to 1000; reduces to the
additive step at small weight_y). RE-RUN exp 0098 UNCHANGED (git pull rebuilds
spark_vi.zip): weight_y=16 should now be STABLE and give the rarity split.
