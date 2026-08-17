---
id: 89
slug: localized-head-41anchor
status: pending
model_class: gated_pc
cohort: population_rare_priority
cohort_def: population_rare_priority
disease: rare_priority
# LOCALIZED HEAD validation at the 41-anchor scale, before whole-Mondo (insight 0071).
# Exact 0085 config (closure, 41-anchor, 3-domain, ridge-only) with the ONE delta:
# localize_head=true — each node's logistic reads ONLY its topic support (gated block +
# ancestors + background, DagLayout.allowed(c)), not all K, so the per-node Newton is
# O(|support|^3) not O(K^3). ADR 0042 done right: hierarchy in the head SUPPORT, not a
# closure product (which collapses with the gate). Read: does the LOCALIZED head match
# the DENSE head's conditional AUC / calibration / per-node reliability (0082/0085)? If
# yes, locality is validated and whole-Mondo K~3,800 becomes tractable as ONE co-fit.
extra_domains: measurement,drug
label_mask_mode: closure
localize_head: true
# --- everything else identical to 0085/0082 ---
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

# 0089 — Localized head at 41-anchor scale (validate before whole-Mondo)

Config-only variation of 0085: `localize_head: true`. The gated co-fit head now reads,
per node, only `DagLayout.allowed(c)` — background + its block + ancestors' blocks
(~O(depth) topics) — instead of all K=101. At K=101 the compute saving is negligible;
the point is a **quality equivalence check**: does constraining the head to local topics
preserve the conditional AUC / calibration / per-node reliability we got with the dense
head (0082 co-fit ECE 0.0098, 0085 per-node max 0.72; cond AUC ~0.70)?

## Why this is the go/no-go for whole-Mondo

The dense head is O(K³·C) / O(C·K²) memory — ~850 GB at whole-Mondo K≈3,800 (insight
0071). The localized head is O(C·depth³) — trivial. If it matches the dense head here, the
whole-Mondo backbone (exp 0088) is fittable as ONE co-fit (not a piecemeal cascade). If it
degrades, we learn the head needs the far-away contrast topics and reconsider.

## What to read

- **co-fit head conditional AUC / top-1 vs majority** vs 0085's dense numbers — equal or
  better ⇒ locality is free.
- **per-node reliability (mean / max ECE)** vs 0085 (mean 0.095, max 0.72) — does
  restricting support help or hurt the degenerate small nodes?
- **|w_CK|max** — should stay bounded (ridge unchanged); locality shouldn't change it much.
- **λ-mass specialization** — unchanged (topic side is identical; only the head support
  changed).

## Run

```bash
cd ~/repos/CHARMPheno && git pull origin claude/spectral-anchor-topic-k-200nqp && \
  CHARM_SPARK_CONF='spark.locality.wait=0s' make -C analysis/cloud exp ID=89
```

Paste the two `[conditional sharpening: ...]` blocks (incl. per-node reliability) so we
can put localized-vs-dense side by side against 0085.

## Run log

### Run 1 (localized support = path + ancestors, NO siblings) — collapses at depth 0 exactly as predicted; readout LR unchanged → purely a head-support issue

The co-fit head degraded, but **only at the wide top level**:

| | 0085 dense head | 0089 run 1 (localized, no siblings) |
|---|---|---|
| cond AUC depth 0 (29 siblings) | 0.7035 | **0.5743** (near chance) |
| cond AUC depth 1 | 0.6659 | 0.6685 |
| cond AUC depth 2 | 0.6983 | 0.7031 |
| co-fit macro AUC | 0.7092 | 0.6278 |
| co-fit ECE | 0.0130 | 0.0317 |

- **The readout LR is unchanged** (macro AUC 0.7316 vs 0085's 0.7234; cond AUC/ECE
  identical to unsup) — so the topic side is untouched; this is *purely* the head's
  support. |w|max bounced (129→284→929, all <1000) — the ill-supported depth-0 nodes
  thrash.
- **Diagnosis (as predicted in the run-1 caveat):** a root child's support was
  `background + its own block` only (root has no block, and run 1 included only
  ancestors). To rank it against its **28 siblings** the head must see *their* topics
  ("high on me AND low on them"); without them it sees only its own activation and can't
  contrast → collapse. Depth 1/2 survived because their fan-out is small (2–7 siblings)
  and own-block activation carries it.
- **Fix:** the support must include SIBLINGS (the closure objective's contrast set) —
  `DagLayout.allowed_with_siblings(c)`. Still O(depth + local fan-out) ≪ K, so it stays
  the whole-Mondo scale fix. This is now the default when `localize_head` is on.

### Run 2 (localized support = path + ancestors + SIBLINGS) — the sibling fix works; localized ≈ dense; VALIDATED for whole-Mondo

The sibling fix recovered the depth-0 collapse. Co-fit head cond AUC by depth:

| | run 1 (no sibs) | run 2 (+sibs) | dense 0085 |
|---|---|---|---|
| depth 0 (29 sibs) | 0.574 | **0.685** | 0.704 |
| depth 1 | 0.669 | 0.670 | 0.666 |
| depth 2 | 0.703 | 0.703 | 0.698 |
| macro AUC | 0.628 | 0.696 | 0.709 |
| co-fit ECE | 0.032 | 0.014 | 0.013 |

- **Depth-0 recovered** from near-chance (0.574) to 0.685 — the sibling contrast set is
  exactly what a wide-fan-out node needs. Depth 1/2 match dense (they already had their
  few siblings' signal via own-block activation).
- **Localized+siblings ≈ dense**, within ~0.013 macro AUC. A small residual gap remains at
  **depth 0** (0.685 vs 0.704, Δ−0.018): a top-level node's support is bg + own + its ~29
  siblings (~38 of 101 topics), so it still can't use *distant* (non-sibling) topics as
  contrast the way the dense head can. Depths 1/2 are dead-on.
- **Readout LR unchanged** (0.7206 vs 0085's 0.7234; topic side untouched) — confirms the
  gap is purely the head's restricted support, as designed.
- **|w| excursed to ~500 mid-fit then came back** (ridge caught it) — the localized head's
  smaller per-node problems are a touch less stable than dense; not divergent, but watch it
  at Mondo scale (head_l2 bump if needed).
- PC vs unsup ~neutral-to-slightly-negative (readout Δ−0.009, cond Δ−0.008) — within the
  marginal-PC-benefit noise on this information-limited data (insight 0064); not a
  localization effect per se.

**Read — VALIDATED.** Localized+siblings recovers the run-1 collapse and matches the dense
head within ~0.01–0.02 AUC. Since the dense head is a hard 850 GB wall at whole-Mondo
(insight 0071), the localized head is the enabling architecture, and the residual gap is a
small, acceptable trade. Crucially the gap is at the **flat top level** — a 41-anchor
artifact (root→29 diseases). The real Mondo tree has DEPTH (root→~25 body systems →
branches), so most discrimination is *within-branch* (depths 1/2), exactly where
localization matched dense EXACTLY. Expect the Mondo fit to localize *better* than this
worst-case flat DAG. Green light for the whole-Mondo run (with the cost-profile watch on
high-fan-out parents).
