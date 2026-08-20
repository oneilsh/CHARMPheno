---
id: 102
slug: mondo-cv-path-cousins-head
status: done
model_class: gated_pc
cohort: population_mondo_cardiovascular
cohort_def: population_mondo_cardiovascular
disease: rare_priority
# STRUCTURAL SUPPORT EXPANSION (Family A, exact Newton kept). exp 0099 localized co-fit head
# capped at 0.585 (localized ceiling ~0.635 vs full-K 0.706 — localization is ~0.07 lossy).
# exp 0101 tried DENSE full-K to recover it and hit the O(C*K^2) Hessian wall (4.3 GiB driver
# collect + preemptible-executor thrash, never finished). This run keeps the head LOCALIZED
# and BOUNDED (shuffleable, exact Newton) but WIDENS the support from closure+siblings to
# closure + siblings-of-every-ancestor (path_cousins) — the contrast set up the whole
# hierarchy. Tests whether a wider-but-bounded head recovers the ~0.07 the closure-only head
# lost, WITHOUT the dense head's cost. Single delta vs 0099: head_support=path_cousins.
dag_source: mondo
mondo_branch: MONDO:0004995
min_positives: 100
mondo_version: 2026-06-02
mondo_cache_dir: data/mondo
extra_domains: measurement,drug
label_mask_mode: closure
localize_head: true
head_support: path_cousins_kids
head_intercept: true
head_standardize: true
doc_concentration: 0.5
readout_sample_frac: 0.3
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

# 0102 — Mondo cardiovascular: path-cousins localized head (bounded support expansion)

**Why.** The co-fit shaping head has been the blocker: 0099's localized head capped at 0.585,
and the sign of PC's effect on the readout tracks the co-fit head's quality (a weak supervisor
shapes the topics the wrong way). Full-K is the best *estimator* (0.706 readout / 0.742 head)
but a dense full-K *co-fit* head is not shuffleable — 0101 hit the O(C·K²) Hessian wall (4.3 GiB
> driver cap) and thrashed on preemptible executors. So the plan is **bounded support, exact
Newton kept**: recover full-K-class head quality without the dense cost.

This is the first of two bounded-support variants (structural): widen each node's head support
from **closure + immediate siblings** (`allowed_with_siblings`) to **closure + the siblings of
every ancestor on its root-path** (`allowed_with_path_cousins`). That gives the head the
contrast set at *every* level of the hierarchy, not just the leaf level — the out-of-immediate-
neighborhood signal the closure-only head misses. Support stays O(depth·fan-out) ≪ K, so the
per-node Fisher (C·S²) and solve (C·S³) stay localized-cheap: the cost profile should print a
`head support/node` a few × larger than 0099's (~18) but still tiny vs dense K=444.

Single delta vs 0099: `head_support=path_cousins`.

## What to read (make -C analysis/cloud report ID=102)

1. **co-fit head macro AUC** — does it rise above 0099's **0.585** toward the full-K 0.706? The
   direct test that a wider bounded support recovers the localization loss.
2. **HEADLINE gated_pc vs unsup_gated (pc_topics_lr)** — does the readout LIFT (Δ>0)? If the
   better supervisor now helps, structural expansion is the fix.
3. **[cost] head support/node** — confirm it stayed bounded (expect p50 ~30–50, not K); the run
   should NOT hit the dense-head 657 MB Fisher / driver-collect wall, and should survive
   executor preemption (light stats → cheap recompute).
4. **corr_relΔλ, |w_CK|max** — standardized head, so |w| large is cosmetic; judge by co-fit AUC
   + readout Δ.

## Interpreting

- **co-fit AUC up AND readout Δ>0:** structural expansion recovers head quality at bounded cost
  — the production head. Next: compare against MI-selected top-m (the data-driven variant) to
  see if selection beats structure.
- **co-fit AUC up but readout still flat/negative:** confirms the *headroom* hypothesis — the
  unsup gate is already near-ceiling and a better supervisor can't move it. Pivot to "PC helps
  only where the gate is weak (rare tail)".
- **co-fit AUC still ~0.585:** path-cousins didn't capture the missing signal → the lost ~0.07
  lives OUTSIDE the DAG neighborhood entirely (comorbidity) → go to MI-selected support.

## Run

```bash
cd ~/repos/CHARMPheno && git pull origin claude/spectral-anchor-topic-k-200nqp && \
  CHARM_SPARK_CONF='spark.locality.wait=0s' make -C analysis/cloud exp ID=102
make -C analysis/cloud report ID=102
```

## Run log

**2026-08-20 — path_cousins INSUFFICIENT; iterating to +children (path_cousins_kids).**
Head-formulation ladder (frozen θ, localized=path_cousins support): localized ceiling
(sklearn oracle, intercept+standardized) = **0.625** vs **full-K readout = 0.684**. The gap
barely moved from 0099's siblings-only (0.635 vs 0.706 → 0.071) to path_cousins (0.059) — so
widening the contrast set UP the tree recovered almost none of the localization loss. The
missing ~0.06 is not in the DAG neighborhood (siblings or path-cousins). Feasible & fast: 30s/iter,
no collect wall. Stability note: at weight_y=16 the path_cousins head over-drove near the end
(corr_relΔλ=0.137, ‖grad_y‖→9.6e14, ELBO −137M→−210M last iter) — flag for a weight_y/trust pass.
DEV confirmed working (supervised fit ran 30 iters; unsup twin ran 100 because baseline_max_iter
wasn't cut — now fixed). readout (dev, 30 iter): gated_pc pc_topics_lr macro AUC 0.684.
→ Now testing head_support=path_cousins_kids (adds v's children's blocks, the subtype signal) to
see if in-subtree descendant signal helps where up-tree cousins did not. If still ~0.625 ceiling,
the gap is out-of-neighborhood comorbidity → MI-selected support (in progress).

**2026-08-20 — FULL 100-iter run (`CHARM_DEV=0`, path_cousins_kids): the record. Dev CONFIRMED;
the head ceiling is real; PC negative in EVERY rarity quartile, including Q1.**

- **co-fit head (as trained) 0.567** — vs 0.572 at dev-30-iter: full training does NOT recover
  it; the ceiling is real, not undertraining. `|w_CK|max=1.5e5` (the vanishing-relative-ridge
  blowup again, insight 0067's signature). Ladder (frozen θ, localized support): engine Newton
  converged 0.549 → +fixed ridge 0.534 → +fixed+intercept 0.601 → sklearn oracle **0.612** →
  **full-K readout 0.688**. Even the localized oracle sits 0.076 below full-K — the missing
  signal is out-of-DAG-neighborhood, as the path_cousins step already indicated.
- **HEADLINE pc_topics_lr: 0.6876 vs unsup 0.7395 (Δ−0.0519)**; AP 0.4830 vs 0.5396 (Δ−0.0567);
  node P@R0.9 Δ−0.0157; detection AP Δ+0.0000.
- **Quartile rarity split (176 shared nodes, +ct edges [57, 175, 490]):** Q1 rarest (+ct 20–57)
  AUC Δ−0.0348 / AP Δ−0.0388; Q2 Δ−0.0561; Q3 Δ−0.0594; Q4 common Δ−0.0572. **Negative in all
  four quartiles — the Q1 rare-tail-rescue hypothesis (insight 0066's predicted headroom) is
  refuted at this head quality.** Q1 is the *least* negative, but not positive.
- **Conditional:** cond AUC Δ−0.0519, cond AP Δ−0.0567, multiclass top1 Δ−0.0276.

Reads: (1) the `CHARM_DEV` ranking loop is **validated** — dev (0.681/0.739, head 0.572) and
full (0.688/0.7395, head 0.567) tell the same story with the same ordering; (2) the PC-arc
closeout (`docs/reports/2026-08-20-pc-arc-closeout-…`) now rests on a full-run record, not dev
numbers; (3) per its §6, the revival condition (co-fit ≥ gate ≈ 0.74) stands, now with the
rare-tail escape hatch closed at current head quality — a future co-fit head must clear the bar
on Q1 too, not just macro.
