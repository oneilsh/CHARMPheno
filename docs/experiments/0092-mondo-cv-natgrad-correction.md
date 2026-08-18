---
id: 92
slug: mondo-cv-natgrad-correction
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
readout_sample_frac: 0.3
# --- the deltas vs 0091: natural-gradient correction (engine, automatic) + recalibrated wy ---
weight_y: 10.0
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

# 0092 — Mondo cardiovascular: natural-gradient supervised correction (the fix)

The make-or-break run for the PC thesis at scale. insight 0072 root-caused the
"PC is neutral" result as a SCALING BUG: the supervised λ-correction was a raw-gradient
step (`∝ 1/λ²`) that vanished at whole-population λ (~1e6). ADR 0044 replaces it with
the natural gradient (`∂L/∂E[logβ] = grad_eb·eb`), scale-stable. `topic_trust` is
retired. `weight_y` re-calibrated to 10 (was 50 compensating for the dead correction).

## What to read (use `make -C analysis/cloud report ID=92`)

REQUIRES the `||grad_y||` diagnostics (commits b67c3c3 / 73050d8) — the cluster was at
ca0e3c4; `git pull` before re-running. Local reproduction (scratch:
realistic_localized.py) PROVED the natural-gradient correction shapes topics HARD when
a node has signal (UNSUP topics-LR 0.53 → PC 0.95, Δ+0.42, corr 0.06–0.12, |w|~1e4),
and could NOT reproduce a corr≈0 at K=20/D=2500 even with localization. So the working
hypothesis for THIS run's neutral-PC is **whole-Mondo shaping STARVATION**: ~3,800
nodes, most with ~100 positives out of ~1e6, so each localized Fisher `H_c`/gradient
`g_c` is degenerate → the newton solve returns `w_c≈0` → `grad_topics≈0` → `corr≈0`.
The report now disambiguates this directly:

1. **`||grad_y||` + `|w_CK|max` trajectory (first/mid/last/peak)** — THE decider.
   * `||grad_y||` peak ≈ 0 AND `|w_CK|max` ≈ 0 (a converged head sits ~70) across ALL
     iters ⇒ CONFIRMED starvation: the localized head never trains. `corr_relΔλ=0` bit-
     exact is then the seed-iteration value (grad ∝ w_CK = 0), not a correction bug.
     → the fix is on the SPARSITY axis (well-populated node selection, cross-node
     hierarchical shrinkage so rare nodes borrow strength, or a pooled shaping signal) —
     NOT head convergence (which the local ladder showed TRADES AWAY the readout).
   * `||grad_y||` peak > 0 but `corr_relΔλ` ≈ 0 ⇒ the correction isn't applying (build) —
     re-open. (Local repro says this shouldn't happen at ca0e3c4+.)
2. **`corr_relΔλ` trajectory** — peak, not just last. A nonzero peak = shaping fired on
   SOME iter (head had magnitude then); a bit-exact-0 peak = never fired.
3. **HEADLINE `gated_pc vs unsup_gated` readout** — the payoff. Δ>0 = shaping recovered.
   Under the starvation hypothesis this stays ≈0 until the sparsity fix.
4. **ELBO trend** — should rise; a fall = the (live) correction destabilizing λ.

## Sequence after this

- **If starvation confirmed** (grad/|w|≈0 across iters): the lever is per-node label
  sparsity, not the head. Options to prototype (local repro can be pushed into the sparse
  regime to pick one): (a) restrict to nodes with ≥N effective positives; (b) hierarchical
  shrinkage — rare nodes' head borrows from ancestors+siblings (a proper prior on `w_c`,
  not the vanishing relative ridge); (c) a pooled/ancestor-tied shaping signal so a rare
  leaf still moves its ancestor topics. The co-fit head **intercept** is DEMOTED — local
  frozen-θ ladder showed it neutral (0.982 vs 0.983); it's a co-fit-beauty nicety, not the
  deliverable lever.
- **If grad alive but corr≈0**: correction-application bug — re-open the engine path.
- Re-confirm any positive result on the 41-anchor 0089 setup (a second scale) before
  rewriting the neutral-PC insights.

## Run

```bash
# MUST pull first: the ||grad_y||/|w| diagnostics (b67c3c3, 73050d8) post-date the
# ca0e3c4 the cluster ran the first time, and they are what makes this run self-diagnosing.
cd ~/repos/CHARMPheno && git pull origin claude/spectral-anchor-topic-k-200nqp && \
  CHARM_SPARK_CONF='spark.locality.wait=0s' make -C analysis/cloud exp ID=92
make -C analysis/cloud report ID=92
```

## Run log

_(pending first run)_
