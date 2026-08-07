# VI-native Prediction-Constrained topic model in `spark-vi` — design

**Date:** 2026-08-07
**Status:** Design (grounded in the code). Implementation staged (increment 1 first).
**Oracle:** the in-memory `analysis/pc/` reference (exact, slow) validates every increment.

## Bottom line

- The **unsupervised SVI machinery is fully present and reusable**; PC's **supervised content is
  100% new code**. `spark-vi` has no sLDA / logistic head / label-aware topic model (grep-confirmed:
  STM "covariates" are a *prevalence* prior regression, not outcome prediction; gated-LDA "labels"
  are DAG gating).
- PC-VI = `[VIRunner SVI loop, unchanged]` + `[label-free local step]` + `[global step = unsupervised
  natural-gradient λ + weight_y-scaled supervised topic correction + logistic-head SGD]`.
- **`OnlineSTM` is the precedent** that de-risks the hard part: it already seeds extra global params in
  `initialize_global`, accumulates their partial sufficient stats per-partition in `local_update`, and
  applies a **ρ-blended non-conjugate M-step in `update_global`** (ridge M-step on `Γ`, M-step on `Σ`),
  all damped by the runner's Robbins–Monro `rho_t`. PC's head + supervised correction attach the same way.

## Reuse map (file:line)

**Reused unchanged:** `VIRunner.fit` minibatch/RM/tree-reduce loop (`core/runner.py:197-380`);
`transform`/`infer_local` plumbing (`runner.py:382-408`); `VIModel` contract + default `combine_stats`
(`core/model.py:26-213`); RM schedule + `subsamplingRate`→minibatch + corpus-scaling
(`runner.py:190-259`); label-free per-doc inference `_cavi_doc_inference` (`models/topic/lda.py:50-95`);
mllib Estimator/Model scaffolding + `_vector_to_bow_document` (`mllib/topic/lda.py`, `mllib/_common.py`).

**Reused as pattern (copy shape, new math):** `OnlineLDA.update_global` λ natural-gradient
(`lda.py:356-357`); the whole `OnlineSTM` extra-global lifecycle — seed in `initialize_global`
(`stm.py:573-612`), partial stats in `local_update`, ρ-blended non-conjugate M-step in `update_global`
(`stm.py:755-880`), richer row type `STMDocument` (`types.py:49-67`), shim column threading
(`mllib/topic/stm.py`).

**New:** `PCDocument` row type; `OnlinePCLDA(VIModel)` in `models/topic/pc.py`; the supervised head math
(logistic `log σ`, `∂/∂w_CK`, `∂/∂π`) ported to numpy from `analysis/pc/head.py` + `slda_reference.py`;
`PCEstimator`/`PCModel` shim in `mllib/topic/pc.py` (threads `labelCol`, `labelMaskCol`, `weightY`).

## Architecture (update equations, high level)

- **Local step (`local_update`, per Spark partition, LABEL-FREE — the label never enters here):** infer
  label-free `π_d`; emit `lambda_stats` (unsupervised, as LDA) and — for observed cells only,
  `obs_dc = y_rowmask[d]·label_mask[d,c]` — the partial head gradient `Σ_d ∂loss_y/∂w_CK` and partial
  supervised topic gradient `Σ_d ∂loss_y/∂topics`. All dense additive arrays → default `combine_stats`
  sums them; `treeReduce` bounds driver memory.
- **Global step (`update_global`, driver, at `rho_t`):** (a) unsupervised λ natural-gradient (unchanged);
  (b) **supervised topic correction** `λ ← λ − ρ·weight_y·(∂loss_y/∂topics stats)` (non-conjugate,
  gradient); (c) **head SGD** `w_CK ← w_CK − ρ·(∂loss_y/∂w_CK + weight_y·λ_w·2·w_CK)` (STM-`Γ` template).
  λ's unsupervised part stays closed-form; head + supervised correction are gradient steps the RM `rho_t`
  already damps.
- **`infer_local` = the identical label-free routine** as `local_update` ⇒ train/test π consistency (the
  faithfulness invariant, mirroring `OnlineLDA.infer_local` at `lda.py:433-459`).
- **Distribution:** identical to OnlineLDA/STM — per-partition local + partial stats, driver-side global.
  Semi-supervised asymmetry is free (unobserved cells contribute 0 to supervised stats, full lambda_stats).

## Staged implementation + validation (oracle = `analysis/pc`)

- **Increment 1 — unsupervised SVI path (`weight_y=0`).** Stand up `OnlinePCLDA` + `PCEstimator`/`PCModel`
  + `PCDocument` end-to-end with the head present but `weight_y=0`, so `update_global` reduces to the LDA
  λ step and the head stays at init. **Zero supervised risk.** Validate: on the `analysis/pc` synthetic
  known-signal generator, **recovery parity** (Hungarian-matched topic cosine within tol; heldout
  θ/predictive structure vs `PCTopicModel(weight_y=0)`) — *not* numeric identity (see estimator gap).
- **Increment 2 — supervised global correction (`weight_y>0`).** Add head SGD + supervised topic
  correction to `update_global`, emit supervised partial stats from `local_update`. Validate: (a) a
  `check_grad`-style test that the per-minibatch supervised gradient matches `analysis/pc`'s gradient on a
  tiny fixed batch; (b) on vendored `toy_bars_3x3` + the Hughes-regime synthetic, PC strict-beats the
  two-stage baseline on heldout AUC, reproducing `analysis/pc/evaluate.py` numbers within a stochastic-SVI
  tolerance band.

## Decisions / open questions (mostly bite at increment 2)

1. **Estimator gap (increment-1 tolerance).** The reference's label-free local step is **NEF-MAP point π**
   (`nef_map_pi_DK`, 100 unrolled exp-grad steps); `OnlineLDA`'s is **mean-field CAVI Dirichlet γ**. Same
   role, different estimator — they agree on strong planted signal but not bit-for-bit even at
   `weight_y=0`. So increment-1 oracles at **recovery parity**. Choice: reuse **L-CAVI** (zero new risk)
   vs a **L-NEF** short unroll (faithful to the reference estimator). Recommend L-CAVI for increment 1;
   revisit only if increment-2's grad-check needs the reference's exact estimator.
2. **The supervised gradient & the autograd charter (biggest, increment 2).** `spark-vi` is numpy/scipy
   only by charter; the supervised topic correction `∂loss_y/∂topics` flows through `π_d=f(topics)`.
   Options: **(a)** hand-code the numpy VJP through a **short** NEF unroll (5–20 steps vs the reference's
   100) — faithful, fast, charter-clean, but genuinely new math; **(b)** allow `autograd` in `pc.py` only
   (autograd already ships to the cluster overlay) — fast to build, but heavier per-executor and partly
   re-imports the gradient-through-inference cost the VI port exists to escape; **(c)** free-π
   "supervised-VI" (label-shaped local step, no through-inference gradient) — tractable but **not
   faithful** (train/test π mismatch — the exact thing the reference removes; off the table for the
   faithful port). Design recommendation: **(a)**, validated by increment-2's `check_grad`.
3. **Short-unroll depth** — how few NEF steps preserve recovery (the reference fixes 100; SVI takes many
   noisy steps so may tolerate far fewer). Empirical, settle in increment 2.
4. **RM ↔ `weight_y` coupling** — one `rho_t` damps λ, head, and correction; large `weight_y` + aggressive
   early `rho` can destabilize the head (STM hit analogous softmax-saturation runaways). May need a
   separate head step-size or `weight_y` warmup. Watch the ELBO/AUC trace.

## File plan

- `spark-vi/spark_vi/models/topic/types.py` — add `PCDocument` (BOW + `y` (C,) + `label_mask` (C,)).
- `spark-vi/spark_vi/models/topic/pc.py` — `OnlinePCLDA(VIModel)`.
- `spark-vi/spark_vi/mllib/topic/pc.py` — `PCEstimator`/`PCModel` (mirror `mllib/topic/lda.py`).
- `spark-vi/tests/test_pc_lda.py` + `test_pc_lda_shim.py` — increment 1 (`weight_y=0` recovery vs
  `analysis/pc`) then increment 2 (`check_grad` + AUC-beats-two-stage on `toy_bars`).
