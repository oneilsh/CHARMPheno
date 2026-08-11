# PC (Prediction-Constrained) walkthrough-review plan — "faithful Hughes, adapted for scalable spark-vi"

**Date:** 2026-08-11
**Branch:** `claude/faithful-flat-pc`
**Scope:** the Prediction-Constrained topic-model production path — the faithful
in-memory reference, its VI/Spark port, the reusable mllib shim, the eval/diagnostics
harness, and the cloud/experiment orchestration. Framed throughout as: *this replicates
Hughes et al.'s Prediction-Constrained sLDA, with the adjustments required to make it
truly scalable under the `spark-vi` distributed-SVI framework.*

**Format:** bottom-up lesson plan (mirrors the case-finding 7-lesson walkthrough in
`docs/REVIEW_LOG.md`). Each lesson lists what to read, the **Hughes-fidelity claim**, the
**scalability adaptation** made for spark-vi, and what a reviewer should **scrutinize**
(invariants that must hold, risks to probe). Log the actual session as a dated `##`
section at the top of `docs/REVIEW_LOG.md` when done.

**Deliberately OUT of scope (deferred to a later arc):**
- **Export + dashboard machinery for PC** — the topic-viewer / faithful-record-completion
  / conditioning-readout path that LDA/STM/gated-STM have (`docs/superpowers/plans/*dashboard*`)
  was never built for PC. Acknowledged gap; defer.
- **The head-convergence ceiling** (head plateaus below the batch-LR ceiling when topics
  don't converge in-budget) — parked as a refinement (Polyak-average / more iters),
  documented in insight 0065.
- **A settled "production" config** (which `head_optimizer` / schedule for a real study).

---

## The Hughes-fidelity ledger (the through-line to check first)

Read `docs/hughes-comparison.md` and ADR 0038 alongside this. The review's spine is
verifying each row: *is the spark-vi adaptation faithful to the Hughes intent, and is the
deviation justified by scalability rather than convenience?*

| Hughes et al. (AISTATS 2018 / JAMA 2020) | spark-vi adaptation | why (scalability) | checked by |
|---|---|---|---|
| Batch L-BFGS joint fit of {φ, η} | distributed **minibatch SVI** (`VIRunner` + `OnlinePCLDA`) | 46k docs, cluster, streaming stats | increment-1 equivalence (`weight_y=0` == `OnlineLDA`); toy `PC > two-stage` gate |
| Per-doc π = MAP via NEF exponentiated-gradient **to convergence** | differentiable **fixed-unroll CAVI** (`_cavi_theta_anp`) for the topic correction; converged `infer_local` for scoring | bound the autograd tape per minibatch doc | `theta_mismatch` localizer (routine gap negligible, cos 0.95) |
| η (logistic head) co-fit under the same batch objective | **Newton/IRLS head** via aggregatable `(g_c, H_c)` sufficient stats | converge a coupled non-conjugate param under minibatch SVI, scale-invariantly, no runner change | insight 0065, ADR 0039, `head_vs_lr_cosine` localizer |
| Gibbs-LDA warm initialization (avoid bad optima) | unsupervised **phase-1 warm-start** (`weight_y=0`, fresh RM schedule) | same purpose, in-framework | `test_pc_warm_start.py` |
| Reported clinical model = **two-stage** (topics → LR/extra-trees), JAMA 2020 | `pc_topics_lr` readout (converged LR on final topics) | same measure of representation quality, convergence-robust | `analysis/pc/evaluate.py` |

If a row's adaptation is unfaithful or unjustified, that's the review's most important find.

---

## Lesson 1 — The faithful reference (`analysis/pc/`): the correctness oracle

**Read:** `analysis/pc/{objective,model,head,generative,variants}.py`, `slda_reference.py`,
`tests/` (esp. the `toy_bars_3x3` gate). ADR 0038.

**Hughes fidelity:** this is the Prediction-Constrained sLDA *in memory* — the trusted
oracle, deliberately NOT the scalable target. `slda_reference.calc_loss__slda` encodes the
reference PC objective; `model.py` is the L-BFGS joint fit; the vendored `toy_bars_3x3`
(K_fit < K_dom, one low-mass predictive topic) is the discriminating gate where PC beats
two-stage (0.56 → 0.91).

**Scrutinize:** Is the PC objective (data NLL + `weight_y`·label NLL, label-free π) the
reference's? Is π solved to a MAP (not a fixed unroll) here? Does the toy gate actually
separate PC from two-stage? This oracle is the ground truth for Lessons 2–4 — if it's
wrong, everything downstream is validated against the wrong thing.

## Lesson 2 — The VI port (`spark_vi/models/topic/pc.py`, `OnlinePCLDA`)

**Read:** the module docstring (increment-1 / increment-2 framing), `initialize_global`,
`local_update`, `update_global`, `_cavi_theta_anp`, `_per_doc_sup_nll`,
`_supervised_batch_value_and_grad`. ADR 0038.

**Hughes fidelity + scalability adaptation:** the central adaptation. **Increment 1**
(`weight_y == 0`) delegates wholesale to `OnlineLDA` — the faithfulness invariant is
byte-for-byte identity with unsupervised LDA on that path (so the two-stage baseline's
topics are exactly an LDA fit). **Increment 2** reshapes the *global* topics by
differentiating the label NLL *through* the label-free per-doc CAVI — this is what makes
it prediction-*constrained*, not a two-stage classifier bolt-on. Hughes's batch fit
becomes VIRunner minibatch SVI; Hughes's NEF-MAP π becomes the bounded differentiable
`_cavi_theta_anp` unroll.

**Scrutinize:** the increment-1 equivalence (is it truly OnlineLDA, or does the head
leak?); the semi-supervised masking (`label_mask` folds `y_rowmask`); that the
differentiated π equals the *predicted* π (same `expElogbeta`); the autograd tape is
bounded per doc (no cross-doc/partition leakage).

## Lesson 3 — Gradient correctness: the digamma-Jacobian (`_grad_topics_to_lambda`)

**Read:** `_grad_topics_to_lambda` + its finite-difference test
(`test_supervised_lambda_gradient_matches_finite_difference`).

**Why it matters:** the supervised topic correction must be `∂loss/∂λ` (Dirichlet-COUNT
space), not the autograd-native `∂loss/∂expElogbeta` (topic-PROBABILITY space). The missing
chain-rule through `expElogbeta = exp(ψ(λ) − ψ(Σλ))` was a real bug: the raw gradient was
~65× too large and ~33° mis-directed, ratcheting Σλ into a runaway. The fix is the exact
closed-form digamma/trigamma Jacobian.

**Scrutinize:** re-derive the Jacobian; confirm the finite-difference test asserts BOTH
that the transformed gradient matches AND that the raw one does NOT (guards the fix from
silently regressing). Confirm the trust-region cap is now a light guard, not load-bearing.

## Lesson 4 — The head optimizer (`sgd` / `adam` / `newton`) — **TOP SCRUTINY**

**Read:** ADR 0039, insight 0065; `update_global` (the three head branches),
`_supervised_head_hessian`, the `head_optimizer`/`head_lr`/`head_newton_ridge` params;
`test_pc_lda.py::test_newton_head_*`. Optional context:
`spark-vi/tests/manual_pc_head_direction_repro.py`,
`analysis/pc/diagnostics/head_optimizer_diagnosis.py`.

**Hughes fidelity + scalability adaptation:** the newest and least-weathered piece.
Hughes co-fits η under a batch objective; under minibatch SVI a single first-order step
per iteration provably does NOT converge the coupled head (insight 0065 — sgd/adam land
~orthogonal to the batch-LR direction). The adaptation: a per-iteration **ridge-Newton
(IRLS)** step using only aggregatable sufficient statistics — `g_c = Σ_d (p−y)π_d` and
`H_c = Σ_d p(1−p)π_dπ_dᵀ`, both additive doc-sums.

**Scrutinize HARD (this is where a reviewer earns their keep):**
- **Scale-invariance claim.** The runner scales every stat by `corpus/batch`; verify
  `H⁻¹g` genuinely cancels it (both `g` and `H` scaled → solve invariant). This is the
  load-bearing reason there's no runner change and no raw θ on the driver.
- **Aggregation.** `head_hess_stat` (C×K×K) sums through the delegate's `combine_stats`
  like every dense stat — confirm it's additive and that `grad_wCK_stat` is the paired `g`.
- **Stability.** Per-minibatch Newton oscillates without damping and can spike on a
  near-singular minibatch `H_c` (observed `|w_CK|` → 26.7). `head_lr` (EMA damping) +
  `head_newton_ridge` (relative ridge) mitigate but don't eliminate — is that acceptable,
  or should there be a hard per-cell head trust-region (cf. `topic_trust`)?
- **Known limit.** The head plateaus below the ceiling when topics don't converge — is
  this correctly attributed (topic-drift, not a head bug) and is the parked fix right?
- **Default safety.** `head_optimizer='sgd'` default → prior behavior byte-for-byte;
  confirm `adam`/`newton` are opt-in and the `weight_y==0` path is untouched.

## Lesson 5 — The mllib shim (`spark_vi/mllib/topic/pc.py`): the reusable artifact

**Read:** `PCEstimator`/`PCModel`, `_build_model_and_config`, `_PC_DEFAULTS`, the param
declarations, `transform` (topicDistribution + probabilityCol), save/resume/warmStartFrom;
`test_mllib_pc_persistence.py`, `test_pc_warm_start.py`. ADR 0009 (shim pattern).

**Why this lesson stands alone:** this is the **"any engineer" surface** — a standard
MLlib-shaped `Estimator`/`Model` that anyone using `spark-vi` can pick up *independently of
charmpheno*, feed a features/label/mask DataFrame, and get a prediction-constrained topic
model with persistence. The scalability + reusability payoff of the whole port lands here.

**Scrutinize:** param round-trip through save/load (including the new
`headOptimizer`/`headNewtonRidge` and the Newton stat buffers); warm-start (fresh RM,
init-from-checkpoint) vs resume (continue decayed schedule) semantics; that the shim adds
only the (inert at `weightY=0`) head over the LDA shim; the transform's `probabilityCol`
= `sigmoid(w_CK · θ)` on the same θ as `topicDistributionCol`.

## Lesson 6 — Evaluate + diagnostics (`analysis/pc/evaluate.py`, `analysis/pc/diagnostics/`)

**Read:** `evaluate.py` (`_bundle_masked`, `_lr_proba_per_label_masked`,
`head_vs_lr_direction_cosine`, `lr_coefs_per_label`/`cosine_per_label`, `format_results_table`,
`min_label_count` masking); `diagnostics/{inspect_run.py, head_optimizer_diagnosis.py}`;
the driver's `_theta_mismatch_diagnostic` / `_anp_theta_df`. `test_evaluate.py`.

**What to check:** `pc_topics_lr` is the convergence-robust measure of *topic* quality
(a fresh LR on final topics) — the right way to judge whether supervision helped, distinct
from the co-fit head's own AUC. The localizers (`head_vs_lr_cosine`, `theta_mismatch`)
drove the entire head diagnosis and run under `--eval-only` (no refit) — verify they
measure what they claim. `min_label_count` masks sub-20-count drug cells (AoU small-cell
privacy floor) — confirm it suppresses both the AUC and the printed counts.

## Lesson 7 — Cloud driver + orchestration (`analysis/cloud/pc_antidepressant_cloud.py`, `scripts/run_experiment.py`)

**Read:** the driver main path (cohort → fused-vocab BOW → person-split → fit → transform
→ baselines → per-drug scoring), both `_run_vi_backend` / `_run_vi_backend_fullyobserved`,
`_vi_two_stage_bundle`, the argparse knob surface, warm-start phase-1/phase-2, `--eval-only`;
`build_pc_args` + defaults; `test_pc_antidepressant_driver.py`,
`test_pc_antidepressant_vi_backend.py`, `test_run_experiment_pc.py`; experiments 0072–0075.

**What to check:** the distributed two-stage baseline (SVI-consistent, OOM-safe — replaced
an in-memory collect); the fully-observed vs semi-supervised split; that every knob threads
model → shim → driver → `build_pc_args` and is recorded in `meta`/`params`; the eval-only
path (restores K + weight_y from the checkpoint, re-runs scoring + diagnostics, no refit).

---

## Suggested session shape

One bottom-up pass (Lessons 1 → 7), scrutiny concentrated on **Lesson 4 (Newton head)**
and the **fidelity ledger**. Timing argument for doing this now: the VI-PC model, the
eval/diagnostics harness, and the Newton head are about to be **reused for the Mondo
rare-disease cohorts** (still AoU data, Mondo-mapping-based identification) — validate the
reusable core before Mondo builds on it. Log findings + any shipped refactors as a dated
section atop `docs/REVIEW_LOG.md`.
