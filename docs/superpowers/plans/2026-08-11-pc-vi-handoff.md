# VI-native Prediction-Constrained topic model — session handoff (2026-08-11)

**Branch:** `claude/faithful-flat-pc` (all work committed + pushed; latest ~`f45172b`).
**Goal:** faithful Hughes-et-al. Prediction-Constrained (PC) topic-model replication
on All-of-Us OMOP, cohort `mdd_stable_treatment`, experiment **0072**.

---

## TL;DR — where we are

- **Big win (done):** found + fixed a real gradient bug in the VI-PC supervised
  topic correction (a missing digamma-Jacobian). Killed the Σλ runaway. **Hughes
  replicated on AoU.**
- **Head issue — DIAGNOSED then FIXED (2026-08-11, Runs 7–9 / exp 0072–0075):**
  1. **Diagnosis (localizer):** the joint head read 0.52 vs a batch LR's 0.615 because
     it was **TRAINED but non-CONVERGENT** — sgd/adam take ONE noisy gradient step per
     SVI iteration and never reach the logistic optimum (cos +0.09 to the LR direction,
     invariant to lr; the θ-routine mismatch is negligible, cos train-vs-scoring 0.95).
     Not θ, not optimizer choice, not lr — pure per-iteration non-convergence. The
     eval-only `head_vs_lr_cosine` / `theta_mismatch` localizers (in the driver's diag
     block, run under `--eval-only`, no refit) proved this.
  2. **Fix (shipped): `head_optimizer='newton'`** — one aggregatable ridge-Newton (IRLS)
     step per SVI iteration. g_c=Σ(p−y)π and H_c=Σp(1−p)ππ' are additive corpus-scaled
     doc-sums, so H⁻¹g is **scale-invariant** and needs no raw θ on the driver → no
     runner change, no over-partitioning issue. Result (0074/0075): head **0.52→0.60**,
     cos **0.09→0.35**; damping (`head_lr` 0.3, `head_newton_ridge` 0.05) stabilized
     `|w_CK|` (steady 6–8). Head plateaus at ~0.60 (not the 0.62 ceiling) because the
     **topics never converge in 100 supervised iters** (α/Σλ still drifting) — the head
     chases a moving target; closable with more iters / Polyak-averaging, but that
     doesn't change the AoU conclusion.
- **AoU PC conclusion (fair test, done):** with a valid head signal finally driving the
  topic correction, `pc_topics_lr` = **0.619** vs unsupervised two-stage 0.614 / codes
  0.613 — supervision helps **marginally** on AoU, as expected where the signal is
  already in the topics. **Hughes replicated; PC's topic-shaping ≈ no benefit here.**
- **NEXT — pivot to Mondo rare-disease (still AoU data, Mondo-mapping-based cohort/label
  identification):** the hidden-low-mass-signal regime where topic-shaping should give a
  real `pc_topics_lr` gain (toy: 0.56→0.91). The Newton head + the whole diagnostic
  toolchain carry over. Design the cohort/label first.

## IMMEDIATE: confirm Run 6 before concluding

The last pasted result (VI-PC head 0.522, `grad_cavi_iters` θ-fix) may be stale.
The driver log header does NOT print `grad_cavi_iters`, but the results JSON params
do. Have the user run:
```
RUN_DIR=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0072-pc-vi-stable-treatment-hughes
grep -o '"grad_cavi_iters": [0-9]*' "$RUN_DIR/pc_results.json"
```
- `50` → real Run 6: the θ-depth fix genuinely did NOT move the head → θ-depth is
  disproven as the head cause → go to the local head diagnosis.
- `20` / no match → old code/result (kicked off before `git pull`) → re-pull, re-run.

## The banked science result

On AoU `mdd_stable_treatment` (N=45,991; 34,515 train / 11,476 test; 10 Hughes drugs):

| model | macro AUC |
|---|---|
| **PC-topics + LR** (`pc_topics_lr`) | **~0.61** |
| LR-on-codes | 0.613 |
| unsupervised two-stage (topics + LR) | 0.614 |
| VI-PC *head* (the joint SVI head) | **0.52** ← the open issue |

**Reading:** a 50-topic representation captures ~all the raw-code predictive signal
(0.61 ≈ 0.613) — **Hughes replicated**. Supervision ≈ unsupervised here (0.61 ≈
0.614): PC's topic-shaping adds nothing on AoU because the predictable signal is
already in the unsupervised topics — a *data* finding, Hughes-consistent. (Toy,
hidden low-mass signal: PC 0.56→0.91 — where shaping earns its keep.)

## The gradient bug (FIXED)

The supervised topic correction subtracted `∂loss/∂expElogbeta` (topic-PROBABILITY
space) directly from `λ` (Dirichlet-COUNT space), skipping the chain rule through
`expElogbeta = exp(ψ(λ) − ψ(Σλ))`. Finite-difference vs the true `∂loss/∂λ`: the
applied gradient was **~65× too large and ~33° mis-directed** (cosine 0.84). One
defect → every symptom (trust-region only capped the 65×; a mis-directed push still
ratcheted Σλ 6e5→1.58e8; wrong direction degraded topics).

- **Fix:** `_grad_topics_to_lambda(grad_eb, lam)` in
  `spark-vi/spark_vi/models/topic/pc.py` — exact closed-form digamma-Jacobian
  (trigamma) transform. Finite-difference-EXACT (test
  `test_supervised_lambda_gradient_matches_finite_difference` in
  `spark-vi/tests/test_pc_lda.py`).
- **Consequence:** `weight_y`'s effective scale dropped ~65×, so **`weight_y` 100 →
  1000**. Now stable at ANY `weight_y` (toy sweep: Σλ pinned ~7.6e4 at wy up to
  3000; PC AUC 0.57→0.91→0.92). Runaway gone (confirmed on AoU: Σλ flat ~6e5 at
  wy=1000).

## The OPEN head issue — DIAGNOSED LOCALLY (2026-08-11); optimizer is SOUND

**Update (2026-08-11): the local diagnosis is done and it FALSIFIES the ridge
hypothesis below, plus every other optimizer/representation theory.** See
`analysis/pc/diagnostics/head_optimizer_diagnosis.py` (runs in ~40 s, numpy+sklearn,
no cluster) and the "Local head-optimizer diagnosis" section in
`docs/experiments/0072-*.md`.

The EXACT `update_global` head update (per-doc-MEAN logistic grad + ridge, RM `rho_t`)
was reproduced on synthetic θ and compared to a batch `LogisticRegression` on the same
θ. It reaches the LR ceiling (within SGD noise) under **all six** stressors:
1. **Ridge** `lambda_w` 0.001→0 (eff. 60×→0×): 0.623 vs 0.623 — AUC is scale-invariant,
   so ridge shrinks ‖w‖ but not the ranking. **The ridge hypothesis is wrong.**
2. **Budget/LR**: converged by ~iter 100 at every `head_lr_scale` — not budget-starved.
3. **Static θ mismatch** (fit on `_cavi_theta_anp` 50-unroll, score on converged
   `infer_local`): routines agree cosine 0.9988; AUC delta **+0.000**. **θ-depth is
   wrong too** — so `grad_cavi_iters` 20→50 was never going to help (consistent with
   Run 6 = 0.522).
4. **Online topic drift** (β moves during head SGD): penalty +0.006 — head recovers.
5. **Class imbalance** to 5% prevalence + no-intercept head: 0.57–0.62, no collapse.

**Reframed conclusion.** `proba_DC = sigmoid(w_CK·θ)` uses the *same* converged θ that
`pc_topics_lr` reads (verified in `PCModel._transform`). Given the head provably
reaches the LR ceiling, there is **no reproducible mechanism** for a 0.52 co-fit head
on topics that give `pc_topics_lr` 0.61. The head **optimizer is sound**; the 0.52 is
a **run-specific artifact** (stale/misconfigured fit, or a near-untrained / mis-directed
`w_CK` in that run). **Do NOT plumb `--head-l2` — it's a dead end.**

- **Toy vs AoU tell (now explained):** the toy passed (0.907) because its signal is
  strong; on AoU the *signal* is weak (LR itself only reaches 0.61), and the head
  tracks the LR ceiling wherever it is. Not a weak-signal optimizer failure.
- **Cheap next checks — on the ARTIFACTS, not another knob:** (a) `grep grad_cavi_iters
  pc_results.json`; (b) `w_CK_absmax` in the JSON/log (~0 ⇒ head never trained);
  (c) cosine(co-fit `w_CK[c]`, fresh `LR.coef_` on the same train θ) — ~0 ⇒ trained to
  the wrong direction; aligned but AUC 0.52 ⇒ a scoring/eval-wiring bug in that run.

## Run log (0072) — full history in `docs/experiments/0072-*.md`

| Run | config | VI-PC head | pc_topics_lr | note |
|---|---|---|---|---|
| 1 | cold K=25 wy=100 | 0.545 | — | buggy gradient |
| 2 | warm K=50 wy=100 | 0.521 | 0.604 | Σλ blew up 6e5→1.58e8 (bug); two-stage 0.614 |
| 3 | eval-only localize | — | 0.604 | topics mildly degraded, head the bigger gap |
| 4 | **fixed grad**, wy=1000 | 0.518 | 0.612 | Σλ FLAT — fix confirmed on AoU |
| 5 | +0.2 subsampling, 300 iter, hlr=2 | 0.522 | 0.608 | head didn't move (option B) |
| 6 | +grad_cavi_iters=50 | 0.522* | 0.615 | θ-depth fix — *confirm not stale* |

## Key decisions / concepts (so they aren't re-litigated)

- **Option A vs B for the head.** A = topics → external classifier (= JAMA 2020's
  actual method; = our `pc_topics_lr` = 0.61 = already in hand). B = converge the
  joint η head (= AISTATS 2018 + spark-vi's all-VI spirit). User chose **B**; the
  open head issue is B's remaining blocker.
- **Why the two papers differ:** AISTATS 2018 = methods paper (joint head is the
  contribution); JAMA 2020 = clinical paper (topics → LR/extra-trees, for
  comparisons + reporting). See `docs/hughes-comparison.md`.
- **The co-fit head MUST be differentiable** (its gradient shapes the topics) →
  logistic or MLP, NOT random forest (piecewise-constant, ∂f/∂π≈0). RF/extra-trees
  are fine as the *downstream* predictor on frozen topics (JAMA), not co-fit.
- **Target is ~0.60–0.65** (AISTATS antidepressant task; NOT the 0.67–0.71 that was
  wrongly attributed earlier — that number appears in neither paper).

## Config knobs added this session (all: driver argparse + build_pc_args + meta/params + tests)

- `--min-label-count` (default 20): mask drug labels with <N heldout cells (AoU
  small-cell floor + noisy AUC); suppresses counts in the printed table too.
- Distributed SVI two-stage baseline (`_vi_two_stage_bundle`, `_collect_topics_labels`)
  — replaced the in-memory `PCTopicModel(weight_y=0)` that OOM'd the driver.
- `--skip-two-stage` / `--baseline-max-iter`: skip/cap the (second SVI) two-stage.
- `--head-lr-scale` / `--weight-y-warmup-iters`: head step controls.
- `--topic-trust`: supervised topic-correction per-cell trust region (now a light
  guard, not load-bearing, post-fix).
- `--grad-cavi-iters`: differentiable CAVI unroll depth (θ-depth).
- `pc_topics_lr` diagnostic (ALWAYS-ON): PC's own supervised topics + external LR —
  the head-vs-topics localizer. Runs on a checkpoint via `--eval-only`.
- StandardScaler in the LR baselines (`_lr_proba_per_label_masked`) so lbfgs
  converges on high-dim counts.
- `--eval-only` now honors the checkpoint's K + weight_y (metadata + two-stage K).

## Files

- `spark-vi/spark_vi/models/topic/pc.py` — `OnlinePCLDA`; `_grad_topics_to_lambda`
  (THE fix); `_cavi_theta_anp` (differentiable θ, `grad_cavi_iters`); `update_global`
  (head SGD + topic correction, head ridge `lambda_w=0.001`); `local_update`.
- `spark-vi/spark_vi/mllib/topic/pc.py` — `PCEstimator`/`PCModel` params
  (`gradCaviIters`, `headLrScale`, `topicTrust`, `weightYWarmupIters`, `weightY`,
  `lambdaW`, `warmStartFrom`, save/resume).
- `analysis/cloud/pc_antidepressant_cloud.py` — the driver: `mdd_stable_treatment`
  path, all argparse knobs, `_vi_two_stage_bundle`, `_collect_topics_labels`,
  `pc_topics_lr` diagnostic, eval-only, meta/params JSON.
- `analysis/pc/evaluate.py` — `_grad_topics_to_lambda` finite-diff oracle,
  `_lr_proba_per_label_masked` (StandardScaler), `_bundle_masked` (`min_label_count`),
  `format_results_table`, `multitask_baseline_probas`.
- `scripts/run_experiment.py` — `build_pc_args` (all knob threading + defaults).
- `docs/experiments/0072-pc-vi-stable-treatment-hughes.md` — the run log (Runs 1–6).
- `docs/hughes-comparison.md` — AISTATS 2018 vs JAMA 2020 vs our two cohorts.
- `experiments/defaults/mdd_stable_treatment.yaml` — cohort defaults (subsampling
  0.2, weight_y etc. come from the 0072 frontmatter overrides).
- `charmpheno/charmpheno/omop/cohorts.py` — `mdd_stable_treatment` cohort,
  `_HUGHES_ANTIDEPRESSANTS`, `all_history_feature_events`.

## Commands

```
RUN_DIR=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0072-pc-vi-stable-treatment-hughes

# fresh fit (K/subsampling change needs a clean run dir):
cd ~/repos/CHARMPheno && git pull
rm -rf "$RUN_DIR"
cd analysis/cloud && make exp ID=72

# score a saved checkpoint WITHOUT re-fitting (fast; the pc_topics_lr localizer):
cd analysis/cloud && CHARM_DRIVER_MEMORY=12g make pc-antidepressant \
  PC_AD_ARGS="--cohort mdd_stable_treatment --backend vi --eval-only --skip-two-stage \
              --save-dir $RUN_DIR --vocab-size 5000 --min-df 20 --min-patient-count 20 \
              --age-min 18 --age-max 80 --min-days 90 --max-gap-days 395 \
              --min-history-events 2 --min-label-count 20 --out $RUN_DIR/pc_results.json"
```
`RUNS_DIR` is defined in `analysis/cloud/Makefile`. Note a `0072-multidomain-…`
sibling dir shares the number on this cluster — always use the full `-pc-vi-…` slug.

## Test envs (local)

- **argv / run_experiment** (`scripts/tests/test_run_experiment_pc.py`): `.venv-pc`
  (no pyspark). `.venv-pc/bin/python -m pytest ...`.
- **Spark** (`analysis/cloud/tests/*`, `charmpheno/tests/test_cohorts.py`,
  `analysis/pc/tests/*`): the poetry env
  `charmpheno-8OdyXQl_-py3.11`, `JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64`,
  run from `charmpheno/` via `poetry run python -m pytest ../analysis/...`.
- **spark-vi slow** (`spark-vi/tests/test_pc_lda.py`): `-m slow`, and
  `PYTHONPATH=<repo>:<repo>/spark-vi` with the poetry python. The
  `test_pc_supervised_beats_two_stage_on_heldout_auc` gate uses `weight_y=1000`
  (recalibrated post-fix).
- PDFs: `poppler-utils` installed; use `pdftotext` (pypdf/cryptography broken).
  Egress: PMC via curl (browser UA) works; medRxiv/jamanetwork blocked;
  `proceedings.mlr.press` via curl.

## Next steps

1. ~~Confirm Run 6 / local head-optimization diagnosis~~ **DONE** — optimizer proven
   sound; ridge + θ-depth falsified (`analysis/pc/diagnostics/head_optimizer_diagnosis.py`).
2. **Localize the 0.52 artifact from the SAVED run** (cheap, on artifacts — no re-fit):
   `grep grad_cavi_iters pc_results.json`; read `w_CK_absmax` (~0 ⇒ untrained head);
   cosine(co-fit `w_CK[c]`, fresh `LR.coef_` on the same train θ). This tells us
   whether it was a stale run, an untrained head, a mis-directed head, or a scoring bug.
   **Do NOT plumb `--head-l2`** — the diagnosis shows it can't be the fix.
3. **Pivot:** Mondo rare-disease space — the hidden-low-mass-signal regime where PC
   topic-shaping should beat unsupervised (unlike AoU). Design the cohort/label.
4. **Optional:** enrich the antidepressant example (richer features/models) to give
   supervision headroom.
