# VI-native Prediction-Constrained topic model — session handoff (2026-08-11)

**Branch:** `claude/faithful-flat-pc` (all work committed + pushed; latest ~`f45172b`).
**Goal:** faithful Hughes-et-al. Prediction-Constrained (PC) topic-model replication
on All-of-Us OMOP, cohort `mdd_stable_treatment`, experiment **0072**.

---

## TL;DR — where we are

- **Big win (done):** found + fixed a real gradient bug in the VI-PC supervised
  topic correction (a missing digamma-Jacobian). Killed the Σλ runaway. **Hughes
  replicated on AoU.**
- **Open thread (not fixed):** the *joint SVI head* under-performs a batch LR on
  the *same* topics (0.52 vs 0.615), and is invariant to every knob tried — a
  head-**optimization** issue (best guess: ridge/gradient-scaling), needing a
  **local** diagnosis, not another 3h cluster run.
- **Immediate pending action:** confirm whether the last result (Run 6) is really
  the `grad_cavi_iters=50` run (see below).
- **Pivot the user wants:** move toward the **Mondo rare-disease** space, where PC's
  topic-shaping should actually help (hidden low-mass signal — the regime where the
  toy showed PC 0.56→0.91, unlike AoU where the signal is already captured).

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

## The OPEN head issue (NOT fixed) — for next session

VI-PC head = **0.52**, provably invariant to: `max_iter` (200/300), `subsampling`
(0.05/0.2), `head_lr_scale` (1/2/3), and `grad_cavi_iters` (20/50, pending Run-6
confirm). Batch LR on the *same* θ = 0.615. So it is **not** representation, **not**
convergence, **not** θ-depth.

- **Toy vs AoU tell:** the head hit 0.907 on the toy (strong signal) but 0.52 on
  AoU (weak signal). Same head machinery. → an optimization issue that erases *weak*
  signal while preserving strong.
- **Best hypothesis (UNCONFIRMED — verify locally, no cluster):** the head uses a
  per-doc-MEAN data gradient (`grad_wCK * inv_n`) against a FIXED ridge
  (`lambda_w=0.001`, `update_global` in `pc.py` ~line 573). sklearn LR balances the
  SUM of per-doc losses against its ridge, so our head may over-regularize by ~the
  corpus factor → shrinks away weak signal. **Diagnosis plan:** reproduce the head
  fit on synthetic weak-signal θ (numpy, like `_cavi_theta_anp` + a manual SGD head
  loop, or drive `OnlinePCLDA` on a small local Spark), compare to a batch
  `LogisticRegression` on the same θ, and sweep `lambda_w` / the data-gradient
  scaling. If lowering `lambda_w` (or corpus-scaling the head data gradient) closes
  0.52→0.61, that's the fix. Then plumb `--head-l2` (lambdaW) like the other knobs.
- **Ruled out:** topic degradation (topics are good, `pc_topics_lr` 0.61); θ-depth
  (Run 6, pending confirm).

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

1. **Confirm Run 6** (grep `grad_cavi_iters` in the JSON).
2. **If θ-depth disproven:** local diagnosis of the head-optimization gap
   (`lambda_w` ridge / per-doc-mean vs corpus-sum data gradient) on synthetic
   weak-signal θ. No cluster. Plumb `--head-l2` if that's the lever.
3. **Pivot:** Mondo rare-disease space — the hidden-low-mass-signal regime where PC
   topic-shaping should beat unsupervised (unlike AoU). Design the cohort/label.
4. **Optional:** enrich the antidepressant example (richer features/models) to give
   supervision headroom.
