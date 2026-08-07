# Faithful Flat Prediction-Constrained (PC) Topic Model — Combined Spec & Plan

**Date:** 2026-08-07
**Status:** Design + plan (proposed; not yet built). Hand-off doc for a fresh thread.
**Motivation:** the supervised-topic-model lit review (see `docs/references.md` §"Topic
hierarchies, supervision & gating" and the PC entries under "Phenotyping / EHR topic models").
PC training (Hughes, Hope, Weiner, McCoy, Perlis, Sudderth & Doshi-Velez 2017/2018) fixes sLDA's
"supervision drowned out by the word likelihood" failure by posing prediction as a *constraint*
on a semi-supervised generative model. This builds a **faithful flat PC** as a new model in our
framework, validated on known signal, then applies it to antidepressant treatment stability on
All of Us OMOP.

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development (or
> executing-plans). Steps use `- [ ]` checkboxes. Do the phases IN ORDER; the whole point of the
> sequencing is to separate "is PC implemented correctly" from "does the antidepressant signal exist."

---

## Branch strategy (decided)

**Branch off `main`** (`git fetch origin main && git checkout -b <pc-branch> origin/main`).
Rationale (dependency-checked 2026-08-07): all flat-PC deps (`OnlineLDA`, SVI engine, mllib
shim, `concentration_optimization`) are on `main`; `stm` is already merged (`main..stm = 0`).
Flat PC needs **none** of the experimental DAG/gating/multidomain work (which sits ~197 commits
diverged on `claude/hybrid-domain-reliability-review-*` and is messy in places), so inheriting it
is pure cost. Faithful PC is gradient-based, so it does **not** need PG (which lives unmerged on
`pg-stm`). Flat PC is a clean, general capability that should be mergeable back to main.

Deferred integration (NOT this plan): gated/hierarchical PC = (DAG work on the experimental branch)
+ (PG on `pg-stm`) + (this PC layer). That is why Phase 1 factors the PC layer to be **composable**
(see Non-goals / Task 1) — so the layer moves onto the gated model later instead of forking.

---

## Part I — Spec / Design

### Goal

A **faithful, flat** prediction-constrained topic model — plain LDA generative core + a supervised
prediction head + the PC constrained objective, semi-supervised — as a new VIModel through all three
layers (core → mllib shim → charmpheno/driver). Validate it on **known signal** (synthetic + the
authors' public reference), then apply it to **antidepressant treatment stability on All of Us OMOP**.

### What PC is (and what it is NOT)

- **Not a new generative model** — it is an *objective + a supervised head* on top of a topic model.
  Generative term over ALL docs; prediction constraint over LABELED docs.
- **The objective** (the faithful part — do NOT reduce it to naive sLDA up-weighting): minimize the
  generative loss `−log p(x)` **subject to** prediction loss ≤ ε, i.e. a Lagrangian
  `−log p(x) + λ·(loss(y, predict(z̄)) − ε)` where the generative term corrupts neither the labeled
  docs' likelihood (Hughes et al. show naive up-weighting misbehaves) nor the semi-supervised
  asymmetry. Unlabeled docs → generative only; labeled docs → generative + constraint.
- **The head**: `ŷ = softmax/GLM(ηᵀ z̄)` on the empirical topic frequencies `z̄` (NOT θ), one weight
  vector per class. Use a **probabilistic** loss (log-loss) so the output is a calibrated `P(y)`.
- **The dial**: ε (target prediction quality) / λ (multiplier), tuned on validation.

### Faithful scope (match Hughes 2017/2018)

- Flat LDA generative core (no gating, no hierarchy, no multi-domain-structured β).
- Semi-supervised (unlabeled + labeled docs).
- Gradient-based variational optimization (as the reference does).
- Bag-of-codewords input; per-class heldout AUC as the eval.

### Non-goals (explicit — these are the NEXT models, not this one)

- **No gating / DAG / hierarchy** — that is gated-PC, a later model that reuses this PC layer.
- **No Pólya-Gamma** — faithful PC is gradient-based; PG is the gated/hierarchical-head substrate later.
- **No precision/FDR-targeted or class-imbalanced constraint** — this build is *faithful*; the
  precision-targeted variant (the real differentiator for rare subtypes) is the *next* iteration and
  must be measured against this faithful baseline, not baked in.
- **No causal / heterogeneous-treatment-effect framing** — med choice as HTE is a separate arc
  (parked in `TOPIC_STATE_MODELING.md`).
- **Composability requirement:** even though those are out of scope, factor the **PC head + constrained
  objective as a layer separable from the flat-LDA base**, so it later attaches to `GatedOnlineLDA` /
  a DAG head without a rewrite.

### Global constraints

- Engine (`spark_vi`) stays id-agnostic; concept-ids and clinical vocabulary only at the driver edge.
- Reproduce the authors' behavior: use the public reference code
  (`github.com/dtak/prediction-constrained-topic-models`) as a **correctness oracle**, the same way
  `fit_gated` oracles the gated SVI.
- Test harness: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_pc_lda.py -q` (engine);
  driver tests under `analysis/cloud/tests/`.
- Commit trailer per the executing session's convention. Push only when the user asks.

---

## Part II — Plan (staged; validate the machine before the application)

### Phase A — Core (validate the objective on KNOWN signal first)

- [ ] **Task A0:** branch off `main` (see Branch strategy).
- [ ] **Task A1 — PC objective + head, composably factored.** Add a supervised head (`η`, log-loss
  on `z̄`) and the PC constrained objective as a layer over `OnlineLDA`. TDD: (1) failing test that
  the head + objective reduce to plain LDA when λ=0; (2) implement; (3) pass. Keep the head/objective
  separable from the base model (a mixin or wrapper), not welded to flat LDA.
- [ ] **Task A2 — synthetic known-signal validation.** Generate synthetic docs with a *planted*
  label-predictive topic; assert PC recovers heldout AUC ≫ chance and beats an unsupervised-LDA→logistic
  two-stage baseline. This is the "the machine works" gate.
- [ ] **Task A3 — reference oracle.** Reproduce a result (or a small toy from the dtak repo) to confirm
  the objective matches the authors' (not naive up-weighting). Document any divergence.

### Phase B — Shim + driver

- [ ] **Task B1 — mllib shim** (`spark_vi/mllib/topic/`): an Estimator/Model pair mirroring the LDA shim,
  exposing the label column + ε/λ.
- [ ] **Task B2 — charmpheno driver** (`analysis/cloud/`): fit + heldout-AUC eval; per-class AUC table
  vs a logistic-regression-on-codes baseline and a Gibbs-LDA→logistic baseline (the Hughes comparison set).

### Phase C — Application (only after A+B are trusted): antidepressant stability on AoU OMOP

- [ ] **Task C0 — verify/port OMOP drug-cohort primitives.** The MDD cohort + antidepressant `drug_era`
  new-user logic + 90-day-stability outcome reuse the GLP-1-comparator primitives (insight 0041) —
  confirm they are on `main` or cherry-pick them; do NOT drag in unrelated experimental code.
- [ ] **Task C1 — cohort + features.** MDD cohort (condition); 11 standard antidepressants (OMOP
  drug concepts / ATC descendants); **outcome = ≥90-day continued prescription** (drug_era continuity)
  as the effectiveness proxy; features = ICD/CPT/Rx codewords (fused vocab). Record N, positive rate.
- [ ] **Task C2 — fit + report.** Per-drug heldout AUC (PC vs logistic-regression vs Gibbs-LDA), the
  shape to reproduce being the paper's (~0.67–0.71 PC vs ~0.55–0.64 LR). Write `docs/experiments/00NN-*`.
- [ ] **Task C3 — honest readout.** State up front that a null on AoU is ambiguous *only if* Phase A
  passed (machine trusted) — then a null is a *data* finding (AoU med-completeness / cross-system
  censoring make 90-day stability noisier than the single-hospital original), not a bug.

---

## Decision / sequencing

Phase A gates everything: **do not run the antidepressant application until PC recovers planted
synthetic signal and matches the reference oracle.** That keeps a null result on AoU interpretable
(data, not code). After this faithful baseline lands, the *next* threads are the differentiators:
(1) precision/FDR-targeted + class-imbalanced constraint (the rare-subtype fix), (2) the DAG-consistent
head + gating/SAGE representation (gated-PC, reusing this PC layer + PG), (3) scale. Each is measured
against this faithful baseline.
