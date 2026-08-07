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

### Placement (revised 2026-08-07 — decided with user)

Faithful PC is optimized by **full-batch gradient descent**, not variational inference, so it does
**NOT** live in `spark-vi` core (which is a VI library: numpy/scipy only, `VIModel`/`VIRunner`
contract shaped around the distributed SVI iteration). Instead:

- **In-memory faithful PC → `analysis/pc/`** — a new **id-agnostic** subpackage (numpy/scipy only,
  no autograd/torch; hand-coded gradients via `scipy.optimize` / a small Adam). This is the
  **reference / correctness oracle**, the analogue of `fit_gated` for the gated SVI path. It is what
  Phase A validates and Phase C runs.
- **Pull-it-back-out path (future, NOT this plan):** once the reference is trusted, reimplement the
  factored **PC head + constrained objective** as a **VIModel layer over `OnlineLDA`/`GatedOnlineLDA`**
  — the distributed, VI-native *production* path — validated against this in-memory reference as the
  oracle (mirroring `fit_gated` ↔ `GatedOnlineLDA`). Keeping the head + objective factored (Task A1)
  is exactly what makes that port a re-wiring rather than a rewrite.

Consequence for the three-layer framing: the in-memory reference needs **no mllib/Spark shim**
(that layer is only for the future VI-native port). Phase B collapses to a thin driver-facing API in
`analysis/pc/`; Phase C is the AoU application on top of it.

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
- Test harness (revised): `.venv-pc/bin/python -m pytest analysis/pc/tests/ -q` (fresh-container venv;
  numpy/scipy/scikit-learn/pytest + `autograd` for the faithful reference).
- Commit trailer per the executing session's convention. Push only when the user asks.

### FAITHFUL π-inference (decided 2026-08-07, after reading the reference code)

Reading `slda_loss__autograd.py` + `calc_nef_map_pi_d_K__autograd.py`: Hughes infer each doc's π by
**generative MAP from words only** (`pi_estimation_mode='missing_y'` — label-FREE, identical at train
and test), then the label loss reshapes the **global topics** by autograd differentiating *through*
that π-inference. Our first A2 pass instead gave each doc a free label-shaped π (train-π ≠ test-π).
**Decision: the faithful reference matches Hughes** — generative-MAP π (unrolled NEF exponentiated-
gradient) + autograd through π→topics. `autograd` is added to `analysis/pc/` ONLY (isolated to the
non-VI reference; the VI core stays numpy/scipy). A1's hand-coded objective + the A2 free-π
`PCTopicModel` are **preserved** as the seed for the future VI port.

Parked VI-port fork (future, not this plan): the VI implementation faces the same choice —
**supervised-VI** (label-shaped E-step = the free-π analogue; tractable in the SVI runner, drops into
`local_update`, but reintroduces the train/test θ mismatch that makes sLDA's supervision weak) vs.
**PC-VI** (label-free E-step + constraint pushed to the globals; the real differentiator, harder in
SVI). A1's factoring (pure head fns returning `grad_Pi`) supports both attachment points.

---

## Part II — Plan (staged; validate the machine before the application)

### Phase A — Core (validate the objective on KNOWN signal first)

All Phase-A code lands in the new **`analysis/pc/`** subpackage (see Placement), numpy/scipy only,
id-agnostic. Tests under `analysis/pc/tests/`.

- [x] **Task A0:** branch off `main` (see Branch strategy). — done: `claude/faithful-flat-pc`.
- [ ] **Task A1 — PC objective + head, composably factored.** In `analysis/pc/`, implement (a) a
  supervised head `ŷ = softmax(ηᵀz̄)` with log-loss on the empirical topic frequencies `z̄` (NOT θ),
  and (b) the PC constrained objective `−log p(x) + λ·(loss(y, predict(z̄)) − ε)` — generative over
  ALL docs, prediction term over LABELED docs only (semi-supervised asymmetry). Factor the **head +
  objective as standalone functions** (pure `(params, data) → (value, grad)`), separable from any
  base model, so the future VI-native port re-wires rather than rewrites. TDD: (1) failing test that
  with λ=0 the objective's gradient/optimum reduces to plain LDA (topics match an unsupervised fit);
  (2) implement; (3) pass. Gradients hand-coded + checked against `scipy.optimize.check_grad`.
- [x] **Task A1 done** — `analysis/pc/{head,generative,objective}.py`; 67 tests; check_grad ~1e-9/8e-7; faithful invariants pinned. (commit 46e2bc6)
- [ ] **Task A2 — synthetic known-signal validation.** (Builds the minimal `PCTopicModel` fit/transform/predict_proba wrapper it needs — that IS B1's core, so B1 collapses to polish.) Generate synthetic docs with a *planted*
  label-predictive topic; assert PC recovers heldout AUC ≫ chance and beats an unsupervised-LDA→logistic
  two-stage baseline. This is the "the machine works" gate.
- [ ] **Task A3 — reference oracle.** Reproduce a result (or a small toy from the dtak repo) to confirm
  the objective matches the authors' (not naive up-weighting). Document any divergence.

### Phase B — Driver-facing API + baselines

(No mllib/Spark shim — that belongs to the future VI-native port, not the in-memory reference.)

- [ ] **Task B1 — driver-facing fit/transform API** in `analysis/pc/`: a small `PCTopicModel`-style
  class (`fit(X, y, K, lambda, eps, ...)`, `transform(X) → z̄`, `predict_proba(X)`) wrapping the
  Task-A1 objective + an optimizer (L-BFGS-B / Adam), with the label column + ε/λ as first-class args.
- [ ] **Task B2 — baselines + eval harness**: heldout per-class AUC; PC vs a logistic-regression-on-codes
  baseline and an unsupervised-LDA→logistic (two-stage) baseline — the Hughes comparison set.

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
