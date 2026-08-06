# Patient Covariates for Gated Case-Finding (Prevalence × Prediction) — Combined Spec & Plan

**Date:** 2026-08-06
**Status:** Design + plan (proposed; not yet built). Hand-off doc for a fresh thread.
**Motivation:** insights [0026](../../insights/0026-stm-prevalence-gives-prevalence-not-content-fidelity.md)
(prevalence gives prevalence, not content), [0021](../../insights/0021-cohort-corpora-two-anchor-mass-concentration.md)
(α-optimization amplifies anchor concentration), [0064](../../insights/0064-lr-ranking-edge-yields-zero-fdr-discoveries-ranker-not-discoverer.md)
(LR-ranking is a ranker, not a discoverer).
**References:** STM covariates (Roberts, Stewart & Airoldi 2016); **DMR** — Dirichlet-multinomial
regression, Mimno & McCallum 2008 (*UAI*), the Dirichlet-side analogue of STM prevalence.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax. Do the tasks IN ORDER — the cheap axis (prediction) may answer the
> whole question before the expensive axis (DMR) is built.

---

## Part I — Spec / Design

### Goal

Add patient covariates `x_d` (age, sex, site, birth-era — demographic/nuisance adjusters, **not**
the ontology gating label) to the gated hierarchical case-finding model at **two** points, and
measure each with a 2×2:

1. **Prevalence (patient → topic):** covariate-dependent Dirichlet prior over topics, DMR-style —
   `α_{d,k} = exp(a_{node(k)} + Λ_{node(k)}ᵀ x_d)`. Reshapes *which topics a patient expresses*.
2. **Prediction (patient → decision):** covariate-adjusted per-node case-finding classifier —
   the placement score combined with `x_d` in a small trained readout. Adjusts *the decision only*.

**Explicitly NOT in scope:** content covariates (β deviations / SAGE), any logistic-normal
document prior (STM's η/Σ), and a jointly-trained probit head. See Non-goals.

### Why this is incremental, not greenfield

- **The production fit already learns a per-node α, which IS a DMR with a one-hot-node design.**
  `GatedOnlineLDA` (`gated_lda.py`) optimizes a tied per-node α via `gated_alpha_newton_step`
  (`concentration_optimization.py:177`). Prevalence covariates = **extra columns in a regression
  already run**: widen the design from node-one-hot to `[node-one-hot | x_d]`.
- **The covariate sidecar already exists.** `_covariates_load.py` builds a formulaic design matrix
  from `person_df`, cached as parquet keyed by `(formula, person_mod, cdr, cohort)`, with a
  `validate_label_not_covariate` guard. It is currently wired only to the **STM** pipeline; the
  work is rewiring it into the case-finding corpus.
- **Prediction covariates are post-fit** — they live in `dag_placement.evaluate` / the readout, no
  engine change.
- **Stays Dirichlet.** No η/Σ anywhere, so vertex-sparsity / rare-phenotype recovery is preserved
  (insight 0028). This is the whole reason to do DMR instead of STM prevalence.

### The two levers, with shapes

Toy sizes: `V=5000` codes, `N=5` non-root nodes, `n_bg=40`, `tpn=2`, `K=50`, `P=6` covariates.

| object | meaning | shape (toy) |
|---|---|---|
| `x_d` | patient covariate vector | `P = 6` |
| `Λ` | covariate → per-node prior weights (DMR) | `P × (N+1) = 6 × 6` |
| `α_{d}` | this patient's Dirichlet param over topics | `K = 50` (via `α_{d,k}=exp(a_{node(k)}+Λ_{node(k)}ᵀx_d)`) |
| `γ_l` | prediction covariate weights, node `l` | `P = 6` |

- **Prevalence (DMR):** E-step uses per-doc `α_d[allowed]` in the γ recurrence
  (`gamma = alpha + Σ_m ω_m·evidence`); M-step Newton on `Λ` (extends `gated_alpha_newton_step`).
- **Prediction:** per-node classifier on `[placement_score, x_d]`, fit post-hoc in the eval layer.

### What we reuse unchanged

- `TopicBlockPartition` / `DagLayout` / `allowed` sets — model-agnostic.
- The α-Newton machinery (`concentration_optimization.py`) — DMR is its covariate-augmented case.
- The covariate sidecar (`_covariates_load.py`, `_covariates_cache.py`) + `validate_label_not_covariate`.
- `dag_placement.evaluate` outputs (AUC/AP by depth, MRR, FDR block) as the 2×2 metrics.
- The `fit_gated` collapsed-Gibbs oracle — extend it with a DMR α to validate the SVI engine, per the
  existing oracle-vs-SVI validation pattern.

### The 2×2 and the pre-registered hypothesis

| | prediction cov OFF | prediction cov ON |
|---|---|---|
| **prevalence cov OFF** | baseline (today) | cheap: adjust decision only |
| **prevalence cov ON** | DMR: reshape topics | both |

Metric: case-finding AUC / AP / FDR from `evaluate`, by depth. **Pre-registered hypothesis**
(insight 0026): prevalence covariates reshape *which* topics fire but not their *content*, so they
may do little for case-finding, while the cheap prediction axis captures demographic confounding
directly at the decision. **If cell (OFF-prev, ON-pred) ≈ (ON-prev, ON-pred), DMR is not earning its
keep for case-finding** — a result worth ~2 days (axis 2) instead of a week (axis 1). Running it as a
2×2 (not DMR-first) is the point.

### Non-goals

- **No content covariates** (β / SAGE deviations) — β stays as fit today.
- **No logistic-normal / STM Σ** — the document prior stays Dirichlet. (This is the design's core
  constraint; it is what keeps rare-phenotype recovery intact.)
- **No jointly-trained probit head** — the post-fit covariate-adjusted classifier stands in for
  axis 2. A joint head is a *later* upgrade, only if the 2×2 shows prediction covariates matter.
- **No new covariate data engineering** — reuse the existing sidecar; do not re-derive covariates.

### Risks / watch-items

- **α-concentration interaction (insight 0021):** α-optimization already amplifies anchor
  concentration; covariate columns in α could worsen it. The oracle check (Task 4) is the guard.
- **Identifiability:** node intercept vs covariate main effects need a reference convention (drop-one
  / sum-to-zero), the estimable-functions question the identifiability compiler already handles
  (insight 0055; Searle). Pick a convention and document it.
- **Per-doc α cost:** minor — computing `α_d` per doc in the SVI E-step is a `P×(N+1)` matvec.

---

## Part II — Plan

### Global constraints

- **Engine (`spark_vi`) stays id-agnostic:** covariates enter as numpy arrays + a design matrix;
  concept-ids and covariate *names* live only at the driver edge (`analysis/cloud`).
- **Dirichlet only — no η/Σ, no logistic-normal.** Any PR that adds a Gaussian document-topic prior
  is out of scope for this plan.
- **Backward compatibility (hard requirement):** with covariates absent (or `Λ` = intercept-only,
  `x_d` all-zero), the fit must be **byte-identical** to the current `GatedOnlineLDA`, and
  `evaluate` must return every current key unchanged. Pin this with a test.
- **Oracle-validated:** the DMR SVI path is validated against an extended `fit_gated` (or a direct
  DMR-likelihood reference), same as the existing gated-LDA validation.
- **Commit trailer:** follow this repo/session's convention (`Co-Authored-By: <model> …`); the
  executing thread uses its own session's trailer. Push only when the user asks.
- **Test harness** — engine: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_gated_lda.py tests/test_dag_placement.py -q` (use `-k <name>` for focused iteration). Driver:
  `.venv/bin/python -m pytest analysis/cloud/tests/test_dag_placement_cloud.py -q`.
- Exploratory research code. Structural tests + the byte-identical/oracle checks; do not gold-plate.

### File structure

- **Modify** `analysis/cloud/dag_placement_cloud.py` — join the covariate sidecar into the corpus by
  `person_id`; thread `x_d` to the fit (Task 2) and the eval (Task 1); add `--prev-cov`/`--pred-cov`
  flags for the 2×2 (Task 5).
- **Modify** `spark-vi/spark_vi/inference/concentration_optimization.py` — DMR Newton on `Λ`
  (covariate-augmented `gated_alpha_newton_step`).
- **Modify** `spark-vi/spark_vi/models/topic/gated_lda.py` — per-doc `α_d` in the gated E-step;
  call the DMR M-step from `update_global`.
- **Modify** `spark-vi/spark_vi/models/topic/dag_placement.py` — `evaluate` gains an optional
  covariate-adjusted per-node classifier; extend `fit_gated` oracle with DMR α (Task 4).
- **Reuse** `analysis/cloud/_covariates_load.py`, `_covariates_cache.py` (+ `validate_label_not_covariate`).
- **Tests:** `spark-vi/tests/test_gated_lda.py`, `spark-vi/tests/test_dag_placement.py`,
  `analysis/cloud/tests/test_dag_placement_cloud.py`.
- **New:** `docs/experiments/00NN-covariate-prevalence-prediction-2x2.md` (claim the next free id).

---

### Task 0 — Shared plumbing: route the covariate sidecar into the case-finding corpus

Both axes need `x_d` per document; this is the common prerequisite.

- [ ] **Test first** (`analysis/cloud/tests/test_dag_placement_cloud.py`): a corpus assembled with a
  covariate formula carries a per-doc covariate vector aligned to `person_id`; `validate_label_not_covariate`
  raises when the gating node column appears in the formula.
- [ ] Run — expect FAIL.
- [ ] Implement: in `dag_placement_cloud.py`, load-or-build the sidecar via `_covariates_load`
  (same cache-key discipline as STM), left-join to the corpus on `person_id`, carry `x_d` (dense
  `P`-vector, standardized) through `transform`. Absent formula → no covariate column (baseline path).
- [ ] Run — PASS. Confirm the baseline (no formula) corpus is unchanged.

### Task 1 — Axis 2 (cheap): covariate-adjusted prediction

Post-fit only; no engine change.

- [ ] **Test first** (`test_dag_placement.py`): on synthetic data where a covariate confounds
  node membership, a per-node classifier on `[placement_score, x_d]` recovers higher adjusted AUC
  than the placement score alone; with no covariates it reproduces the current analytic metric.
- [ ] Run — expect FAIL.
- [ ] Implement: add an optional covariate-adjusted per-node readout to `evaluate` (or a sibling
  helper) — fit a per-node logistic regression on `[placement_score, x_d]`, report adjusted
  AUC/AP alongside the existing analytic numbers. Backward-compatible (no `x_d` → current behavior).
- [ ] Run — PASS.

### Task 2 — Axis 1a (DMR): per-doc α in the gated E-step

- [ ] **Test first** (`test_gated_lda.py`): with `Λ` = intercept-only (covariate weights zero), the
  E-step γ recurrence is byte-identical to the current per-node-α path.
- [ ] Run — expect FAIL (until the per-doc α path exists and reduces correctly).
- [ ] Implement: compute `α_d = exp(design(x_d) @ Λ)` restricted to `allowed`, use it in the CAVI
  γ recurrence in place of the shared `alpha[allowed]`. Initialize `Λ` so the intercept column
  equals today's learned per-node α → exact reduction when covariates are zero/absent.
- [ ] Run — PASS. Byte-identical baseline confirmed.

### Task 3 — Axis 1b (DMR): Newton M-step on Λ

- [ ] **Test first** (`test_gated_lda.py` / `test_concentration_optimization`): the DMR Newton step
  recovers a planted covariate→prevalence effect on synthetic data; with a one-hot-only design it
  matches `gated_alpha_newton_step` to tolerance.
- [ ] Run — expect FAIL.
- [ ] Implement: extend `gated_alpha_newton_step` to optimize `Λ` over the covariate-augmented design
  via the Dirichlet-multinomial-regression gradient/Hessian (digamma terms, same structure as the
  existing α Newton), with `x_d`-weighted sufficient-stat accumulation across the minibatch reduce.
  Wire it into `update_global`. Apply the chosen identifiability convention.
- [ ] Run — PASS.

### Task 4 — Oracle: DMR reference to validate the SVI path

- [ ] Extend `fit_gated` (or add a small direct-likelihood DMR reference) with a per-doc `α_d`, and
  add an SVI-vs-oracle agreement test on synthetic covariate data (mirrors the existing gated-LDA
  oracle test). This is the guard against the insight-0021 concentration interaction.
- [ ] Run — PASS.

### Task 5 — The 2×2 runner + experiment doc

- [ ] Add `--prev-cov {on,off}` and `--pred-cov {on,off}` flags to `dag_placement_cloud.py`; a thin
  sweep script runs the four cells on one cohort (reuse the corpus/bundle cache so assemble runs once).
- [ ] Create `docs/experiments/00NN-covariate-prevalence-prediction-2x2.md`: record the layout, the
  covariate formula, the pre-registered hypothesis (0026), and the four-cell AUC/AP/FDR table.
- [ ] Run the 2×2; fill the results table; write the verdict — did DMR (prevalence) add anything over
  the cheap prediction adjustment?

---

## Decision / sequencing

Task 0 → Task 1 first (shared plumbing + the cheap axis). If the (OFF-prev, ON-pred) cell already
captures the covariate signal, **stop** — DMR is not worth building for case-finding, and that is a
valid, useful negative. Build Tasks 2–4 (DMR) only if you want covariates shaping the *representation*
(prevalence), not just the decision — or for the profiling use where reshaped topics matter.
