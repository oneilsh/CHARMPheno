# Faithful STM Generative Concentration via Exported η-Variance — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make simulated/atlas patients concentrated (a dominant phenotype + comorbidities) rather than an over-diffuse "rainbow," by exporting the fitted model's empirical per-topic document η-variance and using it — with no magic constant and no user knob — to rescale the exported unit-diagonal correlation into the generative covariance.

**Architecture:** The generative draw is θ = softmax(η), η ~ Normal(Γᵀx, Σ). ADR 0034 fits a unit-diagonal correlation Σ (variance pinned to 1 for stability), so the dashboard's draw (which reused `correlation.R` as Σ) is over-diffuse. We add an export-time distributed pass (reusing the STM per-doc E-step) that computes the empirical between-document η-variance per topic, export it as `eta_var` in `correlation.json`, and have the dashboard build the generative covariance `Σ[i][j] = R[i][j]·√(var_i·var_j)`. The topic-correlation heatmap keeps using `R` (a display choice). No re-fit — export only.

**Tech Stack:** Python 3.12 (spark_vi library + charmpheno export, pytest, PySpark), Svelte 5 + TypeScript + vitest (dashboard).

## Global Constraints

- **spark_vi stays domain-agnostic** — the η-variance is a general topic-model statistic; NO CHARM/OMOP/medical names or references in spark_vi.
- No LaTeX anywhere: Unicode Greek (η Σ Γ μ θ) + plain text; write `E(β)` not `E[β]`.
- Cite literature: Chan/Welford 1979 for the parallel variance combine; Blei & Lafferty 2007 for the logistic-normal model.
- No new dependency (numpy/scipy/vitest already present); hand-rolled front-end numerics.
- TDD throughout: failing test first, watch it fail, minimal implementation, watch it pass, commit.
- Export changes touch BOTH builders: `analysis/local/build_dashboard.py` (real) and `analysis/cloud/build_dashboard_cloud.py` (mirrored + `py_compile`/`ast.parse` parity).
- The scale is **data-driven and per-fit** — no hardcoded concentration constant, and NO user-facing "sharpness" knob (an interim slider from commit f26193f is REMOVED in Task 3).
- `eta_var` is optional on the bundle: dashboards without it fall back to unit variance (`Σ = R`), byte-identical to prior behavior.
- Payoff needs a **re-export** (not a re-fit): `make build-dashboard-exp ID=28` for population_cancer after the export code lands.
- Commit messages end with exactly: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Concurrency: the `stm` branch has a parallel arc (cohorts/experiments). These tasks touch `spark_vi/mllib/topic/stm.py`, `charmpheno/export`, `analysis/*/build_dashboard*`, and `dashboard/` — verify commit chains; `git add` only intended files (an external auto-stager may add others).

---

## Task 1: spark_vi — `corpus_eta_variance_gated` distributed pass

**Files:**
- Modify: `spark-vi/spark_vi/mllib/topic/stm.py` (add the function next to `corpus_mean_topic_proportions_gated_rdd`)
- Test: `spark-vi/tests/test_corpus_eta_variance.py` (create)

**Interfaces:**
- Consumes: `_stm_doc_inference` and the arg-prep shown in `infer_local` (`OnlineSTM`, `models/topic/stm.py:755`) — `expElogbeta = exp(digamma(lambda) - digamma(lambda.sum(axis=1,keepdims=True)))`, `allowed = partition.allowed_indices(row.groups)`, `Sigma_inv_allowed = safe_inverse(Sigma[np.ix_(allowed, allowed)])`, `reference = reference_index`. `STMDocument` fields `indices, counts, x, groups`.
- Produces:
  - `corpus_eta_variance_gated_rdd(doc_rdd, global_params, partition, *, lbfgs_max_iter=50, lbfgs_tol=1e-4, reference=None, depth=2) -> np.ndarray` (length-K per-topic variance; reference and never-allowed topics → 0).
  - `corpus_eta_variance_gated(docs, global_params, partition, *, lbfgs_max_iter=50, lbfgs_tol=1e-4, reference=None) -> np.ndarray` (numpy, in-memory list of `STMDocument`; same output; used by tests + small corpora).
  - Internal `_eta_welford_local(rows, ...)` and `_eta_welford_combine(a, b)` helpers, unit-testable without Spark.

**Status:** an implementation was drafted ad-hoc during planning (subagent). Treat its commit as this task's implementation and run the task review against it; if absent/incomplete, implement per below.

- [ ] **Step 1: Write the failing tests** (`spark-vi/tests/test_corpus_eta_variance.py`)
  - Welford combine: feed two batches of known η vectors, assert merged per-topic variance == `np.var(all, ddof=1)`.
  - numpy `corpus_eta_variance_gated`: small synthetic `STMDocument` set with a planted per-topic η spread → recovered variance in the right ballpark.
  - Gating: a foreground topic's variance is computed ONLY from its group's documents (a background-only doc must not contribute) — a fixture where excluding those docs changes the answer.
- [ ] **Step 2: Run tests, verify they fail** (`cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && python3 -m pytest spark-vi/tests/test_corpus_eta_variance.py -q`) — Expected: import/attribute error.
- [ ] **Step 3: Implement** the two functions + Welford helpers, mirroring `corpus_mean_topic_proportions_gated_rdd`'s mapPartitions/treeReduce + default-arg broadcast idiom, running `_stm_doc_inference` per doc and accumulating per-topic Welford `(n, mean, M2)` over ALLOWED topics with finite `eta_hat` only. `var_k = M2_k/(n_k-1)` for `n_k>1` else 0; reference → 0. Cite Chan/Welford 1979 in a comment; domain-agnostic docstring.
- [ ] **Step 4: Run tests, verify pass**; also `python3 -m pytest spark-vi/tests/test_mllib_stm.py -q` (mirrored pass not broken).
- [ ] **Step 5: Commit** (`spark-vi/spark_vi/mllib/topic/stm.py`, `spark-vi/tests/test_corpus_eta_variance.py`): `feat(spark-vi): corpus_eta_variance_gated — empirical per-topic document eta variance`.

---

## Task 2: charmpheno — export `eta_var` into `correlation.json` (both builders)

**Files:**
- Modify: `charmpheno/charmpheno/export/correlation.py` (`build_correlation_json` gains `eta_var`)
- Modify: `analysis/local/build_dashboard.py` (assemble the doc RDD, call `corpus_eta_variance_gated_rdd`, pass `eta_var`)
- Modify: `analysis/cloud/build_dashboard_cloud.py` (mirror the wiring; `ast.parse`/`py_compile` parity)
- Test: `charmpheno/tests/test_correlation_export.py` (extend)

**Interfaces:**
- Consumes: `corpus_eta_variance_gated_rdd` (Task 1); the STM checkpoint `result` (has `global_params`: lambda/Gamma/Sigma, the partition, reference index); the corpus BOW + covariates + groups already loaded in `build_dashboard.py` for `corpus_mean_topic_proportions_gated` / stats.
- Produces: `build_correlation_json(..., eta_var=None)` adds `"eta_var"` to the returned dict — a list of per-DISPLAY-topic variances **in the same compacted display space and order as `topic_order`/`R`** (reuse the exact reference-exclusion + compaction the reference-topic fix `7dec572` established; the reference topic is excluded from `topic_order`, and `eta_var` must align to the emitted rows). When `eta_var is None`, the key is omitted (older-path safe).

- [ ] **Step 1: Write the failing test** in `charmpheno/tests/test_correlation_export.py`: `build_correlation_json(..., eta_var=[...])` emits `"eta_var"` aligned to `topic_order` (same length, reference excluded); and omits it when `eta_var=None`.
- [ ] **Step 2: Run, verify fail** (`python3 -m pytest charmpheno/tests/test_correlation_export.py -k eta_var -q`) — KeyError.
- [ ] **Step 3: Implement** the `correlation.py` field (map the K-space `eta_var` from Task 1 into the compacted display order used for `R`/`topic_order`, excluding the reference — mirror how `R`/`support` are built). In `analysis/local/build_dashboard.py`: build the `STMDocument` RDD (join BOW + covariates + groups — reuse the same inputs used to build the gated-mean `cov_group_rdd`, adding the BOW indices/counts), call `corpus_eta_variance_gated_rdd(doc_rdd, result.global_params (or the arrays), partition, reference=..., depth=2)`, and pass the result into `build_correlation_json(...)`. If assembling the doc RDD is more than a small join, factor a helper. Mirror in `build_dashboard_cloud.py`.
- [ ] **Step 4: Run tests, verify pass**; both drivers parse: `cd analysis/cloud && python3 -c "import ast; ast.parse(open('build_dashboard_cloud.py').read()); print('ok')"` and same for local.
- [ ] **Step 5: Commit** (correlation.py, both build_dashboard*, test): `feat(export): correlation.json carries per-topic eta_var (generative concentration scale)`.

Note for the executor: run the export end-to-end only on the cluster (out of scope here) — local verification is the unit test + `ast.parse` parity, as with the other export tasks. The USER re-exports exp 0028 afterward.

---

## Task 3: dashboard — consume `eta_var` at face value; remove the interim sharpness knob

**Files:**
- Modify: `dashboard/src/lib/conditioning/logisticNormal.ts` (`buildGenerativeSigma` drop the `concentration` param; fix the misleading comment), `dashboard/src/lib/conditioning/recordPosterior.ts` (drop `concentration`), `dashboard/src/lib/store.ts` (remove `drawConcentration`), `dashboard/src/lib/tabs/Simulator.svelte` (remove the "phenotype sharpness" slider + its use)
- Test: `dashboard/src/lib/conditioning/logisticNormal.test.ts` (update)

**Interfaces:**
- Produces: `buildGenerativeSigma(correlation, freeIdx): number[][]` (no `concentration`), `Σ[a][b] = R[i][j]·s_a·s_b`, `s_k = √(correlation.eta_var ? (eta_var[order[freeIdx[k]]] ?? 1) : 1)`. Both samplers drop their `concentration` argument. Draw uses `eta_var` at face value (the data scale); unit fallback when absent.

- [ ] **Step 1: Update the test** — replace the "concentration multiplier increases dominance" test with an **`eta_var`-driven** version: `sampleConditionedTheta` with a large `eta_var` produces MORE concentrated θ (higher top-topic mass / fewer effective topics over N draws) than `eta_var` absent — a real statistical assertion (the mechanism is now the data scale, not a knob). Keep the "fallback (no eta_var) == R sub-block byte-identical" test and the existing sampler/recordPosterior tests (empty-prefix `toEqual` etc.).
- [ ] **Step 2: Run the updated test, watch the concentration-knob test fail** (arg removed) — `cd dashboard && npm run test -- src/lib/conditioning/logisticNormal.test.ts`.
- [ ] **Step 3: Implement** — drop `concentration` from `buildGenerativeSigma` (compute `s_k` from `eta_var` only), from `sampleConditionedTheta` and `sampleRecordPosterior` (and the empty-prefix delegation); remove `drawConcentration` from `store.ts`; remove the sharpness slider markup/CSS + `drawConcentration` import + the `concentration:` factory arg from `Simulator.svelte`. Fix the `buildGenerativeSigma` comment to state the mechanism correctly: scaling R up by the empirical `eta_var` raises the η variance, and softmax of higher-variance η yields MORE peaked θ (more concentrated patients) — the exported per-topic empirical document η variance sets the scale.
- [ ] **Step 4: Run** `npm run test` (full suite green), `npx svelte-check --threshold error` (baseline 4 errors/2 warnings), `npx vite build`.
- [ ] **Step 5: Commit** (the 5 files): `refactor(dashboard): generative Sigma uses exported eta_var at face value; drop interim sharpness knob`.

---

## Task 4: dashboard — Simulate Cohort → Explore Cohort shared-cohort wiring

**Files:**
- Modify: `dashboard/src/lib/cohort.ts` (`CohortConditioning` gains `prefixCounts?`/`beta?`; `drawOne` uses `sampleRecordPosterior` when a prefix is present), `dashboard/src/lib/tabs/Simulator.svelte` (write the generated cohort to the shared `cohort` store + refit the patient projection), `dashboard/src/lib/tabs/Patient.svelte` (verify it displays `$cohort` — Explore Cohort's controls were already removed)
- Test: `dashboard/src/lib/cohort.test.ts` (extend)

**Interfaces:**
- Consumes: `sampleRecordPosterior` (already committed), `simulatorConditioning`, `simulatorPrefix`, `cohort` store, `ensurePatientProjection`.
- Produces: `CohortConditioning` gains `prefixCounts?: Map<number,number>` and `beta?: number[][]`; when present + STM, `drawOne` draws θ via `sampleRecordPosterior` (else `sampleConditionedTheta`). `Simulator.svelte.simulate()` also calls `generateCohort({... conditioning: { mode:'set', values, group, bundle, prefixCounts, beta } })`, writes it to `cohort`, and calls `ensurePatientProjection()` so Explore Cohort's UMAP updates.

- [ ] **Step 1: Write the failing test** (`cohort.test.ts`): a `set`-mode cohort WITH a prefix concentrating on a topic yields higher mean mass on that topic than the no-prefix cohort (reuse/extend the gated STM fixture; non-vacuous).
- [ ] **Step 2: Run, verify fail** — `cd dashboard && npm run test -- src/lib/cohort.test.ts` (prefix ignored).
- [ ] **Step 3: Implement** the `cohort.ts` prefix path (thread `prefixCounts`/`beta`; `drawOne` picks `sampleRecordPosterior` when `prefixCounts?.size`); wire `Simulator.svelte` to generate + write the shared cohort + `ensurePatientProjection()`; confirm `Patient.svelte` renders `$cohort` (no re-added controls).
- [ ] **Step 4: Run** `npm run test`, `npx svelte-check --threshold error`, `npx vite build` — all green.
- [ ] **Step 5: Commit**: `feat(dashboard): Simulate Cohort generates the shared cohort (prefix-conditioned) that Explore Cohort displays`.

---

## Task 5 (LAST): ADR 0036 — record-completion posterior + eta_var generative covariance

**Files:**
- Create: `docs/decisions/0036-dashboard-stm-record-completion-and-generative-concentration.md`
- Modify: `docs/decisions/0035-dashboard-logistic-normal-forward-sampler.md` (cross-reference note)

- [ ] **Step 1: Write the ADR** — Status Accepted. Context: ADR 0035 shipped forward-only conditioning reusing `correlation.R` as Σ; two problems surfaced — (a) the Simulator ignored the prefix (each sample an independent prior draw); (b) `R` is a unit-diagonal correlation (ADR 0034 variance floor), so draws are over-diffuse. Decision: (1) record completion is the logistic-normal posterior over η given the prefix (Fisher scoring to the mode + backtracking line search + Laplace draw; empty prefix reduces to the prior draw); (2) the generative covariance is the exported unit-diagonal correlation rescaled by the empirical per-topic document η-variance `eta_var` (computed at export via the per-doc E-step; general, data-driven, no magic constant; no user knob) — the heatmap keeps the correlation for display; (3) a diagonal-loading PD guard makes any group Σ sub-block factorable. Consequences: coherent, prefix-driven simulated patients; Simulate Cohort and the Patient Atlas share one generation path; requires a re-export (not a re-fit) to carry `eta_var`; the Laplace covariance is a Gauss-Newton approximation (mode exact). Unicode Greek, no LaTeX; cite Blei & Lafferty 2007.
- [ ] **Step 2: Cross-reference** 0035 (top note: extended by 0036).
- [ ] **Step 3: Commit**: `docs(adr): 0036 STM record-completion posterior + eta_var generative concentration (extends 0035)`.

---

## Self-Review

**Spec coverage:** exported data-driven η-scale (Tasks 1–2); dashboard consumes it, no knob (Task 3); prefix conditioning already shipped (record-completion posterior, commits a3fdb0d/a31d409/b12ee6e/f26193f) + Simulate→Explore wiring (Task 4); ADR (Task 5); heatmap unchanged (Tasks 2–3 keep `R` for display); re-export (not re-fit) — user action, noted. ✓

**Type/interface consistency:** `eta_var` is per-DISPLAY-topic in `topic_order` compacted space in BOTH the export (Task 2) and the dashboard consumer (`buildGenerativeSigma` indexes `eta_var[order[freeIdx[k]]]`, Task 3) — must match the reference-excluded compaction from `7dec572`. `corpus_eta_variance_gated_rdd` signature identical in Tasks 1/2. `CohortConditioning.prefixCounts?/beta?` (Task 4) match `sampleRecordPosterior`'s args.

**Placeholder scan:** dashboard tasks carry concrete edits + test intent; Python export/library tasks carry exact signatures, the functions to reuse (`_stm_doc_inference`/`infer_local`/`corpus_mean_topic_proportions_gated_rdd`), and unit/`ast.parse` verification (full cluster export is the user's re-export, as with prior export tasks).

**Executor notes:** run FE from `dashboard/`; Python from repo root with the project venv. Task 1 may already be implemented ad-hoc — review it rather than re-implement. Task 3 partially reverts commit f26193f (the interim slider). The reference-topic compaction (`7dec572`) is the alignment contract for `eta_var` — get it exactly right or the dashboard indexes the wrong topics.
