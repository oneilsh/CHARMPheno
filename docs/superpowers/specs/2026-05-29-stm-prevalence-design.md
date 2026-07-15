# Structural Topic Model (prevalence-only) — Design

**Date:** 2026-05-29
**Status:** Brainstorm-grade design, awaiting user review.
**Scope:** Add a Structural Topic Model variant (prevalence covariates only) to the spark-vi framework and integrate it through charmpheno's corpus / experiment-tracking / dashboard pipeline. Supersedes the parked [per-bin α covariate stub](2026-05-13-per-bin-alpha-covariate.md), which STM-prevalence subsumes as a special case (one-hot bin indicators in x_d).

---

## Context

Topic models in this codebase have been covariate-free so far. LDA and HDP discover phenotypes from co-occurring codes, and any patient-level metadata (age, sex, cohort, condition history) is used only post hoc to describe topic distributions. The natural next step is to make a covariate a first-class part of the model: phenotypes that are common in patients with X covariate get higher prior mass *in those patients*, conditionally on the covariate.

Two options were on the table:
- **Per-bin asymmetric α** (the [2026-05-13 stub](2026-05-13-per-bin-alpha-covariate.md)) — preserves Dirichlet-multinomial conjugacy; categorical covariates only; one categorical at a time.
- **Structural Topic Model (Roberts/Stewart/Airoldi)** — logistic-normal prior on θ with a regression on covariates; supports continuous covariates, interactions, and multiple covariates simultaneously. Non-conjugate, so the per-doc update becomes a numerical optimization.

This spec commits to STM-prevalence-only. Reasons:

1. STM with one-hot encoded categoricals strictly subsumes per-bin α. The per-bin α design has no use case STM does not also serve.
2. The current cohort definitions and experiment plans (cancer / dementia / general; age stratification; sex effects) want continuous covariates and multiple-covariates-at-once — neither of which per-bin α supports.
3. The non-conjugacy cost is contained to the per-doc inner loop and decomposes cleanly for Spark mini-batch SVI (sufficient statistics analysis in the math section below). Scale is not blocked.

**Out of scope:** content covariates (covariate-dependent β via SAGE-style log-linear factors). Same-phenotype-different-content-by-bin is the other half of STM's full feature set; for OMOP codes the use case is weak and the engineering is heavy. Documented as a v1.x follow-on if a clinical use case emerges.

## Goals

1. **Add `OnlineSTM` to spark-vi** as a new `VIModel` consuming an `STMDocument` row type that extends `BOWDocument` with a per-doc covariate vector `x_d`. Pure-numpy contract, no DataFrame or domain concepts in the engine.
2. **Add an MLlib shim** (`spark_vi.mllib.topic.stm`) that accepts either a pre-built `covariates` Vector column *or* an R-style `covariate_formula` string. The formula path uses `formulaic` and a Spark-native categorical-level discovery mechanism (the "schema-frame" trick described below). `formulaic` becomes an optional spark-vi dependency.
3. **Add a fit driver and covariate sidecar** to charmpheno: `analysis/cloud/stm_bigquery_cloud.py` materializes patient-level covariates from BigQuery into a sidecar parquet alongside the corpus parquet; the driver joins corpus + sidecar at fit time. Sidecar over baked-in keeps the expensive corpus build (BigQuery scan + CountVectorizer fit) stable across covariate formula changes.
4. **Wire STM through the experiment-tracking wrapper** so `make next-exp`, `make exp ID=…`, `make eval-exp`, and `make build-dashboard-exp` all dispatch correctly for `model_class: stm`.
5. **Add a dashboard adapter** (`adapt_stm`) writing Γ̂ into the dashboard bundle. The Γ visualization itself is its own brainstorm and spec; this design ships the bundle-side plumbing only.

## The math (prevalence-only STM)

### Generative process

```
For each phenotype k: β_k ~ Dirichlet(η)           # unchanged from LDA, shared
For each document d:
  η_d ~ N(Γ x_d, Σ)                                # NEW: logit-θ has a Gaussian prior
  θ_d = softmax(η_d)                               # NEW
  For each token n in doc d:
    z_dn ~ Categorical(θ_d)                        # unchanged
    w_dn ~ Categorical(β_{z_dn})                   # unchanged
```

Where x_d is the doc's covariate vector (P-dim), Γ is the P×K coefficient matrix mapping covariates to logit-topic-prevalence, and Σ is a K×K residual covariance (K-diagonal in v1 — the same factorization the R `stm` package uses).

### Variational family

```
q(β_k)   = Dirichlet(λ_k)                          # unchanged from LDA
q(η_d)   = N(μ_d, ν_d)  with diagonal ν_d          # NEW
q(z_dn) = Categorical(φ_dn), collapsed             # same Lee/Seung trick as LDA
```

β stays Dirichlet-conjugate. The only non-conjugate piece is q(η_d) → softmax(η_d) coupling, contained per-doc.

### Per-doc inference (Laplace approximation, two-phase)

The per-doc ELBO objective in η_d:
```
L(η_d) = -½ (η_d - Γ x_d)ᵀ Σ⁻¹ (η_d - Γ x_d)
       + Σ_w n_dw · log( Σ_k softmax(η_d)_k · E[β_{k,w}] )
```

**Step (a) — MAP via L-BFGS.** scipy `minimize(method='L-BFGS-B')` with analytic gradient, cold-start at η_d = 0 each outer iteration. Cold-start preserves the stateless `local_update` contract (no per-doc state persisted across outer iters, which would break mini-batch sampling semantics). Typical inner-iter count: 20–30 per doc per outer iter. Per-iter gradient cost is O(K + V_d · K), comparable to one LDA CAVI iter — the per-doc cost factor vs LDA is ~2–3×, not an order of magnitude.

**Step (b) — analytic Hessian at MAP for Laplace covariance.** Once L-BFGS converges to η̂_d, evaluate the analytic Hessian once:
```
H(η̂_d) = -Σ⁻¹ - Σ_w n_dw · [ diag(p) - p pᵀ ]  · (β-projection terms)
         where p = softmax(η̂_d)
```
and set ν_d = (-H)⁻¹. This is the canonical two-step Laplace pattern (R `stm`, PyMC's Laplace, etc.). Using L-BFGS's running inverse-Hessian approximation as ν_d is the lazy path that inherits L-BFGS's quasi-Newton approximation error; the analytic Hessian at the MAP is exact and adds ~5% to per-doc cost (one extra K²-scale evaluation at the end of L-BFGS). For K=80, V_d=20 (typical): ~32K float ops per doc.

### M-step sufficient statistics

Each partition accumulates (additive across docs in the partition):

| Statistic | Shape | Purpose |
|---|---|---|
| `lambda_stats` | K × V | β SVI step (same as LDA's λ_stats today) |
| `XtX` | P × P | denominator of Γ OLS |
| `XtMu` | P × K | numerator of Γ OLS: Σ_d x_d μ_dᵀ |
| `residual_diag` | K | Σ_d (μ_d - Γ x_d)² + diag(ν_d) for Σ |
| `n_docs` | scalar | corpus scaling |
| ELBO terms | scalars | doc evidence, doc KL accumulation |

All additive. None require per-doc state to flow back to the driver — only aggregated cross-products. Combine via the default `combine_stats`.

### M-step on driver

**β** — same SVI natural-gradient step as LDA's λ path today (no change):
```
lambda_new = (1 - ρ) · lambda + ρ · (eta + expElogbeta * lambda_stats)
```

**Γ** — closed-form ridge regression on aggregated cross-products, blended with ρ in mini-batch mode (stochastic-EM):
```
Γ̂_target = (XᵀX + λ_ridge I)⁻¹ XᵀMu        # P×P solve on driver
Γ_new    = (1 - ρ) · Γ + ρ · Γ̂_target
```
Default `λ_ridge = 1e-6` for numerical stability against zero-variance or collinear columns. Tunable.

**Σ** — diagonal sample covariance of residuals + Laplace variance correction, ρ-blended:
```
σ²_k_target = (1 / n_docs) · residual_diag_k
σ²_k_new   = (1 - ρ) · σ²_k + ρ · σ²_k_target
```

### ELBO

```
ELBO = doc_loglik_sum                              # accumulated in local_update
     - doc_eta_kl_sum                              # KL(N(μ_d, ν_d) || N(Γ x_d, Σ))
     - sum_k KL(Dirichlet(λ_k) || Dirichlet(η))    # global β KL, on driver
```

The doc-η KL closes in form (Gaussian-Gaussian) — computed in `local_update`. Same accumulation pattern as LDA's per-doc Dirichlet KL today.

### Mini-batch convergence note

Mini-batch SVI under Robbins-Monro converges to a *neighborhood* of the full-batch optimum, not to the same point. This is the same posture as LDA today — we accept that mini-batch λ does not match full-batch λ identically. The relevant validation criterion (phase 2 of implementation, below) is **qualitative agreement**, not identity:

- ELBO at convergence within ~1% of full-batch.
- β: top-N tokens per topic substantially overlap; high topic-level cosine similarity.
- Γ̂: signs agree with full-batch; magnitudes within a small constant factor.

A failure mode that would block mini-batch shipping: Γ̂ flips signs vs full-batch on a small validation corpus, or β topics fragment unrecognizably.

## Engineering

### Spark-vi: pure-numpy engine (models layer)

**New row type — `spark_vi.models.topic.types.STMDocument`:**

```python
@dataclass(frozen=True, slots=True)
class STMDocument:
    indices: np.ndarray   # sorted int32, len = n_unique
    counts: np.ndarray    # float64, len = n_unique
    length: int           # total tokens
    x: np.ndarray         # float64, shape (P,)  — NEW
```

Lives alongside `BOWDocument`; same module. The engine never learns what `x` means — only its shape and dtype. No `person_id`, no `doc_id` (the BOWDocument doc explicitly carves these out as caller-domain concerns; STM inherits the same boundary).

**New model — `spark_vi.models.topic.stm.OnlineSTM(VIModel)`:**

Constructor signature:
```python
OnlineSTM(
    K: int,
    vocab_size: int,
    P: int,                          # NEW: covariate dimension
    eta: float | None = None,
    sigma_init: float = 1.0,         # NEW: initial diagonal Σ entries
    sigma_ridge: float = 1e-6,       # NEW: Γ regression ridge
    lbfgs_max_iter: int = 50,        # NEW: per-doc L-BFGS inner iters
    lbfgs_tol: float = 1e-4,         # NEW: per-doc L-BFGS convergence tol
    optimize_eta: bool = False,
    random_seed: int | None = None,
)
```

Global params dictionary:
```
{
    "lambda": (K, V) ndarray,        # same as LDA — Dirichlet for β
    "eta":    0-d ndarray,           # same as LDA — symmetric Dirichlet prior on β
    "Gamma":  (P, K) ndarray,        # NEW: covariate → logit-θ mean regression
    "Sigma":  (K,) ndarray,          # NEW: K-diagonal residual covariance
}
```

`local_update` runs per-doc L-BFGS + analytic Hessian, accumulates the sufficient stats listed above. `update_global` applies SVI step on λ, ridge-regression-then-ρ-blend on Γ, sample-cov-then-ρ-blend on Σ. `compute_elbo`, `infer_local`, `iteration_summary`, `get_metadata`, `iteration_diagnostics` follow the same patterns as `OnlineLDA`.

### Spark-vi: MLlib shim (DataFrame-aware layer)

**`spark_vi.mllib.topic.stm`** — DataFrame-aware estimator. Two input paths:

```python
# Path A: caller pre-builds covariates
StreamingSTM(
    K=80,
    features_col="features",         # BOW vector column, as today for LDA
    covariates_col="covariates",     # DenseVector covariate column
    covariate_names=[...],           # list of length P, for metadata
)

# Path B: caller supplies formula + covariate DataFrame
StreamingSTM(
    K=80,
    features_col="features",
    covariate_formula="~ age + sex + cohort + sex:cohort",
    covariate_df=patient_covariates_df,   # joined to BOW by an explicit key
    join_key="person_id",
    quantile_error=0.01,             # reserved for v1.x splines
    max_levels=10_000,               # cardinality bound per categorical
)
```

Path B implementation — the **schema-frame discovery pattern**:

1. Parse the formula via `formulaic.Formula(covariate_formula)`. Walk the term tree to identify categorical columns (explicit `C(col)` *or* columns with `StringType` / `BooleanType` Spark dtype).
2. For each categorical, bound cardinality via `approxCountDistinct`; if over `max_levels`, raise. Otherwise materialize the level set via `df.select(col).distinct().collect()`, sort lexicographically for determinism.
3. Build a small pandas "schema-frame" containing each level of each categorical at least once, plus placeholder rows for continuous columns (value 0.0). For 5 categoricals × 10 levels each, ~10 rows.
4. Call `Formula(covariate_formula).get_model_matrix(schema_df).model_spec`. formulaic captures the level set in `transform_state`; the resulting `ModelSpec` is data-independent at application time.
5. Validate: assert no stateful transforms beyond what we set up (no `bs`, `ns`, `cr`, `scale`, `center`). If `model_spec.transform_state` contains anything we did not pre-seed with our discovered levels, raise with a v1-scope error message pointing to workarounds (bin continuous covariates categorically, or pre-compute spline basis columns).
6. Broadcast the `ModelSpec`. In each partition's mapPartitions block, materialize the per-row design matrix via `model_spec.get_model_matrix(partition_df)` and construct `STMDocument` rows with `x = row_of_design_matrix`.
7. Persist the fitted `ModelSpec` plus covariate names alongside the `VIResult` so transform-time inputs apply the same encoding.

`formulaic` is added as an **optional** spark-vi dependency (declared in `[project.optional-dependencies]`, group name `formula`). Path A imposes no new dependency.

### Charmpheno: covariate sidecar + driver

**Covariate sidecar parquet.** Lives next to the corpus parquet under the same run directory. Schema: `(person_id, x as DenseVector)`. Built by a new helper `charmpheno.omop.covariates.build_patient_covariate_sidecar(spark, person_df, covariate_formula, out_path)` that:

1. Selects the columns referenced by the formula from `person_df`.
2. Hands the (column subset, formula) to the MLlib shim's formula-fitting helper (which runs the schema-frame discovery and produces a `ModelSpec`).
3. Applies the `ModelSpec` to produce per-person x vectors.
4. Writes `(person_id, x)` to parquet.
5. Returns the `ModelSpec` to be persisted alongside.

**Corpus manifest.** Gains a `covariate_sidecar` field pointing to the sidecar parquet (or `null` for LDA / HDP runs). Backward-compatible.

**Fit driver — `analysis/cloud/stm_bigquery_cloud.py`.** Mirrors the existing `lda_bigquery_cloud.py` structure:

1. Load corpus parquet via the shared `_corpus_load` module.
2. Load covariate sidecar parquet.
3. Broadcast-join sidecar to corpus by `person_id`. Result: BOW DataFrame with an added `covariates` DenseVector column.
4. Construct `StreamingSTM` via Path A (covariates pre-built; covariate names from the sidecar's persisted `ModelSpec`).
5. Fit via the standard VIRunner.
6. Persist VIResult, fit log, and the `ModelSpec` (for transform-time round-trip).

The formula is consumed at *sidecar-build* time, not fit time. The driver itself sees pre-built covariates, which keeps the formula-handling code in one place.

### Experiment tracking wrapper integration

`scripts/run_experiment.py` gains STM dispatch:

- `model_class: stm` recognized alongside `lda`, `hdp`.
- Defaults file `experiments/defaults/_base.yaml` gains STM-relevant keys with sensible defaults: `covariate_formula: "~ 1"` (intercept only — degenerates to LDA-like behavior), `sigma_ridge: 1e-6`, `sigma_init: 1.0`, `lbfgs_max_iter: 50`.
- `build_lda_args` becomes `build_fit_args` with model-class branching; the cloud-driver dispatch picks `lda_bigquery_cloud.py` vs `stm_bigquery_cloud.py` vs `hdp_bigquery_cloud.py` based on `model_class`.
- `make next-exp`, `make exp ID=…`, `make eval-exp`, `make build-dashboard-exp` continue to work uniformly via the wrapper's model-class branching. (The eval and build phases are agnostic to model class once the artifacts are in standard VIResult shape.)

A separate covariate-sidecar build step is needed before fit; this can either be a new Make target (`make build-covariates EXP=…`) or folded into the fit driver's startup (auto-build sidecar if missing). Recommend the latter for ergonomics — the user shouldn't have to remember a separate build step.

### Dashboard adapter

`charmpheno/charmpheno/export/dashboard.py` gains `adapt_stm(vi_result, model_spec, ...)` per the existing extension-point pattern documented in [2026-05-13-dashboard-design.md:57](2026-05-13-dashboard-design.md). The adapter:

1. Surfaces β, α-equivalent (the marginal prior from Γ̂ averaged across the corpus's empirical covariate distribution), and Γ̂ itself.
2. Round-trips the `ModelSpec` so client-side rendering can label Γ rows by covariate term names.
3. Includes covariate-row metadata: each row in Γ corresponds to a formula-expanded term (e.g., `sex[T.male]`, `age`, `sex[T.male]:cohort[T.cancer]`).

The Γ̂ *visualization* — what the dashboard does with the per-topic per-covariate effect matrix — is a separate brainstorm and spec. This design ships the bundle-side plumbing only. A reasonable v1 default is a per-topic table showing top-K covariate effects with confidence intervals; richer visualizations (heatmap, partial dependence) follow.

## Implementation phasing

1. **Full-batch `OnlineSTM` in spark-vi (engine).** Reference impl. Pure-numpy contract. Validate against R `stm` package on a small synthetic corpus (synthetic data with known Γ, recover Γ̂ within tolerance).
2. **Mini-batch STM.** ρ-blending on Γ and Σ. Validate qualitative agreement with full-batch on the same synthetic corpus (criterion above: ELBO within 1%, β top-N overlap, Γ̂ signs agree).
3. **MLlib shim with formulaic.** Path A first (pre-built covariates), then Path B (formula + schema-frame discovery + validation rejecting stateful transforms). Sharp-corner tests.
4. **Charmpheno covariate sidecar.** `build_patient_covariate_sidecar` helper, manifest field, BigQuery materialization in the driver.
5. **Charmpheno fit driver + experiment-tracking integration.** `stm_bigquery_cloud.py`, `run_experiment.py` STM dispatch, defaults YAML keys.
6. **Dashboard adapter.** `adapt_stm` writing Γ̂ + ModelSpec into the bundle. (Γ visualization design separate.)

Each phase produces a working, testable artifact. Phases 1–2 are the spark-vi work and can ship as a spark-vi release independent of charmpheno consumption. Phases 3–6 are the charmpheno integration.

## Sharp corners (user-facing pitfalls)

To be documented in the MLlib shim's docstring and in a "Pitfalls" subsection of the engine README:

1. **Numeric-without-`C()` is treated as continuous.** A column of integer cohort IDs (`cohort_id = 1, 2, 3`) used in a formula as bare `cohort_id` is treated as a continuous slope. Wrap categoricals encoded as ints with `C(cohort_id)` explicitly. The shim warns when a low-cardinality int column appears bare in the formula.
2. **Default reference level is alphabetic.** For semantic asymmetry (control vs treatment), use `C(col, contr.treatment(reference="control"))` explicitly.
3. **Unseen levels at transform time error out.** Correct behavior — we cannot extrapolate to unknown levels. The shim wraps formulaic's error with a clear message pointing to the unseen value.
4. **Per-patient covariates replicate per-doc.** A patient with 100 docs contributes 100× to Γ̂ vs a patient with 1 doc. Statistically correct under the iid-doc model, but surprising. Documented; no patient-weighting in v1.
5. **Numerical scale of x matters.** Per-doc L-BFGS struggles with `Γ x_d` blowing up. Center / scale unbounded continuous covariates upstream.
6. **Zero-variance / collinear columns.** Default `sigma_ridge = 1e-6` handles. Configurable.
7. **NaN in covariates.** Fail loudly at fit start: "covariate `age` has N null values; STM requires non-null covariates." No silent doc-dropping.
8. **Interaction explosion.** Warn if P > 100 after formula expansion. Configurable via shim param.

## Deferred (v1.x and beyond)

Tracked here so each can stand alone as its own work:

1. **Splines and standardization.** `bs(x)`, `ns(x)`, `cr(x)`, `scale(x)`, `center(x)`. Implementation path is clear (Spark `approxQuantile` for knot positions, `mean`/`stddev` aggregations for standardization, formula rewrite to inject explicit knots/centers); v1 rejects them with a clear error.
2. **Content covariates (full STM).** Covariate-dependent β via SAGE-style log-linear. Largest single piece of remaining STM functionality.
3. **Patient-weighted Γ.** Downweight docs of high-doc-count patients so each patient contributes equally. Changes the model; requires its own ADR.
4. **Per-doc L-BFGS warm-start.** Persist (μ_d, ν_d) per doc across outer iters. Would change the stateless `local_update` contract; relevant only if per-doc Laplace cost becomes a wall-clock blocker on real corpora.
5. **Newton's method for MAP (instead of L-BFGS).** Quadratic convergence, fewer inner iters, but per-iter K³ solve dominates at K ≳ 50. Not worth pursuing unless L-BFGS proves inadequate.
6. **Full-rank Σ (vs K-diagonal).** Captures topic correlations in the prior. Larger sufficient stats (K² instead of K), real but bounded engineering. Defer until empirical Σ-diagonal residuals show off-diagonal structure worth capturing.
7. **Γ visualization in dashboard.** This spec ships the bundle plumbing; the visualization design is its own brainstorm.
8. **HDP-STM cross.** STM machinery on top of HDP's GEM-stick base distribution. Possible but not designed; defer.

## Open risks

1. **Mini-batch vs full-batch qualitative agreement.** The phase-2 validation is the genuine risk. If Γ̂ signs flip or β topics fragment, we ship full-batch-only STM and document the cause in an ADR. Full-batch STM is still useful for cohort-sized corpora (one cohort, ≲1M patients fits in cluster memory).
2. **Per-doc L-BFGS cost factor in practice.** Claimed ~2-3× LDA's CAVI per outer iter, based on per-iter complexity arguments. Phase 1 will measure on real corpora; if the constant factor is materially larger (say 10×) we revisit the warm-start question.
3. **Γ as a useful dashboard surface.** STM's clinical interpretability story rests on Γ̂ being legibly visualizable. We have no design for that yet; a v1 launch with no Γ visualization is a hollow product. The Γ-viz brainstorm should happen in parallel with phase 1–2 engine work, not after.

## References

- Roberts, Stewart, Tingley 2014. "Structural Topic Models for Open-Ended Survey Responses." *American Journal of Political Science*.
- Roberts, Stewart, Airoldi 2016. "A Model of Text for Experimentation in the Social Sciences." *JASA*. Authoritative algorithmic derivation, including the per-doc Laplace + closed-form M-step structure.
- Roberts, Stewart, Tingley. `stm` R package. CRAN. Reference single-process implementation; phase 1 of this work targets agreement with it on synthetic data.
- Hoffman, Blei, Bach 2010. "Online learning for LDA." NIPS. The natural-gradient SVI step the β update reuses.
- Hoffman, Blei, Wang, Paisley 2013. "Stochastic Variational Inference." *JMLR*. Frames the LDA α update (closed-form-target ρ-blended) as part of SVI — the same pattern STM's Γ, Σ updates inherit.
- Cappé, Moulines 2009. "On-Line Expectation-Maximization Algorithm for Latent Data Models." *JRSS-B*. Convergence theory for the stochastic-EM smoothing pattern applied here to Γ and Σ.
- Lee, Seung 2001. "Algorithms for non-negative matrix factorization." NIPS. The per-doc φ trick reused unchanged from `OnlineLDA`.
- [2026-05-13 per-bin α covariate stub](2026-05-13-per-bin-alpha-covariate.md). Predecessor design; STM-prevalence subsumes it.
- [2026-05-13 dashboard design](2026-05-13-dashboard-design.md). Defines the `adapt_<class>` extension point this spec plugs into.
- [`spark_vi.core.model.VIModel`](../../spark-vi/spark_vi/core/model.py). Contract `OnlineSTM` implements.
- [`spark_vi.models.topic.lda.OnlineLDA`](../../spark-vi/spark_vi/models/topic/lda.py). The reference for what `OnlineSTM` mirrors structurally (CAVI → L-BFGS, single α → Γ, M-step natural gradient on λ unchanged).
