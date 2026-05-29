# 0022 — STM (prevalence-only) supersedes per-bin α for covariate-aware topic modeling

**Status:** Accepted
**Date:** 2026-05-29
**Related:** Supersedes [2026-05-13 per-bin α covariate stub](../superpowers/specs/2026-05-13-per-bin-alpha-covariate.md); design spec [2026-05-29 STM prevalence-only design](../superpowers/specs/2026-05-29-stm-prevalence-design.md); ADR 0023 (STM inference design); ADR 0024 (formulaic in MLlib shim); ADR 0025 (covariate sidecar layout)

## Context

CHARMPheno's topic models have been covariate-free: LDA and HDP discover phenotypes from co-occurring codes, and patient-level metadata (age, sex, cohort) is used only post hoc to describe topic distributions. The natural next step is to make a covariate a first-class part of the model so that, e.g., a cancer-cohort document gets prior mass on phenotypes that are common in cancer patients *conditionally on cohort membership*, not just as an empirical observation after fit.

Two paths were on the table:

1. **Per-bin asymmetric α** (the [2026-05-13 stub](../superpowers/specs/2026-05-13-per-bin-alpha-covariate.md)). Each document carries a bin_id; per-bin Dirichlet prior α^(b) replaces the global α. Preserves Dirichlet-multinomial conjugacy and the existing Spark VI engine. Single categorical covariate at a time; categorical-only.

2. **Structural Topic Model (Roberts/Stewart/Airoldi).** Logistic-normal prior on θ_d with a regression on covariates: η_d ~ N(Γ x_d, Σ); θ_d = softmax(η_d). Supports continuous covariates, multiple covariates simultaneously, and interaction terms. Non-conjugate — the per-doc update becomes a numerical optimization.

The per-bin α stub was deliberately parked as the "cheap path that buys most of the value without leaving the existing engine." A brainstorm on 2026-05-29 revisited this in light of two facts:

- STM-prevalence with one-hot encoded categoricals **strictly subsumes per-bin α** as a special case (a one-hot bin indicator x_d makes Γ collapse to per-bin Dirichlet means).
- The actual covariate use cases on the table — age stratification, sex effects, cohort comparison, and combinations — want either continuous covariates or multiple-categoricals-at-once, neither of which per-bin α supports.

Building per-bin α first would mean wiring covariate plumbing through the doc spec, MLlib shim, persistence, and dashboard adapter — *and then doing it again* for STM. The two paths share the same plumbing surface but per-bin α covers a strict subset of STM's value.

## Decision

CHARMPheno commits to STM (prevalence-only) as the covariate path for v1. The [2026-05-13 per-bin α stub](../superpowers/specs/2026-05-13-per-bin-alpha-covariate.md) is superseded by this ADR and the [2026-05-29 design spec](../superpowers/specs/2026-05-29-stm-prevalence-design.md).

### What "prevalence-only" means

Covariates influence θ_d (per-document topic prevalence) via Γ. β (topic-word distributions) stays shared across the corpus and Dirichlet-conjugate, *unchanged from LDA*. Same-phenotype-different-content-by-covariate ("content covariates" via SAGE-style log-linear factors) is the other half of STM's full feature set; it is explicitly out of scope for v1.

### What v1 supports in the formula surface

- Categorical covariates via dummy coding (`C(col)` or auto-detected from `StringType`/`BooleanType`).
- Continuous covariates as linear terms.
- Interaction terms (`a:b`, `a*b`).
- Intercept handling (`~ 1 + ...` default; `~ 0 + ...` to drop).
- Per-row Python transforms inside `I(...)` (e.g., `I(age**2)`).
- Explicit factor reference-level control (`C(col, contr.treatment(reference="..."))`).

### What v1 explicitly rejects

- **Splines** (`bs(x, df=k)`, `ns(x, df=k)`, `cr(x, df=k)`). Data-dependent knot placement needs Spark `approxQuantile` plumbing that is not in v1's scope. Validation at fit-start raises a clear error pointing to workarounds.
- **Standardization in the formula** (`scale(x)`, `center(x)`). Same reason — data-dependent fitting that v1 doesn't plumb. Callers center/scale upstream.
- **Content covariates** (κ matrices on log β). Different inference machinery (SAGE-style sparse multinomial logistic regression for β); weak OMOP use case relative to engineering cost.

**Escape hatches** for users who want what v1 doesn't support:

- Continuous covariates with non-linear effects → bin into categoricals (`age_decile` as a string column, then `C(age_decile)`).
- Continuous covariates with spline structure → pre-compute the spline basis columns yourself (numpy `np.polynomial`, scipy `BSpline`, or formulaic outside the shim) and pass them as raw continuous covariates. The shim sees them as ordinary linear terms.

## Alternatives considered

1. **Build per-bin α as v1, defer STM to v1.x.** Rejected. Per-bin α is strictly subsumed by STM-with-categoricals; we would build the same doc spec, MLlib shim, persistence, dashboard adapter plumbing twice. The "cheaper engine" argument for per-bin α (preserves Dirichlet conjugacy) is real but localized — it saves work in `OnlineSTM`'s per-doc inner loop only, not in any of the surrounding scaffolding.

2. **Build STM-prevalence and STM-content together (full STM).** Rejected for v1. Content covariates require SAGE-style log-linear regression on β, which replaces the Dirichlet-multinomial conjugate update with a fundamentally different M-step (L1-regularized sparse multinomial regression on K × V × C surfaces). The OMOP code use case is weak — codes are already structured and we don't have a "different vocabulary by author" story like text does. Tracked as a possible v1.x or v2 follow-on if a concrete use case emerges.

3. **Skip both — keep covariate-free fit and do ad hoc post-hoc analysis.** Rejected. Post-hoc covariate analysis estimates θ_d marginally and then describes how θ_d varies with x_d, but never lets x_d *inform topic discovery itself*. A phenotype that would have been "chemo cytopenia in peds onc" can emerge under prior-aware modeling because the model puts mass on it preferentially in the relevant bin; post-hoc analysis can only observe whatever emerged under the covariate-free prior.

## Consequences

- **The [per-bin α stub](../superpowers/specs/2026-05-13-per-bin-alpha-covariate.md) is deprecated.** That document remains in `docs/superpowers/specs/` for historical reference; it gains a header note pointing here.

- **STM brings non-conjugacy.** The per-doc inner loop becomes a numerical optimization (L-BFGS for MAP + analytic Hessian for the Laplace covariance). See ADR 0023 for the inference design.

- **The MLlib shim gains a formula surface and an optional dependency on `formulaic`.** See ADR 0024.

- **The charmpheno corpus pipeline gains a covariate sidecar.** See ADR 0025.

- **v1 scope is a real constraint.** Splines and standardization are not "we forgot to implement" — they are rejected at fit-start time with an error that points to the workarounds above. The v1.x path to enabling them is mapped (Spark `approxQuantile` for knot positions, `mean`/`stddev` aggregations for standardization, formula rewrite to inject explicit values) but not implemented.

- **STM is not a strict superset of LDA at the engineering level.** LDA-shaped models continue to use `OnlineLDA`; STM is a sibling model. A run with `covariate_formula: "~ 1"` (intercept only) reduces to a one-dimensional Γ but still pays the per-doc L-BFGS cost. LDA remains the right choice for covariate-free fits.

## Open follow-ups

- A v1.x ticket for splines and standardization, with the design path noted above.
- A separate brainstorm for content covariates (full STM) if and when an OMOP use case justifies it.
- The Γ̂ visualization surface in the dashboard is its own design question (the bundle plumbing is in the [STM design spec](../superpowers/specs/2026-05-29-stm-prevalence-design.md); the visualization itself is not).
