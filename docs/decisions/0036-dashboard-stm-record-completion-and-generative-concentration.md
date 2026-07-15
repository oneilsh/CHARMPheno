# 0036 — Dashboard STM record-completion posterior + eta_var generative concentration

**Status:** Accepted
**Date:** 2026-07-03

## Context

ADR [0035](0035-dashboard-logistic-normal-forward-sampler.md) shipped the
forward-only logistic-normal sampler: η ~ Normal(Γᵀx, Σ_allowed), θ = softmax(η),
reusing the exported unit-diagonal correlation `R` (ADR
[0034](0034-stm-blockwise-unit-diagonal-correlation-sigma.md)) as Σ. It explicitly
deferred the prefix-posterior E-step (decision 4/D) and left the concentration
question open (Σ = R has variance pinned to 1 by construction, which is the
fitting parameterization, not necessarily the right generative scale). Two
problems surfaced once the forward sampler was in use:

**1. The Simulator ignored the prefix.** Starting conditions entered a record
(the prefix — a partial phenotype history the user builds up as "observed
codes") were not used to condition generation: each reported sample was an
independent draw from the covariate/group prior, so a fixed prefix produced a
different, unrelated set of top phenotypes on every "generate" click. The
prefix was consumed elsewhere for display but the reported θ never conditioned
on it.

**2. `R` is unit-diagonal, so prior draws are over-diffuse.** ADR 0034 pins
Σ_ii ≡ 1 at every M-step to kill the variance-runaway failure mode (insight
[0033](../insights/0033-gated-fullcov-variance-runaway-is-an-init-identifiability-failure.md)).
That is the right fitting parameterization, but it discards the actual scale of
η: drawing η ~ Normal(Γᵀx, R) treats every topic as if it had unit
between-document variance, regardless of how peaked or diffuse real documents
actually are on that topic. Sampled patients were correspondingly under- or
over-concentrated relative to the fitted corpus.

An interim fix for problem 2 added a user-facing "phenotype sharpness" slider
(`drawConcentration`, commit f26193f) that multiplied the rescaled variance by
a manual knob. This was tried and removed (commit 2409520): it reintroduced a
tunable magic constant with no principled default, exactly the kind of knob
the project prefers to avoid in favor of a general, data-driven quantity.

## Decision

**1. Record completion is the logistic-normal posterior over η given the
prefix, not an independent prior draw.**
`sampleRecordPosterior`
([recordPosterior.ts](../../dashboard/src/lib/conditioning/recordPosterior.ts))
treats the prefix as observed multinomial counts over topic-code pairs and
finds the mode of the true posterior

    p(η | prefix) ∝ Normal(η; Γᵀx, Σ_allowed) · Π_w  p(code w | θ(η))^{n_w}

by Fisher scoring (Newton's method under the expected-information / Gauss-Newton
curvature, accumulating the prior precision Σ⁻¹ plus the per-observation
multinomial curvature `n_w (diag(θ) − θθᵀ)` at each iterate), with a
backtracking (Armijo-style) line search on the step length so a strong
likelihood (a long prefix) cannot overshoot the mode and oscillate — the bare
Newton step is accepted at full length (α=1) whenever it improves the
objective, and halved otherwise. At the mode, the draw is a Laplace
approximation: η ~ Normal(η*, H⁻¹), H being the accumulated curvature
(prior precision + data term) at η*.

**The empty-prefix case reduces exactly to the prior draw.** With no
observations there is no likelihood term, so the mode is Γᵀx and the Laplace
covariance is Σ_allowed itself; `sampleRecordPosterior` special-cases
`prefixCounts.size === 0` by delegating directly to `sampleConditionedTheta`
(ADR 0035's forward sampler), which keeps this invariant exact by construction
rather than relying on the optimizer to converge to it. This is why cohort
generation (no prefix) and record completion (a prefix) are one code path, not
two: `sampleRecordPosterior` is a strict generalization of the prior draw.

Landed across a31d409 (posterior + Fisher scoring), b12ee6e (the backtracking
line search — the initial bare Newton step could fail to converge on strongly
informative prefixes), and a3fdb0d (the Simulator reporting fix: it had been
re-running a gating-unaware Dirichlet E-step on top of the conditioned draw,
re-diffusing an already-conditioned sample back toward the prior — the
"rainbow mess" symptom — instead of reporting the conditioned draw as-is).

**2. The generative covariance rescales the fitted correlation by the
empirical per-document η-variance, `eta_var`.**
Σ[i][j] = R[i][j] · √(eta_var_i · eta_var_j)
(`buildGenerativeSigma`,
[logisticNormal.ts](../../dashboard/src/lib/conditioning/logisticNormal.ts)).
`eta_var` is the between-document variance of the per-document posterior mode
η̂_d, computed once at **export** time (not re-fit) by running the same per-doc
Laplace E-step `infer_local`/`_stm_doc_inference` already used for inference,
then accumulating a per-topic streaming mean + M2 (Welford's algorithm; Chan,
Golub & LeVeque 1979 for the distributed tree-reduce combine) over the corpus —
`corpus_eta_variance_gated_rdd` (distributed) /
`corpus_eta_variance_gated` (in-memory), both in
[stm.py](../../spark-vi/spark_vi/mllib/topic/stm.py#L178). Gating is respected:
a foreground topic's variance reflects only the documents in its own group,
because a background-only document's allowed set excludes that topic entirely
(η = −∞ there, skipped in the accumulator); the reference topic and any topic
allowed by zero documents get variance 0.

This is a general, data-driven quantity with **no magic constant and no user
knob** — the interim "phenotype sharpness" slider (commit f26193f) was tried
and removed (commit 2409520) in favor of using the exported `eta_var` at face
value. With `eta_var` absent (older bundles, or non-STM models) `buildGenerativeSigma`
falls back to unit variance per topic, which is byte-identical to ADR 0035's
original behavior (Σ = R).

The correlation heatmap is unaffected: it keeps displaying `R` (the fitted
correlation, unit-diagonal by ADR 0034), because co-movement direction is what
the heatmap communicates, not generative scale. `eta_var` is consumed only by
the generative samplers.

**Critical contract: `eta_var` is exported positionally, aligned to the R rows
(`topic_order`), not by display topic id.** `correlation.py`'s export
([correlation.py](../../charmpheno/charmpheno/export/correlation.py)) builds
`eta_var` and `R` in the same loop pass over the same row order, with the
reference topic excluded from both exactly as `topic_order` excludes it — so
`eta_var[r]` and `R[r]` refer to the same free topic for every row index `r`.
The dashboard consumer must index `eta_var` by R-row position
(`eta_var[r]`), never by the compacted display topic id (`topic_order[r]`,
which differs from `r` whenever a reference topic or a k-anonymity gap shifts
the compaction — true on production bundles). `buildGenerativeSigma` indexes
positionally (fixed in commit a3bd4d0, which corrected an earlier
display-id-indexed version of the same function); this positional contract is
the one future consumers of `eta_var` must preserve.

**3. A diagonal-loading PD guard makes any gated group Σ sub-block
factorable.**
`choleskyPD`
([linalg.ts](../../dashboard/src/lib/conditioning/linalg.ts)) wraps the
hand-rolled Cholesky factorization (ADR 0035 decision 2) with adaptive
diagonal loading: on factorization failure it adds a geometrically growing
multiple of the identity (starting at a negligible fraction of the mean
diagonal) and retries, up to 60 attempts. A rescaled-but-still-PD Σ needs no
loading and is returned unperturbed at `t=0`; the guard exists for background ∪
group sub-blocks that can be indefinite at the boundary of the exported
correlation's support (ADR 0034 leaves cross-foreground entries at their prior
value rather than estimating them). `solveSPD`/`invSPD` (used by the Fisher
scoring in decision 1) route through the same guarded factorization.

## Alternatives considered

- **Keep the prior draw for record completion, only display the prefix.**
  Rejected: this is the status quo ADR 0036 fixes (problem 1) — sampled
  records bore no relationship to the observed prefix, which defeats the
  purpose of a "complete this record" feature.
- **A user-facing concentration/sharpness slider.** Tried (commit f26193f) and
  removed (commit 2409520). A manual multiplier is a tunable knob with no
  principled default and no data to set it from; the corpus already contains
  the answer (how concentrated real documents actually are), so exporting and
  using that value is more general and requires no user judgment call.
- **Re-fit the model to add `eta_var`.** Rejected as unnecessary: `eta_var` is
  computed from the same per-doc E-step the fitted model already supports at
  inference time, over the existing corpus and converged global parameters —
  no new M-step or re-estimation of Σ, Γ, or λ is needed, only one additional
  pass accumulating a streaming per-topic variance. This keeps the cost to an
  export-time addition rather than a fit-time one.
- **Full Newton step with no line search for the posterior mode.** Rejected
  after b12ee6e: a strongly informative prefix (many observed codes) can make
  the bare Fisher-scoring step overshoot the mode and oscillate without
  converging. Backtracking line search costs a handful of extra objective
  evaluations per iteration in the common case (well-behaved prefixes still
  take the full step almost every iteration) and guarantees monotone
  improvement in the well-behaved and pathological cases alike.
- **Exact Bayesian marginal posterior (no Laplace approximation).** Rejected
  as impractical: the true posterior over η given a multinomial-observation
  likelihood and a Gaussian prior has no closed form. The Laplace
  approximation (Gaussian at the mode, curvature from the Hessian) is the
  standard tractable choice for logistic-normal / correlated topic models
  (Blei & Lafferty 2007) and is exact for the empty-prefix case (decision 1).

## Consequences

- **Record completion is coherent.** A fixed prefix now conditions the
  generated θ toward phenotypes consistent with the observed codes, instead of
  producing an unrelated draw on every sample; the Simulator's previous
  "rainbow mess" (re-diffusion through a gating-unaware E-step, then later an
  unconverged Fisher-scoring step) is resolved by decision 1's fixes.
- **Simulate Cohort and the Patient Atlas share one generation path.** Commit
  7808108 wires the Simulator's "simulate" action to regenerate the shared
  cohort store via the same `sampleRecordPosterior`/`sampleConditionedTheta`
  path the Patient Atlas (Explore Cohort) reads, so the atlas reflects what was
  just simulated rather than a stale on-load cohort generated a different way.
- **Sampled patients are realistically concentrated, not maximally diffuse.**
  Rescaling by `eta_var` restores the corpus's actual between-document spread;
  a topic that real documents express sharply (low η-variance) generates
  sharply, and one that varies widely generates widely — general and
  data-driven, with the fitted correlation `R` still supplying co-movement
  direction.
- **Requires a re-export, not a re-fit, to carry `eta_var`.** Bundles exported
  before this change have no `eta_var` key and silently fall back to Σ = R
  (unit variance), which is exactly ADR 0035's original behavior — no
  breakage, but no concentration improvement until the bundle is re-exported.
- **The Laplace covariance is a Gauss-Newton approximation; only the mode is
  exact.** Fisher scoring converges to the true posterior mode (a stationary
  point of the exact log-posterior), but the reported covariance H⁻¹ uses the
  expected-information (Gauss-Newton) curvature rather than the exact Hessian
  of the log-likelihood term. This is the standard Laplace/IRLS treatment for
  exponential-family observation models and is exact in the empty-prefix limit
  (decision 1), where the data term vanishes and H⁻¹ = Σ_allowed exactly.
- **No new npm dependency.** The Fisher scoring, line search, and Laplace draw
  reuse the hand-rolled linear algebra (`choleskyPD`, `solveSPD`, `invSPD`)
  already introduced for ADR 0035's forward sampler.

## Addendum (2026-07-03) — generative scale is a single pooled `c`, not per-topic `eta_var`

Decision 2 above rescaled the generative covariance per-topic:
Σ[i][j] = R[i][j] · √(eta_var_i · eta_var_j), with `eta_var` the empirical
between-document variance of each topic's posterior-mode η, estimated at
export by `corpus_eta_variance_gated_rdd`/`corpus_eta_variance_gated`. Two
findings after that shipped led to a pivot.

**1. Per-topic `eta_var` came out about 10x too compressed.** The fitted
correlation R is unit-diagonal by construction (Σ_ii ≡ 1 at every M-step, ADR
[0034](0034-stm-blockwise-unit-diagonal-correlation-sigma.md)) — that
parameterization shrinks η itself during fitting, so the empirical
between-document variance measured against the fitted β/R afterward is
correspondingly compressed relative to the corpus's true generative scale —
insight [0030](../insights/0030-spectral-init-closes-stm-sigma-blowup-on-real-data.md)
measured that natural η-scale at ≈7.6 on comparable real data (a spectral-init
fit that lets Σ's diagonal breathe rather than pinning it), roughly 10x the
compressed empirical `eta_var` here. Rescaling by `eta_var` at face value
under-concentrated sampled patients.

**2. Estimating a per-topic FREE diagonal at fit time reopened the
variance-runaway failure mode.** exp 0032 tried letting each topic's diagonal
float freely during fitting (`estimate_sigma_diagonal`) to recover the natural
per-topic scale directly, rather than rescaling post hoc. A low-document
("low-ess") topic's diagonal exploded during optimization — the same max-eigenvalue
runaway insight [0033](../insights/0033-gated-fullcov-variance-runaway-is-an-init-identifiability-failure.md)
diagnosed for the full-Σ case: an under-constrained free parameter with too
little data support has no pooling to anchor it, so it is free to climb
without bound. A per-topic free diagonal at fit time reintroduces exactly that
under-constrained degree of freedom, just restricted to the diagonal instead
of the full matrix.

**Decision: the shipped generative scale is a single pooled scalar `c`,
estimated at EXPORT time with β and R FROZEN.** `corpus_eta_scale_gated`
(in-memory) / `corpus_eta_scale_gated_rdd` (distributed), both in
[stm.py](../../spark-vi/spark_vi/mllib/topic/stm.py), replace the per-topic
`eta_var` accumulation with an iterated law-of-total-variance EM: each round
runs the same per-document Laplace E-step as inference under the prior
Σ = c·R, accumulates — pooled over every free, observed topic — both terms of
Var(η) = Var_d(E[η|d]) + E_d(Var[η|d]) (the between-document Welford variance
of the posterior mode η̂, plus the mean per-document Laplace posterior
variance), and updates c toward the self-consistent value. Re-broadcasting the
prior as c·R each round lets c climb from an initial c=1 to the level at which
the E-step is self-consistent; convergence is typically 5-10 rounds.

This is runaway-safe by construction: β and R are frozen (no M-step, no
re-estimation of the fitted correlation), and c is a single number pooled
across every free observed topic, so a low-document topic's noise is averaged
against every other topic's rather than being free to climb on its own — the
exact degree of freedom that caused the finding-2 runaway is simply not
present in this estimator. It remains data-driven — no magic constant, no user
knob — because c comes from the corpus's own posterior spread, the same
principle decision 2 established for `eta_var`.

Σ_gen = c·R is emitted as `correlation.json`'s `eta_scale` field (a scalar,
distinct from the retired-for-generation `eta_var` array) and consumed by
`buildGenerativeSigma` ([logisticNormal.ts](../../dashboard/src/lib/conditioning/logisticNormal.ts)),
which now prefers `eta_scale` (s_k = √eta_scale for every free row) over the
per-topic `eta_var` fallback, over unit variance (Σ = R) when neither is
present. `eta_var` is kept in the export and the dashboard type for back-compat
only; it is no longer the generation input.

**It under-corrects modestly.** The Laplace approximation used at each E-step
carries the usual posterior-variance bias (the reported curvature is the
Gauss-Newton/expected-information Hessian, not the exact one — the same caveat
noted in decision 1's Consequences), so the converged `c` is a conservative
(slightly low) estimate of the true generative scale rather than an exact
match. This is the accepted, understood trade-off: a modest under-correction
is preferable to either the finding-1 10x compression or the finding-2
runaway.

## References

- Blei, D. M. & Lafferty, J. D. (2007). "A Correlated Topic Model of Science."
  *Annals of Applied Statistics*, 1(1), 17–35. — the logistic-normal prior this
  ADR's posterior conditions, and the standard reference for Laplace posterior
  approximation in correlated topic models.
- Chan, T. F., Golub, G. H. & LeVeque, R. J. (1979). "Updating Formulae and a
  Pairwise Algorithm for Computing Sample Variances." Technical Report
  STAN-CS-79-773, Stanford University. — the parallel/pairwise Welford
  combination `corpus_eta_variance_gated_rdd` uses to tree-reduce per-partition
  (n, mean, M2) triples.
- ADR [0035](0035-dashboard-logistic-normal-forward-sampler.md) — the forward
  logistic-normal sampler this ADR extends: the prefix-posterior E-step
  (decision 4/D there) and the concentration scale were both left open and are
  resolved here.
- ADR [0034](0034-stm-blockwise-unit-diagonal-correlation-sigma.md) — the
  unit-diagonal correlation Σ whose pinned variance motivates rescaling by
  `eta_var` for generation.
- ADR [0031](0031-stm-k1-reference-topic-parameterization.md) — the reference
  topic parameterization (η pinned to 0), which both the posterior mode search
  and `eta_var`'s accumulator respect.
- insight [0028](../insights/0028-dirichlet-vs-logistic-normal-rare-phenotype-recovery-gated-lda-is-plda.md)
  — the generative-process difference between Dirichlet and logistic-normal
  that motivates faithful logistic-normal sampling in the first place.
