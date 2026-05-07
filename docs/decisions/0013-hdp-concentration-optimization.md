# ADR 0013 — Concentration-parameter optimization for OnlineHDP

**Status:** Accepted
**Date:** 2026-05-07
**Related:** ADR 0010 (LDA concentration optimization),
ADR 0011 (Online HDP v1 design),
ADR 0012 (HDP MLlib shim).

## Context

ADR 0011 shipped OnlineHDP v1 with γ, α, η held fixed at user-set values
and explicitly deferred optimization: *"the math is its own piece of work,
and ADR 0010 already templates the Newton machinery on LDA. Punt to a
follow-on ADR + spec pair after v1 lands."* ADR 0012 deferred the
matching MLlib `optimize*` flags for the same reason.

This ADR is the follow-on. γ optimization is the headline appeal of HDP
— it's how the model auto-discovers effective topic count instead of
inheriting whatever the user guessed. Without it, HDP behaves like a
heavier LDA. ADR 0010's Newton machinery for LDA's α/η is
near-templateable: the η pattern transfers directly to HDP's η, and HDP's
γ and α turn out to admit a *closed-form* M-step that's even simpler than
LDA's α-Newton.

## Decisions

### γ and α get a closed-form M-step (Beta(1, β) ELBO maximizer)

HDP's γ and α are scalar concentrations on Beta(1, β) priors over
stick-breaking factors, not Dirichlet vectors like LDA's α. The
β-dependent ELBO contribution of N independent Beta(1, β) factors is

    L(β) = N · log β + (β − 1) · S        where S = Σ E_q[log(1 − W)]

(the +log β prefactor and the −β coefficient on the log-(1−W) term are
the only β-dependent pieces of Beta(1, β)'s log-density; Γ(1) makes the
rest β-free). The derivative

    L'(β) = N/β + S

has a single root at

    β* = -N / S

S is always negative (sum of logs of values in (0, 1)); N > 0; so β* > 0
automatically. The second derivative L″(β) = −N/β² < 0 confirms β* is
the unique maximum. **No Hessian inversion, no iteration** — the M-step
is exact in one step.

Use sites:

- **γ**: N = T − 1, S = Σ_t [ψ(v_t) − ψ(u_t + v_t)] from the corpus
  Beta posteriors q(W_t) = Beta(u_t, v_t).
- **α**: N = D · (K − 1), S = Σ_d Σ_k [ψ(b_jk) − ψ(a_jk + b_jk)]
  accumulated across the (corpus-scaled) minibatch from the per-doc
  posteriors q(V_jk) = Beta(a_jk, b_jk).

ρ_t damping wraps the result so it lands in the SVI natural-gradient
framework: `β_new = (1 − ρ_t) · β_old + ρ_t · β*`. Floor at 1e-3 (same
as LDA).

Why this works for HDP and *not* for LDA's α: LDA's α prior is a
*Dirichlet*, whose log-partition function is gammaln(Σ α_k) — the sum
inside gammaln makes L(α) non-quadratic in α and forces Newton with a
diagonal-plus-rank-1 Hessian. HDP's γ and α are scalar Beta(1, β)
concentrations, whose log-partition in β is just `log β` — quadratic
enough that the M-step closes.

### η stays as scalar Newton

The HDP topic-word prior is symmetric Dirichlet(η · 1_V), same shape as
LDA's η — gammaln of a sum, no closed form. Reuse Hoffman 2010 §3.4's
scalar Newton step (the existing `eta_newton_step` helper, lifted to a
shared module — see "Helper home" below).

The α/η stat asymmetry from LDA carries over: γ and η are global stats
computable from the just-updated (u, v) and λ alone, while α requires
per-doc accumulation in `local_update`.

### Default flags: γ on, α on, η off

- `optimize_gamma = True` — γ is HDP's headline knob; closed form is
  cheap; turning it off would be the surprising choice.
- `optimize_alpha = True` — matches LDA's `optimizeDocConcentration`
  default after ADR 0010 (which itself was set to True for MLlib parity).
- `optimize_eta = False` — matches LDA's `optimizeTopicConcentration`
  default. Hoffman 2010 §3.4 notes that η is the least-stable
  concentration in SVI; opt-in for users who want it.

### Move γ, α, η into `global_params` (LDA pattern)

In v1, γ, α, η lived on `self` as instance attributes. With optimization,
the values change per iteration and need to be broadcast to the next
local_update. LDA already moved its α and η into `global_params` for
exactly this reason (post-ADR 0010); HDP follows the same pattern.

Implications:

- `initialize_global` seeds `global_params` with γ, α, η from the
  constructor.
- `local_update`, `update_global`, `compute_elbo`, `infer_local`,
  `iteration_summary` all read γ/α/η from `global_params` — `self.gamma`
  etc. are reserved as "initial values used at init time."
- The MLlib shim's trained accessors (`trainedAlpha()`,
  `trainedCorpusConcentration()`, `trainedTopicConcentration()`) now
  read from `result.global_params` so they reflect optimization.

### Helper home: `spark_vi.inference.concentration_optimization`

LDA's `_alpha_newton_step` and `_eta_newton_step` previously lived in
`spark_vi.models.lda` as private helpers. With HDP needing the η helper
verbatim and a new `beta_concentration_closed_form` for γ/α, those are
hoisted into a new shared module `spark_vi.inference.concentration_optimization`.
LDA keeps back-compat aliases (`from … import alpha_newton_step as
_alpha_newton_step`) so existing tests and probes import unchanged.

This creates the first module in `spark_vi.inference/`, intended as the
home for variational-inference primitives shared across models. Future
candidates (natural-gradient updates, ELBO term computations) will land
here too.

## Alternatives considered

- **Newton for γ and α, matching LDA's η pattern.** Equivalent in one
  iteration to the closed form (γ and α are convex unimodal scalars), but
  more code and a Hessian to maintain for no benefit. Rejected.
- **Per-stick adaptive ρ_t for the M-steps.** Hoffman 2010 derives shared
  ρ_t from the natural-gradient view; no evidence yet that γ/α need
  separate damping. Reuse λ's ρ_t (same as LDA).
- **Asymmetric per-vocab η.** ADR 0010 deferred this for LDA; same
  rationale here (MLlib doesn't surface it; SVI on per-vocab η is least
  stable; tractable but real numerical work). Deferred.
- **Keep γ, α, η on `self` instead of moving to `global_params`.** Less
  refactor churn but breaks the "single source of truth = global_params"
  invariant LDA relies on, and complicates the trained-accessor pattern.
  Rejected.
- **Skip the closed-form derivation and treat γ/α like η (scalar Newton)
  in code, just to keep the helpers uniform.** Rejected — math is a
  learning-walkthrough; ad-hoc unification would obscure that the Beta
  case really is simpler.

## Consequences

- The MLlib shim adds three Params: `optimizeDocConcentration` (default
  True), `optimizeCorpusConcentration` (default True),
  `optimizeTopicConcentration` (default False). The
  `_validate_unsupported_params` block that asserted these flags
  *don't* exist (per ADR 0012) is dropped.
- **Default behavior changes**: a freshly-constructed
  `OnlineHDPEstimator()` now optimizes γ and α automatically.
  Pre-ADR-0013 fits with default constructor args produced the same
  γ/α as the user supplied; post-ADR-0013 fits will produce γ/α that
  reflect the data.
- Trained accessors read from `result.global_params` rather than
  fit-time scalars, so post-fit introspection naturally surfaces the
  optimized values.
- The `iteration_summary` diagnostic line now includes current
  γ/α/η values so cloud-driver logs surface optimization progress
  iteration-by-iteration.
- HDP's v3 roadmap is closed for the "optimization" half of ADR 0011's
  v2 plan. Remaining v2/v3 items (lazy-λ sparse-vocab update, warmup
  ablation, held-out perplexity track) are independent.
- Departure from Wang's reference: the AISTATS 2011 paper holds γ/α
  fixed; the closed-form M-step we use here comes from the variational-EM
  stick-breaking literature (Blei & Jordan 2006 for DP mixtures, naturally
  extending to HDP's two-stick structure).

## Verification

```
cd spark-vi && make test
```

- 6 unit tests on the helpers (closed-form recovery on synthetic Beta
  draws, helper-import smoke checks, validation rejections) in
  `tests/test_concentration_optimization.py`.
- 9 unit tests on the HDP wiring (s_alpha emission, γ/α/η movement,
  floor enforcement, compute_elbo reading from global_params,
  iteration_summary surfacing) appended to `tests/test_online_hdp_unit.py`.
- 2 new integration tests (γ/α moves, η-on smoke) appended to
  `tests/test_online_hdp_integration.py`. The pre-existing 30-iter
  ELBO endpoint trend test now exercises optimization on the new
  defaults.
- 4 new shim tests (defaults, translation, accessors-when-off,
  accessors-when-on) appended to `tests/test_mllib_hdp.py`.

End-to-end on the cloud driver: re-run `make hdp-bq-smoke` after merge
and confirm `iteration_summary`'s log line shows γ evolving each
iteration; spot-check that `model.activeTopicCount(0.95)` differs from a
fixed-γ run (γ optimization typically shrinks the active count toward
the data-driven topic count).

## References

- Hoffman, Blei, Bach 2010. Online learning for LDA. NIPS. §3.4 for the
  η-Newton template.
- Blei, Ng, Jordan 2003. Latent Dirichlet Allocation. JMLR. Appendix A
  for the structured-Hessian Newton (LDA's α; not used here but quoted
  for contrast with the Beta closed form).
- Blei, Jordan 2006. Variational inference for Dirichlet process
  mixtures. Bayesian Analysis. The DP-mixture stick concentration's
  closed-form M-step inherits to HDP's γ and α directly.
- Wang, Paisley, Blei 2011. Online VI for HDP. AISTATS. Holds γ/α fixed
  in experiments; this ADR deviates by adding the closed-form M-step.
