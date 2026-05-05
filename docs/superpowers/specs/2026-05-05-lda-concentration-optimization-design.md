# LDA Concentration-Parameter Optimization (v1)

## Context

The vanilla-LDA shim at [`spark-vi/spark_vi/mllib/lda.py`](../../../spark-vi/spark_vi/mllib/lda.py) currently treats the Dirichlet concentration parameters α (`docConcentration`) and η (`topicConcentration`) as static hyperparameters. The Estimator already exposes `optimizeDocConcentration` as a Param but rejects it when set to `True` (see [`_validate_unsupported_params`](../../../spark-vi/spark_vi/mllib/lda.py#L46-L78)), and there is no equivalent toggle for η. This was a deliberate v0 punt — we wanted to ship the shim before tackling Newton-Raphson updates (recorded in [ADR 0009](../../decisions/0009-mllib-shim.md)).

We are revisiting now because the next model on the roadmap is **online HDP** (Wang, Paisley, Blei 2011), which has *two* concentration parameters (γ for the corpus stick, α for the doc stick) that are part of the model's core appeal — the whole point of HDP is to let the data choose the topic count via γ. Implementing concentration optimization on LDA first is cheaper than going straight to HDP because:

1. The LDA math is well-trodden and verifiable against MLlib's reference implementation (`OnlineLDAOptimizer.updateAlpha`).
2. We already have an ELBO trend test ([`test_vanilla_lda_elbo_smoothed_trend_is_non_decreasing`](../../../spark-vi/tests/test_lda_integration.py#L70-L98)) that catches sign errors in any new ELBO-driven update.
3. The two HDP concentration updates (γ scalar, α scalar) reuse the same Newton machinery we are building here. Building this on LDA is therefore both an immediate win (LDA users get α optimization) and a de-risking step for HDP.

**Intended outcome:** `optimizeDocConcentration=True` works end-to-end (and is the new default, matching MLlib); `optimizeTopicConcentration=True` works (default off); ELBO trend test still passes; new recovery tests confirm α and η move toward known synthetic ground truth on small corpora.

## Scope

### In v1

- **Asymmetric α optimization** (vector, length K) via Blei et al 2003 §5.4 Newton-Raphson. Hessian has a diagonal-plus-rank-1 structure that admits a closed-form Sherman-Morrison Newton step in O(K). Inputs to the Estimator may now be either scalar (broadcast to symmetric vector) or length-K vector.
- **Symmetric scalar η optimization** via the analogous scalar Newton update (Hoffman et al 2010 §3.4).
- **Damping**: both updates reuse the existing Robbins-Monro `ρ_t = (τ₀ + t + 1)^-κ` already controlling λ. Hoffman 2010 §3.3 derives this from the natural-gradient view (α and λ are both global parameters with the same step). No new learning-rate Params.
- **Default flips**: `optimizeDocConcentration=True` (MLlib parity, drops our v0 divergence); `optimizeTopicConcentration=False` (conservative; MLlib has no equivalent so there is no parity target).
- **Numerical floor**: clip α and η component-wise to `[1e-3, ∞)` after each Newton step to keep digamma/trigamma finite if the step overshoots.

### Not in v1 (explicit)

- **Asymmetric η** (per-vocabulary vector η, length V). Length-V Newton has tractable but real numerical work; MLlib does not do this; mini-batch SVI is least stable on η. Revisit when a use case appears.
- **Per-iteration warmup** (skipping early iters before optimizing). Hoffman 2010, MLlib, and gensim all start at iter 0; matching precedent.
- **Custom learning rate for concentrations.** We reuse λ's `ρ_t`. Add a separate schedule only if practice demands it.
- **MLWritable persistence of optimized α/η values.** The shim does not yet implement `MLWritable` (an ADR 0009 v1 punt). Once that lands, the fitted α vector and η scalar must round-trip — but that is a separate piece of work.

## Math

### Asymmetric α (length K)

ELBO terms involving α (the part that depends on α):
```
L(α) = D · [log Γ(Σ_k α_k) − Σ_k log Γ(α_k)]  +  Σ_d Σ_k (α_k − 1) · E[log θ_dk]
```
where `E[log θ_dk] = ψ(γ_dk) − ψ(Σ_j γ_dj)` comes from the per-doc Dirichlet variational posterior `q(θ_d) = Dirichlet(γ_d)`. We already compute `γ_d` inside CAVI (see [`_cavi_doc_inference`](../../../spark-vi/spark_vi/models/lda.py#L100-L155)) — `E[log θ_dk]` is one digamma call away.

**Gradient** (length K):
```
g_k = D · [ψ(Σ_j α_j) − ψ(α_k)]  +  Σ_d E[log θ_dk]
```

**Hessian** (K×K) — diagonal plus rank-1:
```
H = c · 1·1ᵀ − diag(d_k)
where c = D · ψ′(Σ_j α_j),  d_k = D · ψ′(α_k)
```
The off-diagonal entries are `c`; the diagonal entries are `c − d_k`. Since `ψ′` is positive (trigamma) and the diagonal dominates in the regime we care about, H is negative-definite → the objective is concave → Newton ascent is safe.

**Newton step** via Sherman-Morrison gives a closed-form O(K) update (no K×K matrix is materialized):
```
Δα_k = (g_k − b) / d_k
where  b = Σ_j (g_j / d_j) / (Σ_j 1/d_j − 1/c)
```
This matches Blei et al 2003 §5.4 and MLlib's `OnlineLDAOptimizer.updateAlpha`. We will bit-match the MLlib Scala source during implementation as a sanity check.

**Mini-batch scaling**: the per-doc sum `Σ_d E[log θ_dk]` is taken over the batch and scaled by `D / |batch|` (corpus / batch size), matching the same factor we apply to λ stats per [ADR 0005](../../decisions/0005-mini-batch-sampling.md).

**Online combination** (Hoffman 2010 §3.3): `α ← α + ρ_t · Δα` where `Δα = −H⁻¹·g` is the full Newton step from current α. Same `ρ_t` as λ.

### Symmetric scalar η

ELBO terms involving η:
```
L(η) = K · log Γ(V·η) − K·V · log Γ(η)  +  (η − 1) · Σ_t Σ_v E[log φ_tv]
```
where `E[log φ_tv] = ψ(λ_tv) − ψ(Σ_v′ λ_tv′)` comes from the global topic posterior `q(φ_t) = Dirichlet(λ_t)`. K = number of topics, V = vocabulary size.

**Gradient** (scalar):
```
g(η) = K·V · [ψ(V·η) − ψ(η)]  +  Σ_t Σ_v E[log φ_tv]
```

**Hessian** (scalar):
```
H(η) = K·V² · ψ′(V·η) − K·V · ψ′(η)
```
Negative across the operating range η ∈ [10⁻³, ~10²] (verified by the inequality `V · ψ′(V·η) < ψ′(η)` for V > 1 in that regime).

**Newton step**: `Δη = −g/H`. Same `ρ_t` damping as α and λ.

### Why the η stat is cheaper to integrate than α

The α stat `Σ_d Σ_k E[log θ_dk]` decomposes per-doc — `local_update` must accumulate a length-K vector across docs in its partition. The η stat `Σ_t Σ_v E[log φ_tv]` does **not** — it depends only on current global λ, which `update_global` already has in hand. So η optimization adds zero new return values from `local_update`. This same structural asymmetry will show up in HDP (γ stats are global, α stats are per-doc), which is why building both flavors here is the right de-risking step.

## Implementation surfaces

### 1. `spark_vi/models/lda.py` — VanillaLDA

Reference: [`spark-vi/spark_vi/models/lda.py`](../../../spark-vi/spark_vi/models/lda.py).

- Make `self.alpha` a numpy array of length K. Constructor accepts scalar (broadcast to symmetric length-K vector) or 1-D array of length K. Default unchanged: `1/K` symmetric.
- Add constructor flags `optimize_alpha: bool = False`, `optimize_eta: bool = False` (the Estimator plumbs these through; runner-direct callers default off).
- `local_update`: when `optimize_alpha`, accumulate length-K vector `e_log_theta_sum` over docs in the partition and return it as a new stat key. The `n_docs` scalar (already returned for diagnostics) is sufficient to scale on the global side.
- `update_global`:
  - When `optimize_alpha`: corpus-scale the `e_log_theta_sum`, compute Newton step Δα via the closed-form Sherman-Morrison formula, apply ρ_t damping, floor at `1e-3`.
  - When `optimize_eta`: compute `Σ_t Σ_v E[log φ_tv]` directly from current λ, compute scalar Newton step Δη, apply ρ_t damping, floor at `1e-3`.
- New module-private helpers (pure functions, easy to unit-test):
  - `_alpha_newton_step(alpha: np.ndarray, e_log_theta_sum_scaled: np.ndarray, D: float) -> np.ndarray`
  - `_eta_newton_step(eta: float, lambda_: np.ndarray, K: int, V: int) -> float`

### 2. `spark_vi/mllib/lda.py` — VanillaLDAEstimator / Model / shared Params

Reference: [`spark-vi/spark_vi/mllib/lda.py`](../../../spark-vi/spark_vi/mllib/lda.py).

- Add `optimizeTopicConcentration` Param to `_VanillaLDAParams`, default False, type bool, with help text mirroring `optimizeDocConcentration`.
- Flip `optimizeDocConcentration` default to True.
- [`_validate_unsupported_params`](../../../spark-vi/spark_vi/mllib/lda.py#L46-L78): drop the `optimizeDocConcentration=True` rejection; drop the vector-`docConcentration` rejection. Keep all other validations (`optimizer != "online"`, etc.).
- [`_build_model_and_config`](../../../spark-vi/spark_vi/mllib/lda.py#L82-L122): pass `optimize_alpha` and `optimize_eta` through to the VanillaLDA constructor; pass `docConcentration` through unchanged when length > 1 (it is now legal).
- New `VanillaLDAModel.alpha` accessor: expose the final fitted α vector so callers can introspect the optimization outcome. Mirror with `topicConcentration` accessor for fitted η.

### 3. `spark_vi/core/runner.py`

Reference: [`spark-vi/spark_vi/core/runner.py`](../../../spark-vi/spark_vi/core/runner.py). **No changes.** The runner already passes `local_update` outputs verbatim through `combine_stats` and `update_global`; new stat keys are transparent to it.

### 4. Tests

In [`spark-vi/tests/test_mllib_lda.py`](../../../spark-vi/tests/test_mllib_lda.py):

- **Update** [`test_default_params_match_mllib_lda`](../../../spark-vi/tests/test_mllib_lda.py#L8-L24): include `optimizeDocConcentration` in the parity sweep (now matches MLlib's True).
- **Delete and replace** [`test_optimize_doc_concentration_defaults_false_diverging_from_mllib`](../../../spark-vi/tests/test_mllib_lda.py#L36-L45): the divergence is gone; replace with a positive parity assertion.
- **Invert** [`test_optimize_doc_concentration_true_raises`](../../../spark-vi/tests/test_mllib_lda.py#L122-L128): True is now legal; assert it builds and the fit produces a model whose α has actually moved from initialization.
- **Invert** [`test_vector_doc_concentration_raises`](../../../spark-vi/tests/test_mllib_lda.py#L130-L136): vector α is now legal; assert it builds and round-trips through `_build_model_and_config`.

New unit tests (pure-Python, no Spark):

- `test_alpha_newton_step_recovers_known_alpha_on_synthetic`: construct known α = [0.1, 0.5, 0.9], synthesize the corresponding `e_log_theta_sum` from many sampled `θ_d` vectors, run Newton iterations on `_alpha_newton_step` from a uniform 1/K start, assert recovered α within 0.05 of truth.
- `test_eta_newton_step_recovers_known_eta_on_synthetic`: analogous test for `_eta_newton_step`.
- `test_alpha_newton_step_floors_at_1e-3`: pathological gradient driving α below 0; assert clipping kicks in.

New Spark integration test (marked `slow`) in [`spark-vi/tests/test_lda_integration.py`](../../../spark-vi/tests/test_lda_integration.py):

- `test_alpha_optimization_drifts_toward_corpus_truth`: synthetic LDA corpus generated with a known asymmetric α, fit with `optimizeDocConcentration=True`, assert final α is closer to truth than the 1/K initialization (L1 distance reduced by ≥30%).

Existing test that must continue to pass without changes:

- [`test_vanilla_lda_elbo_smoothed_trend_is_non_decreasing`](../../../spark-vi/tests/test_lda_integration.py#L70-L98) — this is the regression gate. Any sign error in either Newton update will break the smoothed ELBO trend.

## Verification

1. `pytest spark-vi/tests/test_mllib_lda.py -v` — all updated unit tests pass.
2. `pytest spark-vi/tests/test_lda_integration.py -v -m slow` — ELBO trend test passes; new α-drift integration test passes.
3. `pytest spark-vi/tests/ -v` — full spark-vi suite still green (no regressions in CAVI / λ update / persistence machinery).
4. Manual on-cluster smoke (next session): `make lda-bq-smoke` completes; `model.alpha` accessor returns a vector measurably moved away from 1/K.

## Literature references

For the implementation phase. Math walkthroughs ("teach the math while implementing") happen in code comments and the implementation conversation, not in this spec.

- **Blei, Ng, Jordan (2003), "Latent Dirichlet Allocation," §5.4** — Newton-Raphson formulation for asymmetric Dirichlet MLE; the diagonal-plus-rank-1 Hessian and Sherman-Morrison closed-form. Primary reference for the α update.
- **Hoffman, Blei, Bach (2010), "Online Learning for Latent Dirichlet Allocation," NIPS, §3.3-3.4** — mini-batch scaling and ρ-damping for online α and η updates. Primary reference for the online integration.
- **Wallach, Mimno, McCallum (2009), "Rethinking LDA: Why Priors Matter"** — empirical case for asymmetric α / symmetric η (the asymmetry pattern we are adopting).
- **Minka (2003), "Estimating a Dirichlet distribution"** — alternative fixed-point iteration; we go with Blei's Newton, but cite for completeness.
- **MLlib reference**: `org.apache.spark.mllib.clustering.LDAOptimizer.OnlineLDAOptimizer.updateAlpha` (Scala source, public). Bit-level reference for the Newton step formula. Cross-checked during implementation.

## Out of scope

- **HDP itself** — the eventual goal. Once this lands and the ELBO trend test confirms the Newton machinery is correct, HDP becomes a translation exercise: γ uses η's machinery (global stat), α uses LDA-α's scalar form (per-doc stat).
- **Asymmetric η** (per-vocab). Parking lot.
- **Concentration-specific learning rate**. Parking lot, add only if we observe instability.
- **Warmup iterations** before optimization. Parking lot, same justification.
- **MLWritable round-trip of optimized α / η**. Tied to ADR 0009's deferred persistence work.
