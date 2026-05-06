# Online HDP — Design Spec

**Date:** 2026-05-06
**Status:** Draft, pending user review
**Scope:** Implementation of the Online Hierarchical Dirichlet Process topic
model (Wang, Paisley, Blei 2011) as a `VIModel` in the `spark_vi` framework,
with a clinical `CharmPhenoHDP` wrapper. Replaces the existing 60-line stub at
[`spark-vi/spark_vi/models/online_hdp.py`](../../../spark-vi/spark_vi/models/online_hdp.py).
Does **not** cover the MLlib-style estimator shim, cloud or local driver
scripts, concentration parameter optimization, or the lazy-lambda sparse
update — those are explicitly deferred to a follow-on ADR + spec pair after
v1 is built and unit-tested.

---

## Context

`VanillaLDA` is shipping and validated against pyspark MLlib's reference
`OnlineLDAOptimizer`. The framework's contract methods, mini-batch SVI, broadcast
lifecycle, persistence format, and ELBO-trend testing pattern are battle-tested.
What's missing is the model that motivated the framework in the first place: the
nonparametric Bayesian topic model that lets the data choose how many phenotypes
exist instead of forcing the user to guess K up front.

Online HDP is structurally similar to LDA but mechanically more involved. Where
LDA has one hidden variable per token (the topic indicator) and a per-doc
Dirichlet over topics, HDP has *two* levels of stick-breaking — corpus-level
sticks β over an unbounded set of atoms, and per-document sticks π that pick
finitely many atoms from β. Two new latent variables get added: `c_jt` (which
corpus atom does doc-atom t map to) and `z_jn` (which doc-atom does token n
pick). The variational distribution factorizes over these as q(c)·q(z)·q(β')·q(π')·q(φ),
and the doc-level coordinate-ascent runs over four blocks (a, b, φ, ζ in
the paper's notation) instead of LDA's two (γ, φ).

This spec lays out the math mapping, the framework integration shape, the test
plan, and the v1 / v2 split. The brainstorming session that preceded it settled
the major design forks:

- **Skip the lazy-lambda sparse-vocabulary update** in v1. Wang's reference code
  uses per-word timestamps + a running cumulative log-discount to amortize the
  natural-gradient (1−ρ) shrinkage across vocabulary words. At our clinical
  scale (V ≈ 5-10k concept_ids, much smaller than NLP V≈100k), full-V digamma
  per minibatch is cheap, and skipping the lazy update simplifies the
  distributed story enormously. Revisit only if profiling justifies the
  complexity.
- **Hold γ and α fixed** at user-set values. Concentration optimization (γ for
  the corpus stick, α for the doc stick) is well-known to materially shape
  topic discovery in HDP, and ADR 0010 already templates the Newton machinery
  on LDA. But it is its own piece of work — we punt to a follow-on ADR.
- **Defer optimal_ordering, MLlib shim, cloud/local drivers** to v2 once the
  core is built and unit-tested. This mirrors the ordering used for vanilla
  LDA: model first, shim second.
- **Real frozen-globals HDP doc CAVI** for `transform()`, not Wang's
  `hdp_to_lda` collapse shortcut. Required for held-out evaluation during
  training, the eventual on-device serving path, and the future
  patient-train/visit-infer enhancement. We never need the LDA-collapse
  shortcut — dropped from scope entirely.

## Goals

1. **Ship a real Online HDP model** running end-to-end through the existing
   `VIRunner` machinery, returning a fitted model with finite ELBO trace,
   non-degenerate λ, and learned corpus stick weights.
2. **Add `infer_local` for HDP** as the per-document frozen-globals doc-CAVI
   primitive. Same contract VanillaLDA uses (per ADR 0007). This is what the
   wrapper's `transform()` calls.
3. **Update the clinical wrapper** at
   [`charmpheno/phenotype/charm_pheno_hdp.py`](../../../charmpheno/charmpheno/phenotype/charm_pheno_hdp.py)
   to use descriptive names externally (`max_topics`, `max_doc_topics`) while
   the inner `OnlineHDP` keeps `T`, `K` symbols matching the paper / reference
   code.
4. **Realign the architecture docs** so the existing T/K naming inversion
   between paper and code is documented in one place readers will encounter.
5. **Land an ADR** recording the v1 scope choices (skip lazy update, fixed γ/α,
   no optimal ordering, real HDP transform, etc.) for future-self clarity.

## Scope

### In v1

- `OnlineHDP(VIModel)` with `initialize_global`, `local_update`,
  `update_global`, `compute_elbo`, `infer_local`, `iteration_summary`.
- Per-doc CAVI implementing paper Eqs 15-18, with the iter < 3 warmup trick
  preserved (Wang's empirical stability heuristic, not derived in the paper).
- M-step on (λ, u, v) implementing paper Eqs 22-27 (natural-gradient SVI).
- Full ELBO decomposition driver-side, including KL terms on the corpus stick
  and the topic Dirichlet.
- Updated `CharmPhenoHDP` wrapper with `.fit()` (via VIRunner) and
  `.transform()` (per-doc frozen-globals inference returning θ vectors).
- Replacement of the existing 60-line stub at `spark-vi/spark_vi/models/online_hdp.py`.
- Unit tests on math helpers, slow-tier ELBO-trend smoke and synthetic recovery
  tests, and a Wang-reference cross-check fixture for one fixed seed.
- ADR 0011 documenting the v1 scope decisions.
- Architecture-doc updates fixing T/K naming exposition.

### Not in v1 (explicit)

- **Lazy-lambda sparse-vocabulary update.** Wang's `m_timestamp` / `m_r`
  trick for amortizing (1-ρ) shrinkage across vocabulary words. Revisit only
  if profiling shows full-V digamma is the bottleneck.
- **Concentration parameter optimization for γ and α.** HDP-specific
  natural-gradient updates per Wang/Paisley/Blei §3.2. Will be its own ADR
  + spec pair, building on the LDA Newton machinery from ADR 0010. γ is
  particularly important because it controls how many topics get discovered;
  fixing it is a real limitation but the right v1 punt.
- **Topic optimal-ordering.** Wang's `optimal_ordering` sorts by descending
  λ_sum after each M-step. Useful for visualization, breaks reproducible topic
  indices across runs. VanillaLDA does not do this either; if we add it later,
  it should be a post-fit `.reorder_by_usage()` helper, not an in-loop
  side-effect.
- **MLlib shim and driver scripts.** Following the LDA precedent (ADR 0009
  added the shim *after* the model was built and validated), the
  `spark_vi.mllib.HDP` shim plus `analysis/local/` and `analysis/cloud/`
  driver scripts land as v2 work.
- **Held-out perplexity track during training.** Free once `infer_local`
  exists, would give a less-noisy convergence signal complementary to the
  ELBO. Worth doing for both LDA and HDP in a single follow-on.
- **Patient-train / visit-infer pipeline.** Train the HDP on patient-level
  documents (long, sharp θ posteriors), then run frozen-globals inference at
  visit granularity (sharp Stage-2 inputs). Documented at
  [`TOPIC_STATE_MODELING.md L507-523`](../../architecture/TOPIC_STATE_MODELING.md#L507-L523).
  V1 trains at visit granularity throughout; the standalone `infer_local`
  entry point we ship in v1 is what enables the future split.
- **`warmup_iters=0` ablation.** Once v1 lands, run a one-off informal
  comparison of ELBO trajectories with and without the iter < 3 warmup trick
  to feed a v2 decision: keep, parameterize, or drop.

### Never (dropped from scope)

- **`hdp_to_lda` collapse helper.** Wang's mechanism for turning a fitted HDP
  into an LDA-equivalent for held-out scoring. We don't need LDA-shaped HDP
  outputs; our `transform` returns real HDP doc-CAVI posteriors.
- **`simulate`, `print_topics`, `plot_topics`, `reorder_by_usage`** as
  built-in convenience methods. Useful as user-side helpers but not required;
  add if and when an actual use case demands them.

## Math

The notation in this section follows the **code convention**: `T` = corpus-level
truncation, `K` = doc-level truncation. The Wang/Paisley/Blei 2011 paper inverts
this (`K` = corpus, `T` = doc); the architecture-doc update in this spec makes
that inversion explicit so future readers don't get confused. All references
below to "paper Eq N" use the paper's symbols, and the code below uses ours.

### Variational distribution

Following paper Eq 12-13, fully factorized:

```
q(β', π', c, z, φ) = q(β') · q(c) · q(z) · q(π') · q(φ)
```

with:
- `q(β'_k) = Beta(u_k, v_k)`           — corpus sticks, k = 0..T-2
- `q(π'_jt) = Beta(a_jt, b_jt)`        — doc sticks, t = 0..K-2 (per doc)
- `q(c_jk = t) = var_phi[k, t]`        — doc-atom k → corpus-atom t (per doc)
- `q(z_jn = k) = phi[n, k]`            — token n → doc-atom k (per doc)
- `q(φ_t) = Dirichlet(λ_t)`            — topic-word, t = 0..T-1

`var_phi` and `phi` are *per-document* and live only inside doc-CAVI; only
their suff-stats survive into M-step. (`u`, `v`, `λ`) are global params owned
by the driver.

**Per-token vs. per-unique-word indexing.** The paper writes `q(z_jn = k)` for
each token n. In code we collapse by unique word: `phi` has shape
`(Wt, K)` where `Wt` is the number of unique words in the document, and the
per-token weighting is recovered by multiplying by `counts` at the points
indicated in the pseudocode below. Mathematically equivalent — `phi[w, k]`
is shared across all `counts[w]` tokens of word w. Same convention LDA uses
in [`_cavi_doc_inference`](../../../spark-vi/spark_vi/models/lda.py#L45-L92)
and the same one Wang's reference and the intel-spark port both follow.

### Doc-CAVI inner loop (paper Eqs 15-18)

For one document with `Wt` unique words, integer `counts ∈ ℝ^Wt`, and indices
`doc.indices ∈ ℤ^Wt`:

**Setup.** `Elogbeta_doc = Elogbeta[:, doc.indices]` of shape `(T, Wt)`,
`Elog_sticks_corpus ∈ ℝ^T` from `expect_log_sticks(u, v)`. These come from the
broadcast of the current global params.

**Initialize per-doc state.**
```
a = ones(K-1)
b = alpha * ones(K-1)
phi = (1/K) * ones((Wt, K))
Elog_sticks_doc = expect_log_sticks(a, b)        # length-K
```

**Iterate up to `cavi_max_iter`, breaking on doc-ELBO convergence:**

```
# 1) var_phi update — paper Eq 17. Shape (K, T).
log_var_phi = phi.T @ (Elogbeta_doc * counts[None, :]).T          # (K, T)
if it >= 3:
    log_var_phi += Elog_sticks_corpus[None, :]
log_var_phi = log_normalize_rows(log_var_phi)
var_phi = exp(log_var_phi)

# 2) phi update — paper Eq 18. Shape (Wt, K).
log_phi = (var_phi @ Elogbeta_doc).T                              # (Wt, K)
if it >= 3:
    log_phi += Elog_sticks_doc[None, :]
log_phi = log_normalize_rows(log_phi)
phi = exp(log_phi)

# 3) a, b update — paper Eqs 15-16. Shape (K-1,) each.
phi_w = phi * counts[:, None]                                     # (Wt, K)
phi_sum = phi_w.sum(axis=0)                                       # (K,)
a = 1.0 + phi_sum[:K-1]
b = alpha + cumsum(phi_sum[1:][::-1])[::-1]                       # b[t] = α + Σ_{s>t}
Elog_sticks_doc = expect_log_sticks(a, b)

# 4) Doc-ELBO convergence test
elbo = _doc_elbo(...)
if it > 0 and abs(elbo - prev_elbo) / abs(prev_elbo) < cavi_tol:
    break
prev_elbo = elbo
```

**The `it < 3` warmup.** Drops the prior-correction terms (`Elog_sticks_corpus`
on var_phi, `Elog_sticks_doc` on phi) for the first three iterations. Empirical
trick from Wang's reference code; not derived in the paper. Both the Python
reference and the intel-spark Scala port preserve it. We keep it; v2 will
ablate via a `warmup_iters=0` experiment.

### Per-doc emissions added to partition stats

```
lambda_stats[:, doc.indices] += var_phi.T @ (phi * counts[:, None]).T   # (T, Wt)
var_phi_sum_stats           += var_phi.sum(axis=0)                      # (T,)
doc_loglik_sum    += sum(phi.T * (var_phi @ (Elogbeta_doc * counts[None, :])))
doc_z_term_sum    += sum((Elog_sticks_doc[None, :] - log_phi) * phi)
doc_c_term_sum    += sum((Elog_sticks_corpus[None, :] - log_var_phi) * var_phi)
doc_stick_kl_sum  += beta_kl(a, b, prior_a=1.0, prior_b=alpha)
n_docs            += 1
```

### Helper: `expect_log_sticks(a, b)`

Returns paper's `E[log β_k]` (or `E[log π_t]` at doc level) under
`β_k = β'_k · Π_{l<k}(1−β'_l)` with `β'_k ~ Beta(a_k, b_k)`:

```python
def expect_log_sticks(a, b):
    # a, b are length (T-1,) for corpus or (K-1,) for doc.
    dig_sum = digamma(a + b)
    Elog_W   = digamma(a) - dig_sum     # E[log β']
    Elog_1mW = digamma(b) - dig_sum     # E[log(1 - β')]
    out = zeros(len(a) + 1)
    out[:-1] = Elog_W
    out[1:] += cumsum(Elog_1mW)
    return out
```

The trailing entry handles truncation: `q(β'_T = 1) = 1` so `E[log β'_T] = 0`
and only the cumulative `E[log(1-β')]` contributes. Standard Sethuraman-style
stick-breaking expectation.

### M-step (paper Eqs 22-27)

```python
def update_global(global_params, target_stats, learning_rate):
    rho = learning_rate
    lam, u, v = global_params["lambda"], global_params["u"], global_params["v"]
    s = target_stats["var_phi_sum_stats"]   # (T,)
    s_tail = cumsum(s[1:][::-1])[::-1]      # (T-1,)

    new_lambda = (1 - rho) * lam + rho * (eta + target_stats["lambda_stats"])
    new_u      = (1 - rho) * u   + rho * (1.0   + s[:T-1])
    new_v      = (1 - rho) * v   + rho * (gamma + s_tail)

    return {"lambda": new_lambda, "u": new_u, "v": new_v}
```

The runner pre-scales `target_stats` by `D / batch_size` per ADR 0005 so
`lambda_stats` and `var_phi_sum_stats` arrive already corpus-scaled.
Learning rate `ρ_t = (τ + t)^(−κ)` is computed by the runner from `VIConfig`.

### ELBO

Paper Eq 14 decomposes into per-doc and corpus-level pieces. Per-doc terms are
already aggregated by `local_update`:

```
ELBO_per_doc = doc_loglik_sum + doc_z_term_sum + doc_c_term_sum - doc_stick_kl_sum
```

Driver-side `compute_elbo` subtracts the corpus-level KL terms:

```
KL_corpus = KL[q(β') ‖ p(β')]   # Beta(u_k, v_k) ‖ Beta(1, gamma), summed k=0..T-2
          + KL[q(φ) ‖ p(φ)]      # Dirichlet(λ_t) ‖ Dirichlet(η · 1_V), summed t=0..T-1

ELBO = ELBO_per_doc - KL_corpus
```

`KL[q(φ) ‖ p(φ)]` reuses VanillaLDA's `_dirichlet_kl` helper directly.
`KL[q(β') ‖ p(β'))` is a standard closed-form Beta KL — implemented as a
new module-private helper.

In minibatch mode, `ELBO_per_doc` is the minibatch sum (not corpus-scaled).
We follow VanillaLDA's choice: `compute_elbo` reports the unscaled minibatch
contribution + corpus-level KL. The ELBO-trend test interprets this as a noisy
unbiased estimator and checks for monotone *smoothed-endpoint* improvement, not
true monotonicity (see test plan).

### Initialization

- **`λ` (T × V)**: `Gamma(shape=gamma_shape, scale=1/gamma_shape)` where
  `gamma_shape = 100` by default, matching VanillaLDA. Departs from Wang's
  reference (`Gamma(1,1) · D · 100 / (T·V) − η`); his D-scale-then-η-cancel
  is undocumented and not derived from anything. Match-LDA is the boring,
  validated choice.
- **`u` = ones(T-1)**, **`v` = gamma * ones(T-1)**: paper-following init at
  the prior mean `Beta(1, γ)`. Departs from Wang's reference
  (`v = [T-1, T-2, ..., 1]`, "make a uniform at beginning") — that's an
  empirical bias toward low topic indices, undocumented in the paper, and we
  get the same effect more cleanly via a future `reorder_by_usage` helper.
- **Per-doc CAVI state** (`a, b, phi`): reset every time `_doc_e_step` runs.
  `a = 1, b = α, phi = uniform(K)`. Matches Wang's reference, no choice here.

## Implementation surfaces

### 1. `spark-vi/spark_vi/models/online_hdp.py` — replace stub

The current 60-line stub is fully replaced (not edited). Expected size ~600
lines, mirroring the structure of [`lda.py`](../../../spark-vi/spark_vi/models/lda.py).

**Module layout (top to bottom):**

- Module docstring + paper / reference-code citations.
- Pure helpers (free functions, easy to unit-test):
  - `_expect_log_sticks(a, b)` — paper Sethuraman-style stick expectation.
  - `_log_normalize_rows(M)` — numerically stable row-wise log-normalize.
  - `_beta_kl(u, v, prior_a, prior_b)` — closed-form Beta KL.
  - `_doc_e_step(indices, counts, Elogbeta_doc, Elog_sticks_corpus, alpha, K, max_iter, tol, warmup=3)`
    → returns `(a, b, phi, var_phi, log_phi, log_var_phi, doc_elbo_terms)`.
- `class OnlineHDP(VIModel)`:
  - `__init__(T, K, vocab_size, alpha=1.0, gamma=1.0, eta=0.01, gamma_shape=100.0, cavi_max_iter=100, cavi_tol=1e-4)`
    with full validation matching VanillaLDA's style.
  - `initialize_global(data_summary)` — emits `{lambda, u, v}` per the init
    rules above.
  - `local_update(rows, global_params)` — runs `_doc_e_step` per row, scatters
    suff-stats, returns the dict listed in [Per-doc emissions](#per-doc-emissions-added-to-partition-stats).
  - `update_global(global_params, target_stats, learning_rate)` — paper
    Eqs 22-27 SVI step.
  - `compute_elbo(global_params, aggregated_stats)` — sums per-doc terms +
    adds corpus-level KL.
  - `infer_local(row, global_params)` — single-doc frozen-globals doc-CAVI;
    returns `{a, b, phi, var_phi}` (caller derives θ from these — see wrapper).
  - `iteration_summary(global_params)` — short string for live-training
    display, derived from `(u, v)` and `λ`: effective active-topic count
    `#{k : E[β_k] > 1/(2T)}` (rough threshold for "this corpus topic carries
    real mass"), top-3 corpus stick weights, and the spread of `λ` row
    norms.

**Random seeding.** Follows VanillaLDA exactly: `np.random` with `gamma_shape`
init for λ. Carry over the same TODO from
[`lda.py:302-304`](../../../spark-vi/spark_vi/models/lda.py#L302-L304) about
per-doc deterministic seeds for reference-impl bit-matching. Fix lands in one
place when we do it for both models.

### 2. `charmpheno/charmpheno/phenotype/charm_pheno_hdp.py` — wrapper update

Replace the existing 64-line stub. Constructor gains descriptive names:

```python
class CharmPhenoHDP:
    def __init__(
        self,
        *,
        vocab_size: int,
        max_topics: int = 150,        # → inner T (corpus truncation)
        max_doc_topics: int = 15,     # → inner K (doc truncation)
        eta: float = 0.01,
        alpha: float = 1.0,
        gamma: float = 1.0,
        gamma_shape: float = 100.0,
        cavi_max_iter: int = 100,
        cavi_tol: float = 1e-4,
    ) -> None: ...
```

`omega` is removed (was an error in the bootstrap stub — never matched
anything in the paper or reference code). `max_topics` (was already present)
is preserved for clinical-user familiarity. `max_doc_topics`, `gamma`,
`gamma_shape`, `cavi_max_iter`, `cavi_tol` are new.

`.fit()` already delegates through `VIRunner` and works once the inner port
lands. New `.transform(data_rdd) -> RDD[(visit_id, theta_d)]` that calls the
runner's transform path → per-row `OnlineHDP.infer_local` → packages
`θ_d[t] = Σ_k π_k(a, b) · var_phi[k, t]` where `π_k(a, b)` is the doc
stick-breaking mean. This is the per-visit topic-proportion vector that
Stage 2 OU eventually consumes.

### 3. Architecture doc updates (in same commit as the implementation)

**[`docs/architecture/SPARK_VI_FRAMEWORK.md`](../../architecture/SPARK_VI_FRAMEWORK.md)
L260-306** — the OnlineHDP code sketch currently writes `(T=150, K=15, ...)`
without saying which truncation is which. Re-emit the sketch with explicit
inline comments labelling `T = corpus truncation` and `K = doc truncation`,
and point to the actual implementation for the live signature.

**[`docs/architecture/TOPIC_STATE_MODELING.md`](../../architecture/TOPIC_STATE_MODELING.md)
L289-301** — the param table already uses our convention (T = corpus,
K = doc). Add a short paragraph immediately after the table making the
paper-vs-code naming inversion explicit, since both Wang's reference Python
and the intel-spark Scala port use `T = corpus, K = doc` (matching us) but
the AISTATS paper uses `K = corpus, T = doc`. Single source of truth lives
in this paragraph; the rest of the doc and the code can refer back to it.

### 4. New ADR `docs/decisions/0011-online-hdp-design.md`

Records the v1 scope decisions: skip lazy update, fixed γ/α, no in-loop
optimal ordering, real frozen-globals transform (not LDA collapse), keep
warmup as default. Same shape as ADRs 0008-0010. Cites this spec as the
implementation detail.

## Tests

### Fast tier (in `make test`, finishes in seconds)

In `spark-vi/tests/test_online_hdp_unit.py` (new file, no Spark dependency):

- `test_expect_log_sticks_known_values` — for `T=3, a=[1,1], b=[1,1]`,
  hand-derive the expected vector and assert match within 1e-12.
- `test_log_normalize_rows_simplex_invariant` — random log-prob input, check
  `exp(out).sum(axis=1) ≈ 1` and `out` rows differ from input rows by a
  constant.
- `test_beta_kl_zero_when_priors_match` — `beta_kl(u=1, v=γ, prior_a=1, prior_b=γ) ≈ 0`.
- `test_beta_kl_positive_when_posterior_differs` — flip posterior, assert > 0.
- `test_doc_e_step_shape_and_simplex_contract` — 10-word doc, 5 doc atoms,
  10 corpus atoms; assert output shapes, `phi.sum(axis=1) ≈ 1`,
  `var_phi.sum(axis=1) ≈ 1`, `a > 0`, `b > 0`, no NaN/Inf.
- `test_doc_e_step_per_iter_elbo_nondecreasing` — enable a debug-mode
  assertion inside `_doc_e_step` that the per-iter doc ELBO never drops by
  more than `1e-9` (numerical noise tolerance). Coordinate ascent guarantees
  monotone increase; any decrease is a bug.
- `test_combine_stats_default_sum_works_for_hdp_keys` — instantiate two
  HDP suff-stats dicts, run them through `model.combine_stats`, assert
  elementwise sum.
- `test_update_global_rho_zero_is_identity` — assert `update_global` with
  `rho=0` returns the input globals unchanged.
- `test_update_global_rho_one_replaces_with_target` — assert
  `update_global` with `rho=1` replaces λ with `eta + target_stats[lambda_stats]`,
  u with `1 + s[:T-1]`, v with `γ + s_tail`.
- `test_initialize_global_shape_and_validity` — λ is `(T, V)` with all
  entries positive, u and v are `(T-1,)` with `u = 1` and `v = γ`.

### Slow tier (`@pytest.mark.slow`, run via `make test-all`)

In `spark-vi/tests/test_online_hdp_integration.py` (new file, Spark fixture):

- `test_online_hdp_short_fit_returns_finite_elbo_trace` — D=200, V=50, T=10,
  K=5 synthetic corpus. Run `VIRunner.fit` for 10 iterations. Assert result
  has positive finite λ, valid u, v, and a finite ELBO trace of length ≥ 10.
- `test_online_hdp_elbo_smoothed_endpoints_show_overall_improvement` — direct
  analogue of the LDA test at
  [`test_lda_integration.py:71-119`](../../../spark-vi/tests/test_lda_integration.py#L71-L119).
  D=200, V=50, T=10, K=5. Run for at least 30 iterations. Apply 10-iter
  moving-average smoothing. Assert `smooth[-1] > smooth[0]`. **Not** a
  monotonicity check; the docstring will spell out the same caveats as the
  LDA version. If empirical noise demands it, the test docstring permits
  falling back to a "first-quartile mean < last-quartile mean" comparison
  (window 15-20) before declaring HDP broken.
- `test_online_hdp_synthetic_recovery_top_topics` — D=2000 docs from a known
  HDP with 5 active true topics, fit with T=20 truncation. Assert that the
  top-5 topics by `var_phi_sum_stats` recover the true word distributions
  to cosine similarity > 0.9 after Hungarian matching.
- `test_online_hdp_infer_local_round_trip` — fit on D=200, then call
  `infer_local` on a held-out doc; assert convergence of doc-CAVI within
  `cavi_max_iter`, simplex-valid θ_d, mass concentrated on a small subset
  of corpus topics.
- `test_online_hdp_doc_e_step_matches_wang_reference_fixture` — for one
  fixed seed, single doc, 5 CAVI iterations: compare our `_doc_e_step`
  outputs `(a, b, phi, var_phi)` against precomputed expected values from
  Wang's `online-hdp/onlinehdp.py:doc_e_step` run standalone. Stored as a
  JSON fixture under `tests/fixtures/`. Tolerance ~1e-5. This is the
  "is the math right" gate. (Note: the fixture has to be generated once
  externally — Wang's code is Python 2 — and committed alongside the test.)

In `charmpheno/tests/test_charm_pheno_hdp.py` (new file):

- `test_charm_pheno_hdp_constructor_validates` — bad inputs raise.
- `test_charm_pheno_hdp_fit_smoke_tiny` — tiny synthetic (D=20, V=20,
  T=5, K=3); `.fit()` completes without raising; `VIResult` has finite
  ELBO trace.
- `test_charm_pheno_hdp_transform_returns_simplex_thetas` — fit on tiny
  synthetic, transform same data, assert output rows are length-T and
  sum-to-1 within 1e-9.

### Informal manual checks (not formalized)

During implementation, examining the ELBO trace and topic-word distributions
visually — same way LDA was sanity-checked. No fixture artifacts.

## Verification

1. `make test` — all fast-tier unit tests pass; suite still finishes in <10s.
2. `make test-all` — all slow-tier integration tests pass; ELBO trend test
   gates regressions, synthetic recovery test confirms correctness.
3. `make zip` and `make build` both produce green artifacts (no new C
   extensions or non-flat layout introduced).
4. `pre-commit run --all-files` clean — no committed test data files larger
   than the existing limits, no .parquet/.csv/etc. in the wrong places.
5. The ADR is written and committed; `SPARK_VI_FRAMEWORK.md` and
   `TOPIC_STATE_MODELING.md` updates land in the same commit as the
   implementation.

## Literature references

For implementation. Math walkthroughs ("teach the math while implementing")
happen in code comments and the implementation conversation, not in this
spec.

- **Wang, Paisley, Blei (2011), "Online Variational Inference for the
  Hierarchical Dirichlet Process," AISTATS** — primary algorithmic
  reference. All equation references in this spec point here. Section 3.1
  derives the doc-level CAVI updates (Eqs 15-18); Section 3.2 derives
  the natural-gradient SVI step (Eqs 22-27).
- **Teh, Jordan, Beal, Blei (2006), "Hierarchical Dirichlet Processes,"
  JASA** — the underlying nonparametric model definition. Background
  only; we do not re-derive anything from it.
- **Hoffman, Blei, Wang, Paisley (2013), "Stochastic Variational
  Inference," JMLR** — general SVI framework; same one we built
  VanillaLDA against. The minibatch-rescale-then-natural-gradient pattern
  applies unchanged here.
- **Sethuraman (1994), "A constructive definition of Dirichlet priors"** —
  cited only via Wang 2011's reuse of his stick-breaking construction;
  the actual `expect_log_sticks` formula falls out of standard
  Beta-distribution algebra.
- **Wang's reference Python implementation** at https://github.com/blei-lab/online-hdp,
  particularly `onlinehdp.py:doc_e_step` and `onlinehdp.py:update_lambda`.
  Used as the bit-matching cross-check fixture and as the source of the
  iter < 3 warmup trick.
- **intel-spark TopicModeling Scala port** at https://github.com/intel-spark/TopicModeling,
  particularly `OnlineHDP.scala`. Confirms the algorithmic reading of
  Wang's reference; we explicitly diverge from its `chunk.collect()`
  driver-side E-step in favor of `mapPartitions` + `treeReduce`.

## Open questions

- **Wang-fixture generation logistics.** Wang's reference is Python 2 with
  pinned dependencies. Generating the JSON fixture for the
  `test_online_hdp_doc_e_step_matches_wang_reference_fixture` test will
  involve a small Docker image or vendored 2-to-3 port of just the
  `doc_e_step` function. Decide during implementation; not a blocker for
  the spec.
