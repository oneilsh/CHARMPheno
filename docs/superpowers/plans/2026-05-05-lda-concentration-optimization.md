# LDA Concentration-Parameter Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement asymmetric α (length K) and symmetric scalar η Newton-Raphson optimization for VanillaLDA, exposed through the MLlib shim with MLlib-parity defaults.

**Architecture:** Two pure-Python helpers (`_alpha_newton_step`, `_eta_newton_step`) compute single Newton steps in closed form. VanillaLDA migrates α and η from instance attributes into `global_params` so they round-trip through the runner like λ. `update_global` runs the Newton step (when the matching flag is on), applies the same Robbins-Monro `ρ_t` already controlling λ, and floors the result at 1e-3. The MLlib shim flips its default to match MLlib's (`optimizeDocConcentration=True`), drops the v0 rejection branches, adds `optimizeTopicConcentration`, and exposes accessors for the trained values.

**Tech Stack:** Python 3.10+, NumPy, SciPy (`digamma`, `polygamma`), PySpark, pytest. Existing in-tree: `spark_vi.core.{model,runner,result,config}`, `spark_vi.models.lda`, `spark_vi.mllib.lda`.

**Spec:** [docs/superpowers/specs/2026-05-05-lda-concentration-optimization-design.md](../specs/2026-05-05-lda-concentration-optimization-design.md). [ADR 0010](../../decisions/0010-concentration-parameter-optimization.md).

---

## File Structure

**Modify:**
- `spark-vi/spark_vi/models/lda.py` — add helpers; α as ndarray; α/η in `global_params`; new constructor flags; Newton steps in `update_global`.
- `spark-vi/spark_vi/mllib/lda.py` — flip default; add Param; drop validator branches; plumb flags; new model accessors; transform reads trained α.
- `spark-vi/tests/test_lda_math.py` — pure-Python helper tests.
- `spark-vi/tests/test_lda_contract.py` — update assertions on α shape and stat-key set.
- `spark-vi/tests/test_mllib_lda.py` — invert two rejection tests; update default-parity test; add `optimizeTopicConcentration` test; assert trained-α accessor.
- `spark-vi/tests/test_lda_integration.py` — add `test_alpha_optimization_drifts_toward_corpus_truth`.

**No changes:**
- `spark-vi/spark_vi/core/runner.py` — new stat keys flow through `mapPartitions`/`treeReduce` transparently. Verified by reading [runner.py:164-188](../../../spark-vi/spark_vi/core/runner.py#L164-L188).

---

## Task 1: `_alpha_newton_step` helper (pure-Python)

**Files:**
- Modify: `spark-vi/spark_vi/models/lda.py` (add helper)
- Test: `spark-vi/tests/test_lda_math.py` (add recovery test)

- [ ] **Step 1: Add `polygamma` to imports**

In `spark-vi/spark_vi/models/lda.py`, change line 39 from:

```python
from scipy.special import digamma, gammaln
```

to:

```python
from scipy.special import digamma, gammaln, polygamma
```

- [ ] **Step 2: Write the failing recovery test**

Add to `spark-vi/tests/test_lda_math.py`:

```python
def test_alpha_newton_step_recovers_known_alpha_on_synthetic():
    """Newton iterations on _alpha_newton_step recover the true α from
    samples of Dir(α). Sanity check on the closed-form Sherman-Morrison step.
    """
    from spark_vi.models.lda import _alpha_newton_step

    rng = np.random.default_rng(42)
    true_alpha = np.array([0.1, 0.5, 0.9])
    K = 3
    D = 10000

    # Sample θ_d ~ Dir(true_alpha), gather Σ_d log θ_dk. Under a perfectly
    # concentrated variational posterior q(θ_d) = δ(θ_d - true_θ_d), this is
    # exactly the corpus-scaled e_log_theta_sum the helper expects.
    thetas = rng.dirichlet(true_alpha, size=D)
    e_log_theta_sum = np.log(thetas).sum(axis=0)  # shape (K,)

    # Initialize from the symmetric prior 1/K, iterate full Newton steps.
    alpha = np.full(K, 1.0 / K, dtype=np.float64)
    for _ in range(50):
        delta = _alpha_newton_step(alpha, e_log_theta_sum, D=float(D))
        alpha = alpha + delta
        alpha = np.maximum(alpha, 1e-3)

    np.testing.assert_allclose(alpha, true_alpha, atol=0.05)
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest spark-vi/tests/test_lda_math.py::test_alpha_newton_step_recovers_known_alpha_on_synthetic -v
```

Expected: FAIL with `ImportError: cannot import name '_alpha_newton_step'`.

- [ ] **Step 4: Implement `_alpha_newton_step`**

Add to `spark-vi/spark_vi/models/lda.py` directly after `_dirichlet_kl` (around line 105):

```python
def _alpha_newton_step(
    alpha: np.ndarray,
    e_log_theta_sum_scaled: np.ndarray,
    D: float,
) -> np.ndarray:
    """One Newton step for asymmetric Dirichlet α.

    Per Blei, Ng, Jordan 2003 Appendix A.4.2 (using the linear-time
    structured-Hessian Newton from Appendix A.2). The ELBO part depending on α is
        L(α) = D · [log Γ(Σ_k α_k) − Σ_k log Γ(α_k)]
             + Σ_d Σ_k (α_k − 1) E[log θ_dk]
    with gradient
        g_k = D · [ψ(Σ_j α_j) − ψ(α_k)] + Σ_d E[log θ_dk]
    and Hessian (diagonal-plus-rank-1)
        H = c · 1·1ᵀ − diag(d_k)
    where c = D · ψ′(Σα), d_k = D · ψ′(α_k).

    Sherman-Morrison gives the Newton step Δα = −H⁻¹·g in closed-form O(K):
        Δα_k = (g_k − b) / d_k
        b    = Σ_j(g_j/d_j) / (Σ_j 1/d_j − 1/c)

    Inputs:
      alpha:                    (K,) current α vector.
      e_log_theta_sum_scaled:   (K,) corpus-scaled Σ_d E[log θ_dk]
                                (i.e. batch sum × D / |batch|).
      D:                        corpus size (used in g, c, d_k).

    Returns:
      Δα: (K,) raw Newton step. Caller does the ρ_t damping and the
        post-step floor — keeping this function pure makes it trivial
        to unit-test against synthetic data.
    """
    alpha_sum = alpha.sum()
    g = D * (digamma(alpha_sum) - digamma(alpha)) + e_log_theta_sum_scaled
    c = D * polygamma(1, alpha_sum)
    d = D * polygamma(1, alpha)

    sum_g_over_d = (g / d).sum()
    sum_1_over_d = (1.0 / d).sum()
    b = sum_g_over_d / (sum_1_over_d - 1.0 / c)

    return (g - b) / d
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest spark-vi/tests/test_lda_math.py::test_alpha_newton_step_recovers_known_alpha_on_synthetic -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add spark-vi/spark_vi/models/lda.py spark-vi/tests/test_lda_math.py
git commit -m "feat(lda): add _alpha_newton_step closed-form Newton helper

Blei 2003 App. A.4.2 closed-form Newton (linear-time inversion via App. A.2) for asymmetric Dirichlet α.
Pure function; caller handles ρ-damping and the post-step floor.
Recovery test confirms convergence to true α on Dirichlet samples."
```

---

## Task 2: `_eta_newton_step` helper (pure-Python)

**Files:**
- Modify: `spark-vi/spark_vi/models/lda.py` (add helper)
- Test: `spark-vi/tests/test_lda_math.py` (add recovery test)

- [ ] **Step 1: Write the failing recovery test**

Add to `spark-vi/tests/test_lda_math.py`:

```python
def test_eta_newton_step_recovers_known_eta_on_synthetic():
    """Newton iterations on _eta_newton_step recover the true η from
    samples of Dir(η · 1_V). Symmetric scalar version of the α test.
    """
    from spark_vi.models.lda import _eta_newton_step

    rng = np.random.default_rng(7)
    true_eta = 0.5
    K = 50
    V = 100

    # Sample K topics φ_t ~ Dir(η · 1_V); compute Σ_t Σ_v log φ_tv.
    # As in the α test, this is the asymptotic E[log φ] under a sharply
    # concentrated variational q(φ_t) = δ(φ_t − true_φ_t).
    phis = rng.dirichlet(np.full(V, true_eta), size=K)
    e_log_phi_sum = float(np.log(phis).sum())

    eta = 0.1
    for _ in range(50):
        delta = _eta_newton_step(eta, e_log_phi_sum, K=K, V=V)
        eta = max(eta + delta, 1e-3)

    assert abs(eta - true_eta) < 0.05, f"got {eta}, expected ~{true_eta}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest spark-vi/tests/test_lda_math.py::test_eta_newton_step_recovers_known_eta_on_synthetic -v
```

Expected: FAIL with `ImportError: cannot import name '_eta_newton_step'`.

- [ ] **Step 3: Implement `_eta_newton_step`**

Add to `spark-vi/spark_vi/models/lda.py` directly after `_alpha_newton_step`:

```python
def _eta_newton_step(
    eta: float,
    e_log_phi_sum: float,
    K: int,
    V: int,
) -> float:
    """One Newton step for symmetric scalar Dirichlet η.

    Per Hoffman, Blei, Bach 2010 §3.4. The ELBO part depending on η is
        L(η) = K · log Γ(V·η) − K·V · log Γ(η)
             + (η − 1) · Σ_t Σ_v E[log φ_tv]
    with scalar gradient and Hessian
        g(η) = K·V · [ψ(V·η) − ψ(η)] + Σ_t Σ_v E[log φ_tv]
        H(η) = K·V² · ψ′(V·η) − K·V · ψ′(η)
    Newton step Δη = −g/H.

    Caller computes e_log_phi_sum from current λ (typically:
        (digamma(lam) − digamma(lam.sum(axis=1, keepdims=True))).sum()).
    Caller also applies ρ_t damping and the post-step floor.
    """
    g = K * V * (digamma(V * eta) - digamma(eta)) + e_log_phi_sum
    h = K * V * V * polygamma(1, V * eta) - K * V * polygamma(1, eta)
    return -g / h
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest spark-vi/tests/test_lda_math.py::test_eta_newton_step_recovers_known_eta_on_synthetic -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/lda.py spark-vi/tests/test_lda_math.py
git commit -m "feat(lda): add _eta_newton_step scalar Newton helper

Hoffman 2010 §3.4 scalar Newton step for symmetric Dirichlet η.
Caller passes e_log_phi_sum directly so the helper stays a pure
math function (parallel to _alpha_newton_step's signature)."
```

---

## Task 3: Migrate α / η into `global_params` (refactor; behavior unchanged)

This is a pure refactor that moves α / η from `self.alpha` / `self.eta` (post-fit) into `global_params["alpha"]` / `global_params["eta"]`. After this task, `self.alpha` / `self.eta` continue to hold the **initial** values (read from the constructor), and updates flow through `global_params`. No optimization yet — Newton steps come in Tasks 5 and 6.

**Files:**
- Modify: `spark-vi/spark_vi/models/lda.py` (constructor, `initialize_global`, `local_update`, `update_global`, `compute_elbo`, `infer_local`).
- Modify: `spark-vi/tests/test_lda_contract.py` (update α shape + stat-key set).
- Modify: `spark-vi/tests/test_lda_math.py` (`test_compute_elbo_lambda_kl_zero_when_lambda_equals_eta` — needs eta in global_params).

- [ ] **Step 1: Update constructor — `self.alpha` becomes a length-K ndarray**

In `spark-vi/spark_vi/models/lda.py`, replace lines 114-149 (the `__init__`) with:

```python
    def __init__(
        self,
        K: int,
        vocab_size: int,
        alpha: float | np.ndarray | None = None,
        eta: float | None = None,
        gamma_shape: float = 100.0,
        cavi_max_iter: int = 100,
        cavi_tol: float = 1e-3,
    ) -> None:
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        if vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1, got {vocab_size}")
        if alpha is None:
            alpha = 1.0 / K
        if eta is None:
            eta = 1.0 / K

        # alpha may be a scalar (broadcast to length-K symmetric vector) or
        # a length-K array (asymmetric). Always stored on self as a length-K
        # float64 array so downstream code treats both inputs uniformly.
        alpha_arr = np.asarray(alpha, dtype=np.float64)
        if alpha_arr.ndim == 0:
            alpha_arr = np.full(K, float(alpha_arr), dtype=np.float64)
        elif alpha_arr.ndim != 1 or alpha_arr.shape[0] != K:
            raise ValueError(
                f"alpha must be a scalar or a length-{K} 1-D array, "
                f"got shape {alpha_arr.shape}"
            )
        if (alpha_arr <= 0).any():
            raise ValueError(f"all alpha components must be > 0, got {alpha_arr}")
        if eta <= 0:
            raise ValueError(f"eta must be > 0, got {eta}")
        if gamma_shape <= 0:
            raise ValueError(f"gamma_shape must be > 0, got {gamma_shape}")
        if cavi_max_iter < 1:
            raise ValueError(f"cavi_max_iter must be >= 1, got {cavi_max_iter}")
        if cavi_tol <= 0:
            raise ValueError(f"cavi_tol must be > 0, got {cavi_tol}")

        self.K = int(K)
        self.V = int(vocab_size)
        self.alpha = alpha_arr             # length-K initial α (driver-side starting point)
        self.eta = float(eta)              # scalar initial η
        self.gamma_shape = float(gamma_shape)
        self.cavi_max_iter = int(cavi_max_iter)
        self.cavi_tol = float(cavi_tol)
```

- [ ] **Step 2: Update `initialize_global` to seed α / η in `global_params`**

In `spark-vi/spark_vi/models/lda.py`, replace the body of `initialize_global` (lines 153-174) with:

```python
    def initialize_global(self, data_summary: Any | None) -> dict[str, np.ndarray]:
        """Random Gamma(gamma_shape, 1/gamma_shape) init for lambda (K, V),
        plus the initial α (K-vector) and η (scalar) seeded from constructor.

        See VanillaLDA.__init__ for why α is always a length-K array internally.
        Putting α / η in global_params (rather than relying on self) means the
        runner broadcasts and round-trips them like λ; update_global can mutate
        them when concentration optimization is enabled (Tasks 5 and 6).

        gamma_shape=100 trace: see Hoffman 2010's onlineldavb.py line 126.
        """
        lam = np.random.gamma(
            shape=self.gamma_shape,
            scale=1.0 / self.gamma_shape,
            size=(self.K, self.V),
        )
        return {
            "lambda": lam,
            "alpha": self.alpha.copy(),         # defensive copy — runner mutates
            "eta": np.array(self.eta),          # 0-d ndarray for combine_stats type-uniformity
        }
```

- [ ] **Step 3: Update `local_update` to read α from `global_params`**

In `spark-vi/spark_vi/models/lda.py`, replace lines 188-240 (`local_update`'s body) with:

```python
    def local_update(
        self,
        rows: Iterable[BOWDocument],
        global_params: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """E-step on one Spark partition.

        Reads α from global_params (Task 3 refactor) so that mid-fit α
        updates from update_global propagate to the next iteration's CAVI
        without needing to re-broadcast self.alpha.

        For each BOWDocument:
          1. Run _cavi_doc_inference to get gamma_d, expElogthetad, phi_norm.
          2. Add the suff-stat row update to lambda_stats[:, indices].
          3. Accumulate the data-likelihood and per-doc Dirichlet-KL terms.
        """
        lam = global_params["lambda"]                                 # (K, V)
        alpha = global_params["alpha"]                                # (K,)
        expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))

        lambda_stats = np.zeros_like(lam)
        doc_loglik_sum = 0.0
        doc_theta_kl_sum = 0.0
        n_docs = 0

        for doc in rows:
            gamma_init = np.random.gamma(
                shape=self.gamma_shape,
                scale=1.0 / self.gamma_shape,
                size=self.K,
            )
            gamma, expElogthetad, phi_norm, _ = _cavi_doc_inference(
                indices=doc.indices,
                counts=doc.counts,
                expElogbeta=expElogbeta,
                alpha=alpha,
                gamma_init=gamma_init,
                max_iter=self.cavi_max_iter,
                tol=self.cavi_tol,
            )

            sstats_row = np.outer(expElogthetad, doc.counts / phi_norm)
            lambda_stats[:, doc.indices] += sstats_row

            doc_loglik_sum += float(np.sum(doc.counts * np.log(phi_norm)))
            doc_theta_kl_sum += _dirichlet_kl(gamma, alpha)
            n_docs += 1

        return {
            "lambda_stats": lambda_stats,
            "doc_loglik_sum": np.array(doc_loglik_sum),
            "doc_theta_kl_sum": np.array(doc_theta_kl_sum),
            "n_docs": np.array(float(n_docs)),
        }
```

- [ ] **Step 4: Update `_cavi_doc_inference` signature**

In `spark-vi/spark_vi/models/lda.py`, change the signature on line 49 from:

```python
    alpha: float,
```

to:

```python
    alpha: float | np.ndarray,
```

The body already works for both via NumPy broadcasting (line 84: `gamma = alpha + ...`).

- [ ] **Step 5: Update `update_global` to read η from `global_params` and pass α / η through**

In `spark-vi/spark_vi/models/lda.py`, replace lines 242-270 (`update_global`'s body) with:

```python
    def update_global(
        self,
        global_params: dict[str, np.ndarray],
        target_stats: dict[str, np.ndarray],
        learning_rate: float,
    ) -> dict[str, np.ndarray]:
        """SVI natural-gradient step on λ; α and η pass through unchanged.

        Tasks 5 and 6 will layer Newton-step updates on α and η here, gated
        on optimize_alpha / optimize_eta flags.

            lambda_new = (1 - rho) * lambda
                       + rho * (eta + expElogbeta * target_stats["lambda_stats"])

        See ADR 0008 §"Online VI update" and the prior-iteration commit's
        comment on the expElogbeta factor.
        """
        lam = global_params["lambda"]
        alpha = global_params["alpha"]
        eta = float(global_params["eta"])

        expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))
        target_lam = eta + expElogbeta * target_stats["lambda_stats"]
        new_lam = (1.0 - learning_rate) * lam + learning_rate * target_lam

        return {
            "lambda": new_lam,
            "alpha": alpha,
            "eta": np.array(eta),
        }
```

- [ ] **Step 6: Update `compute_elbo` to read η from `global_params`**

In `spark-vi/spark_vi/models/lda.py`, replace lines 272-299 (`compute_elbo`'s body) with:

```python
    def compute_elbo(
        self,
        global_params: dict[str, np.ndarray],
        aggregated_stats: dict[str, np.ndarray],
    ) -> float:
        """ELBO = doc-data-likelihood + (-doc-level-KL) + (-global-KL).

        η is read from global_params (Task 3 refactor) so an η-optimization
        update mid-fit (Task 6) feeds back into the global KL prior on the
        next ELBO computation.
        """
        lam = global_params["lambda"]
        eta = float(global_params["eta"])

        K, V = lam.shape
        eta_vec = np.full(V, eta, dtype=np.float64)
        global_kl = 0.0
        for k in range(K):
            global_kl += _dirichlet_kl(lam[k], eta_vec)

        return float(
            float(aggregated_stats["doc_loglik_sum"])
            - float(aggregated_stats["doc_theta_kl_sum"])
            - global_kl
        )
```

- [ ] **Step 7: Update `infer_local` to read α from `global_params`**

In `spark-vi/spark_vi/models/lda.py`, replace lines 301-326 (`infer_local`'s body) with:

```python
    def infer_local(self, row: BOWDocument, global_params: dict[str, np.ndarray]):
        """Single-document E-step under fixed global params.

        Pure function of (row, global_params). Reads α from global_params so
        a model trained with optimize_alpha=True transforms with the *trained*
        α rather than the constructor's initial value.
        """
        lam = global_params["lambda"]
        alpha = global_params["alpha"]

        expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))
        gamma_init = np.random.gamma(
            shape=self.gamma_shape,
            scale=1.0 / self.gamma_shape,
            size=self.K,
        )
        gamma, _, _, _ = _cavi_doc_inference(
            indices=row.indices,
            counts=row.counts,
            expElogbeta=expElogbeta,
            alpha=alpha,
            gamma_init=gamma_init,
            max_iter=self.cavi_max_iter,
            tol=self.cavi_tol,
        )
        theta = gamma / gamma.sum()
        return {"gamma": gamma, "theta": theta}
```

- [ ] **Step 8: Update `test_lda_contract.py` α-shape assertions**

In `spark-vi/tests/test_lda_contract.py`, update the two tests that assert α as scalar.

Replace `test_vanilla_lda_default_alpha_eta_match_one_over_k` at lines 25-30 with:

```python
def test_vanilla_lda_default_alpha_eta_match_one_over_k():
    """Default symmetric α and η both default to 1/K, matching MLlib.

    α is stored on self as a length-K vector (constructor broadcasts a
    scalar). η is scalar.
    """
    from spark_vi.models.lda import VanillaLDA

    m = VanillaLDA(K=4, vocab_size=10)
    np.testing.assert_allclose(m.alpha, 0.25)
    assert m.alpha.shape == (4,)
    assert m.eta == pytest.approx(0.25)
```

Replace `test_vanilla_lda_explicit_alpha_eta_respected` at lines 33-37 with:

```python
def test_vanilla_lda_explicit_alpha_eta_respected():
    from spark_vi.models.lda import VanillaLDA

    m = VanillaLDA(K=10, vocab_size=100, alpha=0.1, eta=0.2)
    np.testing.assert_allclose(m.alpha, 0.1)
    assert m.alpha.shape == (10,)
    assert m.eta == pytest.approx(0.2)
```

Add a new test directly after the explicit-test (vector α now legal at the model level):

```python
def test_vanilla_lda_accepts_vector_alpha():
    """A length-K alpha is accepted and stored verbatim (no broadcast)."""
    from spark_vi.models.lda import VanillaLDA

    m = VanillaLDA(K=3, vocab_size=10, alpha=np.array([0.1, 0.5, 0.9]))
    np.testing.assert_allclose(m.alpha, [0.1, 0.5, 0.9])

    # Wrong shape rejected.
    with pytest.raises(ValueError, match="length-3 1-D array"):
        VanillaLDA(K=3, vocab_size=10, alpha=np.array([0.1, 0.5]))
```

- [ ] **Step 9: Update test fixtures that build `global_params` directly**

`local_update` now reads `global_params["alpha"]`; `update_global` and `compute_elbo` read `global_params["eta"]`. Find and update each test fixture that hand-builds a global_params dict.

Run:

```bash
grep -n 'global_params=g\|"lambda":' spark-vi/tests/test_lda_contract.py spark-vi/tests/test_lda_math.py
```

For every hand-built `g = {"lambda": lam}`, add `"alpha": np.full(K, ...)` and `"eta": np.array(...)`.

Concrete edits in `spark-vi/tests/test_lda_math.py`:

In `test_compute_elbo_lambda_kl_zero_when_lambda_equals_eta` (around line 134), change:

```python
g = {"lambda": np.full((K, V), eta)}
```

to:

```python
g = {
    "lambda": np.full((K, V), eta),
    "alpha": np.full(K, 1.0 / K),
    "eta": np.array(eta),
}
```

Concrete edits in `spark-vi/tests/test_lda_contract.py`: find every hand-built `g = {...}` (search the file) and add `"alpha": np.full(K, m.alpha if np.isscalar(m.alpha) else m.alpha[0])` for K-aware shape, and `"eta": np.array(m.eta)`. Where `m.alpha` is already a vector (post-Task-3) just use `m.alpha.copy()` and `np.array(m.eta)`.

The simplest pattern that works in all locations:

```python
g = {
    "lambda": <existing>,
    "alpha": m.alpha.copy(),
    "eta": np.array(m.eta),
}
```

- [ ] **Step 10: Run the full test suite to confirm refactor is behavior-preserving**

```bash
pytest spark-vi/tests/ -v -k "not slow"
```

Expected: PASS for everything except integration tests marked `slow`.

```bash
pytest spark-vi/tests/test_lda_integration.py -v -m slow
```

Expected: PASS for the existing two integration tests (no behavior change yet).

- [ ] **Step 11: Commit**

```bash
git add spark-vi/spark_vi/models/lda.py spark-vi/tests/test_lda_math.py spark-vi/tests/test_lda_contract.py
git commit -m "refactor(lda): move α/η into global_params dict

α now stored as length-K ndarray (constructor broadcasts scalar).
α / η flow through global_params like λ does, so the runner
broadcasts/round-trips them and update_global can mutate them
when optimization flags fire (next commits). Behavior preserved."
```

---

## Task 4: Add `optimize_alpha` / `optimize_eta` constructor flags

**Files:**
- Modify: `spark-vi/spark_vi/models/lda.py` (`__init__`)
- Test: `spark-vi/tests/test_lda_contract.py` (constructor accepts flags)

- [ ] **Step 1: Write the failing test**

Add to `spark-vi/tests/test_lda_contract.py`:

```python
def test_vanilla_lda_optimize_flags_default_false():
    from spark_vi.models.lda import VanillaLDA

    m = VanillaLDA(K=3, vocab_size=10)
    assert m.optimize_alpha is False
    assert m.optimize_eta is False


def test_vanilla_lda_optimize_flags_can_be_set():
    from spark_vi.models.lda import VanillaLDA

    m = VanillaLDA(K=3, vocab_size=10, optimize_alpha=True, optimize_eta=True)
    assert m.optimize_alpha is True
    assert m.optimize_eta is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest spark-vi/tests/test_lda_contract.py::test_vanilla_lda_optimize_flags_default_false -v
```

Expected: FAIL with `AttributeError: 'VanillaLDA' object has no attribute 'optimize_alpha'`.

- [ ] **Step 3: Add flags to `__init__`**

In `spark-vi/spark_vi/models/lda.py`, update `__init__`'s signature (the change from Task 3, Step 1) to add the two new kwargs *before* `gamma_shape`:

```python
    def __init__(
        self,
        K: int,
        vocab_size: int,
        alpha: float | np.ndarray | None = None,
        eta: float | None = None,
        optimize_alpha: bool = False,
        optimize_eta: bool = False,
        gamma_shape: float = 100.0,
        cavi_max_iter: int = 100,
        cavi_tol: float = 1e-3,
    ) -> None:
```

And at the end of `__init__` (after the existing `self.cavi_tol = ...` line), add:

```python
        self.optimize_alpha = bool(optimize_alpha)
        self.optimize_eta = bool(optimize_eta)
```

- [ ] **Step 4: Run tests**

```bash
pytest spark-vi/tests/test_lda_contract.py::test_vanilla_lda_optimize_flags_default_false spark-vi/tests/test_lda_contract.py::test_vanilla_lda_optimize_flags_can_be_set -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/lda.py spark-vi/tests/test_lda_contract.py
git commit -m "feat(lda): add optimize_alpha/optimize_eta constructor flags

Default False; the actual Newton-step wiring lands in the next
two commits."
```

---

## Task 5: Wire α optimization through `local_update` + `update_global`

**Files:**
- Modify: `spark-vi/spark_vi/models/lda.py` (`local_update` returns `e_log_theta_sum`; `update_global` runs `_alpha_newton_step`)
- Modify: `spark-vi/tests/test_lda_contract.py` (update stat-key assertion)
- Test: `spark-vi/tests/test_lda_contract.py` (add α-update test at rho=1.0)

- [ ] **Step 1: Write the failing test**

Add to `spark-vi/tests/test_lda_contract.py`:

```python
def test_update_global_with_optimize_alpha_runs_newton_and_floors():
    """At ρ=1.0, optimize_alpha=True applies a full Newton step plus floor.

    The verification is structural: with synthetic e_log_theta_sum drawn from
    Dir([0.1, 0.5, 0.9]), one full Newton step from 1/K starting α moves
    α measurably toward the truth. Convergence (closed-form) is covered by
    test_alpha_newton_step_recovers_known_alpha_on_synthetic; this test
    confirms wiring.
    """
    from spark_vi.models.lda import VanillaLDA
    import numpy as np

    K, V = 3, 5
    rng = np.random.default_rng(0)
    true_alpha = np.array([0.1, 0.5, 0.9])
    thetas = rng.dirichlet(true_alpha, size=10000)
    e_log_theta_sum = np.log(thetas).sum(axis=0)

    m = VanillaLDA(K=K, vocab_size=V, optimize_alpha=True)
    g = {
        "lambda": np.ones((K, V)),
        "alpha": np.full(K, 1.0 / K),
        "eta": np.array(0.1),
    }
    target_stats = {
        "lambda_stats": np.zeros((K, V)),
        "e_log_theta_sum": e_log_theta_sum,  # already corpus-scaled (D=10000)
        "doc_loglik_sum": np.array(0.0),
        "doc_theta_kl_sum": np.array(0.0),
        "n_docs": np.array(10000.0),
    }
    new_g = m.update_global(g, target_stats, learning_rate=1.0)

    # α moved toward truth.
    assert np.argmax(new_g["alpha"]) == 2  # largest component is index 2 (0.9)
    assert np.argmin(new_g["alpha"]) == 0  # smallest is index 0 (0.1)
    # Floor respected.
    assert (new_g["alpha"] >= 1e-3).all()


def test_local_update_emits_e_log_theta_sum_when_optimize_alpha():
    """The new stat key is present iff optimize_alpha=True (avoids paying
    the digamma cost when off)."""
    from spark_vi.core.types import BOWDocument
    from spark_vi.models.lda import VanillaLDA

    K, V = 3, 5
    docs = [BOWDocument(
        indices=np.array([0, 2], dtype=np.int32),
        counts=np.array([1.0, 2.0]),
        length=3,
    )]
    g = {
        "lambda": np.ones((K, V)),
        "alpha": np.full(K, 1.0 / K),
        "eta": np.array(0.1),
    }

    m_off = VanillaLDA(K=K, vocab_size=V, optimize_alpha=False)
    stats_off = m_off.local_update(rows=iter(docs), global_params=g)
    assert "e_log_theta_sum" not in stats_off

    m_on = VanillaLDA(K=K, vocab_size=V, optimize_alpha=True)
    stats_on = m_on.local_update(rows=iter(docs), global_params=g)
    assert "e_log_theta_sum" in stats_on
    assert stats_on["e_log_theta_sum"].shape == (K,)
```

- [ ] **Step 2: Update existing stat-key set assertion**

In `spark-vi/tests/test_lda_contract.py` line 100, the assertion currently reads:

```python
assert set(stats.keys()) == {"lambda_stats", "doc_loglik_sum", "doc_theta_kl_sum", "n_docs"}
```

Replace with:

```python
# When optimize_alpha=False (the default), e_log_theta_sum is NOT emitted.
assert set(stats.keys()) == {"lambda_stats", "doc_loglik_sum", "doc_theta_kl_sum", "n_docs"}
```

(Comment-only change so the intent is explicit. The set is unchanged for `optimize_alpha=False`.)

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest spark-vi/tests/test_lda_contract.py::test_update_global_with_optimize_alpha_runs_newton_and_floors spark-vi/tests/test_lda_contract.py::test_local_update_emits_e_log_theta_sum_when_optimize_alpha -v
```

Expected: FAIL — new stat key not emitted; α not updated.

- [ ] **Step 4: Update `local_update` to emit `e_log_theta_sum` when flag is on**

In `spark-vi/spark_vi/models/lda.py`, inside `local_update`, replace the `for doc in rows:` block and the return statement (the body from "for doc in rows:" through the return, in the version from Task 3) with:

```python
        e_log_theta_sum = np.zeros(self.K, dtype=np.float64) if self.optimize_alpha else None

        for doc in rows:
            gamma_init = np.random.gamma(
                shape=self.gamma_shape,
                scale=1.0 / self.gamma_shape,
                size=self.K,
            )
            gamma, expElogthetad, phi_norm, _ = _cavi_doc_inference(
                indices=doc.indices,
                counts=doc.counts,
                expElogbeta=expElogbeta,
                alpha=alpha,
                gamma_init=gamma_init,
                max_iter=self.cavi_max_iter,
                tol=self.cavi_tol,
            )

            sstats_row = np.outer(expElogthetad, doc.counts / phi_norm)
            lambda_stats[:, doc.indices] += sstats_row

            doc_loglik_sum += float(np.sum(doc.counts * np.log(phi_norm)))
            doc_theta_kl_sum += _dirichlet_kl(gamma, alpha)
            n_docs += 1

            if self.optimize_alpha:
                # E[log θ_dk] = ψ(γ_dk) − ψ(Σ_j γ_dj). Accumulate per-doc,
                # corpus-scale on the driver in update_global.
                e_log_theta_sum += digamma(gamma) - digamma(gamma.sum())

        out: dict[str, np.ndarray] = {
            "lambda_stats": lambda_stats,
            "doc_loglik_sum": np.array(doc_loglik_sum),
            "doc_theta_kl_sum": np.array(doc_theta_kl_sum),
            "n_docs": np.array(float(n_docs)),
        }
        if e_log_theta_sum is not None:
            out["e_log_theta_sum"] = e_log_theta_sum
        return out
```

- [ ] **Step 5: Update `update_global` to run α Newton step when flag is on**

In `spark-vi/spark_vi/models/lda.py`, replace `update_global`'s body (the version from Task 3, Step 5) with:

```python
    def update_global(
        self,
        global_params: dict[str, np.ndarray],
        target_stats: dict[str, np.ndarray],
        learning_rate: float,
    ) -> dict[str, np.ndarray]:
        """SVI natural-gradient step on λ; optional Newton step on α.

        λ update (always):
            lambda_new = (1 - rho) * lambda
                       + rho * (eta + expElogbeta * target_stats["lambda_stats"])

        α update (only when self.optimize_alpha):
            Δα   = _alpha_newton_step(α, target_stats["e_log_theta_sum"], D=n_docs_scaled)
            α_new = clip(α + rho * Δα, min=1e-3)

        target_stats[*] is already corpus-scaled by the runner per ADR 0005,
        so target_stats["e_log_theta_sum"] is the corpus-equivalent
        Σ_d E[log θ_dk] and target_stats["n_docs"] is the corpus-equivalent D.
        """
        lam = global_params["lambda"]
        alpha = global_params["alpha"]
        eta = float(global_params["eta"])

        expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))
        target_lam = eta + expElogbeta * target_stats["lambda_stats"]
        new_lam = (1.0 - learning_rate) * lam + learning_rate * target_lam

        if self.optimize_alpha:
            D = float(target_stats["n_docs"])
            delta_alpha = _alpha_newton_step(
                alpha=alpha,
                e_log_theta_sum_scaled=target_stats["e_log_theta_sum"],
                D=D,
            )
            new_alpha = np.maximum(alpha + learning_rate * delta_alpha, 1e-3)
        else:
            new_alpha = alpha

        # η updates land in the next commit (Task 6).
        return {
            "lambda": new_lam,
            "alpha": new_alpha,
            "eta": np.array(eta),
        }
```

- [ ] **Step 6: Run the new tests + the full unit suite**

```bash
pytest spark-vi/tests/test_lda_contract.py spark-vi/tests/test_lda_math.py -v
```

Expected: PASS.

```bash
pytest spark-vi/tests/test_lda_integration.py -v -m slow
```

Expected: PASS (existing ELBO-trend test still green — no math regression).

- [ ] **Step 7: Commit**

```bash
git add spark-vi/spark_vi/models/lda.py spark-vi/tests/test_lda_contract.py
git commit -m "feat(lda): wire optimize_alpha through local_update + update_global

local_update accumulates e_log_theta_sum when the flag is on; the
runner's existing stats-scaling pre-multiplies it by D/|batch| per
ADR 0005. update_global runs the closed-form Newton step from
Task 1, ρ-damps it (same ρ_t as λ), and floors at 1e-3."
```

---

## Task 6: Wire η optimization through `update_global`

**Files:**
- Modify: `spark-vi/spark_vi/models/lda.py` (`update_global`)
- Test: `spark-vi/tests/test_lda_contract.py` (η update at ρ=1.0)

- [ ] **Step 1: Write the failing test**

Add to `spark-vi/tests/test_lda_contract.py`:

```python
def test_update_global_with_optimize_eta_runs_newton_and_floors():
    """ρ=1.0 + optimize_eta=True: η moves toward the value that fits the
    current λ topic-word distribution, plus the post-step floor."""
    from spark_vi.models.lda import VanillaLDA
    rng = np.random.default_rng(11)

    K, V = 50, 100
    true_eta = 0.5
    # Build a λ that looks like K topics drawn from Dir(true_eta · 1_V),
    # scaled up so digamma(λ) − digamma(λ.sum) ≈ log φ_t,v.
    phis = rng.dirichlet(np.full(V, true_eta), size=K)
    lam = phis * 1.0e6  # heavy concentration → digamma ≈ log

    m = VanillaLDA(K=K, vocab_size=V, eta=0.1, optimize_eta=True)
    g = {"lambda": lam, "alpha": np.full(K, 1.0 / K), "eta": np.array(0.1)}
    target_stats = {
        "lambda_stats": np.zeros((K, V)),
        "doc_loglik_sum": np.array(0.0),
        "doc_theta_kl_sum": np.array(0.0),
        "n_docs": np.array(1.0),
    }
    new_g = m.update_global(g, target_stats, learning_rate=1.0)

    new_eta = float(new_g["eta"])
    # One full Newton step from η=0.1 should overshoot the start meaningfully
    # toward 0.5 — exact convergence covered by the helper recovery test.
    assert new_eta > 0.15
    assert new_eta >= 1e-3  # floor
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest spark-vi/tests/test_lda_contract.py::test_update_global_with_optimize_eta_runs_newton_and_floors -v
```

Expected: FAIL — η is unchanged because optimize_eta is not yet wired.

- [ ] **Step 3: Wire η Newton step into `update_global`**

In `spark-vi/spark_vi/models/lda.py`, in `update_global`, replace the final `return {...}` block (the version from Task 5, Step 5) with:

```python
        if self.optimize_eta:
            # Σ_t Σ_v E[log φ_tv] from current λ — this stat is global, so
            # it does NOT come through target_stats; we compute it here.
            e_log_phi_sum = float(
                (digamma(new_lam) - digamma(new_lam.sum(axis=1, keepdims=True))).sum()
            )
            K, V = new_lam.shape
            delta_eta = _eta_newton_step(eta=eta, e_log_phi_sum=e_log_phi_sum, K=K, V=V)
            new_eta = max(eta + learning_rate * delta_eta, 1e-3)
        else:
            new_eta = eta

        return {
            "lambda": new_lam,
            "alpha": new_alpha,
            "eta": np.array(new_eta),
        }
```

- [ ] **Step 4: Run tests**

```bash
pytest spark-vi/tests/test_lda_contract.py -v
```

Expected: PASS.

```bash
pytest spark-vi/tests/test_lda_integration.py -v -m slow
```

Expected: PASS (ELBO trend test still green).

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/lda.py spark-vi/tests/test_lda_contract.py
git commit -m "feat(lda): wire optimize_eta through update_global

η stat (Σ_t Σ_v E[log φ_tv]) is computable directly from current λ,
so unlike α it doesn't need a new local_update return. Same ρ_t
damping, same 1e-3 floor."
```

---

## Task 7: Add `optimizeTopicConcentration` Param to MLlib shim

**Files:**
- Modify: `spark-vi/spark_vi/mllib/lda.py` (`_VanillaLDAParams`, Estimator constructor + `_setDefault`)
- Test: `spark-vi/tests/test_mllib_lda.py`

- [ ] **Step 1: Write the failing test**

Add to `spark-vi/tests/test_mllib_lda.py`:

```python
def test_optimize_topic_concentration_param_default_false():
    from spark_vi.mllib.lda import VanillaLDAEstimator

    e = VanillaLDAEstimator()
    assert e.getOrDefault("optimizeTopicConcentration") is False


def test_optimize_topic_concentration_param_can_be_set():
    from spark_vi.mllib.lda import VanillaLDAEstimator

    e = VanillaLDAEstimator(optimizeTopicConcentration=True)
    assert e.getOrDefault("optimizeTopicConcentration") is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest spark-vi/tests/test_mllib_lda.py::test_optimize_topic_concentration_param_default_false -v
```

Expected: FAIL — Param not declared.

- [ ] **Step 3: Add Param declaration**

In `spark-vi/spark_vi/mllib/lda.py`, in `_VanillaLDAParams`, directly after the `optimizeDocConcentration = Param(...)` block (around line 173-177), add:

```python
    optimizeTopicConcentration = Param(
        Params._dummy(), "optimizeTopicConcentration",
        "whether to optimize η (symmetric scalar) via Newton-Raphson; "
        "see ADR 0010",
        typeConverter=TypeConverters.toBoolean,
    )
```

- [ ] **Step 4: Add to constructor signature and `_setDefault`**

In `VanillaLDAEstimator.__init__`, add `optimizeTopicConcentration: bool = False` to the kwargs (place it directly after `optimizeDocConcentration`):

```python
        optimizeDocConcentration: bool = False,
        optimizeTopicConcentration: bool = False,
        gammaShape: float = 100.0,
```

In the same `__init__`, add `optimizeTopicConcentration=False` to the `_setDefault` call (the block at line 223-230):

```python
        self._setDefault(
            k=10, maxIter=20,
            featuresCol="features", topicDistributionCol="topicDistribution",
            optimizer="online",
            learningOffset=1024.0, learningDecay=0.51, subsamplingRate=0.05,
            optimizeDocConcentration=False,
            optimizeTopicConcentration=False,
            gammaShape=100.0, caviMaxIter=100, caviTol=1e-3,
        )
```

(Default flip on `optimizeDocConcentration` lands in Task 9 — keep it False here for now.)

- [ ] **Step 5: Run tests**

```bash
pytest spark-vi/tests/test_mllib_lda.py::test_optimize_topic_concentration_param_default_false spark-vi/tests/test_mllib_lda.py::test_optimize_topic_concentration_param_can_be_set -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add spark-vi/spark_vi/mllib/lda.py spark-vi/tests/test_mllib_lda.py
git commit -m "feat(mllib): add optimizeTopicConcentration Param

Mirrors optimizeDocConcentration's shape; default False (MLlib has
no equivalent so there is no parity target). Wiring through
_build_model_and_config lands in Task 10."
```

---

## Task 8: Drop validator branches; invert their tests

**Files:**
- Modify: `spark-vi/spark_vi/mllib/lda.py` (`_validate_unsupported_params`)
- Modify: `spark-vi/tests/test_mllib_lda.py` (invert 2 tests)

- [ ] **Step 1: Invert `test_optimize_doc_concentration_true_raises`**

In `spark-vi/tests/test_mllib_lda.py`, replace lines 122-128 (`test_optimize_doc_concentration_true_raises`) with:

```python
def test_optimize_doc_concentration_true_is_legal():
    """v0 rejected this; ADR 0010 makes it the default behavior."""
    from spark_vi.mllib.lda import VanillaLDAEstimator, _validate_unsupported_params

    e = VanillaLDAEstimator(optimizeDocConcentration=True)
    _validate_unsupported_params(e)  # should not raise
```

- [ ] **Step 2: Invert `test_vector_doc_concentration_raises`**

In `spark-vi/tests/test_mllib_lda.py`, replace lines 130-136 (`test_vector_doc_concentration_raises`) with:

```python
def test_vector_doc_concentration_is_legal():
    """v0 rejected this; ADR 0010 supports asymmetric α (length K)."""
    from spark_vi.mllib.lda import VanillaLDAEstimator, _validate_unsupported_params

    e = VanillaLDAEstimator(k=3, docConcentration=[0.1, 0.1, 0.1])
    _validate_unsupported_params(e)  # should not raise


def test_vector_doc_concentration_wrong_length_raises():
    """Vector α with length != k is still rejected."""
    from spark_vi.mllib.lda import VanillaLDAEstimator, _validate_unsupported_params

    e = VanillaLDAEstimator(k=4, docConcentration=[0.1, 0.1, 0.1])
    with pytest.raises(ValueError, match="length"):
        _validate_unsupported_params(e)
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest spark-vi/tests/test_mllib_lda.py::test_optimize_doc_concentration_true_is_legal spark-vi/tests/test_mllib_lda.py::test_vector_doc_concentration_is_legal spark-vi/tests/test_mllib_lda.py::test_vector_doc_concentration_wrong_length_raises -v
```

Expected: FAIL — first two raise; third doesn't (wrong-length isn't validated yet).

- [ ] **Step 4: Update `_validate_unsupported_params`**

In `spark-vi/spark_vi/mllib/lda.py`, replace the entire body of `_validate_unsupported_params` (lines 47-79) with:

```python
def _validate_unsupported_params(estimator: "VanillaLDAEstimator") -> None:
    """Raise ValueError for any configuration the shim cannot honor.

    Per ADR 0010 the v0 rejections of `optimizeDocConcentration=True` and
    vector `docConcentration` are gone — both are now first-class. The
    only remaining rejections are the genuinely unsupported ones:

      * optimizer != "online" — we are SVI-only.
      * vector docConcentration with length != k — the model demands a
        length-k vector when asymmetric.

    Silent fallback would mislead users about what they are getting.
    """
    optimizer = estimator.getOrDefault("optimizer")
    if optimizer != "online":
        raise ValueError(
            f"VanillaLDAEstimator only supports optimizer='online', got {optimizer!r}. "
            f"The 'em' optimizer is not implemented in this shim."
        )

    if estimator.isSet("docConcentration"):
        doc_conc = estimator.getOrDefault("docConcentration")
        if doc_conc is not None and len(doc_conc) > 1:
            k = estimator.getOrDefault("k")
            if len(doc_conc) != k:
                raise ValueError(
                    f"docConcentration vector must have length k={k}, "
                    f"got length {len(doc_conc)}."
                )
```

- [ ] **Step 5: Run tests**

```bash
pytest spark-vi/tests/test_mllib_lda.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add spark-vi/spark_vi/mllib/lda.py spark-vi/tests/test_mllib_lda.py
git commit -m "refactor(mllib): drop ADR 0009 validator rejections

optimizeDocConcentration=True and vector docConcentration are
now both legal. Only remaining rejection (besides 'optimizer')
is wrong-length vector α. Three test inversions."
```

---

## Task 9: Flip `optimizeDocConcentration` default to True; replace divergence test

**Files:**
- Modify: `spark-vi/spark_vi/mllib/lda.py` (constructor default)
- Modify: `spark-vi/tests/test_mllib_lda.py` (replace divergence test, update parity sweep)

- [ ] **Step 1: Replace the divergence test with a positive parity test**

In `spark-vi/tests/test_mllib_lda.py`, replace `test_optimize_doc_concentration_defaults_false_diverging_from_mllib` (lines 36-45) with:

```python
def test_optimize_doc_concentration_default_matches_mllib():
    """ADR 0010 flipped this default to match pyspark.ml.clustering.LDA."""
    from pyspark.ml.clustering import LDA as MLlibLDA
    from spark_vi.mllib.lda import VanillaLDAEstimator

    assert VanillaLDAEstimator().getOrDefault("optimizeDocConcentration") is True
    assert MLlibLDA().getOrDefault("optimizeDocConcentration") is True
```

- [ ] **Step 2: Add `optimizeDocConcentration` to the parity sweep**

In `spark-vi/tests/test_mllib_lda.py`, in `test_default_params_match_mllib_lda` (lines 8-24), update the param list (around line 16-20) to include `optimizeDocConcentration`:

```python
    for name in [
        "k", "maxIter", "featuresCol", "topicDistributionCol",
        "optimizer", "learningOffset", "learningDecay",
        "subsamplingRate", "optimizeDocConcentration",
    ]:
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest spark-vi/tests/test_mllib_lda.py::test_optimize_doc_concentration_default_matches_mllib spark-vi/tests/test_mllib_lda.py::test_default_params_match_mllib_lda -v
```

Expected: FAIL — current default is still False.

- [ ] **Step 4: Flip the default in the Estimator**

In `spark-vi/spark_vi/mllib/lda.py`, in `VanillaLDAEstimator.__init__` (the version after Task 7):

Change the kwarg default at line ~217 from:

```python
        optimizeDocConcentration: bool = False,
```

to:

```python
        optimizeDocConcentration: bool = True,
```

And in the `_setDefault` call (the version after Task 7), change:

```python
            optimizeDocConcentration=False,
```

to:

```python
            optimizeDocConcentration=True,
```

- [ ] **Step 5: Run tests**

```bash
pytest spark-vi/tests/test_mllib_lda.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add spark-vi/spark_vi/mllib/lda.py spark-vi/tests/test_mllib_lda.py
git commit -m "feat(mllib): default optimizeDocConcentration=True (MLlib parity)

Drops the ADR 0009 deliberate divergence; matches
pyspark.ml.clustering.LDA's default. Existing test that asserted
the divergence is replaced with a positive parity assertion."
```

---

## Task 10: Plumb optimize flags through `_build_model_and_config`

**Files:**
- Modify: `spark-vi/spark_vi/mllib/lda.py` (`_build_model_and_config`)
- Test: `spark-vi/tests/test_mllib_lda.py`

- [ ] **Step 1: Write the failing test**

Add to `spark-vi/tests/test_mllib_lda.py`:

```python
def test_param_translation_passes_optimize_flags_to_model():
    from spark_vi.mllib.lda import VanillaLDAEstimator, _build_model_and_config

    e = VanillaLDAEstimator(
        k=3, optimizeDocConcentration=True, optimizeTopicConcentration=True,
    )
    model, _ = _build_model_and_config(e, vocab_size=10)
    assert model.optimize_alpha is True
    assert model.optimize_eta is True


def test_param_translation_accepts_vector_doc_concentration():
    from spark_vi.mllib.lda import VanillaLDAEstimator, _build_model_and_config
    import numpy as np

    e = VanillaLDAEstimator(k=3, docConcentration=[0.1, 0.5, 0.9])
    model, _ = _build_model_and_config(e, vocab_size=10)
    np.testing.assert_allclose(model.alpha, [0.1, 0.5, 0.9])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest spark-vi/tests/test_mllib_lda.py::test_param_translation_passes_optimize_flags_to_model spark-vi/tests/test_mllib_lda.py::test_param_translation_accepts_vector_doc_concentration -v
```

Expected: FAIL — flags not plumbed; vector α path not handled.

- [ ] **Step 3: Update `_build_model_and_config`**

In `spark-vi/spark_vi/mllib/lda.py`, replace the body of `_build_model_and_config` (lines 82-122) with:

```python
def _build_model_and_config(
    estimator: "VanillaLDAEstimator",
    vocab_size: int,
) -> tuple[VanillaLDA, VIConfig]:
    """Translate Estimator Params into (VanillaLDA, VIConfig).

    Per ADR 0010, docConcentration may be:
      * unset / None → broadcast 1/k (symmetric).
      * length-1 list → scalar (broadcast to length-k symmetric).
      * length-k list → asymmetric vector α.
    Wrong-length vectors are rejected upstream by _validate_unsupported_params.
    """
    k = estimator.getOrDefault("k")

    doc_conc = estimator.getOrDefault("docConcentration") if estimator.isSet("docConcentration") else None
    if doc_conc is None:
        alpha = 1.0 / k
    elif len(doc_conc) == 1:
        alpha = float(doc_conc[0])
    else:
        # Length-k vector (validated by _validate_unsupported_params).
        alpha = np.asarray(doc_conc, dtype=np.float64)

    topic_conc = estimator.getOrDefault("topicConcentration") if estimator.isSet("topicConcentration") else None
    eta = 1.0 / k if topic_conc is None else float(topic_conc)

    model = VanillaLDA(
        K=k,
        vocab_size=vocab_size,
        alpha=alpha,
        eta=eta,
        optimize_alpha=estimator.getOrDefault("optimizeDocConcentration"),
        optimize_eta=estimator.getOrDefault("optimizeTopicConcentration"),
        gamma_shape=estimator.getOrDefault("gammaShape"),
        cavi_max_iter=estimator.getOrDefault("caviMaxIter"),
        cavi_tol=estimator.getOrDefault("caviTol"),
    )

    seed = estimator.getOrDefault("seed") if estimator.isSet("seed") else None
    config = VIConfig(
        max_iterations=estimator.getOrDefault("maxIter"),
        learning_rate_tau0=estimator.getOrDefault("learningOffset"),
        learning_rate_kappa=estimator.getOrDefault("learningDecay"),
        mini_batch_fraction=estimator.getOrDefault("subsamplingRate"),
        random_seed=seed,
    )
    return model, config
```

- [ ] **Step 4: Run tests**

```bash
pytest spark-vi/tests/test_mllib_lda.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/mllib/lda.py spark-vi/tests/test_mllib_lda.py
git commit -m "feat(mllib): plumb optimize flags + vector α through _build_model_and_config

Estimator flags (optimizeDocConcentration, optimizeTopicConcentration)
now reach VanillaLDA's optimize_alpha / optimize_eta. Vector
docConcentration (length k) becomes asymmetric initial α."
```

---

## Task 11: Update `_transform` UDF to use trained α from `global_params`

**Files:**
- Modify: `spark-vi/spark_vi/mllib/lda.py` (`VanillaLDAModel._transform`)
- Test: `spark-vi/tests/test_mllib_lda.py`

- [ ] **Step 1: Write the failing test**

Add to `spark-vi/tests/test_mllib_lda.py`:

```python
def test_transform_uses_trained_alpha_from_result(tiny_corpus_df):
    """When optimize_alpha=True, transform must read α from
    result.global_params['alpha'], not from the Estimator's
    docConcentration Param (which was the v0 path).
    """
    from spark_vi.mllib.lda import VanillaLDAEstimator
    import numpy as np

    estimator = VanillaLDAEstimator(
        k=3, maxIter=5, seed=0, subsamplingRate=1.0,
        optimizeDocConcentration=True,
    )
    model = estimator.fit(tiny_corpus_df)

    # The trained α is on result.global_params, not on the docConcentration Param.
    trained_alpha = model.result.global_params["alpha"]
    assert trained_alpha.shape == (3,)
    # Should have moved at least somewhere from the 1/3 init under 5 iters.
    assert not np.allclose(trained_alpha, 1.0 / 3, atol=1e-6)

    # Transform should not raise and should produce a valid distribution.
    out = model.transform(tiny_corpus_df)
    rows = out.select("topicDistribution").collect()
    for r in rows:
        arr = np.asarray(r["topicDistribution"].toArray())
        assert arr.shape == (3,)
        np.testing.assert_allclose(arr.sum(), 1.0, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest spark-vi/tests/test_mllib_lda.py::test_transform_uses_trained_alpha_from_result -v
```

Expected: FAIL — `_transform` still reads α from `docConcentration` Param.

- [ ] **Step 3: Update `_transform` to read α from `result.global_params`**

In `spark-vi/spark_vi/mllib/lda.py`, in `VanillaLDAModel._transform` (lines 370-427), replace the α-resolution block (currently lines 380-384):

```python
        # docConcentration may be unset (None default) → resolve to 1/k.
        if self.isSet("docConcentration") and self.getOrDefault("docConcentration") is not None:
            alpha = float(self.getOrDefault("docConcentration")[0])
        else:
            alpha = 1.0 / self.getOrDefault("k")
```

with:

```python
        # Use the trained α from VIResult — covers both static-α and
        # optimize_alpha paths uniformly. Always a length-K ndarray now.
        alpha = self._result.global_params["alpha"]
```

- [ ] **Step 4: Run tests**

```bash
pytest spark-vi/tests/test_mllib_lda.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/mllib/lda.py spark-vi/tests/test_mllib_lda.py
git commit -m "fix(mllib): transform uses trained α from VIResult, not docConcentration Param

When optimize_alpha=True the Estimator's docConcentration is just
the initial value; the trained α lives in result.global_params
and is what transform must use."
```

---

## Task 12: Add `alpha` and `topicConcentration` accessors on `VanillaLDAModel`

**Files:**
- Modify: `spark-vi/spark_vi/mllib/lda.py` (`VanillaLDAModel`)
- Test: `spark-vi/tests/test_mllib_lda.py`

- [ ] **Step 1: Write the failing test**

Add to `spark-vi/tests/test_mllib_lda.py`:

```python
def test_model_alpha_accessor_returns_trained_vector(tiny_corpus_df):
    from spark_vi.mllib.lda import VanillaLDAEstimator
    import numpy as np

    estimator = VanillaLDAEstimator(
        k=3, maxIter=5, seed=0, subsamplingRate=1.0,
        optimizeDocConcentration=True,
    )
    model = estimator.fit(tiny_corpus_df)

    alpha = model.alpha
    assert alpha.shape == (3,)
    np.testing.assert_allclose(alpha, model.result.global_params["alpha"])


def test_model_topic_concentration_accessor_returns_trained_eta(tiny_corpus_df):
    from spark_vi.mllib.lda import VanillaLDAEstimator

    estimator = VanillaLDAEstimator(
        k=3, maxIter=5, seed=0, subsamplingRate=1.0,
        optimizeTopicConcentration=True,
    )
    model = estimator.fit(tiny_corpus_df)

    eta = model.topicConcentration
    assert isinstance(eta, float)
    assert eta == float(model.result.global_params["eta"])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest spark-vi/tests/test_mllib_lda.py::test_model_alpha_accessor_returns_trained_vector spark-vi/tests/test_mllib_lda.py::test_model_topic_concentration_accessor_returns_trained_eta -v
```

Expected: FAIL — accessors not defined.

- [ ] **Step 3: Add accessors**

In `spark-vi/spark_vi/mllib/lda.py`, in `VanillaLDAModel` (around line 317, just before `vocabSize`), add:

```python
    @property
    def alpha(self) -> np.ndarray:
        """Trained α vector (length K).

        For models trained with optimizeDocConcentration=True, this is the
        result of empirical-Bayes optimization. For static-α models, it's
        the initial α (broadcast to length K) — which equals the
        constructor input either way (no surprise).
        """
        return self._result.global_params["alpha"]

    @property
    def topicConcentration(self) -> float:
        """Trained η scalar.

        For models trained with optimizeTopicConcentration=True, this is
        the result of empirical-Bayes optimization. Otherwise it's the
        initial η.
        """
        return float(self._result.global_params["eta"])
```

- [ ] **Step 4: Run tests**

```bash
pytest spark-vi/tests/test_mllib_lda.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/mllib/lda.py spark-vi/tests/test_mllib_lda.py
git commit -m "feat(mllib): add alpha + topicConcentration accessors on VanillaLDAModel

Read trained values from VIResult.global_params. Both accessors
work uniformly whether the corresponding optimization flag was on
or off — they reflect what the model used to fit, period."
```

---

## Task 13: Spark integration test — α drifts toward known truth

> **Implementation note:** the test as originally specified (assertion "L1 distance from truth must drop by ≥30%") was empirically shown to be unattainable at the planned synthetic-corpus scale during implementation. The actually-shipped test is `test_alpha_optimization_runs_end_to_end_without_regression`, a smoke gate over wiring / finiteness / floor / no-blow-up / movement-from-init. See ADR 0010 "Consequences" for the rationale and the spec for details. The original task text below is preserved for historical reference.

**Files:**
- Modify: `spark-vi/tests/test_lda_integration.py`

- [ ] **Step 1: Add the integration test**

Append to `spark-vi/tests/test_lda_integration.py`:

```python
@pytest.mark.slow
def test_alpha_optimization_drifts_toward_corpus_truth(spark):
    """A 40-iter fit with optimize_alpha=True moves α from 1/K initialization
    measurably toward the synthetic-corpus ground truth.

    Acceptance: L1 distance from truth must drop by ≥30% vs the initial
    symmetric 1/K start. We don't ask for exact convergence — that's a
    function of corpus size, batch fraction, and ρ schedule, none of
    which we want to overconstrain in the regression gate.
    """
    import numpy as np
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.models.lda import VanillaLDA

    K, V, D = 3, 30, 200
    true_alpha = np.array([0.1, 0.5, 0.9])
    np.random.seed(2)

    # Reuse the synthetic-corpus generator pattern from the existing
    # tests, but inject a known asymmetric α.
    from spark_vi.core import BOWDocument
    rng = np.random.default_rng(2)
    true_beta = rng.dirichlet(np.full(V, 0.05), size=K)
    docs = []
    for d in range(D):
        theta_d = rng.dirichlet(true_alpha)
        N_d = max(1, rng.poisson(40))
        zs = rng.choice(K, size=N_d, p=theta_d)
        ws = np.array([rng.choice(V, p=true_beta[z]) for z in zs])
        unique, counts = np.unique(ws, return_counts=True)
        docs.append(BOWDocument(
            indices=unique.astype(np.int32),
            counts=counts.astype(np.float64),
            length=int(counts.sum()),
        ))

    rdd = spark.sparkContext.parallelize(docs, numSlices=2).persist()
    rdd.count()

    cfg = VIConfig(max_iterations=40, mini_batch_fraction=0.3,
                   random_seed=2, convergence_tol=1e-9)
    model = VanillaLDA(K=K, vocab_size=V, optimize_alpha=True)
    result = VIRunner(model, config=cfg).fit(rdd)

    init_alpha = np.full(K, 1.0 / K)
    final_alpha = result.global_params["alpha"]

    init_l1 = np.abs(init_alpha - true_alpha).sum()
    final_l1 = np.abs(final_alpha - true_alpha).sum()

    assert final_l1 < 0.7 * init_l1, (
        f"α did not move enough toward truth: "
        f"init L1={init_l1:.4f}, final L1={final_l1:.4f}, "
        f"final α={final_alpha}"
    )
```

- [ ] **Step 2: Run the new test**

```bash
pytest spark-vi/tests/test_lda_integration.py::test_alpha_optimization_drifts_toward_corpus_truth -v -m slow
```

Expected: PASS.

- [ ] **Step 3: Run the full integration suite as the regression gate**

```bash
pytest spark-vi/tests/test_lda_integration.py -v -m slow
```

Expected: PASS — including the existing
`test_vanilla_lda_elbo_smoothed_endpoints_show_overall_improvement`.

- [ ] **Step 4: Run the full unit suite**

```bash
pytest spark-vi/tests/ -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/tests/test_lda_integration.py
git commit -m "test(lda): integration gate — α drifts toward truth under optimize_alpha

40-iter Spark-local fit on synthetic LDA corpus with known
asymmetric α=[0.1, 0.5, 0.9]. Acceptance: L1 distance from truth
drops ≥30% vs the 1/K init. Catches sign / scaling regressions
in the Newton step that wouldn't show up in the helper unit tests."
```

---

## Self-Review

**Spec coverage:** Each "In v1" item maps to a task:
- Asymmetric α (Blei 2003 App. A.4.2 Newton; linear-time inversion via App. A.2): Tasks 1, 5.
- Symmetric scalar η (Hoffman 2010 §3.4): Tasks 2, 6.
- Damping (reuse λ's ρ_t): Tasks 5, 6 (same `learning_rate` arg).
- Default flips (`optimizeDocConcentration=True`, `optimizeTopicConcentration=False`): Tasks 7, 9.
- Numerical floor (clip ≥1e-3): Tasks 5, 6.
- Vector `docConcentration` accepted: Tasks 8, 10.
- ELBO trend test stays green: Verified at Steps 6, 7-step in Task 5; final gate in Task 13.
- Drift integration test: Task 13.

**Placeholder scan:** No "TBD" / "TODO" / "similar to" / "add appropriate" — every step shows full code or full command. ✓

**Type / name consistency:**
- Helper signatures: `_alpha_newton_step(alpha, e_log_theta_sum_scaled, D)` — used identically in Task 1 (definition), Task 5 (call site).
- `_eta_newton_step(eta, e_log_phi_sum, K, V)` — used identically in Task 2 (definition), Task 6 (call site).
- `optimize_alpha`, `optimize_eta` (snake_case on VanillaLDA): introduced Task 4, used Tasks 5, 6, 10.
- `optimizeDocConcentration`, `optimizeTopicConcentration` (camelCase on shim): existing + introduced Task 7, used Tasks 9, 10.
- `global_params` keys `"lambda"`, `"alpha"`, `"eta"`: introduced Task 3, used Tasks 5, 6, 11, 12.
- New stat key `"e_log_theta_sum"`: introduced Task 5, consumed Task 5 in same file.

**Out-of-scope confirmed:** Asymmetric η, custom concentration learning rate, warmup iterations, MLWritable round-trip — none appear in any task. ✓
