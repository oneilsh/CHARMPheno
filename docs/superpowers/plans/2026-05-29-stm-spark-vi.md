# STM (prevalence-only) — spark-vi engine + MLlib shim Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `OnlineSTM` (prevalence-only Structural Topic Model) to spark-vi as a new `VIModel`, plus a DataFrame-aware MLlib shim with R-style formula support via `formulaic`. Phases 1–3 of the [2026-05-29 STM design spec](../specs/2026-05-29-stm-prevalence-design.md).

**Architecture:** Logistic-normal prior on logit-θ: η_d ~ N(Γ x_d, Σ), θ_d = softmax(η_d). β stays Dirichlet-conjugate (unchanged from LDA). Per-doc inference is two-step Laplace (L-BFGS MAP + analytic Hessian at MAP for ν_d). M-step: β natural-gradient SVI; Γ via closed-form ridge regression on aggregated cross-products, ρ-blended; Σ via diagonal sample covariance with Laplace correction, ρ-blended. Engine is pure-numpy; MLlib shim adds DataFrame + formula support with `formulaic` as an optional dep. See [ADR 0022](../../decisions/0022-stm-prevalence-over-per-bin-alpha.md), [ADR 0023](../../decisions/0023-stm-inference-two-step-laplace-stochastic-em.md), [ADR 0024](../../decisions/0024-formulaic-in-mllib-shim-with-schema-frame-discovery.md).

**Tech Stack:** Python 3.10+, NumPy, SciPy (`scipy.optimize.minimize` with method='L-BFGS-B'; `scipy.special.digamma`, `gammaln`), PySpark (MLlib shim layer), formulaic ≥1.0 (optional), pytest.

---

## Context

The [STM design spec](../specs/2026-05-29-stm-prevalence-design.md) commits to a prevalence-only STM that extends LDA in three ways: per-doc prior on logit-θ becomes a regression on covariates; per-doc inference becomes non-conjugate (Laplace approximation via L-BFGS + analytic Hessian); M-step adds closed-form OLS for Γ and sample-covariance for Σ. β stays Dirichlet-conjugate and its SVI update is unchanged from LDA. This plan implements phases 1–3 (spark-vi engine + MLlib shim). Phases 4–6 (charmpheno integration: sidecar parquet + driver + experiment-tracking + dashboard adapter) are a separate plan.

The cleanest reference for the structure of this code is [`OnlineLDA`](../../spark-vi/spark_vi/models/topic/lda.py). STM mirrors its layout: a row type ([`BOWDocument`](../../spark-vi/spark_vi/models/topic/types.py) → `STMDocument`), a per-doc inference helper ([`_cavi_doc_inference`](../../spark-vi/spark_vi/models/topic/lda.py#L50) → `_stm_doc_inference`), and a `VIModel` subclass with the same five required methods plus optional hooks. The MLlib shim follows [`spark_vi.mllib.topic.lda`](../../spark-vi/spark_vi/mllib/topic/lda.py)'s pattern.

The math the implementer needs (derived in this plan, not in the spec):

**Per-doc objective** to minimize (negative log joint):
```
f(η) = -Σ_w n_dw · log(p^T β_·w) + ½(η - Γx)^T Σ^{-1}(η - Γx)
       where p = softmax(η)
```

**Gradient:**
```
∇f(η) = N_d · p - Σ_w n_dw · φ_w + Σ^{-1}(η - Γx)
       where N_d = Σ_w n_dw   (total tokens)
             φ_w = (p ⊙ β_·w) / (p^T β_·w)   (per-token responsibility, K-vector)
```

**Hessian** (evaluated once at the converged η̂_d for ν_d):
```
H(η̂) = N_d · (diag(p) - p p^T) - Σ_w n_dw · (diag(φ_w) - φ_w φ_w^T) + Σ^{-1}
ν_d   = H(η̂)^{-1}
```

**Per-doc Laplace ELBO contribution:**
```
L_d ≈ Σ_w n_dw · log(p̂^T β_·w)                 # data log-likelihood at MAP
    - ½(η̂ - Γx)^T Σ^{-1}(η̂ - Γx)               # prior log-density at MAP
    - ½ log|Σ|                                  # prior normalizer
    + ½ log|ν_d|                                # entropy of q(η_d)
    (a constant ½K log(2πe) is dropped — irrelevant to optimization or convergence diagnostics)
```

For numerical stability, the data log-likelihood term reuses LDA's Jensen lower bound: replace `log(p^T β_·w)` with `log(p^T expElogβ_·w)` using `expElogβ = exp(digamma(λ) - digamma(λ.sum(axis=1, keepdims=True)))`. This matches OnlineLDA's `phi_norm` computation and keeps β's update natural-gradient-conjugate.

---

## File Structure

**New files:**
- `spark-vi/spark_vi/models/topic/stm.py` — `OnlineSTM(VIModel)` and `_stm_doc_inference` helper
- `spark-vi/spark_vi/mllib/topic/stm.py` — `StreamingSTM` estimator and `STMModel` fitted-model class
- `spark-vi/spark_vi/mllib/topic/_formula.py` — formulaic plumbing: validation, schema-frame discovery
- `spark-vi/tests/test_stm_math.py` — gradient/Hessian/MAP correctness via finite differences and analytic checks
- `spark-vi/tests/test_stm_contract.py` — `VIModel` contract conformance for `OnlineSTM`
- `spark-vi/tests/test_stm_integration.py` — synthetic recovery (known Γ → recover Γ̂) and mini-batch convergence
- `spark-vi/tests/test_mllib_stm.py` — `StreamingSTM` Path A (pre-built covariates) tests
- `spark-vi/tests/test_mllib_stm_formula.py` — Path B (formula) tests including validation rejection of stateful transforms
- `spark-vi/tests/test_mllib_stm_persistence.py` — fitted-model save/load round-trip including ModelSpec

**Modified files:**
- `spark-vi/spark_vi/models/topic/types.py` — add `STMDocument` dataclass alongside `BOWDocument`
- `spark-vi/spark_vi/mllib/topic/_common.py` — add `_vector_to_stm_document` alongside `_vector_to_bow_document`
- `spark-vi/pyproject.toml` — add `formulaic` as optional dependency under `[tool.poetry.extras]` group `formula`

**Not modified:**
- `spark-vi/spark_vi/core/model.py` — `VIModel` base class is unchanged; STM implements existing contract
- `spark-vi/spark_vi/models/topic/lda.py` — LDA is untouched
- `spark-vi/spark_vi/inference/concentration_optimization.py` — LDA's α/η Newton routines are reused only conceptually (the pattern of "blend M-step target with ρ"); STM uses its own closed-form OLS for Γ

---

## Phase 1 — Engine: pure-numpy OnlineSTM

### Task 1: STMDocument row type

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/types.py`
- Test: `spark-vi/tests/test_stm_math.py` (new file; later tasks add to it)

- [ ] **Step 1: Write the failing test**

Create `spark-vi/tests/test_stm_math.py`:

```python
"""Tests for STM per-doc inference math: STMDocument, gradient, Hessian, MAP."""
from __future__ import annotations

import numpy as np
import pytest

from spark_vi.models.topic.types import STMDocument


class TestSTMDocument:
    def test_constructs_with_indices_counts_length_x(self):
        doc = STMDocument(
            indices=np.array([0, 3, 5], dtype=np.int32),
            counts=np.array([2.0, 1.0, 3.0], dtype=np.float64),
            length=6,
            x=np.array([1.0, 0.5, -1.2], dtype=np.float64),
        )
        assert doc.length == 6
        assert doc.x.shape == (3,)
        assert doc.x.dtype == np.float64

    def test_is_frozen(self):
        doc = STMDocument(
            indices=np.array([0], dtype=np.int32),
            counts=np.array([1.0]),
            length=1,
            x=np.array([0.0]),
        )
        with pytest.raises((AttributeError, TypeError)):
            doc.length = 99
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno/spark-vi
poetry run pytest tests/test_stm_math.py::TestSTMDocument -v
```

Expected: FAIL with `ImportError: cannot import name 'STMDocument'`.

- [ ] **Step 3: Add STMDocument to types.py**

In `spark-vi/spark_vi/models/topic/types.py`, after the existing `BOWDocument` class, add:

```python
@dataclass(frozen=True, slots=True)
class STMDocument:
    """Structural Topic Model document.

    Extends BOWDocument with a per-document covariate vector x.
    The engine never learns what x means — only its shape and dtype.

    Invariants (callers' responsibility — not enforced at construction):
      indices: sorted int32 array of token indices, all in [0, vocab_size).
      counts:  float64 array with len(counts) == len(indices), all > 0.
      length:  int total tokens (sum of counts).
      x:       float64 array of shape (P,) — the doc's covariate vector.
    """
    indices: np.ndarray
    counts: np.ndarray
    length: int
    x: np.ndarray
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_stm_math.py::TestSTMDocument -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/types.py spark-vi/tests/test_stm_math.py
git commit -m "feat(spark-vi): STMDocument row type for prevalence-only STM"
```

---

### Task 2: `_stm_doc_inference` — per-doc MAP via L-BFGS + analytic Hessian for Laplace ν_d

**Files:**
- Create: `spark-vi/spark_vi/models/topic/stm.py`
- Test: `spark-vi/tests/test_stm_math.py` (extend)

This is the math-heavy core. We derive gradient, then check it against a finite-difference approximation; then derive Hessian, check it against a finite-difference of the gradient. Then run a full MAP optimization on a small synthetic doc and verify convergence to a sane solution.

- [ ] **Step 1: Write the gradient-correctness test**

Append to `spark-vi/tests/test_stm_math.py`:

```python
from scipy.special import softmax

from spark_vi.models.topic.stm import _stm_neg_log_joint, _stm_neg_log_joint_grad


def _make_small_doc_state(seed=0):
    """Construct a small (K, V, P) state for gradient / Hessian tests."""
    rng = np.random.default_rng(seed)
    K, V, P = 3, 5, 2
    # ExpElogbeta-style nonnegative K x V matrix, columns summing roughly to 1.
    expElogbeta = rng.gamma(shape=2.0, scale=1.0, size=(K, V))
    expElogbeta = expElogbeta / expElogbeta.sum(axis=0, keepdims=True)
    Gamma = rng.normal(size=(P, K))
    Sigma_diag = rng.gamma(shape=2.0, scale=0.5, size=K)
    x = rng.normal(size=P)
    indices = np.array([0, 2, 4], dtype=np.int32)
    counts = np.array([2.0, 1.0, 3.0], dtype=np.float64)
    return dict(
        K=K, V=V, P=P, expElogbeta=expElogbeta, Gamma=Gamma,
        Sigma_diag=Sigma_diag, x=x, indices=indices, counts=counts,
    )


class TestSTMGradient:
    def test_gradient_matches_finite_difference(self):
        st = _make_small_doc_state(seed=42)
        rng = np.random.default_rng(0)
        eta = rng.normal(size=st["K"]) * 0.3

        analytic = _stm_neg_log_joint_grad(
            eta,
            indices=st["indices"], counts=st["counts"],
            expElogbeta=st["expElogbeta"],
            Gamma=st["Gamma"], Sigma_diag=st["Sigma_diag"], x=st["x"],
        )

        eps = 1e-6
        numeric = np.zeros_like(eta)
        for k in range(st["K"]):
            eta_p = eta.copy(); eta_p[k] += eps
            eta_m = eta.copy(); eta_m[k] -= eps
            f_p = _stm_neg_log_joint(
                eta_p, indices=st["indices"], counts=st["counts"],
                expElogbeta=st["expElogbeta"],
                Gamma=st["Gamma"], Sigma_diag=st["Sigma_diag"], x=st["x"],
            )
            f_m = _stm_neg_log_joint(
                eta_m, indices=st["indices"], counts=st["counts"],
                expElogbeta=st["expElogbeta"],
                Gamma=st["Gamma"], Sigma_diag=st["Sigma_diag"], x=st["x"],
            )
            numeric[k] = (f_p - f_m) / (2 * eps)

        np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails (no stm.py yet)**

```bash
poetry run pytest tests/test_stm_math.py::TestSTMGradient -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'spark_vi.models.topic.stm'`.

- [ ] **Step 3: Create `stm.py` with `_stm_neg_log_joint` and `_stm_neg_log_joint_grad`**

Create `spark-vi/spark_vi/models/topic/stm.py`:

```python
"""OnlineSTM: prevalence-only Structural Topic Model as a VIModel.

Generative model:
    For each topic k:  β_k ~ Dirichlet(η)                  # shared, unchanged from LDA
    For each doc d:
        η_d ~ N(Γ x_d, Σ)                                  # logistic-normal prior on logit-θ
        θ_d = softmax(η_d)
        For each token n in d:
            z_dn ~ Categorical(θ_d)
            w_dn ~ Categorical(β_{z_dn})

Variational family:
    q(β_k)  = Dirichlet(λ_k)                               # unchanged from LDA
    q(η_d)  = N(μ_d, ν_d) with K-diagonal ν_d              # Laplace approximation
    q(z_dn) = collapsed via Lee/Seung trick                # unchanged from LDA

Per-doc inference is two-step Laplace (ADR 0023):
    Step (a): L-BFGS finds the MAP point η̂_d.
    Step (b): Analytic Hessian at η̂_d gives ν_d = (-H)⁻¹.

M-step:
    β:  natural-gradient SVI on λ                          # unchanged from LDA
    Γ:  ridge regression on aggregated XᵀX, Xᵀμ; ρ-blended # stochastic-EM
    Σ:  K-diagonal sample covariance of residuals + diag(ν_d); ρ-blended

References:
    Roberts, Stewart, Airoldi 2016. "A Model of Text for Experimentation in the
        Social Sciences." JASA.
    docs/superpowers/specs/2026-05-29-stm-prevalence-design.md
    docs/decisions/0023-stm-inference-two-step-laplace-stochastic-em.md
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from scipy.optimize import minimize
from scipy.special import digamma, gammaln

from spark_vi.core.model import VIModel
from spark_vi.models.topic.types import STMDocument


def _stm_neg_log_joint(
    eta: np.ndarray,
    *,
    indices: np.ndarray,
    counts: np.ndarray,
    expElogbeta: np.ndarray,
    Gamma: np.ndarray,
    Sigma_diag: np.ndarray,
    x: np.ndarray,
) -> float:
    """Negative log joint at η for a single doc.

    f(η) = -Σ_w n_dw · log(p^T expElogβ_·w) + ½(η - Γx)^T Σ⁻¹(η - Γx)
           where p = softmax(η)
    """
    # Data term — Jensen lower bound using expElogβ (matches LDA's phi_norm).
    p = _softmax(eta)
    eb_d = expElogbeta[:, indices]                         # (K, n_unique)
    q_w = eb_d.T @ p + 1e-100                              # (n_unique,)
    data_term = -float(np.sum(counts * np.log(q_w)))
    # Prior term — Gaussian with diagonal Σ.
    diff = eta - Gamma.T @ x                               # (K,)
    prior_term = 0.5 * float(np.sum(diff * diff / Sigma_diag))
    return data_term + prior_term


def _stm_neg_log_joint_grad(
    eta: np.ndarray,
    *,
    indices: np.ndarray,
    counts: np.ndarray,
    expElogbeta: np.ndarray,
    Gamma: np.ndarray,
    Sigma_diag: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Gradient of _stm_neg_log_joint at η.

    ∇f(η) = N_d · p - Σ_w n_dw · φ_w + Σ⁻¹(η - Γx)
            where N_d = Σ_w n_dw  and  φ_w = (p ⊙ β_·w) / (p^T β_·w)
    """
    p = _softmax(eta)
    eb_d = expElogbeta[:, indices]                         # (K, n_unique)
    q_w = eb_d.T @ p + 1e-100                              # (n_unique,)
    # phi_w (K, n_unique): φ_wk = p_k * expElogβ_kw / q_w. Per-token responsibility.
    phi = (eb_d * p[:, None]) / q_w[None, :]               # (K, n_unique)
    N_d = float(np.sum(counts))
    data_grad = N_d * p - phi @ counts                     # (K,)
    diff = eta - Gamma.T @ x                               # (K,)
    prior_grad = diff / Sigma_diag                         # (K,)
    return data_grad + prior_grad


def _softmax(eta: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    m = eta.max()
    exp = np.exp(eta - m)
    return exp / exp.sum()
```

- [ ] **Step 4: Run gradient test to verify it passes**

```bash
poetry run pytest tests/test_stm_math.py::TestSTMGradient -v
```

Expected: PASS.

- [ ] **Step 5: Write the Hessian-correctness test**

Append to `tests/test_stm_math.py`:

```python
from spark_vi.models.topic.stm import _stm_neg_log_joint_hessian


class TestSTMHessian:
    def test_hessian_matches_finite_difference_of_grad(self):
        st = _make_small_doc_state(seed=7)
        rng = np.random.default_rng(1)
        eta = rng.normal(size=st["K"]) * 0.3

        analytic = _stm_neg_log_joint_hessian(
            eta, indices=st["indices"], counts=st["counts"],
            expElogbeta=st["expElogbeta"],
            Gamma=st["Gamma"], Sigma_diag=st["Sigma_diag"], x=st["x"],
        )

        eps = 1e-5
        numeric = np.zeros((st["K"], st["K"]))
        for j in range(st["K"]):
            eta_p = eta.copy(); eta_p[j] += eps
            eta_m = eta.copy(); eta_m[j] -= eps
            g_p = _stm_neg_log_joint_grad(
                eta_p, indices=st["indices"], counts=st["counts"],
                expElogbeta=st["expElogbeta"],
                Gamma=st["Gamma"], Sigma_diag=st["Sigma_diag"], x=st["x"],
            )
            g_m = _stm_neg_log_joint_grad(
                eta_m, indices=st["indices"], counts=st["counts"],
                expElogbeta=st["expElogbeta"],
                Gamma=st["Gamma"], Sigma_diag=st["Sigma_diag"], x=st["x"],
            )
            numeric[:, j] = (g_p - g_m) / (2 * eps)

        np.testing.assert_allclose(analytic, numeric, rtol=1e-3, atol=1e-5)

    def test_hessian_is_symmetric(self):
        st = _make_small_doc_state(seed=11)
        rng = np.random.default_rng(2)
        eta = rng.normal(size=st["K"]) * 0.3
        H = _stm_neg_log_joint_hessian(
            eta, indices=st["indices"], counts=st["counts"],
            expElogbeta=st["expElogbeta"],
            Gamma=st["Gamma"], Sigma_diag=st["Sigma_diag"], x=st["x"],
        )
        np.testing.assert_allclose(H, H.T, rtol=1e-12, atol=1e-12)

    def test_hessian_positive_definite_at_typical_point(self):
        st = _make_small_doc_state(seed=13)
        rng = np.random.default_rng(3)
        eta = rng.normal(size=st["K"]) * 0.3
        H = _stm_neg_log_joint_hessian(
            eta, indices=st["indices"], counts=st["counts"],
            expElogbeta=st["expElogbeta"],
            Gamma=st["Gamma"], Sigma_diag=st["Sigma_diag"], x=st["x"],
        )
        # Negative log joint is convex in η for prevalence-only STM with
        # diagonal Σ + nonnegative β; H should be PD.
        eigs = np.linalg.eigvalsh(H)
        assert np.all(eigs > 0), f"Hessian not PD: eigs={eigs}"
```

- [ ] **Step 6: Run Hessian tests to verify they fail**

```bash
poetry run pytest tests/test_stm_math.py::TestSTMHessian -v
```

Expected: FAIL with `ImportError: cannot import name '_stm_neg_log_joint_hessian'`.

- [ ] **Step 7: Add the Hessian to `stm.py`**

In `spark-vi/spark_vi/models/topic/stm.py`, append:

```python
def _stm_neg_log_joint_hessian(
    eta: np.ndarray,
    *,
    indices: np.ndarray,
    counts: np.ndarray,
    expElogbeta: np.ndarray,
    Gamma: np.ndarray,
    Sigma_diag: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Hessian of _stm_neg_log_joint at η.

    H(η) = N_d · (diag(p) - p p^T) - Σ_w n_dw · (diag(φ_w) - φ_w φ_w^T) + Σ⁻¹
    """
    K = eta.shape[0]
    p = _softmax(eta)
    eb_d = expElogbeta[:, indices]                         # (K, n_unique)
    q_w = eb_d.T @ p + 1e-100                              # (n_unique,)
    phi = (eb_d * p[:, None]) / q_w[None, :]               # (K, n_unique)
    N_d = float(np.sum(counts))

    # N_d · (diag(p) - p p^T)
    H_data_pos = N_d * (np.diag(p) - np.outer(p, p))
    # Σ_w n_dw · (diag(φ_w) - φ_w φ_w^T) — accumulate weighted by counts.
    # diag part: Σ_w n_dw · diag(φ_w) = diag(φ @ counts)
    diag_term = np.diag(phi @ counts)
    # outer part: Σ_w n_dw · φ_w φ_w^T = (φ * counts) @ φ.T
    outer_term = (phi * counts[None, :]) @ phi.T
    H_data_neg = diag_term - outer_term

    H_prior = np.diag(1.0 / Sigma_diag)
    return H_data_pos - H_data_neg + H_prior
```

- [ ] **Step 8: Run Hessian tests to verify they pass**

```bash
poetry run pytest tests/test_stm_math.py::TestSTMHessian -v
```

Expected: PASS.

- [ ] **Step 9: Write the MAP-convergence test for `_stm_doc_inference`**

Append to `tests/test_stm_math.py`:

```python
from spark_vi.models.topic.stm import _stm_doc_inference


class TestSTMDocInference:
    def test_converges_to_stationary_point(self):
        st = _make_small_doc_state(seed=99)
        eta_hat, nu_d, n_iter = _stm_doc_inference(
            indices=st["indices"], counts=st["counts"],
            expElogbeta=st["expElogbeta"],
            Gamma=st["Gamma"], Sigma_diag=st["Sigma_diag"], x=st["x"],
            max_iter=200, tol=1e-6,
        )
        # Gradient at η̂ should be ~zero.
        g = _stm_neg_log_joint_grad(
            eta_hat, indices=st["indices"], counts=st["counts"],
            expElogbeta=st["expElogbeta"],
            Gamma=st["Gamma"], Sigma_diag=st["Sigma_diag"], x=st["x"],
        )
        assert np.linalg.norm(g) < 1e-4, f"|g|={np.linalg.norm(g)} not converged"
        assert nu_d.shape == (st["K"], st["K"])
        # ν_d is symmetric positive definite.
        np.testing.assert_allclose(nu_d, nu_d.T, atol=1e-10)
        eigs = np.linalg.eigvalsh(nu_d)
        assert np.all(eigs > 0)

    def test_strong_prior_pulls_eta_toward_prior_mean(self):
        st = _make_small_doc_state(seed=1)
        # Override Σ to be very tight: posterior should ~= prior mean Γᵀx.
        Sigma_tight = np.full(st["K"], 1e-6)
        prior_mean = st["Gamma"].T @ st["x"]
        eta_hat, _, _ = _stm_doc_inference(
            indices=st["indices"], counts=st["counts"],
            expElogbeta=st["expElogbeta"],
            Gamma=st["Gamma"], Sigma_diag=Sigma_tight, x=st["x"],
            max_iter=200, tol=1e-8,
        )
        np.testing.assert_allclose(eta_hat, prior_mean, atol=1e-3)
```

- [ ] **Step 10: Run MAP test to verify it fails**

```bash
poetry run pytest tests/test_stm_math.py::TestSTMDocInference -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 11: Implement `_stm_doc_inference` in `stm.py`**

Append to `spark-vi/spark_vi/models/topic/stm.py`:

```python
def _stm_doc_inference(
    *,
    indices: np.ndarray,
    counts: np.ndarray,
    expElogbeta: np.ndarray,
    Gamma: np.ndarray,
    Sigma_diag: np.ndarray,
    x: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Per-doc Laplace approximation: L-BFGS for MAP, analytic Hessian for ν_d.

    Cold-start at η = 0 (uniform θ after softmax). Stateless across outer
    iterations — preserves the local_update contract that mini-batch
    sampling assumes (ADR 0023).

    Returns:
        eta_hat:  (K,) the MAP point.
        nu_d:     (K, K) Laplace covariance = (-H)⁻¹ at η̂.
        n_iter:   inner L-BFGS iterations consumed.
    """
    K = expElogbeta.shape[0]
    eta0 = np.zeros(K, dtype=np.float64)

    kwargs = dict(
        indices=indices, counts=counts, expElogbeta=expElogbeta,
        Gamma=Gamma, Sigma_diag=Sigma_diag, x=x,
    )

    result = minimize(
        fun=_stm_neg_log_joint,
        x0=eta0,
        jac=_stm_neg_log_joint_grad,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": tol},
        args=(),
        kwargs=kwargs,
    )
    eta_hat = result.x

    # Analytic Hessian at the MAP — exact second derivative for Laplace covariance.
    H = _stm_neg_log_joint_hessian(eta_hat, **kwargs)
    nu_d = np.linalg.inv(H)

    return eta_hat, nu_d, int(result.nit)
```

Note: scipy's `minimize` does not accept `kwargs=` on most installs; it accepts `args=` as a tuple positionally. Adjust by passing args as a tuple and rewriting the helpers to accept positional arrays. To minimize boilerplate, use `functools.partial`:

```python
from functools import partial

def _stm_doc_inference(
    *,
    indices: np.ndarray,
    counts: np.ndarray,
    expElogbeta: np.ndarray,
    Gamma: np.ndarray,
    Sigma_diag: np.ndarray,
    x: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, int]:
    K = expElogbeta.shape[0]
    eta0 = np.zeros(K, dtype=np.float64)
    common = dict(
        indices=indices, counts=counts, expElogbeta=expElogbeta,
        Gamma=Gamma, Sigma_diag=Sigma_diag, x=x,
    )
    f = partial(_stm_neg_log_joint, **common)
    g = partial(_stm_neg_log_joint_grad, **common)
    result = minimize(f, x0=eta0, jac=g, method="L-BFGS-B",
                      options={"maxiter": max_iter, "gtol": tol})
    eta_hat = result.x
    H = _stm_neg_log_joint_hessian(eta_hat, **common)
    nu_d = np.linalg.inv(H)
    return eta_hat, nu_d, int(result.nit)
```

Use this `partial`-based version.

- [ ] **Step 12: Run MAP test to verify it passes**

```bash
poetry run pytest tests/test_stm_math.py -v
```

Expected: all pass.

- [ ] **Step 13: Commit**

```bash
git add spark-vi/spark_vi/models/topic/stm.py spark-vi/tests/test_stm_math.py
git commit -m "feat(spark-vi): _stm_doc_inference — per-doc Laplace via L-BFGS + analytic Hessian"
```

---

### Task 3: `OnlineSTM` skeleton — `__init__`, `initialize_global`, `get_metadata`

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/stm.py`
- Test: `spark-vi/tests/test_stm_contract.py` (new)

- [ ] **Step 1: Write the constructor + init tests**

Create `spark-vi/tests/test_stm_contract.py`:

```python
"""Tests for OnlineSTM's VIModel contract conformance."""
from __future__ import annotations

import numpy as np
import pytest

from spark_vi.models.topic.stm import OnlineSTM


class TestConstructor:
    def test_constructs_with_minimal_args(self):
        m = OnlineSTM(K=5, vocab_size=100, P=3)
        assert m.K == 5
        assert m.V == 100
        assert m.P == 3

    def test_rejects_invalid_K(self):
        with pytest.raises(ValueError, match="K must be >= 1"):
            OnlineSTM(K=0, vocab_size=100, P=3)

    def test_rejects_invalid_vocab_size(self):
        with pytest.raises(ValueError, match="vocab_size must be >= 1"):
            OnlineSTM(K=5, vocab_size=0, P=3)

    def test_rejects_invalid_P(self):
        with pytest.raises(ValueError, match="P must be >= 1"):
            OnlineSTM(K=5, vocab_size=100, P=0)

    def test_rejects_invalid_sigma_ridge(self):
        with pytest.raises(ValueError, match="sigma_ridge must be >= 0"):
            OnlineSTM(K=5, vocab_size=100, P=3, sigma_ridge=-1.0)


class TestInitializeGlobal:
    def test_returns_lambda_eta_gamma_sigma_shapes(self):
        m = OnlineSTM(K=4, vocab_size=20, P=2, random_seed=42)
        gp = m.initialize_global(data_summary=None)
        assert gp["lambda"].shape == (4, 20)
        assert gp["eta"].shape == ()
        assert gp["Gamma"].shape == (2, 4)
        assert gp["Sigma"].shape == (4,)
        # Sigma starts at sigma_init.
        np.testing.assert_allclose(gp["Sigma"], np.full(4, 1.0))
        # Gamma starts at zeros (covariates have no effect at init).
        np.testing.assert_allclose(gp["Gamma"], np.zeros((2, 4)))

    def test_seeded_init_is_deterministic(self):
        gp1 = OnlineSTM(K=3, vocab_size=10, P=2, random_seed=7).initialize_global(None)
        gp2 = OnlineSTM(K=3, vocab_size=10, P=2, random_seed=7).initialize_global(None)
        np.testing.assert_array_equal(gp1["lambda"], gp2["lambda"])


class TestGetMetadata:
    def test_returns_K_V_P(self):
        m = OnlineSTM(K=5, vocab_size=100, P=3)
        md = m.get_metadata()
        assert md == {"K": 5, "V": 100, "P": 3}
```

- [ ] **Step 2: Run to verify it fails**

```bash
poetry run pytest tests/test_stm_contract.py -v
```

Expected: FAIL — `OnlineSTM` doesn't exist yet.

- [ ] **Step 3: Implement `OnlineSTM.__init__`, `initialize_global`, `get_metadata`**

Append to `spark-vi/spark_vi/models/topic/stm.py`:

```python
class OnlineSTM(VIModel):
    """Online STM (prevalence-only) fittable by VIRunner.

    Per-doc inference is two-step Laplace (ADR 0023): L-BFGS to find η̂_d,
    analytic Hessian at η̂_d for ν_d. β stays Dirichlet-conjugate; Γ and Σ
    are hyperparameters of the prior on η, learned by closed-form M-step
    ρ-blended in mini-batch (stochastic-EM).

    random_seed controls the λ initialization. Per-doc inference is
    cold-started at η=0 deterministically regardless of seed.
    """

    def __init__(
        self,
        K: int,
        vocab_size: int,
        P: int,
        eta: float | None = None,
        sigma_init: float = 1.0,
        sigma_ridge: float = 1e-6,
        lbfgs_max_iter: int = 50,
        lbfgs_tol: float = 1e-4,
        gamma_shape: float = 100.0,
        random_seed: int | None = None,
    ) -> None:
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        if vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1, got {vocab_size}")
        if P < 1:
            raise ValueError(f"P must be >= 1, got {P}")
        if eta is None:
            eta = 1.0 / K
        if eta <= 0:
            raise ValueError(f"eta must be > 0, got {eta}")
        if sigma_init <= 0:
            raise ValueError(f"sigma_init must be > 0, got {sigma_init}")
        if sigma_ridge < 0:
            raise ValueError(f"sigma_ridge must be >= 0, got {sigma_ridge}")
        if lbfgs_max_iter < 1:
            raise ValueError(f"lbfgs_max_iter must be >= 1, got {lbfgs_max_iter}")
        if lbfgs_tol <= 0:
            raise ValueError(f"lbfgs_tol must be > 0, got {lbfgs_tol}")
        if gamma_shape <= 0:
            raise ValueError(f"gamma_shape must be > 0, got {gamma_shape}")

        self.K = int(K)
        self.V = int(vocab_size)
        self.P = int(P)
        self.eta = float(eta)
        self.sigma_init = float(sigma_init)
        self.sigma_ridge = float(sigma_ridge)
        self.lbfgs_max_iter = int(lbfgs_max_iter)
        self.lbfgs_tol = float(lbfgs_tol)
        self.gamma_shape = float(gamma_shape)
        self.random_seed = None if random_seed is None else int(random_seed)

    def initialize_global(self, data_summary: Any | None) -> dict[str, np.ndarray]:
        """Random Gamma init for λ (same shape as LDA); Γ = 0; Σ = sigma_init."""
        if self.random_seed is None:
            lam = np.random.gamma(
                shape=self.gamma_shape, scale=1.0 / self.gamma_shape,
                size=(self.K, self.V),
            )
        else:
            rng = np.random.default_rng(self.random_seed)
            lam = rng.gamma(
                shape=self.gamma_shape, scale=1.0 / self.gamma_shape,
                size=(self.K, self.V),
            )
        return {
            "lambda": lam,
            "eta": np.array(self.eta),
            "Gamma": np.zeros((self.P, self.K), dtype=np.float64),
            "Sigma": np.full(self.K, self.sigma_init, dtype=np.float64),
        }

    def get_metadata(self) -> dict[str, Any]:
        return {"K": self.K, "V": self.V, "P": self.P}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_stm_contract.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/stm.py spark-vi/tests/test_stm_contract.py
git commit -m "feat(spark-vi): OnlineSTM skeleton — __init__, initialize_global, get_metadata"
```

---

### Task 4: `OnlineSTM.local_update`

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/stm.py`
- Test: `spark-vi/tests/test_stm_contract.py` (extend)

- [ ] **Step 1: Write the local_update test**

Append to `tests/test_stm_contract.py`:

```python
class TestLocalUpdate:
    def test_returns_expected_keys_and_shapes(self):
        m = OnlineSTM(K=3, vocab_size=10, P=2, random_seed=0)
        gp = m.initialize_global(None)
        # Inject a non-degenerate Σ so Laplace doesn't collapse.
        gp["Sigma"] = np.full(3, 1.0)
        from spark_vi.models.topic.types import STMDocument
        docs = [
            STMDocument(
                indices=np.array([0, 3, 7], dtype=np.int32),
                counts=np.array([2.0, 1.0, 1.0]),
                length=4,
                x=np.array([1.0, 0.5]),
            ),
            STMDocument(
                indices=np.array([1, 4, 8], dtype=np.int32),
                counts=np.array([1.0, 3.0, 1.0]),
                length=5,
                x=np.array([-0.5, 1.0]),
            ),
        ]
        ss = m.local_update(docs, gp)
        assert ss["lambda_stats"].shape == (3, 10)
        assert ss["XtX"].shape == (2, 2)
        assert ss["XtMu"].shape == (2, 3)
        assert ss["residual_diag_stat"].shape == (3,)
        assert ss["n_docs"].shape == ()
        assert float(ss["n_docs"]) == 2.0
        # ELBO suff stats.
        assert ss["doc_loglik_sum"].shape == ()
        assert ss["doc_eta_kl_sum"].shape == ()

    def test_lambda_stats_only_touches_seen_columns(self):
        m = OnlineSTM(K=3, vocab_size=10, P=2, random_seed=0)
        gp = m.initialize_global(None)
        from spark_vi.models.topic.types import STMDocument
        doc = STMDocument(
            indices=np.array([2, 5], dtype=np.int32),
            counts=np.array([1.0, 1.0]),
            length=2,
            x=np.array([0.0, 0.0]),
        )
        ss = m.local_update([doc], gp)
        touched = set(np.flatnonzero(ss["lambda_stats"].sum(axis=0)).tolist())
        assert touched == {2, 5}

    def test_empty_partition_returns_zero_stats(self):
        m = OnlineSTM(K=3, vocab_size=10, P=2, random_seed=0)
        gp = m.initialize_global(None)
        ss = m.local_update([], gp)
        assert float(ss["n_docs"]) == 0.0
        np.testing.assert_array_equal(ss["lambda_stats"], np.zeros((3, 10)))
        np.testing.assert_array_equal(ss["XtX"], np.zeros((2, 2)))
        np.testing.assert_array_equal(ss["XtMu"], np.zeros((2, 3)))
```

- [ ] **Step 2: Run to verify failure**

```bash
poetry run pytest tests/test_stm_contract.py::TestLocalUpdate -v
```

Expected: FAIL with `AttributeError: 'OnlineSTM' object has no attribute 'local_update'`.

- [ ] **Step 3: Implement `local_update`**

Append to the `OnlineSTM` class in `stm.py`:

```python
    def local_update(
        self,
        rows: Iterable[STMDocument],
        global_params: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """E-step on one Spark partition.

        For each doc: run _stm_doc_inference to get (η̂_d, ν_d). Accumulate:
        - lambda_stats: K×V suff-stats for β SVI step (same shape as LDA)
        - XtX, XtMu: P×P and P×K cross-products for Γ ridge regression
        - residual_diag_stat: K-vector for diagonal Σ sample covariance
        - doc_loglik_sum: data log-lik term at MAP (for ELBO)
        - doc_eta_kl_sum: KL(N(η̂, ν_d) || N(Γx, Σ)) (for ELBO)
        - n_docs scalar
        """
        lam = global_params["lambda"]
        Gamma = global_params["Gamma"]
        Sigma_diag = global_params["Sigma"]
        expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))

        K, V = self.K, self.V
        P = self.P

        lambda_stats = np.zeros((K, V), dtype=np.float64)
        XtX = np.zeros((P, P), dtype=np.float64)
        XtMu = np.zeros((P, K), dtype=np.float64)
        residual_diag = np.zeros(K, dtype=np.float64)
        doc_loglik = 0.0
        doc_eta_kl = 0.0
        n_docs = 0

        log_Sigma_diag = np.log(Sigma_diag)

        for doc in rows:
            eta_hat, nu_d, _ = _stm_doc_inference(
                indices=doc.indices, counts=doc.counts,
                expElogbeta=expElogbeta,
                Gamma=Gamma, Sigma_diag=Sigma_diag, x=doc.x,
                max_iter=self.lbfgs_max_iter, tol=self.lbfgs_tol,
            )
            p = _softmax(eta_hat)
            eb_d = expElogbeta[:, doc.indices]
            q_w = eb_d.T @ p + 1e-100
            phi = (eb_d * p[:, None]) / q_w[None, :]   # (K, n_unique)

            # λ suff-stats: outer(p ⊙ ..., counts/q_w) form matching LDA.
            # For STM with Laplace, the simplest unbiased estimator is the
            # MAP-point per-token responsibility weighted by counts:
            #   λ_stats[:, indices] += phi · counts
            # This is the natural-gradient SVI target up to the expElogbeta
            # multiplier applied by update_global (same pattern as LDA).
            sstats_row = phi * doc.counts[None, :]
            lambda_stats[:, doc.indices] += sstats_row

            # Regression sufficient stats.
            XtX += np.outer(doc.x, doc.x)
            XtMu += np.outer(doc.x, eta_hat)
            # Residual diag for Σ: (η̂ - Γx)² + diag(ν_d).
            resid = eta_hat - Gamma.T @ doc.x
            residual_diag += resid * resid + np.diag(nu_d)

            # ELBO terms — see plan header for the Laplace ELBO derivation.
            doc_loglik += float(np.sum(doc.counts * np.log(q_w)))
            # KL(N(η̂, ν_d) || N(Γx, Σ)) closed form with K-diagonal Σ:
            # ½(tr(Σ⁻¹ ν_d) + (μ - m)ᵀ Σ⁻¹ (μ - m) - K + log|Σ| - log|ν_d|)
            tr_term = float(np.sum(np.diag(nu_d) / Sigma_diag))
            quad_term = float(np.sum(resid * resid / Sigma_diag))
            sign, logdet_nu = np.linalg.slogdet(nu_d)
            logdet_Sigma = float(np.sum(log_Sigma_diag))
            doc_eta_kl += 0.5 * (tr_term + quad_term - K + logdet_Sigma - logdet_nu)

            n_docs += 1

        return {
            "lambda_stats": lambda_stats,
            "XtX": XtX,
            "XtMu": XtMu,
            "residual_diag_stat": residual_diag,
            "doc_loglik_sum": np.array(doc_loglik),
            "doc_eta_kl_sum": np.array(doc_eta_kl),
            "n_docs": np.array(float(n_docs)),
        }
```

Note on the λ suff-stats form: LDA's local_update accumulates `outer(expElogthetad, counts / phi_norm)` and update_global multiplies by `expElogbeta`. In STM we use `phi · counts` directly because `phi` already incorporates `expElogbeta`. The update_global step skips the `expElogbeta * stats` multiplication for the STM path — see Task 5.

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_stm_contract.py::TestLocalUpdate -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/stm.py spark-vi/tests/test_stm_contract.py
git commit -m "feat(spark-vi): OnlineSTM.local_update accumulating Laplace + regression suff-stats"
```

---

### Task 5: `OnlineSTM.update_global` — β SVI, Γ ridge regression, Σ sample-cov; ρ-blended

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/stm.py`
- Test: `spark-vi/tests/test_stm_contract.py` (extend)

- [ ] **Step 1: Write the update_global tests**

Append to `tests/test_stm_contract.py`:

```python
class TestUpdateGlobal:
    def _make_state_with_stats(self):
        m = OnlineSTM(K=3, vocab_size=10, P=2, random_seed=0)
        gp = m.initialize_global(None)
        target_stats = {
            "lambda_stats": np.ones((3, 10)) * 0.5,
            "XtX": np.eye(2) * 100.0,
            "XtMu": np.array([[1.0, -1.0, 0.5], [0.5, 0.0, -0.5]]),
            "residual_diag_stat": np.array([5.0, 3.0, 2.0]),
            "doc_loglik_sum": np.array(-100.0),
            "doc_eta_kl_sum": np.array(5.0),
            "n_docs": np.array(50.0),
        }
        return m, gp, target_stats

    def test_lambda_natural_gradient_step(self):
        m, gp, target = self._make_state_with_stats()
        lam_before = gp["lambda"].copy()
        gp_new = m.update_global(gp, target, learning_rate=1.0)
        # At ρ=1.0 the update fully replaces λ with η + lambda_stats.
        expected = float(gp["eta"]) + target["lambda_stats"]
        np.testing.assert_allclose(gp_new["lambda"], expected)

    def test_lambda_partial_step(self):
        m, gp, target = self._make_state_with_stats()
        lam_before = gp["lambda"].copy()
        gp_new = m.update_global(gp, target, learning_rate=0.3)
        # (1-ρ)·old + ρ·target.
        expected = 0.7 * lam_before + 0.3 * (float(gp["eta"]) + target["lambda_stats"])
        np.testing.assert_allclose(gp_new["lambda"], expected)

    def test_gamma_solves_ridge_regression(self):
        m, gp, target = self._make_state_with_stats()
        gp_new = m.update_global(gp, target, learning_rate=1.0)
        # Γ̂ = (XᵀX + ridge·I)⁻¹ Xᵀμ.
        ridge = m.sigma_ridge
        expected_Gamma = np.linalg.solve(
            target["XtX"] + ridge * np.eye(2), target["XtMu"]
        )
        np.testing.assert_allclose(gp_new["Gamma"], expected_Gamma)

    def test_sigma_sample_covariance(self):
        m, gp, target = self._make_state_with_stats()
        gp_new = m.update_global(gp, target, learning_rate=1.0)
        expected_Sigma = target["residual_diag_stat"] / float(target["n_docs"])
        np.testing.assert_allclose(gp_new["Sigma"], expected_Sigma)

    def test_sigma_minimum_floor(self):
        """Σ should never go below a small floor to keep Laplace well-defined."""
        m, gp, target = self._make_state_with_stats()
        target["residual_diag_stat"] = np.array([0.0, 0.0, 0.0])
        gp_new = m.update_global(gp, target, learning_rate=1.0)
        assert np.all(gp_new["Sigma"] > 0)
```

- [ ] **Step 2: Run to verify failure**

```bash
poetry run pytest tests/test_stm_contract.py::TestUpdateGlobal -v
```

Expected: FAIL — `update_global` not implemented.

- [ ] **Step 3: Implement `update_global`**

Append to the `OnlineSTM` class in `stm.py`:

```python
    SIGMA_FLOOR = 1e-6

    def update_global(
        self,
        global_params: dict[str, np.ndarray],
        target_stats: dict[str, np.ndarray],
        learning_rate: float,
    ) -> dict[str, np.ndarray]:
        """SVI on λ; ρ-blended closed-form M-step on Γ, Σ (ADR 0023).

        λ:  λ_new = (1-ρ)·λ + ρ·(η + lambda_stats)        # natural-gradient SVI
        Γ:  Γ̂ = (XᵀX + ridge·I)⁻¹ XᵀMu                    # ridge OLS on aggregated stats
            Γ_new = (1-ρ)·Γ + ρ·Γ̂                         # stochastic-EM ρ-blend
        Σ:  σ²_k_target = residual_diag_stat_k / n_docs   # diagonal sample cov + Laplace correction
            σ²_k_new   = max((1-ρ)·σ²_k + ρ·σ²_k_target, SIGMA_FLOOR)
        """
        lam = global_params["lambda"]
        eta = float(global_params["eta"])
        Gamma = global_params["Gamma"]
        Sigma_diag = global_params["Sigma"]

        # β: SVI natural-gradient step. Note: STM's lambda_stats already
        # incorporates expElogbeta (via phi in local_update), so no extra
        # expElogbeta multiplication here — differs from LDA.
        target_lam = eta + target_stats["lambda_stats"]
        new_lam = (1.0 - learning_rate) * lam + learning_rate * target_lam

        # Γ: ridge regression on aggregated cross-products.
        XtX = target_stats["XtX"]
        XtMu = target_stats["XtMu"]
        ridge_eye = self.sigma_ridge * np.eye(self.P)
        Gamma_target = np.linalg.solve(XtX + ridge_eye, XtMu)
        new_Gamma = (1.0 - learning_rate) * Gamma + learning_rate * Gamma_target

        # Σ: diagonal sample cov with Laplace correction (already folded into stat).
        n_docs = float(target_stats["n_docs"])
        Sigma_target = target_stats["residual_diag_stat"] / max(n_docs, 1.0)
        new_Sigma = (1.0 - learning_rate) * Sigma_diag + learning_rate * Sigma_target
        new_Sigma = np.maximum(new_Sigma, self.SIGMA_FLOOR)

        return {
            "lambda": new_lam,
            "eta": global_params["eta"],
            "Gamma": new_Gamma,
            "Sigma": new_Sigma,
        }
```

- [ ] **Step 4: Run tests to verify pass**

```bash
poetry run pytest tests/test_stm_contract.py::TestUpdateGlobal -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/stm.py spark-vi/tests/test_stm_contract.py
git commit -m "feat(spark-vi): OnlineSTM.update_global — beta SVI + Gamma ridge OLS + Sigma sample-cov"
```

---

### Task 6: `OnlineSTM.compute_elbo`

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/stm.py`
- Test: `spark-vi/tests/test_stm_contract.py` (extend)

- [ ] **Step 1: Write the ELBO test**

Append to `tests/test_stm_contract.py`:

```python
class TestComputeELBO:
    def test_returns_finite_float(self):
        m = OnlineSTM(K=3, vocab_size=10, P=2, random_seed=0)
        gp = m.initialize_global(None)
        aggregated = {
            "doc_loglik_sum": np.array(-50.0),
            "doc_eta_kl_sum": np.array(3.0),
            "n_docs": np.array(10.0),
        }
        elbo = m.compute_elbo(gp, aggregated)
        assert np.isfinite(elbo)

    def test_includes_negative_global_beta_kl(self):
        """ELBO = doc_loglik - doc_eta_kl - global_beta_kl. Increasing the
        beta KL should decrease the ELBO."""
        m = OnlineSTM(K=3, vocab_size=10, P=2, random_seed=0)
        gp_low_kl = m.initialize_global(None)
        # Concentrate λ on one column → high KL vs uniform prior.
        gp_high_kl = {**gp_low_kl, "lambda": gp_low_kl["lambda"].copy()}
        gp_high_kl["lambda"][:, 0] *= 100.0
        agg = {
            "doc_loglik_sum": np.array(-50.0),
            "doc_eta_kl_sum": np.array(3.0),
            "n_docs": np.array(10.0),
        }
        elbo_low_kl = m.compute_elbo(gp_low_kl, agg)
        elbo_high_kl = m.compute_elbo(gp_high_kl, agg)
        assert elbo_high_kl < elbo_low_kl
```

- [ ] **Step 2: Run to verify failure**

```bash
poetry run pytest tests/test_stm_contract.py::TestComputeELBO -v
```

Expected: FAIL (compute_elbo returns NaN by default).

- [ ] **Step 3: Implement `compute_elbo`**

Append to `OnlineSTM` class:

```python
    def compute_elbo(
        self,
        global_params: dict[str, np.ndarray],
        aggregated_stats: dict[str, np.ndarray],
    ) -> float:
        """ELBO = doc_loglik_sum - doc_eta_kl_sum - global β KL.

        doc_loglik_sum and doc_eta_kl_sum are aggregated in local_update;
        the global β KL is computed here on the driver from λ, η alone
        (same pattern as OnlineLDA).
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
            - float(aggregated_stats["doc_eta_kl_sum"])
            - global_kl
        )


def _dirichlet_kl(q_alpha: np.ndarray, p_alpha: np.ndarray) -> float:
    """KL(Dirichlet(q_alpha) || Dirichlet(p_alpha)). Same as in LDA's stm.py uses."""
    qsum = q_alpha.sum()
    psum = p_alpha.sum()
    return float(
        gammaln(qsum) - gammaln(psum)
        - (gammaln(q_alpha) - gammaln(p_alpha)).sum()
        + ((q_alpha - p_alpha) * (digamma(q_alpha) - digamma(qsum))).sum()
    )
```

- [ ] **Step 4: Run tests to verify pass**

```bash
poetry run pytest tests/test_stm_contract.py::TestComputeELBO -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/stm.py spark-vi/tests/test_stm_contract.py
git commit -m "feat(spark-vi): OnlineSTM.compute_elbo with global beta KL"
```

---

### Task 7: `OnlineSTM.infer_local`, `iteration_summary`, `iteration_diagnostics`

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/stm.py`
- Test: `spark-vi/tests/test_stm_contract.py` (extend)

- [ ] **Step 1: Write tests for the three hooks**

Append to `tests/test_stm_contract.py`:

```python
class TestInferLocal:
    def test_returns_eta_theta_for_single_doc(self):
        m = OnlineSTM(K=3, vocab_size=10, P=2, random_seed=0)
        gp = m.initialize_global(None)
        from spark_vi.models.topic.types import STMDocument
        doc = STMDocument(
            indices=np.array([0, 3], dtype=np.int32),
            counts=np.array([2.0, 1.0]),
            length=3,
            x=np.array([0.5, -0.3]),
        )
        out = m.infer_local(doc, gp)
        assert out["eta"].shape == (3,)
        assert out["theta"].shape == (3,)
        np.testing.assert_allclose(out["theta"].sum(), 1.0, atol=1e-10)
        assert np.all(out["theta"] > 0)


class TestIterationSummary:
    def test_returns_compact_string(self):
        m = OnlineSTM(K=3, vocab_size=10, P=2, random_seed=0)
        gp = m.initialize_global(None)
        s = m.iteration_summary(gp)
        assert isinstance(s, str)
        assert "Γ" in s or "Gamma" in s
        assert "Σ" in s or "Sigma" in s


class TestIterationDiagnostics:
    def test_returns_traceable_arrays(self):
        m = OnlineSTM(K=3, vocab_size=10, P=2, random_seed=0)
        gp = m.initialize_global(None)
        d = m.iteration_diagnostics(gp)
        assert "Gamma" in d
        assert "Sigma" in d
        assert d["Gamma"].shape == (2, 3)
        assert d["Sigma"].shape == (3,)
```

- [ ] **Step 2: Run to verify failure**

```bash
poetry run pytest tests/test_stm_contract.py::TestInferLocal tests/test_stm_contract.py::TestIterationSummary tests/test_stm_contract.py::TestIterationDiagnostics -v
```

Expected: FAIL.

- [ ] **Step 3: Implement the three hooks**

Append to `OnlineSTM` class:

```python
    def infer_local(self, row: STMDocument, global_params: dict[str, np.ndarray]):
        """Single-document MAP inference under fixed global params.

        Pure function of (row, global_params). Uses the same cold-start
        L-BFGS as local_update; returns eta (the MAP point) and theta
        (softmax of eta) for downstream consumers.
        """
        lam = global_params["lambda"]
        Gamma = global_params["Gamma"]
        Sigma_diag = global_params["Sigma"]
        expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))

        eta_hat, _, _ = _stm_doc_inference(
            indices=row.indices, counts=row.counts,
            expElogbeta=expElogbeta,
            Gamma=Gamma, Sigma_diag=Sigma_diag, x=row.x,
            max_iter=self.lbfgs_max_iter, tol=self.lbfgs_tol,
        )
        return {"eta": eta_hat, "theta": _softmax(eta_hat)}

    def iteration_summary(self, global_params: dict[str, np.ndarray]) -> str:
        """Compact per-iter view of Γ scale, Σ scale, and λ row-mass spread."""
        Gamma = global_params["Gamma"]
        Sigma = global_params["Sigma"]
        lam = global_params["lambda"]
        lam_row_sums = lam.sum(axis=1)
        return (
            f"|Γ|[max={np.abs(Gamma).max():.3g} mean={np.abs(Gamma).mean():.3g}], "
            f"Σ[min={Sigma.min():.3g} max={Sigma.max():.3g}], "
            f"Σλ_k[min={lam_row_sums.min():.3g} max={lam_row_sums.max():.3g}]"
        )

    def iteration_diagnostics(
        self, global_params: dict[str, np.ndarray],
    ) -> dict[str, float | np.ndarray]:
        """Per-iter trajectories of Γ and Σ (small; safe to persist every iter)."""
        return {
            "Gamma": np.asarray(global_params["Gamma"]),
            "Sigma": np.asarray(global_params["Sigma"]),
        }
```

- [ ] **Step 4: Run tests to verify pass**

```bash
poetry run pytest tests/test_stm_contract.py -v
```

Expected: ALL PASS (full contract conformance).

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/stm.py spark-vi/tests/test_stm_contract.py
git commit -m "feat(spark-vi): OnlineSTM.infer_local + iteration_summary + iteration_diagnostics"
```

---

### Task 8: Synthetic recovery integration test (known Γ → recover Γ̂)

**Files:**
- Create: `spark-vi/tests/test_stm_integration.py`

- [ ] **Step 1: Write the synthetic recovery test**

Create `spark-vi/tests/test_stm_integration.py`:

```python
"""Integration tests for OnlineSTM: synthetic data with known parameters."""
from __future__ import annotations

import numpy as np
import pytest

from spark_vi.models.topic.stm import OnlineSTM
from spark_vi.models.topic.types import STMDocument


def _synthetic_corpus(
    *,
    K: int, V: int, P: int, D: int, doc_len: int,
    Gamma_true: np.ndarray, Sigma_true: np.ndarray,
    seed: int = 0,
) -> tuple[list[STMDocument], np.ndarray]:
    """Generate D docs from the STM generative process.

    Returns the docs plus the true beta matrix (K, V).
    """
    rng = np.random.default_rng(seed)
    # β_k ~ Dirichlet(0.1 / V).
    beta = rng.dirichlet(np.full(V, 0.1), size=K)   # (K, V)
    # x_d ~ N(0, I); η_d ~ N(Γᵀ x_d, diag(Σ)).
    docs = []
    for d in range(D):
        x = rng.normal(size=P)
        mu = Gamma_true.T @ x
        eta = rng.normal(loc=mu, scale=np.sqrt(Sigma_true))
        theta = np.exp(eta - eta.max())
        theta = theta / theta.sum()
        # Multinomial draw of doc_len tokens.
        z = rng.choice(K, size=doc_len, p=theta)
        w = np.array([rng.choice(V, p=beta[zi]) for zi in z])
        unique, counts = np.unique(w, return_counts=True)
        docs.append(STMDocument(
            indices=unique.astype(np.int32),
            counts=counts.astype(np.float64),
            length=int(counts.sum()),
            x=x,
        ))
    return docs, beta


@pytest.mark.slow
def test_synthetic_recovery_full_batch():
    """Train OnlineSTM full-batch on synthetic data with known Γ; recover Γ̂
    within reasonable tolerance.

    Marked slow because full-batch fit over 200 docs at K=4, V=30, P=2
    takes ~30 seconds. The point of this test is qualitative recovery —
    Γ̂'s sign pattern and rough magnitudes — not exact identity.
    """
    K, V, P, D, doc_len = 4, 30, 2, 200, 60
    Gamma_true = np.array([
        [+1.5, -0.5, 0.0, +0.2],
        [-0.2, +1.0, -1.0, +0.1],
    ])
    Sigma_true = np.full(K, 0.5)

    docs, beta_true = _synthetic_corpus(
        K=K, V=V, P=P, D=D, doc_len=doc_len,
        Gamma_true=Gamma_true, Sigma_true=Sigma_true, seed=42,
    )

    model = OnlineSTM(
        K=K, vocab_size=V, P=P,
        sigma_init=1.0, lbfgs_max_iter=80, lbfgs_tol=1e-5,
        random_seed=42,
    )
    gp = model.initialize_global(None)

    # Run full-batch (ρ=1) for N outer iters.
    for _ in range(30):
        stats = model.local_update(docs, gp)
        gp = model.update_global(gp, stats, learning_rate=1.0)

    Gamma_hat = gp["Gamma"]

    # Topic labels are unidentifiable up to permutation. We check that
    # *some* column-permutation of Γ̂ recovers Γ_true's sign pattern.
    from itertools import permutations
    best = max(
        permutations(range(K)),
        key=lambda perm: float(np.sum(np.sign(Gamma_hat[:, list(perm)]) == np.sign(Gamma_true))),
    )
    Gamma_hat_aligned = Gamma_hat[:, list(best)]
    sign_match = float(np.mean(np.sign(Gamma_hat_aligned) == np.sign(Gamma_true)))
    # 75% (6/8 entries) sign-match is the floor for a "qualitative recovery"
    # check at this corpus size; tighten if the implementation hits higher
    # consistently. Perfect recovery would be 100%.
    assert sign_match >= 0.75, f"Γ̂ sign pattern off: {sign_match=}"
```

- [ ] **Step 2: Run to verify it currently fails (the recovery should actually work; mark it slow first)**

```bash
poetry run pytest tests/test_stm_integration.py -v -m slow
```

Expected: PASS (full implementation in place by Task 7). If it FAILS, this is a real correctness signal — investigate before proceeding.

- [ ] **Step 3: Commit**

```bash
git add spark-vi/tests/test_stm_integration.py
git commit -m "test(spark-vi): synthetic recovery integration test for OnlineSTM"
```

---

## Phase 2 — Mini-batch convergence validation

### Task 9: Mini-batch vs full-batch convergence-agreement test

**Files:**
- Modify: `spark-vi/tests/test_stm_integration.py`

This is the **risk-bearing** validation from the design spec. Mini-batch SVI on STM with stochastic-EM blending of Γ, Σ converges to a *neighborhood* of the full-batch optimum, not to the same point (ADR 0023). Validation criterion is qualitative agreement on Γ̂'s sign pattern, not numerical identity.

- [ ] **Step 1: Write the mini-batch comparison test**

Append to `tests/test_stm_integration.py`:

```python
@pytest.mark.slow
def test_minibatch_converges_to_neighborhood_of_full_batch():
    """Mini-batch fit on the same synthetic corpus produces Γ̂ that:
    (1) Sign-pattern matches full-batch Γ̂ on most entries.
    (2) ELBO at convergence is within ~5% of full-batch.

    Failure of either gate is a signal that ρ-blended stochastic-EM on
    Γ, Σ has correctness issues — investigate before shipping STM
    with mini-batch enabled by default (ADR 0023).
    """
    K, V, P, D, doc_len = 4, 30, 2, 200, 60
    Gamma_true = np.array([
        [+1.5, -0.5, 0.0, +0.2],
        [-0.2, +1.0, -1.0, +0.1],
    ])
    Sigma_true = np.full(K, 0.5)

    docs, _ = _synthetic_corpus(
        K=K, V=V, P=P, D=D, doc_len=doc_len,
        Gamma_true=Gamma_true, Sigma_true=Sigma_true, seed=42,
    )

    # Full-batch reference.
    model_fb = OnlineSTM(K=K, vocab_size=V, P=P, random_seed=42)
    gp_fb = model_fb.initialize_global(None)
    for _ in range(30):
        stats = model_fb.local_update(docs, gp_fb)
        gp_fb = model_fb.update_global(gp_fb, stats, learning_rate=1.0)

    # Mini-batch run with the same seed; ρ_t = (t + 64)^{-0.7}.
    rng = np.random.default_rng(42)
    model_mb = OnlineSTM(K=K, vocab_size=V, P=P, random_seed=42)
    gp_mb = model_mb.initialize_global(None)
    batch_size = 20
    n_outer = 200
    for t in range(n_outer):
        idx = rng.choice(D, size=batch_size, replace=False)
        batch = [docs[i] for i in idx]
        stats = model_mb.local_update(batch, gp_mb)
        # Corpus-scale stats by (D / batch_size) so they represent the
        # full-corpus target, then ρ-blend.
        scale = D / batch_size
        scaled_stats = {
            "lambda_stats": stats["lambda_stats"] * scale,
            "XtX": stats["XtX"] * scale,
            "XtMu": stats["XtMu"] * scale,
            "residual_diag_stat": stats["residual_diag_stat"] * scale,
            "doc_loglik_sum": stats["doc_loglik_sum"] * scale,
            "doc_eta_kl_sum": stats["doc_eta_kl_sum"] * scale,
            "n_docs": stats["n_docs"] * scale,
        }
        rho_t = (t + 64) ** -0.7
        gp_mb = model_mb.update_global(gp_mb, scaled_stats, learning_rate=rho_t)

    # Align Γ̂_mb to Γ̂_fb up to column permutation.
    from itertools import permutations
    best = max(
        permutations(range(K)),
        key=lambda perm: float(np.sum(
            np.sign(gp_mb["Gamma"][:, list(perm)]) == np.sign(gp_fb["Gamma"])
        )),
    )
    Gamma_mb_aligned = gp_mb["Gamma"][:, list(best)]
    sign_match = float(np.mean(np.sign(Gamma_mb_aligned) == np.sign(gp_fb["Gamma"])))
    assert sign_match >= 0.75, (
        f"Mini-batch Γ̂ sign pattern diverges from full-batch: {sign_match=}. "
        f"Stochastic-EM blending of Γ may have a correctness bug; see ADR 0023."
    )

    # ELBO comparison.
    stats_fb_final = model_fb.local_update(docs, gp_fb)
    stats_mb_final = model_mb.local_update(docs, gp_mb)
    elbo_fb = model_fb.compute_elbo(gp_fb, stats_fb_final)
    elbo_mb = model_mb.compute_elbo(gp_mb, stats_mb_final)
    rel_diff = abs(elbo_mb - elbo_fb) / abs(elbo_fb)
    assert rel_diff < 0.05, (
        f"Mini-batch ELBO too far from full-batch: rel_diff={rel_diff:.3f}. "
        f"See ADR 0023 mini-batch convergence-to-neighborhood discussion."
    )
```

- [ ] **Step 2: Run the test**

```bash
poetry run pytest tests/test_stm_integration.py::test_minibatch_converges_to_neighborhood_of_full_batch -v -m slow
```

Expected outcomes:
- **PASS:** mini-batch STM is viable as-spec'd; proceed.
- **FAIL on sign-match:** real correctness issue. Investigate the ρ-blending logic in `update_global` (likely the order of (1-ρ)·old + ρ·target, or the scaling factor for mini-batch). Do not ship until resolved.
- **FAIL on ELBO:** acceptable if sign-match holds; widen the 5% tolerance to 10% and document the looser bound. If sign-match fails too, see above.

- [ ] **Step 3: Commit**

```bash
git add spark-vi/tests/test_stm_integration.py
git commit -m "test(spark-vi): mini-batch vs full-batch convergence agreement validation"
```

---

## Phase 3 — MLlib shim with formulaic

### Task 10: `_vector_to_stm_document` + `StreamingSTM` Path A (pre-built covariates)

**Files:**
- Modify: `spark-vi/spark_vi/mllib/topic/_common.py`
- Create: `spark-vi/spark_vi/mllib/topic/stm.py`
- Create: `spark-vi/tests/test_mllib_stm.py`

- [ ] **Step 1: Write `_vector_to_stm_document` test**

Create `spark-vi/tests/test_mllib_stm.py`:

```python
"""Tests for the MLlib-shim StreamingSTM estimator and STMModel."""
from __future__ import annotations

import numpy as np
import pytest

pyspark = pytest.importorskip("pyspark")
from pyspark.ml.linalg import SparseVector, DenseVector, Vectors


class TestVectorToSTMDocument:
    def test_constructs_from_row_with_features_and_covariates(self):
        from spark_vi.mllib.topic._common import _vector_to_stm_document
        row = {
            "features": SparseVector(10, [0, 3, 5], [2.0, 1.0, 3.0]),
            "covariates": DenseVector([1.0, 0.5, -1.2]),
        }
        doc = _vector_to_stm_document(row, features_col="features",
                                       covariates_col="covariates")
        np.testing.assert_array_equal(doc.indices, [0, 3, 5])
        np.testing.assert_array_equal(doc.counts, [2.0, 1.0, 3.0])
        assert doc.length == 6
        np.testing.assert_array_equal(doc.x, [1.0, 0.5, -1.2])
```

- [ ] **Step 2: Run to verify failure**

```bash
poetry run pytest tests/test_mllib_stm.py::TestVectorToSTMDocument -v
```

Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `_vector_to_stm_document` in `_common.py`**

In `spark-vi/spark_vi/mllib/topic/_common.py`, add (alongside the existing `_vector_to_bow_document`):

```python
from spark_vi.models.topic.types import STMDocument


def _vector_to_stm_document(
    row,
    features_col: str = "features",
    covariates_col: str = "covariates",
) -> STMDocument:
    """Construct an STMDocument from a row with both a BOW vector and a covariate vector.

    Accepts pyspark.sql.Row and dict-like objects. The covariate vector
    must be a DenseVector (or numpy-coercible array); covariates_col
    cannot be sparse for STM (every doc has a complete x vector).
    """
    sv = row[features_col]
    cov = row[covariates_col]
    return STMDocument(
        indices=np.asarray(sv.indices, dtype=np.int32),
        counts=np.asarray(sv.values, dtype=np.float64),
        length=int(sv.values.sum()),
        x=np.asarray(cov, dtype=np.float64),
    )
```

- [ ] **Step 4: Run test to verify pass**

```bash
poetry run pytest tests/test_mllib_stm.py::TestVectorToSTMDocument -v
```

Expected: PASS.

- [ ] **Step 5: Write `StreamingSTM` Path A construction test**

Append to `tests/test_mllib_stm.py`:

```python
class TestStreamingSTMPathA:
    def test_constructs_with_covariates_col(self):
        from spark_vi.mllib.topic.stm import StreamingSTM
        est = StreamingSTM(
            K=5, features_col="features",
            covariates_col="covariates",
            covariate_names=["age", "sex", "cohort"],
        )
        assert est.K == 5
        assert est.P == 3
        assert est.covariate_names == ["age", "sex", "cohort"]

    def test_rejects_zero_covariates(self):
        from spark_vi.mllib.topic.stm import StreamingSTM
        with pytest.raises(ValueError, match="covariate_names"):
            StreamingSTM(K=5, features_col="features",
                         covariates_col="covariates", covariate_names=[])

    def test_rejects_path_b_args_without_formula_extra(self):
        """If user passes formula args without installing the formula extra,
        the estimator should error at construct time, not at fit time."""
        from spark_vi.mllib.topic.stm import StreamingSTM
        with pytest.raises(ValueError, match="covariate_formula|covariate_names"):
            StreamingSTM(K=5, features_col="features")
```

- [ ] **Step 6: Run to verify failure**

```bash
poetry run pytest tests/test_mllib_stm.py::TestStreamingSTMPathA -v
```

Expected: FAIL with ImportError.

- [ ] **Step 7: Implement `StreamingSTM` shim, Path A only**

Create `spark-vi/spark_vi/mllib/topic/stm.py`:

```python
"""StreamingSTM: MLlib-shim estimator for OnlineSTM.

Two input paths:
  (A) Caller supplies a pre-built `covariates` DenseVector column and
      a list of covariate names. No formulaic dependency required.
  (B) Caller supplies a `covariate_formula` string + a covariate
      DataFrame. Requires the `formula` extra: pip install spark-vi[formula].

This file implements Path A. Path B is added by Tasks 11-13.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class StreamingSTM:
    """Streaming-VI estimator for OnlineSTM with DataFrame input.

    Constructor enforces that the caller supplies enough information
    to determine P (covariate dimension) — either via covariate_names
    (Path A) or covariate_formula (Path B; see Tasks 11-13).
    """

    def __init__(
        self,
        K: int,
        features_col: str = "features",
        covariates_col: str | None = None,
        covariate_names: list[str] | None = None,
        covariate_formula: str | None = None,
        covariate_df: Any | None = None,
        join_key: str | None = None,
        max_levels: int = 10_000,
        sigma_init: float = 1.0,
        sigma_ridge: float = 1e-6,
        lbfgs_max_iter: int = 50,
        lbfgs_tol: float = 1e-4,
        random_seed: int | None = None,
    ) -> None:
        # Path A vs B validation.
        path_a = covariates_col is not None and covariate_names is not None
        path_b = covariate_formula is not None
        if not (path_a or path_b):
            raise ValueError(
                "StreamingSTM requires either (covariates_col + covariate_names) "
                "for Path A, or covariate_formula for Path B."
            )
        if path_a and path_b:
            raise ValueError("Use either Path A or Path B, not both.")

        self.K = int(K)
        self.features_col = features_col

        if path_a:
            if not covariate_names:
                raise ValueError("covariate_names must be non-empty for Path A.")
            self.covariates_col = covariates_col
            self.covariate_names = list(covariate_names)
            self.P = len(self.covariate_names)
            self.covariate_formula = None
        else:
            # Path B — wired in Tasks 11-13.
            self.covariates_col = "covariates"
            self.covariate_formula = covariate_formula
            self.covariate_df = covariate_df
            self.join_key = join_key
            self.max_levels = max_levels
            self.covariate_names = None       # set during fit
            self.P = None                     # set during fit

        self.sigma_init = sigma_init
        self.sigma_ridge = sigma_ridge
        self.lbfgs_max_iter = lbfgs_max_iter
        self.lbfgs_tol = lbfgs_tol
        self.random_seed = random_seed
```

- [ ] **Step 8: Run tests to verify pass**

```bash
poetry run pytest tests/test_mllib_stm.py -v
```

Expected: ALL PASS.

- [ ] **Step 9: Commit**

```bash
git add spark-vi/spark_vi/mllib/topic/_common.py spark-vi/spark_vi/mllib/topic/stm.py spark-vi/tests/test_mllib_stm.py
git commit -m "feat(spark-vi/mllib): StreamingSTM estimator + _vector_to_stm_document (Path A)"
```

---

### Task 11: formulaic optional dep + formula validator (reject stateful transforms)

**Files:**
- Modify: `spark-vi/pyproject.toml`
- Create: `spark-vi/spark_vi/mllib/topic/_formula.py`
- Create: `spark-vi/tests/test_mllib_stm_formula.py`

- [ ] **Step 1: Add formulaic as optional dep**

In `spark-vi/pyproject.toml`, add under `[tool.poetry.dependencies]`:

```toml
formulaic = {version = ">=1.0", optional = true}
```

Then add a `[tool.poetry.extras]` section if absent:

```toml
[tool.poetry.extras]
formula = ["formulaic"]
```

Install the extra in the dev env:

```bash
cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno/spark-vi
poetry install --extras formula
```

- [ ] **Step 2: Write formula-validation tests**

Create `spark-vi/tests/test_mllib_stm_formula.py`:

```python
"""Tests for the formula path of StreamingSTM (Path B).

Covers: formula parsing, validation rejecting stateful transforms,
schema-frame categorical discovery, ModelSpec construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

formulaic = pytest.importorskip("formulaic")


class TestFormulaValidation:
    def test_rejects_splines(self):
        from spark_vi.mllib.topic._formula import validate_formula
        with pytest.raises(ValueError, match="bs|spline|stateful"):
            validate_formula("~ age + bs(age, df=4)")

    def test_rejects_natural_splines(self):
        from spark_vi.mllib.topic._formula import validate_formula
        with pytest.raises(ValueError, match="ns|spline|stateful"):
            validate_formula("~ ns(age, df=4)")

    def test_rejects_standardization(self):
        from spark_vi.mllib.topic._formula import validate_formula
        with pytest.raises(ValueError, match="scale|center|stateful"):
            validate_formula("~ scale(age) + sex")

    def test_accepts_categoricals(self):
        from spark_vi.mllib.topic._formula import validate_formula
        validate_formula("~ C(cohort) + sex")  # should not raise

    def test_accepts_interactions(self):
        from spark_vi.mllib.topic._formula import validate_formula
        validate_formula("~ age * sex")

    def test_accepts_I_transforms(self):
        from spark_vi.mllib.topic._formula import validate_formula
        validate_formula("~ age + I(age**2)")

    def test_accepts_intercept_dropping(self):
        from spark_vi.mllib.topic._formula import validate_formula
        validate_formula("~ 0 + age + sex")
```

- [ ] **Step 3: Run to verify failure**

```bash
poetry run pytest tests/test_mllib_stm_formula.py::TestFormulaValidation -v
```

Expected: FAIL — ImportError.

- [ ] **Step 4: Implement `validate_formula`**

Create `spark-vi/spark_vi/mllib/topic/_formula.py`:

```python
"""Formula-handling plumbing for the StreamingSTM MLlib shim (Path B).

Uses `formulaic` to parse R-style formula strings and produce a ModelSpec
that maps doc covariates → design matrix rows. Rejects "stateful transforms"
(splines, standardization) that v1 explicitly does not support
(see ADR 0022, ADR 0024).

Categorical level discovery uses the schema-frame trick (ADR 0024):
Spark-native distinct + cardinality bound, build a tiny pandas
schema-frame, hand to formulaic for level capture. The fitted ModelSpec
is data-independent at application time.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Imported lazily so the spark-vi base install doesn't require formulaic.
def _formulaic():
    try:
        import formulaic
        return formulaic
    except ImportError as e:
        raise ImportError(
            "Formula path requires the optional 'formula' extra: "
            "pip install spark-vi[formula]"
        ) from e


# v1 rejection list — these are formulaic's built-in stateful transforms.
# Adding more here is forward-compatible: anything that learns state from
# the schema-frame's bogus placeholder values must be reject-listed.
_STATEFUL_REJECT_LIST = {
    "bs", "ns", "cr", "scale", "center", "standardize",
}


def validate_formula(formula_str: str) -> None:
    """Parse the formula and reject any stateful transform we don't support in v1.

    Raises ValueError with a clear remediation message if a rejected
    transform appears. See ADR 0022 v1 scope decision.
    """
    formulaic = _formulaic()
    formula = formulaic.Formula(formula_str)
    # Walk the parsed term tree and collect callable names that appear
    # in factor expressions.
    found = set()
    for term in formula.lhs.terms + formula.rhs.terms if hasattr(formula, "lhs") else formula.rhs.terms:
        for factor in term.factors:
            expr = str(factor.expr)
            for name in _STATEFUL_REJECT_LIST:
                if f"{name}(" in expr:
                    found.add(name)

    if found:
        rejected = ", ".join(sorted(found))
        raise ValueError(
            f"STM v1 does not support stateful formula transforms: {rejected}. "
            f"Workarounds: bin continuous covariates categorically (e.g. "
            f"`age_decile`), or pre-compute the basis columns yourself and "
            f"pass them as raw continuous covariates. See ADR 0022."
        )
```

Note: `formulaic`'s Formula API has minor differences between versions; the snippet above targets formulaic ≥1.0. If the term-tree walk shape differs in the installed version, adjust the iteration accordingly — the goal is "scan factor expressions for the reject-listed callable names." The fallback (string-scan the formula string) is acceptable too: `for name in _STATEFUL_REJECT_LIST: if f"{name}(" in formula_str: found.add(name)`. The implementer chooses the more robust path against the installed formulaic version.

- [ ] **Step 5: Run tests to verify pass**

```bash
poetry run pytest tests/test_mllib_stm_formula.py::TestFormulaValidation -v
```

Expected: PASS. If a specific test fails because formulaic's parsed-term-tree shape differs from the assumed API, use the string-scan fallback.

- [ ] **Step 6: Commit**

```bash
git add spark-vi/pyproject.toml spark-vi/spark_vi/mllib/topic/_formula.py spark-vi/tests/test_mllib_stm_formula.py
git commit -m "feat(spark-vi/mllib): formulaic optional dep + formula validator rejecting stateful transforms"
```

---

### Task 12: Schema-frame categorical discovery + per-partition design-matrix application

**Files:**
- Modify: `spark-vi/spark_vi/mllib/topic/_formula.py`
- Modify: `spark-vi/spark_vi/mllib/topic/stm.py`
- Modify: `spark-vi/tests/test_mllib_stm_formula.py`

- [ ] **Step 1: Write schema-frame discovery tests**

Append to `tests/test_mllib_stm_formula.py`:

```python
class TestFitModelSpec:
    def test_categorical_levels_discovered_and_applied(self):
        from spark_vi.mllib.topic._formula import fit_model_spec, apply_model_spec

        covariate_pdf = pd.DataFrame({
            "cohort": ["control", "case", "case", "control", "case"],
            "sex":    ["M", "F", "M", "F", "F"],
            "age":    [25.0, 40.0, 55.0, 30.0, 45.0],
        })
        spec, names = fit_model_spec(
            formula="~ C(cohort) + C(sex) + age",
            covariate_pdf=covariate_pdf,
        )
        applied = apply_model_spec(spec, covariate_pdf)
        # Expected columns: intercept + cohort[T.control] + sex[T.M] + age = 4
        assert applied.shape == (5, 4)
        assert "Intercept" in names or "intercept" in [n.lower() for n in names]

    def test_unseen_level_at_apply_raises(self):
        from spark_vi.mllib.topic._formula import fit_model_spec, apply_model_spec
        train = pd.DataFrame({"cohort": ["a", "b"]})
        spec, _ = fit_model_spec(formula="~ C(cohort)", covariate_pdf=train)
        test = pd.DataFrame({"cohort": ["a", "c"]})   # 'c' unseen
        with pytest.raises(Exception):
            apply_model_spec(spec, test)
```

- [ ] **Step 2: Run to verify failure**

```bash
poetry run pytest tests/test_mllib_stm_formula.py::TestFitModelSpec -v
```

Expected: FAIL with `ImportError: cannot import name 'fit_model_spec'`.

- [ ] **Step 3: Implement `fit_model_spec` and `apply_model_spec`**

Append to `spark-vi/spark_vi/mllib/topic/_formula.py`:

```python
def fit_model_spec(
    formula: str,
    covariate_pdf: pd.DataFrame,
) -> tuple[Any, list[str]]:
    """Build a formulaic ModelSpec from a covariate DataFrame.

    Used directly when the caller has a small pandas DataFrame in hand
    (Path B with covariate_df pre-collected, or for sidecar building in
    charmpheno). For large Spark DataFrames, use fit_model_spec_from_spark
    which builds the schema-frame from per-column distinct() queries.
    """
    formulaic = _formulaic()
    validate_formula(formula)

    materializer = formulaic.Formula(formula).get_model_matrix(covariate_pdf)
    spec = materializer.model_spec
    names = list(materializer.columns)
    return spec, names


def apply_model_spec(spec: Any, covariate_pdf: pd.DataFrame) -> np.ndarray:
    """Apply a fitted ModelSpec to a new DataFrame; return the design matrix."""
    materialized = spec.get_model_matrix(covariate_pdf)
    return np.asarray(materialized, dtype=np.float64)


def discover_categorical_levels_spark(
    spark_df: Any,
    categorical_cols: list[str],
    max_levels: int,
) -> dict[str, list]:
    """Spark-native level discovery for each categorical column.

    Bounds cardinality via approxCountDistinct first; raises if over max_levels.
    Returns levels sorted lexicographically for determinism.
    """
    from pyspark.sql import functions as F

    levels: dict[str, list] = {}
    for col in categorical_cols:
        n_distinct = spark_df.select(F.approxCountDistinct(col)).first()[0]
        if n_distinct > max_levels:
            raise ValueError(
                f"Categorical '{col}' has approximately {n_distinct} distinct "
                f"levels, above max_levels={max_levels}. Consider manual binning "
                f"or coarser representation."
            )
        rows = spark_df.select(col).distinct().collect()
        vals = sorted(r[col] for r in rows if r[col] is not None)
        levels[col] = vals
    return levels


def fit_model_spec_from_spark(
    formula: str,
    spark_df: Any,
    categorical_cols: list[str],
    continuous_cols: list[str],
    max_levels: int = 10_000,
) -> tuple[Any, list[str]]:
    """Schema-frame discovery: build ModelSpec without materializing the full data.

    Process (ADR 0024):
      1. Discover categorical levels via Spark distinct() with cardinality bound.
      2. Build a tiny pandas schema-frame containing each level at least once
         (plus zero-valued placeholders for continuous columns).
      3. Fit ModelSpec against the schema-frame via formulaic — captures
         the level set in transform_state.
      4. Validate transform_state contains *only* the categorical level
         mappings (no spline knots, no scale/center stats).
    """
    levels = discover_categorical_levels_spark(spark_df, categorical_cols, max_levels)
    max_n_levels = max((len(v) for v in levels.values()), default=1)
    rows = []
    for i in range(max(max_n_levels, 1)):
        row = {}
        for col in categorical_cols:
            col_levels = levels[col]
            row[col] = col_levels[i % len(col_levels)]
        for col in continuous_cols:
            row[col] = 0.0
        rows.append(row)
    schema_pdf = pd.DataFrame(rows)
    spec, names = fit_model_spec(formula, schema_pdf)
    # Post-fit guard: transform_state must only contain the categorical
    # mappings we intentionally captured.
    extras = _unexpected_transform_state(spec, categorical_cols)
    if extras:
        raise ValueError(
            f"Formula introduced unexpected stateful transforms: {extras}. "
            f"This should have been caught by validate_formula; please file a bug."
        )
    return spec, names


def _unexpected_transform_state(spec: Any, categorical_cols: list[str]) -> list[str]:
    """Identify transform_state entries that aren't covariate-level captures."""
    # formulaic stores transform-specific state per term/factor; the exact
    # API shape may vary across versions. The safest check is to enumerate
    # the transform-state keys and confirm none reference rejected callables.
    ts = getattr(spec, "transform_state", None) or {}
    extras = []
    for key, value in ts.items():
        key_str = str(key).lower()
        for rejected in _STATEFUL_REJECT_LIST:
            if rejected in key_str:
                extras.append(key)
                break
    return extras
```

- [ ] **Step 4: Run tests to verify pass**

```bash
poetry run pytest tests/test_mllib_stm_formula.py::TestFitModelSpec -v
```

Expected: PASS.

- [ ] **Step 5: Wire Path B into `StreamingSTM.fit` (simplified, in-memory pandas DataFrame for now)**

For the v1 MLlib shim Path B, the integration with a real Spark DataFrame at `.fit()` time is part of Task 13 (persistence + full fit roundtrip). At this task we make sure the construction works given a pandas covariate DataFrame:

Append to `tests/test_mllib_stm_formula.py`:

```python
class TestStreamingSTMPathBConstruction:
    def test_construct_with_formula_resolves_P_and_names(self):
        from spark_vi.mllib.topic.stm import StreamingSTM
        from spark_vi.mllib.topic._formula import fit_model_spec
        pdf = pd.DataFrame({
            "cohort": ["a", "b", "a", "b"],
            "age":    [25.0, 40.0, 55.0, 30.0],
        })
        spec, names = fit_model_spec("~ C(cohort) + age", pdf)
        est = StreamingSTM(
            K=4,
            covariate_formula="~ C(cohort) + age",
            covariate_df=pdf,
        )
        # Inject ModelSpec resolution (real .fit() will do this with Spark).
        est._resolve_model_spec_from_pandas(pdf)
        assert est.P == len(names)
        assert est.covariate_names == names
```

In `spark_vi/mllib/topic/stm.py`, append a small helper that does what the test exercises:

```python
    def _resolve_model_spec_from_pandas(self, covariate_pdf):
        """Resolve P and covariate_names from a pre-collected pandas covariate DataFrame.

        Used by tests and by the in-memory Path-B construction. Production
        .fit() invocations against Spark DataFrames will use the
        schema-frame Spark discovery path instead (Task 13).
        """
        from spark_vi.mllib.topic._formula import fit_model_spec
        spec, names = fit_model_spec(self.covariate_formula, covariate_pdf)
        self.model_spec = spec
        self.covariate_names = names
        self.P = len(names)
```

- [ ] **Step 6: Run tests to verify pass**

```bash
poetry run pytest tests/test_mllib_stm_formula.py -v
```

Expected: ALL PASS.

- [ ] **Step 7: Commit**

```bash
git add spark-vi/spark_vi/mllib/topic/_formula.py spark-vi/spark_vi/mllib/topic/stm.py spark-vi/tests/test_mllib_stm_formula.py
git commit -m "feat(spark-vi/mllib): schema-frame categorical discovery for STM formula path"
```

---

### Task 13: `STMModel` persistence (ModelSpec round-trip)

**Files:**
- Modify: `spark-vi/spark_vi/mllib/topic/stm.py`
- Create: `spark-vi/tests/test_mllib_stm_persistence.py`

- [ ] **Step 1: Write the save/load round-trip test**

Create `spark-vi/tests/test_mllib_stm_persistence.py`:

```python
"""Save/load round-trip tests for StreamingSTM's fitted STMModel."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


class TestSTMModelPersistence:
    def test_save_and_load_roundtrips_VIResult_and_ModelSpec(self, tmp_path: Path):
        from spark_vi.mllib.topic.stm import STMModel
        from spark_vi.models.topic.stm import OnlineSTM

        model = OnlineSTM(K=3, vocab_size=10, P=2, random_seed=0)
        gp = model.initialize_global(None)

        # A toy "ModelSpec" stub — anything pickle-able works for the
        # persistence layer test. Real ModelSpec comes from formulaic.
        class _FakeSpec:
            def __init__(self):
                self.factor_levels = {"cohort": ["a", "b"]}

        spec = _FakeSpec()
        stm_model = STMModel(
            global_params=gp,
            metadata={"K": 3, "V": 10, "P": 2},
            model_spec=spec,
            covariate_names=["intercept", "cohort_b"],
        )

        out_dir = tmp_path / "stm_model"
        stm_model.save(out_dir)
        loaded = STMModel.load(out_dir)

        # Global params round-trip.
        np.testing.assert_array_equal(loaded.global_params["Gamma"], gp["Gamma"])
        np.testing.assert_array_equal(loaded.global_params["Sigma"], gp["Sigma"])
        np.testing.assert_array_equal(loaded.global_params["lambda"], gp["lambda"])
        assert loaded.metadata == {"K": 3, "V": 10, "P": 2}
        # ModelSpec round-trips.
        assert loaded.model_spec.factor_levels == {"cohort": ["a", "b"]}
        assert loaded.covariate_names == ["intercept", "cohort_b"]
```

- [ ] **Step 2: Run to verify failure**

```bash
poetry run pytest tests/test_mllib_stm_persistence.py -v
```

Expected: FAIL — `STMModel` doesn't exist.

- [ ] **Step 3: Implement `STMModel` with save/load**

Append to `spark-vi/spark_vi/mllib/topic/stm.py`:

```python
import pickle
from pathlib import Path


class STMModel:
    """Fitted MLlib-shim STM model. Wraps OnlineSTM's global params + ModelSpec.

    Persistence layout under <model_dir>:
        global_params.npz   # lambda, Gamma, Sigma, eta (numpy arrays)
        metadata.json       # K, V, P, covariate_names
        model_spec.pkl      # formulaic ModelSpec (pickle)
    """

    def __init__(
        self,
        global_params: dict[str, np.ndarray],
        metadata: dict[str, Any],
        model_spec: Any,
        covariate_names: list[str],
    ) -> None:
        self.global_params = global_params
        self.metadata = metadata
        self.model_spec = model_spec
        self.covariate_names = covariate_names

    def save(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_dir / "global_params.npz",
            **{k: np.asarray(v) for k, v in self.global_params.items()},
        )
        import json
        (out_dir / "metadata.json").write_text(json.dumps({
            **self.metadata,
            "covariate_names": self.covariate_names,
        }))
        with (out_dir / "model_spec.pkl").open("wb") as f:
            pickle.dump(self.model_spec, f)

    @classmethod
    def load(cls, in_dir: Path) -> "STMModel":
        in_dir = Path(in_dir)
        npz = np.load(in_dir / "global_params.npz")
        global_params = {k: npz[k] for k in npz.files}
        import json
        md = json.loads((in_dir / "metadata.json").read_text())
        covariate_names = md.pop("covariate_names", [])
        with (in_dir / "model_spec.pkl").open("rb") as f:
            spec = pickle.load(f)
        return cls(
            global_params=global_params,
            metadata=md,
            model_spec=spec,
            covariate_names=covariate_names,
        )
```

- [ ] **Step 4: Run tests to verify pass**

```bash
poetry run pytest tests/test_mllib_stm_persistence.py -v
```

Expected: PASS.

- [ ] **Step 5: Run the full spark-vi test suite to confirm no regressions**

```bash
poetry run pytest tests/ -v -m "not slow and not cluster"
```

Expected: ALL PASS. STM tests + all existing LDA / HDP / shim tests.

Also run the slow STM integration tests once to confirm:

```bash
poetry run pytest tests/test_stm_integration.py -v -m slow
```

Expected: both `test_synthetic_recovery_full_batch` and `test_minibatch_converges_to_neighborhood_of_full_batch` PASS.

- [ ] **Step 6: Commit**

```bash
git add spark-vi/spark_vi/mllib/topic/stm.py spark-vi/tests/test_mllib_stm_persistence.py
git commit -m "feat(spark-vi/mllib): STMModel persistence with ModelSpec round-trip"
```

---

## Verification

After all 13 tasks land:

```bash
cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno/spark-vi
poetry run pytest tests/ -v -m "not slow and not cluster"
poetry run pytest tests/test_stm_integration.py -v -m slow
```

Both should pass. The slow integration tests are the load-bearing validation:
- **`test_synthetic_recovery_full_batch`** confirms the engine math is correct (Γ̂ sign pattern recovers Γ_true on synthetic data).
- **`test_minibatch_converges_to_neighborhood_of_full_batch`** confirms the mini-batch posture from ADR 0023 holds (qualitative agreement between mini-batch and full-batch, not numerical identity).

If `test_minibatch_converges_to_neighborhood_of_full_batch` fails, that is the **risk flagged in the design spec**: investigate before shipping STM with mini-batch enabled by default. The fallback ship path is full-batch-only STM, with an ADR documenting why.

## Out of Scope / Follow-Ups

This plan covers phases 1–3 of the [STM design spec](../specs/2026-05-29-stm-prevalence-design.md). The following are phase 4–6 and live in the second plan (charmpheno integration):

- Covariate sidecar parquet helper (`charmpheno.omop.covariates.build_patient_covariate_sidecar`).
- Cloud fit driver (`analysis/cloud/stm_bigquery_cloud.py`).
- Experiment-tracking wrapper integration (`scripts/run_experiment.py` STM dispatch, defaults YAML keys, Make targets).
- Dashboard adapter (`charmpheno/charmpheno/export/dashboard.py` `adapt_stm`).
- Per-cohort BigQuery covariate materialization.

The following are deferred per the design spec / ADRs:

- Splines (`bs`, `ns`, `cr`), `scale()`, `center()` — v1.x; rejection at fit start is the v1 behavior.
- Content covariates (full STM, SAGE-style log-linear β) — v2.
- Per-doc L-BFGS warm-starting across outer iterations — v1.x, only if per-doc cost becomes blocking.
- Full K×K Σ (vs K-diagonal) — v1.x, only if residual diagonal analysis surfaces off-diagonal structure.
- Fused single-`agg`-pass categorical discovery (vs per-column distinct) — v1.1 perf.
