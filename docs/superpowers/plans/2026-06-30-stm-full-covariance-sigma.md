# Full-covariance Σ for STM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace OnlineSTM's diagonal (mean-field) logistic-normal covariance with a full (K−1)×(K−1) Σ — the CTM treatment (Blei & Lafferty 2007) — so the model captures and exposes topic correlations.

**Architecture:** The per-doc Laplace machinery already computes full K×K Hessians and a full ν_d; only the prior term, the Σ representation/M-step, the ELBO KL, and storage are diagonal. We cut those over to a full SPD matrix Σ, generalize the lazy block M-step (ADR 0027) to a per-pair scatter + support matrix, add the inverse-Wishart + diagonal-shrink regularizers, and a min_pair_support floor. Spec: [docs/superpowers/specs/2026-06-30-stm-full-covariance-sigma-design.md](../specs/2026-06-30-stm-full-covariance-sigma-design.md).

**Tech Stack:** Python 3.12, NumPy/SciPy, PySpark (Spark 3.5), pytest. Engine in `spark-vi/spark_vi/models/topic/stm.py`; MLlib shim in `spark-vi/spark_vi/mllib/topic/stm.py`; drivers in `analysis/`.

## Global Constraints

- spark-vi stays DOMAIN-AGNOSTIC: integer token ids only, never OMOP/EHR vocabulary terms in the library.
- NO LaTeX in comments/docstrings/docs — plain text + Unicode Greek (Σ η μ Γ ν ρ ψ β λ) only.
- No personal info in committed artifacts; markdown-linkable code refs in docs.
- CITE literature in docstrings for any method/default from the literature (inverse-Wishart → Blei & Lafferty 2007; diagonal-shrink → Roberts et al. stm).
- REPLACE the diagonal path entirely — no `covariance_type` toggle, no backward-compat / no promote-on-load shim. Legacy diagonal checkpoints do not reload.
- The reference topic is pinned at η=0 and carries NO Σ row/column; Σ is over the K−1 free topics (or all K when `reference_topic=False`). "Free topics" below means the prior-side topics `allowed_free` already computed in `local_update` ([stm.py:531-534](../../../spark-vi/spark_vi/models/topic/stm.py#L531-L534)).
- TDD: every behavior gets a test you watched fail first. Run engine tests with `cd spark-vi && python -m pytest`; shim/driver tests need `PYTHONPATH="charmpheno:spark-vi:$PYTHONPATH"` from repo root. The venv already has `setuptools<74` + `formulaic`.
- Σ regularizer parameterization (so the scalar knobs stay consistent with today's inverse-gamma): the inverse-Wishart scale is Ψ = `sigma_prior_count`·`sigma_prior_scale`·I and dof ν = `sigma_prior_count`; the per-entry MAP is Σ_ij = (S_ij + Ψ_ij) / (N_ij + ν). Diagonal Ψ_kk = ν·`sigma_prior_scale` gives prior mean `sigma_prior_scale`; off-diagonal Ψ_ij = 0 gives prior mean 0 — matching today's diagonal inverse-gamma exactly in the diagonal case.

---

## File Structure

- **`spark-vi/spark_vi/models/topic/stm.py`** (modify) — the engine. Per-doc functions take a precision matrix; Σ becomes a matrix; M-step, ELBO, init change; new SPD/correlation helpers.
- **`spark-vi/spark_vi/models/topic/_linalg.py`** (create) — small, pure SPD helpers: `nearest_spd`, `safe_inverse`, `topic_correlation`. Isolated so they unit-test without Spark.
- **`spark-vi/spark_vi/mllib/topic/stm.py`** (modify) — `StreamingSTM`: new params, persist Σ matrix + correlation R + support N, metadata.
- **`analysis/cloud/stm_bigquery_cloud.py`**, **`analysis/local/fit_stm_local.py`**, **`scripts/run_experiment.py`** (modify) — thread the new knobs (mirror the existing `sigma_prior_*` / `--spectral-*` threading).
- **`spark-vi/tests/test_stm_full_sigma.py`**, **`spark-vi/tests/test_linalg.py`** (create) — engine + helper tests.
- **`docs/decisions/`, `docs/insights/`, `docs/experiments/`** (create) — ADR, insight, two experiment docs.

Existing tests that assert the diagonal representation (`test_stm_reference.py`, `test_stm_contract.py`, `test_mllib_stm.py`, `test_stm_integration.py`) will need updates as Σ changes shape; each task names the ones it touches.

---

## Task 1: Per-doc prior takes a full precision matrix

Switch the three per-doc pure functions from a diagonal `Sigma_diag` (K-vector) to a full precision matrix `Sigma_inv` (n_sub×n_sub), so the Gaussian prior is (η−μ)ᵀ Σ⁻¹ (η−μ). Keep callers byte-identical by passing `np.diag(1.0/Sigma_diag)` for now (the storage cut-over is Task 2).

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/stm.py` — `_stm_neg_log_joint` ([108-131](../../../spark-vi/spark_vi/models/topic/stm.py#L108-L131)), `_stm_neg_log_joint_grad` ([134-158](../../../spark-vi/spark_vi/models/topic/stm.py#L134-L158)), `_stm_neg_log_joint_hessian` ([161-192](../../../spark-vi/spark_vi/models/topic/stm.py#L161-L192)), and the single `common` dict they are called through in `_stm_doc_inference` ([266-269](../../../spark-vi/spark_vi/models/topic/stm.py#L266-L269)). `local_update` is NOT touched this task — it still passes `Sigma_diag` into `_stm_doc_inference`, which converts internally; that outer signature changes in Task 3.
- Test: `spark-vi/tests/test_stm_full_sigma.py` (create).

**Interfaces:**
- Produces: `_stm_neg_log_joint(eta, *, indices, counts, expElogbeta, Gamma, Sigma_inv, x)`, same for `_grad` and `_hessian`; `Sigma_inv` is an (n,n) precision matrix where n = len(eta). Prior term = 0.5·diffᵀ·Sigma_inv·diff, grad += Sigma_inv·diff, Hessian += Sigma_inv.

- [ ] **Step 1: Write the failing test** (`spark-vi/tests/test_stm_full_sigma.py`)

```python
import numpy as np
from spark_vi.models.topic.stm import (
    _stm_neg_log_joint, _stm_neg_log_joint_grad, _stm_neg_log_joint_hessian,
)

def _toy():
    rng = np.random.default_rng(0)
    K, Vn = 3, 4
    eta = rng.normal(size=K)
    indices = np.array([0, 1, 2, 3], dtype=np.int32)
    counts = np.array([2.0, 1.0, 3.0, 1.0])
    expElogbeta = np.abs(rng.normal(size=(K, Vn))) + 0.1
    Gamma = rng.normal(size=(2, K))
    x = np.array([1.0, 0.5])
    return dict(eta=eta, indices=indices, counts=counts,
               expElogbeta=expElogbeta, Gamma=Gamma, x=x), K

def test_full_precision_prior_matches_diagonal_special_case():
    common, K = _toy()
    eta = common.pop("eta")
    sigma_diag = np.array([1.5, 2.0, 0.5])
    Sigma_inv = np.diag(1.0 / sigma_diag)
    # Prior term with full precision == diagonal hand-computation.
    diff = eta - common["Gamma"].T @ common["x"]
    expected_prior = 0.5 * float(diff @ Sigma_inv @ diff)
    f = _stm_neg_log_joint(eta, Sigma_inv=Sigma_inv, **common)
    # Recompute the data term alone by zeroing the prior (Sigma_inv = 0).
    data_only = _stm_neg_log_joint(eta, Sigma_inv=np.zeros((K, K)), **common)
    assert np.isclose(f - data_only, expected_prior)

def test_full_precision_grad_and_hessian_use_Sigma_inv():
    common, K = _toy()
    eta = common.pop("eta")
    Sigma_inv = np.array([[2.0, 0.3, 0.0],
                          [0.3, 1.0, 0.1],
                          [0.0, 0.1, 4.0]])
    diff = eta - common["Gamma"].T @ common["x"]
    g = _stm_neg_log_joint_grad(eta, Sigma_inv=Sigma_inv, **common)
    g_data = _stm_neg_log_joint_grad(eta, Sigma_inv=np.zeros((K, K)), **common)
    assert np.allclose(g - g_data, Sigma_inv @ diff)
    H = _stm_neg_log_joint_hessian(eta, Sigma_inv=Sigma_inv, **common)
    H_data = _stm_neg_log_joint_hessian(eta, Sigma_inv=np.zeros((K, K)), **common)
    assert np.allclose(H - H_data, Sigma_inv)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd spark-vi && python -m pytest tests/test_stm_full_sigma.py -v`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'Sigma_inv'`.

- [ ] **Step 3: Implement** — in the three functions, rename the `Sigma_diag` parameter to `Sigma_inv` and change the prior math:
  - `_stm_neg_log_joint`: replace `prior_term = 0.5 * float(np.sum(diff * diff / Sigma_diag))` with `prior_term = 0.5 * float(diff @ Sigma_inv @ diff)`. Update the docstring prior line and remove the "diagonal Σ" wording.
  - `_stm_neg_log_joint_grad`: replace `prior_grad = diff / Sigma_diag` with `prior_grad = Sigma_inv @ diff`.
  - `_stm_neg_log_joint_hessian`: replace `H_prior = np.diag(1.0 / Sigma_diag)` with `H_prior = Sigma_inv`.
  - In `_stm_doc_inference`, the `common` dict ([266-269](../../../spark-vi/spark_vi/models/topic/stm.py#L266-L269)) currently passes `Sigma_diag=sub_Sigma` (where `sub_Sigma = Sigma_diag[allowed]`); change that one line to `Sigma_inv=np.diag(1.0 / sub_Sigma)` so the per-doc behavior is byte-identical this task. `_stm_doc_inference`'s own outer `Sigma_diag` parameter is unchanged here (Task 3 converts it).

- [ ] **Step 4: Run to verify it passes**

Run: `cd spark-vi && python -m pytest tests/test_stm_full_sigma.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Run the existing engine suite to confirm byte-identity**

Run: `cd spark-vi && python -m pytest tests/test_stm_reference.py tests/test_stm_contract.py tests/test_stm_integration.py -q`
Expected: PASS (unchanged — the diagonal-precision passed in keeps results identical).

- [ ] **Step 6: Commit**

```bash
git add spark-vi/spark_vi/models/topic/stm.py spark-vi/tests/test_stm_full_sigma.py
git commit -m "refactor(stm): per-doc prior takes full precision matrix Sigma_inv"
```

---

## Task 2: SPD/correlation helpers in a new `_linalg` module

Create the small, pure linear-algebra helpers the engine needs: a Cholesky-based inverse with eigenvalue-floor repair, a nearest-SPD projection (for the gated-assembly case), and the correlation extraction. Isolated so they test without Spark and don't bloat `stm.py`.

**Files:**
- Create: `spark-vi/spark_vi/models/topic/_linalg.py`
- Test: `spark-vi/tests/test_linalg.py` (create)

**Interfaces:**
- Produces: `safe_inverse(M, cond_cap=1e-10) -> np.ndarray` (SPD inverse, eigenvalue-floor repair if not PD); `nearest_spd(M, floor=1e-8) -> np.ndarray` (symmetrize + floor eigenvalues to `floor`); `topic_correlation(Sigma) -> np.ndarray` (R_ij = Σ_ij/sqrt(Σ_ii·Σ_jj), diagonal exactly 1).

- [ ] **Step 1: Write the failing test** (`spark-vi/tests/test_linalg.py`)

```python
import numpy as np
import pytest
from spark_vi.models.topic._linalg import safe_inverse, nearest_spd, topic_correlation

def test_safe_inverse_matches_inv_for_spd():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(5, 5)); M = A @ A.T + np.eye(5)
    assert np.allclose(safe_inverse(M), np.linalg.inv(M))

def test_safe_inverse_repairs_indefinite():
    M = np.diag([1.0, -2.0, 3.0])  # indefinite
    inv = safe_inverse(M)
    w = np.linalg.eigvalsh(inv)
    assert np.all(w > 0)  # SPD result despite indefinite input

def test_nearest_spd_floors_eigenvalues_and_symmetrizes():
    M = np.array([[1.0, 0.9, 0.9],
                  [0.9, 1.0, -0.9],
                  [0.9, -0.9, 1.0]])  # symmetric but indefinite
    S = nearest_spd(M, floor=1e-6)
    assert np.allclose(S, S.T)
    assert np.min(np.linalg.eigvalsh(S)) >= 1e-6 - 1e-12

def test_nearest_spd_is_identity_on_spd():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(4, 4)); M = A @ A.T + np.eye(4)
    assert np.allclose(nearest_spd(M), M)

def test_topic_correlation_unit_diagonal_and_values():
    Sigma = np.array([[4.0, 2.0], [2.0, 9.0]])
    R = topic_correlation(Sigma)
    assert np.allclose(np.diag(R), 1.0)
    assert np.isclose(R[0, 1], 2.0 / np.sqrt(4.0 * 9.0))
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd spark-vi && python -m pytest tests/test_linalg.py -v`
Expected: FAIL with `ModuleNotFoundError: ... _linalg`.

- [ ] **Step 3: Implement** (`spark-vi/spark_vi/models/topic/_linalg.py`)

```python
"""Pure SPD / correlation helpers for the full-covariance STM (no Spark deps).

safe_inverse mirrors the per-doc Hessian repair (_spd_inverse in stm.py) but for
Sigma. nearest_spd projects an assembled (possibly indefinite) covariance to the
nearest SPD matrix — needed because the gated per-pair M-step stitches Sigma from
inconsistent doc subsets and can break positive-definiteness (design spec C2).
"""
from __future__ import annotations
import numpy as np

def safe_inverse(M: np.ndarray, cond_cap: float = 1e-10) -> np.ndarray:
    """Inverse of a matrix meant to be SPD; eigenvalue-floor repair if not PD."""
    try:
        np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(0.5 * (M + M.T))
        floor = max(w.max() * cond_cap, 1e-12)
        w = np.maximum(w, floor)
        return (V * (1.0 / w)) @ V.T
    return np.linalg.inv(M)

def nearest_spd(M: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    """Symmetrize and floor eigenvalues at `floor`. Identity (within fp) on SPD
    inputs whose eigenvalues already exceed the floor."""
    S = 0.5 * (M + M.T)
    w, V = np.linalg.eigh(S)
    if np.min(w) >= floor:
        return S
    w = np.maximum(w, floor)
    return (V * w) @ V.T

def topic_correlation(Sigma: np.ndarray) -> np.ndarray:
    """Correlation matrix R_ij = Sigma_ij / sqrt(Sigma_ii Sigma_jj); unit diagonal."""
    d = np.sqrt(np.clip(np.diag(Sigma), 1e-300, None))
    R = Sigma / np.outer(d, d)
    np.fill_diagonal(R, 1.0)
    return R
```

Note: `nearest_spd` on an SPD input returns the symmetrized matrix `S` (bit-equal to the input when already symmetric), satisfying `test_nearest_spd_is_identity_on_spd`.

- [ ] **Step 4: Run to verify it passes**

Run: `cd spark-vi && python -m pytest tests/test_linalg.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/_linalg.py spark-vi/tests/test_linalg.py
git commit -m "feat(stm): SPD + correlation helpers for full-covariance Sigma"
```

---

## Task 3: Σ representation cut-over (non-gated full-cov MLE)

The core change: store Σ as a (K)×(K) matrix, compute Σ⁻¹ once per global step, accumulate the full residual outer-product scatter + a support matrix, and produce the full-covariance MLE M-step + full ELBO KL. Scope this task to the **non-gated and gated-without-cross-terms MLE** (no regularizers yet — Task 4; the gated sub-block prior + per-pair floor refinements come in Tasks 5-6). After this task the non-gated path is full-covariance end-to-end.

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/stm.py` — `__init__` ([335-397](../../../spark-vi/spark_vi/models/topic/stm.py#L335-L397)), `initialize_global` ([419-457](../../../spark-vi/spark_vi/models/topic/stm.py#L419-L457)), `local_update` ([461-566](../../../spark-vi/spark_vi/models/topic/stm.py#L461-L566)), `update_global` ([570-650](../../../spark-vi/spark_vi/models/topic/stm.py#L570-L650)), `compute_elbo` ([652+](../../../spark-vi/spark_vi/models/topic/stm.py#L652)), `_stm_doc_inference` ([229-320](../../../spark-vi/spark_vi/models/topic/stm.py#L229-L320)).
- Test: `spark-vi/tests/test_stm_full_sigma.py` (extend), and update `spark-vi/tests/test_stm_reference.py`, `tests/test_stm_contract.py`, `tests/test_stm_integration.py` for the new Σ shape.

**Interfaces:**
- Produces: `global_params["Sigma"]` is now a (K,K) SPD matrix (was a K-vector). `local_update` returns `residual_outer_stat` (K,K) and `n_pairs_stat` (K,K) instead of `residual_diag_stat` (K,) and `n_docs_per_topic` (K,) — but KEEP `n_docs_per_topic` too (still needed for the Γ lazy rule). `_stm_doc_inference` takes `Sigma_inv` (K,K full precision) instead of `Sigma_diag`.

- [ ] **Step 1: Write the failing test** — non-gated planted-Σ recovery + shape.

```python
def test_initialize_global_sigma_is_matrix():
    from spark_vi.models.topic.stm import OnlineSTM
    m = OnlineSTM(K=4, vocab_size=10, P=2, reference_topic=False)
    gp = m.initialize_global(None)
    assert gp["Sigma"].shape == (4, 4)
    assert np.allclose(gp["Sigma"], np.eye(4) * m.sigma_init)

def test_nongated_recovers_planted_correlated_sigma():
    """One global M-step from planted (eta, Gamma) recovers off-diagonal Sigma."""
    from spark_vi.models.topic.stm import OnlineSTM
    from spark_vi.models.topic.types import STMDocument
    rng = np.random.default_rng(3)
    K, V, P, D = 3, 8, 1, 4000
    Sigma_true = np.array([[1.0, 0.6, 0.0],
                           [0.6, 1.0, 0.0],
                           [0.0, 0.0, 1.0]])
    mu = np.zeros(K)
    etas = rng.multivariate_normal(mu, Sigma_true, size=D)
    beta = np.abs(rng.normal(size=(K, V))) + 0.1
    beta /= beta.sum(1, keepdims=True)
    docs = []
    for d in range(D):
        theta = np.exp(etas[d]); theta /= theta.sum()
        probs = theta @ beta
        counts_full = rng.multinomial(60, probs)
        idx = np.nonzero(counts_full)[0].astype(np.int32)
        docs.append(STMDocument(indices=idx, counts=counts_full[idx].astype(float),
                                length=int(counts_full.sum()),
                                x=np.array([1.0]), groups=frozenset()))
    m = OnlineSTM(K=K, vocab_size=V, P=P, reference_topic=False)
    gp = m.initialize_global(None)
    gp["lambda"] = beta * 200.0  # seed beta near truth so the E-step is informative
    gp["Gamma"] = np.zeros((P, K))
    stats = m.local_update(docs, gp)
    gp2 = m.update_global(gp, stats, learning_rate=1.0)
    R = gp2["Sigma"] / np.sqrt(np.outer(np.diag(gp2["Sigma"]), np.diag(gp2["Sigma"])))
    assert R[0, 1] > 0.3          # recovers the planted positive correlation
    assert abs(R[0, 2]) < 0.2     # uncorrelated pair stays near zero
    assert np.min(np.linalg.eigvalsh(gp2["Sigma"])) > 0  # SPD
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd spark-vi && python -m pytest tests/test_stm_full_sigma.py -k sigma -v`
Expected: FAIL (shape mismatch / off-diagonal stays 0 under the diagonal M-step).

- [ ] **Step 3: Implement the cut-over.**

  **(a) `initialize_global`** — change the two `"Sigma": np.full(self.K, self.sigma_init, ...)` lines ([438](../../../spark-vi/spark_vi/models/topic/stm.py#L438), [455](../../../spark-vi/spark_vi/models/topic/stm.py#L455)) to `"Sigma": np.eye(self.K, dtype=np.float64) * self.sigma_init`.

  **(b) `local_update`** — compute the precision once and accumulate full scatter + support:
  - After `Sigma_diag = global_params["Sigma"]` ([478](../../../spark-vi/spark_vi/models/topic/stm.py#L478)), rename to `Sigma = global_params["Sigma"]` and add `from spark_vi.models.topic._linalg import safe_inverse` (top of file) and `Sigma_inv = safe_inverse(Sigma)`.
  - Replace the `residual_diag = np.zeros(K)` accumulator with `residual_outer = np.zeros((K, K))` and `n_pairs = np.zeros((K, K))`. Keep `n_docs_per_topic` (still drives the Γ lazy rule).
  - In the `_stm_doc_inference` call ([503-509](../../../spark-vi/spark_vi/models/topic/stm.py#L503-L509)) pass `Sigma_inv=Sigma_inv` instead of `Sigma_diag=Sigma_diag`.
  - Replace the residual accumulation ([539-541](../../../spark-vi/spark_vi/models/topic/stm.py#L539-L541)) with a full outer product over `allowed_free`:
    ```python
    resid = eta_hat[allowed_free] - (Gamma.T @ doc.x)[allowed_free]
    af = allowed_free
    contrib = np.outer(resid, resid) + nu_d[np.ix_(af, af)]
    residual_outer[np.ix_(af, af)] += contrib
    n_pairs[np.ix_(af, af)] += 1.0
    n_docs_per_topic[allowed_free] += 1.0
    ```
  - Update the per-doc ELBO KL block ([546-553](../../../spark-vi/spark_vi/models/topic/stm.py#L546-L553)) to the full form using the marginal sub-block precision over `af`:
    ```python
    sub_Sigma = Sigma[np.ix_(af, af)]
    sub_Sigma_inv = safe_inverse(sub_Sigma)
    sub_nu = nu_d[np.ix_(af, af)]
    tr_term = float(np.trace(sub_Sigma_inv @ sub_nu))
    quad_term = float(resid @ sub_Sigma_inv @ resid)
    _s, logdet_nu = np.linalg.slogdet(sub_nu)
    _s2, logdet_Sigma = np.linalg.slogdet(sub_Sigma)
    doc_eta_kl += 0.5 * (tr_term + quad_term - len(af) + logdet_Sigma - logdet_nu)
    ```
    (`resid` here is the dense over-`af` vector from the line above; drop the old K-length `resid` array.)
  - Return `residual_outer_stat=residual_outer` and `n_pairs_stat=n_pairs` in the dict (replace `residual_diag_stat`); keep `n_docs_per_topic`.

  **(c) `update_global`** — full-Σ MLE M-step replacing the diagonal block ([629-643](../../../spark-vi/spark_vi/models/topic/stm.py#L629-L643)):
  ```python
  from spark_vi.models.topic._linalg import nearest_spd
  S = target_stats["residual_outer_stat"]
  N = target_stats["n_pairs_stat"]
  Sigma_target = Sigma.copy()  # lazy: entries with no support stay current
  with np.errstate(invalid="ignore", divide="ignore"):
      mle = np.where(N > 0, S / np.where(N > 0, N, 1.0), Sigma)
  present = N > 0
  Sigma_target[present] = mle[present]
  new_Sigma = (1.0 - learning_rate) * Sigma + learning_rate * Sigma_target
  new_Sigma = nearest_spd(new_Sigma + self.sigma_ridge * np.eye(self.K),
                          floor=self.SIGMA_FLOOR)
  ```
  Rename the local `Sigma_diag = global_params["Sigma"]` ([593](../../../spark-vi/spark_vi/models/topic/stm.py#L593)) to `Sigma`. Return `"Sigma": new_Sigma`. (The IW prior + diag-shrink go in Task 4; this task is plain MLE so the `sigma_prior_scale` branch is temporarily dropped — Task 4 restores it as the IW generalization.)

  **(d) `_stm_doc_inference`** — change the signature param `Sigma_diag` → `Sigma_inv` (full K×K). Where it currently does `sub_Sigma = Sigma_diag[allowed]` ([264](../../../spark-vi/spark_vi/models/topic/stm.py#L264)) and builds `common` with `Sigma_inv=np.diag(1.0/sub_Sigma)` (from Task 1), replace with `sub_Sigma_inv = Sigma_inv[np.ix_(allowed, allowed)]` and `Sigma_inv=sub_Sigma_inv` in `common`. (For non-gated `allowed` is all topics; the gated marginal-vs-conditional refinement is Task 5 — for now slicing the precision is the conditional form, corrected in Task 5.) Add a `# NOTE(Task 5): replace precision-slice with marginal sub-block inv(Sigma_AA)` comment so the follow-up is explicit.

  **(e) `compute_elbo`** — read `Sigma` as a matrix; its Gaussian-KL term must mirror the per-doc full KL in (b). Inspect [compute_elbo](../../../spark-vi/spark_vi/models/topic/stm.py#L652) and replace any `Sigma_diag`/`np.log(Sigma)`/`1.0/Sigma` usage with the full-matrix `slogdet`/`safe_inverse` forms.

- [ ] **Step 4: Run the new test**

Run: `cd spark-vi && python -m pytest tests/test_stm_full_sigma.py -v`
Expected: PASS.

- [ ] **Step 5: Update + run the existing engine suites** (Σ shape changed)

Update `test_stm_reference.py`, `test_stm_contract.py`, `test_stm_integration.py`: any assertion on `Sigma` being a K-vector (e.g. `.shape == (K,)`, `Sigma[k]`) becomes the matrix form (`.shape == (K, K)`, `Sigma[k, k]`); the reference-topic tests asserting `Gamma[:, 0] == 0` are unchanged. Run:
`cd spark-vi && python -m pytest tests/test_stm_reference.py tests/test_stm_contract.py tests/test_stm_integration.py -q`
Expected: PASS after the shape updates.

- [ ] **Step 6: Commit**

```bash
git add spark-vi/spark_vi/models/topic/stm.py spark-vi/tests/
git commit -m "feat(stm): full-covariance Sigma representation + non-gated MLE M-step"
```

---

## Task 4: Inverse-Wishart prior + diagonal-shrink regularizers

Restore and generalize the Σ prior as the conjugate inverse-Wishart (replacing the diagonal inverse-gamma), and add the stm diagonal-shrink lever. Both default off (reduce to Task 3's MLE).

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/stm.py` — `__init__` (new `sigma_diag_shrink` param + validation; keep `sigma_prior_scale`/`sigma_prior_count`), `update_global` (IW MAP + diag-shrink).
- Test: `spark-vi/tests/test_stm_full_sigma.py` (extend).

**Interfaces:**
- Consumes: `residual_outer_stat` (S), `n_pairs_stat` (N) from Task 3.
- Produces: `OnlineSTM(..., sigma_prior_scale=None, sigma_prior_count=0.0, sigma_diag_shrink=0.0)`. Σ_ij = (S_ij + Ψ_ij)/(N_ij + ν) with Ψ = ν·sigma_prior_scale·I, ν = sigma_prior_count; then Σ ← (1−w)Σ + w·diag(diag(Σ)), w = sigma_diag_shrink.

- [ ] **Step 1: Write the failing tests**

```python
def _planted_docs(rng, K=3, V=8, D=2000):
    from spark_vi.models.topic.types import STMDocument
    Sigma_true = np.array([[1.0,0.6,0.0],[0.6,1.0,0.0],[0.0,0.0,1.0]])
    etas = rng.multivariate_normal(np.zeros(K), Sigma_true, size=D)
    beta = np.abs(rng.normal(size=(K,V)))+0.1; beta/=beta.sum(1,keepdims=True)
    docs=[]
    for d in range(D):
        th=np.exp(etas[d]); th/=th.sum(); cf=rng.multinomial(60, th@beta)
        idx=np.nonzero(cf)[0].astype(np.int32)
        docs.append(STMDocument(indices=idx, counts=cf[idx].astype(float),
                    length=int(cf.sum()), x=np.array([1.0]), groups=frozenset()))
    return docs, beta

def test_iw_prior_off_equals_mle():
    from spark_vi.models.topic.stm import OnlineSTM
    rng = np.random.default_rng(4); docs, beta = _planted_docs(rng)
    def run(**kw):
        m = OnlineSTM(K=3, vocab_size=8, P=1, reference_topic=False, **kw)
        gp = m.initialize_global(None); gp["lambda"]=beta*200.0; gp["Gamma"]=np.zeros((1,3))
        return m.update_global(gp, m.local_update(docs, gp), 1.0)["Sigma"]
    assert np.allclose(run(), run(sigma_prior_scale=None, sigma_prior_count=0.0))

def test_iw_prior_shrinks_toward_scale():
    from spark_vi.models.topic.stm import OnlineSTM
    rng = np.random.default_rng(5); docs, beta = _planted_docs(rng)
    m = OnlineSTM(K=3, vocab_size=8, P=1, reference_topic=False,
                  sigma_prior_scale=5.0, sigma_prior_count=1e6)  # huge pseudo-count
    gp = m.initialize_global(None); gp["lambda"]=beta*200.0; gp["Gamma"]=np.zeros((1,3))
    Sig = m.update_global(gp, m.local_update(docs, gp), 1.0)["Sigma"]
    assert np.allclose(np.diag(Sig), 5.0, atol=0.2)        # diagonal pulled to scale
    assert np.allclose(Sig - np.diag(np.diag(Sig)), 0.0, atol=0.2)  # off-diag pulled to 0

def test_diag_shrink_one_diagonalizes():
    from spark_vi.models.topic.stm import OnlineSTM
    rng = np.random.default_rng(6); docs, beta = _planted_docs(rng)
    m = OnlineSTM(K=3, vocab_size=8, P=1, reference_topic=False, sigma_diag_shrink=1.0)
    gp = m.initialize_global(None); gp["lambda"]=beta*200.0; gp["Gamma"]=np.zeros((1,3))
    Sig = m.update_global(gp, m.local_update(docs, gp), 1.0)["Sigma"]
    assert np.allclose(Sig - np.diag(np.diag(Sig)), 0.0, atol=1e-9)  # fully diagonal

def test_sigma_diag_shrink_validation():
    from spark_vi.models.topic.stm import OnlineSTM
    import pytest
    with pytest.raises(ValueError):
        OnlineSTM(K=3, vocab_size=8, P=1, sigma_diag_shrink=1.5)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd spark-vi && python -m pytest tests/test_stm_full_sigma.py -k "iw_prior or diag_shrink" -v`
Expected: FAIL (`unexpected keyword 'sigma_diag_shrink'`, off-diag not shrunk).

- [ ] **Step 3: Implement.**
  - `__init__`: add `sigma_diag_shrink: float = 0.0` after `sigma_prior_count` ([344](../../../spark-vi/spark_vi/models/topic/stm.py#L344)); validate `if not (0.0 <= sigma_diag_shrink <= 1.0): raise ValueError(...)`; `self.sigma_diag_shrink = float(sigma_diag_shrink)`. Keep existing `sigma_prior_scale`/`sigma_prior_count` validation.
  - `update_global`: replace the Task-3 MLE block with the IW MAP + diag-shrink:
    ```python
    S = target_stats["residual_outer_stat"]; N = target_stats["n_pairs_stat"]
    nu = self.sigma_prior_count
    Psi = np.zeros((self.K, self.K))
    if self.sigma_prior_scale is not None:
        np.fill_diagonal(Psi, nu * self.sigma_prior_scale)
    denom = N + nu
    Sigma_target = Sigma.copy()
    present = denom > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        mapped = (S + Psi) / np.where(present, denom, 1.0)
    Sigma_target[present] = mapped[present]
    new_Sigma = (1.0 - learning_rate) * Sigma + learning_rate * Sigma_target
    w = self.sigma_diag_shrink
    if w > 0.0:
        new_Sigma = (1.0 - w) * new_Sigma + w * np.diag(np.diag(new_Sigma))
    new_Sigma = nearest_spd(new_Sigma + self.sigma_ridge * np.eye(self.K),
                            floor=self.SIGMA_FLOOR)
    ```
    Note: with `sigma_prior_scale=None` and `sigma_prior_count=0`, `Psi=0`, `nu=0`, `denom=N`, recovering the Task-3 MLE (`present = N>0`). The diagonal case (Ψ_kk = ν·scale) reproduces the old inverse-gamma `(resid + count*scale)/(n + count)` exactly.

- [ ] **Step 4: Run to verify it passes**

Run: `cd spark-vi && python -m pytest tests/test_stm_full_sigma.py -k "iw_prior or diag_shrink or validation" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/stm.py spark-vi/tests/test_stm_full_sigma.py
git commit -m "feat(stm): inverse-Wishart prior + diagonal-shrink for full Sigma"
```

---

## Task 5: Gated marginal sub-block prior

Correct the gated per-doc prior to the **marginal** sub-block N(μ_A, Σ_AA) (spec Q4): for a gated doc, the prior precision is inv(Σ_{A,A}), the inverse of the sub-block — not the sub-block of the inverse (which is the conditional). Precompute one inverse per distinct allowed-set in the minibatch.

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/stm.py` — `local_update` (build a per-allowed-set precision cache; pass marginal sub-precision into `_stm_doc_inference`), `_stm_doc_inference` (accept a ready precision over `allowed` instead of slicing the global precision).
- Test: `spark-vi/tests/test_stm_full_sigma.py` (extend).

**Interfaces:**
- Produces: `_stm_doc_inference(..., Sigma_inv_allowed=<(|A|,|A|) marginal precision over `allowed`>)` — the caller passes the precision already restricted to the doc's allowed set, computed as `safe_inverse(Sigma[np.ix_(allowed, allowed)])`.

- [ ] **Step 1: Write the failing test** — marginal differs from the precision-slice (conditional), and equals the hand-computed inverse of the sub-block.

```python
def test_gated_prior_uses_marginal_subblock_not_conditional():
    from spark_vi.models.topic import stm as stm_mod
    from spark_vi.models.topic.stm import _stm_doc_inference
    from spark_vi.models.topic._linalg import safe_inverse
    Sigma = np.array([[2.0, 0.8, 0.5],
                      [0.8, 1.5, 0.3],
                      [0.5, 0.3, 1.0]])
    allowed = np.array([0, 2], dtype=np.int64)  # drop topic 1
    marginal = safe_inverse(Sigma[np.ix_(allowed, allowed)])
    conditional = safe_inverse(Sigma)[np.ix_(allowed, allowed)]
    assert not np.allclose(marginal, conditional)  # they genuinely differ
    # capture the precision actually used by patching the hessian builder
    seen = {}
    orig = stm_mod._stm_neg_log_joint_hessian
    def spy(eta, **kw): seen["Sigma_inv"] = kw["Sigma_inv"]; return orig(eta, **kw)
    stm_mod._stm_neg_log_joint_hessian = spy
    try:
        _stm_doc_inference(
            indices=np.array([0,1],dtype=np.int32), counts=np.array([1.0,1.0]),
            expElogbeta=np.ones((3,2))*0.5, Gamma=np.zeros((1,3)),
            Sigma_inv_allowed=marginal, x=np.array([1.0]),
            allowed=allowed, reference=None, max_iter=5)
    finally:
        stm_mod._stm_neg_log_joint_hessian = orig
    assert np.allclose(seen["Sigma_inv"], marginal)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd spark-vi && python -m pytest tests/test_stm_full_sigma.py -k marginal -v`
Expected: FAIL (`_stm_doc_inference` has no `Sigma_inv_allowed`).

- [ ] **Step 3: Implement.**
  - `_stm_doc_inference`: replace the `Sigma_inv` parameter with `Sigma_inv_allowed` (already restricted to `allowed`, shape (|A|,|A|)). Remove the internal `Sigma_inv[np.ix_(allowed, allowed)]` slice; use `Sigma_inv_allowed` directly to build `common` (`Sigma_inv=Sigma_inv_allowed`). Remove the Task-3 NOTE comment.
  - `local_update`: build a cache keyed by the allowed-set so each distinct combination is inverted once:
    ```python
    Sigma = global_params["Sigma"]
    _subprec_cache: dict[tuple, np.ndarray] = {}
    def _marginal_precision(allowed):
        key = tuple(allowed.tolist())
        P = _subprec_cache.get(key)
        if P is None:
            P = safe_inverse(Sigma[np.ix_(allowed, allowed)])
            _subprec_cache[key] = P
        return P
    ```
    In the doc loop, after `allowed = part.allowed_indices(doc.groups)`, call `Sigma_inv_allowed = _marginal_precision(allowed)` and pass it to `_stm_doc_inference(..., Sigma_inv_allowed=Sigma_inv_allowed)` (drop the old `Sigma_inv=Sigma_inv` argument; you can drop the whole-matrix `Sigma_inv = safe_inverse(Sigma)` from Task 3 if it's now unused — but the ELBO KL block still needs `safe_inverse(Sigma[np.ix_(af,af)])`, which is the same per-`af` inverse; reuse `_marginal_precision(allowed_free)` there).

- [ ] **Step 4: Run to verify it passes**

Run: `cd spark-vi && python -m pytest tests/test_stm_full_sigma.py -k "marginal or sigma or iw_prior" -v`
Expected: PASS (marginal test + Task 3/4 tests still green — non-gated `allowed` = all topics, so the marginal precision over the full set equals the global precision).

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/stm.py spark-vi/tests/test_stm_full_sigma.py
git commit -m "feat(stm): gated marginal sub-block prior (inv of Sigma_AA)"
```

---

## Task 6: Per-pair support floor (min_pair_support) + multi-group cross-covariance

The scatter/support accumulation from Task 3 already restricts to `allowed_free × allowed_free`, so cross-group covariance is informed by comorbid (multi-group) docs automatically. This task adds the **min_pair_support floor**: cross-pairs with fewer than `min_pair_support` co-activating docs are zeroed out of the scatter (fall back to prior), as a robustness + small-cell-privacy guard.

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/stm.py` — `__init__` (`min_pair_support` param), `update_global` (apply the floor before the IW MAP).
- Test: `spark-vi/tests/test_stm_full_sigma.py` (extend).

**Interfaces:**
- Produces: `OnlineSTM(..., min_pair_support=1)`. In `update_global`, entries with `N_ij < min_pair_support` get `S_ij → 0` and `N_ij → 0` (treated as uninformed → prior/lazy).

- [ ] **Step 1: Write the failing tests** — (a) a cross-group entry informed only by enough comorbid docs survives; (b) below the floor it falls back to prior.

```python
def _gated_multigroup_docs(rng, n_comorbid):
    """background topics {0,1}; group A foreground {2}; group B foreground {3}.
    n_comorbid docs belong to BOTH A and B (co-activate topics 2 and 3)."""
    from spark_vi.models.topic.types import STMDocument
    docs = []
    def mk(groups, idx):
        cf = np.zeros(6); 
        for i in idx: cf[i] = 5.0
        ix = np.nonzero(cf)[0].astype(np.int32)
        return STMDocument(indices=ix, counts=cf[ix], length=int(cf.sum()),
                           x=np.array([1.0]), groups=frozenset(groups))
    for _ in range(400): docs.append(mk(["A"], [0,1,4]))   # vocab 4 ~ topic2 word
    for _ in range(400): docs.append(mk(["B"], [0,1,5]))   # vocab 5 ~ topic3 word
    for _ in range(n_comorbid): docs.append(mk(["A","B"], [0,1,4,5]))
    return docs

def _fit_block(min_pair_support, n_comorbid):
    from spark_vi.models.topic.stm import OnlineSTM
    from spark_vi.models.topic.partition import TopicBlockPartition
    rng = np.random.default_rng(7)
    part = TopicBlockPartition(group_var="g", background_k=2,
                               foreground=(("A",1),("B",1)))  # K=4
    m = OnlineSTM(K=4, vocab_size=6, P=1, reference_topic=False,
                  topic_blocks=part, min_pair_support=min_pair_support)
    gp = m.initialize_global(None); gp["Gamma"]=np.zeros((1,4))
    docs = _gated_multigroup_docs(rng, n_comorbid)
    stats = m.local_update(docs, gp)
    return m.update_global(gp, stats, 1.0)["Sigma"], part

def test_cross_group_covariance_from_comorbid_docs():
    Sig, part = _fit_block(min_pair_support=10, n_comorbid=300)
    a = part.block_indices("A")[0]; b = part.block_indices("B")[0]
    assert Sig[a, b] != 0.0                     # informed cross-group entry
    assert np.min(np.linalg.eigvalsh(Sig)) > 0  # SPD

def test_thin_cross_group_falls_back_to_prior():
    Sig, part = _fit_block(min_pair_support=50, n_comorbid=5)  # below floor
    a = part.block_indices("A")[0]; b = part.block_indices("B")[0]
    assert Sig[a, b] == 0.0    # 5 comorbid docs < floor 50 -> not estimated
    assert np.min(np.linalg.eigvalsh(Sig)) > 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd spark-vi && python -m pytest tests/test_stm_full_sigma.py -k "cross_group or thin_cross" -v`
Expected: FAIL (`unexpected keyword 'min_pair_support'`).

- [ ] **Step 3: Implement.**
  - `__init__`: add `min_pair_support: int = 1` after `sigma_diag_shrink`; validate `if min_pair_support < 1: raise ValueError(...)`; `self.min_pair_support = int(min_pair_support)`.
  - `update_global`, immediately before building `denom`/`mapped` (Task 4): apply the floor.
    ```python
    if self.min_pair_support > 1:
        thin = N < self.min_pair_support
        S = np.where(thin, 0.0, S)
        N = np.where(thin, 0.0, N)
    ```
    (Diagonal and background pairs have full support and never trip it; only thin cross-group cells do. After zeroing, those cells get the prior via the IW Ψ / lazy-keep, then SPD repair completes the matrix.)

- [ ] **Step 4: Run to verify it passes**

Run: `cd spark-vi && python -m pytest tests/test_stm_full_sigma.py -k "cross_group or thin_cross" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/stm.py spark-vi/tests/test_stm_full_sigma.py
git commit -m "feat(stm): min_pair_support floor for thin cross-group covariance"
```

---

## Task 7: Correlation extraction + Σ-matrix diagnostics

Expose the correlation matrix and the support matrix as outputs, and generalize the per-iter Σ diagnostic (ADR 0030) to the full-matrix health signals.

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/stm.py` — add a `topic_correlation_matrix(global_params)` method (delegates to `_linalg.topic_correlation`); the diagnostic-trace hook that currently logs `Sigma[min...max]` (find via `grep -n "Sigma\[" spark-vi/spark_vi/models/topic/stm.py` and the VIRunner diagnostic callback).
- Test: `spark-vi/tests/test_stm_full_sigma.py` (extend).

**Interfaces:**
- Produces: `OnlineSTM.topic_correlation_matrix(global_params) -> (K,K)` correlation; the diagnostic dict gains `sigma_cond` (condition number), `sigma_eig_min`, `sigma_eig_max`, `max_abs_offdiag_corr`.

- [ ] **Step 1: Write the failing test**

```python
def test_topic_correlation_matrix_from_sigma():
    from spark_vi.models.topic.stm import OnlineSTM
    m = OnlineSTM(K=3, vocab_size=8, P=1, reference_topic=False)
    gp = m.initialize_global(None)
    gp["Sigma"] = np.array([[4.0,2.0,0.0],[2.0,9.0,0.0],[0.0,0.0,1.0]])
    R = m.topic_correlation_matrix(gp)
    assert np.allclose(np.diag(R), 1.0)
    assert np.isclose(R[0,1], 2.0/np.sqrt(36.0))
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd spark-vi && python -m pytest tests/test_stm_full_sigma.py -k correlation -v`
Expected: FAIL (no `topic_correlation_matrix`).

- [ ] **Step 3: Implement** — add to `OnlineSTM`:
```python
def topic_correlation_matrix(self, global_params):
    """Topic correlation R_ij = Sigma_ij / sqrt(Sigma_ii Sigma_jj) (Blei & Lafferty
    2007 logistic-normal correlation). See _linalg.topic_correlation."""
    from spark_vi.models.topic._linalg import topic_correlation
    return topic_correlation(global_params["Sigma"])
```
Locate the Σ diagnostic line (`grep -n "Sigma\[" spark-vi/spark_vi/models/topic/stm.py`) and, where it computes the per-topic `Sigma[min...max]`, add the full-matrix signals from `np.linalg.eigvalsh(Sigma)` (min/max/cond) and `np.max(np.abs(topic_correlation(Sigma) - np.eye(K)))`. Keep the log line plain-text (no LaTeX).

- [ ] **Step 4: Run to verify it passes**

Run: `cd spark-vi && python -m pytest tests/test_stm_full_sigma.py -k correlation -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/stm.py spark-vi/tests/test_stm_full_sigma.py
git commit -m "feat(stm): topic correlation extraction + full-Sigma diagnostics"
```

---

## Task 8: Shim persistence + metadata (StreamingSTM)

Thread the new knobs through `StreamingSTM`, persist Σ (matrix), the correlation R, and the support N, and record provenance in metadata.

**Files:**
- Modify: `spark-vi/spark_vi/mllib/topic/stm.py` — `StreamingSTM.__init__` (new params), `fit` (pass to `OnlineSTM`; record `stm_hardening`), the `STMModel` save path (persist `correlation` + `support`).
- Test: `spark-vi/tests/test_mllib_stm.py` (extend).

**Interfaces:**
- Consumes: `OnlineSTM(..., sigma_diag_shrink, min_pair_support)`, `topic_correlation_matrix`.
- Produces: `StreamingSTM(..., sigma_diag_shrink=0.0, min_pair_support=1)`; persisted `params/Sigma.npy` (K×K), `params/correlation.npy`, metadata `stm_hardening` gains `sigma_diag_shrink`, `min_pair_support`.

- [ ] **Step 1: Write the failing test** — `StreamingSTM` forwards the params and the fitted model exposes a (K,K) Σ + correlation.

```python
def test_streaming_stm_full_sigma_metadata_and_shapes(spark, tiny_stm_dataset):
    from spark_vi.mllib.topic.stm import StreamingSTM
    est = StreamingSTM(K=4, covariate_formula="~ x", sigma_diag_shrink=0.0,
                       min_pair_support=3, reference_topic=False)
    model = est.fit(tiny_stm_dataset, max_iter=2)
    assert model.global_params["Sigma"].shape == (4, 4)
    assert model.metadata["stm_hardening"]["min_pair_support"] == 3
    assert model.metadata["stm_hardening"]["sigma_diag_shrink"] == 0.0
```
(Use the existing `spark`/dataset fixtures in `test_mllib_stm.py`; mirror the closest existing `StreamingSTM.fit` test for construction.)

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH="charmpheno:spark-vi:$PYTHONPATH" python -m pytest spark-vi/tests/test_mllib_stm.py -k full_sigma -v`
Expected: FAIL (`unexpected keyword 'sigma_diag_shrink'`).

- [ ] **Step 3: Implement** — in `StreamingSTM.__init__` add `sigma_diag_shrink: float = 0.0` and `min_pair_support: int = 1` (mirror the existing `sigma_prior_scale`/`sigma_prior_count` storage at [stm.py:175-178](../../../spark-vi/spark_vi/mllib/topic/stm.py#L175-L178)); forward both into the `OnlineSTM(...)` construction in `fit`; add both keys to the `metadata.setdefault("stm_hardening", {...})` dict ([stm.py:353-358](../../../spark-vi/spark_vi/mllib/topic/stm.py#L353-L358)). In the `STMModel` save (find via `grep -n "def save\|np.save\|params/" spark-vi/spark_vi/mllib/topic/stm.py`), also write `correlation.npy` from `model.topic_correlation_matrix(global_params)`.

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH="charmpheno:spark-vi:$PYTHONPATH" python -m pytest spark-vi/tests/test_mllib_stm.py -k full_sigma -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/mllib/topic/stm.py spark-vi/tests/test_mllib_stm.py
git commit -m "feat(stm): StreamingSTM threads diag-shrink/min_pair_support, persists correlation"
```

---

## Task 9: Driver + run_experiment threading

Thread `--sigma-diag-shrink` and `--min-pair-support` (plus the now-IW `--sigma-prior-*`) through both drivers and `build_stm_args`, mirroring the existing knob pattern.

**Files:**
- Modify: `analysis/cloud/stm_bigquery_cloud.py`, `analysis/local/fit_stm_local.py` (argparse + `StreamingSTM(...)` forwarding), `scripts/run_experiment.py` (`build_stm_args`).
- Test: `analysis/cloud/tests/test_stm_driver_partition.py`, `scripts/tests/test_run_experiment.py`, `tests/scripts/test_fit_stm_local.py`.

**Interfaces:**
- Produces: `--sigma-diag-shrink FLOAT` (default 0.0), `--min-pair-support INT` (default 1); `build_stm_args` emits them when present in the experiment frontmatter.

- [ ] **Step 1: Write the failing tests** — parse + forward + emit. Mirror the existing `--sigma-prior-scale` / spectral tests already in these files.

```python
# scripts/tests/test_run_experiment.py
def test_build_stm_args_emits_full_sigma_knobs():
    from run_experiment import build_stm_args
    eff = _base_stm_effective()  # existing helper
    eff.update({"sigma_diag_shrink": 0.25, "min_pair_support": 30})
    args = build_stm_args(eff, checkpoint_dir=None)  # match the real signature
    assert "--sigma-diag-shrink" in args and "0.25" in args
    assert "--min-pair-support" in args and "30" in args
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH="charmpheno:spark-vi:$PYTHONPATH" python -m pytest scripts/tests/test_run_experiment.py -k full_sigma -v`
Expected: FAIL.

- [ ] **Step 3: Implement** — add `p.add_argument("--sigma-diag-shrink", type=float, default=0.0, help="...")` and `p.add_argument("--min-pair-support", type=int, default=1, help="...")` to both drivers right after the existing `--sigma-prior-count` argument; forward `sigma_diag_shrink=args.sigma_diag_shrink, min_pair_support=args.min_pair_support` into `StreamingSTM(...)`. In `build_stm_args` ([scripts/run_experiment.py:475-487](../../../scripts/run_experiment.py#L475-L487)), emit each when present: `if effective.get("sigma_diag_shrink"): hardening.extend(["--sigma-diag-shrink", str(...)])`; same for `min_pair_support`.

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH="charmpheno:spark-vi:$PYTHONPATH" python -m pytest scripts/tests/test_run_experiment.py analysis/cloud/tests/test_stm_driver_partition.py -k "full_sigma or sigma" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add analysis/ scripts/
git commit -m "feat(stm): thread sigma-diag-shrink + min-pair-support through drivers"
```

---

## Task 10: Synthetic gated SPD-assembly + integration coverage

Add the deliberately-adversarial SPD-assembly test (background correlated with two foregrounds, cross-foreground pinned) and a small end-to-end local-driver run, to lock the central numerical risk and the full pipeline.

**Files:**
- Test: `spark-vi/tests/test_stm_full_sigma.py` (extend), `tests/scripts/test_fit_stm_local.py` (extend).

- [ ] **Step 1: Write the failing tests**

```python
def test_assembled_sigma_is_spd_when_cross_block_inconsistent():
    """background topic 0 correlates strongly with foreground 1 AND 2, but no doc
    co-activates 1 and 2 -> entry (1,2) pinned 0. Assembled Sigma must be repaired
    to SPD (eigenvalues > 0), not returned indefinite."""
    from spark_vi.models.topic.stm import OnlineSTM
    from spark_vi.models.topic.partition import TopicBlockPartition
    from spark_vi.models.topic.types import STMDocument
    rng = np.random.default_rng(8)
    part = TopicBlockPartition(group_var="g", background_k=1,
                               foreground=(("A",1),("B",1)))  # K=3
    docs = []
    for _ in range(500):  # group A: topics {0,1}
        docs.append(STMDocument(indices=np.array([0,1],dtype=np.int32),
            counts=np.array([8.0,8.0]), length=16, x=np.array([1.0]), groups=frozenset(["A"])))
    for _ in range(500):  # group B: topics {0,2}
        docs.append(STMDocument(indices=np.array([0,2],dtype=np.int32),
            counts=np.array([8.0,8.0]), length=16, x=np.array([1.0]), groups=frozenset(["B"])))
    m = OnlineSTM(K=3, vocab_size=3, P=1, reference_topic=False, topic_blocks=part)
    gp = m.initialize_global(None); gp["Gamma"]=np.zeros((1,3))
    Sig = m.update_global(gp, m.local_update(docs, gp), 1.0)["Sigma"]
    assert np.min(np.linalg.eigvalsh(Sig)) > 0
    assert np.allclose(Sig, Sig.T)
```

For `tests/scripts/test_fit_stm_local.py`, add `test_fit_stm_local_full_sigma_end_to_end`: run `fit_main` on the existing gated sim with `--min-pair-support 2`, assert `rc == 0`, the saved `Sigma.npy` is `(K, K)`, and `correlation.npy` exists.

- [ ] **Step 2: Run to verify they fail/pass appropriately**

Run: `cd spark-vi && python -m pytest tests/test_stm_full_sigma.py -k spd_when_cross -v`
Then: `PYTHONPATH="charmpheno:spark-vi:$PYTHONPATH" python -m pytest tests/scripts/test_fit_stm_local.py -k full_sigma -v`
Expected: both PASS (the engine code already exists from Tasks 3-8; these lock behavior). If the SPD test fails, the `nearest_spd` floor in `update_global` (Task 3/4) needs the ridge add-in — fix there.

- [ ] **Step 3: Commit**

```bash
git add spark-vi/tests/test_stm_full_sigma.py tests/scripts/test_fit_stm_local.py
git commit -m "test(stm): gated SPD-assembly + full-Sigma local end-to-end"
```

---

## Task 11: Docs — ADR, insight, experiments

Record the architecture decision (full Σ supersedes diagonal), an insight stub for the result, and two pre-registered experiments (100 iters each). No code.

**Files:**
- Create: `docs/decisions/0033-stm-full-covariance-sigma.md`, `docs/experiments/0020-stm-cancer-fullcov-nongated.md`, `docs/experiments/0021-stm-comorbid-fullcov-gated-multigroup.md`.
- Modify: `docs/decisions/0028-*.md` (note ADR 0028-B sampler is now enabled by full Σ), `docs/decisions/0030-*.md` (Σ diagnostic generalized), `docs/superpowers/specs/2026-06-30-stm-full-covariance-sigma-design.md` (status → Implemented).

- [ ] **Step 1: Write ADR 0033** — context (diagonal forwent topic correlation; the design decisions 1-7 from the spec), decision (replace diagonal with full Σ; marginal gated sub-block; multi-group cross-covariance via comorbid docs; per-pair lazy + min_pair_support floor; IW + diagonal-shrink regularizers; the three-layer SPD strategy), consequences (the diagonal results 0015/0017/insight 0030 become the prior baseline; ADR 0028-B sampler enabled; no backward compat), references (Blei & Lafferty 2007, Roberts et al. stm, ADR 0027/0028/0030/0031). Cite literature; no LaTeX; markdown-linkable refs.

- [ ] **Step 2: Write the two experiment docs** — mirror the `docs/experiments/0017-*.md` frontmatter+prose format. 0020: non-gated cancer cohort, `K: 40`, `max_iter: 100`, full Σ (no gating), report the correlation matrix + topic quality (NPMI vs exp 0017) + Σ conditioning. 0021: a comorbid cohort (cancer + dementia) with multi-group membership, `min_pair_support` set, validating cross-group covariance from comorbid patients + the support floor. Both include the hypothesis / what-to-watch / decision tree sections.

- [ ] **Step 3: Update ADR 0028 + 0030 + the spec status** — one-line notes pointing to ADR 0033 / the full-Σ enablement; flip the spec `Status:` to `Implemented`.

- [ ] **Step 4: Commit**

```bash
git add docs/
git commit -m "docs(stm): ADR 0033 full-covariance Sigma + exps 0020/0021 + spec status"
```

---

## Plan complete — execution

After all tasks: dispatch the final whole-branch review (most capable model) over `git diff <plan-base> HEAD`, address findings via one fix subagent, then run the two cluster experiments (`make exp ID=20`, `make exp ID=21`). Push only when the user asks.
