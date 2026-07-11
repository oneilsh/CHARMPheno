# PG-STM Inference Core Implementation Plan (sub-project 1/3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A single-machine, full-batch Pólya-Gamma variational core for the gated
stick-breaking logistic-normal topic model, with an exact Gibbs cross-check, validated to
the milestone-1 checkpoint (runaway cured + structure recovered + VI≈Gibbs).

**Architecture:** Stick-breaking multinomial link → PG augmentation makes the per-doc stick
logits ψ conditionally Gaussian → conjugate coordinate-ascent VI (ω, ψ, Σ via Inverse-
Wishart, Γ ridge, β Dirichlet) with one delta-method approximation for E[log θ] in the
token-assignment step. An exact PG-Gibbs sampler audits that approximation and the Σ
posterior.

**Tech Stack:** Python, NumPy, SciPy, `polyagamma==2.0.2` (`random_polyagamma`), pytest.

**Design doc:** `docs/superpowers/specs/2026-07-11-pg-stm-inference-core-design.md`

## Global Constraints

- **Correctness is defined by the tests.** Every non-trivial update is validated against an
  independent reference (analytic conjugate formula, brute-force/grid posterior, or Monte-
  Carlo expectation), not against its own re-derivation. Where a formula is given, the test
  is the check that it was transcribed right.
- **On-path VI:** the full-batch coordinate ascent must be structured so each update is a
  pure function of (global params, per-doc suff-stats) — this is the Tier-3 SVI kernel at
  batch = full corpus. No design choice that blocks minibatching later.
- **Parameterization caveat (load-bearing for tests):** `gated_ln_corpus` plants from
  `softmax(η)` with a known *softmax*-logit `Sigma_true`. PG-STM uses the **stick-breaking**
  link, whose Σ is the *stick*-logit covariance — a different object. So β recovery is the
  only cross-model comparison that's valid (topic-word structure is link-agnostic); **never
  assert PG-STM's Σ ≈ the planted `Sigma_true`.** All Σ correctness is link-internal:
  VI-vs-Gibbs agreement, boundedness, and the within-model MLE-vs-IW contrast (T7).
- **Gating is block-structured:** a doc's ψ lives on its allowed background∪group sticks
  (`TopicBlockPartition.allowed_indices`); Σ is (K−1)×(K−1) block-structured; cross-group
  entries are updated only from within-block scatter (never co-active under single-label
  gating) — mirror the current STM's marginal-precision logic.
- **No reference topic** (stick-breaking is inherently identified). Σ, Γ are (K−1)-dim.
- **Dependency:** `polyagamma==2.0.2`, API `random_polyagamma(h=<trials>, z=<logit>)` for a
  PG(h, z) draw; verified installing + sampling in the dev env.
- **Single machine, no Spark, no export/dashboard** (sub-projects 2, 3).

## Notation (used verbatim below)

- K topics, K−1 sticks. σ(x)=1/(1+e^−x) logistic. For a doc: topic counts n∈R^K (from token
  assignments), N=Σ n_k. Stick k "successes" a_k=n_k, "trials at risk" b_k=Σ_{j≥k} n_j, PG
  linear term κ_k=a_k−b_k/2. Per-doc prior mean μ_d=Γᵀx_d, prior cov Σ. Per-doc ψ posterior
  N(m_d, V_d). s_k=σ(m_k)(1−σ(m_k)).

## File Structure

- `spark-vi/spark_vi/models/topic/pg_stm.py` — the whole core (link + PG conditionals +
  global updates + `PGSTMVI` full-batch driver + `pg_stm_gibbs` sampler). One module for the
  prototype; sub-project #2 may split it.
- Tests (per concern): `test_pg_stm_link.py` (T1), `test_pg_stm_conditionals.py` (T2),
  `test_pg_stm_updates.py` (T3), `test_pg_stm_assignment.py` (T4), `test_pg_stm_nested.py` (T5),
  `test_pg_stm_vi.py` (T6), `test_pg_stm_gibbs.py` (T7), `test_pg_stm_runaway.py` (T8).
- **NESTED gating (T5+):** Tasks 1–4 build FLAT stick-breaking block primitives; Task 5 composes
  them into the nested per-group [gate, foreground] + shared background structure (K−1 sticks
  total, a doc uses |allowed|−1). Tasks 6–8 build the gated driver/Gibbs/runaway on it.
- `polyagamma==2.0.2` (wheels; verified) — dev env + later the cluster image.

## Test environment

Run from `spark-vi/`: `python -m pytest tests/<file> -v`. Reuse `tests/_stm_synth.py`:
`gated_ln_corpus(*, group_weights, fg_per_group, bg_k, V, D, doc_len, ...)` plants from a
**logistic-normal** (so Σ recovery is testable), `synthetic_gated_corpus(...)`,
`planted_recovery(beta_hat, planted_beta, thresh=0.5)`,
`foreground_recovers_group(beta_hat, partition, group, planted_beta)`,
`final_sigma_range(gp)`, and `fit_stm(docs, *, K, V, sigma_init, n_iter, seed)` (the point-EM
baseline). `TopicBlockPartition`: `.K`, `.groups`, `.background_indices()`,
`.block_indices(g)`, `.allowed_indices(frozenset)`. `STMDocument`: `.indices`, `.counts`,
`.x`, `.groups`.

---

### Task 1: Stick-breaking link + count factorization

Pure, deterministic functions — the bijection between stick logits ψ and simplex point θ,
and the per-doc PG count factorization. Foundational; everything else consumes these.

**Files:**
- Create: `spark-vi/spark_vi/models/topic/pg_stm.py`
- Test: `spark-vi/tests/test_pg_stm_link.py`

**Interfaces — Produces:**
- `stick_to_simplex(psi: np.ndarray) -> np.ndarray` — ψ shape (K−1,) → θ shape (K,), sums to 1.
- `simplex_to_stick(theta: np.ndarray) -> np.ndarray` — inverse, θ (K,) → ψ (K−1,).
- `stick_trials(n: np.ndarray) -> np.ndarray` — topic counts n (K,) → trials-at-risk b (K−1,),
  `b[k] = n[k:].sum()`. (Successes a = n[:K−1]; κ = a − b/2.)

- [ ] **Step 1: Write the failing tests**

```python
# spark-vi/tests/test_pg_stm_link.py
import numpy as np
from spark_vi.models.topic.pg_stm import stick_to_simplex, simplex_to_stick, stick_trials


def test_stick_to_simplex_sums_to_one():
    rng = np.random.default_rng(0)
    for _ in range(20):
        psi = rng.normal(size=5)          # K-1 = 5 -> K = 6
        theta = stick_to_simplex(psi)
        assert theta.shape == (6,)
        assert np.all(theta > 0)
        assert abs(theta.sum() - 1.0) < 1e-12


def test_roundtrip_psi_theta_psi():
    rng = np.random.default_rng(1)
    for _ in range(20):
        psi = rng.normal(size=7)
        theta = stick_to_simplex(psi)
        psi2 = simplex_to_stick(theta)
        assert np.allclose(psi, psi2, atol=1e-9)


def test_stick_trials_at_risk():
    n = np.array([3.0, 0.0, 5.0, 2.0])      # K=4, N=10
    b = stick_trials(n)                      # K-1 = 3
    assert np.allclose(b, [10.0, 7.0, 7.0])  # N, N-3, N-3-0
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd spark-vi && python -m pytest tests/test_pg_stm_link.py -v`
Expected: FAIL — `cannot import name 'stick_to_simplex'`.

- [ ] **Step 3: Implement**

```python
# spark-vi/spark_vi/models/topic/pg_stm.py
"""Pólya-Gamma variational core for the gated stick-breaking logistic-normal
topic model (design 2026-07-11-pg-stm-inference-core-design.md). Single machine,
full-batch VI + exact Gibbs cross-check. References: Polson/Scott/Windle 2013 (PG);
Linderman/Johnson/Adams 2015 (stick-breaking multinomial + PG); Blei/Lafferty 2007
(logistic-normal topic model)."""
from __future__ import annotations

import numpy as np
from scipy.special import expit  # logistic sigmoid


def stick_to_simplex(psi: np.ndarray) -> np.ndarray:
    """Stick-breaking map: psi (K-1,) -> theta (K,) on the simplex.
    theta_k = sigma(psi_k) * prod_{j<k}(1 - sigma(psi_j)); last topic gets the remainder."""
    psi = np.asarray(psi, dtype=np.float64)
    sig = expit(psi)                          # (K-1,)
    theta = np.empty(psi.shape[0] + 1, dtype=np.float64)
    remaining = 1.0
    for k in range(psi.shape[0]):
        theta[k] = remaining * sig[k]
        remaining *= (1.0 - sig[k])
    theta[-1] = remaining
    return theta


def simplex_to_stick(theta: np.ndarray) -> np.ndarray:
    """Inverse map: theta (K,) -> psi (K-1,). sigma(psi_k) = theta_k / (1 - sum_{j<k} theta_j)."""
    theta = np.asarray(theta, dtype=np.float64)
    psi = np.empty(theta.shape[0] - 1, dtype=np.float64)
    remaining = 1.0
    for k in range(theta.shape[0] - 1):
        frac = np.clip(theta[k] / remaining, 1e-15, 1.0 - 1e-15)
        psi[k] = np.log(frac) - np.log1p(-frac)   # logit(frac)
        remaining -= theta[k]
    return psi


def stick_trials(n: np.ndarray) -> np.ndarray:
    """Per-stick trials-at-risk b (K-1,): b[k] = sum_{j>=k} n[j]."""
    n = np.asarray(n, dtype=np.float64)
    return np.cumsum(n[::-1])[::-1][:-1].copy()
```

- [ ] **Step 4: Run to verify they pass**

Run: `cd spark-vi && python -m pytest tests/test_pg_stm_link.py -v`  Expected: PASS (3).

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/pg_stm.py spark-vi/tests/test_pg_stm_link.py
git commit -m "feat(pg-stm): stick-breaking link + PG count factorization"
```

---

### Task 2: PG conditionals — ω update and the per-doc ψ Gaussian posterior

The heart of the augmentation. Given topic counts and the Gaussian prior, PG makes ψ
conditionally Gaussian. Two pieces: the ω expectation/draw, and the ψ posterior (m, V).

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/pg_stm.py`
- Test: `spark-vi/tests/test_pg_stm_conditionals.py`

**Interfaces:**
- Consumes: `stick_trials` (T1); `polyagamma.random_polyagamma`.
- Produces:
  - `omega_expectation(b: np.ndarray, c: np.ndarray) -> np.ndarray` — VI mean
    `E[ω_k] = (b_k/(2 c_k)) tanh(c_k/2)`, with `c_k = sqrt(E[ψ_k²])`; `c_k→0` limit = `b_k/4`.
  - `omega_sample(b, psi, rng) -> np.ndarray` — Gibbs draw `ω_k ~ PG(b_k, ψ_k)` via
    `random_polyagamma(h=b, z=psi, random_state=rng)`.
  - `psi_posterior(n, b, mu, Sigma_inv, omega) -> tuple[np.ndarray, np.ndarray]` — returns
    `(m, V)` with `V = (Sigma_inv + diag(omega))^{-1}`, `m = V @ (Sigma_inv @ mu + kappa)`,
    `kappa = n[:len(b)] - b/2`. All arrays over the doc's stick set (caller slices to allowed).

- [ ] **Step 1: Write the failing tests**

```python
# spark-vi/tests/test_pg_stm_conditionals.py
import numpy as np
from scipy.integrate import quad
from spark_vi.models.topic.pg_stm import omega_expectation, omega_sample, psi_posterior


def test_omega_expectation_matches_pg_mean():
    # E[PG(b,c)] = (b/2c) tanh(c/2); check vs a large-sample MC of random_polyagamma
    from polyagamma import random_polyagamma
    rng = np.random.default_rng(0)
    b = np.array([2.0, 5.0]); c = np.array([0.7, 1.3])
    mc = random_polyagamma(h=np.repeat(b, 200000).reshape(2, -1).T,
                           z=np.repeat(c, 200000).reshape(2, -1).T,
                           random_state=rng).mean(axis=0)
    assert np.allclose(omega_expectation(b, c), mc, rtol=2e-2)


def test_omega_expectation_zero_limit():
    b = np.array([4.0])
    val = omega_expectation(b, np.array([1e-9]))
    assert abs(val[0] - 1.0) < 1e-6           # b/4 = 1.0


def test_psi_posterior_matches_bruteforce_1stick():
    # K=2 (one stick). Posterior over psi given a Binomial(N, sigma(psi)) likelihood and
    # Gaussian prior N(mu, s2) is, under PG with a FIXED omega, exactly N(m, V). Verify the
    # returned (m, V) equals the closed-form Gaussian obtained by completing the square.
    n = np.array([7.0, 3.0]); b = np.array([10.0])       # a=7, b=10, kappa=2
    mu = np.array([0.4]); s2 = 1.7
    Sigma_inv = np.array([[1.0 / s2]])
    omega = np.array([0.9])
    m, V = psi_posterior(n, b, mu, Sigma_inv, omega)
    kappa = n[:1] - b / 2
    V_ref = 1.0 / (1.0 / s2 + omega[0])
    m_ref = V_ref * (mu[0] / s2 + kappa[0])
    assert np.allclose(V, [[V_ref]], atol=1e-12)
    assert np.allclose(m, [m_ref], atol=1e-12)


def test_psi_posterior_two_sticks_coupled_prior():
    # With a correlated 2x2 prior, V must be (Sigma_inv + diag(omega))^-1 exactly.
    n = np.array([4.0, 2.0, 1.0]); b = np.array([7.0, 3.0])
    mu = np.array([0.1, -0.2])
    Sigma = np.array([[1.5, 0.6], [0.6, 1.2]]); Sigma_inv = np.linalg.inv(Sigma)
    omega = np.array([0.5, 0.8])
    m, V = psi_posterior(n, b, mu, Sigma_inv, omega)
    V_ref = np.linalg.inv(Sigma_inv + np.diag(omega))
    kappa = n[:2] - b / 2
    m_ref = V_ref @ (Sigma_inv @ mu + kappa)
    assert np.allclose(V, V_ref, atol=1e-12)
    assert np.allclose(m, m_ref, atol=1e-12)
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd spark-vi && python -m pytest tests/test_pg_stm_conditionals.py -v`
Expected: FAIL — import error.

- [ ] **Step 3: Implement**

Append to `pg_stm.py`:

```python
from polyagamma import random_polyagamma


def omega_expectation(b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Variational mean of the PG auxiliary: E[omega_k] = (b_k/(2 c_k)) tanh(c_k/2),
    c_k = sqrt(E[psi_k^2]). tanh(c/2)/c -> 1/2 as c->0, so the limit is b/4."""
    b = np.asarray(b, dtype=np.float64); c = np.asarray(c, dtype=np.float64)
    out = np.empty_like(b)
    small = c < 1e-6
    out[small] = b[small] / 4.0
    cc = c[~small]
    out[~small] = b[~small] / (2.0 * cc) * np.tanh(cc / 2.0)
    return out


def omega_sample(b: np.ndarray, psi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Exact Gibbs draw omega_k ~ PG(b_k, psi_k)."""
    return random_polyagamma(h=np.asarray(b, dtype=np.float64),
                             z=np.asarray(psi, dtype=np.float64), random_state=rng)


def psi_posterior(n, b, mu, Sigma_inv, omega):
    """Per-doc Gaussian posterior over the stick logits under PG augmentation.
    V = (Sigma_inv + diag(omega))^-1 ; m = V (Sigma_inv mu + kappa) ; kappa = a - b/2."""
    b = np.asarray(b, dtype=np.float64)
    kappa = np.asarray(n, dtype=np.float64)[:b.shape[0]] - b / 2.0
    prec = np.asarray(Sigma_inv, dtype=np.float64) + np.diag(np.asarray(omega, dtype=np.float64))
    V = np.linalg.inv(prec)
    m = V @ (np.asarray(Sigma_inv, dtype=np.float64) @ np.asarray(mu, dtype=np.float64) + kappa)
    return m, V
```

- [ ] **Step 4: Run to verify they pass**

Run: `cd spark-vi && python -m pytest tests/test_pg_stm_conditionals.py -v`  Expected: PASS (4).

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/pg_stm.py spark-vi/tests/test_pg_stm_conditionals.py
git commit -m "feat(pg-stm): PG omega update (VI mean + Gibbs draw) and psi Gaussian posterior"
```

---

### Task 3: Global conjugate updates — Inverse-Wishart Σ, ridge Γ, Dirichlet β

The global M-step, all conjugate. The IW Σ update is the one that used to run away, now a
proper posterior. Block-structured for gating.

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/pg_stm.py`
- Test: `spark-vi/tests/test_pg_stm_updates.py`

**Interfaces — Produces:**
- `sigma_iw_posterior_mean(scatter, n_docs, *, Psi0, nu0, dim) -> np.ndarray` — IW posterior
  mean `E[Σ] = (Psi0 + scatter) / (nu0 + n_docs - dim - 1)`. `scatter = Σ_d (e_d e_dᵀ + V_d)`,
  `e_d = m_d − μ_d`.
- `gamma_ridge(M, X, *, ridge) -> np.ndarray` — `Γ̂ = (XᵀX + ridge·I)⁻¹ XᵀM`, M the (D,K−1)
  stacked posterior means m_d, X the (D,P) covariates. Returns (P, K−1).
- `beta_dirichlet_mean(word_topic_stats, *, eta) -> np.ndarray` — `(K,V)` row-normalized
  `(eta + stats)`.

Block-structure is applied by the *caller* (T5): it accumulates `scatter` only within the
background block and each group block (never cross-group), passing the block sub-matrices to
`sigma_iw_posterior_mean`. This task provides the pure conjugate updates.

- [ ] **Step 1: Write the failing tests**

```python
# spark-vi/tests/test_pg_stm_updates.py
import numpy as np
from spark_vi.models.topic.pg_stm import (
    sigma_iw_posterior_mean, gamma_ridge, beta_dirichlet_mean)


def test_iw_posterior_mean_recovers_planted_cov_large_n():
    # Draw e_d ~ N(0, Sigma_true); the IW posterior mean -> Sigma_true as D grows.
    rng = np.random.default_rng(0)
    dim = 4
    A = rng.normal(size=(dim, dim)); Sigma_true = A @ A.T + np.eye(dim)
    D = 20000
    E = rng.multivariate_normal(np.zeros(dim), Sigma_true, size=D)
    scatter = E.T @ E                                   # sum e_d e_d^T (V_d = 0 here)
    Psi0 = np.eye(dim); nu0 = dim + 2
    est = sigma_iw_posterior_mean(scatter, D, Psi0=Psi0, nu0=nu0, dim=dim)
    assert np.allclose(est, Sigma_true, atol=0.15)


def test_iw_posterior_mean_is_finite_and_pd_with_zero_data():
    # The whole point: even with NO informative data, the proper prior gives a finite PD mean.
    dim = 3
    est = sigma_iw_posterior_mean(np.zeros((dim, dim)), 0, Psi0=2.0*np.eye(dim), nu0=dim+2, dim=dim)
    assert np.all(np.isfinite(est))
    assert np.all(np.linalg.eigvalsh(est) > 0)


def test_gamma_ridge_recovers_planted():
    rng = np.random.default_rng(1)
    D, P, Km1 = 5000, 3, 4
    X = rng.normal(size=(D, P)); Gamma_true = rng.normal(size=(P, Km1))
    M = X @ Gamma_true + 0.01 * rng.normal(size=(D, Km1))
    est = gamma_ridge(M, X, ridge=1e-6)
    assert np.allclose(est, Gamma_true, atol=0.05)


def test_beta_dirichlet_mean_normalizes():
    stats = np.array([[10.0, 0.0, 2.0], [0.0, 5.0, 5.0]])
    beta = beta_dirichlet_mean(stats, eta=0.1)
    assert beta.shape == (2, 3)
    assert np.allclose(beta.sum(axis=1), 1.0)
    assert beta[0, 0] > beta[0, 1]                      # more mass where more counts
```

- [ ] **Step 2: Run to verify they fail** — import error.

- [ ] **Step 3: Implement**

Append to `pg_stm.py`:

```python
def sigma_iw_posterior_mean(scatter, n_docs, *, Psi0, nu0, dim):
    """Inverse-Wishart posterior mean E[Sigma] = (Psi0 + scatter)/(nu0 + n_docs - dim - 1).
    Proper prior (nu0 > dim + 1) => finite PD mean even at n_docs = 0 (the runaway cure)."""
    denom = nu0 + n_docs - dim - 1.0
    return (np.asarray(Psi0, dtype=np.float64) + np.asarray(scatter, dtype=np.float64)) / denom


def gamma_ridge(M, X, *, ridge):
    """Ridge regression of stacked posterior means M (D, K-1) on covariates X (D, P)."""
    X = np.asarray(X, dtype=np.float64); M = np.asarray(M, dtype=np.float64)
    P = X.shape[1]
    return np.linalg.solve(X.T @ X + ridge * np.eye(P), X.T @ M)


def beta_dirichlet_mean(word_topic_stats, *, eta):
    """Row-normalized Dirichlet posterior mean of the (K,V) topic-word matrix."""
    lam = np.asarray(word_topic_stats, dtype=np.float64) + eta
    return lam / lam.sum(axis=1, keepdims=True)
```

- [ ] **Step 4: Run to verify they pass** — PASS (4).

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/pg_stm.py spark-vi/tests/test_pg_stm_updates.py
git commit -m "feat(pg-stm): conjugate global updates — IW Sigma, ridge Gamma, Dirichlet beta"
```

---

### Task 4: Delta-method E[log θ] + token responsibilities

The single VI approximation, isolated and tested against Monte-Carlo. Then the LDA-style
responsibilities that produce the per-doc topic counts.

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/pg_stm.py`
- Test: `spark-vi/tests/test_pg_stm_assignment.py`

**Interfaces — Produces:**
- `expected_log_theta(m, v) -> np.ndarray` — `(K,)` from stick posterior mean `m (K−1,)`,
  marginal var `v (K−1,)`. Formula: with `ls_plus_k = log σ(m_k) − 0.5 v_k s_k`,
  `ls_minus_k = log σ(−m_k) − 0.5 v_k s_k`, `s_k = σ(m_k)(1−σ(m_k))`:
  `E[log θ_k] = ls_plus_k + Σ_{j<k} ls_minus_j` for k<K, and `E[log θ_K] = Σ_j ls_minus_j`.
- `token_responsibilities(doc_indices, elog_theta, elog_beta, allowed) -> (phi, n)` — φ is
  `(len(indices), K)`, `φ_{n,k} ∝ exp(elog_theta_k + elog_beta_{k, w_n})` restricted to
  `allowed`, normalized per token; `n (K,) = Σ_n φ_{n,k}·count_n`.

- [ ] **Step 1: Write the failing tests**

```python
# spark-vi/tests/test_pg_stm_assignment.py
import numpy as np
from spark_vi.models.topic.pg_stm import (
    expected_log_theta, token_responsibilities, stick_to_simplex)


def test_expected_log_theta_matches_montecarlo():
    rng = np.random.default_rng(0)
    m = np.array([0.3, -0.5, 0.1]); v = np.array([0.4, 0.2, 0.6])   # K-1=3 -> K=4
    samples = rng.normal(m, np.sqrt(v), size=(400000, 3))
    logtheta = np.log(np.array([stick_to_simplex(s) for s in samples[:20000]]))
    mc = logtheta.mean(axis=0)
    approx = expected_log_theta(m, v)
    assert np.allclose(approx, mc, atol=3e-3)


def test_expected_log_theta_zero_var_is_exact_logtheta():
    m = np.array([0.3, -0.5, 0.1]); v = np.zeros(3)
    assert np.allclose(expected_log_theta(m, v), np.log(stick_to_simplex(m)), atol=1e-12)


def test_token_responsibilities_normalize_and_respect_gating():
    elog_theta = np.log(np.array([0.4, 0.3, 0.2, 0.1]))
    elog_beta = np.log(np.array([                    # (K=4, V=3)
        [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.34, 0.33, 0.33]]))
    idx = np.array([0, 1]); allowed = np.array([0, 1, 2])   # topic 3 masked out
    phi, n = token_responsibilities(idx, elog_theta, elog_beta, allowed, counts=np.array([2.0, 1.0]))
    assert np.allclose(phi.sum(axis=1), 1.0)
    assert np.allclose(phi[:, 3], 0.0)              # gated-out topic gets zero
    assert abs(n.sum() - 3.0) < 1e-9                # total mass = total tokens
```

- [ ] **Step 2: Run to verify they fail** — import error.

- [ ] **Step 3: Implement**

Append to `pg_stm.py` (note the `token_responsibilities` signature adds `counts=`):

```python
def expected_log_theta(m, v):
    """Delta-method E[log theta] under q(psi_k)=N(m_k, v_k). Second-order Taylor of log-sigma:
    E[log sigma(psi_k)] ~ log sigma(m_k) - 0.5 v_k s_k, s_k = sigma(m_k)(1-sigma(m_k))."""
    m = np.asarray(m, dtype=np.float64); v = np.asarray(v, dtype=np.float64)
    sig = expit(m); s = sig * (1.0 - sig)
    log_sig = np.log(sig); log_1msig = np.log1p(-sig)
    ls_plus = log_sig - 0.5 * v * s          # E[log sigma(psi_k)]
    ls_minus = log_1msig - 0.5 * v * s       # E[log (1-sigma(psi_k))]
    K = m.shape[0] + 1
    out = np.empty(K, dtype=np.float64)
    cum = np.concatenate([[0.0], np.cumsum(ls_minus)])   # cum[k] = sum_{j<k} ls_minus_j
    out[:K - 1] = ls_plus + cum[:K - 1]
    out[K - 1] = cum[K - 1]                               # = sum_j ls_minus_j
    return out


def token_responsibilities(doc_indices, elog_theta, elog_beta, allowed, *, counts):
    """LDA-style responsibilities restricted to the allowed topic set.
    phi_{n,k} ∝ exp(elog_theta_k + elog_beta_{k, w_n}) for k in allowed, else 0."""
    K = elog_theta.shape[0]
    log_unnorm = elog_theta[None, :] + elog_beta[:, doc_indices].T   # (n_tok, K)
    mask = np.full(K, -np.inf); mask[np.asarray(allowed)] = 0.0
    log_unnorm = log_unnorm + mask[None, :]
    log_unnorm -= log_unnorm.max(axis=1, keepdims=True)
    phi = np.exp(log_unnorm); phi /= phi.sum(axis=1, keepdims=True)
    n = (phi * np.asarray(counts, dtype=np.float64)[:, None]).sum(axis=0)
    return phi, n
```

- [ ] **Step 4: Run to verify they pass** — PASS (3).

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/pg_stm.py spark-vi/tests/test_pg_stm_assignment.py
git commit -m "feat(pg-stm): delta-method E[log theta] + gated token responsibilities"
```

---

### Task 5: Nested composition layer (gate + per-block flat stick-breaking)

Compose the Task-1/Task-4 flat primitives into the nested gated structure: a per-group gate
stick splits background vs foreground, then flat stick-breaking within each block. Pure
functions, validated against quadrature — the intricate part is the composed `E[log θ]`.

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/pg_stm.py`
- Test: `spark-vi/tests/test_pg_stm_nested.py`

**Interfaces — Produces** (all operate on a group-g doc's active sticks, split into the three
groups background / gate / foreground; the Task-6 driver handles the global-index mapping):
- `gated_theta(psi_bg, psi_gate, psi_fg) -> theta` — `theta = concat(σ(psi_gate)·stick_to_simplex(psi_bg),
  (1−σ(psi_gate))·stick_to_simplex(psi_fg))`, length `len(psi_bg)+1 + len(psi_fg)+1` = B+m_g. `psi_gate`
  is a scalar.
- `gated_expected_log_theta(m_bg, v_bg, m_gate, v_gate, m_fg, v_fg) -> elog_theta` — length B+m_g:
  background entries `= E[log σ(ψ_gate)] + expected_log_theta(m_bg, v_bg)`, foreground entries
  `= E[log(1−σ(ψ_gate))] + expected_log_theta(m_fg, v_fg)`, where the gate terms use the SAME
  delta method as one stick: `E[log σ(ψ_gate)] = log σ(m_gate) − 0.5 v_gate s + (v_gate²/8) q`,
  `s=σ(m_gate)(1−σ(m_gate))`, `q=−s(1−2σ)²+2s²` (reuse the Task-4 helper — factor the single-stick
  `E[log σ]`/`E[log(1−σ)]` out of `expected_log_theta` so both call it).
- `gated_counts(n_bg, n_fg) -> (gate_a, gate_b, b_bg, b_fg)` — `gate_a = n_bg.sum()`,
  `gate_b = n_bg.sum()+n_fg.sum()`, `b_bg = stick_trials(n_bg)`, `b_fg = stick_trials(n_fg)`.
  (The gate is one binomial: N_bg successes of N; each block uses flat `stick_trials`.)

- [ ] **Step 1: Write the failing tests**

```python
# spark-vi/tests/test_pg_stm_nested.py
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import expit
from spark_vi.models.topic.pg_stm import (
    gated_theta, gated_expected_log_theta, gated_counts,
    stick_to_simplex, stick_trials)


def test_gated_theta_sums_to_one_and_composes():
    rng = np.random.default_rng(0)
    for _ in range(20):
        psi_bg = rng.normal(size=3)      # B=4 background topics
        psi_gate = float(rng.normal())
        psi_fg = rng.normal(size=2)      # m_g=3 foreground topics
        theta = gated_theta(psi_bg, psi_gate, psi_fg)
        assert theta.shape == (4 + 3,)
        assert np.all(theta > 0) and abs(theta.sum() - 1.0) < 1e-12
        # background mass = sigma(gate), foreground mass = 1-sigma(gate)
        assert abs(theta[:4].sum() - expit(psi_gate)) < 1e-12
        assert abs(theta[4:].sum() - (1 - expit(psi_gate))) < 1e-12
        # within-block proportions match the flat map
        assert np.allclose(theta[:4] / theta[:4].sum(), stick_to_simplex(psi_bg))
        assert np.allclose(theta[4:] / theta[4:].sum(), stick_to_simplex(psi_fg))


def test_gated_counts():
    n_bg = np.array([3.0, 0.0, 5.0])     # N_bg=8
    n_fg = np.array([2.0, 4.0])          # N_fg=6
    ga, gb, b_bg, b_fg = gated_counts(n_bg, n_fg)
    assert ga == 8.0 and gb == 14.0
    assert np.allclose(b_bg, stick_trials(n_bg))
    assert np.allclose(b_fg, stick_trials(n_fg))


def _quad_elog_gated(m_bg, v_bg, m_gate, v_gate, m_fg, v_fg, nodes=64):
    # Exact E[log theta] via per-stick Gaussian quadrature (deterministic reference).
    x, w = hermegauss(nodes); w = w / np.sqrt(2 * np.pi)
    def e_log_sig(m, v, sign):  # E[log sigma(sign*psi)], psi~N(m,v)
        return float(w @ np.log(expit(sign * (m + np.sqrt(v) * x))))
    def e_log_theta_flat(m, v):
        K = len(m) + 1
        lp = np.array([e_log_sig(m[j], v[j], +1) for j in range(len(m))])
        lm = np.array([e_log_sig(m[j], v[j], -1) for j in range(len(m))])
        out = np.empty(K); cum = np.concatenate([[0.0], np.cumsum(lm)])
        out[:K-1] = lp + cum[:K-1]; out[K-1] = cum[K-1]; return out
    eg_bg = e_log_sig(m_gate, v_gate, +1)      # E[log sigma(gate)]
    eg_fg = e_log_sig(m_gate, v_gate, -1)      # E[log (1-sigma(gate))]
    return np.concatenate([eg_bg + e_log_theta_flat(m_bg, v_bg),
                           eg_fg + e_log_theta_flat(m_fg, v_fg)])


def test_gated_expected_log_theta_matches_quadrature():
    m_bg = np.array([0.3, -0.5]); v_bg = np.array([0.4, 0.2])
    m_gate, v_gate = 0.2, 0.3
    m_fg = np.array([0.1]); v_fg = np.array([0.5])
    approx = gated_expected_log_theta(m_bg, v_bg, m_gate, v_gate, m_fg, v_fg)
    exact = _quad_elog_gated(m_bg, v_bg, m_gate, v_gate, m_fg, v_fg)
    assert np.allclose(approx, exact, atol=2e-3)   # same delta-method accuracy as Task 4
```

- [ ] **Step 2: Run to verify they fail** — import error for `gated_theta`.

- [ ] **Step 3: Implement.** Add `gated_theta`, `gated_counts`, and `gated_expected_log_theta` to
  `pg_stm.py`. Refactor Task 4's `expected_log_theta` to expose the single-stick delta-method
  `E[log σ(m,v)]` / `E[log(1−σ(m,v))]` as a small helper (e.g. `_elog_sigmoid(m, v, sign)`) that both
  `expected_log_theta` and the gate term in `gated_expected_log_theta` call — DRY, and it keeps the
  gate's approximation identical to the sticks'. Re-run the Task-4 suite to confirm the refactor is
  behavior-preserving (`test_pg_stm_assignment.py` still green).

- [ ] **Step 4: Run to verify they pass** — `pytest tests/test_pg_stm_nested.py tests/test_pg_stm_assignment.py -v` green.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/pg_stm.py spark-vi/tests/test_pg_stm_nested.py
git commit -m "feat(pg-stm): nested composition — gate + per-block stick-breaking (theta, E[log theta], counts)"
```

---

### Task 6: Gated full-batch PG-VI driver + block-Σ IW + planted recovery

Assemble the gated coordinate ascent: global stick layout (K−1 = background ∪ per-group
[gate, foreground]), per-doc active sub-vector E-step under the block-Σ marginal, block-
structured IW M-step, the `sigma_mode` flag. Integration task; correctness gate = recovery.

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/pg_stm.py`
- Test: `spark-vi/tests/test_pg_stm_vi.py`

**Interfaces — Produces:**
- `class PGSTMVI(K, V, partition, *, P, n_iter=200, Psi0_scale=1.0, nu0=None, gamma_ridge=1e-6,
  beta_eta=0.1, sigma_mode="iw", seed=0)`; `fit(docs) -> {"beta": (K,V), "Gamma": (P,K−1),
  "Sigma": (K−1,K−1), "psi_mean": (D,K−1), "psi_var": (D,K−1), "sigma_max_trace": list[float]}`.
- `sigma_mode`: `"iw"` → block `sigma_iw_posterior_mean`; `"mle"` → un-regularized `scatter/n`
  per block fed back (Task-8 isolation only). Production is `"iw"`.

**Stick layout (implement as a small helper `stick_layout(partition)`):** background sticks =
global indices `0..B−2`; group g (in `partition.groups` order) occupies `m_g` consecutive
indices starting at `B−1 + Σ_{g'<g} m_{g'}`: the first is the **gate**, the next `m_g−1` are
the foreground sticks. Total `K−1`. A group-g doc's ACTIVE global stick indices =
`background_slice ∪ {gate_g} ∪ fg_g_slice`; its allowed TOPIC indices =
`partition.allowed_indices(doc.groups)` (background topics ∪ group-g topics).

**Per-iteration coordinate ascent** (consumes Tasks 1–5):
1. E-step per doc (group g): `Sigma_inv_active = safe_inverse(Sigma[ix_(active, active)])`,
   `mu_active = (Gamma.T @ x)[active]`. Inner loop to a small tol:
   responsibilities over allowed topics (`token_responsibilities` with a full-K `elog_theta`
   that is `gated_expected_log_theta` on allowed, −inf elsewhere) → counts `n_allowed`, split
   into `n_bg`/`n_fg` → `gated_counts` → per-stick `omega_expectation(b, c)` with
   `b = [gate_b, b_bg..., b_fg...]` ordered to match `active`, `c = sqrt(m² + diag(V))` →
   `psi_posterior(...)` over active sticks (with `n`/`b` being the successes/trials per active
   stick: gate success = `gate_a`, background/foreground successes = the block counts) →
   `gated_expected_log_theta`.
2. Accumulate: word-topic stats (β); `(X, M=stacked active means placed into full K−1 with
   inactive left at prior mean)` for Γ; block-wise `scatter` and per-block doc counts for Σ —
   background block from ALL docs (every doc is active on background), each group block
   `[gate_g, fg_g]` from that group's docs only, background↔group cross-block from that group's
   docs, group↔group' left at prior.
3. M-step: `beta_dirichlet_mean`; `gamma_ridge`; per block `sigma_iw_posterior_mean` (or
   `scatter/n` if `sigma_mode="mle"`); assemble the block-Σ (group↔group' at prior/0). Record
   `max|Σ|` into `sigma_max_trace`.

Numerical guard (Task-1 watch-item): clip `psi` to a safe range (e.g. ±30) before
`stick_to_simplex`/`gated_theta` to avoid the sigmoid-underflow `nan` on any transient extreme
iterate; log if the clip ever trips.

- [ ] **Step 1: Write the failing test**

```python
# spark-vi/tests/test_pg_stm_vi.py
import numpy as np
from spark_vi.models.topic.pg_stm import PGSTMVI
from tests._stm_synth import gated_ln_corpus, planted_recovery, foreground_recovers_group


def _corpus(seed=0):
    return gated_ln_corpus(group_weights={"A": 0.5, "B": 0.5}, fg_per_group=1, bg_k=2,
                           V=60, D=600, doc_len=40, seed=seed)


def test_pgvi_recovers_planted_structure():
    docs, planted, part = _corpus(seed=0)
    P = docs[0].x.shape[0]
    out = PGSTMVI(K=part.K, V=60, partition=part, P=P, n_iter=150, seed=0).fit(docs)
    assert planted_recovery(out["beta"], planted["beta"]) >= 0.75
    for g in part.groups:
        assert foreground_recovers_group(out["beta"], part, g, planted["beta"])
    assert out["Sigma"].shape == (part.K - 1, part.K - 1)
    assert np.all(np.isfinite(out["Sigma"])) and np.max(np.abs(out["Sigma"])) < 1e3
    assert len(out["sigma_max_trace"]) == 150
```

- [ ] **Step 2: Run to verify it fails** — `cannot import name 'PGSTMVI'`.

- [ ] **Step 3: Implement `stick_layout` + `PGSTMVI`.** Compose the Task 1–5 primitives per the
  recipe above. Every update a pure function of (globals, per-doc suff-stats), so it ports to
  SVI. Init `beta` from smoothed random counts, `Gamma=0`, `Sigma=I` (K−1). Cache
  `active`/`Sigma_inv_active` per distinct group. The recovery test is the correctness gate; if
  recovery is marginal, raise `n_iter` or E-step inner rounds — do NOT loosen the 0.75 threshold
  without recording why.

- [ ] **Step 4: Run to verify it passes** — `pytest tests/test_pg_stm_vi.py -v` PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/pg_stm.py spark-vi/tests/test_pg_stm_vi.py
git commit -m "feat(pg-stm): gated PG-VI driver (nested, block-Sigma IW, sigma_mode) + recovery"
```

---

### Task 7: Gated PG-Gibbs cross-check + VI≈Gibbs

Exact blocked Gibbs over the nested structure; audits the delta-method and the block-Σ
posterior.

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/pg_stm.py`
- Test: `spark-vi/tests/test_pg_stm_gibbs.py`

**Interfaces — Produces:** `pg_stm_gibbs(docs, K, V, partition, *, P, n_iter=400, burn=200,
seed=0, ...) -> {"beta", "Gamma", "Sigma", "Sigma_samples"}`.

Per sweep, per doc (group g): sample `z` per token from `Categorical(θ_d·β_{·,w})` with
`θ_d = gated_theta(ψ_bg, ψ_gate_g, ψ_fg_g)` computed **exactly** from current ψ (no delta
method); form `n_bg`/`n_fg`; `gated_counts`; sample `ω = omega_sample(b, ψ_active, rng)` per
active stick; sample `ψ_active ~ N(m, V)` from `psi_posterior`; sample block-Σ from
`scipy.stats.invwishart` (background block, per-group `[gate,fg]` block, cross via the joint);
conjugate Γ, β. Keep post-burn Σ samples. Reuse the Task-6 `stick_layout`/`active` handling.

- [ ] **Step 1: Write the failing tests**

```python
# spark-vi/tests/test_pg_stm_gibbs.py
import numpy as np
from spark_vi.models.topic.pg_stm import pg_stm_gibbs, PGSTMVI
from tests._stm_synth import gated_ln_corpus, planted_recovery


def _corpus(seed=1):
    return gated_ln_corpus(group_weights={"A": 0.5, "B": 0.5}, fg_per_group=1, bg_k=2,
                           V=60, D=500, doc_len=40, seed=seed)


def test_gibbs_recovers_planted():
    docs, planted, part = _corpus()
    P = docs[0].x.shape[0]
    out = pg_stm_gibbs(docs, K=part.K, V=60, partition=part, P=P, n_iter=400, burn=200, seed=0)
    assert planted_recovery(out["beta"], planted["beta"]) >= 0.75


def test_vi_matches_gibbs_on_sigma():
    docs, planted, part = _corpus()
    P = docs[0].x.shape[0]
    vi = PGSTMVI(K=part.K, V=60, partition=part, P=P, n_iter=150, seed=0).fit(docs)
    gb = pg_stm_gibbs(docs, K=part.K, V=60, partition=part, P=P, n_iter=400, burn=200, seed=0)
    def corr(S):
        d = np.sqrt(np.diag(S)); return S / np.outer(d, d)
    # Compare the SHARED background block's correlation (present in every doc, best-identified).
    B = len(part.background_indices())
    assert np.allclose(corr(vi["Sigma"])[:B-1, :B-1], corr(gb["Sigma"])[:B-1, :B-1], atol=0.15)
```

- [ ] **Step 2: Run to verify they fail** — import error.

- [ ] **Step 3: Implement `pg_stm_gibbs`** per the sweep above, reusing Task 1/2/5 primitives,
  the Task-6 `stick_layout`, and `scipy.stats.invwishart`.

- [ ] **Step 4: Run to verify they pass** — PASS (2). Gibbs is stochastic; atol=0.15 on the
  background-block correlation is deliberately loose. Raise `n_iter`/`burn` if it flakes; a
  genuine VI–Gibbs correlation mismatch is a real finding to report, not to paper over.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/pg_stm.py spark-vi/tests/test_pg_stm_gibbs.py
git commit -m "feat(pg-stm): gated PG-Gibbs cross-check + VI-vs-Gibbs Sigma agreement"
```

---

### Task 8: Gated runaway checkpoint (the milestone-1 gate)

Reproduce the Σ runaway and show the IW posterior cures it, isolating the estimator at fixed
link (`sigma_mode` mle vs iw on the same gated model).

**Files:**
- Create: `spark-vi/tests/test_pg_stm_runaway.py`
- Create: `docs/experiments/0049-pg-stm-runaway-checkpoint.md`

**Interfaces — Consumes:** `PGSTMVI` (Task 6); `gated_ln_corpus`, `fit_stm`, `final_sigma_range`.

- [ ] **Step 1: Write the gating tests**

```python
# spark-vi/tests/test_pg_stm_runaway.py
import numpy as np
from spark_vi.models.topic.pg_stm import PGSTMVI
from tests._stm_synth import gated_ln_corpus, fit_stm, final_sigma_range


def _scarce_corpus(seed=0):
    # Group B's foreground used by ~3% of docs (ess ~ 15-30): the weakly-identified-variance
    # regime that drove the point-EM runaway (insight 0033).
    return gated_ln_corpus(group_weights={"A": 0.97, "B": 0.03}, fg_per_group=1, bg_k=2,
                           V=60, D=1000, doc_len=40, seed=seed)


def test_DECISIVE_estimator_isolation_mle_diverges_iw_bounded():
    # SAME gated nested model, SAME E-step; vary ONLY the Sigma M-step: MLE (point estimate
    # fed back, no trust region) vs IW (proper posterior). Isolates the estimator from the link.
    docs, planted, part = _scarce_corpus()
    P = docs[0].x.shape[0]
    mle = PGSTMVI(K=part.K, V=60, partition=part, P=P, n_iter=150, sigma_mode="mle", seed=0).fit(docs)
    iw = PGSTMVI(K=part.K, V=60, partition=part, P=P, n_iter=150, sigma_mode="iw", seed=0).fit(docs)
    assert max(mle["sigma_max_trace"]) > 1e3                    # MLE diverges over iterations
    assert np.all(np.isfinite(iw["Sigma"])) and np.max(np.abs(iw["Sigma"])) < 1e2   # IW bounded


def test_scarce_topic_gets_wide_not_divergent_posterior_under_iw():
    docs, planted, part = _scarce_corpus()
    P = docs[0].x.shape[0]
    out = PGSTMVI(K=part.K, V=60, partition=part, P=P, n_iter=150, sigma_mode="iw", seed=0).fit(docs)
    pv = out["psi_var"]
    assert np.all(np.isfinite(pv))
    assert pv.max() < 1e2 and pv.max() > pv.mean()   # elevated but bounded


def test_CONTEXT_current_softmax_point_em_also_blows_up_here():
    # Context only (link AND estimator both differ from PG-VI): confirms the regime triggers
    # the DOCUMENTED softmax point-EM pathology.
    docs, planted, part = _scarce_corpus()
    gp = fit_stm(docs, K=part.K, V=60, sigma_init=1.0, n_iter=200, seed=0,
                 estimate_sigma_diagonal=True)
    lo, hi = final_sigma_range(gp)
    assert hi > 1e3
```

- [ ] **Step 2: Run to verify.** The DECISIVE test drives the milestone: `sigma_mode="mle"`
  must diverge (`sigma_max_trace` explodes) and `"iw"` stay bounded — same gated model, same
  E-step, only the Σ M-step differs. If `"mle"` does NOT diverge, increase scarcity (lower B's
  share, raise D) until the un-regularized point estimate blows up. The CONTEXT test should pass
  immediately; if not, it's context only and does NOT block the milestone.

- [ ] **Step 3: No new implementation** — the gate over Task 6's `"iw"` path. If `"iw"` fails to
  stay bounded, the defect is in Task 6 (or the bet); investigate with systematic-debugging, do
  NOT weaken the bound thresholds.

- [ ] **Step 4: Run the full gate** — `pytest tests/test_pg_stm_runaway.py -v`. Milestone 1.

- [ ] **Step 5: Record the checkpoint + commit.** Write
  `docs/experiments/0049-pg-stm-runaway-checkpoint.md`: the three synthetic results (MLE
  `max sigma_max_trace`, IW `max|Σ|`, VI–Gibbs background-block correlation atol, recovery
  score) and the **checkpoint verdict** — bet holds → proceed to sub-project #2; or bet fails →
  the publishable stop. Numeric values are the one section filled on completion.

```bash
git add spark-vi/tests/test_pg_stm_runaway.py docs/experiments/0049-pg-stm-runaway-checkpoint.md
git commit -m "test(pg-stm): gated runaway reproduction + milestone-1 checkpoint gate"
```

---

## Post-implementation (controller)

- All 8 task tests green = milestone 1 reached. Read the checkpoint verdict in exp 0049.
- **Bet holds** → the design's sub-project #2 (distributed SVI) is the next brainstorm.
- **Bet fails** (PG-VI `"iw"` also diverges / VI≠Gibbs on the shared background block) → STOP;
  publishable negative result. Write the insight; do not proceed to #2.
- `polyagamma` must be added to the cluster image before sub-project #2 (it ships wheels).

## Self-Review notes

- **Spec coverage:** flat stick-breaking primitives (T1), PG ω + ψ-Gaussian (T2), IW Σ + Γ + β
  (T3), delta-method E[log θ] (T4) — all reused as block primitives; nested composition (gate +
  per-block, T5); gated full-batch on-path PG-VI + block-Σ IW + recovery (T6); gated Gibbs +
  VI≈Gibbs (T7); gated runaway cure + checkpoint (T8). Nested = K−1 sticks (background ∪
  per-group [gate, foreground]); a doc uses |allowed|−1. No reference topic.
- **Type consistency:** flat sticks (Ki−1) within a block of Ki topics; `gated_theta` length
  B+m_g; ψ/Σ/Γ are (K−1); `PGSTMVI.fit` and `pg_stm_gibbs` return the same `Sigma` (K−1,K−1)
  the tests compare; `sigma_mode`/`sigma_max_trace` consistent across T6/T8.
- **Research honesty:** T1–T5 formulas validated against brute-force/quadrature references;
  T6/T7 correctness is end-to-end recovery + VI≈Gibbs; T8 is the estimator-isolation gate. The
  derive-to-pass-a-test parts are T6's gated E-step assembly + block-Σ accumulation (recipe in
  the design; recovery is the check) and T7's Gibbs sweep.
- **Parameterization caveat (unchanged):** `gated_ln_corpus` is softmax-planted, so only β
  recovery is cross-model; all Σ checks are link-internal (VI≈Gibbs on the shared background
  block, boundedness, MLE-vs-IW).
