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
- Tests (per concern): `spark-vi/tests/test_pg_stm_link.py` (T1),
  `test_pg_stm_conditionals.py` (T2), `test_pg_stm_updates.py` (T3),
  `test_pg_stm_assignment.py` (T4), `test_pg_stm_vi.py` (T5),
  `test_pg_stm_gibbs.py` (T6), `test_pg_stm_runaway.py` (T7).
- `spark-vi/pyproject.toml` (or the test env's requirements) — add `polyagamma`.

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

### Task 5: Full-batch PG-VI driver + planted recovery

Assemble the coordinate ascent (the Tier-3 SVI kernel at batch=full). This is an integration
task; its correctness test is end-to-end **structure recovery** on a planted logistic-normal
corpus.

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/pg_stm.py`
- Test: `spark-vi/tests/test_pg_stm_vi.py`

**Interfaces — Produces:**
- `class PGSTMVI` with `__init__(self, K, V, partition, *, P, n_iter=200, Psi0_scale=1.0,
  nu0=None, gamma_ridge=1e-6, beta_eta=0.1, sigma_mode="iw", seed=0)` and
  `fit(self, docs) -> dict` returning `{"beta": (K,V), "Gamma": (P,K−1), "Sigma": (K−1,K−1),
  "psi_mean": (D,K−1), "psi_var": (D,K−1), "sigma_max_trace": list[float]}` where
  `sigma_max_trace` is `max|Σ|` per iteration (for the T7 divergence check).
- `sigma_mode` selects the Σ M-step: `"iw"` (default) → `sigma_iw_posterior_mean` (proper
  posterior); `"mle"` → the un-regularized point estimate `scatter_block / n_block` fed back
  each iteration (the pre-ADR-0034 behavior). The flag exists ONLY to isolate the estimator
  in T7 at fixed link/E-step; production is always `"iw"`.

**The coordinate ascent per iteration** (consumes T1–T4):
1. E-step per doc: use current `(Gamma, Sigma, beta)`. Compute `mu_d = Gamma.T @ x_d`,
   `elog_beta = log(beta)` (or Dirichlet E[log β]); iterate a few inner rounds of
   {responsibilities → counts `n` → `b = stick_trials(n)` → `c = sqrt(m² + diag(V))` →
   `omega = omega_expectation(b, c)` → `(m, V) = psi_posterior(n, b, mu, Sigma_inv, omega)` →
   `elog_theta = expected_log_theta(m, diag(V))`} to a small tol. Restrict all per-doc arrays
   to `allowed = partition.allowed_indices(doc.groups)` and the corresponding stick indices.
2. Accumulate global stats: word-topic stats for β; `(X, M)` for Γ; block-wise
   `scatter_block = Σ_d (e_d e_dᵀ + V_d)` and per-block doc counts for Σ.
3. M-step: `beta = beta_dirichlet_mean(...)`; `Gamma = gamma_ridge(M, X, ridge=...)`;
   for each block, `Sigma[block] = sigma_iw_posterior_mean(scatter_block, n_block, ...)`;
   cross-group entries left at 0 (never co-active).

Gating detail: the doc's stick set is the allowed topics in block order; `psi_posterior`
operates on that sub-vector with `Sigma_inv` the inverse of the allowed sub-block (marginal
precision — mirror the current STM's `safe_inverse(Sigma[ix_(allowed,allowed)])`).

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
    model = PGSTMVI(K=part.K, V=60, partition=part, P=P, n_iter=150, seed=0)
    out = model.fit(docs)
    assert planted_recovery(out["beta"], planted["beta"]) >= 0.75
    for g in part.groups:
        assert foreground_recovers_group(out["beta"], part, g, planted["beta"])
    # Sigma is a valid (K-1)x(K-1) PD-ish correlation-carrying matrix, and BOUNDED
    assert out["Sigma"].shape == (part.K - 1, part.K - 1)
    assert np.all(np.isfinite(out["Sigma"]))
    assert np.max(np.abs(out["Sigma"])) < 1e3           # bounded (no runaway on clean data)
```

- [ ] **Step 2: Run to verify it fails** — `cannot import name 'PGSTMVI'`.

- [ ] **Step 3: Implement `PGSTMVI`.** Compose the T1–T4 primitives into the coordinate
  ascent described above. Keep every update a pure function of (globals, per-doc suff-stats)
  so it ports to SVI. Initialize `beta` from a smoothed random count matrix, `Gamma=0`,
  `Sigma=I`. Cache `allowed`/`Sigma_inv_allowed` per distinct group-set. Full reference for
  the block accumulation + per-doc E-step loop is in the design doc's Inference section; the
  recovery test is the correctness gate. Log per-iteration max|Σ| and a convergence delta.

- [ ] **Step 4: Run to verify it passes**

Run: `cd spark-vi && python -m pytest tests/test_pg_stm_vi.py -v`
Expected: PASS. (If recovery is marginal, raise `n_iter` or the E-step inner rounds; do NOT
loosen the 0.75 recovery threshold without recording why.)

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/pg_stm.py spark-vi/tests/test_pg_stm_vi.py
git commit -m "feat(pg-stm): full-batch PG-VI driver (SVI kernel) + planted recovery"
```

---

### Task 6: Exact PG-Gibbs cross-check + VI≈Gibbs agreement

The audit. An exact blocked Gibbs sampler over (z, ψ, ω, Σ, Γ, β); its job is to confirm the
VI's delta-method and Σ posterior aren't artifacts.

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/pg_stm.py`
- Test: `spark-vi/tests/test_pg_stm_gibbs.py`

**Interfaces — Produces:**
- `pg_stm_gibbs(docs, K, V, partition, *, P, n_iter=400, burn=200, ...) -> dict` — returns
  posterior means `{"beta", "Gamma", "Sigma", "Sigma_samples": (n_kept, K−1, K−1)}`.

Per sweep: sample `z` per token from `Categorical(θ_d · β_{·,w})` with `θ_d =
stick_to_simplex(ψ_d)` (**exact θ, no delta-method**); form counts `n`, `b=stick_trials(n)`;
sample `ω_d = omega_sample(b, ψ_d[allowed], rng)`; sample `ψ_d ~ N(m, V)` from `psi_posterior`;
sample `Σ ~ IW(Psi0 + scatter, nu0 + D)` block-wise (`scipy.stats.invwishart`); sample/`update
Γ, β` conjugately. Keep post-burn Σ samples.

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
    # Correlation structure agrees: compare standardized Sigma (correlation) off-diagonals.
    def corr(S):
        d = np.sqrt(np.diag(S)); return S / np.outer(d, d)
    assert np.allclose(corr(vi["Sigma"]), corr(gb["Sigma"]), atol=0.15)
```

- [ ] **Step 2: Run to verify they fail** — import error.

- [ ] **Step 3: Implement `pg_stm_gibbs`** per the sweep above, reusing T1/T2 primitives and
  `scipy.stats.invwishart`. Share the block/`allowed` handling with `PGSTMVI`.

- [ ] **Step 4: Run to verify they pass** — PASS (2). (Gibbs is stochastic; the atol=0.15 on
  the correlation is deliberately loose. If it fails, first raise `n_iter`/`burn`; a genuine
  VI–Gibbs *correlation* mismatch is a real finding to report, not to paper over.)

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/pg_stm.py spark-vi/tests/test_pg_stm_gibbs.py
git commit -m "feat(pg-stm): exact PG-Gibbs cross-check + VI-vs-Gibbs Sigma agreement"
```

---

### Task 7: The runaway reproduction + the milestone-1 checkpoint

The decisive test: reproduce the Σ runaway with the current point-estimate EM, and show
PG-VI stays bounded and gives the weakly-identified topic a *wide* posterior.

**Files:**
- Create: `spark-vi/tests/test_pg_stm_runaway.py`
- Create: `docs/experiments/0049-pg-stm-runaway-checkpoint.md` (records the checkpoint result)

**Interfaces — Consumes:** `PGSTMVI` (T5); `fit_stm` and `final_sigma_range` (from
`tests/_stm_synth.py`); `gated_ln_corpus` with a doc-scarce planted topic.

- [ ] **Step 1: Write the failing/gating tests**

```python
# spark-vi/tests/test_pg_stm_runaway.py
import numpy as np
from spark_vi.models.topic.pg_stm import PGSTMVI
from tests._stm_synth import gated_ln_corpus, fit_stm, final_sigma_range


def _scarce_corpus(seed=0):
    # A group whose foreground topic is used by very few docs (ess ~ 15): the weakly-
    # identified-variance regime that drove the point-EM runaway (insight 0033).
    return gated_ln_corpus(group_weights={"A": 0.97, "B": 0.03}, fg_per_group=1, bg_k=2,
                           V=60, D=1000, doc_len=40, seed=seed)


def test_DECISIVE_estimator_isolation_mle_diverges_iw_bounded():
    # The clean test: SAME stick-breaking model, SAME E-step, vary ONLY the Sigma M-step.
    # MLE (point estimate fed back, no trust region) vs IW (proper posterior). Isolates the
    # ESTIMATOR from the link — the crux of the whole bet.
    docs, planted, part = _scarce_corpus()
    P = docs[0].x.shape[0]
    mle = PGSTMVI(K=part.K, V=60, partition=part, P=P, n_iter=150, sigma_mode="mle", seed=0).fit(docs)
    iw = PGSTMVI(K=part.K, V=60, partition=part, P=P, n_iter=150, sigma_mode="iw", seed=0).fit(docs)
    assert max(mle["sigma_max_trace"]) > 1e3                    # MLE diverges over iterations
    assert np.all(np.isfinite(iw["Sigma"])) and np.max(np.abs(iw["Sigma"])) < 1e2   # IW bounded


def test_scarce_topic_gets_wide_not_divergent_posterior_under_iw():
    # Under IW, the weakly-identified stick shows a WIDE but finite per-doc posterior variance
    # (honest uncertainty), not a divergent point.
    docs, planted, part = _scarce_corpus()
    P = docs[0].x.shape[0]
    out = PGSTMVI(K=part.K, V=60, partition=part, P=P, n_iter=150, sigma_mode="iw", seed=0).fit(docs)
    pv = out["psi_var"]
    assert np.all(np.isfinite(pv))
    assert pv.max() < 1e2 and pv.max() > pv.mean()  # elevated but bounded


def test_CONTEXT_current_softmax_point_em_also_blows_up_here():
    # Secondary/context (NOT the isolation — link AND estimator both differ from PG-VI):
    # confirms this regime genuinely triggers the DOCUMENTED softmax point-EM pathology.
    docs, planted, part = _scarce_corpus()
    gp = fit_stm(docs, K=part.K, V=60, sigma_init=1.0, n_iter=200, seed=0,
                 estimate_sigma_diagonal=True)     # UN-PINNED: the config that blew up
    lo, hi = final_sigma_range(gp)
    assert hi > 1e3
```

- [ ] **Step 2: Run to verify.** The DECISIVE test drives the milestone: `sigma_mode="mle"`
  must reproduce a divergence (`sigma_max_trace` explodes) and `"iw"` must stay bounded — same
  link, same E-step, only the Σ M-step differs, so a pass is unambiguous evidence the *proper
  prior/posterior* is what cures it. If `"mle"` does NOT diverge on this regime, increase
  scarcity (lower B's share, raise D) until the un-regularized point estimate blows up — the
  divergence must be real for the contrast to mean anything. The CONTEXT test (softmax
  point-EM) should pass immediately; if it doesn't, it is context only and does NOT block the
  milestone (note it and move on — the decisive test is the isolation one).

- [ ] **Step 3: No new implementation** — this task is the gate over T5's `"iw"` path. If the
  `"iw"` arm fails to stay bounded, the defect is in T5 (or the bet itself); investigate with
  systematic-debugging, do not weaken the bound thresholds.

- [ ] **Step 4: Run the full gate**

Run: `cd spark-vi && python -m pytest tests/test_pg_stm_runaway.py -v`
Expected: point-EM runs away (PASS on the divergence assertion), PG-VI bounded + wide
posterior (PASS). This is milestone 1.

- [ ] **Step 5: Record the checkpoint + commit**

Write `docs/experiments/0049-pg-stm-runaway-checkpoint.md`: the three synthetic results
(runaway reproduced at hi=<value>; PG-VI max|Σ|=<value>; VI–Gibbs correlation atol; recovery
score), and the **checkpoint verdict** — bet holds → proceed to sub-project #2; or bet fails
→ the publishable stop. Leave the numeric values as the one section filled on completion.

```bash
git add spark-vi/tests/test_pg_stm_runaway.py docs/experiments/0049-pg-stm-runaway-checkpoint.md
git commit -m "test(pg-stm): runaway reproduction + milestone-1 checkpoint gate"
```

---

## Post-implementation (controller)

- All 7 task tests green = milestone 1 reached. Read the checkpoint verdict in exp 0049.
- **Bet holds** → the design's sub-project #2 (distributed SVI) is the next brainstorm.
- **Bet fails** (PG-VI also diverges / VI≠Gibbs) → STOP; that is the publishable negative
  result. Write the insight; do not proceed to #2.
- `polyagamma` must be added to the cluster image before sub-project #2 (it ships wheels).

## Self-Review notes

- **Spec coverage:** stick-breaking link (T1), PG ω + ψ-Gaussian (T2), IW Σ + Γ + β (T3),
  delta-method E[log θ] (T4), full-batch on-path PG-VI + recovery (T5), Gibbs cross-check +
  VI≈Gibbs (T6), runaway cure + checkpoint (T7). No reference topic (Σ,Γ are K−1 throughout).
  Block-structured Σ handled in T5's accumulation. `polyagamma` dependency in T2. All covered.
- **Type consistency:** ψ/sticks are (K−1)-dim, θ/β rows are K-dim, Σ/Γ are (K−1); `stick_trials`
  returns (K−1); `omega_expectation`/`psi_posterior` operate on the doc's allowed stick sub-set;
  `PGSTMVI.fit` and `pg_stm_gibbs` return the same `Sigma` (K−1,K−1) shape the tests compare.
- **Research honesty:** T2/T4 formulas are validated against brute-force/Monte-Carlo references
  (not self-consistency); T5/T6 correctness is end-to-end recovery + VI≈Gibbs; T7 is the gate.
  The one place the implementer must derive-to-pass-a-test (not transcribe) is T5's E-step inner
  loop and block accumulation — the design doc gives the recipe, the recovery test is the check.
