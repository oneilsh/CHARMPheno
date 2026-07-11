# Multivariate-t Per-Document Scale (c, ν) Diagnostic — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a flagged, no-refit diagnostic that calibrates a per-document
concentration scale s_d under a multivariate-t prior — a held-out (c, ν) sweep
emitting the calibrated pair plus two falsifiable readouts — without touching
the shipped `eta_scale` export.

**Architecture:** Per-document explicit EM over (η, s_d) reusing the existing
gated Laplace E-step; a 2-D held-out sweep mirroring
`corpus_heldout_scale_sweep_gated`; a driver flag writing `t_prior_scale.json`.
Additive throughout — the ν=∞ path reproduces today's single-c model, so nothing
existing changes behavior.

**Tech Stack:** Python, NumPy, SciPy (`scipy.optimize`), PySpark RDD
(`mapPartitions`/`treeReduce`), pytest.

**Design doc:** `docs/superpowers/specs/2026-07-10-tprior-per-document-scale-design.md`

## Global Constraints

- **No hardcoded scale.** The sweep emits (c*, ν*) per corpus; never introduce a
  scale constant. The bias-corrected ~6 stays a research finding, not code.
- **No baked-in verdicts.** Readouts emit numbers only (spreads, quantiles) — no
  thresholds, no pass/fail. Same no-verdict contract as the dedup gate.
- **ν=∞ nests the current model exactly** (within numerical tol) — this is a
  test, not a comment.
- **Inference vs scoring role split** (as in `corpus_heldout_scale_sweep_gated`):
  INFERENCE uses `expElogbeta` (exp-digamma of lambda); SCORING the held-out
  predictive uses `beta_prob = lambda/rowsum` (E[β]) via `_predictive_loglik`.
  Conflating them silently miscalibrates the scale.
- **Common split across knobs.** The per-doc visible/held split seed is
  `seed + doc_index`, independent of c, ν, and f — every knob sees the identical
  split (controlled comparison). Warm-start is the ONLY thing that varies the
  optimizer; it must be result-preserving.
- **`_json_safe` at the distributed-function boundary** (numpy scalars/arrays in
  the summary broke a prior cluster run) — import from
  `spark_vi.eval.topic.concentration_heterogeneity`.
- **Add `t_prior_scale.json` to the driver zip optional-files loop** (the exact
  omission that broke two prior cluster runs).

## s_d closed form (used verbatim in Tasks 2–4)

With μ_allowed = Γ[:, allowed]ᵀ x, diff = η̂[allowed] − μ_allowed,
q_R = diffᵀ · Rinv_allowed · diff (η pinned to 0 at the reference position),
and K_free = |allowed| − (1 if reference else 0):

```
ŝ_d = (ν + q_R / c) / (ν + K_free + 2)
```

Derivation (conditional posterior s_d | η is Inverse-Gamma((ν+K_free)/2,
(ν + q_R/c)/2); its mode is scale/(shape+1)). As ν→∞, ŝ_d→1 (Gaussian nesting).

## File Structure

- `spark-vi/spark_vi/models/topic/stm.py` — add optional `eta_init` warm-start
  arg to `_stm_doc_inference` (Task 1). This is the only edit to the core model
  file; strictly additive (`None` = current behavior).
- `spark-vi/spark_vi/mllib/topic/stm.py` — add `_stm_doc_inference_tprior`
  (Task 2), `corpus_tprior_scale_sweep_gated` + internal `_c_sweep_at_nu`
  (Task 3), `corpus_tprior_scale_sweep_gated_rdd` (Task 4). Same file as the
  existing gated sweeps they mirror.
- `analysis/cloud/build_dashboard_cloud.py` — add the `BUILD_T_PRIOR_SCALE`
  block and the zip-list entry (Task 5).
- `docs/experiments/0048-stm-population-cancer-tprior-scale.md` — cluster run doc
  (Task 5).
- Tests: `spark-vi/tests/test_stm_doc_inference_warmstart.py` (Task 1),
  `spark-vi/tests/test_tprior_doc_inference.py` (Task 2),
  `spark-vi/tests/test_tprior_scale_sweep.py` (Tasks 3–4),
  `analysis/cloud/tests/test_tprior_driver.py` (Task 5).

## Test environment

Run pytest from `spark-vi/` for the library tests
(`cd spark-vi && python -m pytest tests/<file> -v`). The synthetic fixtures live
in `spark-vi/tests/_stm_synth.py` (`synthetic_gated_corpus`, `fit_stm`). The
RDD-parity tests use the existing `spark` pytest fixture
(`spark.sparkContext.parallelize(docs, numSlices=3)`), exactly as
`spark-vi/tests/test_heldout_scale_sweep_marginalized.py` does. Driver tests run
from repo root (`python -m pytest analysis/cloud/tests/<file> -v`).

---

### Task 1: Warm-start arg on `_stm_doc_inference`

The EM η-step (Task 2) and the grid warm-start (Task 3) need to seed the L-BFGS
solve from a previous solution instead of cold zeros. Add an optional full-K
`eta_init`; `None` preserves the exact current path (cold zeros), so every
existing caller is byte-identical.

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/stm.py:231-330` (`_stm_doc_inference`)
- Test: `spark-vi/tests/test_stm_doc_inference_warmstart.py`

**Interfaces:**
- Consumes: existing `_stm_doc_inference(*, indices, counts, expElogbeta, Gamma, Sigma_inv_allowed, x, max_iter=50, tol=1e-4, allowed=None, reference=None) -> (eta_hat, nu_d, nit)`.
- Produces: same, plus keyword `eta_init: np.ndarray | None = None` (a full-K
  vector; only the `allowed` entries are read). `None` ⇒ zeros (unchanged).

- [ ] **Step 1: Write the failing tests**

Create `spark-vi/tests/test_stm_doc_inference_warmstart.py`:

```python
from __future__ import annotations

import numpy as np

from spark_vi.models.topic.stm import _stm_doc_inference
from spark_vi.models.topic._linalg import safe_inverse


def _setup(K=4, V=12, reference=None):
    rng = np.random.default_rng(0)
    beta = np.full((K, V), 1e-3)
    blk = V // K
    for k in range(K):
        beta[k, k * blk:(k + 1) * blk] += 2.0
    beta /= beta.sum(axis=1, keepdims=True)
    lam = beta * (500.0 * V) + 0.01
    from scipy.special import digamma
    expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))
    Gamma = np.zeros((1, K))
    R = np.eye(K)
    allowed = np.arange(K, dtype=np.int64)
    Sigma_inv_allowed = (1.0 / 4.0) * safe_inverse(R)
    indices = np.array([0, 1, blk, blk + 1, 2 * blk], dtype=np.int64)
    counts = np.array([3.0, 2.0, 5.0, 1.0, 4.0])
    x = np.array([1.0])
    return dict(indices=indices, counts=counts, expElogbeta=expElogbeta,
                Gamma=Gamma, Sigma_inv_allowed=Sigma_inv_allowed, x=x,
                allowed=allowed, reference=reference)


def test_eta_init_none_matches_explicit_zeros():
    kw = _setup()
    a = _stm_doc_inference(**kw)
    b = _stm_doc_inference(**kw, eta_init=np.zeros(4))
    assert np.allclose(a[0][kw["allowed"]], b[0][kw["allowed"]], atol=1e-8)


def test_warm_start_reaches_same_mode():
    kw = _setup()
    cold = _stm_doc_inference(**kw)
    warm = _stm_doc_inference(**kw, eta_init=cold[0])   # seed from the solution
    assert np.allclose(cold[0][kw["allowed"]], warm[0][kw["allowed"]], atol=1e-5)


def test_warm_start_reaches_same_mode_reference():
    kw = _setup(reference=0)
    cold = _stm_doc_inference(**kw)
    warm = _stm_doc_inference(**kw, eta_init=cold[0])
    assert np.allclose(cold[0][kw["allowed"]], warm[0][kw["allowed"]], atol=1e-5)
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd spark-vi && python -m pytest tests/test_stm_doc_inference_warmstart.py -v`
Expected: FAIL — `_stm_doc_inference() got an unexpected keyword argument 'eta_init'`.

- [ ] **Step 3: Implement the warm-start arg**

In `_stm_doc_inference`, add `eta_init: np.ndarray | None = None` to the
signature (after `reference`). Build the L-BFGS starting point from it. Two spots:

Non-reference branch (currently `eta0 = np.zeros(n_sub, ...)` at line ~283):
```python
        if eta_init is None:
            eta0 = np.zeros(n_sub, dtype=np.float64)
        else:
            eta0 = np.asarray(eta_init, dtype=np.float64)[allowed].copy()
```

Reference branch (currently `x0=np.zeros(free.shape[0], ...)` at line ~317):
```python
    if eta_init is None:
        x0 = np.zeros(free.shape[0], dtype=np.float64)
    else:
        x0 = np.asarray(eta_init, dtype=np.float64)[allowed][free].copy()
    result = minimize(f_free, x0=x0, jac=g_free, method="L-BFGS-B",
                      options={"maxiter": max_iter, "gtol": tol})
```

Update the docstring with one line: `eta_init` (optional full-K vector) warm-starts
the L-BFGS from the given mode over the allowed (and, under reference, free) set;
`None` uses cold zeros (the default, byte-identical to prior behavior).

- [ ] **Step 4: Run to verify they pass**

Run: `cd spark-vi && python -m pytest tests/test_stm_doc_inference_warmstart.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Guard the existing callers**

Run the STM suites that exercise `_stm_doc_inference` to confirm no regression:
`cd spark-vi && python -m pytest tests/test_concentration_stm.py tests/test_heldout_scale_sweep.py tests/test_heldout_scale_sweep_marginalized.py -q`
Expected: PASS (all green — `eta_init=None` default keeps them byte-identical).

- [ ] **Step 6: Commit**

```bash
git add spark-vi/spark_vi/models/topic/stm.py spark-vi/tests/test_stm_doc_inference_warmstart.py
git commit -m "feat(stm): optional eta_init warm-start on _stm_doc_inference (None=cold, byte-identical)"
```

---

### Task 2: `_stm_doc_inference_tprior` — per-document EM over (η, s_d)

The core new primitive: coordinate-ascent to the joint MAP (η̂, ŝ_d) under the
t-prior, reusing the Task-1 warm-startable η-step and the closed-form s_d update.

**Files:**
- Modify: `spark-vi/spark_vi/mllib/topic/stm.py` (add the function; near the
  existing gated sweep helpers, after `_gated_mode_theta` at line ~361)
- Test: `spark-vi/tests/test_tprior_doc_inference.py`

**Interfaces:**
- Consumes: `_stm_doc_inference` (with `eta_init`, Task 1);
  `safe_inverse` from `spark_vi.models.topic._linalg`.
- Produces:
  `_stm_doc_inference_tprior(*, indices, counts, expElogbeta, Gamma, Rinv_allowed, x, c, nu, allowed, reference=None, eta_init=None, sd_init=1.0, lbfgs_max_iter=50, lbfgs_tol=1e-4, sd_max_iter=10, sd_tol=1e-4) -> (eta_hat, sd_hat, nu_d, n_em)`
  where `Rinv_allowed = inv(R[allowed][:, allowed])` (c-independent; caller
  caches), `nu` may be `math.inf`, `eta_hat` is full-K (−inf off allowed),
  `sd_hat` a float, `nu_d` the Laplace covariance from the final η-step,
  `n_em` the EM sweep count.

- [ ] **Step 1: Write the failing tests**

Create `spark-vi/tests/test_tprior_doc_inference.py`:

```python
from __future__ import annotations

import math
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import digamma

from spark_vi.models.topic._linalg import safe_inverse
from spark_vi.mllib.topic.stm import _stm_doc_inference_tprior
from spark_vi.models.topic.stm import _stm_doc_inference


def _setup(K=4, V=12, reference=None):
    beta = np.full((K, V), 1e-3)
    blk = V // K
    for k in range(K):
        beta[k, k * blk:(k + 1) * blk] += 2.0
    beta /= beta.sum(axis=1, keepdims=True)
    lam = beta * (500.0 * V) + 0.01
    expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))
    Gamma = np.zeros((1, K))
    R = np.eye(K)
    allowed = np.arange(K, dtype=np.int64)
    indices = np.array([0, 1, blk, blk + 1, 2 * blk], dtype=np.int64)
    counts = np.array([3.0, 2.0, 5.0, 1.0, 4.0])
    x = np.array([1.0])
    return dict(indices=indices, counts=counts, expElogbeta=expElogbeta,
                Gamma=Gamma, Rinv_allowed=safe_inverse(R), x=x,
                allowed=allowed, reference=reference), R


def test_nu_inf_reproduces_single_gaussian_solve():
    kw, R = _setup()
    c = 4.0
    eta_t, sd_t, _, n_em = _stm_doc_inference_tprior(**kw, c=c, nu=math.inf)
    assert sd_t == 1.0 and n_em == 1
    Sigma_inv_allowed = (1.0 / c) * kw["Rinv_allowed"]
    eta_g, _, _ = _stm_doc_inference(
        indices=kw["indices"], counts=kw["counts"], expElogbeta=kw["expElogbeta"],
        Gamma=kw["Gamma"], Sigma_inv_allowed=Sigma_inv_allowed, x=kw["x"],
        allowed=kw["allowed"], reference=kw["reference"],
    )
    al = kw["allowed"]
    assert np.allclose(eta_t[al], eta_g[al], atol=1e-8)


def test_sd_update_matches_brute_force_at_fixed_eta():
    kw, R = _setup()
    c, nu = 4.0, 5.0
    # One EM sweep so we have a converged eta at sd=1, then check the sd mode.
    eta_t, sd_t, _, _ = _stm_doc_inference_tprior(
        **kw, c=c, nu=nu, sd_max_iter=1)      # single sd update
    al = kw["allowed"]
    mu = kw["Gamma"].T @ kw["x"]               # (K,) here zero-mean
    diff = eta_t[al] - mu[al]
    q_R = float(diff @ kw["Rinv_allowed"] @ diff)
    K_free = len(al)                           # reference=None
    # brute force: maximize the conditional log-posterior of s given eta
    def neg_logpost(s):
        if s <= 0:
            return np.inf
        return (0.5 * K_free * math.log(s) + q_R / (2.0 * s * c)
                + (nu / 2.0 + 1.0) * math.log(s) + (nu / 2.0) / s)
    opt = minimize_scalar(neg_logpost, bounds=(1e-4, 50.0), method="bounded")
    closed = (nu + q_R / c) / (nu + K_free + 2.0)
    assert abs(closed - opt.x) < 1e-3
    assert abs(sd_t - closed) < 1e-6


def test_em_converges_to_fixed_point():
    kw, R = _setup()
    c, nu = 4.0, 5.0
    eta_t, sd_t, _, n_em = _stm_doc_inference_tprior(
        **kw, c=c, nu=nu, sd_max_iter=50, sd_tol=1e-8)
    assert n_em < 50                            # stopped early = converged
    al = kw["allowed"]
    mu = kw["Gamma"].T @ kw["x"]
    diff = eta_t[al] - mu[al]
    q_R = float(diff @ kw["Rinv_allowed"] @ diff)
    K_free = len(al)
    assert abs(sd_t - (nu + q_R / c) / (nu + K_free + 2.0)) < 1e-5


def test_warm_start_invariance():
    kw, R = _setup()
    c, nu = 4.0, 5.0
    cold = _stm_doc_inference_tprior(**kw, c=c, nu=nu)
    warm = _stm_doc_inference_tprior(**kw, c=c, nu=nu, eta_init=cold[0], sd_init=cold[1])
    al = kw["allowed"]
    assert np.allclose(cold[0][al], warm[0][al], atol=1e-5)
    assert abs(cold[1] - warm[1]) < 1e-5
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd spark-vi && python -m pytest tests/test_tprior_doc_inference.py -v`
Expected: FAIL — `cannot import name '_stm_doc_inference_tprior'`.

- [ ] **Step 3: Implement the EM wrapper**

Add to `spark-vi/spark_vi/mllib/topic/stm.py` (after `_gated_mode_theta`):

```python
def _stm_doc_inference_tprior(
    *, indices, counts, expElogbeta, Gamma, Rinv_allowed, x, c, nu,
    allowed, reference=None, eta_init=None, sd_init=1.0,
    lbfgs_max_iter=50, lbfgs_tol=1e-4, sd_max_iter=10, sd_tol=1e-4,
):
    """Per-document joint MAP (eta, s_d) under the multivariate-t prior
    eta_d | s_d ~ N(mu_d, s_d*c*R), s_d ~ Inverse-Gamma(nu/2, nu/2)
    (design doc 2026-07-10-tprior-per-document-scale-design.md).

    Explicit EM / coordinate ascent: the eta-step is the existing gated Laplace
    solve (``_stm_doc_inference``) at prior precision (1/(s_d*c))*Rinv_allowed,
    warm-started across sweeps; the s_d-step is the closed-form Inverse-Gamma
    mode  s_d = (nu + q_R/c)/(nu + K_free + 2)  with
    q_R = (eta-mu)^T Rinv_allowed (eta-mu) over the allowed set (reference pinned
    at 0) and K_free = |allowed| - (1 if reference else 0). nu=inf recovers the
    single Gaussian solve at s_d=1 (nesting). Returns (eta_hat full-K, s_d float,
    nu_d Laplace cov from the final eta-step, n_em sweeps)."""
    allowed = np.asarray(allowed, dtype=np.int64)
    mu_allowed = (Gamma[:, allowed].T @ x)
    K_free = int(allowed.shape[0] - (1 if reference is not None else 0))

    if not math.isfinite(nu):
        Sigma_inv_allowed = (1.0 / c) * Rinv_allowed
        eta_hat, nu_d, _ = _stm_doc_inference(
            indices=indices, counts=counts, expElogbeta=expElogbeta,
            Gamma=Gamma, Sigma_inv_allowed=Sigma_inv_allowed, x=x,
            max_iter=lbfgs_max_iter, tol=lbfgs_tol,
            allowed=allowed, reference=reference, eta_init=eta_init,
        )
        return eta_hat, 1.0, nu_d, 1

    sd = float(sd_init)
    eta_warm = eta_init
    eta_hat = nu_d = None
    n_em = 0
    for n_em in range(1, sd_max_iter + 1):
        Sigma_inv_allowed = (1.0 / (sd * c)) * Rinv_allowed
        eta_hat, nu_d, _ = _stm_doc_inference(
            indices=indices, counts=counts, expElogbeta=expElogbeta,
            Gamma=Gamma, Sigma_inv_allowed=Sigma_inv_allowed, x=x,
            max_iter=lbfgs_max_iter, tol=lbfgs_tol,
            allowed=allowed, reference=reference, eta_init=eta_warm,
        )
        eta_warm = eta_hat
        diff = eta_hat[allowed] - mu_allowed
        q_R = float(diff @ Rinv_allowed @ diff)
        sd_new = (nu + q_R / c) / (nu + K_free + 2.0)
        if abs(sd_new - sd) < sd_tol:
            sd = sd_new
            break
        sd = sd_new
    return eta_hat, sd, nu_d, n_em
```

Ensure `import math` is present at the top of the module (add if missing).

- [ ] **Step 4: Run to verify they pass**

Run: `cd spark-vi && python -m pytest tests/test_tprior_doc_inference.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/mllib/topic/stm.py spark-vi/tests/test_tprior_doc_inference.py
git commit -m "feat(stm): per-doc t-prior EM inference _stm_doc_inference_tprior (nu=inf nests Gaussian)"
```

---

### Task 3: `corpus_tprior_scale_sweep_gated` (numpy) + readouts

The driver-side 2-D (c, ν) sweep with the two falsifiable readouts, mirroring
`corpus_heldout_scale_sweep_gated` (`stm.py:1129`). An internal `_c_sweep_at_nu`
does one 1-D c-sweep at a fixed ν (warm-starting across c per doc); the grid
loops ν over it, the drift readout loops f over it.

**Files:**
- Modify: `spark-vi/spark_vi/mllib/topic/stm.py` (add both functions after
  `corpus_heldout_scale_sweep_gated`, before its `_rdd` sibling)
- Test: `spark-vi/tests/test_tprior_scale_sweep.py`

**Interfaces:**
- Consumes: `_stm_doc_inference_tprior` (Task 2); `heldout_split`,
  `_predictive_loglik` from `spark_vi.eval.topic.concentration_recovery`;
  `_gated_mode_theta`; `_json_safe` from
  `spark_vi.eval.topic.concentration_heterogeneity`; `safe_inverse`.
- Produces:
  `corpus_tprior_scale_sweep_gated(docs, global_params, partition, *, c_grid, nu_grid, holdout_frac=0.3, drift_fracs=(0.2, 0.3, 0.5), reference=None, seed=0, lbfgs_max_iter=50, lbfgs_tol=1e-4, sd_max_iter=10, sd_tol=1e-4) -> dict`.
  Output dict shape (all JSON-safe; ν=∞ serialized as the string `"inf"`):
  ```
  {"grid": [{"c": float, "nu": float|"inf", "ll": float}, ...],
   "argmax": {"c": float, "nu": float|"inf", "ll": float},
   "n_docs": int,
   "drift": {"fracs": [float,...],
             "gaussian": [{"frac": float, "c_star": float}, ...],
             "tprior":   [{"frac": float, "c_star": float}, ...],
             "gaussian_spread": float, "tprior_spread": float},
   "sd_readout": {"n_docs": int,
                  "sd_quantiles":   {"p10":,"p25":,"p50":,"p75":,"p90":},
                  "sd_c_quantiles": {"p10":,"p25":,"p50":,"p75":,"p90":}}}
  ```
  `_c_sweep_at_nu(docs, expElogbeta, beta_prob, Gamma, R, Rinv_cache, partition, *, c_grid, nu, holdout_frac, reference, seed, ...) -> ({c: mean_ll}, argmax_c)`.

- [ ] **Step 1: Write the failing tests**

Create `spark-vi/tests/test_tprior_scale_sweep.py`:

```python
from __future__ import annotations

import json
import math
import numpy as np

from tests._stm_synth import synthetic_gated_corpus, fit_stm


def _build_fitted_corpus(seed=0):
    docs, planted, part = synthetic_gated_corpus(
        groups=("A", "B"), fg_per_group=1, bg_k=2, V=40, D=40, doc_len=25,
        bg_frac=0.5, seed=seed,
    )
    gp = fit_stm(docs, K=part.K, V=40, sigma_init=1.0, n_iter=20,
                 partition=part, seed=seed)
    return docs, part, {"lambda": gp["lambda"], "Gamma": gp["Gamma"],
                        "Sigma": gp["Sigma"]}


def test_nu_inf_column_matches_gaussian_sweep():
    from spark_vi.mllib.topic.stm import (
        corpus_tprior_scale_sweep_gated, corpus_heldout_scale_sweep_gated,
    )
    docs, part, gp = _build_fitted_corpus(seed=3)
    c_grid = [1, 2, 4, 8]
    gauss = corpus_heldout_scale_sweep_gated(
        docs, gp, part, c_grid=c_grid, holdout_frac=0.3, seed=0)
    t = corpus_tprior_scale_sweep_gated(
        docs, gp, part, c_grid=c_grid, nu_grid=[math.inf],
        holdout_frac=0.3, seed=0)
    by_c = {row["c"]: row["ll"] for row in t["grid"] if row["nu"] == "inf"}
    for c in c_grid:
        assert abs(by_c[c] - gauss["lls"][c]) < 1e-6


def test_grid_argmax_and_structure():
    from spark_vi.mllib.topic.stm import corpus_tprior_scale_sweep_gated
    docs, part, gp = _build_fitted_corpus(seed=5)
    t = corpus_tprior_scale_sweep_gated(
        docs, gp, part, c_grid=[2, 4, 8], nu_grid=[2.5, 5, math.inf],
        holdout_frac=0.3, drift_fracs=(0.2, 0.3), seed=0)
    assert len(t["grid"]) == 9
    assert t["argmax"]["c"] in {2, 4, 8}
    assert t["argmax"]["nu"] in {2.5, 5, "inf"}
    assert all(np.isfinite(r["ll"]) for r in t["grid"])
    assert t["n_docs"] > 0


def test_drift_and_sd_readouts_present():
    from spark_vi.mllib.topic.stm import corpus_tprior_scale_sweep_gated
    docs, part, gp = _build_fitted_corpus(seed=7)
    t = corpus_tprior_scale_sweep_gated(
        docs, gp, part, c_grid=[2, 4, 8], nu_grid=[5, math.inf],
        holdout_frac=0.3, drift_fracs=(0.2, 0.3, 0.5), seed=0)
    d = t["drift"]
    assert [p["frac"] for p in d["gaussian"]] == [0.2, 0.3, 0.5]
    assert [p["frac"] for p in d["tprior"]] == [0.2, 0.3, 0.5]
    assert d["gaussian_spread"] >= 0.0 and d["tprior_spread"] >= 0.0
    s = t["sd_readout"]
    assert s["n_docs"] > 0
    for q in ("p10", "p25", "p50", "p75", "p90"):
        assert q in s["sd_quantiles"] and q in s["sd_c_quantiles"]
    # sd_c = sd * c_star: median should be a positive scale
    assert s["sd_c_quantiles"]["p50"] > 0.0


def test_output_is_json_safe():
    from spark_vi.mllib.topic.stm import corpus_tprior_scale_sweep_gated
    docs, part, gp = _build_fitted_corpus(seed=9)
    t = corpus_tprior_scale_sweep_gated(
        docs, gp, part, c_grid=[2, 4], nu_grid=[5, math.inf],
        holdout_frac=0.3, drift_fracs=(0.2, 0.3), seed=0)
    json.dumps(t)   # must not raise
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd spark-vi && python -m pytest tests/test_tprior_scale_sweep.py -v`
Expected: FAIL — `cannot import name 'corpus_tprior_scale_sweep_gated'`.

- [ ] **Step 3: Implement `_c_sweep_at_nu` and the sweep**

Add to `spark-vi/spark_vi/mllib/topic/stm.py` (after
`corpus_heldout_scale_sweep_gated`). First the shared 1-D c-sweep helper:

```python
def _nu_key(nu):
    return "inf" if not math.isfinite(nu) else float(nu)


def _c_sweep_at_nu(
    docs, *, expElogbeta, beta_prob, Gamma, R, Rinv_cache, partition, c_grid, nu,
    holdout_frac, reference, seed, K, lbfgs_max_iter, lbfgs_tol,
    sd_max_iter, sd_tol,
):
    """One 1-D held-out c-sweep at a fixed nu. Warm-starts eta across c within
    each doc (ordered c_grid). Returns ({c: mean_per_token_ll}, argmax_c)."""
    from spark_vi.eval.topic.concentration_recovery import (
        _predictive_loglik, heldout_split,
    )
    from spark_vi.models.topic._linalg import safe_inverse

    sum_ll = {c: 0.0 for c in c_grid}
    n_tok = {c: 0 for c in c_grid}
    for i, doc in enumerate(docs):
        split = heldout_split(doc, holdout_frac=holdout_frac, seed=seed + i)
        if split is None:
            continue
        visible_doc, held_indices, held_counts = split
        if held_counts.size == 0:
            continue
        allowed = partition.allowed_indices(doc.groups)
        key = tuple(allowed.tolist())
        Rinv_allowed = Rinv_cache.get(key)
        if Rinv_allowed is None:
            Rinv_allowed = safe_inverse(R[np.ix_(allowed, allowed)])
            Rinv_cache[key] = Rinv_allowed
        n_held = int(held_counts.sum())
        eta_warm = None
        for c in c_grid:                      # ordered → warm-start across c
            eta_hat, _sd, _nu_d, _n = _stm_doc_inference_tprior(
                indices=visible_doc.indices, counts=visible_doc.counts,
                expElogbeta=expElogbeta, Gamma=Gamma, Rinv_allowed=Rinv_allowed,
                x=doc.x, c=c, nu=nu, allowed=allowed, reference=reference,
                eta_init=eta_warm, lbfgs_max_iter=lbfgs_max_iter,
                lbfgs_tol=lbfgs_tol, sd_max_iter=sd_max_iter, sd_tol=sd_tol,
            )
            eta_warm = eta_hat
            theta_hat = _gated_mode_theta(eta_hat, allowed, K)
            sum_ll[c] += _predictive_loglik(theta_hat, beta_prob, held_indices, held_counts)
            n_tok[c] += n_held
    lls = {c: (sum_ll[c] / n_tok[c] if n_tok[c] else float("-inf")) for c in c_grid}
    argmax_c = max(lls, key=lls.get)
    return lls, argmax_c
```

Then the public sweep:

```python
def corpus_tprior_scale_sweep_gated(
    docs, global_params, partition, *, c_grid, nu_grid,
    holdout_frac=0.3, drift_fracs=(0.2, 0.3, 0.5), reference=None, seed=0,
    lbfgs_max_iter=50, lbfgs_tol=1e-4, sd_max_iter=10, sd_tol=1e-4,
) -> dict:
    """Driver-side 2-D held-out (c, nu) sweep for the multivariate-t per-document
    scale diagnostic (design doc 2026-07-10-tprior-per-document-scale-design.md).

    Mirrors ``corpus_heldout_scale_sweep_gated`` (same heldout_split, same
    inference-vs-scoring role split, same short-doc skips). Emits the (c, nu)
    grid + argmax, the f-drift readout (c* across drift_fracs at nu=inf vs
    nu=nu*), and the s_d readout (sd and sd*c* quantiles at (c*, nu*), inferred
    on full docs). Both readouts emit numbers only, no verdicts. nu=inf column
    reproduces the Gaussian sweep (nesting). See the RDD sibling for cluster use.
    """
    from scipy.special import digamma
    from spark_vi.eval.topic.concentration_heterogeneity import _json_safe

    lam = np.asarray(global_params["lambda"], dtype=np.float64)
    Gamma = np.asarray(global_params["Gamma"], dtype=np.float64)
    Sigma = np.asarray(global_params["Sigma"], dtype=np.float64)
    K = lam.shape[0]
    lam_rowsum = lam.sum(axis=1, keepdims=True)
    expElogbeta = np.exp(digamma(lam) - digamma(lam_rowsum))
    beta_prob = lam / lam_rowsum
    d = np.diag(Sigma)
    R = Sigma / np.sqrt(np.outer(d, d))
    Rinv_cache: dict[tuple, np.ndarray] = {}

    common = dict(
        expElogbeta=expElogbeta, beta_prob=beta_prob, Gamma=Gamma, R=R,
        Rinv_cache=Rinv_cache, partition=partition, c_grid=list(c_grid),
        holdout_frac=holdout_frac, reference=reference, seed=seed, K=K,
        lbfgs_max_iter=lbfgs_max_iter, lbfgs_tol=lbfgs_tol,
        sd_max_iter=sd_max_iter, sd_tol=sd_tol,
    )

    # --- 2-D grid at holdout_frac ---
    grid = []
    lls_by_nu = {}
    for nu in nu_grid:
        lls, _ = _c_sweep_at_nu(docs, nu=nu, **common)
        lls_by_nu[_nu_key(nu)] = lls
        for c in c_grid:
            grid.append({"c": float(c), "nu": _nu_key(nu), "ll": float(lls[c])})
    best = max(grid, key=lambda r: r["ll"])
    c_star = best["c"]
    nu_star = math.inf if best["nu"] == "inf" else float(best["nu"])

    # --- drift readout: c*(f) at nu=inf vs nu=nu* ---
    def _c_star_at(nu, frac):
        d2 = dict(common); d2["holdout_frac"] = frac
        _, argmax_c = _c_sweep_at_nu(docs, nu=nu, **d2)
        return float(argmax_c)

    gaussian = [{"frac": float(f), "c_star": _c_star_at(math.inf, f)} for f in drift_fracs]
    tprior = [{"frac": float(f), "c_star": _c_star_at(nu_star, f)} for f in drift_fracs]
    def _spread(rows):
        cs = [r["c_star"] for r in rows]
        return float(max(cs) - min(cs))

    # --- s_d readout at (c*, nu*), full docs (no split) ---
    from spark_vi.models.topic._linalg import safe_inverse
    sd_vals = []
    for doc in docs:
        allowed = partition.allowed_indices(doc.groups)
        key = tuple(allowed.tolist())
        Rinv_allowed = Rinv_cache.get(key)
        if Rinv_allowed is None:
            Rinv_allowed = safe_inverse(R[np.ix_(allowed, allowed)])
            Rinv_cache[key] = Rinv_allowed
        _eta, sd, _nu_d, _n = _stm_doc_inference_tprior(
            indices=doc.indices, counts=doc.counts, expElogbeta=expElogbeta,
            Gamma=Gamma, Rinv_allowed=Rinv_allowed, x=doc.x, c=c_star, nu=nu_star,
            allowed=allowed, reference=reference,
            lbfgs_max_iter=lbfgs_max_iter, lbfgs_tol=lbfgs_tol,
            sd_max_iter=sd_max_iter, sd_tol=sd_tol,
        )
        sd_vals.append(sd)
    sd_arr = np.asarray(sd_vals, dtype=np.float64)
    def _q(a):
        ps = np.quantile(a, [0.10, 0.25, 0.50, 0.75, 0.90])
        return {"p10": float(ps[0]), "p25": float(ps[1]), "p50": float(ps[2]),
                "p75": float(ps[3]), "p90": float(ps[4])}

    n_docs = _count_contributing(docs, partition, holdout_frac, seed)

    out = {
        "grid": grid,
        "argmax": {"c": c_star, "nu": _nu_key(nu_star), "ll": best["ll"]},
        "n_docs": n_docs,
        "drift": {"fracs": [float(f) for f in drift_fracs],
                  "gaussian": gaussian, "tprior": tprior,
                  "gaussian_spread": _spread(gaussian),
                  "tprior_spread": _spread(tprior)},
        "sd_readout": {"n_docs": int(sd_arr.size),
                       "sd_quantiles": _q(sd_arr),
                       "sd_c_quantiles": _q(sd_arr * c_star)},
    }
    return _json_safe(out)
```

Add the small contributing-doc counter (mirrors the skip guards) near the helper:

```python
def _count_contributing(docs, partition, holdout_frac, seed):
    from spark_vi.eval.topic.concentration_recovery import heldout_split
    n = 0
    for i, doc in enumerate(docs):
        split = heldout_split(doc, holdout_frac=holdout_frac, seed=seed + i)
        if split is None:
            continue
        _v, _hi, hc = split
        if hc.size == 0:
            continue
        n += 1
    return n
```

- [ ] **Step 4: Run to verify they pass**

Run: `cd spark-vi && python -m pytest tests/test_tprior_scale_sweep.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/mllib/topic/stm.py spark-vi/tests/test_tprior_scale_sweep.py
git commit -m "feat(stm): corpus_tprior_scale_sweep_gated 2-D (c,nu) sweep + drift/sd readouts"
```

---

### Task 4: `corpus_tprior_scale_sweep_gated_rdd` (distributed)

The distributed sweep for a live cluster corpus, mirroring
`corpus_heldout_scale_sweep_gated_rdd` (`stm.py:1262`). It reuses the numpy
per-partition logic by running `_c_sweep_at_nu`-style accumulation inside
`mapPartitions` and `treeReduce`-ing, then does the drift/sd readouts as separate
distributed passes. Must be doc-for-doc identical to the numpy sweep (parity
test).

**Files:**
- Modify: `spark-vi/spark_vi/mllib/topic/stm.py` (add after the numpy sweep)
- Test: add to `spark-vi/tests/test_tprior_scale_sweep.py`

**Interfaces:**
- Consumes: the numpy building blocks from Task 3; the RDD idiom
  (`zipWithIndex().mapPartitions(_local).treeReduce(_combine, depth=depth)`).
- Produces:
  `corpus_tprior_scale_sweep_gated_rdd(doc_rdd, global_params, partition, *, c_grid, nu_grid, holdout_frac=0.3, drift_fracs=(0.2, 0.3, 0.5), reference=None, seed=0, lbfgs_max_iter=50, lbfgs_tol=1e-4, sd_max_iter=10, sd_tol=1e-4, depth=2) -> dict`
  — same output dict shape as the numpy sweep.

- [ ] **Step 1: Write the failing parity test**

Add to `spark-vi/tests/test_tprior_scale_sweep.py`:

```python
class TestRddParity:
    def test_rdd_matches_numpy(self, spark):
        import math
        from spark_vi.mllib.topic.stm import (
            corpus_tprior_scale_sweep_gated,
            corpus_tprior_scale_sweep_gated_rdd,
        )
        docs, part, gp = _build_fitted_corpus(seed=3)
        c_grid = [2, 4, 8]
        nu_grid = [5, math.inf]
        expected = corpus_tprior_scale_sweep_gated(
            docs, gp, part, c_grid=c_grid, nu_grid=nu_grid,
            holdout_frac=0.3, drift_fracs=(0.2, 0.3), seed=0)
        rdd = spark.sparkContext.parallelize(docs, numSlices=3)
        got = corpus_tprior_scale_sweep_gated_rdd(
            rdd, gp, part, c_grid=c_grid, nu_grid=nu_grid,
            holdout_frac=0.3, drift_fracs=(0.2, 0.3), seed=0)
        # grid LLs match doc-for-doc (same splits, same warm-start order per doc)
        exp_by = {(r["c"], r["nu"]): r["ll"] for r in expected["grid"]}
        got_by = {(r["c"], r["nu"]): r["ll"] for r in got["grid"]}
        assert set(exp_by) == set(got_by)
        for k in exp_by:
            assert abs(exp_by[k] - got_by[k]) < 1e-6
        assert got["argmax"] == expected["argmax"]
        assert got["n_docs"] == expected["n_docs"]
        assert abs(got["sd_readout"]["sd_c_quantiles"]["p50"]
                   - expected["sd_readout"]["sd_c_quantiles"]["p50"]) < 1e-6

    def test_rdd_output_json_safe(self, spark):
        import json, math
        from spark_vi.mllib.topic.stm import corpus_tprior_scale_sweep_gated_rdd
        docs, part, gp = _build_fitted_corpus(seed=9)
        rdd = spark.sparkContext.parallelize(docs, numSlices=2)
        got = corpus_tprior_scale_sweep_gated_rdd(
            rdd, gp, part, c_grid=[2, 4], nu_grid=[5, math.inf],
            holdout_frac=0.3, drift_fracs=(0.2, 0.3), seed=0)
        json.dumps(got)
```

Note the parity requirement: warm-start is per-doc (across c within a doc), so it
does NOT cross partition boundaries — numpy and RDD both warm-start within each
doc independently, giving identical LLs regardless of `numSlices`.

- [ ] **Step 2: Run to verify they fail**

Run: `cd spark-vi && python -m pytest tests/test_tprior_scale_sweep.py::TestRddParity -v`
Expected: FAIL — `cannot import name 'corpus_tprior_scale_sweep_gated_rdd'`.

- [ ] **Step 3: Implement the RDD sweep**

Add to `spark-vi/spark_vi/mllib/topic/stm.py`. Structure: a picklable
per-partition accumulator that, for each doc, computes the per-(c, nu) held-out
LL contribution AND the (c*, nu*)-conditional s_d — but (c*, nu*) is unknown
until the grid argmax. So run in two rounds, each its own distributed pass:

Round 1 — grid + drift (both are `_c_sweep_at_nu`-style c-sweeps at various
(nu, frac)); Round 2 — s_d readout at the resolved (c*, nu*).

```python
def corpus_tprior_scale_sweep_gated_rdd(
    doc_rdd, global_params, partition, *, c_grid, nu_grid,
    holdout_frac=0.3, drift_fracs=(0.2, 0.3, 0.5), reference=None, seed=0,
    lbfgs_max_iter=50, lbfgs_tol=1e-4, sd_max_iter=10, sd_tol=1e-4, depth=2,
) -> dict:
    """Distributed multivariate-t (c, nu) sweep — the cluster stand-in for
    ``corpus_tprior_scale_sweep_gated``. Same output dict; doc-for-doc identical
    LLs (warm-start is per-doc, so it never crosses partition boundaries)."""
    from scipy.special import digamma
    from spark_vi.eval.topic.concentration_heterogeneity import _json_safe

    lam = np.asarray(global_params["lambda"], dtype=np.float64)
    Gamma = np.asarray(global_params["Gamma"], dtype=np.float64)
    Sigma = np.asarray(global_params["Sigma"], dtype=np.float64)
    K = lam.shape[0]
    lam_rowsum = lam.sum(axis=1, keepdims=True)
    expElogbeta = np.exp(digamma(lam) - digamma(lam_rowsum))
    beta_prob = lam / lam_rowsum
    d = np.diag(Sigma)
    R = Sigma / np.sqrt(np.outer(d, d))

    sc = doc_rdd.context
    gp_b = sc.broadcast((expElogbeta, beta_prob, Gamma, R))
    c_list = list(c_grid)

    # Each (nu, frac) c-sweep as one distributed pass over indexed docs.
    def _pass(nu, frac):
        _nu, _frac = nu, frac
        def _local(rows, _gp=gp_b, _cl=c_list, _p=partition, _K=K,
                   _ref=reference, _seed=seed, _nu=_nu, _frac=_frac,
                   _li=lbfgs_max_iter, _lt=lbfgs_tol, _si=sd_max_iter, _st=sd_tol):
            import numpy as _np
            from spark_vi.eval.topic.concentration_recovery import (
                _predictive_loglik, heldout_split,
            )
            from spark_vi.models.topic._linalg import safe_inverse
            from spark_vi.mllib.topic.stm import (
                _stm_doc_inference_tprior, _gated_mode_theta,
            )
            eE, bp, Gm, Rr = _gp.value
            cache = {}
            sll = {c: 0.0 for c in _cl}
            ntk = {c: 0 for c in _cl}
            ndoc = 0
            for i, doc in rows:
                split = heldout_split(doc, holdout_frac=_frac, seed=_seed + i)
                if split is None:
                    continue
                vis, hi, hc = split
                if hc.size == 0:
                    continue
                allowed = _p.allowed_indices(doc.groups)
                key = tuple(allowed.tolist())
                Rinv = cache.get(key)
                if Rinv is None:
                    Rinv = safe_inverse(Rr[_np.ix_(allowed, allowed)])
                    cache[key] = Rinv
                ndoc += 1
                nh = int(hc.sum())
                warm = None
                for c in _cl:
                    eta_hat, _sd, _nud, _n = _stm_doc_inference_tprior(
                        indices=vis.indices, counts=vis.counts, expElogbeta=eE,
                        Gamma=Gm, Rinv_allowed=Rinv, x=doc.x, c=c, nu=_nu,
                        allowed=allowed, reference=_ref, eta_init=warm,
                        lbfgs_max_iter=_li, lbfgs_tol=_lt,
                        sd_max_iter=_si, sd_tol=_st,
                    )
                    warm = eta_hat
                    th = _gated_mode_theta(eta_hat, allowed, _K)
                    sll[c] += _predictive_loglik(th, bp, hi, hc)
                    ntk[c] += nh
            return [((sll, ntk), ndoc)]

        def _combine(a, b):
            (sa, ta), na = a
            (sb, tb), nb = b
            return ({c: sa[c] + sb[c] for c in sa},
                    {c: ta[c] + tb[c] for c in ta}), na + nb

        (sll, ntk), ndoc = (
            doc_rdd.zipWithIndex()
            .map(lambda t: (t[1], t[0]))       # (index, doc)
            .mapPartitions(_local).treeReduce(_combine, depth=depth)
        )
        lls = {c: (sll[c] / ntk[c] if ntk[c] else float("-inf")) for c in c_list}
        return lls, max(lls, key=lls.get), ndoc

    # Grid at holdout_frac
    grid = []
    n_docs = 0
    for nu in nu_grid:
        lls, _amax, ndoc = _pass(nu, holdout_frac)
        n_docs = ndoc
        for c in c_list:
            grid.append({"c": float(c), "nu": _nu_key(nu), "ll": float(lls[c])})
    best = max(grid, key=lambda r: r["ll"])
    c_star = best["c"]
    nu_star = math.inf if best["nu"] == "inf" else float(best["nu"])

    # Drift
    gaussian, tprior = [], []
    for f in drift_fracs:
        _l, ag, _n = _pass(math.inf, f); gaussian.append({"frac": float(f), "c_star": float(ag)})
        _l, at, _n = _pass(nu_star, f);  tprior.append({"frac": float(f), "c_star": float(at)})
    def _spread(rows):
        cs = [r["c_star"] for r in rows]; return float(max(cs) - min(cs))

    # s_d readout at (c*, nu*): one map → collect the sd scalars (n is small)
    def _sd_local(doc, _gp=gp_b, _p=partition, _cs=c_star, _ns=nu_star,
                  _ref=reference, _li=lbfgs_max_iter, _lt=lbfgs_tol,
                  _si=sd_max_iter, _st=sd_tol):
        import numpy as _np
        from spark_vi.models.topic._linalg import safe_inverse
        from spark_vi.mllib.topic.stm import _stm_doc_inference_tprior
        eE, bp, Gm, Rr = _gp.value
        allowed = _p.allowed_indices(doc.groups)
        Rinv = safe_inverse(Rr[_np.ix_(allowed, allowed)])
        _eta, sd, _nud, _n = _stm_doc_inference_tprior(
            indices=doc.indices, counts=doc.counts, expElogbeta=eE, Gamma=Gm,
            Rinv_allowed=Rinv, x=doc.x, c=_cs, nu=_ns, allowed=allowed,
            reference=_ref, lbfgs_max_iter=_li, lbfgs_tol=_lt,
            sd_max_iter=_si, sd_tol=_st,
        )
        return float(sd)
    sd_arr = np.asarray(doc_rdd.map(_sd_local).collect(), dtype=np.float64)
    def _q(a):
        ps = np.quantile(a, [0.10, 0.25, 0.50, 0.75, 0.90])
        return {"p10": float(ps[0]), "p25": float(ps[1]), "p50": float(ps[2]),
                "p75": float(ps[3]), "p90": float(ps[4])}

    out = {
        "grid": grid,
        "argmax": {"c": c_star, "nu": _nu_key(nu_star), "ll": best["ll"]},
        "n_docs": int(n_docs),
        "drift": {"fracs": [float(f) for f in drift_fracs],
                  "gaussian": gaussian, "tprior": tprior,
                  "gaussian_spread": _spread(gaussian),
                  "tprior_spread": _spread(tprior)},
        "sd_readout": {"n_docs": int(sd_arr.size),
                       "sd_quantiles": _q(sd_arr),
                       "sd_c_quantiles": _q(sd_arr * c_star)},
    }
    return _json_safe(out)
```

- [ ] **Step 4: Run to verify parity passes**

Run: `cd spark-vi && python -m pytest tests/test_tprior_scale_sweep.py -v`
Expected: PASS (all — numpy + RDD parity + JSON-safe).

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/mllib/topic/stm.py spark-vi/tests/test_tprior_scale_sweep.py
git commit -m "feat(stm): distributed corpus_tprior_scale_sweep_gated_rdd (doc-for-doc numpy parity)"
```

---

### Task 5: Driver flag `BUILD_T_PRIOR_SCALE` + zip + experiment doc

Wire the diagnostic into the dashboard build behind an off-by-default flag,
mirroring the `BUILD_CONCENTRATION_HETEROGENEITY_DIAGNOSTIC` block
(`build_dashboard_cloud.py:969-1028`) exactly — same preconditions, same
self-contained doc_rdd, same never-fatal try/except, same inference-scale
convention — plus the zip-list entry (the omission that broke two prior runs).

**Files:**
- Modify: `analysis/cloud/build_dashboard_cloud.py` (add the block after the
  heterogeneity block ~line 1028; add `"t_prior_scale.json"` to the zip loop at
  ~line 1519-1521)
- Create: `docs/experiments/0048-stm-population-cancer-tprior-scale.md`
- Test: `analysis/cloud/tests/test_tprior_driver.py`

**Interfaces:**
- Consumes: `corpus_tprior_scale_sweep_gated_rdd` (Task 4);
  `_vector_to_stm_document`; the driver's `eta_scale`, `is_stm`, `tbs`,
  `stm_partition`, `stm_cov_df`, `result`, `bow_df`, `out_dir`, `_phase`, `log`.
- Produces: `t_prior_scale.json` in the bundle; env knobs
  `BUILD_T_PRIOR_SCALE`, `BUILD_T_PRIOR_SCALE_DOC_FRAC` (default `0.05`),
  `BUILD_T_PRIOR_SCALE_NU_GRID` (default `2.5,5,10,20,inf`),
  `BUILD_T_PRIOR_SCALE_C_GRID` (default `2,3,4,6,8,12`).

- [ ] **Step 1: Write the failing test**

Create `analysis/cloud/tests/test_tprior_driver.py`. Follow the structure of the
existing driver tests (`analysis/cloud/tests/test_stm_driver_partition.py` for
imports/fixtures). Test the two pure, unit-testable pieces: (a) the grid parser,
(b) the zip loop includes the file when present.

```python
from __future__ import annotations

import zipfile
from pathlib import Path


def test_nu_grid_parser_handles_inf():
    from analysis.cloud.build_dashboard_cloud import _parse_scale_grid
    assert _parse_scale_grid("2.5,5,10,20,inf") == [2.5, 5.0, 10.0, 20.0, float("inf")]
    assert _parse_scale_grid("2,4,8") == [2.0, 4.0, 8.0]


def test_zip_includes_t_prior_scale_when_present(tmp_path: Path):
    from analysis.cloud.build_dashboard_cloud import _zip_optional_files
    out_dir = tmp_path
    (out_dir / "topics.json").write_text("{}")
    (out_dir / "t_prior_scale.json").write_text("{}")
    zip_path = tmp_path / "bundle.zip"
    _zip_optional_files(out_dir, zip_path, required=("topics.json",))
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
    assert "t_prior_scale.json" in names
    assert "topics.json" in names
```

Note: if `_parse_scale_grid` / `_zip_optional_files` do not already exist as
named helpers, this task extracts them from the inline driver code (small,
testable seams) as part of Step 3 — the inline zip loop and any inline grid
parsing move into these two functions and the block calls them. If the reviewer
prefers not to refactor the zip loop, keep the extraction minimal: a
`_zip_optional_files(out_dir, zip_path, required)` that wraps the existing
`for f in (...)` logic with `"t_prior_scale.json"` added to the optional tuple.

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest analysis/cloud/tests/test_tprior_driver.py -v`
Expected: FAIL — import errors for the two helpers.

- [ ] **Step 3: Implement the driver block, helpers, and zip entry**

(a) Add the grid parser near the top-level helpers in
`build_dashboard_cloud.py`:

```python
def _parse_scale_grid(spec: str) -> list[float]:
    """Parse a comma-separated scale grid; 'inf' -> math.inf (for the nu grid)."""
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        out.append(float("inf") if tok.lower() == "inf" else float(tok))
    return out
```

(b) Extract the zip optional-files loop into a helper and add the new file. The
existing loop at ~1519-1521 becomes:

```python
def _zip_optional_files(out_dir, zip_path, required):
    import zipfile
    with zipfile.ZipFile(zip_path, "a", zipfile.ZIP_DEFLATED) as zf:
        existing = set(zf.namelist())
        for f in ("gating.json", "covariate_schema.json",
                  "covariate_effects.json", "correlation.json",
                  "concentration_heterogeneity.json", "t_prior_scale.json"):
            p = out_dir / f
            if p.exists() and f not in existing:
                zf.write(p, arcname=f)
                print(f"[driver]   zip: +{f}", flush=True)
```

Wire the existing zip site to call it (preserving the required-files write that
precedes it), OR — minimal variant — just add `"t_prior_scale.json"` to the
existing inline tuple at line 1521 and have the test call a thin
`_zip_optional_files` that the site also uses. Either way `"t_prior_scale.json"`
MUST be in the optional tuple.

(c) Add the diagnostic block after the heterogeneity block (~line 1028),
mirroring it precisely:

```python
        if os.environ.get("BUILD_T_PRIOR_SCALE") and (
                is_stm and tbs and stm_partition is not None
                and stm_cov_df is not None):
            try:
                from spark_vi.mllib.topic.stm import (
                    corpus_tprior_scale_sweep_gated_rdd,
                )
                from spark_vi.mllib.topic._common import _vector_to_stm_document

                _tp_frac = float(os.environ.get("BUILD_T_PRIOR_SCALE_DOC_FRAC", "0.05"))
                _tp_nu = _parse_scale_grid(
                    os.environ.get("BUILD_T_PRIOR_SCALE_NU_GRID", "2.5,5,10,20,inf"))
                _tp_c = _parse_scale_grid(
                    os.environ.get("BUILD_T_PRIOR_SCALE_C_GRID", "2,3,4,6,8,12"))
                _stm_hardening = result.metadata.get("stm_hardening", {}) or {}
                _tp_ref = 0 if _stm_hardening.get("reference_topic") else None

                _tp_doc_df = bow_df.select("person_id", "features").join(
                    stm_cov_df.select("person_id", "source_cohort", "covariates"),
                    on="person_id", how="inner",
                )
                _tp_doc_rdd = _tp_doc_df.rdd.map(
                    lambda row: _vector_to_stm_document(
                        row, features_col="features",
                        covariates_col="covariates", group_col="source_cohort",
                    )
                )
                # 5% sample: same convention as the heterogeneity diagnostic.
                _tp_sample = _tp_doc_rdd.sample(False, _tp_frac, seed=0)
                log.info("STM: t-prior scale sweep c_grid=%s nu_grid=%s frac=%s",
                         _tp_c, _tp_nu, _tp_frac)
                with _phase("t-prior scale sweep"):
                    _tp = corpus_tprior_scale_sweep_gated_rdd(
                        _tp_sample, result.global_params, stm_partition,
                        c_grid=_tp_c, nu_grid=_tp_nu, reference=_tp_ref, seed=0,
                    )
                import json as _json
                (out_dir / "t_prior_scale.json").write_text(_json.dumps(_tp, indent=2))
                log.info(
                    "t-prior scale (n=%d): argmax c*=%s nu*=%s | drift gauss=%.3f "
                    "tprior=%.3f | sd*c* p50=%.3f",
                    _tp["n_docs"], _tp["argmax"]["c"], _tp["argmax"]["nu"],
                    _tp["drift"]["gaussian_spread"], _tp["drift"]["tprior_spread"],
                    _tp["sd_readout"]["sd_c_quantiles"]["p50"])
            except Exception as _tpexc:   # diagnostic-only: never fatal
                log.warning("t-prior scale sweep failed (%s); bundle UNAFFECTED.", _tpexc)
```

- [ ] **Step 4: Run to verify the test passes**

Run: `python -m pytest analysis/cloud/tests/test_tprior_driver.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Write the experiment doc**

Create `docs/experiments/0048-stm-population-cancer-tprior-scale.md`, cloning the
exp 0047 fit config (same cohort, same gated STM, same
`docs/experiments/0047-stm-population-cancer-scale-diagnostic.md` fit block) with
the flag on. Frontmatter `status: planned`. Body sections: **Goal** (calibrate
(c, ν) on the frozen population_cancer fit; test the two falsifiable
predictions); **Config** (identical fit to 0047; adds
`BUILD_T_PRIOR_SCALE=1`, default grids, `BUILD_T_PRIOR_SCALE_DOC_FRAC=0.05`);
**Command** (`make build-dashboard-exp ID=48` or the exp-0047 build invocation
with the flag exported); **Falsifiable predictions** (verbatim from the design
doc: at (c*, ν*) the tprior drift spread < gaussian drift spread; the sd·c*
distribution's spread is consistent with 6.95/5.60/5.41); **Result** (leave a
`TODO: fill after cluster run` placeholder marked as the one intentional pending
section — this doc is the run's landing pad, filled on completion).

- [ ] **Step 6: Commit**

```bash
git add analysis/cloud/build_dashboard_cloud.py analysis/cloud/tests/test_tprior_driver.py docs/experiments/0048-stm-population-cancer-tprior-scale.md
git commit -m "feat(export): BUILD_T_PRIOR_SCALE flagged diagnostic + zip entry + exp 0048"
```

---

## Post-implementation (controller, after all tasks green)

- Push `origin/stm` and verify HEAD (auto-push can lag; two prior cluster runs
  broke on stale code). The user runs the cluster build; the diagnostic lands in
  `t_prior_scale.json`, then an insight interprets the two readouts.
- The experiment doc's Result section is the intentional post-run TODO — not a
  plan placeholder.

## Self-Review notes

- **Spec coverage:** model+EM (Task 2), 2-D sweep+argmax (Task 3), f-drift
  readout (Task 3/4), ŝ_d readout (Task 3/4), warm-start requirement (Tasks 1–4),
  f-out-of-grid (drift uses its own `_c_sweep_at_nu` passes, not the 2-D grid),
  ν=∞ reuse/nesting (tested Task 3), coarse ν grid + c centered on c* (driver
  defaults, Task 5), `_json_safe` (Tasks 3/4), zip entry (Task 5), no-verdict
  (readouts emit numbers only), no-magic-number (grids are inputs; argmax is
  emitted, never hardcoded). All covered.
- **Type consistency:** `_stm_doc_inference_tprior` returns
  `(eta_hat, sd, nu_d, n_em)` everywhere; sweep output dict shape identical in
  numpy and RDD; `_nu_key` serializes ν=∞ as `"inf"` consistently in grid,
  argmax, and tests.
- **Deliberate deviation from spec test #3:** the spec says "monotone
  non-decreasing objective"; the plan tests EM convergence via fixed-point
  satisfaction + early stop (`test_em_converges_to_fixed_point`), which verifies
  the same convergence guarantee without reconstructing the objective's dropped
  constants. Equivalent coverage, cleaner test.
