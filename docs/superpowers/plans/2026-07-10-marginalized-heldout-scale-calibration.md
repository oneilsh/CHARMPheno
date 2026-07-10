# Marginalized held-out scale calibration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the MAP-plug-in held-out predictive likelihood that calibrates the exported generative `eta_scale` (c) with the marginalized (Laplace-sample) posterior-predictive, so c stops drifting with the held-out fraction; validate as a decomposition (synthetic confirms the MAP artifact, real corpus reads residual drift as misspecification).

**Architecture:** Frozen-β, export-time, single-pooled-scalar calibration is unchanged (ADR 0034/0036). Only the scoring *functional* inside the held-out sweep changes: instead of scoring held-out tokens under one MAP θ̂, average the per-token predictive over S Laplace draws (log-of-average) using the per-doc Laplace covariance ν_d the E-step already returns. Add a synthetic decomposition harness; flip the export sweep to the marginalized scorer.

**Tech Stack:** Python (numpy/scipy), PySpark (RDD sweep), pytest.

## Global Constraints

- No LaTeX anywhere (prose, docstrings, UI); Unicode Greek only (η, θ, Σ, R, β, ν, c).
- Cite literature for any method/default/constant: Wallach, Murray, Salakhutdinov & Mimno 2009 (ICML) for document-completion / plug-in bias; Blei & Lafferty 2007 for logistic-normal + Laplace; Hill 1973 / Jost 2006 for concentration metrics.
- TDD: write the failing test, watch it fail, minimal code to pass, commit.
- **The estimator is log-of-average, per held token:** `Σ_w n_w · log[(1/S) Σ_s (θ(η_s)·β)_w]`. NOT average-of-log. Guard with a dedicated test.
- Numpy/Spark parity: per-doc split seed AND per-doc sample seed are `seed + doc_index`, independent of c, so the c-sweep is a controlled comparison and the RDD path reproduces the numpy oracle doc-for-doc.
- Inference vs scoring split preserved: inference uses `expElogbeta` (exp-digamma of λ); held-out scoring uses `beta_prob` = E[β] = λ/λ.sum(axis=1). Do not conflate.
- The existing plug-in sweep functions (`stm_heldout_ll`, `sweep_heldout`, `corpus_heldout_scale_sweep_gated{,_rdd}` plug-in path) stay intact — they are the decomposition baseline.
- Hash IDs in any row-level log output; aggregates/probabilities may be raw.

## File Structure

- `spark-vi/spark_vi/eval/topic/concentration_recovery.py` — add the Laplace θ-sampler helper, the marginalized scorer, and `stm_marginalized_heldout_ll` / `sweep_heldout_marginalized` (numpy, frozen-β, synthetic). Existing plug-in functions untouched.
- `spark-vi/tests/test_concentration_recovery_marginalized.py` — NEW test module for the above (do not disturb the existing `test_concentration_recovery*.py`).
- `spark-vi/spark_vi/mllib/topic/stm.py` — add `marginalize`/`n_samples` routing to `corpus_heldout_scale_sweep_gated` (:868) and `..._rdd` (:972), capturing ν_d instead of discarding it.
- `spark-vi/tests/test_heldout_scale_sweep_marginalized.py` — NEW; numpy path + numpy/RDD parity.
- `scripts/marginalized_scale_decomposition_experiment.py` — NEW; synthetic decomposition (MAP vs marginalized × holdout fraction) + residual measurement.
- `analysis/local/build_dashboard.py`, `analysis/cloud/build_dashboard_cloud.py` — flip export sweep to `marginalize=True`; record method + per-holdout c\* in `eta_scale_diagnostic`.
- `docs/experiments/0046-marginalized-scale-decomposition/` — synthetic + real-corpus results.
- `docs/insights/00NN-*.md` — the decomposition write-up (number assigned at write time).

---

### Task 1: Laplace θ-sampler + marginalized per-token scorer (frozen-β)

**Files:**
- Modify: `spark-vi/spark_vi/eval/topic/concentration_recovery.py`
- Test: `spark-vi/tests/test_concentration_recovery_marginalized.py`

**Interfaces:**
- Consumes: `_softmax` (stm), `_predictive_loglik` (this module, as the S=1-at-mode reference in tests).
- Produces:
  - `laplace_theta_samples(eta_hat, nu_d, allowed, K, *, reference, n_samples, rng) -> np.ndarray` shape `(n_samples, K)`.
  - `marginalized_predictive_loglik(theta_samples, beta_prob, held_indices, held_counts) -> float`.

- [ ] **Step 1: Write the failing tests**

```python
# spark-vi/tests/test_concentration_recovery_marginalized.py
import numpy as np
from spark_vi.eval.topic.concentration_recovery import (
    laplace_theta_samples, marginalized_predictive_loglik, _predictive_loglik,
)

def test_samples_shape_and_simplex_and_masking():
    K = 5
    allowed = np.array([0, 2, 4])          # topics 1,3 disallowed
    reference = 0                           # reference pinned at eta=0
    eta_hat = np.full(K, -np.inf); eta_hat[allowed] = [0.0, 0.3, -0.2]
    nu_d = np.zeros((K, K))
    free = np.array([2, 4])                 # allowed minus reference
    nu_d[np.ix_(free, free)] = [[0.4, 0.05], [0.05, 0.3]]
    S = 64
    th = laplace_theta_samples(eta_hat, nu_d, allowed, K,
                               reference=reference, n_samples=S,
                               rng=np.random.default_rng(0))
    assert th.shape == (S, K)
    assert np.allclose(th.sum(axis=1), 1.0)         # simplex
    assert np.allclose(th[:, [1, 3]], 0.0)          # disallowed -> exactly 0
    assert (th[:, reference] > 0).all()             # reference alive

def test_zero_covariance_samples_reduce_to_mode_theta():
    # nu_d = 0 -> every draw is the mode -> marginalized == plug-in at the mode.
    K = 4; allowed = np.array([0, 1, 2, 3]); reference = 0
    eta_hat = np.array([0.0, 0.5, -0.3, 0.1])
    nu_d = np.zeros((K, K))
    th = laplace_theta_samples(eta_hat, nu_d, allowed, K, reference=reference,
                               n_samples=8, rng=np.random.default_rng(1))
    mode = th[0]
    assert np.allclose(th, mode)                    # all draws identical
    beta = np.abs(np.random.default_rng(2).normal(size=(K, 6)))
    beta /= beta.sum(axis=1, keepdims=True)
    held_i = np.array([0, 3, 5]); held_c = np.array([2.0, 1.0, 3.0])
    marg = marginalized_predictive_loglik(th, beta, held_i, held_c)
    plug = _predictive_loglik(mode, beta, held_i, held_c)
    assert abs(marg - plug) < 1e-9

def test_log_of_average_not_average_of_log():
    # With real spread, log-of-average (marginalized) must EXCEED average-of-log
    # (Jensen) — this is the ordering that IS the fix.
    K = 3; allowed = np.array([0, 1, 2]); reference = 0
    eta_hat = np.array([0.0, 0.2, -0.1])
    nu_d = np.zeros((K, K)); nu_d[np.ix_([1, 2], [1, 2])] = [[1.5, 0.0], [0.0, 1.5]]
    th = laplace_theta_samples(eta_hat, nu_d, allowed, K, reference=reference,
                               n_samples=256, rng=np.random.default_rng(3))
    beta = np.abs(np.random.default_rng(4).normal(size=(K, 8)))
    beta /= beta.sum(axis=1, keepdims=True)
    held_i = np.array([1, 4, 7]); held_c = np.array([1.0, 2.0, 1.0])
    log_of_avg = marginalized_predictive_loglik(th, beta, held_i, held_c)
    # average-of-log baseline
    avg_of_log = np.mean([_predictive_loglik(t, beta, held_i, held_c) for t in th])
    assert log_of_avg > avg_of_log
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd spark-vi && python -m pytest tests/test_concentration_recovery_marginalized.py -x -q`
Expected: FAIL (ImportError: cannot import name `laplace_theta_samples`).

- [ ] **Step 3: Implement**

Append to `spark_vi/eval/topic/concentration_recovery.py`:

```python
def laplace_theta_samples(
    eta_hat, nu_d, allowed, K, *, reference, n_samples, rng,
):
    """Draw ``n_samples`` theta over the K display topics from the per-doc
    Laplace posterior N(eta_hat, nu_d) restricted to the FREE (allowed,
    non-reference) topics.

    ``eta_hat`` (length K) is the MAP mode (finite on allowed, -inf elsewhere,
    reference at 0); ``nu_d`` (K, K) is the Laplace covariance with a nonzero
    sub-block only on the free topics (exactly the pair returned by
    spark_vi.models.topic.stm._stm_doc_inference). Each returned row assembles
    logits = 0 at the reference, drawn value at each free topic, -inf at
    disallowed topics, then softmax -> theta on the simplex with disallowed
    topics exactly 0. Reference: Blei & Lafferty 2007 (logistic-normal + Laplace
    posterior).
    """
    allowed = np.asarray(allowed)
    if reference is not None and reference in set(allowed.tolist()):
        free_topics = np.array([k for k in allowed.tolist() if k != reference],
                               dtype=np.int64)
        ref_alive = True
    else:
        free_topics = allowed.astype(np.int64)
        ref_alive = False

    F = free_topics.shape[0]
    theta = np.zeros((n_samples, K), dtype=np.float64)
    if F == 0:
        # only the reference is allowed -> all mass on it
        if ref_alive:
            theta[:, reference] = 1.0
        return theta

    mean_free = eta_hat[free_topics]
    cov_free = nu_d[np.ix_(free_topics, free_topics)]
    # SPD Cholesky with a tiny jitter fallback (cov may be numerically singular
    # when a free topic is nearly pinned by the data term).
    try:
        L = np.linalg.cholesky(cov_free)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(cov_free + 1e-10 * np.eye(F))
    z = rng.standard_normal((n_samples, F))
    eta_free = mean_free[None, :] + z @ L.T          # (S, F)

    for s in range(n_samples):
        logits = np.full(K, -np.inf)
        if ref_alive:
            logits[reference] = 0.0
        logits[free_topics] = eta_free[s]
        mx = np.max(logits[np.isfinite(logits)])
        ex = np.where(np.isfinite(logits), np.exp(logits - mx), 0.0)
        theta[s] = ex / ex.sum()
    return theta


def marginalized_predictive_loglik(theta_samples, beta_prob, held_indices, held_counts):
    """Held-out log-likelihood under the MARGINALIZED per-token predictive:
    for each held token w, average the predicted probability (theta_s @ beta)_w
    over the S samples, THEN take the log (log-of-average, not average-of-log --
    the ordering that removes the MAP-plug-in bias; see Wallach et al. 2009,
    "Evaluation methods for topic models", ICML). Returns the SUM over held
    tokens (caller normalizes by corpus held-token count). The 1e-300 floor
    guards log(0) for a term with zero predicted mass under every sample.
    """
    preds = theta_samples @ beta_prob                 # (S, V)
    avg = preds[:, held_indices].mean(axis=0)         # (n_held,)
    return float(np.sum(held_counts * np.log(avg + 1e-300)))
```

- [ ] **Step 4: Run to verify pass**

Run: `cd spark-vi && python -m pytest tests/test_concentration_recovery_marginalized.py -x -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/eval/topic/concentration_recovery.py spark-vi/tests/test_concentration_recovery_marginalized.py
git commit -m "feat(eval): Laplace theta-sampler + marginalized (log-of-average) held-out scorer"
```

---

### Task 2: Frozen-β marginalized held-out sweep + ordering guard

**Files:**
- Modify: `spark-vi/spark_vi/eval/topic/concentration_recovery.py`
- Test: `spark-vi/tests/test_concentration_recovery_marginalized.py`

**Interfaces:**
- Consumes: `heldout_split`, `stm_recover_theta` internals (`_stm_doc_inference`), `laplace_theta_samples`, `marginalized_predictive_loglik`.
- Produces:
  - `stm_marginalized_heldout_ll(docs, beta, *, c, n_samples=64, holdout_frac=0.3, seed=0, max_iter=200, tol=1e-6) -> float`
  - `sweep_heldout_marginalized(docs, beta, *, knobs, n_samples=64, holdout_frac=0.3, seed=0) -> dict` (same `{"lls", "argmax_knob"}` shape as `sweep_heldout`).

- [ ] **Step 1: Write the failing tests**

```python
def test_marginalized_sweep_recovers_planted_scale_and_is_flatter_across_holdout():
    from spark_vi.eval.topic.concentration_recovery import (
        make_shared_beta, plant_corpus, sweep_heldout, sweep_heldout_marginalized,
    )
    beta = make_shared_beta(K=8, V=400, seed=0)
    docs, _ = plant_corpus(beta, D=200, doc_len=60, mechanism="logistic_normal",
                           level=3.0, seed=1)
    knobs = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0]
    # MAP plug-in c* moves across holdout; marginalized c* should be steadier.
    map_lo = sweep_heldout(docs, beta, method="stm", knobs=knobs, holdout_frac=0.5)["argmax_knob"]
    map_hi = sweep_heldout(docs, beta, method="stm", knobs=knobs, holdout_frac=0.9)["argmax_knob"]
    mrg_lo = sweep_heldout_marginalized(docs, beta, knobs=knobs, holdout_frac=0.5, n_samples=64)["argmax_knob"]
    mrg_hi = sweep_heldout_marginalized(docs, beta, knobs=knobs, holdout_frac=0.9, n_samples=64)["argmax_knob"]
    # marginalized drift <= MAP drift (grid steps); primary claim is directional.
    assert abs(knobs.index(mrg_lo) - knobs.index(mrg_hi)) <= abs(knobs.index(map_lo) - knobs.index(map_hi))

def test_sweep_marginalized_return_shape():
    from spark_vi.eval.topic.concentration_recovery import (
        make_shared_beta, plant_corpus, sweep_heldout_marginalized,
    )
    beta = make_shared_beta(K=6, V=200, seed=0)
    docs, _ = plant_corpus(beta, D=40, doc_len=40, mechanism="dirichlet", level=0.3, seed=2)
    out = sweep_heldout_marginalized(docs, beta, knobs=[1.0, 3.0], n_samples=16)
    assert set(out) == {"lls", "argmax_knob"} and set(out["lls"]) == {1.0, 3.0}
```

Note: the drift test is directional (marginalized drift ≤ MAP drift on the grid). If it proves flaky at this synthetic size, the implementer should raise D and/or n_samples and REPORT the observed c\* drift numbers in the task report — the true acceptance is the full experiment (Task 3), not this smoke test.

- [ ] **Step 2: Run to verify they fail** — `ImportError: sweep_heldout_marginalized`.

- [ ] **Step 3: Implement** (mirror `stm_heldout_ll` / `sweep_heldout`, swapping the scorer):

```python
def stm_marginalized_heldout_ll(
    docs, beta, *, c, n_samples=64, holdout_frac=0.3, seed=0,
    max_iter=200, tol=1e-6,
):
    """Marginalized (Laplace-sample) counterpart of stm_heldout_ll: infer the
    visible-token MAP eta_hat AND its Laplace covariance nu_d, draw n_samples
    theta from N(eta_hat, nu_d), and score the held-out half by the log-of-
    average per-token predictive (marginalized_predictive_loglik). Non-gated
    frozen-beta path (allowed=None, reference=None), so free_topics = all K and
    the reference branch is inactive. Per-doc sample rng is seeded seed+i,
    independent of c, matching heldout_split's split seed. Mean per held token."""
    K = beta.shape[0]
    Gamma = np.zeros((1, K)); x = np.array([1.0])
    Sigma_inv = (1.0 / c) * np.eye(K)
    total_ll = 0.0; total_tokens = 0
    for i, doc in enumerate(docs):
        split = heldout_split(doc, holdout_frac=holdout_frac, seed=seed + i)
        if split is None:
            continue
        visible_doc, held_i, held_c = split
        if held_c.size == 0:
            continue
        eta_hat, nu_d, _ = _stm_doc_inference(
            indices=visible_doc.indices, counts=visible_doc.counts,
            expElogbeta=beta, Gamma=Gamma, Sigma_inv_allowed=Sigma_inv, x=x,
            max_iter=max_iter, tol=tol, allowed=None, reference=None,
        )
        th = laplace_theta_samples(
            eta_hat, nu_d, np.arange(K), K, reference=None,
            n_samples=n_samples, rng=np.random.default_rng(seed + i),
        )
        total_ll += marginalized_predictive_loglik(th, beta, held_i, held_c)
        total_tokens += int(held_c.sum())
    return total_ll / total_tokens


def sweep_heldout_marginalized(docs, beta, *, knobs, n_samples=64, holdout_frac=0.3, seed=0):
    """Marginalized analog of sweep_heldout (STM only): score each c via
    stm_marginalized_heldout_ll on the SAME per-doc split/sample seeds, return
    {"lls": {c: mean_ll}, "argmax_knob": best_c}."""
    lls = {c: stm_marginalized_heldout_ll(docs, beta, c=c, n_samples=n_samples,
                                          holdout_frac=holdout_frac, seed=seed)
           for c in knobs}
    return {"lls": lls, "argmax_knob": max(lls, key=lls.get)}
```

Import `_stm_doc_inference` at module top (already imports `_softmax`, `_stm_doc_inference` from `spark_vi.models.topic.stm` — verify and extend the existing import).

- [ ] **Step 4: Run to verify pass** — `pytest tests/test_concentration_recovery_marginalized.py -x -q`.
- [ ] **Step 5: Commit** — `feat(eval): frozen-beta marginalized held-out c-sweep (STM)`.

---

### Task 3: Synthetic decomposition experiment (the evidence gate)

**Files:**
- Create: `scripts/marginalized_scale_decomposition_experiment.py`
- Create: `docs/experiments/0046-marginalized-scale-decomposition/` (results written by the script)
- Test: `scripts/tests/test_marginalized_scale_decomposition_experiment.py` (a fast smoke test on a tiny config)

**What it does:** Plant at a KNOWN generative scale over `make_shared_beta`; sweep c under BOTH `sweep_heldout` (MAP) and `sweep_heldout_marginalized` at holdout ∈ {0.5, 0.7, 0.95}, at a clean regime (K=8, V=400, doc_len=60) and the REAL regime (K=60, V=5000, doc_len=44). Emit a JSON + Markdown table of c\*(estimator, holdout) and the planted scale, plus the **residual drift** of the marginalized estimator (max−min c\* across holdout fractions). CLI flags for a fast smoke config.

**Acceptance (recorded in the results md, not asserted in the smoke test):**
1. MAP c\* drifts across holdout fractions; marginalized c\* is flat and centered on the planted scale (within grid tolerance) — this isolates and confirms the MAP artifact and the fix.
2. Report the marginalized residual drift magnitude — this decides Task 7 (importance sampling): if material, do Task 7; if negligible, skip it.

- [ ] **Step 1** Write a fast smoke test that runs the experiment's core function on a tiny config (K=6, V=120, D=40, one holdout pair, n_samples=16) and asserts it returns a dict with `map_cstar`, `marg_cstar` keyed by holdout, and a numeric `marg_residual_drift`.
- [ ] **Step 2** Run it — FAIL (module missing).
- [ ] **Step 3** Implement the experiment module + a thin `__main__` that writes `docs/experiments/0046-marginalized-scale-decomposition/results{,-real-regime}.{json,md}`. Reuse `plant_corpus`, `sweep_heldout`, `sweep_heldout_marginalized`, `corpus_concentration_summary`. Follow the structure of `scripts/concentration_recovery_experiment.py`.
- [ ] **Step 4** Run smoke test — PASS. Then run the full clean-regime config locally and paste the c\*-vs-holdout table into the task report.
- [ ] **Step 5** Commit — `feat(exp): synthetic MAP-vs-marginalized scale decomposition (exp 0046)`.

> **CONTROLLER GATE after Task 3:** read the marginalized residual-drift number. Surface it to the human with a recommendation (material → run Task 7 before the production flip; negligible → skip Task 7, mark it not-needed). Do not proceed past Task 6 to the real-corpus run without this decision.

---

### Task 4: Production numpy sweep — `marginalize` routing

**Files:**
- Modify: `spark-vi/spark_vi/mllib/topic/stm.py` (`corpus_heldout_scale_sweep_gated`, :868)
- Test: `spark-vi/tests/test_heldout_scale_sweep_marginalized.py`

**Interfaces:**
- Produces: `corpus_heldout_scale_sweep_gated(..., marginalize=False, n_samples=64)` — when `marginalize`, capture `nu_d` from `_stm_doc_inference` and score via `laplace_theta_samples` + `marginalized_predictive_loglik` over the doc's `allowed`/`reference`; else the existing `_gated_mode_theta` + `_predictive_loglik` path (byte-identical to today).

- [ ] **Step 1** Failing tests:
  - `test_marginalize_false_is_byte_identical_to_current`: on a small gated fixture, `marginalize=False` returns the exact same `lls` dict as before this change (guards the baseline).
  - `test_marginalize_true_runs_and_returns_grid`: `marginalize=True, n_samples=16` returns `{"lls","argmax_c","n_docs"}` with the same c-grid keys and finite LLs.
  - `test_gated_zero_nu_reduces_to_plugin`: construct a fixture where the data term dominates so `nu_d ≈ 0`; assert marginalized `lls` ≈ plug-in `lls` within a loose tol (sanity that the gated assembly matches the mode).
- [ ] **Step 2** Run — FAIL (unexpected kwarg `marginalize`).
- [ ] **Step 3** Implement: capture `eta_hat, nu_d, _` (drop the `_` on ν_d); branch the per-c scoring. Reuse the module's existing `allowed`/`reference` and `beta_prob`. Seed the sampler `np.random.default_rng(seed + i)` (same discipline as the split). Import `laplace_theta_samples`, `marginalized_predictive_loglik` inside the function (mirrors the existing local imports of `_predictive_loglik`, `heldout_split`).
- [ ] **Step 4** Run — PASS.
- [ ] **Step 5** Commit — `feat(spark-vi): marginalized scoring option in gated held-out scale sweep (numpy)`.

---

### Task 5: Production RDD sweep — `marginalize` routing + numpy/RDD parity

**Files:**
- Modify: `spark-vi/spark_vi/mllib/topic/stm.py` (`corpus_heldout_scale_sweep_gated_rdd`, :972)
- Test: `spark-vi/tests/test_heldout_scale_sweep_marginalized.py` (add RDD parity; use the local-Spark fixture pattern from the existing sweep tests)

**Interfaces:**
- Produces: `corpus_heldout_scale_sweep_gated_rdd(..., marginalize=False, n_samples=64)` — same math as Task 4 on a distributed corpus; broadcast `marginalize`/`n_samples`; per-doc sample seed `seed + zipWithIndex` so RDD reproduces numpy doc-for-doc.

- [ ] **Step 1** Failing test: `test_rdd_marginalized_matches_numpy` — build a small in-memory doc list, run `corpus_heldout_scale_sweep_gated(marginalize=True, n_samples=16)` and `..._rdd(marginalize=True, n_samples=16)` on the same docs/seed; assert per-c `lls` match to ~1e-9.
- [ ] **Step 2** Run — FAIL.
- [ ] **Step 3** Implement in `_local`: capture ν_d, seed `np.random.default_rng(seed + idx)`, branch scoring identically to Task 4. Add `marginalize`/`n_samples` to the broadcast closure.
- [ ] **Step 4** Run — PASS (plus the existing `_rdd` plug-in parity tests still green).
- [ ] **Step 5** Commit — `feat(spark-vi): marginalized scoring in distributed gated scale sweep + numpy/RDD parity`.

---

### Task 6: Driver wiring — flip export to marginalized, record provenance

**Files:**
- Modify: `analysis/local/build_dashboard.py`, `analysis/cloud/build_dashboard_cloud.py`
- Test: the drivers' existing dashboard-build tests (extend, do not fork)

**Interfaces:**
- The export scale block calls `corpus_heldout_scale_sweep_gated_rdd(..., marginalize=True, n_samples=<default>)`, ships the smoothed argmax c\* as `eta_scale`, and records in `eta_scale_diagnostic`: `method="marginalized_laplace_mc"`, `n_samples`, the per-holdout c\* grid (now expected flat), and the per-c held-out LLs at the canonical holdout.

- [ ] **Step 1** Failing test: assert the driver passes `marginalize=True` and that `eta_scale_diagnostic["method"] == "marginalized_laplace_mc"` in the emitted correlation bundle for a small fixture.
- [ ] **Step 2** Run — FAIL.
- [ ] **Step 3** Implement the flip + provenance dict. Keep a `--no-marginalize-scale` escape (BooleanOptionalAction, default True) so the MAP baseline stays runnable for the real-corpus decomposition (Task 8).
- [ ] **Step 4** Run — PASS.
- [ ] **Step 5** Commit — `feat(export): calibrate eta_scale via marginalized held-out sweep; record provenance`.

---

### Task 7 (CONDITIONAL — only if Task 3 residual is material): importance-sampling de-bias

**Files:**
- Modify: `spark-vi/spark_vi/eval/topic/concentration_recovery.py` (+ `stm.py` scorer branch), tests alongside.

**What:** Self-normalized importance sampling to correct the Laplace under-dispersion residual. Proposal q = the Laplace Gaussian N(η̂, ν_d); weight each sample by `w_s = exp(logjoint(η_s) − logq(η_s))` (unnormalized posterior over proposal); score `p(w_held) ≈ Σ_s w_s (θ(η_s)·β)_w / Σ_s w_s` per held token, then log. Add `estimator="laplace_mc"|"laplace_is"` to the marginalized scorer and thread through the sweep.

- [ ] **Step 1** Failing test: on a deliberately over-dispersed planted config, `laplace_is` recovers the planted scale with SMALLER residual drift across holdout than `laplace_mc`.
- [ ] Steps 2–5: TDD + commit — `feat(eval): importance-sampled de-bias for the marginalized held-out sweep`.

> Skip this task entirely (mark not-needed in the ledger) if the Task-3 residual drift is negligible. Decision is the controller's, surfaced to the human.

---

### Task 8: Real-corpus decomposition run + insight (execution/ops)

**Not a TDD code task** — a cluster run + write-up, driven by the controller/user.

- [ ] Re-run the export scale sweep on the population_cancer fit BOTH ways (`--marginalize-scale` and `--no-marginalize-scale`) at holdout ∈ {0.5, 0.7, 0.95}, as a SEPARATE dashboard bundle so the before/after is side-by-side. (`make build-dashboard-exp ID=28` variants, or the documented override path.)
- [ ] Record `docs/experiments/0046-marginalized-scale-decomposition/real-corpus.md`: MAP c\* curve vs marginalized c\* curve. Reading: marginalized should be flatter; **residual drift = a measurement of misspecification** (concentration heterogeneity), not an estimator artifact.
- [ ] Set the shipped `eta_scale` = the marginalized c\* (holdout-independent by construction if the residual is negligible; else the documented canonical-holdout value).
- [ ] Write `docs/insights/00NN-*.md` documenting the decomposition (MAP artifact confirmed on synthetic; real-corpus residual = misspecification signal) and the shipped number. Add the numbered pointer.
- [ ] Update `docs/REVIEW_LOG.md` and the parking-lot memory (item #2 resolved / partially resolved).

---

## Self-Review notes

- **Spec coverage:** estimator (T1), frozen-β sweep + ordering guard (T2), synthetic decomposition evidence (T3), production numpy+RDD (T4/T5), export flip (T6), conditional IS (T7), real-corpus + write-up (T8). All §5 validation steps map to tasks.
- **Type consistency:** `laplace_theta_samples`/`marginalized_predictive_loglik` signatures are identical in T1 (def), T2 (frozen-β caller), T4/T5 (gated callers).
- **The ordering guard** (log-of-average, not average-of-log) is pinned in T1 (`test_log_of_average_not_average_of_log`) and is the load-bearing correctness test.
- **Baseline preserved:** T4 `test_marginalize_false_is_byte_identical_to_current` guards the plug-in path so the decomposition has a valid before.
