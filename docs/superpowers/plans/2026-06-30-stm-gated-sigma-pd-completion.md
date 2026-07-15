# STM gated-Σ PD completion — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the gated-Σ `min_pair_support` zero-pin + `nearest_spd` flooring with
a maximum-determinant positive-definite completion (covariance selection), so the gated
full-covariance Σ is well-conditioned by construction; and remove the now-unused
inverse-Wishart prior and `sigma_diag_shrink`.

**Architecture:** A new pure `pd_complete(target, observed_mask)` in `_linalg.py`
(driver-side, K×K) runs a two-stage covariance selection — max-det IPS primary, Higham
alternating-projection fallback for non-PD-completable observed parts. `update_global`
forms observed entries as `S/N` over the `N ≥ min_pair_support` mask and calls it. The
corpus-sized scatter/support accumulation is unchanged.

**Tech stack:** Python, NumPy (driver-side dense linear algebra), pytest.

**Design spec:** [2026-06-30-stm-gated-sigma-pd-completion-design.md](../specs/2026-06-30-stm-gated-sigma-pd-completion-design.md)

## Global Constraints

- spark-vi stays domain-agnostic: integer token ids only, no OMOP/EHR vocabulary in the
  library.
- NO LaTeX in docs/docstrings — Unicode Greek (Σ η μ ν λ) + plain text.
- Cite literature for any method/default/constant (Dempster 1972; Grone et al. 1984;
  Speed & Kiiveri 1986; Lauritzen 1996; Higham 2002).
- Markdown-linkable code references in docs (`[name](path#Lstart-Lend)`).
- The completion is driver-side on the K×K Σ; do not touch the distributed
  scatter/support accumulation.
- `pd_complete` is pure NumPy, no Spark, no domain types.
- Full removal (not deprecation) of `sigma_prior_scale`, `sigma_prior_count`,
  `sigma_diag_shrink` — parameters, plumbing, flags, tests. Completed experiment docs
  0018/0022/0023/0024 keep their frontmatter (historical); the now-unknown keys are
  simply not emitted by `build_stm_args`.
- `nearest_spd` is NOT removed — it remains the per-document Laplace-Hessian repair and
  is reused as the completion's PSD projection.

---

### Task 1: `pd_complete` — maximum-determinant PD completion (covariance selection)

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/_linalg.py` (add `pd_complete`)
- Test: `spark-vi/tests/test_linalg_pd_complete.py` (create)

**Interfaces:**
- Consumes: `nearest_spd` (existing, same file) as the PSD projection.
- Produces: `pd_complete(target: np.ndarray, observed_mask: np.ndarray, *, tol: float = 1e-10, max_iter: int = 1000) -> np.ndarray`.

**Algorithm.** Iterative proportional scaling for covariance selection (Dempster 1972;
Speed & Kiiveri 1986), with a Higham (2002) PSD projection providing robustness when the
observed part is not PD-completable. Single loop, conceptually two-stage (the PSD
projection is a no-op in the feasible case, so feasible inputs get the exact max-det
completion):

1. Init `Sigma` = `target` on observed entries, 0 on free off-diagonal entries; force the
   diagonal to `target`'s diagonal. Project to PD once via `nearest_spd`.
2. Repeat until `max |Sigma - Sigma_prev| < tol` or `max_iter`:
   - `Prec = inv(Sigma)`; zero `Prec` on the free (¬observed) entries; `Sigma = inv(Prec)`
     (this drives the free precision entries to zero — conditional independence).
   - Reset `Sigma` on observed entries to `target` (preserve the measured values exactly).
   - If `Sigma` is not PD (a Cholesky raises), project via `nearest_spd` (the Higham
     fallback for non-PD-completable observed parts).
3. Symmetrize and return.

On a decomposable (chordal) observed pattern this converges in one sweep to the
closed-form completion `Sigma[A,B] = Sigma[A,S] · inv(Sigma[S,S]) · Sigma[S,B]` (Grone et
al. 1984; Lauritzen 1996) — used as a test oracle, not a separate code path.

- [ ] **Step 1: Write the failing tests** (`spark-vi/tests/test_linalg_pd_complete.py`)

```python
import numpy as np
import pytest
from spark_vi.models.topic._linalg import pd_complete


def _is_pd(M):
    try:
        np.linalg.cholesky(M)
        return True
    except np.linalg.LinAlgError:
        return False


def test_identity_when_all_observed():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 5))
    Sigma = A @ A.T + np.eye(5)
    out = pd_complete(Sigma, np.ones((5, 5), bool))
    np.testing.assert_allclose(out, Sigma, atol=1e-9)


def test_decomposable_closed_form_oracle():
    # topics: 0 = separator (background), 1 and 2 cond. independent given 0.
    target = np.array([[2.0, 1.0, 0.8],
                       [1.0, 1.5, 0.0],   # (1,2) is free
                       [0.8, 0.0, 1.2]])
    mask = np.array([[1, 1, 1],
                     [1, 1, 0],
                     [1, 0, 1]], bool)
    out = pd_complete(target, mask)
    # Grone/Lauritzen closed form for the free cross entry:
    expected_12 = target[1, 0] * (1.0 / target[0, 0]) * target[0, 2]   # = 0.4
    assert abs(out[1, 2] - expected_12) < 1e-8
    assert abs(out[2, 1] - expected_12) < 1e-8
    # observed entries preserved exactly
    for i, j in [(0, 1), (0, 2), (1, 1), (2, 2), (0, 0)]:
        assert abs(out[i, j] - target[i, j]) < 1e-8
    # zero precision on the free entry (conditional independence)
    prec = np.linalg.inv(out)
    assert abs(prec[1, 2]) < 1e-7
    assert _is_pd(out)


def test_general_nonchordal_pd_and_zero_precision_on_free():
    # 4 topics in a 4-cycle of observed edges: 0-1,1-2,2-3,3-0 observed;
    # 0-2 and 1-3 free (non-decomposable pattern -> IPS must iterate).
    rng = np.random.default_rng(1)
    A = rng.standard_normal((4, 4))
    target = A @ A.T + 2 * np.eye(4)
    mask = np.array([[1, 1, 0, 1],
                     [1, 1, 1, 0],
                     [0, 1, 1, 1],
                     [1, 0, 1, 1]], bool)
    out = pd_complete(target, mask)
    assert _is_pd(out)
    prec = np.linalg.inv(out)
    assert abs(prec[0, 2]) < 1e-6 and abs(prec[1, 3]) < 1e-6   # free -> zero precision
    for i in range(4):                                          # observed preserved
        for j in range(4):
            if mask[i, j]:
                assert abs(out[i, j] - target[i, j]) < 1e-6


def test_non_pd_completable_observed_returns_pd_without_raising():
    # Observed entries that admit no PD completion: a near-rank-deficient block.
    target = np.array([[1.0, 0.99, 0.99],
                       [0.99, 1.0, -0.99],   # inconsistent with the above two
                       [0.99, -0.99, 1.0]])
    mask = np.ones((3, 3), bool)             # all "observed" but not PD
    out = pd_complete(target, mask)          # must not raise
    assert _is_pd(out)
    np.testing.assert_allclose(out, out.T, atol=1e-10)


def test_output_symmetric():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((6, 6))
    target = A @ A.T + np.eye(6)
    mask = np.ones((6, 6), bool)
    mask[0, 4] = mask[4, 0] = False
    out = pd_complete(target, mask)
    np.testing.assert_allclose(out, out.T, atol=1e-10)
```

- [ ] **Step 2: Run the tests, verify they fail** — `pytest spark-vi/tests/test_linalg_pd_complete.py -v` → FAIL (`pd_complete` undefined).

- [ ] **Step 3: Implement `pd_complete`** in `_linalg.py` per the algorithm above. Cite
  Dempster 1972 / Grone et al. 1984 / Speed & Kiiveri 1986 / Lauritzen 1996 / Higham 2002
  in the docstring. NO LaTeX. Keep diagonal always observed (the caller guarantees a true
  diagonal mask). Symmetrize the mask defensively (`mask | mask.T`).

- [ ] **Step 4: Run the tests, verify they pass** — `pytest spark-vi/tests/test_linalg_pd_complete.py -v` → PASS. Tune `tol`/`max_iter` only as needed for the convergence tests; do not loosen the oracle tolerances.

- [ ] **Step 5: Commit** — `feat(spark-vi): pd_complete max-det PD completion (covariance selection)`.

---

### Task 2: Integrate `pd_complete` into the Σ M-step; remove IW prior + diag-shrink from `OnlineSTM`

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/stm.py` (`OnlineSTM.__init__` params; `update_global` Σ M-step)
- Test: `spark-vi/tests/test_stm_contract.py`, `spark-vi/tests/test_stm_sigma_completion.py` (create)

**Interfaces:**
- Consumes: `pd_complete` (Task 1).
- Produces: an `update_global` whose Σ result is the PD completion of the `S/N` observed
  entries over the `N ≥ min_pair_support` mask. `OnlineSTM.__init__` no longer accepts
  `sigma_prior_scale`, `sigma_prior_count`, `sigma_diag_shrink`.

**M-step rewrite** ([stm.py:711-740](../../../spark-vi/spark_vi/models/topic/stm.py#L711-L740)),
replacing the `min_pair_support` zero-pin, the IW-MAP `Psi` blend, the `sigma_diag_shrink`
block, and the trailing `nearest_spd`:

```python
S = target_stats["residual_outer_stat"]
N = target_stats["n_pairs_stat"]
observed = N >= self.min_pair_support          # diagonal has full support -> always True
np.fill_diagonal(observed, True)
with np.errstate(invalid="ignore", divide="ignore"):
    mle = np.where(observed, S / np.where(observed, N, 1.0), 0.0)
Sigma_target = np.where(observed, mle, Sigma)   # free entries: carry prior, will be completed
Sigma_blended = (1.0 - learning_rate) * Sigma + learning_rate * Sigma_target
new_Sigma = pd_complete(Sigma_blended, observed)
```

(The ρ-blend is applied to the observed entries; `pd_complete` then fixes those and
completes the free entries. The free entries' pre-completion values are irrelevant — the
completion overwrites them — so the `np.where(observed, mle, Sigma)` free branch is just a
finite placeholder.)

- [ ] **Step 1: Write the failing tests** (`spark-vi/tests/test_stm_sigma_completion.py`)

```python
import numpy as np
from spark_vi.models.topic.stm import OnlineSTM


def _is_pd(M):
    try:
        np.linalg.cholesky(M)
        return True
    except np.linalg.LinAlgError:
        return False


def test_removed_params_rejected():
    import pytest
    for bad in ("sigma_prior_scale", "sigma_prior_count", "sigma_diag_shrink"):
        with pytest.raises(TypeError):
            OnlineSTM(K=5, V=10, **{bad: 1.0})


def test_completed_sigma_is_pd_and_preserves_observed():
    # Build target_stats so a cross-pair (3,4) is unsupported (N=0) -> free.
    K = 5
    stm = OnlineSTM(K=K, V=8, min_pair_support=2)
    rng = np.random.default_rng(0)
    eta = rng.standard_normal((40, K))
    S = eta.T @ eta
    N = np.full((K, K), 40.0)
    N[3, 4] = N[4, 3] = 0.0                       # thin cross-pair -> free
    gp = stm.initialize_global({"vocab_size": 8})
    stats = {k: np.zeros_like(v) if hasattr(v, "shape") else v
             for k, v in stm.update_global.__defaults__ or {}}  # placeholder; see note
    # (Implementer: assemble a minimal valid target_stats dict matching update_global's
    # contract — lambda_stats, XtX*, residual_outer_stat=S, n_pairs_stat=N, etc.)
    out = stm.update_global(gp, _make_target_stats(S, N), learning_rate=1.0)
    Sigma = out["Sigma"]
    assert _is_pd(Sigma)
    assert abs(np.linalg.inv(Sigma)[3, 4]) < 1e-5    # free pair -> zero precision
```

Note for the implementer: `_make_target_stats(S, N)` is a small local helper that builds
the full `target_stats` dict `update_global` expects (see the existing
`test_stm_contract.py` fixtures for the required keys: `lambda_stats`, `XtX`, `XtX_groups`,
`XtMu`, `residual_outer_stat`, `n_pairs_stat`, `n_docs_per_topic`, `doc_loglik_sum`,
`doc_eta_kl_sum`, `n_docs`). Reuse that pattern rather than the `__defaults__` placeholder.

- [ ] **Step 2: Run, verify fail** — the params are still accepted and there is no
  completion; tests fail.

- [ ] **Step 3: Implement** — remove the three params from `OnlineSTM.__init__` (and their
  validation + attributes); rewrite the Σ M-step as above; update the `update_global`
  docstring's Σ section (drop the IW MAP and diag-shrink lines, describe the completion;
  cite the spec). Keep `min_pair_support` (now the observed/free threshold).

- [ ] **Step 4: Update `test_stm_contract.py`** — any case constructing `OnlineSTM` with
  the removed params, or asserting the old zero-pin/IW/diag-shrink Σ formulas, is updated:
  drop the removed kwargs; replace zero-pin assertions with the completion contract (PD,
  observed preserved, free → zero precision). Run `pytest spark-vi/tests/test_stm_contract.py
  spark-vi/tests/test_stm_sigma_completion.py -v` → PASS.

- [ ] **Step 5: Run the STM engine suite** — `pytest spark-vi/tests/test_stm_math.py
  spark-vi/tests/test_stm_reference.py spark-vi/tests/test_stm_integration.py -q` → PASS
  (fix any fallout from the removed params).

- [ ] **Step 6: Commit** — `feat(spark-vi): gated Σ M-step uses pd_complete; remove IW prior + diag-shrink from OnlineSTM`.

---

### Task 3: Remove IW prior + `sigma_diag_shrink` from the shim, drivers, and run_experiment

**Files:**
- Modify: `spark-vi/spark_vi/mllib/topic/stm.py` (StreamingSTM params + `stm_hardening` metadata)
- Modify: `analysis/cloud/stm_bigquery_cloud.py`, `analysis/local/fit_stm_local.py` (CLI flags)
- Modify: `scripts/run_experiment.py` (`build_stm_args` emission)
- Test: `spark-vi/tests/test_mllib_stm.py`, `analysis/cloud/tests/test_stm_driver_partition.py`, `scripts/tests/test_run_experiment.py`

**Interfaces:**
- Consumes: the Task-2 `OnlineSTM` signature (no IW/diag-shrink params).
- Produces: `StreamingSTM` and the drivers/run_experiment with no `--sigma-prior-scale`,
  `--sigma-prior-count`, `--sigma-diag-shrink` flags or params; `build_stm_args` emits
  none of them.

- [ ] **Step 1: Write/adjust the failing tests** — in `test_mllib_stm.py`, assert
  `StreamingSTM(... )` rejects the three removed kwargs (`pytest.raises(TypeError)`), and
  that `stm_hardening` metadata no longer carries them; in `test_run_experiment.py`, assert
  `build_stm_args` on an experiment whose frontmatter still has `sigma_prior_count` /
  `sigma_diag_shrink` does NOT emit `--sigma-prior-*` / `--sigma-diag-shrink` (unknown keys
  ignored); in the driver test, assert the argparse parser has no such options.

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement the removal** — delete the params from `StreamingSTM.__init__`,
  its forwarding to `OnlineSTM`, and the `stm_hardening` metadata dict; delete the three
  `add_argument` blocks in both drivers and their forwarding into `StreamingSTM`; delete
  the three emission blocks in `build_stm_args`
  ([run_experiment.py:494-508](../../../scripts/run_experiment.py#L494-L508)). Leave
  `min_pair_support` everywhere it appears.

- [ ] **Step 4: Run the affected suites** — `pytest spark-vi/tests/test_mllib_stm.py
  spark-vi/tests/test_mllib_stm_persistence.py scripts/tests/test_run_experiment.py
  analysis/cloud/tests/test_stm_driver_partition.py -q` → PASS.

- [ ] **Step 5: Commit** — `refactor: remove IW prior + sigma_diag_shrink plumbing (shim, drivers, run_experiment)`.

---

### Task 4: exp 0025 + docs (insight 0032 resolution, ADR 0033 amendment)

**Files:**
- Create: `docs/experiments/0025-stm-comorbid-fullcov-gated-pdcompletion.md`
- Modify: `docs/insights/0032-gated-fullcov-recovers-dementia-subphenotypes-and-exposes-spd-assembly-conditioning.md`
- Modify: `docs/decisions/0033-stm-full-covariance-sigma.md`

**Interfaces:** none (docs).

- [ ] **Step 1: Write exp 0025** — frontmatter = exp 0021 (K=50, background_k=30,
  `foreground: "cancer:10,dementia:10"`, `group_var: source_cohort`, `min_pair_support: 10`,
  `reference_topic: true`, `spectral_init: true`, `spectral_method: dense`, `sigma_init: 1.0`,
  `max_iter: 100`) with NO `sigma_prior_*` / `sigma_diag_shrink` keys. Body: the completion
  replaces the zero-pin; hypothesis = cond O(1e1-1e3), `sigma_eig_min` off 1e-6 AND
  `sigma_eig_max` O(1-10) (both ends), no runaway, ELBO ≈ -1.59e6, dementia sub-phenotypes
  preserved, cancer↔dementia sub-block of `correlation.npy` at CI-implied values. Decision
  tree: success → completion validated, make it the gated default; if a topic still runs
  away → the runaway is not the cross-block (re-examine via `stm_sigma_diagnostic`).

- [ ] **Step 2: Update insight 0032** — add the resolution: the four-lever failure is
  resolved by reframing the assembly as covariance selection (zero precision, not
  covariance, on unobserved cross-pairs); `pd_complete` (max-det IPS + Higham fallback);
  exp 0025 tests it. Mark Status accordingly (conditioning RESOLVED-PENDING-0025). Keep NO
  LaTeX; markdown-linkable refs.

- [ ] **Step 3: Amend ADR 0033** — note that decisions 5 (three-layer SPD mitigation) and 7
  (min_pair_support zero-pin) are superseded for the gated cross-group case by the
  PD-completion design (link the spec); decision 6's two regularizers (IW prior,
  `sigma_diag_shrink`) are removed. Add Dempster 1972 / Grone et al. 1984 / Speed-Kiiveri
  1986 / Higham 2002 to references.

- [ ] **Step 4: Verify** — no LaTeX (`grep -nE '\$|\\\(' docs/...`); cross-refs resolve.

- [ ] **Step 5: Commit** — `docs(stm): exp 0025 + insight 0032 resolution + ADR 0033 amendment (PD completion)`.

---

## Validation (post-implementation)

- Full engine + persistence suite green: `pytest spark-vi/tests -q` (excluding slow).
- Analysis suite green: `pytest analysis/cloud/tests scripts/tests -q`.
- Cluster: `make exp ID=25` — success = cond O(1e1-1e3), both eigenvalue ends controlled,
  no variance runaway (`stm_sigma_diagnostic`), ELBO ≈ -1.59e6, dementia sub-phenotypes
  (Alzheimer's/amnestic + vascular) preserved, trustworthy cancer↔dementia R sub-block.

## Self-review notes

- Spec coverage: Task 1 = `pd_complete`; Task 2 = M-step integration + OnlineSTM removal;
  Task 3 = shim/driver/run_experiment removal; Task 4 = exp 0025 + docs. All spec "In
  scope" items covered.
- Type consistency: `pd_complete(target, observed_mask, *, tol, max_iter)` signature is
  identical in Tasks 1 and 2.
- The `_make_target_stats` helper in Task 2's test is the one placeholder; the implementer
  builds it from the documented `update_global` stats contract (keys listed) — not a logic
  placeholder, a fixture-assembly instruction.
