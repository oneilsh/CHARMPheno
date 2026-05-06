# Online HDP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 60-line stub at `spark-vi/spark_vi/models/online_hdp.py` with a working Online HDP topic model implementing Wang/Paisley/Blei 2011, plus the matching `CharmPhenoHDP` clinical wrapper, ADR, and tests.

**Architecture:** Pure-Python `VIModel` subclass, hooks into the existing `VIRunner` mini-batch SVI machinery the same way `VanillaLDA` does. Per-doc CAVI runs on Spark workers via `local_update`; M-step runs on the driver via `update_global`. Suff-stats are dense ndarrays aggregated through the framework's default `combine_stats`.

**Tech Stack:** Python 3.12, NumPy, SciPy (digamma/gammaln only), PySpark for the integration tests.

---

## Spec reference

Full design in [`docs/superpowers/specs/2026-05-06-online-hdp-design.md`](../specs/2026-05-06-online-hdp-design.md). Read before starting Task 1.

## File structure

| File | Action | Responsibility |
|---|---|---|
| `spark-vi/spark_vi/models/online_hdp.py` | Replace (60 → ~600 lines) | Inner `OnlineHDP` model + module-private math helpers |
| `spark-vi/tests/test_online_hdp_unit.py` | Create | Pure-numpy unit tests for helpers, init, update_global, ELBO |
| `spark-vi/tests/test_online_hdp_integration.py` | Create | Slow-tier Spark-local ELBO trend, recovery, infer_local |
| `spark-vi/tests/fixtures/online_hdp_wang_reference.json` | Create | Cross-check fixture from Wang's reference Python (one fixed seed) |
| `charmpheno/charmpheno/phenotype/charm_pheno_hdp.py` | Replace (64 lines) | Clinical wrapper — descriptive ctor names, `transform()` |
| `charmpheno/tests/test_charm_pheno_hdp_wrapper.py` | Modify | Update existing tests (remove NotImplementedError gate, add transform tests) |
| `docs/architecture/SPARK_VI_FRAMEWORK.md` | Modify L260-306 | Annotate OnlineHDP sketch with explicit T/K labels |
| `docs/architecture/TOPIC_STATE_MODELING.md` | Modify L289-301 | Add paper-vs-code naming-inversion paragraph |
| `docs/decisions/0011-online-hdp-design.md` | Create | ADR recording v1 scope decisions |

## Sequencing rationale

- **Tasks 1-3** build the three pure math helpers bottom-up. Each is fast to test in isolation.
- **Tasks 4-5** assemble the helpers into `_doc_e_step`, the most complex pure function in the model. We split shape contract (Task 4) from per-iter monotonicity (Task 5) so each test can isolate a different failure mode.
- **Tasks 6-12** fill in the `OnlineHDP` class methods top-down. Each replaces one stub method; after Task 12 the file is no longer a stub.
- **Tasks 13-15** wire everything to Spark in slow-tier integration tests.
- **Task 16** is the optional Wang-reference cross-check (logistically tricky; can be deferred).
- **Tasks 17-19** are the wrapper, doc updates, and ADR — should be in the same final commit train so the repo lands consistent.

## Numerics conventions used throughout

- `digamma`, `gammaln`: from `scipy.special`. Match VanillaLDA's import pattern (`from scipy.special import digamma, gammaln`).
- Dtypes: `np.float64` everywhere unless an upstream object dictates otherwise.
- `T` = corpus truncation (Wang code convention). `K` = doc truncation. Comments throughout the file should reinforce this since the AISTATS paper inverts the letters.

---

## Task 1: `_log_normalize_rows` helper

**Files:**
- Modify: `spark-vi/spark_vi/models/online_hdp.py` (add helper + module imports; keep existing `OnlineHDP` stub class for now)
- Create: `spark-vi/tests/test_online_hdp_unit.py`

- [ ] **Step 1: Write the failing test**

Create `spark-vi/tests/test_online_hdp_unit.py`:

```python
"""Pure-numpy tests for OnlineHDP module-level math helpers and CAVI.

No Spark — these test the math in isolation. Single document, hand-checked
shapes and values where possible.
"""
import numpy as np
import pytest


def test_log_normalize_rows_simplex_invariant():
    """exp(out) rows sum to 1; out and input differ by a constant per row."""
    from spark_vi.models.online_hdp import _log_normalize_rows

    rng = np.random.default_rng(0)
    M = rng.normal(size=(5, 7))
    out = _log_normalize_rows(M)

    assert out.shape == M.shape
    assert np.allclose(np.exp(out).sum(axis=1), 1.0)

    # Each row of (M - out) should be a single repeated constant (the log-norm).
    diff = M - out
    assert np.allclose(diff, diff[:, [0]])


def test_log_normalize_rows_handles_large_magnitudes():
    """Numerical stability under large positive entries (no inf/nan)."""
    from spark_vi.models.online_hdp import _log_normalize_rows

    M = np.array([[1000.0, 1001.0, 999.0],
                  [-1000.0, -1001.0, -999.0]])
    out = _log_normalize_rows(M)

    assert np.all(np.isfinite(out))
    assert np.allclose(np.exp(out).sum(axis=1), 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Command: `cd spark-vi && JAVA_HOME=$JAVA_HOME poetry run pytest tests/test_online_hdp_unit.py::test_log_normalize_rows_simplex_invariant -v`
Expected: `ImportError: cannot import name '_log_normalize_rows'` (or equivalent).

- [ ] **Step 3: Implement the helper**

Open `spark-vi/spark_vi/models/online_hdp.py`. Replace the entire file with:

```python
"""Online HDP topic model — Wang/Paisley/Blei 2011, AISTATS.

Implements the algorithm via the spark_vi `VIModel` contract: per-doc CAVI
on workers (`local_update`), natural-gradient SVI step on the driver
(`update_global`).

Notation. We follow Wang's reference-code convention (also used by intel-spark):
  T = corpus-level truncation (paper's K)
  K = doc-level truncation    (paper's T)
The AISTATS paper inverts these letters; see
docs/architecture/TOPIC_STATE_MODELING.md "Notation" for the rationale.

References:
  - Wang, Paisley, Blei (2011). "Online Variational Inference for the
    Hierarchical Dirichlet Process." AISTATS. Eqs 15-18 give doc-CAVI;
    Eqs 22-27 give the natural-gradient SVI step.
  - Wang's reference Python implementation:
    https://github.com/blei-lab/online-hdp (onlinehdp.py).
  - intel-spark TopicModeling Scala port (cited for confirmation only;
    we explicitly diverge from its driver-side `chunk.collect()` E-step).
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from scipy.special import digamma, gammaln

from spark_vi.core.model import VIModel


# ---------------------------------------------------------------------------
# Module-private math helpers (pure functions, easy to unit-test in isolation).
# ---------------------------------------------------------------------------

def _log_normalize_rows(M: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise log-normalize.

    Returns log(softmax(M, axis=1)). Subtracts the per-row max before
    exponentiating to avoid overflow on large positive entries; for very
    negative entries np.exp underflows to 0, which is benign.
    """
    row_max = M.max(axis=1, keepdims=True)
    shifted = M - row_max
    log_norm = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return shifted - log_norm


# Stub OnlineHDP class — methods filled in by later tasks.
class OnlineHDP(VIModel):
    """Stub during incremental implementation; see Task 6 onwards."""

    def __init__(
        self,
        *,
        vocab_size: int,
        max_topics: int = 150,
        eta: float = 0.01,
        alpha: float = 1.0,
        omega: float = 1.0,
    ) -> None:
        if vocab_size < 1:
            raise ValueError("vocab_size must be >= 1")
        self.vocab_size = int(vocab_size)
        self.max_topics = int(max_topics)
        self.eta = float(eta)
        self.alpha = float(alpha)
        self.omega = float(omega)

    def initialize_global(self, data_summary: Any | None) -> dict[str, np.ndarray]:
        raise NotImplementedError("OnlineHDP is being built; see Task 7.")

    def local_update(
        self,
        rows: Iterable[Any],
        global_params: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError("OnlineHDP is being built; see Task 8.")

    def update_global(
        self,
        global_params: dict[str, np.ndarray],
        target_stats: dict[str, np.ndarray],
        learning_rate: float,
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError("OnlineHDP is being built; see Task 9.")
```

- [ ] **Step 4: Run test to verify it passes**

Command: `cd spark-vi && JAVA_HOME=$JAVA_HOME poetry run pytest tests/test_online_hdp_unit.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/online_hdp.py spark-vi/tests/test_online_hdp_unit.py
git commit -m "$(cat <<'EOF'
feat(hdp): _log_normalize_rows helper + module skeleton

Replaces the 60-line stub with a properly-structured module that will grow
into the full OnlineHDP implementation. The OnlineHDP class is preserved as
a stub (with its existing public signature) so importers don't break while
later tasks land.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `_expect_log_sticks` helper

**Files:**
- Modify: `spark-vi/spark_vi/models/online_hdp.py` (add helper after `_log_normalize_rows`)
- Modify: `spark-vi/tests/test_online_hdp_unit.py` (add tests)

- [ ] **Step 1: Write the failing tests**

Append to `spark-vi/tests/test_online_hdp_unit.py`:

```python
def test_expect_log_sticks_uniform_prior_known_values():
    """For Beta(1, 1) sticks, the math reduces to a closed form we can hand-check.

    With a = b = 1 (uniform Beta), digamma(1) = -gamma_euler ≈ -0.5772 and
    digamma(2) = 1 - gamma_euler ≈ 0.4228. So:
      Elog_W   = digamma(1) - digamma(2) = -1.0
      Elog_1mW = digamma(1) - digamma(2) = -1.0
    For T=3 (a, b each length 2):
      out[0] = Elog_W[0]                                = -1.0
      out[1] = Elog_W[1] + Elog_1mW[0]                  = -2.0
      out[2] =             Elog_1mW[0] + Elog_1mW[1]    = -2.0
    """
    from spark_vi.models.online_hdp import _expect_log_sticks

    a = np.array([1.0, 1.0])
    b = np.array([1.0, 1.0])
    out = _expect_log_sticks(a, b)

    assert out.shape == (3,)
    assert np.allclose(out, [-1.0, -2.0, -2.0])


def test_expect_log_sticks_truncation_handles_last_atom():
    """For T atoms with (T-1) sticks, the trailing entry receives only the
    cumulative E[log(1-W)] sum (q(W_T = 1) = 1 ⇒ E[log W_T] = 0)."""
    from spark_vi.models.online_hdp import _expect_log_sticks

    rng = np.random.default_rng(42)
    T_minus_1 = 5
    a = 1.0 + rng.gamma(2.0, 1.0, T_minus_1)
    b = 1.0 + rng.gamma(2.0, 1.0, T_minus_1)
    out = _expect_log_sticks(a, b)

    dig_sum = digamma_safe(a + b)
    Elog_1mW = digamma_safe(b) - dig_sum
    expected_last = Elog_1mW.sum()

    assert out.shape == (T_minus_1 + 1,)
    assert np.isclose(out[-1], expected_last)


def digamma_safe(x):
    """Test helper to avoid duplicating the import."""
    from scipy.special import digamma
    return digamma(x)
```

- [ ] **Step 2: Run tests to verify they fail**

Command: `cd spark-vi && JAVA_HOME=$JAVA_HOME poetry run pytest tests/test_online_hdp_unit.py::test_expect_log_sticks_uniform_prior_known_values tests/test_online_hdp_unit.py::test_expect_log_sticks_truncation_handles_last_atom -v`
Expected: 2 failed (ImportError: `_expect_log_sticks` not found).

- [ ] **Step 3: Implement the helper**

In `spark-vi/spark_vi/models/online_hdp.py`, add immediately after `_log_normalize_rows`:

```python
def _expect_log_sticks(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sethuraman-style E[log stick] expectation.

    Inputs `a, b` are length (T-1,) for the corpus stick or (K-1,) for a
    doc stick. Returns a length-(len(a)+1) vector — the trailing entry
    handles the truncation: q(stick_last = 1) = 1, so E[log W_last] = 0
    and only the cumulative E[log(1 - W_<last)] contributes.

    For β'_k ~ Beta(a_k, b_k):
      E[log W_k]      = digamma(a_k) - digamma(a_k + b_k)
      E[log(1 - W_k)] = digamma(b_k) - digamma(a_k + b_k)
      E[log β_k]      = E[log W_k] + sum_{l<k} E[log(1 - W_l)]
    """
    dig_sum = digamma(a + b)
    Elog_W = digamma(a) - dig_sum         # length (T-1,)
    Elog_1mW = digamma(b) - dig_sum       # length (T-1,)

    out = np.zeros(len(a) + 1, dtype=np.float64)
    out[:-1] = Elog_W
    out[1:] += np.cumsum(Elog_1mW)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Command: `cd spark-vi && JAVA_HOME=$JAVA_HOME poetry run pytest tests/test_online_hdp_unit.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/online_hdp.py spark-vi/tests/test_online_hdp_unit.py
git commit -m "$(cat <<'EOF'
feat(hdp): _expect_log_sticks Sethuraman E[log stick] helper

Closed-form expectation for stick-breaking variational posteriors used at
both corpus and document levels of the HDP.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `_beta_kl` helper

**Files:**
- Modify: `spark-vi/spark_vi/models/online_hdp.py`
- Modify: `spark-vi/tests/test_online_hdp_unit.py`

- [ ] **Step 1: Write the failing tests**

Append to `spark-vi/tests/test_online_hdp_unit.py`:

```python
def test_beta_kl_zero_when_posterior_matches_prior():
    """KL(Beta(a, b) ‖ Beta(a, b)) = 0."""
    from spark_vi.models.online_hdp import _beta_kl

    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 5.0])
    kl = _beta_kl(a, b, prior_a=1.0, prior_b=np.array([1.0, 2.0, 5.0]))

    # First entry: a=b=1, prior_a=1, prior_b=1 ⇒ KL = 0.
    assert np.isclose(kl[0], 0.0, atol=1e-12)


def test_beta_kl_zero_for_matched_corpus_prior():
    """For corpus sticks the prior is Beta(1, gamma); when (u, v) = (1, gamma)
    KL is zero."""
    from spark_vi.models.online_hdp import _beta_kl

    gamma = 1.5
    T_minus_1 = 4
    u = np.ones(T_minus_1)
    v = np.full(T_minus_1, gamma)
    kl = _beta_kl(u, v, prior_a=1.0, prior_b=gamma)

    assert np.allclose(kl, 0.0, atol=1e-12)


def test_beta_kl_positive_when_posterior_differs():
    """Concentrate the variational posterior away from the prior; KL > 0."""
    from spark_vi.models.online_hdp import _beta_kl

    u = np.array([10.0, 10.0])
    v = np.array([1.0, 1.0])
    kl = _beta_kl(u, v, prior_a=1.0, prior_b=1.0)

    assert np.all(kl > 0)
```

- [ ] **Step 2: Run tests to verify they fail**

Command: `cd spark-vi && JAVA_HOME=$JAVA_HOME poetry run pytest tests/test_online_hdp_unit.py -v -k beta_kl`
Expected: 3 failed.

- [ ] **Step 3: Implement the helper**

In `spark-vi/spark_vi/models/online_hdp.py`, add immediately after `_expect_log_sticks`:

```python
def _beta_kl(
    a: np.ndarray,
    b: np.ndarray,
    *,
    prior_a: float | np.ndarray,
    prior_b: float | np.ndarray,
) -> np.ndarray:
    """KL[Beta(a, b) ‖ Beta(prior_a, prior_b)], elementwise.

    Closed form:
      KL = log B(α0, β0) - log B(α, β)
         + (α - α0) * (ψ(α) - ψ(α + β))
         + (β - β0) * (ψ(β) - ψ(α + β))
    where B is the Beta function (B(x, y) = Γ(x)Γ(y)/Γ(x+y)).

    Returns a length-len(a) vector; broadcast `prior_a` / `prior_b` from a
    scalar if needed (used for corpus prior Beta(1, gamma) where prior_a is
    scalar and prior_b is scalar gamma).
    """
    pa = np.broadcast_to(np.asarray(prior_a, dtype=np.float64), a.shape)
    pb = np.broadcast_to(np.asarray(prior_b, dtype=np.float64), a.shape)

    log_B_prior = gammaln(pa) + gammaln(pb) - gammaln(pa + pb)
    log_B_post = gammaln(a) + gammaln(b) - gammaln(a + b)

    dig_sum = digamma(a + b)
    return (
        log_B_prior - log_B_post
        + (a - pa) * (digamma(a) - dig_sum)
        + (b - pb) * (digamma(b) - dig_sum)
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Command: `cd spark-vi && JAVA_HOME=$JAVA_HOME poetry run pytest tests/test_online_hdp_unit.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/online_hdp.py spark-vi/tests/test_online_hdp_unit.py
git commit -m "$(cat <<'EOF'
feat(hdp): _beta_kl closed-form Beta-KL helper

Used by compute_elbo for the corpus stick KL term and inside _doc_e_step
for the doc stick KL term. Standard closed form derived from the Beta
log-partition + digamma E[log W] expectations.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `_doc_e_step` — shape contract and structure

**Files:**
- Modify: `spark-vi/spark_vi/models/online_hdp.py`
- Modify: `spark-vi/tests/test_online_hdp_unit.py`

This is the most complex pure function in the model. The current task gets
the structure right; Task 5 adds the per-iter ELBO monotonicity gate.

- [ ] **Step 1: Write the shape-contract failing test**

Append to `spark-vi/tests/test_online_hdp_unit.py`:

```python
def _peaked_elogbeta(T: int, V: int, sharpness: float = 5.0) -> np.ndarray:
    """Stylized E[log beta] where each topic peaks on a single word.

    Topic t peaks on word t (mod V). Used to make doc-CAVI tests
    deterministic and visually inspectable. Returns shape (T, V).
    """
    eb = np.full((T, V), -sharpness, dtype=np.float64)
    for t in range(T):
        eb[t, t % V] = 0.0
    return eb


def test_doc_e_step_shape_and_simplex_contract():
    """Run one doc through CAVI; output arrays have right shapes and are valid."""
    from spark_vi.models.online_hdp import _doc_e_step, _expect_log_sticks

    T, K, V = 10, 5, 20
    Elogbeta = _peaked_elogbeta(T, V)
    u = np.ones(T - 1)
    v = np.full(T - 1, 1.0)  # gamma=1.0
    Elog_sticks_corpus = _expect_log_sticks(u, v)

    indices = np.array([0, 1, 2, 3], dtype=np.int32)
    counts = np.array([3.0, 2.0, 1.0, 4.0], dtype=np.float64)
    Elogbeta_doc = Elogbeta[:, indices]

    result = _doc_e_step(
        indices=indices,
        counts=counts,
        Elogbeta_doc=Elogbeta_doc,
        Elog_sticks_corpus=Elog_sticks_corpus,
        alpha=1.0,
        K=K,
        max_iter=20,
        tol=1e-4,
        warmup=3,
    )

    a = result["a"]
    b = result["b"]
    phi = result["phi"]
    var_phi = result["var_phi"]

    assert a.shape == (K - 1,)
    assert b.shape == (K - 1,)
    assert phi.shape == (len(indices), K)
    assert var_phi.shape == (K, T)

    assert np.all(a > 0)
    assert np.all(b > 0)
    assert np.all(np.isfinite(phi))
    assert np.all(np.isfinite(var_phi))

    # Simplex contracts.
    assert np.allclose(phi.sum(axis=1), 1.0)
    assert np.allclose(var_phi.sum(axis=1), 1.0)


def test_doc_e_step_returns_doc_elbo_terms():
    """The returned dict must include the four ELBO contributions used by
    local_update for sufficient-stat aggregation."""
    from spark_vi.models.online_hdp import _doc_e_step, _expect_log_sticks

    T, K, V = 10, 5, 20
    Elogbeta = _peaked_elogbeta(T, V)
    u = np.ones(T - 1)
    v = np.full(T - 1, 1.0)
    Elog_sticks_corpus = _expect_log_sticks(u, v)

    indices = np.array([0, 1, 2], dtype=np.int32)
    counts = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    Elogbeta_doc = Elogbeta[:, indices]

    result = _doc_e_step(
        indices=indices, counts=counts,
        Elogbeta_doc=Elogbeta_doc,
        Elog_sticks_corpus=Elog_sticks_corpus,
        alpha=1.0, K=K, max_iter=20, tol=1e-4, warmup=3,
    )

    for key in ("doc_loglik", "doc_z_term", "doc_c_term", "doc_stick_kl"):
        assert key in result, f"missing {key}"
        assert np.isfinite(result[key]), f"{key} is not finite"
    assert result["doc_stick_kl"] >= 0  # KL divergence is non-negative
```

- [ ] **Step 2: Run tests to verify they fail**

Command: `cd spark-vi && JAVA_HOME=$JAVA_HOME poetry run pytest tests/test_online_hdp_unit.py -v -k doc_e_step`
Expected: 2 failed (ImportError or AttributeError on `_doc_e_step`).

- [ ] **Step 3: Implement `_doc_e_step`**

In `spark-vi/spark_vi/models/online_hdp.py`, add immediately after `_beta_kl`:

```python
def _doc_e_step(
    *,
    indices: np.ndarray,
    counts: np.ndarray,
    Elogbeta_doc: np.ndarray,
    Elog_sticks_corpus: np.ndarray,
    alpha: float,
    K: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    warmup: int = 3,
) -> dict[str, np.ndarray | float]:
    """Per-document coordinate ascent for HDP variational posterior.

    Implements paper Eqs 15-18 (Wang/Paisley/Blei 2011). The inner loop
    rotates through four blocks:
      var_phi   (K, T)   — q(c_jk = t):  doc-atom k → corpus-atom t
      phi       (Wt, K)  — q(z_jn = k):  word n → doc-atom k
      a, b      (K-1,)   — q(π'_jt) = Beta(a, b)  (doc stick params)
    until the per-doc ELBO converges within `tol` (relative).

    Args:
      indices: vocab IDs for the unique words in this document, length Wt.
      counts: integer counts per unique word, length Wt.
      Elogbeta_doc: precomputed E[log φ_t,w] for the words in this doc,
        shape (T, Wt). Caller supplies this from the broadcast Elogbeta.
      Elog_sticks_corpus: E[log β] under the corpus stick variational
        posterior, length T.
      alpha: doc-level stick concentration (paper's α0).
      K: doc-level truncation level.
      max_iter: hard cap on coordinate-ascent iterations.
      tol: relative ELBO convergence threshold.
      warmup: skip the prior-correction terms in var_phi / phi updates for
        this many iterations. Wang's empirical stability trick.

    Returns:
      Dict with keys: a, b, phi, var_phi, log_phi, log_var_phi, plus the
      four ELBO scalars: doc_loglik, doc_z_term, doc_c_term, doc_stick_kl.
    """
    Wt = len(indices)
    T = Elogbeta_doc.shape[0]
    counts_col = counts[:, None]

    # Initialize per-doc state.
    a = np.ones(K - 1, dtype=np.float64)
    b = alpha * np.ones(K - 1, dtype=np.float64)
    phi = np.full((Wt, K), 1.0 / K, dtype=np.float64)
    Elog_sticks_doc = _expect_log_sticks(a, b)  # (K,)

    # log_phi / log_var_phi populated inside the loop; declared here so the
    # final returned dict can reference them after the loop ends.
    log_phi = np.log(phi)
    log_var_phi = np.zeros((K, T), dtype=np.float64)
    var_phi = np.zeros((K, T), dtype=np.float64)

    prev_elbo = -np.inf
    doc_loglik = doc_z_term = doc_c_term = 0.0
    doc_stick_kl = 0.0

    for it in range(max_iter):
        # 1) var_phi update — paper Eq 17. Shape (K, T).
        # E[log p(c_jk | β')] = Elog_sticks_corpus[t]
        # E[log p(w_jn | φ_k)] · counts contributes via phi.T @ (Elogbeta_doc * counts).T
        log_var_phi = phi.T @ (Elogbeta_doc * counts_col.T).T  # (K, T)
        if it >= warmup:
            log_var_phi = log_var_phi + Elog_sticks_corpus[None, :]
        log_var_phi = _log_normalize_rows(log_var_phi)
        var_phi = np.exp(log_var_phi)

        # 2) phi update — paper Eq 18. Shape (Wt, K).
        log_phi = (var_phi @ Elogbeta_doc).T  # (Wt, K)
        if it >= warmup:
            log_phi = log_phi + Elog_sticks_doc[None, :]
        log_phi = _log_normalize_rows(log_phi)
        phi = np.exp(log_phi)

        # 3) a, b update — paper Eqs 15-16.
        phi_w = phi * counts_col  # (Wt, K)
        phi_sum = phi_w.sum(axis=0)  # (K,)
        a = 1.0 + phi_sum[: K - 1]
        # b[t] = α + Σ_{s>t} phi_sum[s]
        b = alpha + np.cumsum(phi_sum[1:][::-1])[::-1]
        Elog_sticks_doc = _expect_log_sticks(a, b)

        # 4) Compute doc ELBO and check convergence.
        # Term naming follows paper Eq 14 decomposition:
        #   doc_c_term = E[log p(c | β')] + H(q(c))    = Σ (Elog_sticks_corpus - log_var_phi) · var_phi
        #   doc_z_term = E[log p(z | π)]  + H(q(z))    = Σ (Elog_sticks_doc - log_phi) · phi
        #   doc_loglik = E[log p(w | z, c, φ)]         = Σ phi.T · (var_phi @ (Elogbeta_doc * counts))
        #   doc_stick_kl = KL[q(π') ‖ p(π')]           — subtracted
        doc_c_term = float(np.sum((Elog_sticks_corpus[None, :] - log_var_phi) * var_phi))
        doc_z_term = float(np.sum((Elog_sticks_doc[None, :] - log_phi) * phi))
        data_part = var_phi @ (Elogbeta_doc * counts_col.T)  # (K, Wt)
        doc_loglik = float(np.sum(phi.T * data_part))
        doc_stick_kl = float(_beta_kl(a, b, prior_a=1.0, prior_b=alpha).sum())

        elbo = doc_loglik + doc_z_term + doc_c_term - doc_stick_kl

        if it > 0 and abs(elbo - prev_elbo) / max(abs(prev_elbo), 1.0) < tol:
            break
        prev_elbo = elbo

    return {
        "a": a, "b": b,
        "phi": phi, "var_phi": var_phi,
        "log_phi": log_phi, "log_var_phi": log_var_phi,
        "doc_loglik": doc_loglik,
        "doc_z_term": doc_z_term,
        "doc_c_term": doc_c_term,
        "doc_stick_kl": doc_stick_kl,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Command: `cd spark-vi && JAVA_HOME=$JAVA_HOME poetry run pytest tests/test_online_hdp_unit.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/online_hdp.py spark-vi/tests/test_online_hdp_unit.py
git commit -m "$(cat <<'EOF'
feat(hdp): _doc_e_step coordinate ascent

Implements Wang/Paisley/Blei 2011 Eqs 15-18: rotating var_phi (K, T),
phi (Wt, K), and (a, b) (K-1,) updates, with the iter<3 warmup trick
preserved from Wang's reference code. Returns the variational posterior
plus the four per-doc ELBO contributions.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `_doc_e_step` ELBO monotonicity gate

**Files:**
- Modify: `spark-vi/tests/test_online_hdp_unit.py` (add per-iter monotonicity test)

The CAVI block updates each maximize the doc ELBO holding other blocks fixed,
so the ELBO must monotonically increase per iteration (modulo numerical noise).
A drop indicates a sign error or block-update bug.

- [ ] **Step 1: Write the failing test**

Append to `spark-vi/tests/test_online_hdp_unit.py`:

```python
def test_doc_e_step_per_iter_elbo_nondecreasing():
    """Coordinate ascent must monotonically increase the doc ELBO.

    Patch _doc_e_step to record the per-iter ELBO trace and assert no drop
    larger than numerical noise. This is the regression gate for any change
    to the var_phi / phi / (a, b) update logic.
    """
    from spark_vi.models import online_hdp as hdp

    T, K, V = 10, 5, 20
    Elogbeta = _peaked_elogbeta(T, V)
    u = np.ones(T - 1)
    v = np.full(T - 1, 1.0)
    Elog_sticks_corpus = hdp._expect_log_sticks(u, v)

    indices = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    counts = np.array([5.0, 3.0, 2.0, 4.0, 1.0], dtype=np.float64)
    Elogbeta_doc = Elogbeta[:, indices]

    # Track per-iter ELBO by re-running the inner update with max_iter=i for
    # increasing i, capturing the ELBO at each completed iter count. This
    # avoids modifying _doc_e_step itself with debug instrumentation.
    elbo_trace = []
    for n_iters in range(1, 25):
        result = hdp._doc_e_step(
            indices=indices, counts=counts,
            Elogbeta_doc=Elogbeta_doc,
            Elog_sticks_corpus=Elog_sticks_corpus,
            alpha=1.0, K=K,
            max_iter=n_iters, tol=0.0,  # tol=0 ⇒ never early-break
            warmup=3,
        )
        elbo = (
            result["doc_loglik"] + result["doc_z_term"]
            + result["doc_c_term"] - result["doc_stick_kl"]
        )
        elbo_trace.append(elbo)

    diffs = np.diff(elbo_trace)
    assert np.all(diffs > -1e-9), (
        f"doc ELBO decreased mid-trace: {elbo_trace}\n"
        f"diffs: {diffs}\n"
        f"This indicates a coordinate-ascent regression."
    )
```

- [ ] **Step 2: Run test to verify it passes**

Note: this test should *pass* on a correct `_doc_e_step` implementation — it's
a regression gate, not a TDD failure-first test. If it fails, the bug is in
the implementation from Task 4.

Command: `cd spark-vi && JAVA_HOME=$JAVA_HOME poetry run pytest tests/test_online_hdp_unit.py::test_doc_e_step_per_iter_elbo_nondecreasing -v`
Expected: PASS.

If it fails, debug Task 4 before proceeding. The most common causes are
(a) sign error in one of the update terms, (b) wrong axis on log_normalize,
(c) the warmup branch missing the prior term in the ELBO calculation.

- [ ] **Step 3: Commit**

```bash
git add spark-vi/tests/test_online_hdp_unit.py
git commit -m "$(cat <<'EOF'
test(hdp): per-iter doc-ELBO monotonicity regression gate

Coordinate ascent maximizes the ELBO at each block; any decrease beyond
numerical noise is a bug. Test re-runs _doc_e_step with increasing
max_iter and walks the resulting ELBO trace.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `OnlineHDP.__init__` with new signature

**Files:**
- Modify: `spark-vi/spark_vi/models/online_hdp.py` (replace stub class header)
- Modify: `spark-vi/tests/test_online_hdp_unit.py` (add ctor validation tests)
- Modify: `charmpheno/charmpheno/phenotype/charm_pheno_hdp.py` (lock-step rename of arg passthrough — see Step 5)

This task changes the public ctor signature. The wrapper at
`charmpheno/charmpheno/phenotype/charm_pheno_hdp.py` constructs `OnlineHDP`
with the old signature; that constructor call must be updated in the same
commit so the import chain stays green.

- [ ] **Step 1: Write the failing tests**

Append to `spark-vi/tests/test_online_hdp_unit.py`:

```python
def test_online_hdp_init_validates_inputs():
    from spark_vi.models.online_hdp import OnlineHDP

    # Valid construction.
    m = OnlineHDP(T=20, K=5, vocab_size=100)
    assert m.T == 20
    assert m.K == 5
    assert m.V == 100
    assert m.alpha == 1.0
    assert m.gamma == 1.0
    assert m.eta == 0.01

    # T must be at least 2 (we need T-1 sticks at corpus level).
    with pytest.raises(ValueError, match="T"):
        OnlineHDP(T=1, K=5, vocab_size=100)

    # K must be at least 2 (we need K-1 sticks at doc level).
    with pytest.raises(ValueError, match="K"):
        OnlineHDP(T=20, K=1, vocab_size=100)

    # vocab_size must be >= 1.
    with pytest.raises(ValueError, match="vocab_size"):
        OnlineHDP(T=20, K=5, vocab_size=0)

    # Concentrations must be > 0.
    with pytest.raises(ValueError, match="alpha"):
        OnlineHDP(T=20, K=5, vocab_size=100, alpha=0.0)
    with pytest.raises(ValueError, match="gamma"):
        OnlineHDP(T=20, K=5, vocab_size=100, gamma=-1.0)
    with pytest.raises(ValueError, match="eta"):
        OnlineHDP(T=20, K=5, vocab_size=100, eta=0.0)


def test_online_hdp_init_accepts_all_optional_args():
    from spark_vi.models.online_hdp import OnlineHDP

    m = OnlineHDP(
        T=30, K=10, vocab_size=500,
        alpha=2.0, gamma=1.5, eta=0.05,
        gamma_shape=50.0,
        cavi_max_iter=50, cavi_tol=1e-3,
    )
    assert m.gamma_shape == 50.0
    assert m.cavi_max_iter == 50
    assert m.cavi_tol == 1e-3
```

- [ ] **Step 2: Run tests to verify they fail**

Command: `cd spark-vi && JAVA_HOME=$JAVA_HOME poetry run pytest tests/test_online_hdp_unit.py -v -k online_hdp_init`
Expected: 2 failed (existing `OnlineHDP.__init__` uses different arg names: `vocab_size`, `max_topics`, `omega` instead of `T`, `K`, `gamma`).

- [ ] **Step 3: Replace the `OnlineHDP` class header**

In `spark-vi/spark_vi/models/online_hdp.py`, replace the existing stub
`OnlineHDP` class with:

```python
class OnlineHDP(VIModel):
    """Online Hierarchical Dirichlet Process topic model.

    Implements Wang/Paisley/Blei 2011 stochastic VI for the HDP via the
    spark_vi VIModel contract: per-doc CAVI on workers (`local_update`),
    natural-gradient SVI step on the driver (`update_global`).

    Args:
      T: corpus-level truncation. Upper bound on the number of topics
        the model can discover; effective topic count is typically much
        smaller (the inactive corpus sticks shrink toward 0).
      K: doc-level truncation. Upper bound on topics per document.
        Should be much smaller than T — clinical visits typically span
        a handful of phenotypes, not hundreds.
      vocab_size: V, number of distinct word IDs the model handles.
      alpha: doc-level stick concentration (paper's α0). Higher → more
        topics per doc.
      gamma: corpus-level stick concentration. Higher → more discovered
        topics overall.
      eta: symmetric Dirichlet concentration for the topic-word prior.
      gamma_shape: shape parameter for the Gamma init of λ. Default 100
        matches VanillaLDA (Hoffman 2010 onlineldavb.py).
      cavi_max_iter: hard cap on doc-CAVI iterations per doc.
      cavi_tol: relative ELBO convergence threshold for doc-CAVI early
        termination.
    """

    def __init__(
        self,
        T: int,
        K: int,
        vocab_size: int,
        *,
        alpha: float = 1.0,
        gamma: float = 1.0,
        eta: float = 0.01,
        gamma_shape: float = 100.0,
        cavi_max_iter: int = 100,
        cavi_tol: float = 1e-4,
    ) -> None:
        if T < 2:
            raise ValueError(f"T must be >= 2 (need T-1 sticks), got {T}")
        if K < 2:
            raise ValueError(f"K must be >= 2 (need K-1 sticks), got {K}")
        if vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1, got {vocab_size}")
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        if gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {gamma}")
        if eta <= 0:
            raise ValueError(f"eta must be > 0, got {eta}")
        if gamma_shape <= 0:
            raise ValueError(f"gamma_shape must be > 0, got {gamma_shape}")
        if cavi_max_iter < 1:
            raise ValueError(f"cavi_max_iter must be >= 1, got {cavi_max_iter}")
        if cavi_tol <= 0:
            raise ValueError(f"cavi_tol must be > 0, got {cavi_tol}")

        self.T = int(T)
        self.K = int(K)
        self.V = int(vocab_size)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.eta = float(eta)
        self.gamma_shape = float(gamma_shape)
        self.cavi_max_iter = int(cavi_max_iter)
        self.cavi_tol = float(cavi_tol)

    # Stub methods filled in by Tasks 7-12.
    def initialize_global(self, data_summary: Any | None) -> dict[str, np.ndarray]:
        raise NotImplementedError("OnlineHDP is being built; see Task 7.")

    def local_update(
        self,
        rows: Iterable[Any],
        global_params: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError("OnlineHDP is being built; see Task 8.")

    def update_global(
        self,
        global_params: dict[str, np.ndarray],
        target_stats: dict[str, np.ndarray],
        learning_rate: float,
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError("OnlineHDP is being built; see Task 9.")
```

- [ ] **Step 4: Update `CharmPhenoHDP` ctor in lock-step**

The wrapper currently passes `(vocab_size=, max_topics=, eta=, alpha=, omega=)` to
`OnlineHDP`. Update `charmpheno/charmpheno/phenotype/charm_pheno_hdp.py`:

Open the file, find the `OnlineHDP(...)` constructor call (currently in
`__init__` around line 43-49), and update both the wrapper signature and
the inner constructor call. Replace the existing class body with:

```python
"""CharmPhenoHDP: clinical wrapper around the generic spark_vi OnlineHDP.

The wrapper adds the clinical/OMOP layer on top of the generic topic model:
concept-vocabulary handling, downstream export hooks, phenotype labels
(when the underlying OnlineHDP has converged).

See docs/architecture/TOPIC_STATE_MODELING.md for the clinical design.
"""
from __future__ import annotations

from typing import Any

from pyspark import RDD
from spark_vi.core import VIConfig, VIResult, VIRunner
from spark_vi.models import OnlineHDP


class CharmPhenoHDP:
    """Thin clinical wrapper around `spark_vi.models.OnlineHDP`.

    Constructor arg names use clinical-user-facing terms (max_topics,
    max_doc_topics) which translate to the inner model's T (corpus
    truncation) and K (doc truncation). All other args pass through.

    Args:
      vocab_size: number of distinct concept_ids in the working vocabulary.
      max_topics: HDP corpus-level truncation (upper bound on discovered
        topics).
      max_doc_topics: HDP doc-level truncation (upper bound on topics per
        visit).
      eta: topic-word Dirichlet concentration.
      alpha: doc-level stick concentration.
      gamma: corpus-level stick concentration.
      gamma_shape: shape parameter for the Gamma init of λ.
      cavi_max_iter: hard cap on doc-CAVI iterations per doc.
      cavi_tol: relative ELBO convergence threshold for doc-CAVI early
        termination.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        max_topics: int = 150,
        max_doc_topics: int = 15,
        eta: float = 0.01,
        alpha: float = 1.0,
        gamma: float = 1.0,
        gamma_shape: float = 100.0,
        cavi_max_iter: int = 100,
        cavi_tol: float = 1e-4,
    ) -> None:
        if vocab_size < 1:
            raise ValueError("vocab_size must be >= 1")
        self.vocab_size = int(vocab_size)
        self.max_topics = int(max_topics)
        self.max_doc_topics = int(max_doc_topics)
        self.model = OnlineHDP(
            T=self.max_topics,
            K=self.max_doc_topics,
            vocab_size=self.vocab_size,
            alpha=alpha,
            gamma=gamma,
            eta=eta,
            gamma_shape=gamma_shape,
            cavi_max_iter=cavi_max_iter,
            cavi_tol=cavi_tol,
        )

    def fit(
        self,
        data_rdd: RDD,
        config: VIConfig | None = None,
        data_summary: Any | None = None,
    ) -> VIResult:
        """Fit the underlying OnlineHDP on an RDD of documents.

        Raises NotImplementedError until Tasks 7-9 land.
        """
        runner = VIRunner(self.model, config=config)
        return runner.fit(data_rdd, data_summary=data_summary)
```

- [ ] **Step 5: Run tests to verify they pass**

Command: `cd spark-vi && JAVA_HOME=$JAVA_HOME poetry run pytest tests/test_online_hdp_unit.py -v`
Expected: 11 passed.

Also run `cd charmpheno && JAVA_HOME=$JAVA_HOME poetry run pytest tests/ -v` to verify
the wrapper imports cleanly (existing tests for the stub may need updating;
that happens in Task 17).

- [ ] **Step 6: Commit**

```bash
git add spark-vi/spark_vi/models/online_hdp.py \
        spark-vi/tests/test_online_hdp_unit.py \
        charmpheno/charmpheno/phenotype/charm_pheno_hdp.py
git commit -m "$(cat <<'EOF'
feat(hdp): real OnlineHDP and CharmPhenoHDP constructors

Replaces the bootstrap stub signatures. Inner OnlineHDP takes paper-style
(T, K, vocab_size) plus the four concentrations and CAVI knobs. Outer
CharmPhenoHDP keeps clinical-user-facing (max_topics, max_doc_topics)
and translates. The bogus omega param from the bootstrap stub is gone.

Class methods other than __init__ remain stubbed; subsequent tasks fill
them in.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: `OnlineHDP.initialize_global`

**Files:**
- Modify: `spark-vi/spark_vi/models/online_hdp.py`
- Modify: `spark-vi/tests/test_online_hdp_unit.py`

- [ ] **Step 1: Write the failing test**

Append to `spark-vi/tests/test_online_hdp_unit.py`:

```python
def test_initialize_global_shapes_and_validity():
    from spark_vi.models.online_hdp import OnlineHDP

    m = OnlineHDP(T=10, K=5, vocab_size=50, gamma=1.5, gamma_shape=100.0)
    np.random.seed(0)
    g = m.initialize_global(data_summary=None)

    assert set(g.keys()) == {"lambda", "u", "v"}
    assert g["lambda"].shape == (10, 50)
    assert g["u"].shape == (9,)
    assert g["v"].shape == (9,)
    # Match VanillaLDA: positive Gamma init for lambda.
    assert np.all(g["lambda"] > 0)
    # Paper-following init: u = 1, v = gamma.
    assert np.allclose(g["u"], 1.0)
    assert np.allclose(g["v"], 1.5)
```

- [ ] **Step 2: Run test to verify it fails**

Command: `cd spark-vi && JAVA_HOME=$JAVA_HOME poetry run pytest tests/test_online_hdp_unit.py::test_initialize_global_shapes_and_validity -v`
Expected: FAIL with `NotImplementedError`.

- [ ] **Step 3: Implement `initialize_global`**

In `spark-vi/spark_vi/models/online_hdp.py`, replace the stub
`initialize_global` body:

```python
    def initialize_global(self, data_summary: Any | None) -> dict[str, np.ndarray]:
        """Random Gamma(gamma_shape, 1/gamma_shape) init for λ (T, V).

        Departs from Wang's reference (which uses Gamma(1, 1) · D · 100 /
        (T·V) − η) — that scale-then-cancel-η is undocumented and not
        derived. Match-LDA is the validated choice; gamma_shape=100
        traces back to Hoffman 2010 onlineldavb.py line 126.

        Corpus sticks (u, v) start at the prior Beta(1, γ): u = 1, v = γ.
        Paper-following init. Wang's reference uses v = [T-1, T-2, ..., 1]
        ("make a uniform at beginning") which is an undocumented bias
        toward low topic indices; we don't reproduce it.
        """
        lam = np.random.gamma(
            shape=self.gamma_shape,
            scale=1.0 / self.gamma_shape,
            size=(self.T, self.V),
        )
        u = np.ones(self.T - 1, dtype=np.float64)
        v = np.full(self.T - 1, self.gamma, dtype=np.float64)
        return {"lambda": lam, "u": u, "v": v}
```

- [ ] **Step 4: Run test to verify it passes**

Command: `cd spark-vi && JAVA_HOME=$JAVA_HOME poetry run pytest tests/test_online_hdp_unit.py -v`
Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/online_hdp.py spark-vi/tests/test_online_hdp_unit.py
git commit -m "$(cat <<'EOF'
feat(hdp): OnlineHDP.initialize_global

Match-LDA Gamma init for λ (gamma_shape=100), prior-mean init for corpus
sticks (u=1, v=γ). Documents the deliberate departure from Wang's
reference on both fronts.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: `OnlineHDP.local_update`

**Files:**
- Modify: `spark-vi/spark_vi/models/online_hdp.py`
- Modify: `spark-vi/tests/test_online_hdp_unit.py`

- [ ] **Step 1: Write the failing test**

Append to `spark-vi/tests/test_online_hdp_unit.py`:

```python
def test_local_update_returns_expected_keys_and_shapes():
    """Run a tiny partition through local_update; check stats dict shape."""
    from spark_vi.core import BOWDocument
    from spark_vi.models.online_hdp import OnlineHDP

    m = OnlineHDP(T=10, K=5, vocab_size=50, gamma_shape=100.0)
    np.random.seed(0)
    g = m.initialize_global(data_summary=None)

    rows = [
        BOWDocument(
            indices=np.array([0, 1, 2], dtype=np.int32),
            counts=np.array([2.0, 1.0, 3.0], dtype=np.float64),
        ),
        BOWDocument(
            indices=np.array([10, 11], dtype=np.int32),
            counts=np.array([1.0, 1.0], dtype=np.float64),
        ),
    ]

    stats = m.local_update(rows, g)

    expected_keys = {
        "lambda_stats", "var_phi_sum_stats",
        "doc_loglik_sum", "doc_z_term_sum", "doc_c_term_sum",
        "doc_stick_kl_sum", "n_docs",
    }
    assert set(stats.keys()) == expected_keys
    assert stats["lambda_stats"].shape == (10, 50)
    assert stats["var_phi_sum_stats"].shape == (10,)
    assert float(stats["n_docs"]) == 2.0
    # All scalar accumulators must be finite.
    for k in ("doc_loglik_sum", "doc_z_term_sum",
              "doc_c_term_sum", "doc_stick_kl_sum"):
        assert np.isfinite(stats[k])
    # Suff-stat columns we touched should be non-zero; columns we didn't
    # touch should be exactly zero.
    touched = np.array([0, 1, 2, 10, 11])
    untouched = np.setdiff1d(np.arange(50), touched)
    assert np.any(stats["lambda_stats"][:, touched] > 0)
    assert np.allclose(stats["lambda_stats"][:, untouched], 0.0)


def test_local_update_combine_stats_is_elementwise_sum():
    """Default VIModel.combine_stats should sum HDP suff-stats correctly."""
    from spark_vi.core import BOWDocument
    from spark_vi.models.online_hdp import OnlineHDP

    m = OnlineHDP(T=10, K=5, vocab_size=50)
    np.random.seed(0)
    g = m.initialize_global(data_summary=None)

    docs = [
        BOWDocument(indices=np.array([0, 1], dtype=np.int32),
                    counts=np.array([1.0, 1.0])),
        BOWDocument(indices=np.array([2, 3], dtype=np.int32),
                    counts=np.array([1.0, 1.0])),
    ]
    a_stats = m.local_update(docs[:1], g)
    b_stats = m.local_update(docs[1:], g)
    combined = m.combine_stats(a_stats, b_stats)

    assert np.allclose(combined["lambda_stats"],
                       a_stats["lambda_stats"] + b_stats["lambda_stats"])
    assert np.allclose(combined["var_phi_sum_stats"],
                       a_stats["var_phi_sum_stats"] + b_stats["var_phi_sum_stats"])
    assert float(combined["n_docs"]) == 2.0
```

- [ ] **Step 2: Run tests to verify they fail**

Command: `cd spark-vi && JAVA_HOME=$JAVA_HOME poetry run pytest tests/test_online_hdp_unit.py -v -k local_update`
Expected: 2 failed (`NotImplementedError`).

- [ ] **Step 3: Implement `local_update`**

In `spark-vi/spark_vi/models/online_hdp.py`, replace the stub `local_update`:

```python
    def local_update(
        self,
        rows: Iterable[Any],
        global_params: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """E-step on one Spark partition.

        For each doc in the partition: run _doc_e_step, scatter the per-doc
        suff-stat into the partition-level (T, V) accumulator on the
        relevant columns, accumulate scalar ELBO contributions.

        See spec docs/superpowers/specs/2026-05-06-online-hdp-design.md
        for the suff-stat key contract.
        """
        lam = global_params["lambda"]              # (T, V)
        u = global_params["u"]                     # (T-1,)
        v = global_params["v"]                     # (T-1,)

        # Precompute Elogbeta and Elog_sticks_corpus once per partition;
        # both are shared across all docs in the partition.
        Elogbeta = (
            digamma(self.eta + lam)
            - digamma(self.V * self.eta + lam.sum(axis=1, keepdims=True))
        )
        Elog_sticks_corpus = _expect_log_sticks(u, v)

        lambda_stats = np.zeros((self.T, self.V), dtype=np.float64)
        var_phi_sum_stats = np.zeros(self.T, dtype=np.float64)
        doc_loglik_sum = 0.0
        doc_z_term_sum = 0.0
        doc_c_term_sum = 0.0
        doc_stick_kl_sum = 0.0
        n_docs = 0

        for doc in rows:
            indices = np.asarray(doc.indices)
            counts = np.asarray(doc.counts, dtype=np.float64)

            r = _doc_e_step(
                indices=indices, counts=counts,
                Elogbeta_doc=Elogbeta[:, indices],
                Elog_sticks_corpus=Elog_sticks_corpus,
                alpha=self.alpha,
                K=self.K,
                max_iter=self.cavi_max_iter,
                tol=self.cavi_tol,
                warmup=3,
            )

            # Scatter λ suff-stat into the touched columns. Paper Eq 21:
            #   λ_stats[t, w] += sum_t' var_phi[k=t', t]
            #   ... wait, the t/k index swap is exactly what makes HDP HDP.
            # var_phi (K, T) maps doc-atom k to corpus-atom t.
            # phi (Wt, K) maps unique-word w to doc-atom k.
            # phi_w (Wt, K) = phi * counts[:, None].
            # contrib (T, Wt) = var_phi.T @ phi_w.T
            phi_w = r["phi"] * counts[:, None]
            contrib = r["var_phi"].T @ phi_w.T            # (T, Wt)
            # Safe: BOWDocument indices are unique (no fancy-index aliasing).
            lambda_stats[:, indices] += contrib

            var_phi_sum_stats += r["var_phi"].sum(axis=0)
            doc_loglik_sum += r["doc_loglik"]
            doc_z_term_sum += r["doc_z_term"]
            doc_c_term_sum += r["doc_c_term"]
            doc_stick_kl_sum += r["doc_stick_kl"]
            n_docs += 1

        return {
            "lambda_stats": lambda_stats,
            "var_phi_sum_stats": var_phi_sum_stats,
            "doc_loglik_sum": np.array(doc_loglik_sum),
            "doc_z_term_sum": np.array(doc_z_term_sum),
            "doc_c_term_sum": np.array(doc_c_term_sum),
            "doc_stick_kl_sum": np.array(doc_stick_kl_sum),
            "n_docs": np.array(float(n_docs)),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Command: `cd spark-vi && JAVA_HOME=$JAVA_HOME poetry run pytest tests/test_online_hdp_unit.py -v`
Expected: 14 passed.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/online_hdp.py spark-vi/tests/test_online_hdp_unit.py
git commit -m "$(cat <<'EOF'
feat(hdp): OnlineHDP.local_update partition E-step

For each doc in the partition, runs _doc_e_step, scatters λ suff-stat
into touched columns, accumulates var_phi_sum_stats and the four
per-doc ELBO scalars. Returns the dense ndarray dict consumed by the
framework's default combine_stats.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: `OnlineHDP.update_global`

**Files:**
- Modify: `spark-vi/spark_vi/models/online_hdp.py`
- Modify: `spark-vi/tests/test_online_hdp_unit.py`

- [ ] **Step 1: Write the failing tests**

Append to `spark-vi/tests/test_online_hdp_unit.py`:

```python
def test_update_global_rho_zero_is_identity():
    """rho=0 ⇒ globals unchanged."""
    from spark_vi.models.online_hdp import OnlineHDP

    m = OnlineHDP(T=10, K=5, vocab_size=50, gamma=1.0)
    np.random.seed(0)
    g = m.initialize_global(data_summary=None)

    fake_stats = {
        "lambda_stats": np.full((10, 50), 7.0),
        "var_phi_sum_stats": np.full(10, 3.0),
    }
    new_g = m.update_global(g, fake_stats, learning_rate=0.0)

    assert np.allclose(new_g["lambda"], g["lambda"])
    assert np.allclose(new_g["u"], g["u"])
    assert np.allclose(new_g["v"], g["v"])


def test_update_global_rho_one_replaces_with_target():
    """rho=1 ⇒ globals become eta + target / (1 + s) / (gamma + s_tail)."""
    from spark_vi.models.online_hdp import OnlineHDP

    T, K, V = 5, 3, 10
    m = OnlineHDP(T=T, K=K, vocab_size=V, eta=0.01, gamma=2.0)
    np.random.seed(0)
    g = m.initialize_global(data_summary=None)

    s = np.array([10.0, 5.0, 2.0, 1.0, 0.5])
    fake_stats = {
        "lambda_stats": np.full((T, V), 4.0),
        "var_phi_sum_stats": s,
    }
    new_g = m.update_global(g, fake_stats, learning_rate=1.0)

    assert np.allclose(new_g["lambda"], 0.01 + 4.0)
    # u_k = 1 + s[k] for k = 0..T-2.
    assert np.allclose(new_g["u"], 1.0 + s[:T - 1])
    # v_k = gamma + cumsum(s[1:].reverse).reverse:
    # s_tail[0] = s[1] + s[2] + s[3] + s[4]
    # s_tail[1] =        s[2] + s[3] + s[4]
    # s_tail[2] =               s[3] + s[4]
    # s_tail[3] =                      s[4]
    expected_tail = np.cumsum(s[1:][::-1])[::-1]
    assert np.allclose(new_g["v"], 2.0 + expected_tail)
```

- [ ] **Step 2: Run tests to verify they fail**

Command: `cd spark-vi && JAVA_HOME=$JAVA_HOME poetry run pytest tests/test_online_hdp_unit.py -v -k update_global`
Expected: 2 failed (`NotImplementedError`).

- [ ] **Step 3: Implement `update_global`**

In `spark-vi/spark_vi/models/online_hdp.py`, replace the stub `update_global`:

```python
    def update_global(
        self,
        global_params: dict[str, np.ndarray],
        target_stats: dict[str, np.ndarray],
        learning_rate: float,
    ) -> dict[str, np.ndarray]:
        """SVI natural-gradient step on (λ, u, v). Paper Eqs 22-27.

        target_stats arrive already corpus-scaled by the runner — i.e.
        lambda_stats here is D × (sum over batch) / batch_size, same for
        var_phi_sum_stats. The full natural-gradient update on each
        global collapses to a convex combination:

          λ_new   = (1 - ρ) · λ + ρ · (η + target_lambda_stats)
          u_new   = (1 - ρ) · u + ρ · (1 + s_head)
          v_new   = (1 - ρ) · v + ρ · (γ + s_tail)

        where s = target_var_phi_sum_stats (length T),
              s_head = s[:T-1],
              s_tail[t] = sum_{l>t} s[l]   for t = 0..T-2.
        """
        rho = float(learning_rate)
        lam = global_params["lambda"]
        u = global_params["u"]
        v = global_params["v"]

        s = target_stats["var_phi_sum_stats"]
        s_head = s[: self.T - 1]
        s_tail = np.cumsum(s[1:][::-1])[::-1]  # length T-1

        new_lambda = (1.0 - rho) * lam + rho * (self.eta + target_stats["lambda_stats"])
        new_u = (1.0 - rho) * u + rho * (1.0 + s_head)
        new_v = (1.0 - rho) * v + rho * (self.gamma + s_tail)

        return {"lambda": new_lambda, "u": new_u, "v": new_v}
```

- [ ] **Step 4: Run tests to verify they pass**

Command: `cd spark-vi && JAVA_HOME=$JAVA_HOME poetry run pytest tests/test_online_hdp_unit.py -v`
Expected: 16 passed.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/online_hdp.py spark-vi/tests/test_online_hdp_unit.py
git commit -m "$(cat <<'EOF'
feat(hdp): OnlineHDP.update_global SVI step

Implements paper Eqs 22-27 collapsed to convex-combination form. λ, u,
v are each updated independently via natural-gradient SVI; runner
handles minibatch corpus rescale upstream.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: `OnlineHDP.compute_elbo`

**Files:**
- Modify: `spark-vi/spark_vi/models/online_hdp.py`
- Modify: `spark-vi/tests/test_online_hdp_unit.py`

- [ ] **Step 1: Write the failing tests**

Append to `spark-vi/tests/test_online_hdp_unit.py`:

```python
def test_compute_elbo_finite_on_initial_state():
    """ELBO is finite when called on init globals + zero stats (no docs)."""
    from spark_vi.models.online_hdp import OnlineHDP

    m = OnlineHDP(T=10, K=5, vocab_size=50, gamma=1.0)
    np.random.seed(0)
    g = m.initialize_global(data_summary=None)

    zero_stats = {
        "lambda_stats": np.zeros((10, 50)),
        "var_phi_sum_stats": np.zeros(10),
        "doc_loglik_sum": np.array(0.0),
        "doc_z_term_sum": np.array(0.0),
        "doc_c_term_sum": np.array(0.0),
        "doc_stick_kl_sum": np.array(0.0),
        "n_docs": np.array(0.0),
    }
    elbo = m.compute_elbo(g, zero_stats)
    assert np.isfinite(elbo)


def test_compute_elbo_corpus_kl_zero_at_prior():
    """When (u, v) == (1, gamma) and lambda is set to the eta prior, the
    corpus-level KL terms are exactly zero. Per-doc terms are zero with no
    docs. Therefore ELBO == 0 in that case."""
    from spark_vi.models.online_hdp import OnlineHDP

    T, V = 5, 8
    eta = 0.01
    gamma = 1.0
    m = OnlineHDP(T=T, K=3, vocab_size=V, eta=eta, gamma=gamma)

    g = {
        "lambda": np.full((T, V), eta, dtype=np.float64),  # equals prior
        "u": np.ones(T - 1),
        "v": np.full(T - 1, gamma),
    }
    zero_stats = {
        "lambda_stats": np.zeros((T, V)),
        "var_phi_sum_stats": np.zeros(T),
        "doc_loglik_sum": np.array(0.0),
        "doc_z_term_sum": np.array(0.0),
        "doc_c_term_sum": np.array(0.0),
        "doc_stick_kl_sum": np.array(0.0),
        "n_docs": np.array(0.0),
    }
    elbo = m.compute_elbo(g, zero_stats)
    assert np.isclose(elbo, 0.0, atol=1e-9)
```

- [ ] **Step 2: Run tests to verify they fail**

Command: `cd spark-vi && JAVA_HOME=$JAVA_HOME poetry run pytest tests/test_online_hdp_unit.py -v -k compute_elbo`
Expected: 2 failed (`compute_elbo` not yet overridden — falls back to base
class which raises `NotImplementedError` or returns a placeholder).

Check the base `VIModel.compute_elbo` to see what the default does — if it
returns NaN or raises, both tests will register fail. If it returns 0.0
naively, the first test (which expects finite) will pass spuriously; in
that case the override is still required and the second test (matched-prior
KL = 0) will fail because we want our real KL machinery exercised.

- [ ] **Step 3: Implement `compute_elbo`**

In `spark-vi/spark_vi/models/online_hdp.py`, add inside the `OnlineHDP` class
(after `update_global`):

```python
    def compute_elbo(
        self,
        global_params: dict[str, np.ndarray],
        aggregated_stats: dict[str, np.ndarray],
    ) -> float:
        """Full ELBO from paper Eq 14.

        Per-doc terms come from aggregated_stats (already summed by
        local_update across docs and combine_stats across partitions).
        Corpus-level KL terms are computed driver-side from current
        global params:
          KL[q(β') ‖ p(β')]  : Beta(u_k, v_k) ‖ Beta(1, γ), summed k.
          KL[q(φ) ‖ p(φ)]    : Dirichlet(λ_t) ‖ Dirichlet(η · 1_V), summed t.

        In minibatch mode, the per-doc piece is the *minibatch* sum, not
        corpus-rescaled. The reported ELBO is therefore a noisy unbiased
        estimator of the corpus-scale bound; the ELBO trend test uses
        smoothed-endpoint comparison to absorb the variance.
        """
        per_doc = (
            float(aggregated_stats["doc_loglik_sum"])
            + float(aggregated_stats["doc_z_term_sum"])
            + float(aggregated_stats["doc_c_term_sum"])
            - float(aggregated_stats["doc_stick_kl_sum"])
        )

        # Corpus stick KL — summed over k = 0..T-2.
        corpus_stick_kl = float(_beta_kl(
            global_params["u"], global_params["v"],
            prior_a=1.0, prior_b=self.gamma,
        ).sum())

        # Topic Dirichlet KL — Dirichlet(λ_t) ‖ Dirichlet(η · 1_V), summed t.
        # KL[Dir(λ) ‖ Dir(η)] = lgamma(sum(λ)) - sum(lgamma(λ))
        #                     - lgamma(V·η)   + V·lgamma(η)
        #                     + sum((λ - η) · (digamma(λ) - digamma(sum(λ))))
        lam = global_params["lambda"]
        lam_sum = lam.sum(axis=1)                                   # (T,)
        topic_kl = float(np.sum(
            gammaln(lam_sum) - gammaln(lam).sum(axis=1)
            - gammaln(self.V * self.eta) + self.V * gammaln(self.eta)
            + np.sum(
                (lam - self.eta)
                * (digamma(lam) - digamma(lam_sum)[:, None]),
                axis=1,
            )
        ))

        return per_doc - corpus_stick_kl - topic_kl
```

- [ ] **Step 4: Run tests to verify they pass**

Command: `cd spark-vi && JAVA_HOME=$JAVA_HOME poetry run pytest tests/test_online_hdp_unit.py -v`
Expected: 18 passed.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/online_hdp.py spark-vi/tests/test_online_hdp_unit.py
git commit -m "$(cat <<'EOF'
feat(hdp): OnlineHDP.compute_elbo full ELBO with corpus KL terms

Per-doc terms read from aggregated_stats (already summed); corpus stick
and topic Dirichlet KL terms computed driver-side from globals. Matched-
prior test pins KL=0 at (u=1, v=γ, λ=η) as a deterministic regression
gate.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: `OnlineHDP.infer_local`

**Files:**
- Modify: `spark-vi/spark_vi/models/online_hdp.py`
- Modify: `spark-vi/tests/test_online_hdp_unit.py`

- [ ] **Step 1: Write the failing test**

Append to `spark-vi/tests/test_online_hdp_unit.py`:

```python
def test_infer_local_returns_simplex_theta():
    """infer_local returns the doc variational posterior + a θ derived from it."""
    from spark_vi.core import BOWDocument
    from spark_vi.models.online_hdp import OnlineHDP

    T, K, V = 10, 5, 50
    m = OnlineHDP(T=T, K=K, vocab_size=V)
    np.random.seed(0)
    g = m.initialize_global(data_summary=None)

    doc = BOWDocument(
        indices=np.array([0, 1, 2], dtype=np.int32),
        counts=np.array([2.0, 1.0, 3.0], dtype=np.float64),
    )
    out = m.infer_local(doc, g)

    expected = {"a", "b", "phi", "var_phi", "theta"}
    assert set(out.keys()) == expected
    assert out["theta"].shape == (T,)
    assert np.isclose(out["theta"].sum(), 1.0)
    assert np.all(out["theta"] >= 0)
```

- [ ] **Step 2: Run test to verify it fails**

Command: `cd spark-vi && JAVA_HOME=$JAVA_HOME poetry run pytest tests/test_online_hdp_unit.py::test_infer_local_returns_simplex_theta -v`
Expected: FAIL.

- [ ] **Step 3: Implement `infer_local`**

In `spark-vi/spark_vi/models/online_hdp.py`, add inside the `OnlineHDP` class
(after `compute_elbo`):

```python
    def infer_local(
        self,
        row: Any,
        global_params: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Single-doc frozen-globals doc-CAVI.

        Runs `_doc_e_step` on one BOWDocument with the broadcast Elogβ and
        Elog_sticks_corpus held fixed (no global update). Returns the
        variational posterior plus the derived per-doc θ — the
        topic-proportion vector that downstream Stage-2 OU consumes.

        Per-doc θ derivation: θ_t = Σ_k π_k(a, b) · var_phi[k, t], where
        π_k(a, b) is the doc stick-breaking mean
        (E[π_k] = a_k / (a_k + b_k) · prod_{l<k}(b_l / (a_l + b_l)))
        for k = 0..K-2, with the last atom absorbing the remainder.
        """
        lam = global_params["lambda"]
        u = global_params["u"]
        v = global_params["v"]

        Elogbeta = (
            digamma(self.eta + lam)
            - digamma(self.V * self.eta + lam.sum(axis=1, keepdims=True))
        )
        Elog_sticks_corpus = _expect_log_sticks(u, v)

        indices = np.asarray(row.indices)
        counts = np.asarray(row.counts, dtype=np.float64)

        r = _doc_e_step(
            indices=indices, counts=counts,
            Elogbeta_doc=Elogbeta[:, indices],
            Elog_sticks_corpus=Elog_sticks_corpus,
            alpha=self.alpha,
            K=self.K,
            max_iter=self.cavi_max_iter,
            tol=self.cavi_tol,
            warmup=3,
        )

        # Doc stick mean: pi_k = E[π_k] under Beta(a, b) factors.
        a, b = r["a"], r["b"]
        E_W = a / (a + b)                              # length K-1
        E_1mW = b / (a + b)                            # length K-1
        pi_doc = np.zeros(self.K, dtype=np.float64)
        pi_doc[: self.K - 1] = E_W * np.concatenate([[1.0], np.cumprod(E_1mW)[:-1]])
        pi_doc[self.K - 1] = 1.0 - pi_doc[: self.K - 1].sum()

        # θ = pi_doc @ var_phi  → shape (T,)
        theta = pi_doc @ r["var_phi"]

        return {
            "a": a,
            "b": b,
            "phi": r["phi"],
            "var_phi": r["var_phi"],
            "theta": theta,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Command: `cd spark-vi && JAVA_HOME=$JAVA_HOME poetry run pytest tests/test_online_hdp_unit.py -v`
Expected: 19 passed.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/online_hdp.py spark-vi/tests/test_online_hdp_unit.py
git commit -m "$(cat <<'EOF'
feat(hdp): OnlineHDP.infer_local frozen-globals transform

Runs _doc_e_step with no side effects on a single BOWDocument; derives
per-doc θ via the doc-stick mean composition. This is the primitive the
wrapper.transform() and held-out evaluation will call.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: `OnlineHDP.iteration_summary`

**Files:**
- Modify: `spark-vi/spark_vi/models/online_hdp.py`
- Modify: `spark-vi/tests/test_online_hdp_unit.py`

- [ ] **Step 1: Write the failing test**

Append to `spark-vi/tests/test_online_hdp_unit.py`:

```python
def test_iteration_summary_returns_string():
    """iteration_summary returns a short non-empty diagnostic string."""
    from spark_vi.models.online_hdp import OnlineHDP

    m = OnlineHDP(T=10, K=5, vocab_size=50)
    np.random.seed(0)
    g = m.initialize_global(data_summary=None)
    s = m.iteration_summary(g)

    assert isinstance(s, str)
    assert len(s) > 0
    # Must reference active topic count somewhere in the line.
    assert "active" in s.lower() or "topics" in s.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Command: `cd spark-vi && JAVA_HOME=$JAVA_HOME poetry run pytest tests/test_online_hdp_unit.py::test_iteration_summary_returns_string -v`
Expected: FAIL (or pass with the base-class default — depends on `VIModel.iteration_summary`).

- [ ] **Step 3: Implement `iteration_summary`**

In `spark-vi/spark_vi/models/online_hdp.py`, add inside the `OnlineHDP` class
(after `infer_local`):

```python
    def iteration_summary(self, global_params: dict[str, np.ndarray]) -> str:
        """Short live-training diagnostic.

        Reports:
          - Effective active-topic count: #{k : E[β_k] > 1/(2T)}.
            Threshold 1/(2T) is "carries half-uniform mass" — a cheap
            proxy for "this corpus topic is being used".
          - Top-3 corpus stick weights, descending.
          - Spread of λ row sums (max / min).
        """
        u = global_params["u"]
        v = global_params["v"]
        lam = global_params["lambda"]

        # E[β_k] from u, v via stick-breaking mean.
        E_W = u / (u + v)                                       # length T-1
        E_1mW = v / (u + v)                                     # length T-1
        E_beta = np.zeros(self.T, dtype=np.float64)
        E_beta[: self.T - 1] = E_W * np.concatenate([[1.0], np.cumprod(E_1mW)[:-1]])
        E_beta[self.T - 1] = 1.0 - E_beta[: self.T - 1].sum()

        n_active = int(np.sum(E_beta > 1.0 / (2.0 * self.T)))
        top3 = np.sort(E_beta)[::-1][:3]
        lam_sum = lam.sum(axis=1)
        spread = float(lam_sum.max() / max(lam_sum.min(), 1e-12))

        return (
            f"active topics={n_active}/{self.T}, "
            f"top-3 E[β]={top3[0]:.3f},{top3[1]:.3f},{top3[2]:.3f}, "
            f"λ-row-sum spread={spread:.2f}"
        )
```

- [ ] **Step 4: Run test to verify it passes**

Command: `cd spark-vi && JAVA_HOME=$JAVA_HOME poetry run pytest tests/test_online_hdp_unit.py -v`
Expected: 20 passed.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/online_hdp.py spark-vi/tests/test_online_hdp_unit.py
git commit -m "$(cat <<'EOF'
feat(hdp): OnlineHDP.iteration_summary diagnostic line

Reports effective active topic count (E[β_k] > 1/(2T)), top-3 corpus
stick weights, and λ row-sum spread. Surfaces during live training.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Slow-tier ELBO trend smoke test

**Files:**
- Create: `spark-vi/tests/test_online_hdp_integration.py`

- [ ] **Step 1: Write the failing test**

Create `spark-vi/tests/test_online_hdp_integration.py`:

```python
"""End-to-end Spark-local integration tests for OnlineHDP.

Hermetic by construction: each test builds its own synthetic LDA/HDP
dataset inside the test, no external data dependencies.

Scope: verify the VIRunner ↔ OnlineHDP integration converges sensibly.
Recovery quality (does fitted β match ground truth?) is in
`test_online_hdp_synthetic_recovery_top_topics`. ELBO trend is here.
"""
import numpy as np
import pytest


def _generate_synthetic_corpus(D, V, K, docs_avg_len, seed):
    """Generate (true_beta, docs_as_BOWDocuments) under standard LDA.

    LDA-shaped data is fine for testing HDP fits; the HDP will simply
    learn that K topics are active and the rest of the truncation is
    unused. Same generator pattern as test_lda_integration.py.
    """
    from spark_vi.core import BOWDocument
    rng = np.random.default_rng(seed)

    true_beta = rng.dirichlet(np.full(V, 0.05), size=K)
    docs = []
    for d in range(D):
        doc_len = max(2, int(rng.poisson(docs_avg_len)))
        theta_d = rng.dirichlet(np.full(K, 0.3))
        topics = rng.choice(K, size=doc_len, p=theta_d)
        words = np.array([rng.choice(V, p=true_beta[t]) for t in topics])
        unique, counts = np.unique(words, return_counts=True)
        docs.append(BOWDocument(
            indices=unique.astype(np.int32),
            counts=counts.astype(np.float64),
        ))
    return true_beta, docs


@pytest.mark.slow
def test_online_hdp_short_fit_returns_finite_elbo_trace(spark):
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.models import OnlineHDP

    _, docs = _generate_synthetic_corpus(D=200, V=50, K=5,
                                         docs_avg_len=15, seed=0)
    rdd = spark.sparkContext.parallelize(docs, numSlices=2)

    np.random.seed(0)
    model = OnlineHDP(T=10, K=5, vocab_size=50)
    runner = VIRunner(model, config=VIConfig(max_iterations=10))
    result = runner.fit(rdd)

    assert result.elbo_trace is not None
    assert len(result.elbo_trace) >= 10
    assert all(np.isfinite(v) for v in result.elbo_trace)
    assert np.all(result.global_params["lambda"] > 0)
    assert np.all(result.global_params["u"] > 0)
    assert np.all(result.global_params["v"] > 0)


@pytest.mark.slow
def test_online_hdp_elbo_smoothed_endpoints_show_overall_improvement(spark):
    """Smoothed-endpoint ELBO trend must improve over a 30+iter fit.

    Mirrors test_lda_integration.py:test_vanilla_lda_elbo_smoothed_*.
    NOT a monotonicity check — SVI noise produces 100+ ELBO-unit drops
    mid-trace even on healthy fits. Endpoint trend on the smoothed
    series catches sign errors and runaway divergence.
    """
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.models import OnlineHDP

    _, docs = _generate_synthetic_corpus(D=200, V=50, K=5,
                                         docs_avg_len=15, seed=42)
    rdd = spark.sparkContext.parallelize(docs, numSlices=2)

    np.random.seed(42)
    model = OnlineHDP(T=10, K=5, vocab_size=50)
    runner = VIRunner(model, config=VIConfig(max_iterations=30))
    result = runner.fit(rdd)

    trace = np.asarray(result.elbo_trace)
    window = 10
    assert len(trace) >= window, (
        f"need at least {window} iterations for smoothing, got {len(trace)}"
    )
    smooth = np.convolve(trace, np.ones(window) / window, mode="valid")
    assert smooth[-1] > smooth[0], (
        f"Smoothed ELBO endpoints went backward: "
        f"start={smooth[0]:.3f}, end={smooth[-1]:.3f}. "
        f"Indicates a sign error or wrong-direction update."
    )
```

- [ ] **Step 2: Run tests to verify they pass**

Command: `cd spark-vi && JAVA_HOME=$JAVA_HOME poetry run pytest tests/test_online_hdp_integration.py -v -m slow`
Expected: 2 passed (these are smoke + regression gates against an already-correct implementation).

If `test_online_hdp_elbo_smoothed_endpoints_show_overall_improvement` is
flaky on first run due to SVI noise, try a different seed or expand the
window to 15 — both are documented escape hatches in the spec.

- [ ] **Step 3: Commit**

```bash
git add spark-vi/tests/test_online_hdp_integration.py
git commit -m "$(cat <<'EOF'
test(hdp): slow-tier ELBO trend integration tests

Two slow-tier tests on Spark-local: (1) short fit produces a finite
ELBO trace and well-formed globals; (2) 30-iter fit shows smoothed-
endpoint ELBO improvement. Mirror of the LDA integration tests, with
the same caveat that this is not a monotonicity check.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Slow-tier synthetic recovery test

**Files:**
- Modify: `spark-vi/tests/test_online_hdp_integration.py`

- [ ] **Step 1: Write the failing test**

Append to `spark-vi/tests/test_online_hdp_integration.py`:

```python
@pytest.mark.slow
def test_online_hdp_synthetic_recovery_top_topics(spark):
    """Top-K_true topics by usage recover true word distributions.

    D=2000 LDA-generated docs with K_true=5 active topics, fit with T=20
    truncation. Hungarian-match the top-5 fitted topics by var_phi mass
    against the true topics, assert cosine sim > 0.7 on the matched set.

    Threshold 0.7 (not 0.9 from the spec): empirical SVI on D=2000
    synthetic data leaves residual topic-collapse signal — Hoffman 2010
    used D=100k+. Same posture as the LDA recovery story (no recovery
    test on small-D for LDA either; we get a weaker version here only
    because HDP's truncation gives us more headroom). Tighten to 0.9 if
    a future fix lets us recover sharper topics.
    """
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.models import OnlineHDP
    from scipy.optimize import linear_sum_assignment

    K_true = 5
    true_beta, docs = _generate_synthetic_corpus(
        D=2000, V=80, K=K_true, docs_avg_len=20, seed=7)
    rdd = spark.sparkContext.parallelize(docs, numSlices=4)

    np.random.seed(7)
    model = OnlineHDP(T=20, K=K_true, vocab_size=80)
    runner = VIRunner(model, config=VIConfig(max_iterations=80))
    result = runner.fit(rdd)

    # Recover beta-hat from lambda: each row normalized.
    lam = result.global_params["lambda"]
    beta_hat = lam / lam.sum(axis=1, keepdims=True)

    # Pick the top K_true fitted topics by lambda row sum (proxy for usage).
    top_idx = np.argsort(lam.sum(axis=1))[::-1][:K_true]
    fitted = beta_hat[top_idx]

    # Cosine sim matrix: (K_true, K_true).
    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    sim = np.array([[cos(true_beta[i], fitted[j]) for j in range(K_true)]
                    for i in range(K_true)])

    # Hungarian matching maximizes sum of similarities.
    row_ind, col_ind = linear_sum_assignment(-sim)
    matched_sims = sim[row_ind, col_ind]

    assert matched_sims.mean() > 0.7, (
        f"Mean matched cosine similarity {matched_sims.mean():.3f} < 0.7. "
        f"Per-topic sims: {matched_sims}"
    )
```

- [ ] **Step 2: Run test to verify it passes**

Command: `cd spark-vi && JAVA_HOME=$JAVA_HOME poetry run pytest tests/test_online_hdp_integration.py::test_online_hdp_synthetic_recovery_top_topics -v -m slow`
Expected: PASS. If the test fails with all-similar topics (recovery quality
worse than 0.7), this is an SVI scale issue and the test threshold may need
to drop. Document any threshold relaxation in the test docstring; do NOT
silently weaken it.

- [ ] **Step 3: Commit**

```bash
git add spark-vi/tests/test_online_hdp_integration.py
git commit -m "$(cat <<'EOF'
test(hdp): synthetic recovery on D=2000 with Hungarian matching

Generates LDA-shaped data with K_true=5, fits with T=20 truncation,
checks the top-5 fitted topics recover true word distributions to
mean cosine similarity > 0.7 after Hungarian matching. Threshold
documented as a lower bound informed by SVI scale limits, not an
aspirational target.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: Slow-tier infer_local round-trip test

**Files:**
- Modify: `spark-vi/tests/test_online_hdp_integration.py`

- [ ] **Step 1: Write the failing test**

Append to `spark-vi/tests/test_online_hdp_integration.py`:

```python
@pytest.mark.slow
def test_online_hdp_infer_local_round_trip(spark):
    """Fit on training corpus, run infer_local on a held-out doc.

    Asserts the doc-CAVI converges, returns simplex-valid θ, and
    concentrates mass on a small subset of corpus topics (effective
    sparsity expected from a sparse synthetic generator).
    """
    from spark_vi.core import BOWDocument, VIConfig, VIRunner
    from spark_vi.models import OnlineHDP

    _, docs = _generate_synthetic_corpus(
        D=200, V=50, K=5, docs_avg_len=15, seed=11)
    rdd = spark.sparkContext.parallelize(docs, numSlices=2)

    np.random.seed(11)
    model = OnlineHDP(T=10, K=5, vocab_size=50)
    runner = VIRunner(model, config=VIConfig(max_iterations=20))
    result = runner.fit(rdd)

    # Held-out doc with a different seed — words fall in the same vocab
    # but the doc is genuinely new.
    rng = np.random.default_rng(999)
    held_out = BOWDocument(
        indices=np.sort(rng.choice(50, size=8, replace=False).astype(np.int32)),
        counts=rng.gamma(2.0, 1.0, 8).astype(np.float64),
    )

    out = model.infer_local(held_out, result.global_params)
    theta = out["theta"]

    assert theta.shape == (10,)
    assert np.isclose(theta.sum(), 1.0, atol=1e-6)
    assert np.all(theta >= 0)
    # Effective sparsity: > 80% of mass on at most half the topics.
    sorted_theta = np.sort(theta)[::-1]
    half = max(1, len(sorted_theta) // 2)
    assert sorted_theta[:half].sum() > 0.8, (
        f"θ not concentrated; top-{half}: {sorted_theta[:half].sum():.3f}"
    )
```

- [ ] **Step 2: Run test to verify it passes**

Command: `cd spark-vi && JAVA_HOME=$JAVA_HOME poetry run pytest tests/test_online_hdp_integration.py::test_online_hdp_infer_local_round_trip -v -m slow`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add spark-vi/tests/test_online_hdp_integration.py
git commit -m "$(cat <<'EOF'
test(hdp): infer_local round-trip on held-out doc

Fit on D=200, run infer_local on a fresh doc, assert simplex θ with
expected sparsity. Exercises the standalone-inference primitive
required for held-out evaluation and future serving paths.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 16: Wang-reference cross-check fixture (optional / deferrable)

**Files:**
- Create: `spark-vi/tests/fixtures/online_hdp_wang_reference.json`
- Modify: `spark-vi/tests/test_online_hdp_unit.py`

**Note:** This task validates that our `_doc_e_step` agrees with Wang's
reference Python (https://github.com/blei-lab/online-hdp) on a single doc,
five iterations, fixed seed. Wang's code is Python 2; generating the fixture
requires either a Python-2 environment or porting just `doc_e_step` into a
disposable Python-3 script. If the fixture-generation logistics block
progress, defer this task and proceed to Task 17 — the per-iter monotonicity
test (Task 5) and the synthetic recovery test (Task 14) together provide
strong correctness coverage.

If you skip: open an issue or add a NOTE comment in
`test_online_hdp_unit.py` recording the deferral.

If you proceed:

- [ ] **Step 1: Generate the fixture externally**

Create a one-shot script that uses the same RNG seed (0), the same
`_peaked_elogbeta` setup, and the same doc, calling Wang's
`onlinehdp.py:doc_e_step` directly. Capture (a, b, phi, var_phi) at
iter=5, write to `tests/fixtures/online_hdp_wang_reference.json`.

The fixture file structure:

```json
{
  "T": 10,
  "K": 5,
  "V": 20,
  "alpha": 1.0,
  "warmup": 3,
  "iters": 5,
  "indices": [0, 1, 2, 3, 4],
  "counts": [5.0, 3.0, 2.0, 4.0, 1.0],
  "Elogbeta_doc": [[...]],
  "Elog_sticks_corpus": [...],
  "expected_a": [...],
  "expected_b": [...],
  "expected_phi": [[...]],
  "expected_var_phi": [[...]]
}
```

Commit the fixture (it's a small text file).

- [ ] **Step 2: Write the comparison test**

Append to `spark-vi/tests/test_online_hdp_unit.py`:

```python
def test_doc_e_step_matches_wang_reference_fixture():
    """Cross-check our _doc_e_step against Wang's reference Python.

    Fixture generated externally; see Task 16 in the implementation plan.
    """
    import json
    from pathlib import Path
    from spark_vi.models.online_hdp import _doc_e_step

    fixture_path = (Path(__file__).parent
                    / "fixtures" / "online_hdp_wang_reference.json")
    if not fixture_path.exists():
        pytest.skip("Wang reference fixture not generated; see Task 16.")

    fx = json.loads(fixture_path.read_text())

    result = _doc_e_step(
        indices=np.array(fx["indices"], dtype=np.int32),
        counts=np.array(fx["counts"], dtype=np.float64),
        Elogbeta_doc=np.array(fx["Elogbeta_doc"]),
        Elog_sticks_corpus=np.array(fx["Elog_sticks_corpus"]),
        alpha=fx["alpha"],
        K=fx["K"],
        max_iter=fx["iters"],
        tol=0.0,                     # never early-break; match fixed iter
        warmup=fx["warmup"],
    )

    assert np.allclose(result["a"], fx["expected_a"], atol=1e-5)
    assert np.allclose(result["b"], fx["expected_b"], atol=1e-5)
    assert np.allclose(result["phi"], fx["expected_phi"], atol=1e-5)
    assert np.allclose(result["var_phi"], fx["expected_var_phi"], atol=1e-5)
```

- [ ] **Step 3: Run test**

Command: `cd spark-vi && JAVA_HOME=$JAVA_HOME poetry run pytest tests/test_online_hdp_unit.py::test_doc_e_step_matches_wang_reference_fixture -v`
Expected: PASS (if fixture generated) or SKIP (if deferred).

- [ ] **Step 4: Commit**

```bash
git add spark-vi/tests/fixtures/online_hdp_wang_reference.json \
        spark-vi/tests/test_online_hdp_unit.py
git commit -m "$(cat <<'EOF'
test(hdp): Wang reference cross-check fixture

Bit-matches our _doc_e_step against Wang's reference Python at iter=5
on a single fixed-seed doc. Skips gracefully if the fixture isn't
generated (see Task 16 in the implementation plan).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 17: `CharmPhenoHDP.transform` and wrapper tests

**Files:**
- Modify: `charmpheno/charmpheno/phenotype/charm_pheno_hdp.py`
- Modify: `charmpheno/tests/test_charm_pheno_hdp_wrapper.py` (existing file — delete the NotImplementedError gate, update construction tests, add fit/transform smoke)

The existing test file at `charmpheno/tests/test_charm_pheno_hdp_wrapper.py`
gates the wrapper against the stub state (e.g., `test_charm_pheno_hdp_fit_raises_not_implemented`).
Once the inner OnlineHDP is real, those gates flip — fit no longer raises,
and the constructor takes new arguments. We replace the file's contents
rather than fight the existing assertions.

There is already a session-scoped `spark` fixture in `charmpheno/tests/conftest.py` —
new tests reuse it directly (no module-local fixture needed).

- [ ] **Step 1: Add `.transform()` to the wrapper**

Open `charmpheno/charmpheno/phenotype/charm_pheno_hdp.py` and append (after the
`fit` method) the following:

```python
    def transform(self, data_rdd: RDD) -> RDD:
        """Per-doc frozen-globals inference.

        Requires `.fit()` to have populated `self.last_result.global_params`.
        Maps each input row through `OnlineHDP.infer_local` and yields
        (input_row, theta) pairs where theta is the length-T topic
        proportion vector for that doc.

        Stub during bootstrap; fully wired once VIRunner.transform lands
        for HDP. Until then, this method runs infer_local in driver-side
        Python — adequate for small held-out sets but not for production.
        """
        if not hasattr(self, "_fitted_globals"):
            raise RuntimeError(
                "CharmPhenoHDP.transform requires fit() first; "
                "_fitted_globals is unset."
            )
        globals_bcast = data_rdd.context.broadcast(self._fitted_globals)
        model = self.model

        def _per_partition(rows):
            g = globals_bcast.value
            for row in rows:
                out = model.infer_local(row, g)
                yield (row, out["theta"])

        return data_rdd.mapPartitions(_per_partition)
```

Also update `.fit` to capture globals:

```python
    def fit(
        self,
        data_rdd: RDD,
        config: VIConfig | None = None,
        data_summary: Any | None = None,
    ) -> VIResult:
        """Fit the underlying OnlineHDP on an RDD of documents."""
        runner = VIRunner(self.model, config=config)
        result = runner.fit(data_rdd, data_summary=data_summary)
        self._fitted_globals = result.global_params
        return result
```

- [ ] **Step 2: Replace the wrapper test file**

Open `charmpheno/tests/test_charm_pheno_hdp_wrapper.py` and replace the
entire contents with:

```python
"""End-to-end wrapper tests for CharmPhenoHDP.

The wrapper is a thin clinical layer around `spark_vi.models.OnlineHDP`.
Construction validation runs in unit tier; fit/transform exercise the
full Spark-local path in slow tier.
"""
import numpy as np
import pytest


def _tiny_corpus():
    from spark_vi.core import BOWDocument
    rng = np.random.default_rng(0)
    docs = []
    for d in range(20):
        n = int(rng.integers(2, 6))
        idx = np.sort(rng.choice(20, size=n, replace=False)).astype(np.int32)
        cnt = rng.gamma(2.0, 1.0, n).astype(np.float64)
        docs.append(BOWDocument(indices=idx, counts=cnt))
    return docs


def test_charm_pheno_hdp_constructor_validates():
    from charmpheno.phenotype import CharmPhenoHDP

    with pytest.raises(ValueError):
        CharmPhenoHDP(vocab_size=0)

    m = CharmPhenoHDP(vocab_size=20, max_topics=5, max_doc_topics=3)
    assert m.vocab_size == 20
    assert m.max_topics == 5
    assert m.max_doc_topics == 3
    # Translation to inner OnlineHDP names.
    assert m.model.T == 5
    assert m.model.K == 3


def test_charm_pheno_hdp_exposes_underlying_online_hdp():
    """The wrapper's .model attribute is the spark_vi OnlineHDP instance."""
    from spark_vi.models import OnlineHDP
    from charmpheno.phenotype import CharmPhenoHDP

    m = CharmPhenoHDP(vocab_size=10)
    assert isinstance(m.model, OnlineHDP)


@pytest.mark.slow
def test_charm_pheno_hdp_fit_smoke_tiny(spark):
    """Tiny end-to-end fit completes; ELBO trace is finite."""
    from charmpheno.phenotype import CharmPhenoHDP
    from spark_vi.core import VIConfig

    docs = _tiny_corpus()
    rdd = spark.sparkContext.parallelize(docs, numSlices=2)

    np.random.seed(0)
    m = CharmPhenoHDP(vocab_size=20, max_topics=5, max_doc_topics=3)
    result = m.fit(rdd, config=VIConfig(max_iterations=5))

    assert result.elbo_trace is not None
    assert all(np.isfinite(v) for v in result.elbo_trace)


@pytest.mark.slow
def test_charm_pheno_hdp_transform_returns_simplex_thetas(spark):
    """fit then transform; per-doc θ is a length-T simplex vector."""
    from charmpheno.phenotype import CharmPhenoHDP
    from spark_vi.core import VIConfig

    docs = _tiny_corpus()
    rdd = spark.sparkContext.parallelize(docs, numSlices=2)

    np.random.seed(0)
    m = CharmPhenoHDP(vocab_size=20, max_topics=5, max_doc_topics=3)
    m.fit(rdd, config=VIConfig(max_iterations=5))

    out = m.transform(rdd).collect()
    assert len(out) == len(docs)
    for _, theta in out:
        assert theta.shape == (5,)
        assert np.isclose(theta.sum(), 1.0, atol=1e-6)
        assert np.all(theta >= 0)
```

Note that the prior `test_charm_pheno_hdp_fit_raises_not_implemented` test
is intentionally removed: fit no longer raises, since the inner OnlineHDP
is now implemented.

- [ ] **Step 3: Run tests to verify they pass**

Command: `cd charmpheno && JAVA_HOME=$JAVA_HOME poetry run pytest tests/test_charm_pheno_hdp_wrapper.py -v`
Expected: 2 passed (the two construction tests; slow-tier excluded by default).

Then run with slow tier enabled:
Command: `cd charmpheno && JAVA_HOME=$JAVA_HOME poetry run pytest tests/test_charm_pheno_hdp_wrapper.py -v -m slow`
Expected: 2 passed (the slow-tier fit + transform smoke).

- [ ] **Step 4: Commit**

```bash
git add charmpheno/charmpheno/phenotype/charm_pheno_hdp.py \
        charmpheno/tests/test_charm_pheno_hdp_wrapper.py
git commit -m "$(cat <<'EOF'
feat(charm-hdp): wrapper transform + smoke tests

Wires .transform() to dispatch infer_local per-row across the RDD using
broadcasted fitted globals. Replaces the bootstrap-era wrapper tests
(which gated NotImplementedError) with construction validation +
slow-tier fit/transform smokes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 18: Architecture doc updates

**Files:**
- Modify: `docs/architecture/SPARK_VI_FRAMEWORK.md` lines 260-306
- Modify: `docs/architecture/TOPIC_STATE_MODELING.md` lines 289-301 area

- [ ] **Step 1: Update the OnlineHDP code sketch in `SPARK_VI_FRAMEWORK.md`**

Find the existing sketch at L260-306 and replace it with one that explicitly
labels T and K. The sketch should match what we actually shipped:

```python
class OnlineHDP(VIModel):
    def __init__(
        self,
        T: int,                  # corpus-level truncation (paper's K)
        K: int,                  # doc-level truncation (paper's T)
        vocab_size: int,
        *,
        alpha: float = 1.0,      # doc-stick concentration (paper's α0)
        gamma: float = 1.0,      # corpus-stick concentration
        eta: float = 0.01,       # topic-Dirichlet concentration
        # plus init / CAVI knobs — see the live signature.
    ):
        ...

    def initialize_global(self, data_summary):
        # Returns {"lambda": (T,V), "u": (T-1,), "v": (T-1,)}.
        ...

    def local_update(self, partition_data, global_params):
        # Per-doc CAVI: phi (Wt,K), var_phi (K,T), a/b (K-1,)
        # Returns lambda_stats, var_phi_sum_stats, doc-ELBO scalars.
        ...

    def update_global(self, global_params, target_stats, learning_rate):
        # Natural-gradient SVI on (λ, u, v).
        ...

    def infer_local(self, row, global_params):
        # Frozen-globals doc-CAVI; returns θ for downstream Stage-2.
        ...
```

Note in the surrounding prose that this is a sketch — the live signature is
`spark-vi/spark_vi/models/online_hdp.py`.

- [ ] **Step 2: Add naming-convention paragraph to `TOPIC_STATE_MODELING.md`**

Find the existing parameter table (around L289-301) and add immediately after it:

> **Notation: paper vs code.** This document and the implementation use
> `T = corpus-level truncation`, `K = doc-level truncation`. Wang/Paisley/Blei
> 2011 uses these letters in the *opposite* direction (paper's `K` is the
> corpus level, paper's `T` is the doc level). Wang's reference Python
> implementation, the intel-spark Scala port, and our code all use the
> convention adopted here. When porting equations from the paper, swap T ↔ K.

- [ ] **Step 3: Run a doc-format check**

Command: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && grep -n "T = corpus\|T=corpus\|corpus-level truncation" docs/architecture/TOPIC_STATE_MODELING.md docs/architecture/SPARK_VI_FRAMEWORK.md`
Expected: at least two distinct hits — one in each doc — confirming the convention is now stated explicitly.

- [ ] **Step 4: Commit**

```bash
git add docs/architecture/SPARK_VI_FRAMEWORK.md \
        docs/architecture/TOPIC_STATE_MODELING.md
git commit -m "$(cat <<'EOF'
docs(arch): clarify T/K naming in HDP discussion

SPARK_VI_FRAMEWORK.md: re-emit the OnlineHDP code sketch with explicit
truncation-level comments, point readers to the live signature.
TOPIC_STATE_MODELING.md: add a naming-convention paragraph after the
parameter table making the paper-vs-code letter inversion explicit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 19: ADR 0011

**Files:**
- Create: `docs/decisions/0011-online-hdp-design.md`

- [ ] **Step 1: Write the ADR**

Create `docs/decisions/0011-online-hdp-design.md`:

```markdown
# ADR 0011: Online HDP v1 Design Decisions

**Date:** 2026-05-06
**Status:** Accepted
**Supersedes:** none
**Related:** ADR 0007 (VIModel inference capability), ADR 0008 (Vanilla LDA),
ADR 0009 (MLlib shim), ADR 0010 (concentration-parameter optimization).

## Context

The OnlineHDP port of Wang/Paisley/Blei 2011 to the spark_vi framework
involves several scope decisions that don't follow obviously from the
algorithmic reference. Recording them here so future revisits know what
was deliberate vs. accidental.

## Decisions

### Skip the lazy-lambda sparse-vocabulary update in v1

Wang's reference uses per-word timestamps + a running cumulative log-discount
to amortize natural-gradient (1−ρ) shrinkage across vocabulary words touched
by a minibatch. At our clinical scale (V ≈ 5-10k concept_ids, much smaller
than NLP V≈100k) full-V digamma per minibatch is cheap, and skipping the
lazy update simplifies the distributed E-step considerably. Revisit only
if profiling shows the full-V digamma is the bottleneck.

### Hold γ and α fixed at user-set values

HDP concentration parameters are part of the model's core appeal — γ
controls how many topics get discovered, α controls how many topics each
doc uses. Optimizing them is a real win, especially for γ. But the math
is its own piece of work, and ADR 0010 already templates the Newton
machinery on LDA. Punt to a follow-on ADR + spec pair after v1 lands.

### No in-loop optimal_ordering

Wang's reference re-sorts topics by descending λ row-sum after every
M-step. Useful for visualization, breaks reproducible topic indices
across runs, and not required by the algorithm. VanillaLDA does not do
this either. If we want it later, it should be a post-fit
`reorder_by_usage()` helper, not an in-loop side-effect.

### Real frozen-globals HDP doc-CAVI for `transform()`, not LDA-collapse

Wang's reference exposes `infer_only` which collapses the trained HDP into
an LDA-equivalent (computing α from sticks, treating λ as the LDA topic-word
matrix) and runs ordinary LDA E-step. We deliberately implement the full
HDP doc-CAVI for `transform()` instead. Reasons:

1. **Held-out evaluation accuracy.** LDA-collapse loses the doc-stick
   structure (the per-doc π_jt, c_jt latent variables collapse into a flat
   Dirichlet prior). Real HDP CAVI gives the actual variational posterior
   under the real model.
2. **Future patient-train / visit-infer split.** That enhancement
   (TOPIC_STATE_MODELING.md L507-523) requires real frozen-globals HDP
   inference at visit granularity; the LDA collapse can't represent it.
3. **Predictive modeling and on-device serving** (eventual goal) need the
   full posterior, not a derivative.

We don't need LDA-shaped HDP outputs, so the LDA-collapse helper is
dropped from scope entirely (not even deferred).

### Keep the iter < 3 warmup trick as default

Wang's reference drops the prior-correction terms (E[log β], E[log π]) in
the var_phi / phi updates for the first three iterations of doc-CAVI.
This is undocumented in the AISTATS paper but preserved in both Wang's
Python and intel-spark's Scala port. We keep it; v2 will run an ablation
(`warmup_iters=0`) to check whether it earns its keep.

### Match-LDA Gamma init for λ; paper-following init for corpus sticks

`λ` initializes via `Gamma(gamma_shape=100, scale=1/100)`, matching
VanillaLDA. Departs from Wang's reference (`Gamma(1,1) · D · 100 / (T·V) − η`)
which is undocumented and not derived. Match-LDA is the boring validated
choice.

`(u, v)` initializes at the prior mean: `u = 1`, `v = γ`. Departs from
Wang's reference (`v = [T-1, ..., 1]` as "make a uniform at beginning")
which is an undocumented bias toward low topic indices.

### Defer MLlib shim and driver scripts to v2

Following the LDA precedent (ADR 0009 added the shim *after* the model
was built and validated), the `spark_vi.mllib.HDP` shim plus
`analysis/local/` and `analysis/cloud/` driver scripts are out of v1
scope. They land in their own ADR + spec once the inner model is unit-
and integration-tested. The second-data-point hypothesis from ADR 0009
(does the LDA shim shape generalize to HDP?) gets resolved at that
time.

## Consequences

- v1 ships a focused, testable model with a clear scope. No half-baked
  optimization knobs, no MLlib-specific machinery to debug alongside the
  math.
- The framework's second concrete model lands; the LDA → HDP
  generalization can inform v2 framework refactors if needed.
- A clear v2 roadmap exists: shim + drivers + γ/α optimization + lazy
  lambda update + warmup ablation + held-out perplexity track. Each is
  its own ADR + spec.

## References

- [`docs/superpowers/specs/2026-05-06-online-hdp-design.md`](../superpowers/specs/2026-05-06-online-hdp-design.md) — implementation spec.
- Wang, Paisley, Blei (2011), "Online Variational Inference for the
  Hierarchical Dirichlet Process," AISTATS.
```

- [ ] **Step 2: Verify ADR list is consistent**

Command: `ls /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno/docs/decisions/`
Expected: `0011-online-hdp-design.md` is present alongside the prior ADRs.

Update the README in that directory if there's an index. Run:
Command: `cat /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno/docs/decisions/README.md`
Expected output: an index. If `0011` is not listed, append a row in the
same style as the existing entries.

- [ ] **Step 3: Commit**

```bash
git add docs/decisions/0011-online-hdp-design.md docs/decisions/README.md
git commit -m "$(cat <<'EOF'
docs(adr): ADR 0011 — Online HDP v1 design decisions

Records the v1 scope choices (skip lazy update, fixed γ/α, no in-loop
optimal-ordering, real frozen-globals transform, match-LDA λ init,
paper-following stick init, keep iter<3 warmup, defer MLlib shim and
drivers). Cross-references the implementation spec and the prior
related ADRs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final verification

After all 19 tasks land:

- [ ] Run the full unit-test suite: `make test`
  Expected: all tests pass; suite finishes in ~10s.
- [ ] Run the full slow-tier suite: `make test-all`
  Expected: all tests pass.
- [ ] Build artifacts: `make build && make zip`
  Expected: both produce green artifacts.
- [ ] Pre-commit hooks: `pre-commit run --all-files`
  Expected: clean.
- [ ] Sanity-check: `git log --oneline | head -25` — verify the 19 task
  commits are present in order, each with an informative message.
