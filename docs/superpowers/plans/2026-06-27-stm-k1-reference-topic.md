# STM K−1 Reference-Topic Parameterization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `reference_topic` parameterization to `OnlineSTM` that pins one topic's η to 0, removing the softmax translation degeneracy that drives the σ_init collapse/blow-up (insight 0029).

**Architecture:** softmax(η) is invariant to adding a constant to every coordinate, so the per-doc MAP η has a free overall level the data cannot pin — only the weak Σ prior can, which is exactly what lets η drift to the saturation boundary (Σ → 10^10) or, with a tight prior, collapse to near-uniform. We fix topic 0's η ≡ 0 ("the always-on baseline phenotype") and optimize the other topics' η *relative to it*. Topic 0 is always the first background topic (`TopicBlockPartition` lays background first and enforces `background_k >= 1`), so it is in **every** document's allowed set — the only valid anchor for a single global reference when Γ/Σ are shared across docs. Non-gated STM is the degenerate "allowed = all K" case, so the gated formulation is the general one and there is no separate code path.

**Implementation strategy — "clamped-K":** Γ and Σ keep their full-K shape. The reference's η leaves only the *per-doc optimizer* and the *prior-side accumulators* (Γ-regression targets, Σ residual, per-topic doc counts, η KL). Because we simply never accumulate `XtMu`/`residual_diag`/`n_docs_per_topic` for the reference, its Γ-column solve resolves to 0 (A⁻¹·0) and its Σ-entry is skipped by the existing lazy-block rule — so `update_global` needs **no changes at all**. β/λ stay full-K: the reference is a real topic with content (its exp(0)=1 sits in the softmax denominator and it accumulates word stats), only its η has no free prior dimension.

**Tech Stack:** Python, NumPy, SciPy (`scipy.optimize.minimize` L-BFGS-B), pytest. No new dependencies.

## Global Constraints

- **Domain-agnostic:** spark-vi never sees OMOP concept ids/names — integer token ids only. No medical/domain vocabulary in code, tests, or comments.
- **Opt-in, default-off, byte-identical default path:** `reference_topic=False` is the default; with it off, every code path must be numerically identical to the current `stm` branch. The existing STM test suite (85 tests) must stay green unchanged.
- **No LaTeX in prose/docstrings:** use plain text + Unicode Greek (η, ν, θ, β, Σ, Γ, λ, φ). No `$...$` delimiters.
- **Markdown-linkable code refs in any prose** (`[name](path#Lstart-Lend)`).
- **TDD throughout:** failing test first, watch it fail, minimal code, watch it pass, commit.
- **Reference index is topic 0**, justified by the background-first contiguous layout in [partition.py:67-71](spark-vi/spark_vi/models/topic/partition.py#L67-L71) and `background_k >= 1` enforced at [partition.py:23-24](spark-vi/spark_vi/models/topic/partition.py#L23-L24).

---

## File Structure

- `spark-vi/spark_vi/models/topic/stm.py` — the engine. Add `reference_topic` to `OnlineSTM.__init__`, a `_reference_index()` helper, a `reference` kwarg to `_stm_doc_inference`, prior-side exclusion in `local_update`, and a `reference` pass-through in `infer_local`. `update_global` is unchanged.
- `spark-vi/tests/test_stm_reference.py` — NEW. Fast unit tests for the reference inference primitive (Task 1) and the model-level M-step inertness (Task 2).
- `spark-vi/tests/test_stm_integration.py` — extend with the slow end-to-end invariant test (Task 3).
- `analysis/local/stm_ablation.py` — extend with reference rows (Task 3); this is what produces the empirical "does it recover at σ_init=1?" numbers the controller interprets.

---

## Task 1: Reference-pinned per-doc inference

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/stm.py` — `__init__` (around [stm.py:287-343](spark-vi/spark_vi/models/topic/stm.py#L287-L343)), add `_reference_index` after `_effective_partition` ([stm.py:345-350](spark-vi/spark_vi/models/topic/stm.py#L345-L350)), rewrite `_stm_doc_inference` ([stm.py:229-272](spark-vi/spark_vi/models/topic/stm.py#L229-L272)).
- Test: `spark-vi/tests/test_stm_reference.py` (new).

**Interfaces:**
- Consumes: `_stm_neg_log_joint`, `_stm_neg_log_joint_grad`, `_stm_neg_log_joint_hessian`, `_spd_inverse`, `_softmax` (all existing in stm.py); `scipy.optimize.minimize`.
- Produces:
  - `OnlineSTM.__init__(..., reference_topic: bool = False)` — new defaulted kwarg, stored as `self.reference_topic: bool`.
  - `OnlineSTM._reference_index() -> int | None` — returns `0` when `reference_topic` else `None`.
  - `_stm_doc_inference(..., reference: int | None = None)` — when `reference` is a topic id (must be in `allowed`), pins that topic's η to 0 and optimizes the rest of `allowed`. Returns `(eta_hat, nu_d, nit)` where `eta_hat[reference] == 0.0` (finite, not −inf), `nu_d` has the reference row/col exactly 0, and disallowed topics stay at η=−inf / ν=0 as before. With `reference=None` the behavior is unchanged.

- [ ] **Step 1: Write the failing tests**

Create `spark-vi/tests/test_stm_reference.py`:

```python
"""Unit tests for the K-1 reference-topic parameterization (opt-in)."""
from __future__ import annotations

import numpy as np
import pytest

from spark_vi.models.topic.stm import (
    OnlineSTM,
    _stm_doc_inference,
    _softmax,
)


# A tiny deterministic 3-topic / 6-word beta. Three "pure" topics over disjoint
# 2-word blocks. Positivity is all the inference math needs, so we can pass this
# directly as expElogbeta in the primitive-level tests.
_BETA3 = np.array([
    [.45, .45, .02, .02, .03, .03],
    [.02, .02, .45, .45, .03, .03],
    [.03, .03, .02, .02, .45, .45],
])


def test_reference_inference_pins_reference_to_zero():
    """With reference=0, topic 0's eta is exactly 0, it has no variance, and
    theta is still a valid distribution that gives the reference positive mass."""
    eta_hat, nu_d, _ = _stm_doc_inference(
        indices=np.array([2, 3], dtype=np.int32),
        counts=np.array([5.0, 5.0]),
        expElogbeta=_BETA3,
        Gamma=np.zeros((1, 3)),
        Sigma_diag=np.full(3, 5.0),
        x=np.array([1.0]),
        reference=0,
    )
    assert eta_hat[0] == 0.0
    assert nu_d[0, 0] == 0.0
    assert np.all(nu_d[0, :] == 0.0) and np.all(nu_d[:, 0] == 0.0)
    theta = _softmax(eta_hat)
    assert abs(float(theta.sum()) - 1.0) < 1e-12
    assert theta[0] > 0.0


def test_reference_inference_recovers_dominant_topic():
    """A doc made of pure topic-1 words puts most theta mass on topic 1 even
    though topic 0 is pinned as the reference — the free optimization works."""
    eta_hat, _, _ = _stm_doc_inference(
        indices=np.array([2, 3], dtype=np.int32),   # topic-1's words
        counts=np.array([8.0, 8.0]),
        expElogbeta=_BETA3,
        Gamma=np.zeros((1, 3)),
        Sigma_diag=np.full(3, 5.0),
        x=np.array([1.0]),
        reference=0,
    )
    theta = _softmax(eta_hat)
    assert int(np.argmax(theta)) == 1


def test_reference_inference_respects_allowed():
    """reference must sit inside allowed; disallowed topics stay at theta=0 and
    the reference is still pinned to 0."""
    eta_hat, nu_d, _ = _stm_doc_inference(
        indices=np.array([0, 1], dtype=np.int32),
        counts=np.array([4.0, 4.0]),
        expElogbeta=_BETA3,
        Gamma=np.zeros((1, 3)),
        Sigma_diag=np.full(3, 5.0),
        x=np.array([1.0]),
        allowed=np.array([0, 1], dtype=np.int64),   # topic 2 disallowed
        reference=0,
    )
    assert eta_hat[0] == 0.0
    assert eta_hat[2] == -np.inf
    theta = _softmax(eta_hat)
    assert theta[2] == 0.0
    assert abs(float(theta.sum()) - 1.0) < 1e-12


def test_reference_topic_requires_k_at_least_two():
    """reference_topic needs at least one free topic besides the reference."""
    with pytest.raises(ValueError, match="reference_topic requires K >= 2"):
        OnlineSTM(K=1, vocab_size=4, P=1, reference_topic=True)


def test_reference_index_toggles():
    assert OnlineSTM(K=3, vocab_size=4, P=1).  _reference_index() is None
    assert OnlineSTM(K=3, vocab_size=4, P=1, reference_topic=True)._reference_index() == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_stm_reference.py -v`
Expected: FAIL — `_stm_doc_inference()` has no `reference` kwarg (TypeError), and `OnlineSTM` has no `reference_topic` kwarg / `_reference_index`.

- [ ] **Step 3: Add the `reference_topic` constructor kwarg + validation + storage**

In `OnlineSTM.__init__` ([stm.py:287-343](spark-vi/spark_vi/models/topic/stm.py#L287-L343)), add the parameter to the signature (after `topic_blocks=None,`):

```python
        topic_blocks=None,
        reference_topic: bool = False,
    ) -> None:
```

Add validation alongside the existing `topic_blocks` check (after [stm.py:327-329](spark-vi/spark_vi/models/topic/stm.py#L327-L329)):

```python
        if reference_topic and K < 2:
            raise ValueError(
                f"reference_topic requires K >= 2 (need a free topic besides "
                f"the reference), got K={K}")
```

Store it alongside `self.topic_blocks = topic_blocks` ([stm.py:343](spark-vi/spark_vi/models/topic/stm.py#L343)):

```python
        self.topic_blocks = topic_blocks
        self.reference_topic = bool(reference_topic)
```

- [ ] **Step 4: Add the `_reference_index` helper**

Immediately after `_effective_partition` ([stm.py:345-350](spark-vi/spark_vi/models/topic/stm.py#L345-L350)), add:

```python
    def _reference_index(self) -> int | None:
        """Global topic id held at eta=0 when reference_topic is on, else None.

        Topic 0 is always the first background topic (TopicBlockPartition lays
        background out first and enforces background_k >= 1), so it is in EVERY
        document's allowed set — the only place a single global reference can
        live when Gamma/Sigma are shared across docs. softmax(eta) is invariant
        to adding a constant to all coordinates; pinning one to 0 removes that
        translation degeneracy, which otherwise lets eta drift to the
        softmax-saturation boundary and Sigma blow up (insight 0029).
        """
        return 0 if self.reference_topic else None
```

- [ ] **Step 5: Rewrite `_stm_doc_inference` to accept `reference`**

Replace the whole function body ([stm.py:229-272](spark-vi/spark_vi/models/topic/stm.py#L229-L272)) with:

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
    allowed: np.ndarray | None = None,
    reference: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Per-doc Laplace approximation, optionally restricted to an allowed topic
    set and optionally with one topic pinned to eta=0 (the reference).

    When allowed is None, optimizes over all K topics. When allowed is a sorted
    index array, L-BFGS runs only on those topics; disallowed topics are filled
    with eta=-inf (theta exactly 0) and nu_d=0.

    When reference is a topic id (which must be in allowed), that topic's eta is
    held at 0 and only the other allowed topics are optimized. softmax([nu, 0])
    removes the translation degeneracy. Fixing a coordinate to a constant makes
    the reduced gradient/Hessian the corresponding sub-block of the full ones,
    so we reuse the existing grad/Hessian and just delete the reference row/col.
    The reference stays a real topic: its exp(0)=1 is in the softmax denominator
    and it still contributes to the data term. In the returned arrays the
    reference has eta_hat=0 (finite) and nu_d row/col exactly 0 (it is pinned, so
    it carries no posterior variance).
    """
    K = expElogbeta.shape[0]
    if allowed is None:
        allowed = np.arange(K, dtype=np.int64)
    sub_expElogbeta = expElogbeta[allowed]
    sub_Gamma = Gamma[:, allowed]
    sub_Sigma = Sigma_diag[allowed]
    n_sub = allowed.shape[0]
    common = dict(
        indices=indices, counts=counts, expElogbeta=sub_expElogbeta,
        Gamma=sub_Gamma, Sigma_diag=sub_Sigma, x=x,
    )

    if reference is None:
        # Canonical path — unchanged from before.
        eta0 = np.zeros(n_sub, dtype=np.float64)
        f = partial(_stm_neg_log_joint, **common)
        g = partial(_stm_neg_log_joint_grad, **common)
        result = minimize(f, x0=eta0, jac=g, method="L-BFGS-B",
                          options={"maxiter": max_iter, "gtol": tol})
        sub_eta = result.x
        H = _stm_neg_log_joint_hessian(sub_eta, **common)
        sub_nu = _spd_inverse(H)
        eta_hat = np.full(K, -np.inf, dtype=np.float64)
        eta_hat[allowed] = sub_eta
        nu_d = np.zeros((K, K), dtype=np.float64)
        nu_d[np.ix_(allowed, allowed)] = sub_nu
        return eta_hat, nu_d, int(result.nit)

    # Reference parameterization: pin `reference` at eta=0, optimize the rest.
    ref_pos = int(np.searchsorted(allowed, reference))
    free = np.array([i for i in range(n_sub) if i != ref_pos], dtype=np.int64)

    def _full(nu_free: np.ndarray) -> np.ndarray:
        eta_sub = np.zeros(n_sub, dtype=np.float64)   # reference position stays 0
        eta_sub[free] = nu_free
        return eta_sub

    def f_free(nu_free: np.ndarray) -> float:
        return _stm_neg_log_joint(_full(nu_free), **common)

    def g_free(nu_free: np.ndarray) -> np.ndarray:
        return _stm_neg_log_joint_grad(_full(nu_free), **common)[free]

    result = minimize(f_free, x0=np.zeros(free.shape[0], dtype=np.float64),
                      jac=g_free, method="L-BFGS-B",
                      options={"maxiter": max_iter, "gtol": tol})
    eta_sub = _full(result.x)
    H_full = _stm_neg_log_joint_hessian(eta_sub, **common)   # (n_sub, n_sub)
    H_free = H_full[np.ix_(free, free)]
    sub_nu_free = _spd_inverse(H_free)

    eta_hat = np.full(K, -np.inf, dtype=np.float64)
    eta_hat[allowed] = eta_sub                  # reference -> 0, free -> nu
    nu_d = np.zeros((K, K), dtype=np.float64)
    free_topics = allowed[free]
    nu_d[np.ix_(free_topics, free_topics)] = sub_nu_free
    return eta_hat, nu_d, int(result.nit)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_stm_reference.py -v`
Expected: PASS (5 tests).

- [ ] **Step 7: Confirm the default path is untouched**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_stm_integration.py -v`
Expected: PASS — the `reference=None` branch is byte-identical to the prior body, so all existing STM tests stay green.

- [ ] **Step 8: Commit**

```bash
git add spark-vi/spark_vi/models/topic/stm.py spark-vi/tests/test_stm_reference.py
git commit -m "feat(stm): reference-pinned per-doc inference (K-1 reference, opt-in)"
```

---

## Task 2: Wire the reference into the M-step (prior-side exclusion)

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/stm.py` — `local_update` accumulation block ([stm.py:455-473](spark-vi/spark_vi/models/topic/stm.py#L455-L473)) and the inference call ([stm.py:435-441](spark-vi/spark_vi/models/topic/stm.py#L435-L441)); `infer_local` ([stm.py:608-613](spark-vi/spark_vi/models/topic/stm.py#L608-L613)).
- Test: `spark-vi/tests/test_stm_reference.py` (extend).

**Interfaces:**
- Consumes: `OnlineSTM._reference_index()` and `_stm_doc_inference(..., reference=...)` from Task 1.
- Produces: after a fit with `reference_topic=True`, the reference topic (index 0) has `Gamma[:, 0]` all 0 and `Sigma[0] == sigma_init` (inert), while it still accumulates β content (`lambda[0]` row mass > 0). The ELBO stays finite. `update_global` is unchanged. The `reference_topic=False` path is byte-identical (`allowed_free == allowed`).

**Why `update_global` needs no change (read before implementing):** the reference is dropped only from `XtMu`, `residual_diag_stat`, and `n_docs_per_topic` in `local_update`. Then in `update_global`: the background Γ solve `Gamma_target[:, bg] = solve(XtX + ridge, XtMu[:, bg])` ([stm.py:540](spark-vi/spark_vi/models/topic/stm.py#L540)) computes column 0 as A⁻¹ · XtMu[:, 0] = A⁻¹ · 0 = 0, so `Gamma[:, 0]` stays 0; and `present = n_docs_per_topic > 0` ([stm.py:551](spark-vi/spark_vi/models/topic/stm.py#L551)) is False for topic 0, so the lazy rule leaves `Sigma[0]` at its initial `sigma_init`.

- [ ] **Step 1: Write the failing tests** (append to `spark-vi/tests/test_stm_reference.py`)

```python
from spark_vi.models.topic.types import STMDocument
from spark_vi.models.topic.partition import TopicBlockPartition


def _toy_docs(rng, *, V, D, doc_len, K_blocks):
    """D docs over V words; each doc concentrates on one of K_blocks word
    blocks plus light noise. Integer token ids only (domain-agnostic)."""
    block = V // K_blocks
    docs = []
    for _ in range(D):
        b = int(rng.integers(K_blocks))
        toks = np.concatenate([
            rng.integers(b * block, (b + 1) * block, size=doc_len - 3),
            rng.integers(0, V, size=3),
        ])
        u, c = np.unique(toks, return_counts=True)
        docs.append(STMDocument(indices=u.astype(np.int32),
                                counts=c.astype(np.float64),
                                length=int(c.sum()), x=np.array([1.0])))
    return docs


def test_reference_gamma_column_zero_and_sigma_inert():
    """After full-batch updates with reference_topic, the reference topic's
    Gamma column stays 0 and its Sigma entry stays at sigma_init."""
    rng = np.random.default_rng(0)
    V, K = 30, 4
    docs = _toy_docs(rng, V=V, D=60, doc_len=20, K_blocks=K)
    m = OnlineSTM(K=K, vocab_size=V, P=1, sigma_init=3.0,
                  random_seed=1, reference_topic=True)
    gp = m.initialize_global(None)
    for _ in range(8):
        gp = m.update_global(gp, m.local_update(docs, gp), learning_rate=1.0)
    assert np.allclose(gp["Gamma"][:, 0], 0.0)
    assert gp["Sigma"][0] == pytest.approx(3.0)


def test_reference_topic_still_learns_content():
    """The reference is a real topic: its lambda row carries vocabulary mass."""
    rng = np.random.default_rng(0)
    V, K = 30, 4
    docs = _toy_docs(rng, V=V, D=60, doc_len=20, K_blocks=K)
    m = OnlineSTM(K=K, vocab_size=V, P=1, random_seed=1, reference_topic=True)
    gp = m.initialize_global(None)
    for _ in range(8):
        gp = m.update_global(gp, m.local_update(docs, gp), learning_rate=1.0)
    assert gp["lambda"][0].sum() > V  # row mass above the eta=1/K prior floor


def test_reference_elbo_finite():
    """KL over the free subspace stays finite (the reference row/col of nu_d is
    excluded, so the sub-covariance is non-singular)."""
    rng = np.random.default_rng(0)
    V, K = 30, 4
    docs = _toy_docs(rng, V=V, D=60, doc_len=20, K_blocks=K)
    m = OnlineSTM(K=K, vocab_size=V, P=1, random_seed=1, reference_topic=True)
    gp = m.initialize_global(None)
    for _ in range(5):
        stats = m.local_update(docs, gp)
        gp = m.update_global(gp, stats, learning_rate=1.0)
    elbo = m.compute_elbo(gp, m.local_update(docs, gp))
    assert np.isfinite(elbo)


def test_reference_off_does_not_perturb_default():
    """Passing reference_topic=False (the default) is identical to not passing
    it — the kwarg's presence must not change the canonical fit."""
    rng = np.random.default_rng(0)
    V, K = 30, 4
    docs = _toy_docs(rng, V=V, D=40, doc_len=20, K_blocks=K)
    a = OnlineSTM(K=K, vocab_size=V, P=1, random_seed=7)
    b = OnlineSTM(K=K, vocab_size=V, P=1, random_seed=7, reference_topic=False)
    gpa, gpb = a.initialize_global(None), b.initialize_global(None)
    for _ in range(5):
        gpa = a.update_global(gpa, a.local_update(docs, gpa), learning_rate=1.0)
        gpb = b.update_global(gpb, b.local_update(docs, gpb), learning_rate=1.0)
    assert np.array_equal(gpa["Gamma"], gpb["Gamma"])
    assert np.array_equal(gpa["Sigma"], gpb["Sigma"])
    assert np.array_equal(gpa["lambda"], gpb["lambda"])


def test_reference_gated_infer_local_pins_reference():
    """infer_local must use the same parameterization as training, so exported
    theta has the reference pinned to eta=0."""
    rng = np.random.default_rng(0)
    V = 24
    part = TopicBlockPartition(group_var="g", background_k=2,
                               foreground=(("rare", 2),))
    K = part.K
    docs = []
    for i in range(80):
        is_rare = (i % 4 == 0)
        toks = rng.integers(0, V, size=12)
        docs.append(STMDocument(indices=np.unique(toks).astype(np.int32),
                                counts=np.ones(len(np.unique(toks))),
                                length=len(np.unique(toks)),
                                x=np.array([1.0]),
                                groups=frozenset({"rare"}) if is_rare else frozenset()))
    m = OnlineSTM(K=K, vocab_size=V, P=1, random_seed=3,
                  topic_blocks=part, reference_topic=True)
    gp = m.initialize_global(None)
    for _ in range(10):
        gp = m.update_global(gp, m.local_update(docs, gp), learning_rate=0.5)
    out = m.infer_local(docs[0], gp)
    assert out["eta"][0] == 0.0
    assert abs(float(out["theta"].sum()) - 1.0) < 1e-12
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_stm_reference.py -v -k "gamma_column or learns_content or elbo_finite or infer_local"`
Expected: FAIL — `local_update` does not yet exclude the reference, so `Gamma[:, 0]` becomes nonzero, `Sigma[0]` drifts, and `infer_local` does not pin the reference.

- [ ] **Step 3: Pass `reference` into the per-doc inference call**

In `local_update`, just before the doc loop add the reference id (after [stm.py:431](spark-vi/spark_vi/models/topic/stm.py#L431) `log_Sigma_diag = np.log(Sigma_diag)`):

```python
        log_Sigma_diag = np.log(Sigma_diag)
        ref = self._reference_index()
```

Update the inference call ([stm.py:435-441](spark-vi/spark_vi/models/topic/stm.py#L435-L441)) to pass it:

```python
            allowed = part.allowed_indices(doc.groups)
            eta_hat, nu_d, _ = _stm_doc_inference(
                indices=doc.indices, counts=doc.counts,
                expElogbeta=expElogbeta,
                Gamma=Gamma, Sigma_diag=Sigma_diag, x=doc.x,
                max_iter=self.lbfgs_max_iter, tol=self.lbfgs_tol,
                allowed=allowed, reference=ref,
            )
```

- [ ] **Step 4: Exclude the reference from the prior-side accumulators**

Replace the accumulation block ([stm.py:455-473](spark-vi/spark_vi/models/topic/stm.py#L455-L473)) with:

```python
            # Prior-side topics for this doc. With a reference topic, exclude it:
            # it is pinned at eta=0, carries no free Gamma column or Sigma entry,
            # and must stay out of the Gamma-regression targets, the Sigma
            # residual, the per-topic doc counts, and the eta KL. Dropping it
            # from XtMu makes its Gamma solve resolve to 0; dropping it from
            # n_docs_per_topic makes update_global's lazy rule leave Sigma[ref]
            # at sigma_init. When ref is None this is exactly `allowed`, so the
            # canonical path is byte-identical.
            if ref is None:
                allowed_free = allowed
            else:
                allowed_free = allowed[allowed != ref]

            # XtMu / residual_diag / counts only over the free prior topics.
            eta_allowed = eta_hat[allowed_free]
            XtMu[:, allowed_free] += np.outer(doc.x, eta_allowed)
            resid = np.zeros(K, dtype=np.float64)
            resid[allowed_free] = eta_allowed - (Gamma.T @ doc.x)[allowed_free]
            residual_diag[allowed_free] += resid[allowed_free] ** 2 + np.diag(nu_d)[allowed_free]
            n_docs_per_topic[allowed_free] += 1.0

            doc_loglik += float(np.sum(doc.counts * np.log(q_w)))
            # KL over the free prior sub-space only.
            al = allowed_free
            tr_term = float(np.sum(np.diag(nu_d)[al] / Sigma_diag[al]))
            quad_term = float(np.sum(resid[al] ** 2 / Sigma_diag[al]))
            sub_nu = nu_d[np.ix_(al, al)]
            # sub_nu is SPD by construction (_spd_inverse), so slogdet sign is +1.
            _sign, logdet_nu = np.linalg.slogdet(sub_nu)
            logdet_Sigma = float(np.sum(log_Sigma_diag[al]))
            doc_eta_kl += 0.5 * (tr_term + quad_term - len(al) + logdet_Sigma - logdet_nu)
            n_docs += 1
```

(Note: the `p` / `eb_d` / `q_w` / `phi` / `lambda_stats` lines at [stm.py:442-447](spark-vi/spark_vi/models/topic/stm.py#L442-L447) are unchanged — the data term and β stats use the full `allowed`, including the reference's content.)

- [ ] **Step 5: Pin the reference in `infer_local`**

Update the `_stm_doc_inference` call in `infer_local` ([stm.py:608-613](spark-vi/spark_vi/models/topic/stm.py#L608-L613)):

```python
        eta_hat, _, _ = _stm_doc_inference(
            indices=row.indices, counts=row.counts,
            expElogbeta=expElogbeta,
            Gamma=Gamma, Sigma_diag=Sigma_diag, x=row.x,
            max_iter=self.lbfgs_max_iter, tol=self.lbfgs_tol,
            reference=self._reference_index(),
        )
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_stm_reference.py -v`
Expected: PASS (all Task 1 + Task 2 tests).

- [ ] **Step 7: Confirm the full STM suite is still green**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/ -v -k stm`
Expected: PASS — existing STM tests unchanged (`reference_topic=False` path byte-identical), plus the new reference tests.

- [ ] **Step 8: Commit**

```bash
git add spark-vi/spark_vi/models/topic/stm.py spark-vi/tests/test_stm_reference.py
git commit -m "feat(stm): exclude reference topic from M-step prior accumulators"
```

---

## Task 3: Validation harness + the σ_init=1 recovery experiment

**Files:**
- Modify: `analysis/local/stm_ablation.py` — add reference configs to the non-gated table and a reference comparison to the gated block.
- Test: `spark-vi/tests/test_stm_integration.py` (extend with one slow end-to-end invariant test).

**Interfaces:**
- Consumes: `OnlineSTM(reference_topic=True)` (Tasks 1-2); `fit_stm`, `synthetic_ehr_corpus`, `synthetic_gated_corpus`, `planted_recovery`, `foreground_recovers_group`, `final_sigma_range`, `spectral_init_beta`, `TopicBlockPartition` (all existing). `fit_stm` already forwards `**model_kwargs` to `OnlineSTM` ([_stm_synth.py:101-104](spark-vi/tests/_stm_synth.py#L101-L104)), so `reference_topic=True` flows through with no harness change.
- Produces: an extended `stm_ablation.py` whose output the controller reads to answer "does reference + spectral recover the planted topics at σ_init=1?"; and a deterministic integration test asserting the fitted reference is pinned and θ stays valid end-to-end.

**Fairness note (read before editing the ablation):** with `reference_topic`, one topic is spent as the always-on baseline. So a reference run with the same K has one fewer *free* topic. For an apples-to-apples non-gated comparison, the reference configs fit with `K = K_RARE + 1` (K_RARE free topics + 1 baseline), and `planted_recovery` still counts how many of the 8 planted topics any β row covers. The gated block needs no K change: the reference pins one of the 2 background topics, and `foreground_recovers_group` only inspects foreground rows, which are unaffected.

- [ ] **Step 1: Write the failing integration test** (append to `spark-vi/tests/test_stm_integration.py`)

```python
def test_reference_fit_pins_reference_end_to_end():
    """A fitted reference model, gated and non-gated, keeps the reference topic
    pinned (eta=0) through the export path and yields a valid theta. This is the
    deterministic invariant; the *recovery* payoff is measured by the ablation
    script, not asserted at a guessed threshold."""
    import numpy as np
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from _stm_synth import synthetic_ehr_corpus, synthetic_gated_corpus, fit_stm
    from spark_vi.models.topic.stm import OnlineSTM

    # Non-gated.
    docs, _ = synthetic_ehr_corpus(K_rare=4, V=80, D=200, doc_len=25,
                                   bg_frac=0.6, seed=5)
    m = OnlineSTM(K=5, vocab_size=80, P=1, sigma_init=1.0,
                  random_seed=42, reference_topic=True)
    gp = m.initialize_global(None)
    for _ in range(30):
        gp = m.update_global(gp, m.local_update(docs, gp), learning_rate=1.0)
    out = m.infer_local(docs[0], gp)
    assert out["eta"][0] == 0.0
    assert abs(float(out["theta"].sum()) - 1.0) < 1e-12
    assert np.allclose(gp["Gamma"][:, 0], 0.0)

    # Gated.
    docs_g, _, part = synthetic_gated_corpus(groups=("maj", "rare"),
                                             fg_per_group=2, bg_k=2, V=120,
                                             D=200, doc_len=30, bg_frac=0.5,
                                             seed=8)
    mg = OnlineSTM(K=part.K, vocab_size=120, P=1, sigma_init=1.0,
                   random_seed=42, topic_blocks=part, reference_topic=True)
    gpg = mg.initialize_global(None)
    for _ in range(30):
        gpg = mg.update_global(gpg, mg.local_update(docs_g, gpg), learning_rate=0.5)
    og = mg.infer_local(docs_g[0], gpg)
    assert og["eta"][0] == 0.0
    assert np.allclose(gpg["Gamma"][:, 0], 0.0)
```

- [ ] **Step 2: Run it to verify it passes** (it exercises only Task 1-2 code; it should already pass — it is a regression guard, not a red-first test)

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_stm_integration.py::test_reference_fit_pins_reference_end_to_end -v`
Expected: PASS. (If it fails, Tasks 1-2 are incomplete — fix there before continuing.)

- [ ] **Step 3: Add reference configs to the non-gated ablation table**

In `analysis/local/stm_ablation.py`, replace the `configs` list ([stm_ablation.py:62-67](analysis/local/stm_ablation.py#L62-L67)) with:

```python
configs = [
    ("random-init (baseline)",          dict()),
    ("+Sigma-prior",                     dict(sigma_prior_scale=2.0, sigma_prior_count=500.0)),
    ("+spectral",                        dict(_spectral=True)),
    ("+spectral+Sigma-prior",            dict(_spectral=True, sigma_prior_scale=2.0, sigma_prior_count=500.0)),
    ("+reference",                       dict(_reference=True)),
    ("+spectral+reference",              dict(_spectral=True, _reference=True)),
    ("+spectral+reference+Sigma-prior",  dict(_spectral=True, _reference=True, sigma_prior_scale=2.0, sigma_prior_count=500.0)),
]
```

Replace the table loop ([stm_ablation.py:74-87](analysis/local/stm_ablation.py#L74-L87)) with one that honors `_reference` (extra topic for fairness, separate spectral β at K+1):

```python
# Reference runs get one extra topic (the pinned baseline) so they keep K_RARE
# free topics; precompute a matching spectral beta0 at K_RARE+1.
beta0_ref = spectral_init_beta(
    docs, TopicBlockPartition(group_var="", background_k=K_RARE + 1, foreground=()), V)

for config_name, kwargs in configs:
    use_spectral = kwargs.pop("_spectral", False)
    use_reference = kwargs.pop("_reference", False)
    k_fit = K_RARE + 1 if use_reference else K_RARE
    b0 = beta0_ref if use_reference else beta0
    row = f"{config_name:<32}"
    for si in SIGMA_INITS:
        model_kwargs = dict(kwargs)
        if use_reference:
            model_kwargs["reference_topic"] = True
        init_data = {"spectral_beta": b0} if use_spectral else None
        gp = fit_stm(docs, K=k_fit, V=V, sigma_init=si,
                     n_iter=N_ITER, batch=BATCH, seed=42,
                     init_data=init_data, **model_kwargs)
        beta_hat = gp["lambda"] / gp["lambda"].sum(axis=1, keepdims=True)
        rec = planted_recovery(beta_hat, planted, thresh=0.5)
        _, s_max = final_sigma_range(gp)
        row += f"  ({rec}/{K_RARE}, {s_max:.2e})"
    print(row)
```

- [ ] **Step 4: Add reference rows to the gated block**

After the existing block-aware spectral gated run ([stm_ablation.py:144-151](analysis/local/stm_ablation.py#L144-L151)), append:

```python
gp_ref = fit_stm(docs_gated, K=part.K, V=G_V, sigma_init=1,
                 n_iter=G_N_ITER, batch=G_BATCH, seed=42,
                 partition=part, reference_topic=True)
beta_ref = gp_ref["lambda"] / gp_ref["lambda"].sum(axis=1, keepdims=True)
rec_ref = foreground_recovers_group(beta_ref, part, "rare", planted_g, thresh=0.5)
_, smax_ref = final_sigma_range(gp_ref)
print(f"  reference (sigma_init=1):            recovers rare = {rec_ref},  Sigma_max = {smax_ref:.2e}")

gp_sr = fit_stm(docs_gated, K=part.K, V=G_V, sigma_init=1,
                n_iter=G_N_ITER, batch=G_BATCH, seed=42,
                partition=part, reference_topic=True,
                init_data={"spectral_beta": beta0_gated})
beta_sr = gp_sr["lambda"] / gp_sr["lambda"].sum(axis=1, keepdims=True)
rec_sr = foreground_recovers_group(beta_sr, part, "rare", planted_g, thresh=0.5)
_, smax_sr = final_sigma_range(gp_sr)
print(f"  spectral+reference (sigma_init=1):   recovers rare = {rec_sr},  Sigma_max = {smax_sr:.2e}")
```

- [ ] **Step 5: Run the ablation end-to-end (smoke + produce the numbers)**

Run: `.venv/bin/python analysis/local/stm_ablation.py`
Expected: the script completes without error and prints the non-gated table (now with three reference rows) and the gated block (now with reference + spectral+reference at σ_init=1). The numbers are the experiment output — the controller reads and interprets them (they are not asserted in code).

- [ ] **Step 6: Run the full STM suite once more**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/ -v -k stm`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add analysis/local/stm_ablation.py spark-vi/tests/test_stm_integration.py
git commit -m "test(stm): reference-topic ablation rows + end-to-end pin invariant"
```

---

## Controller post-implementation (not a subagent task)

After Task 3, the controller (not a subagent) does the interpretive work, because it requires reading the ablation numbers and judging the modeling outcome:

1. **Read the ablation output** and answer the load-bearing question: does `spectral+reference` recover the planted topics at σ_init=1 where spectral-alone gave 0/8 (non-gated) and does it recover the rare foreground (gated)? Report the verdict to the user with the math taught (the reparam of the Laplace Hessian — this is the standing learning goal).
2. **Update `docs/insights/0029-stm-sigma-init-collapse-blowup-missing-stabilizers.md`** with the K−1 result: whether the third missing stabilizer closes the σ_init=1 gap, or whether reference helps the background level but foreground recovery still needs spectral + the Σ prior.
3. **Write an ADR** (next number is 0030) for the opt-in K−1 reference parameterization: the identifiability rationale, the clamped-K implementation choice (vs true K−1 re-indexing), and the "topic 0 = global background reference" decision. Impersonal, no LaTeX.
4. **Decide the two deferred spectral-init minors** (`find_anchors` O(nV²), `min_marginal_frac=1.0`) — still gated on whether spectral init will run on real V≈3691 data, unchanged by this work.

---

## Self-Review

**Spec coverage:**
- Reference parameterization (pin η=0, optimize rest) → Task 1.
- M-step integration with clamped-K and no `update_global` change → Task 2 (with the inertness rationale).
- Gated-as-general (topic 0 reference valid in every allowed set) → enforced via `_reference_index()` returning 0; non-gated is the `allowed = all K` case; covered by both gated and non-gated tests.
- Default-off byte-identical → `test_reference_off_does_not_perturb_default` + existing suite green (Tasks 1, 2 steps 7).
- σ_init=1 recovery experiment (the load-bearing question) → Task 3 ablation + controller interpretation.
- Export consistency (`infer_local`) → Task 2 step 5 + `test_reference_gated_infer_local_pins_reference`.

**Placeholder scan:** every code step shows complete code; no TBD/handle-edge-cases. The recovery thresholds are deliberately NOT asserted (they are the experiment); the deterministic invariants are asserted instead.

**Type consistency:** `_reference_index() -> int | None`; `_stm_doc_inference(..., reference: int | None = None)`; `reference_topic: bool`. `allowed_free` is `np.ndarray[int64]`. `eta_hat[reference] == 0.0` (finite), `nu_d` reference row/col `0.0`. All consistent across tasks.
