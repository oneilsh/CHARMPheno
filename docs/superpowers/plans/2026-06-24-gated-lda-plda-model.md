# Gated LDA (PLDA) — Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in background/foreground topic-block gating to `OnlineLDA` (the spark-vi engine), implementing Partially Labeled Dirichlet Allocation: each document may express a shared background block plus only its own group's foreground block, via hard topic-masking.

**Architecture:** Mirror the shipped `OnlineSTM` gating exactly — reuse `TopicBlockPartition`/`allowed_indices`, add a `groups` field to `BOWDocument`, add an `allowed=` mask to the per-doc CAVI E-step, and a `topic_blocks` constructor arg + masked `local_update` to `OnlineLDA`. The non-gated path stays byte-identical to current LDA (gating off ⇒ `allowed=None` ⇒ original code path).

**Tech Stack:** Python, numpy, scipy.special (digamma); pytest (pure in-process, no Spark). Design spec: `docs/superpowers/specs/2026-06-24-gated-lda-plda-design.md`. Motivation: insight 0028.

## Global Constraints

- **Non-gated byte-identity:** `OnlineLDA(topic_blocks=None)` and an explicit all-background partition (`TopicBlockPartition(background_k=K, foreground=())`) must produce output identical to current `OnlineLDA`. Enforced by routing both to the unchanged `allowed=None` code path.
- **Engine stays domain-agnostic:** the engine knows only "groups" (frozensets of strings); no clinical/`source_cohort` naming in spark-vi.
- **v1 scope = masking + fixed α.** Learned *asymmetric*-α optimization under masking requires a per-group-simplex Newton step (the closed form in `alpha_newton_step` assumes one shared K-simplex; under masking the ψ(Σα) normalizer varies by a doc's allowed set). v1 therefore **rejects `optimize_alpha=True` together with foreground blocks** (raises). Fixed α < 1 still supplies the document-topic sparsity that recovers rare phenotypes (insight 0028). The deferred v2 derivation is in the Appendix.
- **Mirror, don't invent:** follow `OnlineSTM`'s `_effective_partition`, `_stm_doc_inference(..., allowed=)`, and per-allowed-subspace KL patterns verbatim where applicable.

---

### Task 1: `BOWDocument` gains an optional `groups` field

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/types.py` (the `BOWDocument` dataclass + its `from_spark_row`)
- Test: `spark-vi/tests/test_lda_gating.py` (create)

**Interfaces:**
- Produces: `BOWDocument(indices, counts, length, groups=frozenset())`; `BOWDocument.from_spark_row(row, features_col="features", group_col=None)` populating `groups` from `row[group_col]` (a single label) when `group_col` is given.

- [ ] **Step 1: Write the failing test**

```python
# spark-vi/tests/test_lda_gating.py
import numpy as np
from pyspark.ml.linalg import SparseVector
from spark_vi.models.topic.types import BOWDocument


def test_bowdocument_groups_defaults_empty():
    d = BOWDocument(indices=np.array([0, 1], dtype=np.int32),
                    counts=np.array([1.0, 2.0]), length=3)
    assert d.groups == frozenset()


def test_bowdocument_from_spark_row_reads_group_col():
    row = {"features": SparseVector(3, [0, 2], [2.0, 1.0]), "g": "dementia"}
    d = BOWDocument.from_spark_row(row, group_col="g")
    assert d.groups == frozenset({"dementia"})
    # No group_col -> empty (back-compat).
    d0 = BOWDocument.from_spark_row({"features": SparseVector(3, [0], [1.0])})
    assert d0.groups == frozenset()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd spark-vi && python -m pytest tests/test_lda_gating.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'groups'` is not raised (groups is positional-less), actually FAIL on `from_spark_row` not accepting `group_col`.

- [ ] **Step 3: Add the field + group_col (mirror STMDocument)**

In `types.py`, the `BOWDocument` dataclass — add the field after `length` and extend `from_spark_row`:

```python
@dataclass(frozen=True, slots=True)
class BOWDocument:
    # ... existing docstring; add to invariants:
    #   groups: frozenset[str] of group labels (empty = background only / gating off).
    indices: np.ndarray
    counts: np.ndarray
    length: int
    groups: frozenset = frozenset()

    @classmethod
    def from_spark_row(cls, row, features_col: str = "features",
                       group_col: str | None = None) -> "BOWDocument":
        sv = row[features_col]
        groups = frozenset()
        if group_col is not None and row[group_col] is not None:
            groups = frozenset({str(row[group_col])})
        return cls(
            indices=np.asarray(sv.indices, dtype=np.int32),
            counts=np.asarray(sv.values, dtype=np.float64),
            length=int(sv.values.sum()),
            groups=groups,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd spark-vi && python -m pytest tests/test_lda_gating.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/types.py spark-vi/tests/test_lda_gating.py
git commit -m "feat(spark-vi): BOWDocument gains optional groups field for gated LDA"
```

---

### Task 2: `_cavi_doc_inference` gains an `allowed=` topic mask

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/lda.py` (`_cavi_doc_inference`)
- Test: `spark-vi/tests/test_lda_gating.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `_cavi_doc_inference(indices, counts, expElogbeta, alpha, gamma_init, max_iter, tol, allowed=None)`. `allowed=None` → unchanged full-K path. Otherwise runs CAVI on the `allowed` sub-simplex and scatters back full-K arrays with **0 on disallowed rows** for both `gamma` and `expElogthetad` (so β suff-stats and KL exclude disallowed topics). `phi_norm` is computed over the allowed topics only.

- [ ] **Step 1: Write the failing test**

```python
def test_cavi_allowed_zeros_disallowed_and_matches_full_when_none():
    from spark_vi.models.topic.lda import _cavi_doc_inference
    from scipy.special import digamma
    rng = np.random.default_rng(0)
    K, V = 6, 10
    lam = rng.gamma(2.0, 1.0, size=(K, V))
    expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))
    indices = np.array([1, 4, 7], dtype=np.int32)
    counts = np.array([2.0, 1.0, 3.0])
    alpha = np.full(K, 0.1)
    gamma_init = np.full(K, 1.0)

    allowed = np.array([0, 2, 5], dtype=np.int64)
    gamma, elt, phi_norm, _ = _cavi_doc_inference(
        indices, counts, expElogbeta, alpha, gamma_init, max_iter=50, tol=1e-4,
        allowed=allowed)
    disallowed = np.array([1, 3, 4])
    assert np.allclose(gamma[disallowed], 0.0)
    assert np.allclose(elt[disallowed], 0.0)
    assert (gamma[allowed] > 0).all()

    # allowed=None reproduces the original full-K result exactly.
    g_full, e_full, p_full, _ = _cavi_doc_inference(
        indices, counts, expElogbeta, alpha, gamma_init, max_iter=50, tol=1e-4)
    g_none, e_none, p_none, _ = _cavi_doc_inference(
        indices, counts, expElogbeta, alpha, gamma_init, max_iter=50, tol=1e-4,
        allowed=None)
    assert np.array_equal(g_full, g_none) and np.array_equal(e_full, e_none)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd spark-vi && python -m pytest tests/test_lda_gating.py::test_cavi_allowed_zeros_disallowed_and_matches_full_when_none -v`
Expected: FAIL — `TypeError: _cavi_doc_inference() got an unexpected keyword argument 'allowed'`

- [ ] **Step 3: Add the `allowed` branch**

In `lda.py`, change the signature to add `allowed: np.ndarray | None = None`, keep the existing body as the `allowed is None` fast path (so non-gated is byte-identical), and add the masked branch. Concretely, wrap the existing loop:

```python
def _cavi_doc_inference(
    indices, counts, expElogbeta, alpha, gamma_init, max_iter, tol,
    allowed=None,
):
    """... (existing docstring); when `allowed` is given, optimize theta_d over
    only those topic rows (PLDA hard masking) and return full-K gamma/
    expElogthetad with 0 on disallowed rows."""
    if allowed is None:
        # --- unchanged original full-K path ---
        eb_d = expElogbeta[:, indices]
        gamma = gamma_init.astype(np.float64, copy=True)
        expElogthetad = np.exp(digamma(gamma) - digamma(gamma.sum()))
        phi_norm = eb_d.T @ expElogthetad + 1e-100
        n_iter = 0
        for it in range(1, max_iter + 1):
            n_iter = it
            prev = gamma.copy()
            gamma = alpha + expElogthetad * (eb_d @ (counts / phi_norm))
            expElogthetad = np.exp(digamma(gamma) - digamma(gamma.sum()))
            phi_norm = eb_d.T @ expElogthetad + 1e-100
            if np.mean(np.abs(gamma - prev)) < tol:
                break
        return gamma, expElogthetad, phi_norm, n_iter

    # --- masked sub-simplex path (theta_d over `allowed` topics only) ---
    K = expElogbeta.shape[0]
    eb_d = expElogbeta[np.ix_(allowed, indices)]          # (|A|, n_unique)
    sub_alpha = np.asarray(alpha)[allowed]
    sub_gamma = np.asarray(gamma_init, dtype=np.float64)[allowed].copy()
    sub_elt = np.exp(digamma(sub_gamma) - digamma(sub_gamma.sum()))
    phi_norm = eb_d.T @ sub_elt + 1e-100
    n_iter = 0
    for it in range(1, max_iter + 1):
        n_iter = it
        prev = sub_gamma.copy()
        sub_gamma = sub_alpha + sub_elt * (eb_d @ (counts / phi_norm))
        sub_elt = np.exp(digamma(sub_gamma) - digamma(sub_gamma.sum()))
        phi_norm = eb_d.T @ sub_elt + 1e-100
        if np.mean(np.abs(sub_gamma - prev)) < tol:
            break
    gamma = np.zeros(K, dtype=np.float64)
    expElogthetad = np.zeros(K, dtype=np.float64)
    gamma[allowed] = sub_gamma
    expElogthetad[allowed] = sub_elt
    return gamma, expElogthetad, phi_norm, n_iter
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd spark-vi && python -m pytest tests/test_lda_gating.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/lda.py spark-vi/tests/test_lda_gating.py
git commit -m "feat(spark-vi): _cavi_doc_inference supports allowed-topic masking (PLDA)"
```

---

### Task 3: `OnlineLDA` gains `topic_blocks` + masked `local_update` + α guard

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/lda.py` (`OnlineLDA.__init__`, add `_effective_partition`, `local_update`)
- Test: `spark-vi/tests/test_lda_gating.py`

**Interfaces:**
- Consumes: `_cavi_doc_inference(..., allowed=)` (Task 2), `BOWDocument.groups` (Task 1), `TopicBlockPartition` (existing) — `.allowed_indices(groups)`, `.foreground`.
- Produces: `OnlineLDA(..., topic_blocks=None)`. When `topic_blocks` has foreground blocks, `local_update` masks each doc to `partition.allowed_indices(doc.groups)`; KL is taken over the allowed sub-space. Constructing with foreground blocks AND `optimize_alpha=True` raises `ValueError` (v1 limitation). `_effective_partition()` returns an implicit all-background partition when `topic_blocks is None`.

- [ ] **Step 1: Write the failing tests**

```python
from spark_vi.models.topic.lda import OnlineLDA
from spark_vi.models.topic.partition import TopicBlockPartition
from spark_vi.models.topic.types import BOWDocument


def _fit(model, docs, n_iter=40, lr=0.5):
    gp = model.initialize_global(None)
    for _ in range(n_iter):
        gp = model.update_global(gp, model.local_update(docs, gp), learning_rate=lr)
    return gp


def test_all_background_partition_equals_plain_lda():
    rng = np.random.default_rng(1)
    V = 12
    docs = [BOWDocument(indices=np.unique(rng.integers(0, V, 5).astype(np.int32)),
                        counts=np.ones(1), length=1) for _ in range(60)]
    docs = [BOWDocument(d.indices, np.ones(len(d.indices)), int(len(d.indices)))
            for d in docs]
    plain = _fit(OnlineLDA(K=4, vocab_size=V, random_seed=2), docs)
    allbg = _fit(OnlineLDA(K=4, vocab_size=V, random_seed=2,
                           topic_blocks=TopicBlockPartition("g", 4, ())), docs)
    assert np.allclose(plain["lambda"], allbg["lambda"])


def test_optimize_alpha_with_foreground_raises():
    import pytest
    part = TopicBlockPartition("g", background_k=3, foreground=(("rare", 2),))
    with pytest.raises(ValueError, match="optimize_alpha"):
        OnlineLDA(K=part.K, vocab_size=10, optimize_alpha=True, topic_blocks=part)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd spark-vi && python -m pytest tests/test_lda_gating.py -k "background_partition or foreground_raises" -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'topic_blocks'`

- [ ] **Step 3: Implement constructor, guard, `_effective_partition`, masked `local_update`**

In `OnlineLDA.__init__`, add the param (last, default None) and the guard at the end of `__init__`:

```python
        # ... existing assignments ...
        self.topic_blocks = topic_blocks
        if (topic_blocks is not None and topic_blocks.foreground
                and self.optimize_alpha):
            raise ValueError(
                "optimize_alpha is not supported with foreground topic blocks "
                "(v1): the asymmetric-alpha Newton step assumes one shared "
                "K-simplex, but PLDA masking varies the simplex per document. "
                "Use a fixed alpha (alpha<1 still gives document sparsity); see "
                "docs/superpowers/plans/2026-06-24-gated-lda-plda-model.md Appendix."
            )
```

Add the helper (copy of STM's):

```python
    def _effective_partition(self):
        """The real partition, or an implicit all-background one when None."""
        if self.topic_blocks is not None:
            return self.topic_blocks
        from spark_vi.models.topic.partition import TopicBlockPartition
        return TopicBlockPartition(group_var="", background_k=self.K, foreground=())
```

In `local_update`, compute `gating_on` once, derive `allowed` per doc, thread it into `_cavi_doc_inference`, and take KL over the allowed sub-space. The masked `expElogthetad` is already 0 on disallowed rows, so `sstats_row = np.outer(expElogthetad, counts/phi_norm)` masks the β suff-stats automatically. Modify the loop:

```python
        part = self._effective_partition()
        gating_on = bool(part.foreground)
        # ... existing setup (lambda_stats, expElogbeta, accumulators) ...
        for doc in rows:
            allowed = part.allowed_indices(doc.groups) if gating_on else None
            # ... existing gamma_init draw (unchanged) ...
            gamma, expElogthetad, phi_norm, _ = _cavi_doc_inference(
                indices=doc.indices, counts=doc.counts, expElogbeta=expElogbeta,
                alpha=alpha, gamma_init=gamma_init,
                max_iter=self.cavi_max_iter, tol=self.cavi_tol,
                allowed=allowed,
            )
            sstats_row = np.outer(expElogthetad, doc.counts / phi_norm)
            lambda_stats[:, doc.indices] += sstats_row
            doc_loglik_sum += float(np.sum(doc.counts * np.log(phi_norm)))
            if allowed is None:
                doc_theta_kl_sum += _dirichlet_kl(gamma, alpha)
            else:
                doc_theta_kl_sum += _dirichlet_kl(gamma[allowed], alpha[allowed])
            n_docs += 1
            if self.optimize_alpha:   # gating_on is False here (guarded in __init__)
                e_log_theta_sum += digamma(gamma) - digamma(gamma.sum())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd spark-vi && python -m pytest tests/test_lda_gating.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/lda.py spark-vi/tests/test_lda_gating.py
git commit -m "feat(spark-vi): OnlineLDA topic-block gating (PLDA) with fixed-alpha v1 guard"
```

---

### Task 4: End-to-end gated recovery + masking-integrity test

**Files:**
- Test: `spark-vi/tests/test_lda_gating.py`

**Interfaces:**
- Consumes: `OnlineLDA(topic_blocks=...)`, `BOWDocument(groups=...)`, `TopicBlockPartition` — all from Tasks 1–3.

- [ ] **Step 1: Write the failing test (it will fail only if masking is wrong)**

Mirrors `test_gated_stm_recovers_planted_minority_phenotype` for LDA: a rare-only token cluster expressed only by the rare group must be captured by a foreground topic, and background topics must barely touch it.

```python
def test_gated_lda_recovers_planted_minority_phenotype():
    rng = np.random.default_rng(11)
    V = 12
    bg_tokens = np.arange(0, 8)
    rare_tokens = np.arange(8, 12)
    part = TopicBlockPartition("g", background_k=3, foreground=(("rare", 2),))

    def make_doc(is_rare):
        toks = rng.choice(bg_tokens, size=3, replace=False)
        if is_rare:
            toks = np.concatenate([toks, rng.choice(rare_tokens, size=2, replace=False)])
        toks = np.unique(toks).astype(np.int32)
        return BOWDocument(indices=toks, counts=np.ones(len(toks)), length=len(toks),
                           groups=frozenset({"rare"}) if is_rare else frozenset())

    docs = [make_doc(i % 5 == 0) for i in range(400)]   # ~20% rare
    # Fixed small alpha (<1) supplies document sparsity; optimize_alpha off (v1).
    model = OnlineLDA(K=part.K, vocab_size=V, alpha=1.0 / part.K,
                      random_seed=2, topic_blocks=part)
    gp = _fit(model, docs, n_iter=40, lr=0.5)
    beta = gp["lambda"] / gp["lambda"].sum(axis=1, keepdims=True)
    fg = part.block_indices("rare")
    bg = part.background_indices()
    assert beta[fg][:, rare_tokens].sum(axis=1).max() > 0.5   # a fg topic owns the rare cluster
    assert beta[bg][:, rare_tokens].sum(axis=1).max() < 0.1   # background stays off it
```

- [ ] **Step 2: Run test to verify it passes (green if Tasks 1–3 correct)**

Run: `cd spark-vi && python -m pytest tests/test_lda_gating.py::test_gated_lda_recovers_planted_minority_phenotype -v`
Expected: PASS. If it FAILS (fg mass ≤ 0.5 or bg mass ≥ 0.1), masking is mis-wired — debug Task 2/3 before proceeding (do not weaken the thresholds).

- [ ] **Step 3: Run the full engine test suite (no regressions)**

Run: `cd spark-vi && python -m pytest tests/test_lda_gating.py tests/test_lda_integration.py -v`
Expected: PASS (gating tests + existing LDA integration unaffected — the non-gated path is unchanged).

- [ ] **Step 4: Commit**

```bash
git add spark-vi/tests/test_lda_gating.py
git commit -m "test(spark-vi): gated-LDA recovers planted minority phenotype; background stays off it"
```

---

## Self-Review

**Spec coverage** (against `2026-06-24-gated-lda-plda-design.md` §1–2, the engine scope): §1 BOWDocument.groups → Task 1 ✓. §2 `_cavi_doc_inference` allowed-masking → Task 2 ✓; `OnlineLDA` topic_blocks + `_effective_partition` + masked local_update → Task 3 ✓. §3 asymmetric-α-per-allowed → **explicitly deferred to v2** (guarded, Appendix) — the spec under-scoped this; corrected here. §9 tests: non-gated identity (Task 3), masking integrity + recovery (Task 4); the *hard-regime / beats-STM* validation is a cloud experiment, not a unit test (noted below). §§4–8 (shim, driver, eval, dashboard, runner) are **out of scope for this plan** — follow-on plans.

**Placeholder scan:** none — every step has runnable code/commands.

**Type consistency:** `allowed: np.ndarray | None` consistent across Tasks 2–3; `topic_blocks` / `_effective_partition` / `.foreground` / `.allowed_indices` / `.block_indices` / `.background_indices` all match `TopicBlockPartition`'s real API; `groups=frozenset()` matches `STMDocument`.

## Follow-on (separate plans, not this one)

1. **Shim + driver + runner:** `OnlineLDAEstimator(topic_blocks, doc_group_col)`; `lda_bigquery_cloud.py` gating args + `source_cohort` materialization + `topic_block_spec` metadata + block-aware fit logger; `build_lda_args` gating flags. Then the **payoff cloud run**: gated LDA on `cancer_or_dementia`, 30 bg / 10 cancer / 10 dementia — the 0004/0005 rematch (block-aware NPMI eval already works for any model class). This is where the *hard-regime / beats-gated-STM* validation happens.
2. **Dashboard:** LDA masked corpus-mean-θ prevalence helper (the only model-specific piece; `gating.json` + k-anon are already model-agnostic).
3. **v2 — learned asymmetric α under masking** (Appendix).

## Appendix — v2: asymmetric-α optimization under PLDA masking (deferred)

The closed form in `alpha_newton_step` assumes every topic is in every document's simplex (single `D`, single ψ(Σα)). Under masking, doc `d` uses only its allowed set `A_d`, so the α-ELBO gradient is

```
g_k = Σ_{d: k∈A_d} [ ψ(Σ_{j∈A_d} α_j) − ψ(α_k) ] + Σ_{d: k∈A_d} E[log θ_dk]
```

The normalizer ψ(Σ_{j∈A_d} α_j) takes only **G+1 distinct values** for our 2-block-per-doc case — one per group `g`, with `S_g = Σ_{background} α + Σ_{foreground-g} α`. So it is computable as `Σ_g n_g·ψ(S_g)` with per-topic allowed-doc counts (`n_total·ψ(α_k)` subtracted for background topics; `n_g·ψ(α_k)` for a group-`g` foreground topic). The Hessian is block-structured (the rank-1 coupling differs per group), so v2 needs a per-group-simplex Newton step rather than the single Sherman-Morrison form. Validate against finite-difference gradients before shipping. Until then, fixed α<1 is correct and sufficient to test the PLDA hypothesis.
