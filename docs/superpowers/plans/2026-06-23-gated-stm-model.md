# Gated STM (Model) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in "topic-block gating" mode to the canonical `OnlineSTM` so a rare group of documents gets its own foreground topics (with their own vocabulary content) while a shared background block borrows the full corpus's strength — the model, shim, driver, and eval (the dashboard is a separate Plan 2).

**Architecture:** A new `TopicBlockPartition` value object partitions the K topics into a background block (all docs may express) and one foreground block per group (only that group's docs may express). Hard masking in per-doc inference makes a foreground topic's β/Γ/Σ estimate from its group's documents only; the M-step becomes block-aware (per-block Γ normal equations, per-topic Σ divisor). When the partition is `None`, every code path reduces to an implicit all-background partition that is numerically identical to today's STM.

**Tech Stack:** Python, NumPy, SciPy (L-BFGS), PySpark (mllib-shim VIRunner), pytest.

## Global Constraints

- Canonical no-gating path (partition `None`) must be numerically identical to today's `OnlineSTM`, pinned by a regression test. (verbatim from spec: "the no-gating path stays byte-identical")
- Terminology: use "group" for the gating variable and "background / foreground block" for the topic partition. NEVER introduce "cohort" as a new gating identifier (it already means the OMOP filter and `source_cohort`). The labeler's `background` quality category is a separate, untouched concept.
- Γ keeps shape (P, K); covariates stay keyed `(person_id, source_cohort)`; persisted global_params (`lambda`, `Gamma`, `Sigma`, `eta`) keep identical shapes.
- `group_var` must NOT also appear as a term in the prevalence formula (foreground rank-deficiency). Hard error.
- A foreground block with zero matching documents is a config error (raise). A block with few docs emits a warning.
- No LaTeX in any prose/docstring (Unicode Greek allowed: α β η λ Σ Γ ρ). No emojis in committed files. Code references in docs use markdown links `[name](path#Lstart-Lend)`.
- TDD throughout: failing test first, watch it fail, minimal implementation, watch it pass, commit.
- Commit messages end with: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

**Reference spec:** [docs/superpowers/specs/2026-06-23-gated-stm-background-foreground-design.md](../specs/2026-06-23-gated-stm-background-foreground-design.md)

---

## File Structure

- Create: `spark-vi/spark_vi/models/topic/partition.py` — `TopicBlockPartition` value object (resolve/validate index sets, group→block, manifest round-trip). One responsibility: the gating layout.
- Modify: `spark-vi/spark_vi/models/topic/types.py` — add `groups` field to `STMDocument`.
- Modify: `spark-vi/spark_vi/models/topic/stm.py` — masked per-doc inference, block-aware M-step, `OnlineSTM.topic_blocks`, ELBO + diagnostics under masking.
- Modify: `spark-vi/spark_vi/mllib/topic/_common.py` — `_vector_to_stm_document` group extraction.
- Modify: `spark-vi/spark_vi/mllib/topic/stm.py` — `StreamingSTM` gains `topic_blocks` + `doc_group_col`; `STMModel` persists the partition.
- Modify: `analysis/cloud/stm_bigquery_cloud.py` — build partition from CLI, thread group column, write `topic_block_spec` to manifest, formula guard.
- Modify: `scripts/run_experiment.py` — `build_stm_args` passes gating flags; resume guard adds `topic_block_spec`.
- Modify: `analysis/cloud/eval_coherence_cloud.py` — foreground-aware NPMI (background → full corpus, foreground → group sub-corpus; label by block).
- Create: `docs/decisions/00NN-gated-stm-hard-masking.md` — ADR (number assigned at write time).
- Create/Modify: experiment YAML under `experiments/` + a `docs/experiments/00NN-*.md` writeup.
- Tests: `spark-vi/tests/test_topic_block_partition.py` (new), and additions to `spark-vi/tests/test_stm_math.py`, `test_stm_contract.py`, `test_mllib_stm.py`, `test_mllib_stm_persistence.py`, `scripts/tests/test_run_experiment.py`.

---

## Task 1: TopicBlockPartition value object

**Files:**
- Create: `spark-vi/spark_vi/models/topic/partition.py`
- Test: `spark-vi/tests/test_topic_block_partition.py`

**Interfaces:**
- Produces:
  - `TopicBlockPartition(group_var: str, background_k: int, foreground: tuple[tuple[str, int], ...])` (frozen dataclass)
  - `.K -> int` (background_k + sum of foreground sizes)
  - `.groups -> tuple[str, ...]` (foreground labels in order)
  - `.background_indices() -> np.ndarray` (int64, `arange(background_k)`)
  - `.block_indices(group: str) -> np.ndarray` (int64, that group's contiguous foreground block)
  - `.allowed_indices(groups: frozenset[str]) -> np.ndarray` (int64, sorted: background ∪ blocks of given groups)
  - `.topic_labels() -> list[str]` (length K; `"background"` or the owning group label)
  - `.to_dict() -> dict` / `TopicBlockPartition.from_dict(d) -> TopicBlockPartition`

- [ ] **Step 1: Write the failing tests**

```python
# spark-vi/tests/test_topic_block_partition.py
import numpy as np
import pytest

from spark_vi.models.topic.partition import TopicBlockPartition


def _part():
    return TopicBlockPartition(
        group_var="source_cohort",
        background_k=3,
        foreground=(("cancer", 2), ("dementia", 2)),
    )


def test_k_is_background_plus_foreground():
    assert _part().K == 7


def test_index_blocks_are_contiguous_and_disjoint():
    p = _part()
    np.testing.assert_array_equal(p.background_indices(), [0, 1, 2])
    np.testing.assert_array_equal(p.block_indices("cancer"), [3, 4])
    np.testing.assert_array_equal(p.block_indices("dementia"), [5, 6])


def test_allowed_indices_unions_background_and_group_blocks():
    p = _part()
    np.testing.assert_array_equal(
        p.allowed_indices(frozenset({"cancer"})), [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(
        p.allowed_indices(frozenset()), [0, 1, 2])
    np.testing.assert_array_equal(
        p.allowed_indices(frozenset({"cancer", "dementia"})), [0, 1, 2, 3, 4, 5, 6])


def test_topic_labels():
    assert _part().topic_labels() == [
        "background", "background", "background",
        "cancer", "cancer", "dementia", "dementia"]


def test_unknown_group_in_allowed_indices_raises():
    with pytest.raises(KeyError):
        _part().allowed_indices(frozenset({"nope"}))


def test_rejects_nonpositive_sizes():
    with pytest.raises(ValueError):
        TopicBlockPartition("g", 0, (("a", 2),))
    with pytest.raises(ValueError):
        TopicBlockPartition("g", 3, (("a", 0),))


def test_rejects_duplicate_group_labels():
    with pytest.raises(ValueError):
        TopicBlockPartition("g", 3, (("a", 2), ("a", 1)))


def test_dict_roundtrip():
    p = _part()
    assert TopicBlockPartition.from_dict(p.to_dict()) == p
    assert p.to_dict() == {
        "group_var": "source_cohort",
        "background_k": 3,
        "foreground": [["cancer", 2], ["dementia", 2]],
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd spark-vi && python -m pytest tests/test_topic_block_partition.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'spark_vi.models.topic.partition'`

- [ ] **Step 3: Write the implementation**

```python
# spark-vi/spark_vi/models/topic/partition.py
"""TopicBlockPartition: the gating layout for OnlineSTM background/foreground blocks.

Partitions the K topics into a background block (every document may express it)
and one contiguous foreground block per group (only that group's documents may
express it). The engine consumes only the resolved index sets; the contiguous
layout (background first, then groups in declared order) is a readability
convention. See docs/superpowers/specs/2026-06-23-gated-stm-background-foreground-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TopicBlockPartition:
    group_var: str
    background_k: int
    foreground: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if self.background_k < 1:
            raise ValueError(f"background_k must be >= 1, got {self.background_k}")
        labels = [g for g, _ in self.foreground]
        if len(labels) != len(set(labels)):
            raise ValueError(f"duplicate group labels in foreground: {labels}")
        for g, size in self.foreground:
            if size < 1:
                raise ValueError(f"foreground block '{g}' size must be >= 1, got {size}")

    @property
    def K(self) -> int:
        return self.background_k + sum(size for _, size in self.foreground)

    @property
    def groups(self) -> tuple[str, ...]:
        return tuple(g for g, _ in self.foreground)

    def background_indices(self) -> np.ndarray:
        return np.arange(self.background_k, dtype=np.int64)

    def _block_start(self, group: str) -> int:
        start = self.background_k
        for g, size in self.foreground:
            if g == group:
                return start
            start += size
        raise KeyError(f"unknown group {group!r}; known groups: {self.groups}")

    def block_indices(self, group: str) -> np.ndarray:
        start = self._block_start(group)
        size = dict(self.foreground)[group]
        return np.arange(start, start + size, dtype=np.int64)

    def allowed_indices(self, groups: frozenset[str]) -> np.ndarray:
        parts = [self.background_indices()]
        for g in sorted(groups):
            parts.append(self.block_indices(g))  # raises KeyError on unknown group
        return np.unique(np.concatenate(parts)).astype(np.int64)

    def topic_labels(self) -> list[str]:
        labels = ["background"] * self.background_k
        for g, size in self.foreground:
            labels.extend([g] * size)
        return labels

    def to_dict(self) -> dict:
        return {
            "group_var": self.group_var,
            "background_k": self.background_k,
            "foreground": [[g, size] for g, size in self.foreground],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TopicBlockPartition":
        return cls(
            group_var=d["group_var"],
            background_k=int(d["background_k"]),
            foreground=tuple((g, int(size)) for g, size in d["foreground"]),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd spark-vi && python -m pytest tests/test_topic_block_partition.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/partition.py spark-vi/tests/test_topic_block_partition.py
git commit -m "$(printf 'feat(stm): TopicBlockPartition value object for gating layout\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 2: STMDocument.groups field + doc-builder extraction

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/types.py:49-66`
- Modify: `spark-vi/spark_vi/mllib/topic/_common.py:39-59`
- Test: `spark-vi/tests/test_stm_math.py` (append to the existing `STMDocument` test class)

**Interfaces:**
- Consumes: `STMDocument` (existing).
- Produces:
  - `STMDocument(indices, counts, length, x, groups: frozenset[str] = frozenset())`
  - `_vector_to_stm_document(row, features_col="features", covariates_col="covariates", group_col: str | None = None) -> STMDocument` — when `group_col` is set, reads `row[group_col]` and stores `frozenset({str(value)})` (a list/array value stores all members); when `None`, `groups=frozenset()`.

- [ ] **Step 1: Write the failing tests**

```python
# append in spark-vi/tests/test_stm_math.py
def test_stmdocument_groups_defaults_to_empty_frozenset():
    import numpy as np
    from spark_vi.models.topic.types import STMDocument
    d = STMDocument(indices=np.array([0], dtype=np.int32),
                    counts=np.array([1.0]), length=1, x=np.array([1.0]))
    assert d.groups == frozenset()


def test_stmdocument_carries_groups():
    import numpy as np
    from spark_vi.models.topic.types import STMDocument
    d = STMDocument(indices=np.array([0], dtype=np.int32),
                    counts=np.array([1.0]), length=1, x=np.array([1.0]),
                    groups=frozenset({"cancer"}))
    assert d.groups == frozenset({"cancer"})


def test_vector_to_stm_document_extracts_group_from_column():
    from pyspark.ml.linalg import Vectors
    from spark_vi.mllib.topic._common import _vector_to_stm_document
    row = {"features": Vectors.sparse(3, {0: 2.0}),
           "covariates": Vectors.dense([1.0, 0.5]),
           "source_cohort": "dementia"}
    doc = _vector_to_stm_document(row, group_col="source_cohort")
    assert doc.groups == frozenset({"dementia"})


def test_vector_to_stm_document_no_group_col_yields_empty():
    from pyspark.ml.linalg import Vectors
    from spark_vi.mllib.topic._common import _vector_to_stm_document
    row = {"features": Vectors.sparse(3, {0: 2.0}),
           "covariates": Vectors.dense([1.0, 0.5])}
    doc = _vector_to_stm_document(row)
    assert doc.groups == frozenset()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd spark-vi && python -m pytest tests/test_stm_math.py -k "groups or group_col or group" -v`
Expected: FAIL — `STMDocument` has no `groups` field / `_vector_to_stm_document` has no `group_col` kwarg.

- [ ] **Step 3: Implement**

In `spark-vi/spark_vi/models/topic/types.py`, add the field to `STMDocument` (frozen, slots dataclass):

```python
    indices: np.ndarray
    counts: np.ndarray
    length: int
    x: np.ndarray
    groups: frozenset = frozenset()
```

Update the docstring invariant list to add:
```
      groups:  frozenset[str] of group labels (empty = background only / gating off).
```

In `spark-vi/spark_vi/mllib/topic/_common.py`, update the signature and body:

```python
def _vector_to_stm_document(
    row,
    features_col: str = "features",
    covariates_col: str = "covariates",
    group_col: str | None = None,
) -> STMDocument:
    """Construct an STMDocument from a row with a BOW vector and a covariate vector.

    When group_col is set, row[group_col] supplies the doc's gating group(s):
    a scalar value becomes a singleton frozenset; a list/tuple value stores all
    members. When None, groups is empty (background-only / gating off).
    """
    bow = _vector_to_bow_document(row[features_col])
    cov = row[covariates_col]
    if group_col is None:
        groups = frozenset()
    else:
        val = row[group_col]
        groups = (frozenset(str(v) for v in val)
                  if isinstance(val, (list, tuple))
                  else frozenset({str(val)}))
    return STMDocument(
        indices=bow.indices,
        counts=bow.counts,
        length=bow.length,
        x=np.asarray(cov, dtype=np.float64),
        groups=groups,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd spark-vi && python -m pytest tests/test_stm_math.py -k "groups or group_col or group" -v`
Expected: PASS (4 tests). Also run the full `test_stm_math.py` to confirm the new default field broke nothing: `python -m pytest tests/test_stm_math.py -v` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/types.py spark-vi/spark_vi/mllib/topic/_common.py spark-vi/tests/test_stm_math.py
git commit -m "$(printf 'feat(stm): STMDocument.groups + doc-builder group extraction\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 3: Masked per-doc inference

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/stm.py:170-206` (`_stm_doc_inference`)
- Test: `spark-vi/tests/test_stm_math.py`

**Interfaces:**
- Consumes: existing `_stm_neg_log_joint`, `_stm_neg_log_joint_grad`, `_stm_neg_log_joint_hessian` (these already operate on whatever K-length `expElogbeta`, `Gamma`, `Sigma_diag` they are handed — so masking is achieved by passing them the row/column-restricted views, no change to those three).
- Produces: `_stm_doc_inference(..., allowed: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, int]`. When `allowed` is given (sorted int indices into [0,K)), the L-BFGS optimizes only over those topics; the returned `eta_hat` is a full-K vector with disallowed entries set to `-np.inf` (so `_softmax` yields exactly 0 there), and `nu_d` is full K×K with disallowed rows/cols 0. When `allowed is None`, behavior is identical to today.

- [ ] **Step 1: Write the failing tests**

```python
# append in spark-vi/tests/test_stm_math.py
class TestMaskedDocInference:
    def _setup(self, K=5, V=4):
        import numpy as np
        rng = np.random.default_rng(0)
        expElogbeta = rng.random((K, V)) + 0.1
        Gamma = np.zeros((2, K))
        Sigma = np.ones(K)
        indices = np.array([0, 2], dtype=np.int32)
        counts = np.array([3.0, 1.0])
        x = np.array([1.0, 0.0])
        return expElogbeta, Gamma, Sigma, indices, counts, x

    def test_disallowed_topics_get_zero_theta(self):
        import numpy as np
        from spark_vi.models.topic.stm import _stm_doc_inference, _softmax
        eb, G, S, idx, cnt, x = self._setup()
        allowed = np.array([0, 1, 2], dtype=np.int64)  # topics 3,4 disallowed
        eta_hat, nu_d, _ = _stm_doc_inference(
            indices=idx, counts=cnt, expElogbeta=eb, Gamma=G,
            Sigma_diag=S, x=x, allowed=allowed)
        theta = _softmax(eta_hat)
        assert theta[3] == 0.0 and theta[4] == 0.0
        assert abs(theta[:3].sum() - 1.0) < 1e-9
        # nu_d zero on disallowed rows/cols
        assert np.all(nu_d[3, :] == 0.0) and np.all(nu_d[:, 4] == 0.0)

    def test_allowed_none_matches_full_inference(self):
        import numpy as np
        from spark_vi.models.topic.stm import _stm_doc_inference
        eb, G, S, idx, cnt, x = self._setup()
        a = _stm_doc_inference(indices=idx, counts=cnt, expElogbeta=eb,
                               Gamma=G, Sigma_diag=S, x=x, allowed=None)
        b = _stm_doc_inference(indices=idx, counts=cnt, expElogbeta=eb,
                               Gamma=G, Sigma_diag=S, x=x,
                               allowed=np.arange(eb.shape[0], dtype=np.int64))
        np.testing.assert_allclose(a[0], b[0], atol=1e-8)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd spark-vi && python -m pytest tests/test_stm_math.py -k MaskedDocInference -v`
Expected: FAIL — `_stm_doc_inference` has no `allowed` kwarg.

- [ ] **Step 3: Implement**

Replace `_stm_doc_inference` body in `spark-vi/spark_vi/models/topic/stm.py`:

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
) -> tuple[np.ndarray, np.ndarray, int]:
    """Per-doc Laplace approximation, optionally restricted to an allowed topic set.

    When allowed is None, optimizes over all K topics (canonical). When allowed
    is a sorted index array, the L-BFGS runs only on those topics (the neg-log-
    joint sees the row/column-restricted views); disallowed topics are filled
    with eta=-inf (theta exactly 0) and nu_d=0 in the returned full-K arrays, so
    they contribute nothing to any downstream sufficient statistic.
    """
    K = expElogbeta.shape[0]
    if allowed is None:
        allowed = np.arange(K, dtype=np.int64)
    sub_expElogbeta = expElogbeta[allowed]
    sub_Gamma = Gamma[:, allowed]
    sub_Sigma = Sigma_diag[allowed]
    eta0 = np.zeros(allowed.shape[0], dtype=np.float64)
    common = dict(
        indices=indices, counts=counts, expElogbeta=sub_expElogbeta,
        Gamma=sub_Gamma, Sigma_diag=sub_Sigma, x=x,
    )
    f = partial(_stm_neg_log_joint, **common)
    g = partial(_stm_neg_log_joint_grad, **common)
    result = minimize(f, x0=eta0, jac=g, method="L-BFGS-B",
                      options={"maxiter": max_iter, "gtol": tol})
    sub_eta = result.x
    H = _stm_neg_log_joint_hessian(sub_eta, **common)
    sub_nu = np.linalg.inv(H)

    eta_hat = np.full(K, -np.inf, dtype=np.float64)
    eta_hat[allowed] = sub_eta
    nu_d = np.zeros((K, K), dtype=np.float64)
    nu_d[np.ix_(allowed, allowed)] = sub_nu
    return eta_hat, nu_d, int(result.nit)
```

Note: `_softmax` already subtracts the max before `exp`, so `-inf` entries map to `exp(-inf)=0` cleanly. Confirm no `nan`: the max is always a finite allowed entry because `allowed` is non-empty.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd spark-vi && python -m pytest tests/test_stm_math.py -k MaskedDocInference -v`
Expected: PASS (2 tests). Then full math + contract suite to confirm no regression: `python -m pytest tests/test_stm_math.py tests/test_stm_contract.py -v` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/stm.py spark-vi/tests/test_stm_math.py
git commit -m "$(printf 'feat(stm): masked per-doc inference over an allowed topic set\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 4: Block-aware M-step + OnlineSTM.topic_blocks (the heart)

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/stm.py` (`OnlineSTM.__init__`, `initialize_global`, `local_update`, `update_global`)
- Test: `spark-vi/tests/test_stm_contract.py` (new gating tests)

**Interfaces:**
- Consumes: `TopicBlockPartition` (Task 1), `STMDocument.groups` (Task 2), masked `_stm_doc_inference` (Task 3).
- Produces:
  - `OnlineSTM(K, vocab_size, P, ..., topic_blocks: TopicBlockPartition | None = None)`. If `topic_blocks` is set, `topic_blocks.K == K` is asserted.
  - `local_update` now also returns `XtX_groups` (shape `(G, P, P)` where `G = len(topic_blocks.foreground)`, or `(0, P, P)` when no partition) and `n_docs_per_topic` (shape `(K,)`). `XtX` is renamed conceptually to "all-doc" but the key stays `"XtX"`.
  - `update_global` solves background columns with `XtX` and each group's foreground columns with its `XtX_groups[i]`; Σ divides each topic by its `n_docs_per_topic` entry.

Internal helper: when `topic_blocks is None`, the engine uses an **implicit all-background partition** so every code path unifies — `allowed = arange(K)` for every doc, `XtX_groups` is `(0,P,P)`, and `n_docs_per_topic` is uniformly `n_docs`, which makes `update_global` reduce exactly to today's single-`XtX` solve and scalar Σ divisor.

- [ ] **Step 1: Write the failing tests**

```python
# spark-vi/tests/test_stm_contract.py  (new tests; keep existing ones)
import numpy as np
from spark_vi.models.topic.stm import OnlineSTM
from spark_vi.models.topic.partition import TopicBlockPartition
from spark_vi.models.topic.types import STMDocument


def _docs(rng, n, V, P, groups_fn):
    out = []
    for i in range(n):
        nz = rng.choice(V, size=2, replace=False)
        out.append(STMDocument(
            indices=np.sort(nz).astype(np.int32),
            counts=np.array([2.0, 1.0]),
            length=3, x=rng.random(P), groups=groups_fn(i)))
    return out


def test_canonical_collapse_update_global_matches_original_formulas():
    # With partition=None, update_global must use EXACTLY the pre-gating closed
    # forms: a single XtX solve for Gamma and a scalar n_docs divisor for Sigma.
    # (Together with Task 3's masked==unmasked inference test, this pins the full
    # None path as byte-identical to the original engine.)
    rng = np.random.default_rng(3)
    V, P, K = 6, 2, 4
    docs = _docs(rng, 8, V, P, lambda i: frozenset())
    model = OnlineSTM(K=K, vocab_size=V, P=P, random_seed=1, topic_blocks=None)
    gp0 = model.initialize_global(None)
    stats = model.local_update(list(docs), gp0)
    out = model.update_global(gp0, stats, 0.5)
    # Original closed forms recomputed independently from the stats.
    ridge = model.sigma_ridge * np.eye(P)
    Gamma_target = np.linalg.solve(stats["XtX"] + ridge, stats["XtMu"])
    exp_Gamma = 0.5 * gp0["Gamma"] + 0.5 * Gamma_target
    Sigma_target = stats["residual_diag_stat"] / float(stats["n_docs"])
    exp_Sigma = np.maximum(0.5 * gp0["Sigma"] + 0.5 * Sigma_target, model.SIGMA_FLOOR)
    target_lam = float(gp0["eta"]) + stats["lambda_stats"]
    exp_lam = 0.5 * gp0["lambda"] + 0.5 * target_lam
    # The None-path Sigma divisor must be uniformly n_docs (every topic allowed).
    np.testing.assert_array_equal(stats["n_docs_per_topic"],
                                  np.full(K, float(stats["n_docs"])))
    np.testing.assert_allclose(out["Gamma"], exp_Gamma, atol=1e-12)
    np.testing.assert_allclose(out["Sigma"], exp_Sigma, atol=1e-12)
    np.testing.assert_allclose(out["lambda"], exp_lam, atol=1e-12)


def test_zero_foreground_contribution_from_majority():
    # Majority docs (no groups) must contribute ZERO to foreground lambda rows.
    rng = np.random.default_rng(5)
    V, P = 6, 2
    part = TopicBlockPartition("g", background_k=2, foreground=(("rare", 2),))
    K = part.K  # 4; foreground topics = [2, 3]
    # All docs are majority (no groups) -> foreground never allowed.
    docs = _docs(rng, 10, V, P, lambda i: frozenset())
    model = OnlineSTM(K=K, vocab_size=V, P=P, random_seed=1, topic_blocks=part)
    gp = model.initialize_global(None)
    stats = model.local_update(docs, gp)
    assert np.all(stats["lambda_stats"][2:, :] == 0.0)
    assert np.all(stats["n_docs_per_topic"][2:] == 0.0)
    assert stats["n_docs_per_topic"][0] == 10.0  # background trained on all


def test_block_aware_sigma_divisor_uses_per_topic_counts():
    rng = np.random.default_rng(7)
    V, P = 6, 2
    part = TopicBlockPartition("g", background_k=2, foreground=(("rare", 1),))
    K = part.K  # 3; foreground topic = [2]
    # 6 majority + 4 'rare' docs.
    docs = (_docs(rng, 6, V, P, lambda i: frozenset())
            + _docs(rng, 4, V, P, lambda i: frozenset({"rare"})))
    model = OnlineSTM(K=K, vocab_size=V, P=P, random_seed=1, topic_blocks=part)
    gp = model.initialize_global(None)
    stats = model.local_update(docs, gp)
    np.testing.assert_array_equal(stats["n_docs_per_topic"], [10.0, 10.0, 4.0])


def test_topic_blocks_k_mismatch_raises():
    part = TopicBlockPartition("g", background_k=2, foreground=(("rare", 2),))
    import pytest
    with pytest.raises(ValueError):
        OnlineSTM(K=3, vocab_size=6, P=2, topic_blocks=part)  # part.K == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd spark-vi && python -m pytest tests/test_stm_contract.py -k "canonical_collapse or zero_foreground or block_aware_sigma or k_mismatch" -v`
Expected: FAIL — `OnlineSTM.__init__` has no `topic_blocks` kwarg; stats dict has no `n_docs_per_topic`/`XtX_groups`.

- [ ] **Step 3: Implement**

(3a) `__init__`: add the kwarg and validation. After the existing validations and before assignments, add `topic_blocks=None` to the signature; then:

```python
        if topic_blocks is not None and topic_blocks.K != int(K):
            raise ValueError(
                f"topic_blocks.K ({topic_blocks.K}) != K ({K})")
        self.topic_blocks = topic_blocks
```

Add an effective-partition helper on the class (used by both updates):

```python
    def _effective_partition(self):
        """The real partition, or an implicit all-background one when None."""
        if self.topic_blocks is not None:
            return self.topic_blocks
        from spark_vi.models.topic.partition import TopicBlockPartition
        return TopicBlockPartition(group_var="", background_k=self.K, foreground=())

    def _allowed(self, doc) -> np.ndarray:
        part = self._effective_partition()
        return part.allowed_indices(doc.groups)
```

(3b) `local_update`: thread the allowed set + new accumulators. Replace the accumulator setup and per-doc loop:

```python
        part = self._effective_partition()
        G = len(part.foreground)
        group_order = part.groups  # tuple of labels in block order

        lambda_stats = np.zeros((K, V), dtype=np.float64)
        XtX = np.zeros((P, P), dtype=np.float64)            # all-doc cross-product
        XtX_groups = np.zeros((G, P, P), dtype=np.float64)  # per-group cross-product
        XtMu = np.zeros((P, K), dtype=np.float64)
        residual_diag = np.zeros(K, dtype=np.float64)
        n_docs_per_topic = np.zeros(K, dtype=np.float64)
        doc_loglik = 0.0
        doc_eta_kl = 0.0
        n_docs = 0

        log_Sigma_diag = np.log(Sigma_diag)

        for doc in rows:
            allowed = part.allowed_indices(doc.groups)
            eta_hat, nu_d, _ = _stm_doc_inference(
                indices=doc.indices, counts=doc.counts,
                expElogbeta=expElogbeta,
                Gamma=Gamma, Sigma_diag=Sigma_diag, x=doc.x,
                max_iter=self.lbfgs_max_iter, tol=self.lbfgs_tol,
                allowed=allowed,
            )
            p = _softmax(eta_hat)  # 0 on disallowed
            eb_d = expElogbeta[:, doc.indices]
            q_w = eb_d.T @ p + 1e-100
            phi = (eb_d * p[:, None]) / q_w[None, :]   # (K, n_unique); 0 on disallowed rows
            sstats_row = phi * doc.counts[None, :]
            lambda_stats[:, doc.indices] += sstats_row

            xxT = np.outer(doc.x, doc.x)
            XtX += xxT
            for gi, g in enumerate(group_order):
                if g in doc.groups:
                    XtX_groups[gi] += xxT

            # XtMu / residual_diag / counts only over allowed topics.
            eta_allowed = eta_hat[allowed]
            XtMu[:, allowed] += np.outer(doc.x, eta_allowed)
            resid = np.zeros(K, dtype=np.float64)
            resid[allowed] = eta_allowed - (Gamma.T @ doc.x)[allowed]
            residual_diag[allowed] += resid[allowed] ** 2 + np.diag(nu_d)[allowed]
            n_docs_per_topic[allowed] += 1.0

            doc_loglik += float(np.sum(doc.counts * np.log(q_w)))
            # KL over the allowed sub-space only.
            al = allowed
            tr_term = float(np.sum(np.diag(nu_d)[al] / Sigma_diag[al]))
            quad_term = float(np.sum(resid[al] ** 2 / Sigma_diag[al]))
            sub_nu = nu_d[np.ix_(al, al)]
            sign, logdet_nu = np.linalg.slogdet(sub_nu)
            logdet_Sigma = float(np.sum(log_Sigma_diag[al]))
            doc_eta_kl += 0.5 * (tr_term + quad_term - len(al) + logdet_Sigma - logdet_nu)
            n_docs += 1

        return {
            "lambda_stats": lambda_stats,
            "XtX": XtX,
            "XtX_groups": XtX_groups,
            "XtMu": XtMu,
            "residual_diag_stat": residual_diag,
            "n_docs_per_topic": n_docs_per_topic,
            "doc_loglik_sum": np.array(doc_loglik),
            "doc_eta_kl_sum": np.array(doc_eta_kl),
            "n_docs": np.array(float(n_docs)),
        }
```

(3c) `update_global`: block-aware Γ + per-topic Σ. Replace the Γ and Σ blocks:

```python
        part = self._effective_partition()
        XtX = target_stats["XtX"]
        XtX_groups = target_stats["XtX_groups"]
        XtMu = target_stats["XtMu"]
        ridge_eye = self.sigma_ridge * np.eye(self.P)

        Gamma_target = np.zeros_like(Gamma)
        bg = part.background_indices()
        Gamma_target[:, bg] = np.linalg.solve(XtX + ridge_eye, XtMu[:, bg])
        for gi, g in enumerate(part.groups):
            cols = part.block_indices(g)
            Gamma_target[:, cols] = np.linalg.solve(
                XtX_groups[gi] + ridge_eye, XtMu[:, cols])
        new_Gamma = (1.0 - learning_rate) * Gamma + learning_rate * Gamma_target

        # Sigma: per-topic divisor (background -> n_docs; foreground -> group docs).
        n_per_topic = np.maximum(target_stats["n_docs_per_topic"], 1.0)
        Sigma_target = target_stats["residual_diag_stat"] / n_per_topic
        new_Sigma = (1.0 - learning_rate) * Sigma_diag + learning_rate * Sigma_target
        new_Sigma = np.maximum(new_Sigma, self.SIGMA_FLOOR)
```

(Keep the existing λ SVI step unchanged above these blocks.) Note: when partition is None, `bg = arange(K)`, the group loop is empty, and `n_per_topic` is uniformly `n_docs` (every topic allowed for every doc) — so `Gamma_target = solve(XtX+ridge, XtMu)` and `Sigma_target = residual_diag / n_docs`, identical to the original.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd spark-vi && python -m pytest tests/test_stm_contract.py -v`
Expected: PASS including the four new tests. Then run the full STM suite: `python -m pytest tests/test_stm_math.py tests/test_stm_contract.py tests/test_stm_integration.py -v` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/stm.py spark-vi/tests/test_stm_contract.py
git commit -m "$(printf 'feat(stm): block-aware M-step + OnlineSTM.topic_blocks gating\n\nMasked local_update accumulates per-group XtX and per-topic doc counts;\nupdate_global solves Gamma block-wise and Sigma per-topic. Partition=None\nreduces to an implicit all-background partition, numerically identical to\nthe canonical engine (pinned by test_canonical_collapse).\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 5: Recovery test + per-block diagnostics

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/stm.py` (`iteration_summary`, `iteration_diagnostics`)
- Test: `spark-vi/tests/test_stm_integration.py`

**Interfaces:**
- Consumes: gated `OnlineSTM` (Task 4).
- Produces: `iteration_diagnostics` returns, in addition to `Gamma`/`Sigma`, `n_docs_per_topic_last` is NOT available here (it's a stat, not a global param), so diagnostics report partition-derived static structure: `topic_block_labels` (list[str]) when `topic_blocks` is set. `iteration_summary` appends a per-block Σλ mass summary when gated.

- [ ] **Step 1: Write the failing test (recovery — the "it works" test)**

```python
# spark-vi/tests/test_stm_integration.py  (append)
def test_gated_stm_recovers_planted_minority_phenotype():
    """A planted vocabulary cluster expressed ONLY by the rare group must be
    recovered by a foreground topic, and majority docs must not express it."""
    import numpy as np
    from spark_vi.models.topic.stm import OnlineSTM, _softmax
    from spark_vi.models.topic.partition import TopicBlockPartition
    from spark_vi.models.topic.types import STMDocument

    rng = np.random.default_rng(11)
    V = 12
    # Background vocab = tokens 0..7; rare-only phenotype = tokens 8..11.
    bg_tokens = np.arange(0, 8)
    rare_tokens = np.arange(8, 12)
    part = TopicBlockPartition("g", background_k=3, foreground=(("rare", 2),))
    K = part.K  # 5

    def make_doc(is_rare):
        toks = rng.choice(bg_tokens, size=3, replace=False)
        if is_rare:
            toks = np.concatenate([toks, rng.choice(rare_tokens, size=2, replace=False)])
        toks = np.unique(toks)
        return STMDocument(indices=toks.astype(np.int32),
                           counts=np.ones(len(toks)), length=len(toks),
                           x=np.array([1.0]),
                           groups=frozenset({"rare"}) if is_rare else frozenset())

    docs = [make_doc(i % 5 == 0) for i in range(400)]  # ~20% rare
    model = OnlineSTM(K=K, vocab_size=V, P=1, random_seed=2, topic_blocks=part)
    gp = model.initialize_global(None)
    for _ in range(30):
        stats = model.local_update(docs, gp)
        gp = model.update_global(gp, stats, learning_rate=0.5)

    lam = gp["lambda"]
    beta = lam / lam.sum(axis=1, keepdims=True)
    fg = part.block_indices("rare")
    # Some foreground topic concentrates on the rare-only tokens.
    fg_mass_on_rare = beta[fg][:, rare_tokens].sum(axis=1).max()
    bg_mass_on_rare = beta[part.background_indices()][:, rare_tokens].sum(axis=1).max()
    assert fg_mass_on_rare > 0.5, fg_mass_on_rare
    # Background barely touches rare tokens (majority never expresses them).
    assert bg_mass_on_rare < 0.1, bg_mass_on_rare
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `cd spark-vi && python -m pytest tests/test_stm_integration.py -k recovers_planted -v`
Expected: This test should PASS if Task 4 is correct (it is an end-to-end validation, not a new feature). If it FAILS, the M-step has a bug — debug Task 4 before continuing. Treat a failure here as a Task-4 regression, not a Task-5 implementation gap.

- [ ] **Step 3: Implement the diagnostics additions**

In `iteration_diagnostics`, append block labels when gated:

```python
    def iteration_diagnostics(self, global_params):
        diag = {
            "Gamma": np.asarray(global_params["Gamma"]),
            "Sigma": np.asarray(global_params["Sigma"]),
        }
        if self.topic_blocks is not None:
            diag["topic_block_labels"] = np.asarray(
                self.topic_blocks.topic_labels(), dtype=object)
        return diag
```

In `iteration_summary`, append per-block Σλ mass when gated:

```python
    def iteration_summary(self, global_params):
        Gamma = global_params["Gamma"]
        Sigma = global_params["Sigma"]
        lam = global_params["lambda"]
        lam_row_sums = lam.sum(axis=1)
        base = (
            f"|Γ|[max={np.abs(Gamma).max():.3g} mean={np.abs(Gamma).mean():.3g}], "
            f"Σ[min={Sigma.min():.3g} max={Sigma.max():.3g}], "
            f"Σλ_k[min={lam_row_sums.min():.3g} max={lam_row_sums.max():.3g}]"
        )
        if self.topic_blocks is None:
            return base
        part = self.topic_blocks
        bg_mass = float(lam_row_sums[part.background_indices()].sum())
        fg_bits = []
        for g in part.groups:
            fg_bits.append(f"{g}={float(lam_row_sums[part.block_indices(g)].sum()):.3g}")
        return base + f", blocks[bg={bg_mass:.3g} " + " ".join(fg_bits) + "]"
```

- [ ] **Step 4: Write a small diagnostics test and run**

```python
# spark-vi/tests/test_stm_integration.py  (append)
def test_gated_diagnostics_include_block_labels():
    import numpy as np
    from spark_vi.models.topic.stm import OnlineSTM
    from spark_vi.models.topic.partition import TopicBlockPartition
    part = TopicBlockPartition("g", background_k=2, foreground=(("rare", 1),))
    model = OnlineSTM(K=3, vocab_size=4, P=1, random_seed=1, topic_blocks=part)
    gp = model.initialize_global(None)
    diag = model.iteration_diagnostics(gp)
    assert list(diag["topic_block_labels"]) == ["background", "background", "rare"]
    assert "blocks[bg=" in model.iteration_summary(gp)
```

Run: `cd spark-vi && python -m pytest tests/test_stm_integration.py -k "recovers_planted or block_labels" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/stm.py spark-vi/tests/test_stm_integration.py
git commit -m "$(printf 'feat(stm): gated recovery test + per-block diagnostics\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 6: StreamingSTM gating wiring

**Files:**
- Modify: `spark-vi/spark_vi/mllib/topic/stm.py:66-117` (`__init__`), `:119-236` (`fit`)
- Test: `spark-vi/tests/test_mllib_stm.py`

**Interfaces:**
- Consumes: gated `OnlineSTM` (Task 4), `_vector_to_stm_document(group_col=...)` (Task 2), `TopicBlockPartition` (Task 1).
- Produces: `StreamingSTM(..., topic_blocks: TopicBlockPartition | None = None, doc_group_col: str | None = None)`. `fit` selects `doc_group_col` (when set) into the RDD and passes it to `_vector_to_stm_document`; constructs `OnlineSTM(topic_blocks=self.topic_blocks)`. Validates: `doc_group_col` set iff `topic_blocks` set; and `topic_blocks.group_var` not in `self.covariate_names` (the formula guard).

- [ ] **Step 1: Write the failing test**

```python
# spark-vi/tests/test_mllib_stm.py  (append; reuse the module's spark fixture)
def test_streaming_stm_rejects_group_var_in_formula():
    import pytest
    from spark_vi.mllib.topic.stm import StreamingSTM
    from spark_vi.models.topic.partition import TopicBlockPartition
    part = TopicBlockPartition("source_cohort", background_k=2, foreground=(("cancer", 1),))
    with pytest.raises(ValueError, match="group_var"):
        StreamingSTM(
            K=3, covariates_col="covariates",
            covariate_names=["Intercept", "C(source_cohort)[T.cancer]"],
            topic_blocks=part, doc_group_col="source_cohort")


def test_streaming_stm_requires_group_col_with_partition():
    import pytest
    from spark_vi.mllib.topic.stm import StreamingSTM
    from spark_vi.models.topic.partition import TopicBlockPartition
    part = TopicBlockPartition("source_cohort", background_k=2, foreground=(("cancer", 1),))
    with pytest.raises(ValueError, match="doc_group_col"):
        StreamingSTM(K=3, covariates_col="covariates",
                     covariate_names=["Intercept", "age"], topic_blocks=part)


def test_streaming_stm_gated_fit_smoke(spark):
    from pyspark.ml.linalg import Vectors
    from spark_vi.mllib.topic.stm import StreamingSTM
    from spark_vi.models.topic.partition import TopicBlockPartition
    part = TopicBlockPartition("grp", background_k=2, foreground=(("rare", 1),))
    rows = []
    for i in range(20):
        rare = i % 4 == 0
        rows.append((Vectors.sparse(5, {i % 5: 2.0, (i + 1) % 5: 1.0}),
                     Vectors.dense([1.0, float(i % 2)]),
                     "rare" if rare else "common"))
    df = spark.createDataFrame(rows, ["features", "covariates", "grp"])
    est = StreamingSTM(K=3, covariates_col="covariates",
                       covariate_names=["Intercept", "age"],
                       topic_blocks=part, doc_group_col="grp")
    model = est.fit(df, max_iter=3, subsampling_rate=1.0)
    assert model.global_params["lambda"].shape == (3, 5)
    assert model.metadata  # fit produced a model
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd spark-vi && python -m pytest tests/test_mllib_stm.py -k "group_var or group_col or gated_fit" -v`
Expected: FAIL — `StreamingSTM` has no `topic_blocks`/`doc_group_col` kwargs.

- [ ] **Step 3: Implement**

In `StreamingSTM.__init__`, add params `topic_blocks=None, doc_group_col=None` to the signature, then after the existing Path A/B assignments add:

```python
        self.topic_blocks = topic_blocks
        self.doc_group_col = doc_group_col
        if topic_blocks is not None:
            if doc_group_col is None:
                raise ValueError(
                    "topic_blocks requires doc_group_col (the column naming each "
                    "document's gating group).")
            if self.covariate_names is not None and \
                    _formula_mentions(topic_blocks.group_var, self.covariate_names):
                raise ValueError(
                    f"group_var {topic_blocks.group_var!r} must not also appear in "
                    f"the prevalence formula (foreground regression would be "
                    f"rank-deficient); remove it from the formula terms.")
        elif doc_group_col is not None:
            raise ValueError("doc_group_col set without topic_blocks.")
```

Add a module-level helper near the top of `stm.py`:

```python
def _formula_mentions(group_var: str, covariate_names: list[str]) -> bool:
    """True if group_var appears as a factor in any design-column name,
    e.g. group_var='source_cohort' matches 'C(source_cohort)[T.dementia]'."""
    needle = f"({group_var})"
    return any(needle in name or name == group_var for name in covariate_names)
```

For Path B (formula resolved at fit time via `_resolve_model_spec_from_pandas`), the `covariate_names` are None at construction; re-run the guard inside `fit` right after `covariate_names` becomes available (after the `if self.covariate_names is None: raise` block):

```python
        if self.topic_blocks is not None and \
                _formula_mentions(self.topic_blocks.group_var, self.covariate_names):
            raise ValueError(
                f"group_var {self.topic_blocks.group_var!r} must not also appear "
                f"in the prevalence formula.")
```

In `fit`, pass `topic_blocks` to the model and thread the group column. Change the `OnlineSTM(...)` construction to add `topic_blocks=self.topic_blocks`. Change the RDD build:

```python
        features_col = self.features_col
        covariates_col = self.covariates_col
        group_col = self.doc_group_col
        select_cols = [features_col, covariates_col]
        if group_col is not None:
            select_cols.append(group_col)
        rdd = (
            dataset.select(*select_cols).rdd
            .map(lambda row: _vector_to_stm_document(
                {c: row[i] for i, c in enumerate(select_cols)},
                features_col=features_col,
                covariates_col=covariates_col,
                group_col=group_col,
            ))
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd spark-vi && python -m pytest tests/test_mllib_stm.py -k "group_var or group_col or gated_fit" -v`
Expected: PASS (3 tests). Then full shim suite: `python -m pytest tests/test_mllib_stm.py tests/test_mllib_stm_formula.py -v` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/mllib/topic/stm.py spark-vi/tests/test_mllib_stm.py
git commit -m "$(printf 'feat(stm): StreamingSTM topic_blocks + doc_group_col wiring + formula guard\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 7: Persist the partition in STMModel

**Files:**
- Modify: `spark-vi/spark_vi/mllib/topic/stm.py` (`StreamingSTM.fit` return, `STMModel.__init__/save/load`)
- Test: `spark-vi/tests/test_mllib_stm_persistence.py`

**Interfaces:**
- Consumes: `TopicBlockPartition.to_dict()/from_dict()` (Task 1).
- Produces: `STMModel` carries `topic_blocks: TopicBlockPartition | None`; persisted into `metadata["topic_block_spec"]` (the partition's `to_dict()`), reconstructed on `load`. No new sidecar file — it rides in `manifest.json` metadata.

- [ ] **Step 1: Write the failing test**

```python
# spark-vi/tests/test_mllib_stm_persistence.py  (append)
def test_stmmodel_roundtrips_topic_block_spec(tmp_path):
    import numpy as np
    from spark_vi.mllib.topic.stm import STMModel
    from spark_vi.models.topic.partition import TopicBlockPartition
    part = TopicBlockPartition("source_cohort", background_k=2, foreground=(("cancer", 1),))
    model = STMModel(
        global_params={"lambda": np.ones((3, 4)), "eta": np.array(0.3),
                       "Gamma": np.zeros((2, 3)), "Sigma": np.ones(3)},
        metadata={"topic_block_spec": part.to_dict()},
        model_spec=None, covariate_names=["Intercept", "age"],
        topic_blocks=part)
    model.save(tmp_path)
    loaded = STMModel.load(tmp_path)
    assert loaded.topic_blocks == part
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd spark-vi && python -m pytest tests/test_mllib_stm_persistence.py -k topic_block_spec -v`
Expected: FAIL — `STMModel.__init__` has no `topic_blocks` kwarg.

- [ ] **Step 3: Implement**

In `STMModel.__init__`, add `topic_blocks=None` param and store `self.topic_blocks = topic_blocks`.

In `STMModel.load`, reconstruct from metadata after building `result`:

```python
        spec_dict = result.metadata.get("topic_block_spec")
        topic_blocks = (
            TopicBlockPartition.from_dict(spec_dict) if spec_dict else None)
```

Add the import at the top of `STMModel.load` scope (or module top):
```python
        from spark_vi.models.topic.partition import TopicBlockPartition
```
and pass `topic_blocks=topic_blocks` into the returned `cls(...)`.

In `StreamingSTM.fit`'s `return STMModel(...)`, add `topic_blocks=self.topic_blocks` and, so the spec persists even if the driver does not set it, ensure the metadata carries it:

```python
        metadata = dict(result.metadata)
        if self.topic_blocks is not None:
            metadata.setdefault("topic_block_spec", self.topic_blocks.to_dict())
        return STMModel(
            global_params=result.global_params,
            metadata=metadata,
            model_spec=getattr(self, "model_spec", None),
            covariate_names=list(self.covariate_names),
            n_iterations=result.n_iterations,
            elbo_trace=list(result.elbo_trace),
            converged=result.converged,
            diagnostic_traces=dict(result.diagnostic_traces),
            topic_blocks=self.topic_blocks,
        )
```

(`save` needs no change — `topic_block_spec` is already in `metadata`, which `save_result` persists into `manifest.json`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd spark-vi && python -m pytest tests/test_mllib_stm_persistence.py -v`
Expected: PASS (all, including the new one).

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/mllib/topic/stm.py spark-vi/tests/test_mllib_stm_persistence.py
git commit -m "$(printf 'feat(stm): persist TopicBlockPartition in STMModel metadata\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 8: Driver wiring (stm_bigquery_cloud.py)

**Files:**
- Modify: `analysis/cloud/stm_bigquery_cloud.py` (`parse_args`, `main`)
- Test: manual smoke via `--help`; the unit-tested logic is the partition build, factored into a helper with a test.

**Interfaces:**
- Consumes: `TopicBlockPartition` (Task 1), `StreamingSTM(topic_blocks=, doc_group_col=)` (Task 6).
- Produces: new CLI flags `--background-k` (int) and `--foreground` (string `"label:size,label:size"`); a helper `build_topic_block_partition(group_var, background_k, foreground_arg, K) -> TopicBlockPartition | None` returning `None` when `--background-k`/`--foreground` are unset. The driver passes `topic_blocks` + `doc_group_col=group_var` to `StreamingSTM`, and writes `topic_block_spec` into `corpus_manifest`.

- [ ] **Step 1: Write the failing test for the partition-build helper**

```python
# analysis/cloud/tests/test_stm_driver_partition.py  (new; mirror the dir of other cloud tests)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # analysis/cloud on path

from stm_bigquery_cloud import build_topic_block_partition


def test_build_partition_from_cli():
    p = build_topic_block_partition(
        group_var="source_cohort", background_k=30,
        foreground_arg="cancer:10,dementia:10", K=50)
    assert p.K == 50
    assert p.groups == ("cancer", "dementia")


def test_build_partition_none_when_unset():
    assert build_topic_block_partition(
        group_var="source_cohort", background_k=None,
        foreground_arg=None, K=40) is None


def test_build_partition_k_mismatch_raises():
    import pytest
    with pytest.raises(ValueError, match="K"):
        build_topic_block_partition(
            group_var="source_cohort", background_k=30,
            foreground_arg="cancer:10", K=50)  # 30+10 != 50
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd analysis/cloud && python -m pytest tests/test_stm_driver_partition.py -v`
Expected: FAIL — `build_topic_block_partition` not defined.

- [ ] **Step 3: Implement**

Add the flags in `parse_args` (after the STM-specific flags block):

```python
    p.add_argument("--background-k", type=int, default=None,
                   help="Gating: size of the shared background topic block. "
                        "Set together with --foreground to enable background/"
                        "foreground gating; unset = canonical STM (no gating).")
    p.add_argument("--foreground", default=None,
                   help="Gating: per-group foreground block sizes as "
                        "'label:size,label:size' (e.g. 'cancer:10,dementia:10'). "
                        "The group labels are values of the gating column "
                        "(--group-var). background_k + sum(sizes) must equal --K.")
    p.add_argument("--group-var", default="source_cohort",
                   help="Gating: document column whose value selects a doc's "
                        "foreground block (default: source_cohort). Must NOT also "
                        "appear in --covariate-formula.")
```

Add the helper at module level:

```python
def build_topic_block_partition(*, group_var, background_k, foreground_arg, K):
    """Build a TopicBlockPartition from CLI args, or None when gating is off.

    foreground_arg is 'label:size,label:size'. Asserts background_k + sum(sizes)
    == K so a misconfigured partition fails before the (expensive) fit.
    """
    if background_k is None and foreground_arg is None:
        return None
    if background_k is None or foreground_arg is None:
        raise ValueError("--background-k and --foreground must be set together.")
    from spark_vi.models.topic.partition import TopicBlockPartition
    fg = []
    for piece in foreground_arg.split(","):
        label, _, size = piece.partition(":")
        fg.append((label.strip(), int(size)))
    part = TopicBlockPartition(group_var=group_var, background_k=int(background_k),
                               foreground=tuple(fg))
    if part.K != K:
        raise ValueError(
            f"gating partition K ({part.K}) != --K ({K}); "
            f"background_k + sum(foreground sizes) must equal --K.")
    return part
```

In `main`, build the partition after corpus load and before the fit, and thread it. After the `composite = "source_cohort" in cat_cols` block, add:

```python
        partition = build_topic_block_partition(
            group_var=args.group_var, background_k=args.background_k,
            foreground_arg=args.foreground, K=args.K)
        if partition is not None and not composite:
            # The group column must exist on the corpus; today only the composite
            # path materializes source_cohort. Require group_var == source_cohort
            # until a non-composite group source is plumbed.
            raise SystemExit(
                "gating requires the composite (source_cohort) corpus; "
                "build the combined cohort or omit --background-k/--foreground.")
        if partition is not None:
            # Spec guard: a foreground block with zero documents is a config
            # error; a thin block is a warning. source_cohort is materialized on
            # bow_df by the composite block above.
            present = {r["source_cohort"] for r in
                       bow_df.select("source_cohort").distinct().collect()}
            counts = {r["source_cohort"]: r["count"] for r in
                      bow_df.groupBy("source_cohort").count().collect()}
            for g in partition.groups:
                if g not in present:
                    raise SystemExit(
                        f"gating group {g!r} has zero documents (source_cohort "
                        f"values present: {sorted(present)}). Fix --foreground "
                        f"labels to match the corpus.")
                if counts.get(g, 0) < 100:
                    print(f"[driver]   WARNING: foreground group {g!r} has only "
                          f"{counts.get(g, 0)} docs; its block may be unstable.",
                          flush=True)
```

In the `StreamingSTM(...)` construction, add:

```python
                topic_blocks=partition,
                doc_group_col=(args.group_var if partition is not None else None),
```

In the metadata-augment block, add to the `corpus_manifest` dict (inside `setdefault`):

```python
                "topic_block_spec": (partition.to_dict() if partition is not None else None),
```

- [ ] **Step 4: Run test + `--help` smoke**

Run: `cd analysis/cloud && python -m pytest tests/test_stm_driver_partition.py -v`
Expected: PASS (3 tests).
Run: `cd analysis/cloud && python stm_bigquery_cloud.py --help 2>&1 | grep -E "background-k|foreground|group-var"`
Expected: the three new flags appear in help text.

- [ ] **Step 5: Commit**

```bash
git add analysis/cloud/stm_bigquery_cloud.py analysis/cloud/tests/test_stm_driver_partition.py
git commit -m "$(printf 'feat(stm): gating CLI flags + topic_block_spec in corpus_manifest\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 9: Experiment runner — gating flags + resume guard

**Files:**
- Modify: `scripts/run_experiment.py` (`build_stm_args`, `_resume_corpus_mismatches`)
- Test: `scripts/tests/test_run_experiment.py`

**Interfaces:**
- Consumes: the driver flags from Task 8.
- Produces: `build_stm_args` emits `--background-k`, `--foreground`, `--group-var` when present in `effective`; `_resume_corpus_mismatches` compares `topic_block_spec` (a changed partition blocks resume).

- [ ] **Step 1: Write the failing tests**

```python
# scripts/tests/test_run_experiment.py  (append)
def test_build_stm_args_includes_gating_flags():
    from run_experiment import build_stm_args
    eff = {
        "source_table": "condition_era", "doc_min_length": 20, "K": 50,
        "max_iter": 20, "vocab_size": 10000, "min_df": 20,
        "min_patient_count": 20, "subsampling_rate": 0.2, "tau0": 64.0,
        "kappa": 0.7, "save_interval": 5, "person_mod": 4,
        "covariate_formula": "~ C(sex) + age", "categorical_cols": ["sex"],
        "continuous_cols": ["age"],
        "background_k": 30, "foreground": "cancer:10,dementia:10",
        "group_var": "source_cohort",
    }
    argv = build_stm_args(eff, "/tmp/out")
    assert "--background-k" in argv and "30" in argv
    assert "--foreground" in argv and "cancer:10,dementia:10" in argv
    assert "--group-var" in argv and "source_cohort" in argv


def test_build_stm_args_omits_gating_when_absent():
    from run_experiment import build_stm_args
    eff = {
        "source_table": "condition_era", "doc_min_length": 20, "K": 40,
        "max_iter": 20, "vocab_size": 10000, "min_df": 20,
        "min_patient_count": 20, "subsampling_rate": 0.2, "tau0": 64.0,
        "kappa": 0.7, "save_interval": 5, "person_mod": 4,
        "covariate_formula": "~ C(sex) + age", "categorical_cols": ["sex"],
        "continuous_cols": ["age"],
    }
    argv = build_stm_args(eff, "/tmp/out")
    assert "--background-k" not in argv and "--foreground" not in argv


def test_resume_mismatch_on_changed_partition():
    from run_experiment import _resume_corpus_mismatches
    ck = {"person_mod": 4, "source_table": "condition_era",
          "topic_block_spec": {"group_var": "source_cohort", "background_k": 30,
                               "foreground": [["cancer", 10], ["dementia", 10]]}}
    eff = {"person_mod": 4, "source_table": "condition_era",
           "background_k": 20, "foreground": "cancer:10,dementia:10",
           "group_var": "source_cohort", "K": 40}
    out = _resume_corpus_mismatches(ck, eff)
    assert any("topic_block_spec" in m for m in out)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd scripts && python -m pytest tests/test_run_experiment.py -k "gating or changed_partition" -v`
Expected: FAIL — gating flags not emitted; `topic_block_spec` not compared.

- [ ] **Step 3: Implement**

In `build_stm_args`, before the final `return`, build the gating flags and append:

```python
    gating: list[str] = []
    if effective.get("background_k") is not None and effective.get("foreground"):
        gating = [
            "--background-k", str(effective["background_k"]),
            "--foreground", str(effective["foreground"]),
            "--group-var", str(effective.get("group_var", "source_cohort")),
        ]
```

Then change the final return to include `gating` before the resume-from suffix:

```python
    return common + [
        "--covariate-formula", str(effective["covariate_formula"]),
        "--categorical-cols", ",".join(effective.get("categorical_cols", [])),
        "--continuous-cols", ",".join(effective.get("continuous_cols", [])),
        "--sigma-init", str(effective.get("sigma_init", 1.0)),
        "--sigma-ridge", str(effective.get("sigma_ridge", 1e-6)),
        "--lbfgs-max-iter", str(effective.get("lbfgs_max_iter", 50)),
        "--lbfgs-tol", str(effective.get("lbfgs_tol", 1e-4)),
    ] + gating + (["--resume-from", str(resume_from)] if resume_from is not None else [])
```

In `_resume_corpus_mismatches`, after the `doc_spec` comparison and before `return mismatches`, add the partition comparison (only when the checkpoint has it — preserving the "fields present in checkpoint" rule):

```python
    if "topic_block_spec" in checkpoint_manifest:
        from spark_vi.models.topic.partition import TopicBlockPartition
        ck_spec = checkpoint_manifest["topic_block_spec"]
        ck_part = TopicBlockPartition.from_dict(ck_spec) if ck_spec else None
        want_part = None
        if effective.get("background_k") is not None and effective.get("foreground"):
            fg = tuple((lbl.strip(), int(sz)) for lbl, _, sz in
                       (piece.partition(":") for piece in
                        str(effective["foreground"]).split(",")))
            want_part = TopicBlockPartition(
                group_var=str(effective.get("group_var", "source_cohort")),
                background_k=int(effective["background_k"]), foreground=fg)
        if ck_part != want_part:
            mismatches.append(
                f"topic_block_spec: checkpoint={ck_part!r} != config={want_part!r}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd scripts && python -m pytest tests/test_run_experiment.py -v`
Expected: PASS (all, including the three new tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/run_experiment.py scripts/tests/test_run_experiment.py
git commit -m "$(printf 'feat(run-exp): STM gating flags + topic_block_spec resume guard\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 10: Foreground-aware NPMI eval

**Files:**
- Modify: `analysis/cloud/eval_coherence_cloud.py`
- Test: `analysis/cloud/tests/test_eval_foreground_split.py` (new; unit-test the pure topic-labeling/reference-selection helper)

**Interfaces:**
- Consumes: `TopicBlockPartition` (Task 1), `topic_block_spec` in `corpus_manifest`.
- Produces: a pure helper `foreground_reference_groups(topic_block_spec) -> dict[int, str | None]` mapping each topic index to its group label (`None` = background, scored on the full corpus; a group label = scored on that group's sub-corpus). The eval driver, when `topic_block_spec` is present, builds a per-group filtered reference (`bow_df` filtered to docs whose `source_cohort` is the group) and scores each foreground topic against its group's reference, background topics against the full corpus, and labels topics by block in the printout.

- [ ] **Step 1: Write the failing test for the helper**

```python
# analysis/cloud/tests/test_eval_foreground_split.py  (new)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_coherence_cloud import foreground_reference_groups


def test_maps_topics_to_groups_or_background():
    spec = {"group_var": "source_cohort", "background_k": 2,
            "foreground": [["cancer", 2], ["dementia", 1]]}
    assert foreground_reference_groups(spec) == {
        0: None, 1: None, 2: "cancer", 3: "cancer", 4: "dementia"}


def test_none_spec_yields_empty_map():
    assert foreground_reference_groups(None) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd analysis/cloud && python -m pytest tests/test_eval_foreground_split.py -v`
Expected: FAIL — `foreground_reference_groups` not defined.

- [ ] **Step 3: Implement the helper + wire the eval**

Add the helper near the top of `eval_coherence_cloud.py` (module level):

```python
def foreground_reference_groups(topic_block_spec):
    """Map topic index -> group label (None = background / full-corpus reference).

    Foreground topics are scored against their group's sub-corpus rather than the
    full corpus: scoring a rare phenotype against majority docs that can never
    contain it triggers the NPMI zero-pair penalty (docs/insights/0007). Returns
    {} when there is no gating partition (all topics scored on the full corpus).
    """
    if not topic_block_spec:
        return {}
    from spark_vi.models.topic.partition import TopicBlockPartition
    part = TopicBlockPartition.from_dict(topic_block_spec)
    labels = part.topic_labels()
    return {k: (None if lbl == "background" else lbl)
            for k, lbl in enumerate(labels)}
```

In `main`, after reading the manifest, read the spec and compute the map (place near the `cohort`/`prior_obs_days` reads):

```python
        topic_block_spec = corpus.get("topic_block_spec")
        fg_groups = foreground_reference_groups(topic_block_spec)
        if fg_groups:
            print(f"[driver]   gating: {sum(v is not None for v in fg_groups.values())} "
                  f"foreground topics scored on group sub-corpora", flush=True)
```

In the NPMI-coherence phase, when `fg_groups` is non-empty, compute coherence per reference set. Replace the single `compute_npmi_coherence` call with a split: background topics on the full `reference_rdd`; each group's foreground topics on a `source_cohort`-filtered reference. The `source_cohort` column is recoverable from `doc_id` exactly as the fit driver does it (`F.split(F.col("doc_id"), ":").getItem(0)`); add it to `bow_df` before building the reference when `fg_groups` is non-empty. Merge the per-topic NPMI arrays back into one report indexed by topic. Concretely, after the existing `topic_term` is computed:

```python
            if fg_groups:
                from pyspark.sql import functions as F
                bow_g = bow_df.withColumn(
                    "source_cohort", F.split(F.col("doc_id"), ":").getItem(0))
                per_topic = np.full(topic_term.shape[0], np.nan)
                # Background topics: full-corpus reference.
                bg_idx = [k for k, g in fg_groups.items() if g is None]
                bg_rep = compute_npmi_coherence(
                    topic_term[bg_idx], reference_rdd, top_n=args.top_n,
                    min_pair_count=args.npmi_min_pair_count)
                for j, k in enumerate(bg_idx):
                    per_topic[k] = bg_rep.per_topic_npmi[j]
                # Foreground topics: per-group sub-corpus reference.
                for g in sorted({v for v in fg_groups.values() if v is not None}):
                    g_idx = [k for k, gg in fg_groups.items() if gg == g]
                    g_ref = (bow_g.where(F.col("source_cohort") == g)
                             .rdd.map(BOWDocument.from_spark_row))
                    g_rep = compute_npmi_coherence(
                        topic_term[g_idx], g_ref, top_n=args.top_n,
                        min_pair_count=args.npmi_min_pair_count)
                    for j, k in enumerate(g_idx):
                        per_topic[k] = g_rep.per_topic_npmi[j]
                print("[driver]   per-topic NPMI (block-aware reference):", flush=True)
                labels = foreground_reference_groups(topic_block_spec)
                for k in range(topic_term.shape[0]):
                    blk = "background" if labels[k] is None else labels[k]
                    print(f"[driver]     topic {k:3d} [{blk}] NPMI={per_topic[k]:+.4f}",
                          flush=True)
                print("[driver] EVAL COHERENCE CLOUD PASSED", flush=True)
                return 0
```

(Leave the existing non-gated path — the `report = compute_npmi_coherence(...)` + `print_ranked_report` block — untouched for `fg_groups == {}`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd analysis/cloud && python -m pytest tests/test_eval_foreground_split.py -v`
Expected: PASS (2 tests). Confirm the module still imports cleanly: `python -c "import eval_coherence_cloud"` — Expected: no error.

- [ ] **Step 5: Commit**

```bash
git add analysis/cloud/eval_coherence_cloud.py analysis/cloud/tests/test_eval_foreground_split.py
git commit -m "$(printf 'feat(eval): foreground-aware NPMI (group sub-corpus references)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 11: ADR for the hard-masking decision

**Files:**
- Create: `docs/decisions/00NN-gated-stm-hard-masking.md` (use the next unused number; check `ls docs/decisions/`)

**Interfaces:** none (documentation).

- [ ] **Step 1: Determine the ADR number**

Run: `ls docs/decisions/ | sort | tail -3`
Use the next integer after the highest existing ADR number for `00NN`.

- [ ] **Step 2: Write the ADR**

Content (fill `NN` and the date `2026-06-23`; match the existing ADR file's front-matter style — check one neighbor for the exact header format before writing):

```markdown
# ADR 00NN: Gated STM uses hard masking for background/foreground blocks

**Status:** Accepted
**Date:** 2026-06-23

## Context

Prevalence-only STM gives a rare group prevalence fidelity but not content
fidelity (docs/insights/0026). To give a rare group its own topic *content*
while borrowing the large cohort's strength for shared structure, we add an
opt-in background/foreground topic-block partition. Three mechanisms were
considered for enforcing that only a group's documents express its foreground
topics.

## Decision

Use **hard masking** (approach A): a document's allowed-topic set is
`background union (foreground blocks of its groups)`; per-document inference
optimizes eta only over the allowed set, so disallowed topics have theta exactly
0 and contribute zero sufficient statistics. The M-step is block-aware
(per-block Gamma normal equations, per-topic Sigma divisor). The canonical
no-gating path reduces to an implicit all-background partition, numerically
identical to prior STM.

## Alternatives considered

- **B — soft prevalence prior.** An informative Gaussian prior on Gamma drives
  majority foreground prevalence toward 0, so gating emerges rather than being
  imposed. Smallest engine change and a fully continuous/joint fit, which may
  have its own benefits (no hard structural zeros; the model can let a near-group
  document borrow a foreground topic when the data strongly support it). Rejected
  for v1 because isolation is soft (mild foreground contamination) and it adds a
  prior-strength tuning knob, but explicitly retained as a future option.
- **C — two-pass freeze-background.** Fit background on the full corpus, freeze
  beta, fit foreground on the group residual. Simplest math; rejected because it
  is not a joint fit (background cannot adapt) and adds two-stage checkpoint
  management.

## Consequences

- The group variable must NOT appear in the prevalence formula: within a
  foreground block's group-only document subset the group indicator is constant,
  so the foreground regression would be rank-deficient (only ridge-rescued,
  uninterpretable). Enforced by a guard.
- Group-shifted *background* prevalence (a legitimate separate effect) is NOT
  available via the gating variable; it would require a distinct covariate, a
  possible future extension.
- Foreground content fidelity comes from gating, not from covariate-dependent
  beta; content covariates (SAGE) remain a separate, unbuilt extension.
```

- [ ] **Step 3: Commit**

```bash
git add docs/decisions/00NN-gated-stm-hard-masking.md
git commit -m "$(printf 'docs(adr): gated STM hard-masking decision (A over B/C)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 12: Experiment config + writeup

**Files:**
- Create/Modify: experiment definition under `experiments/` (match the existing experiment-definition format — inspect a neighbor such as `experiments/0003-*.yaml` or the defaults dir first).
- Create: `docs/experiments/00NN-gated-stm-cancer-dementia.md` (next experiment number; check `ls docs/experiments/`).

**Interfaces:** consumes the gating keys (`background_k`, `foreground`, `group_var`) read by `build_stm_args` (Task 9).

- [ ] **Step 1: Inspect the existing experiment format**

Run: `ls experiments/ docs/experiments/ && sed -n '1,40p' docs/experiments/0003-stm-cancer-dementia.md`
Note the YAML key layout and frontmatter so the new files match conventions exactly.

- [ ] **Step 2: Write the experiment definition**

Create an experiment that reuses the `cancer_or_dementia` combined cohort with gating enabled. The defining additions over experiment 0003 (copy 0003's other keys verbatim — same `source_table`, `doc_unit`, `person_mod`, `prior_obs_days`, `covariate_formula` MINUS `C(source_cohort)`):

```yaml
# gating: background block + per-group foreground blocks
K: 50
background_k: 30
foreground: "cancer:10,dementia:10"
group_var: source_cohort
# source_cohort is the gating variable, so it must NOT be in the formula:
covariate_formula: "~ C(sex) + age"
categorical_cols: ["sex"]
continuous_cols: ["age"]
```

- [ ] **Step 3: Write the experiment doc**

`docs/experiments/00NN-gated-stm-cancer-dementia.md` — frontmatter matching the 0003 doc, body stating: goal (validate that the dementia foreground surfaces a dementia-distinctive phenotype prevalence-only STM could not, per insight 0026), the partition (30 background / 10 cancer / 10 dementia), the success criterion (criterion 4 of the spec), and a note that `source_cohort` is the gating variable and therefore absent from the formula.

- [ ] **Step 4: Validate the config parses**

Run the experiment runner in dry/validate mode if available (inspect `python scripts/run_experiment.py --help` for a `--dry-run`/`--print-args` flag; if present, run it for the new experiment id and confirm the gating flags appear in the built argv). If no such flag exists, run:
`cd scripts && python -c "from run_experiment import build_stm_args; import yaml; print('--background-k' in build_stm_args(yaml.safe_load(open('../experiments/<new>.yaml')), '/tmp/o'))"`
Expected: `True`.

- [ ] **Step 5: Commit**

```bash
git add experiments/ docs/experiments/00NN-gated-stm-cancer-dementia.md
git commit -m "$(printf 'data(exp): gated STM cancer/dementia experiment (30bg/10/10)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Final verification

- [ ] Run the full spark-vi STM suite + the cloud/scripts unit tests:

```bash
cd spark-vi && python -m pytest tests/test_topic_block_partition.py tests/test_stm_math.py tests/test_stm_contract.py tests/test_stm_integration.py tests/test_mllib_stm.py tests/test_mllib_stm_formula.py tests/test_mllib_stm_persistence.py -v
cd ../analysis/cloud && python -m pytest tests/test_stm_driver_partition.py tests/test_eval_foreground_split.py -v
cd ../../scripts && python -m pytest tests/test_run_experiment.py -v
```
Expected: all PASS.

- [ ] Confirm canonical preservation explicitly: `test_canonical_collapse_partition_none_matches_baseline` PASSES (the byte-identical guarantee).

- [ ] The real-cohort fit (running the Task 12 experiment on the cluster) is the empirical validation and is performed by the user after this plan lands, not as a plan step.
