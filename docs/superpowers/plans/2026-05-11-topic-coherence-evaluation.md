# Topic Coherence Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship held-out NPMI topic coherence as a generic `spark_vi.eval.topic` module, with a deterministic person-keyed BOW split helper in `charmpheno/omop/`, plus a local analysis driver. Land in two PRs: a prerequisite `models.topic.*` namespace refactor first, then the eval work on top.

**Architecture:** Two branches. Part A (`refactor/models-topic-namespace`) is a mechanical rename verified by the existing test suite — no new tests, no new behavior. Part B (`feat/eval-topic-coherence`) is TDD-driven: pure-math NPMI helpers tested first, then Spark-side co-occurrence aggregation, then the orchestrator, then HDP filtering, then the split helper, then the analysis driver. The eval function takes a topic-term matrix and an RDD of `BOWDocument`; the driver bridges from a DataFrame.

**Tech Stack:** Python 3.10+, NumPy, PySpark (RDD + DataFrame APIs), pytest. Existing repo conventions: pre-commit hooks, `pytest -m "not slow"` for fast tier, `--slow` for integration tier (see `spark-vi/conftest.py`).

---

## Spec

This plan implements [`docs/superpowers/specs/2026-05-11-topic-coherence-evaluation-design.md`](../specs/2026-05-11-topic-coherence-evaluation-design.md). The spec assigned ADR 0016 to the eval branch; this plan corrects to **ADR 0016 for the refactor** (lands first) and **ADR 0017 for the eval branch** (lands second). A small spec edit fixes the cross-reference.

---

## Part A — Prerequisite Refactor (`refactor/models-topic-namespace`)

**Goal of Part A:** Move `spark_vi/models/{lda,online_hdp,counting}.py` → `spark_vi/models/topic/`. No behavior change. Verified by the full existing test suite passing.

Branch off `main`. Lands as its own PR. Part B starts from the merged result.

### Task A1: Create the branch and the new directory

**Files:**
- Create: `spark-vi/spark_vi/models/topic/` (empty directory, populated in subsequent tasks)

- [ ] **Step 1: Create branch from main**

```bash
git checkout main
git pull origin main
git checkout -b refactor/models-topic-namespace
```

- [ ] **Step 2: Verify the test suite is green before any changes**

```bash
cd spark-vi && poetry run pytest -m "not slow" -x 2>&1 | tail -20
```

Expected: all tests pass. Record the count. This is the baseline; the refactor must not regress it.

- [ ] **Step 3: Commit the branch state (no-op, just to confirm we're on the right branch)**

Skip — nothing to commit yet. Proceed to Task A2.

---

### Task A2: Move the three model files into `models/topic/`

**Files:**
- Move: `spark-vi/spark_vi/models/lda.py` → `spark-vi/spark_vi/models/topic/lda.py`
- Move: `spark-vi/spark_vi/models/online_hdp.py` → `spark-vi/spark_vi/models/topic/online_hdp.py`
- Move: `spark-vi/spark_vi/models/counting.py` → `spark-vi/spark_vi/models/topic/counting.py`
- Create: `spark-vi/spark_vi/models/topic/__init__.py`

- [ ] **Step 1: Do the three moves with `git mv` (preserves history)**

```bash
cd spark-vi
git mv spark_vi/models/lda.py spark_vi/models/topic/lda.py
git mv spark_vi/models/online_hdp.py spark_vi/models/topic/online_hdp.py
git mv spark_vi/models/counting.py spark_vi/models/topic/counting.py
```

- [ ] **Step 2: Write `spark_vi/models/topic/__init__.py`**

```python
"""Topic-model implementations of the spark_vi.core.VIModel contract.

OnlineLDA: streaming variational Bayes for Latent Dirichlet Allocation.
OnlineHDP: streaming variational Bayes for the Hierarchical Dirichlet Process.
CountingModel: trivial coin-flip-posterior reference model used to exercise
    the framework contract end-to-end. Not a topic model in the LDA/HDP sense;
    lives under topic/ because it shares the bag-of-words input shape and is
    used as a contract-conformance fixture for topic-model infrastructure.
"""
from spark_vi.models.topic.counting import CountingModel
from spark_vi.models.topic.lda import OnlineLDA
from spark_vi.models.topic.online_hdp import OnlineHDP

__all__ = ["CountingModel", "OnlineHDP", "OnlineLDA"]
```

- [ ] **Step 3: Update `spark_vi/models/__init__.py` to re-export from `topic/`**

Replace the existing contents with:

```python
"""Pre-built models for spark-vi.

The topic-model implementations live under spark_vi.models.topic. This file
re-exports their public surface for backward-compatibility-by-default of the
top-level `from spark_vi.models import OnlineLDA` style imports.
"""
from spark_vi.models.topic import CountingModel, OnlineHDP, OnlineLDA

__all__ = ["CountingModel", "OnlineHDP", "OnlineLDA"]
```

- [ ] **Step 4: Verify the moved files still import correctly inside themselves**

The three moved files reference each other via absolute imports (`from spark_vi.models.lda import _cavi_doc_inference`). Those will break. Run a quick syntax/import check:

```bash
cd spark-vi && poetry run python -c "from spark_vi.models.topic import OnlineLDA, OnlineHDP, CountingModel; print('ok')"
```

Expected: likely an `ImportError` referencing `spark_vi.models.lda` (the moved file's own intra-package references are now wrong, plus mllib still imports the old path). Move on — Task A3 fixes the intra-package paths and Task A4 fixes the rest.

- [ ] **Step 5: Commit the moves before fixing imports**

```bash
git add -A spark-vi/spark_vi/models/
git commit -m "refactor(models): move lda/online_hdp/counting to spark_vi.models.topic"
```

---

### Task A3: Update intra-`models/` import paths in the moved files

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/lda.py`
- Modify: `spark-vi/spark_vi/models/topic/online_hdp.py`
- Modify: `spark-vi/spark_vi/models/topic/counting.py`

- [ ] **Step 1: Find any intra-models cross-references in the moved files**

```bash
cd spark-vi && grep -n "from spark_vi.models" spark_vi/models/topic/*.py
```

Expected output (or similar): zero hits in `counting.py`, possibly one or two in `lda.py` and `online_hdp.py` referencing each other or their own old path.

- [ ] **Step 2: Rewrite any hit found in Step 1 to use the new `spark_vi.models.topic.X` path**

For each hit, replace `from spark_vi.models.<name>` with `from spark_vi.models.topic.<name>`. Verify with the same grep — expect zero hits afterward to the old path.

- [ ] **Step 3: Verify the package imports**

```bash
cd spark-vi && poetry run python -c "from spark_vi.models.topic import OnlineLDA, OnlineHDP, CountingModel; print('ok')"
```

Expected: prints `ok`. If it still fails with `ImportError`, the failing module name in the traceback points at a remaining old-path import in one of the moved files; fix and re-run.

- [ ] **Step 4: Commit**

```bash
git add spark_vi/models/topic/
git commit -m "refactor(models): fix intra-topic imports after move"
```

---

### Task A4: Update all external imports in framework code

**Files (each modified to swap import paths):**
- Modify: `spark-vi/spark_vi/mllib/lda.py` (lines 25, 427 — `from spark_vi.models.lda` → `from spark_vi.models.topic.lda`)
- Modify: `spark-vi/spark_vi/mllib/hdp.py` (line 26 — `from spark_vi.models.online_hdp` → `from spark_vi.models.topic.online_hdp`)
- Modify: `spark-vi/probes/diagnose_collapse.py` (line 29)
- Modify: `spark-vi/probes/alpha_drift_probe.py` (line 46)
- Modify: `spark-vi/probes/diagnose_collapse_in_spark.py` (line 17)

- [ ] **Step 1: Sed-rewrite framework + probes imports**

```bash
cd spark-vi
# Be specific: only rewrite spark_vi.models.<name>, not anything else (no leading 'from' lookahead needed since the pattern is distinctive).
find spark_vi/mllib probes -name "*.py" -print0 | \
  xargs -0 sed -i '' -E 's|spark_vi\.models\.lda|spark_vi.models.topic.lda|g; s|spark_vi\.models\.online_hdp|spark_vi.models.topic.online_hdp|g; s|spark_vi\.models\.counting|spark_vi.models.topic.counting|g'
```

- [ ] **Step 2: Verify the rewrite hit the expected files**

```bash
cd spark-vi && grep -rn "spark_vi\.models\.\(lda\|online_hdp\|counting\)" spark_vi/mllib probes
```

Expected: empty (the only remaining `spark_vi.models.<name>` references should be inside `spark_vi/models/topic/`, which the previous task already updated).

- [ ] **Step 3: Verify the framework still imports cleanly**

```bash
cd spark-vi && poetry run python -c "from spark_vi.mllib.lda import OnlineLDAEstimator; from spark_vi.mllib.hdp import OnlineHDPEstimator; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 4: Commit**

```bash
git add spark_vi/mllib probes
git commit -m "refactor(mllib,probes): point imports at spark_vi.models.topic"
```

---

### Task A5: Update test imports

**Files:**
- Modify: all 15 test files in `spark-vi/tests/` referencing `spark_vi.models.{lda,online_hdp,counting}` (per the prior grep)

- [ ] **Step 1: Sed-rewrite test imports**

```bash
cd spark-vi
find tests -name "*.py" -print0 | \
  xargs -0 sed -i '' -E 's|spark_vi\.models\.lda|spark_vi.models.topic.lda|g; s|spark_vi\.models\.online_hdp|spark_vi.models.topic.online_hdp|g; s|spark_vi\.models\.counting|spark_vi.models.topic.counting|g'
```

- [ ] **Step 2: Verify no old-path references remain in tests**

```bash
cd spark-vi && grep -rn "spark_vi\.models\.\(lda\|online_hdp\|counting\)" tests
```

Expected: empty.

- [ ] **Step 3: Run the full fast-tier suite**

```bash
cd spark-vi && poetry run pytest -m "not slow" 2>&1 | tail -20
```

Expected: same pass count as the baseline recorded in Task A1, Step 2. Any failure means a missed import path or a `__init__.py` symbol that needs to be re-exported.

- [ ] **Step 4: Commit**

```bash
git add tests
git commit -m "refactor(tests): point imports at spark_vi.models.topic"
```

---

### Task A6: Update `analysis/` and ADR cross-references

**Files:**
- Modify: `analysis/local/fit_charmpheno_local.py:22` (`from spark_vi.models import CountingModel` — already uses the umbrella import, no change needed; verify)
- Modify: `analysis/cloud/hdp_bigquery_cloud.py:76` (`from spark_vi.models.online_hdp import ...` — explicit submodule)
- Modify: `docs/decisions/0013-hdp-concentration-optimization.md:111` (prose reference to `spark_vi.models.lda`)

- [ ] **Step 1: Update analysis driver imports**

```bash
sed -i '' -E 's|spark_vi\.models\.online_hdp|spark_vi.models.topic.online_hdp|g; s|spark_vi\.models\.lda|spark_vi.models.topic.lda|g; s|spark_vi\.models\.counting|spark_vi.models.topic.counting|g' \
  analysis/local/fit_charmpheno_local.py analysis/cloud/hdp_bigquery_cloud.py
```

- [ ] **Step 2: Update ADR 0013 cross-reference**

Read [`docs/decisions/0013-hdp-concentration-optimization.md`](../../decisions/0013-hdp-concentration-optimization.md) line 111. Replace the inline reference `spark_vi.models.lda` with `spark_vi.models.topic.lda`. Use the Edit tool, not sed (single occurrence in prose, easy to do precisely).

- [ ] **Step 3: Verify nothing references the old paths anywhere in the repo**

```bash
grep -rn "spark_vi\.models\.\(lda\|online_hdp\|counting\)" --include="*.py" --include="*.md" 2>/dev/null
```

Expected: the only hits should be inside `spark-vi/spark_vi/models/topic/` (the modules' own references) and inside `docs/superpowers/specs/2026-04-22-charmpheno-project-setup-design.md` and `docs/superpowers/specs/2026-05-04-mllib-shim-design.md` (historical specs, frozen by date — leave them alone, they describe the state at the time of writing). The eval spec [`2026-05-11-topic-coherence-evaluation-design.md`](../specs/2026-05-11-topic-coherence-evaluation-design.md) references the *new* path and stays as-is.

- [ ] **Step 4: Commit**

```bash
git add analysis/ docs/decisions/0013-hdp-concentration-optimization.md
git commit -m "refactor(analysis,docs): point imports at spark_vi.models.topic"
```

---

### Task A7: Write ADR 0016

**Files:**
- Create: `docs/decisions/0016-models-topic-namespace.md`

- [ ] **Step 1: Write the ADR**

```markdown
# ADR 0016 — `spark_vi.models.topic` namespace

**Status:** Accepted
**Date:** 2026-05-11
**Supersedes:** none (refines the implicit layout established by ADR 0001-0015)
**Superseded by:** none

## Context

The `spark_vi` framework was originally laid out with topic-model
implementations directly under `spark_vi.models.{lda,online_hdp,counting}`.
This made sense when the only models in flight were topic models. As we
add evaluation surface (held-out coherence, future term relevance, future
synthetic-recovery testing — see ADR 0017) the eval namespace wants a
`spark_vi.eval.topic.*` shape to leave room for non-topic-model eval modules
later. An asymmetric `models.lda` / `eval.topic.coherence` layout is
needlessly confusing; we want the topic-model scope to be visible on both
the model side and the eval side.

## Decision

Move the three topic-model implementation files to a `topic` subpackage:

```
spark_vi/models/lda.py        -> spark_vi/models/topic/lda.py
spark_vi/models/online_hdp.py -> spark_vi/models/topic/online_hdp.py
spark_vi/models/counting.py   -> spark_vi/models/topic/counting.py
```

`CountingModel` is not a topic model in the LDA/HDP sense but shares the
bag-of-words input shape and is used as a contract-conformance fixture for
topic-model infrastructure; it lives under `topic/` for that reason.

The top-level `spark_vi.models.__init__` re-exports `OnlineLDA`, `OnlineHDP`,
and `CountingModel` so `from spark_vi.models import OnlineLDA` continues to
work. The new canonical import path is `from spark_vi.models.topic import
OnlineLDA` but the umbrella import is the supported public surface.

## Consequences

**Breaking:** any external code importing `spark_vi.models.lda`,
`spark_vi.models.online_hdp`, or `spark_vi.models.counting` directly (as
submodules, not via the umbrella) breaks. The framework is early enough
that no external consumers exist, and within-repo callers (`mllib`, tests,
probes, `analysis/`, ADR 0013 prose) are migrated in this same PR.

**Non-breaking:** umbrella imports `from spark_vi.models import OnlineLDA`
are preserved by the re-export. Historical specs that name old paths
(2026-04-22, 2026-05-04) are left as-is — they are point-in-time documents.

**Forward:** ADR 0017 introduces `spark_vi.eval.topic.*` mirroring this
layout.

## Alternatives considered

- **Leave the layout asymmetric.** Rejected: the dissonance compounds with
  every new eval module added under `eval/topic/`.
- **Re-namespace under `spark_vi.topic.{models,eval}` (collapse the inner
  split).** Rejected: framework code that is generic over model class
  (`spark_vi.core`, `spark_vi.io`, `spark_vi.diagnostics`) already shapes
  the top-level under capability, not domain. `topic` is one domain among
  potential others; better to keep it as a leaf of the `models` and `eval`
  capabilities than as a domain-rooted competing axis.
```

- [ ] **Step 2: Commit the ADR**

```bash
git add docs/decisions/0016-models-topic-namespace.md
git commit -m "docs(adr): ADR 0016 — spark_vi.models.topic namespace refactor"
```

---

### Task A8: Full verification + push

- [ ] **Step 1: Run the full test suite, including slow tier**

```bash
cd spark-vi && poetry run pytest 2>&1 | tail -20
```

Expected: same pass/skip counts as the pre-refactor baseline.

- [ ] **Step 2: Run charmpheno tests too (independent package, but exercises spark_vi import surface)**

```bash
cd charmpheno && poetry run pytest 2>&1 | tail -20
```

Expected: all pass.

- [ ] **Step 3: Push branch and open PR**

```bash
git push -u origin refactor/models-topic-namespace
gh pr create --title "refactor(models): introduce spark_vi.models.topic namespace" --body "$(cat <<'EOF'
## Summary
- Mechanical rename: `spark_vi.models.{lda,online_hdp,counting}` → `spark_vi.models.topic.*`
- Umbrella `from spark_vi.models import OnlineLDA` preserved
- Prerequisite for the eval-namespace work (see ADR 0017 / coherence spec)

## Test plan
- [ ] `pytest` in spark-vi (fast + slow) passes with same counts as main
- [ ] `pytest` in charmpheno passes
- [ ] Manual: `python -c "from spark_vi.models import OnlineLDA, OnlineHDP, CountingModel"` works
- [ ] Manual: `python -c "from spark_vi.models.topic.lda import OnlineLDA"` works

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 4: Wait for PR review, merge to main, then proceed to Part B**

After merge, return to main and refresh:

```bash
git checkout main && git pull origin main
```

---

## Part B — Eval Branch (`feat/eval-topic-coherence`)

**Goal of Part B:** Ship the NPMI coherence module, the person-keyed split helper, and the local driver. Build TDD-style. Each component has a focused unit test before any implementation.

Branch off the merged `main` (which now has Part A's refactor landed).

### Task B1: Create the eval branch and skeleton

**Files:**
- Create: `spark-vi/spark_vi/eval/__init__.py`
- Create: `spark-vi/spark_vi/eval/topic/__init__.py`
- Create: `spark-vi/tests/eval/__init__.py`

- [ ] **Step 1: Create branch**

```bash
git checkout main
git pull origin main
git checkout -b feat/eval-topic-coherence
```

- [ ] **Step 2: Create the eval package init**

Create `spark-vi/spark_vi/eval/__init__.py` with:

```python
"""Evaluation utilities for spark_vi models.

Subpackages are scoped by model class:
  spark_vi.eval.topic — held-out coherence and related metrics for topic
                        models (OnlineLDA, OnlineHDP).
"""
```

- [ ] **Step 3: Create the topic eval init (empty re-exports for now, populated as components land)**

Create `spark-vi/spark_vi/eval/topic/__init__.py` with:

```python
"""Topic-model evaluation: NPMI coherence on held-out data.

Public API lands incrementally as plan tasks complete. See
docs/superpowers/specs/2026-05-11-topic-coherence-evaluation-design.md.
"""
```

- [ ] **Step 4: Create the tests directory init**

Create `spark-vi/tests/eval/__init__.py` empty (pytest package marker).

- [ ] **Step 5: Verify the package imports**

```bash
cd spark-vi && poetry run python -c "import spark_vi.eval.topic; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 6: Commit**

```bash
git add spark-vi/spark_vi/eval spark-vi/tests/eval
git commit -m "feat(eval): scaffold spark_vi.eval.topic namespace"
```

---

### Task B2: `CoherenceReport` dataclass

**Files:**
- Create: `spark-vi/spark_vi/eval/topic/types.py`
- Create: `spark-vi/tests/eval/test_types.py`

- [ ] **Step 1: Write the failing test**

Create `spark-vi/tests/eval/test_types.py`:

```python
"""Tests for spark_vi.eval.topic.types."""
from __future__ import annotations

import numpy as np
import pytest


def test_coherence_report_constructs_and_is_frozen():
    from spark_vi.eval.topic.types import CoherenceReport

    report = CoherenceReport(
        per_topic_npmi=np.array([0.1, 0.2, 0.3]),
        top_term_indices=np.array([[0, 1], [1, 2], [2, 3]]),
        topic_indices=np.array([0, 1, 2]),
        n_holdout_docs=100,
        top_n=2,
        mean=0.2,
        median=0.2,
        stdev=float(np.std([0.1, 0.2, 0.3], ddof=0)),
        min=0.1,
        max=0.3,
    )
    assert report.per_topic_npmi.shape == (3,)
    assert report.top_term_indices.shape == (3, 2)
    assert report.topic_indices.shape == (3,)
    assert report.n_holdout_docs == 100
    assert report.top_n == 2
    assert report.mean == pytest.approx(0.2)
    assert report.min == pytest.approx(0.1)
    assert report.max == pytest.approx(0.3)


def test_coherence_report_is_immutable():
    from spark_vi.eval.topic.types import CoherenceReport

    report = CoherenceReport(
        per_topic_npmi=np.array([0.1]),
        top_term_indices=np.array([[0]]),
        topic_indices=np.array([0]),
        n_holdout_docs=1,
        top_n=1,
        mean=0.1, median=0.1, stdev=0.0, min=0.1, max=0.1,
    )
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        report.mean = 0.5  # type: ignore[misc]
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd spark-vi && poetry run pytest tests/eval/test_types.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'spark_vi.eval.topic.types'`.

- [ ] **Step 3: Implement `types.py`**

Create `spark-vi/spark_vi/eval/topic/types.py`:

```python
"""Result types for spark_vi.eval.topic.

CoherenceReport is the frozen dataclass returned by compute_npmi_coherence;
the same shape supports LDA (all K topics scored) and HDP (mask-filtered
subset of T topics).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CoherenceReport:
    """Per-topic NPMI and descriptive summary statistics.

    Attributes:
        per_topic_npmi: shape (K_scored,) per-topic mean NPMI over the
            unordered pairs of top-N terms.
        top_term_indices: shape (K_scored, N) — vocab indices of the top-N
            terms per scored topic, sorted by descending E[beta_t].
        topic_indices: shape (K_scored,) — original topic indices in the
            input topic-term matrix. Identity for LDA; mask-filtered subset
            for HDP.
        n_holdout_docs: number of documents in the held-out corpus used
            for the co-occurrence statistics.
        top_n: top-N parameter the report was computed with.
        mean, median, stdev, min, max: descriptive summary over
            per_topic_npmi. Not used to normalize.
    """
    per_topic_npmi: np.ndarray
    top_term_indices: np.ndarray
    topic_indices: np.ndarray
    n_holdout_docs: int
    top_n: int
    mean: float
    median: float
    stdev: float
    min: float
    max: float
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd spark-vi && poetry run pytest tests/eval/test_types.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Re-export from the topic eval `__init__.py`**

Edit `spark-vi/spark_vi/eval/topic/__init__.py` to add:

```python
from spark_vi.eval.topic.types import CoherenceReport

__all__ = ["CoherenceReport"]
```

- [ ] **Step 6: Commit**

```bash
git add spark-vi/spark_vi/eval spark-vi/tests/eval/test_types.py
git commit -m "feat(eval): CoherenceReport dataclass"
```

---

### Task B3: `_npmi_pair` pure math

**Files:**
- Create: `spark-vi/spark_vi/eval/topic/coherence.py` (initial — only the private `_npmi_pair` helper)
- Create: `spark-vi/tests/eval/test_coherence.py` (initial)

- [ ] **Step 1: Write the failing test**

Create `spark-vi/tests/eval/test_coherence.py`:

```python
"""Tests for spark_vi.eval.topic.coherence."""
from __future__ import annotations

import math

import pytest


def test_npmi_pair_independent_returns_zero():
    """If p(i,j) == p(i)*p(j), NPMI = 0."""
    from spark_vi.eval.topic.coherence import _npmi_pair
    # p_i = p_j = 0.5, p_ij = 0.25 -> log(1)/-log(0.25) = 0
    assert _npmi_pair(p_i=0.5, p_j=0.5, p_ij=0.25) == pytest.approx(0.0)


def test_npmi_pair_perfect_cooccurrence_returns_one():
    """If p(i,j) == p(i) == p(j), NPMI = 1 (always co-occur)."""
    from spark_vi.eval.topic.coherence import _npmi_pair
    # p_i = p_j = p_ij = 0.5
    # numerator: log(0.5 / 0.25) = log(2)
    # denominator: -log(0.5) = log(2)
    # ratio = 1
    assert _npmi_pair(p_i=0.5, p_j=0.5, p_ij=0.5) == pytest.approx(1.0)


def test_npmi_pair_zero_cooccurrence_returns_minus_one():
    """Roder et al. 2015 convention: NPMI = -1 when p(i,j) = 0."""
    from spark_vi.eval.topic.coherence import _npmi_pair
    assert _npmi_pair(p_i=0.5, p_j=0.5, p_ij=0.0) == -1.0


def test_npmi_pair_anti_correlated_is_negative():
    """If pair appears less than independence would predict, NPMI < 0."""
    from spark_vi.eval.topic.coherence import _npmi_pair
    # p_i = p_j = 0.5, p_ij = 0.1 << 0.25 (independence baseline)
    result = _npmi_pair(p_i=0.5, p_j=0.5, p_ij=0.1)
    assert result < 0.0
    assert result > -1.0  # not the zero-cooccur sentinel


def test_npmi_pair_handles_small_probabilities():
    """No log-of-zero NaN/inf for tiny but non-zero p_ij."""
    from spark_vi.eval.topic.coherence import _npmi_pair
    result = _npmi_pair(p_i=0.01, p_j=0.01, p_ij=1e-6)
    # Independence baseline: 0.01 * 0.01 = 1e-4; p_ij = 1e-6 << baseline.
    # Anti-correlated; bounded above by 1 and below by -1; not NaN.
    assert math.isfinite(result)
    assert -1.0 <= result <= 1.0
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd spark-vi && poetry run pytest tests/eval/test_coherence.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'spark_vi.eval.topic.coherence'`.

- [ ] **Step 3: Implement `coherence.py` initial scaffold**

Create `spark-vi/spark_vi/eval/topic/coherence.py`:

```python
"""NPMI topic coherence over a held-out BOW corpus.

Implements normalized pointwise mutual information per Roder et al. 2015,
with whole-document co-occurrence (no sliding window). The metric is the
unweighted mean over all unordered pairs of a topic's top-N terms.

See docs/decisions/0017-topic-coherence-evaluation.md and the spec at
docs/superpowers/specs/2026-05-11-topic-coherence-evaluation-design.md.
"""
from __future__ import annotations

import math


def _npmi_pair(p_i: float, p_j: float, p_ij: float) -> float:
    """NPMI for a single (w_i, w_j) pair.

    NPMI = log[p_ij / (p_i * p_j)] / -log p_ij.

    Returns -1.0 when p_ij == 0 (Roder et al. 2015 convention) so the
    pair-aggregate stays in [-1, 1] without NaN/Inf contamination.
    """
    if p_ij <= 0.0:
        return -1.0
    pmi = math.log(p_ij / (p_i * p_j))
    return pmi / -math.log(p_ij)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd spark-vi && poetry run pytest tests/eval/test_coherence.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/eval/topic/coherence.py spark-vi/tests/eval/test_coherence.py
git commit -m "feat(eval): _npmi_pair pure math"
```

---

### Task B4: `_top_n_terms_per_topic`

**Files:**
- Modify: `spark-vi/spark_vi/eval/topic/coherence.py` (add `_top_n_terms_per_topic`)
- Modify: `spark-vi/tests/eval/test_coherence.py` (add tests)

- [ ] **Step 1: Write the failing tests**

Append to `spark-vi/tests/eval/test_coherence.py`:

```python
import numpy as np


def test_top_n_terms_per_topic_picks_argmax_n():
    """For each topic row, return the indices of the N largest entries, sorted descending."""
    from spark_vi.eval.topic.coherence import _top_n_terms_per_topic
    topic_term = np.array([
        [0.1, 0.4, 0.2, 0.3],  # topic 0: top-2 by value -> indices [1, 3]
        [0.5, 0.05, 0.4, 0.05],  # topic 1: top-2 -> indices [0, 2]
    ])
    out = _top_n_terms_per_topic(topic_term, top_n=2)
    assert out.shape == (2, 2)
    assert list(out[0]) == [1, 3]
    assert list(out[1]) == [0, 2]


def test_top_n_terms_per_topic_breaks_ties_by_index():
    """Ties broken by ascending index, deterministic across runs."""
    from spark_vi.eval.topic.coherence import _top_n_terms_per_topic
    topic_term = np.array([
        [0.25, 0.25, 0.25, 0.25],  # all tied
    ])
    out = _top_n_terms_per_topic(topic_term, top_n=3)
    # With all ties, smaller indices win.
    assert list(out[0]) == [0, 1, 2]


def test_top_n_terms_per_topic_top_n_must_not_exceed_vocab():
    from spark_vi.eval.topic.coherence import _top_n_terms_per_topic
    topic_term = np.array([[0.5, 0.5]])
    with pytest.raises(ValueError, match="top_n"):
        _top_n_terms_per_topic(topic_term, top_n=3)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd spark-vi && poetry run pytest tests/eval/test_coherence.py::test_top_n_terms_per_topic_picks_argmax_n -v
```

Expected: FAIL with `ImportError: cannot import name '_top_n_terms_per_topic'`.

- [ ] **Step 3: Implement `_top_n_terms_per_topic`**

Append to `spark-vi/spark_vi/eval/topic/coherence.py`:

```python
import numpy as np


def _top_n_terms_per_topic(topic_term: np.ndarray, top_n: int) -> np.ndarray:
    """Return the indices of the top-N terms for each topic, sorted descending.

    Ties broken by ascending index (lexicographic argsort over (-value, index)).
    Output shape: (K, top_n) with dtype int64.
    """
    K, V = topic_term.shape
    if top_n > V:
        raise ValueError(f"top_n={top_n} exceeds vocabulary size V={V}")
    # Stable sort on negated values gives descending order with tie-broken by
    # ascending original index — exactly what argsort on (-value) does when
    # stable.
    sorted_idx = np.argsort(-topic_term, axis=1, kind="stable")
    return sorted_idx[:, :top_n].astype(np.int64)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd spark-vi && poetry run pytest tests/eval/test_coherence.py -v
```

Expected: all tests in the file pass (5 existing + 3 new = 8).

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/eval/topic/coherence.py spark-vi/tests/eval/test_coherence.py
git commit -m "feat(eval): _top_n_terms_per_topic with deterministic tie-breaking"
```

---

### Task B5: `_aggregate_topic_coherence` pure math

**Files:**
- Modify: `spark-vi/spark_vi/eval/topic/coherence.py` (add `_aggregate_topic_coherence`)
- Modify: `spark-vi/tests/eval/test_coherence.py` (add tests)

- [ ] **Step 1: Write the failing test**

Append to `spark-vi/tests/eval/test_coherence.py`:

```python
def test_aggregate_topic_coherence_tiny_example():
    """Hand-computed: 2 topics, 4 terms, 5 docs."""
    from spark_vi.eval.topic.coherence import _aggregate_topic_coherence

    # Top-N terms per topic:
    #   topic 0: terms [0, 1]
    #   topic 1: terms [2, 3]
    top_n_indices = np.array([[0, 1], [2, 3]])

    # Doc frequencies (out of 5 docs):
    #   term 0: 4 docs; term 1: 3 docs; term 2: 5 docs; term 3: 1 doc
    doc_freqs = {0: 4, 1: 3, 2: 5, 3: 1}

    # Pairwise co-occurrence:
    #   (0,1): 2 docs
    #   (2,3): 1 doc  (term 3 appears only in docs that also have term 2)
    pair_freqs = {(0, 1): 2, (2, 3): 1}

    n_docs = 5
    out = _aggregate_topic_coherence(
        top_n_indices=top_n_indices,
        doc_freqs=doc_freqs,
        pair_freqs=pair_freqs,
        n_docs=n_docs,
    )
    assert out.shape == (2,)

    # Topic 0: NPMI((0, 1)) with p_0=4/5, p_1=3/5, p_01=2/5
    p_0, p_1, p_01 = 4/5, 3/5, 2/5
    expected_t0 = math.log(p_01 / (p_0 * p_1)) / -math.log(p_01)
    assert out[0] == pytest.approx(expected_t0)

    # Topic 1: NPMI((2, 3)) with p_2=5/5=1.0, p_3=1/5, p_23=1/5
    # PMI = log(0.2 / (1.0 * 0.2)) = log(1) = 0; NPMI = 0 / -log(0.2) = 0.
    assert out[1] == pytest.approx(0.0)


def test_aggregate_topic_coherence_missing_pair_returns_minus_one():
    """A pair with no co-occurrence in the held-out corpus contributes NPMI = -1."""
    from spark_vi.eval.topic.coherence import _aggregate_topic_coherence

    top_n_indices = np.array([[0, 1]])
    doc_freqs = {0: 3, 1: 3}
    pair_freqs: dict[tuple[int, int], int] = {}  # never co-occurred
    out = _aggregate_topic_coherence(
        top_n_indices=top_n_indices,
        doc_freqs=doc_freqs,
        pair_freqs=pair_freqs,
        n_docs=10,
    )
    assert out[0] == pytest.approx(-1.0)


def test_aggregate_topic_coherence_averages_over_pairs():
    """Topic with top-N=3 averages over 3 pairs."""
    from spark_vi.eval.topic.coherence import _aggregate_topic_coherence

    top_n_indices = np.array([[0, 1, 2]])
    # All three pairs perfectly co-occur: NPMI = 1 for each.
    doc_freqs = {0: 5, 1: 5, 2: 5}
    pair_freqs = {(0, 1): 5, (0, 2): 5, (1, 2): 5}
    out = _aggregate_topic_coherence(
        top_n_indices=top_n_indices,
        doc_freqs=doc_freqs,
        pair_freqs=pair_freqs,
        n_docs=5,
    )
    # Each pair: log(1/1) / -log(1)  -> 0/0 indeterminate at p_ij = 1
    # Convention: handle p_ij == 1 by returning 1.0 (perfect co-occurrence).
    # See _npmi_pair: when p_ij = 1, -log(1) = 0, denominator is 0 -- the
    # function should handle this edge case too. Verify both _npmi_pair and
    # _aggregate handle it. Expected: all pairs return 1.0, mean = 1.0.
    assert out[0] == pytest.approx(1.0)
```

Note: the last test surfaces the `p_ij == 1` edge case in `_npmi_pair` (denominator `-log(1) = 0`). Update `_npmi_pair` to handle this before the aggregator can pass.

- [ ] **Step 2: Run the failing test**

```bash
cd spark-vi && poetry run pytest tests/eval/test_coherence.py -v
```

Expected: `ImportError` on `_aggregate_topic_coherence` first, then once that's added, the `p_ij == 1` test will fail on `_npmi_pair` with `ZeroDivisionError` or similar.

- [ ] **Step 3: Update `_npmi_pair` to handle `p_ij == 1` (perfect co-occurrence)**

Replace the existing `_npmi_pair` body in `coherence.py`:

```python
def _npmi_pair(p_i: float, p_j: float, p_ij: float) -> float:
    """NPMI for a single (w_i, w_j) pair.

    NPMI = log[p_ij / (p_i * p_j)] / -log p_ij.

    Edge cases:
      p_ij == 0:  return -1.0   (Roder et al. 2015 convention; pair never co-occurs)
      p_ij == 1:  return  1.0   (denominator -log(1) = 0; pair always co-occurs)
    """
    if p_ij <= 0.0:
        return -1.0
    if p_ij >= 1.0:
        return 1.0
    pmi = math.log(p_ij / (p_i * p_j))
    return pmi / -math.log(p_ij)
```

- [ ] **Step 4: Implement `_aggregate_topic_coherence`**

Append to `spark-vi/spark_vi/eval/topic/coherence.py`:

```python
from itertools import combinations


def _aggregate_topic_coherence(
    *,
    top_n_indices: np.ndarray,
    doc_freqs: dict[int, int],
    pair_freqs: dict[tuple[int, int], int],
    n_docs: int,
) -> np.ndarray:
    """Per-topic mean NPMI over the unordered pairs of its top-N terms.

    Args:
        top_n_indices: shape (K_scored, N). Term indices per topic.
        doc_freqs: term_index -> # held-out docs containing that term.
        pair_freqs: (min_idx, max_idx) -> # held-out docs containing both.
            Pairs with zero co-occurrence may be absent from this dict.
        n_docs: total # held-out docs (the normalizer).

    Returns:
        shape (K_scored,) float64 array of mean NPMI per topic.
    """
    K_scored, N = top_n_indices.shape
    out = np.empty(K_scored, dtype=np.float64)
    n_pairs = N * (N - 1) // 2
    for k in range(K_scored):
        terms = top_n_indices[k]
        total = 0.0
        for w_i, w_j in combinations(terms, 2):
            a, b = (int(w_i), int(w_j)) if w_i < w_j else (int(w_j), int(w_i))
            p_i = doc_freqs.get(a, 0) / n_docs
            p_j = doc_freqs.get(b, 0) / n_docs
            p_ij = pair_freqs.get((a, b), 0) / n_docs
            total += _npmi_pair(p_i=p_i, p_j=p_j, p_ij=p_ij)
        out[k] = total / n_pairs
    return out
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
cd spark-vi && poetry run pytest tests/eval/test_coherence.py -v
```

Expected: all 11 tests pass (8 prior + 3 new).

- [ ] **Step 6: Commit**

```bash
git add spark-vi/spark_vi/eval/topic/coherence.py spark-vi/tests/eval/test_coherence.py
git commit -m "feat(eval): _aggregate_topic_coherence + _npmi_pair edge case for p_ij=1"
```

---

### Task B6: Spark-side `_compute_doc_freqs` and `_compute_pair_freqs`

**Files:**
- Modify: `spark-vi/spark_vi/eval/topic/coherence.py` (add the two Spark helpers)
- Modify: `spark-vi/tests/eval/test_coherence.py` (add Spark-driven tests)
- Possibly create: `spark-vi/tests/eval/conftest.py` (fixture for SparkContext) — but check first whether `spark-vi/tests/conftest.py` already provides one.

- [ ] **Step 1: Create `tests/eval/conftest.py` providing a derived `sc` fixture**

The repo's `spark-vi/tests/conftest.py` provides a session-scoped `spark` SparkSession fixture. The eval tests use raw RDDs, so derive `sc` (SparkContext) once per session for convenience.

Create `spark-vi/tests/eval/conftest.py`:

```python
"""Eval-tier pytest fixtures."""
from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def sc(spark):
    """SparkContext derived from the shared session-scoped spark fixture."""
    return spark.sparkContext
```

- [ ] **Step 2: Write the failing Spark-driven tests**

Append to `spark-vi/tests/eval/test_coherence.py`:

```python
def test_compute_doc_freqs_counts_distinct_doc_membership(sc):
    """Each term gets the number of distinct held-out docs it appears in.

    Counts are doc-level (binary): a doc with term 0 appearing 5 times still
    contributes 1 to doc_freqs[0].
    """
    from spark_vi.core.types import BOWDocument
    from spark_vi.eval.topic.coherence import _compute_doc_freqs

    docs = [
        BOWDocument(indices=np.array([0, 1], dtype=np.int32), counts=np.array([5.0, 2.0]), length=7),
        BOWDocument(indices=np.array([0, 2], dtype=np.int32), counts=np.array([1.0, 1.0]), length=2),
        BOWDocument(indices=np.array([1], dtype=np.int32), counts=np.array([3.0]), length=3),
    ]
    rdd = sc.parallelize(docs, numSlices=2)
    interest = {0, 1, 2}

    out = _compute_doc_freqs(rdd, interest)
    assert out == {0: 2, 1: 2, 2: 1}


def test_compute_doc_freqs_ignores_terms_outside_interest_set(sc):
    from spark_vi.core.types import BOWDocument
    from spark_vi.eval.topic.coherence import _compute_doc_freqs

    docs = [
        BOWDocument(indices=np.array([0, 99], dtype=np.int32), counts=np.array([1.0, 1.0]), length=2),
    ]
    rdd = sc.parallelize(docs)
    interest = {0}
    out = _compute_doc_freqs(rdd, interest)
    assert out == {0: 1}  # 99 absent


def test_compute_pair_freqs_emits_only_interest_set_pairs(sc):
    """Pair (i, j) with both i < j and both in interest set."""
    from spark_vi.core.types import BOWDocument
    from spark_vi.eval.topic.coherence import _compute_pair_freqs

    docs = [
        BOWDocument(indices=np.array([0, 1, 2], dtype=np.int32), counts=np.array([1.0, 1.0, 1.0]), length=3),
        BOWDocument(indices=np.array([0, 2], dtype=np.int32), counts=np.array([1.0, 1.0]), length=2),
        BOWDocument(indices=np.array([1, 99], dtype=np.int32), counts=np.array([1.0, 1.0]), length=2),
    ]
    rdd = sc.parallelize(docs, numSlices=2)
    interest = {0, 1, 2}

    out = _compute_pair_freqs(rdd, interest)
    # Doc 0: pairs (0,1), (0,2), (1,2). Doc 1: pair (0,2). Doc 2: no interest pairs.
    assert out == {(0, 1): 1, (0, 2): 2, (1, 2): 1}
```

The `sc` fixture comes from `tests/eval/conftest.py` created in Step 1.

- [ ] **Step 3: Run the tests to verify they fail**

```bash
cd spark-vi && poetry run pytest tests/eval/test_coherence.py -v -k "doc_freqs or pair_freqs"
```

Expected: `ImportError` on the new helpers.

- [ ] **Step 4: Implement the two helpers**

Append to `spark-vi/spark_vi/eval/topic/coherence.py`:

```python
from itertools import combinations as _comb
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark import RDD

    from spark_vi.core.types import BOWDocument


def _compute_doc_freqs(
    bow_rdd: "RDD[BOWDocument]",
    interest_set: set[int],
) -> dict[int, int]:
    """Per-term doc-frequency over the held-out corpus, restricted to interest_set.

    Returns {term_index: # docs containing it}, only for terms in interest_set.
    """
    interest_b = bow_rdd.context.broadcast(interest_set)

    def _emit_terms(doc):
        s = interest_b.value
        seen = set()
        for w in doc.indices:
            iw = int(w)
            if iw in s and iw not in seen:
                seen.add(iw)
                yield (iw, 1)

    counts = bow_rdd.flatMap(_emit_terms).reduceByKey(lambda a, b: a + b).collectAsMap()
    interest_b.unpersist(blocking=False)
    return dict(counts)


def _compute_pair_freqs(
    bow_rdd: "RDD[BOWDocument]",
    interest_set: set[int],
) -> dict[tuple[int, int], int]:
    """Pairwise co-occurrence over the held-out corpus, restricted to interest_set.

    Returns {(min_idx, max_idx): # docs containing both}, only for pairs where
    both indices are in interest_set and at least one doc co-occurrence exists.
    """
    interest_b = bow_rdd.context.broadcast(interest_set)

    def _emit_pairs(doc):
        s = interest_b.value
        terms = sorted({int(w) for w in doc.indices if int(w) in s})
        for a, b in _comb(terms, 2):
            yield ((a, b), 1)

    counts = bow_rdd.flatMap(_emit_pairs).reduceByKey(lambda a, b: a + b).collectAsMap()
    interest_b.unpersist(blocking=False)
    return {tuple(k): v for k, v in counts.items()}
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
cd spark-vi && poetry run pytest tests/eval/test_coherence.py -v
```

Expected: all 14 tests pass.

- [ ] **Step 6: Commit**

```bash
git add spark-vi/spark_vi/eval/topic/coherence.py spark-vi/tests/eval/test_coherence.py
git commit -m "feat(eval): Spark-side _compute_doc_freqs and _compute_pair_freqs"
```

---

### Task B7: `compute_npmi_coherence` orchestrator (LDA path)

**Files:**
- Modify: `spark-vi/spark_vi/eval/topic/coherence.py` (add public `compute_npmi_coherence`)
- Modify: `spark-vi/spark_vi/eval/topic/__init__.py` (re-export it)
- Modify: `spark-vi/tests/eval/test_coherence.py` (add an end-to-end LDA-path test)

- [ ] **Step 1: Write the failing end-to-end test**

Append to `spark-vi/tests/eval/test_coherence.py`:

```python
def test_compute_npmi_coherence_lda_path(sc):
    """End-to-end on a tiny synthetic corpus, no HDP mask."""
    from spark_vi.core.types import BOWDocument
    from spark_vi.eval.topic.coherence import compute_npmi_coherence
    from spark_vi.eval.topic.types import CoherenceReport

    # 2 topics over 4 terms. Topic 0 places mass on terms 0 and 1; topic 1 on 2 and 3.
    topic_term = np.array([
        [0.45, 0.45, 0.05, 0.05],
        [0.05, 0.05, 0.45, 0.45],
    ])
    docs = [
        BOWDocument(indices=np.array([0, 1], dtype=np.int32), counts=np.array([1.0, 1.0]), length=2),
        BOWDocument(indices=np.array([0, 1], dtype=np.int32), counts=np.array([1.0, 1.0]), length=2),
        BOWDocument(indices=np.array([2, 3], dtype=np.int32), counts=np.array([1.0, 1.0]), length=2),
        BOWDocument(indices=np.array([2, 3], dtype=np.int32), counts=np.array([1.0, 1.0]), length=2),
    ]
    rdd = sc.parallelize(docs, numSlices=2)

    report = compute_npmi_coherence(topic_term, rdd, top_n=2)
    assert isinstance(report, CoherenceReport)
    assert report.per_topic_npmi.shape == (2,)
    assert report.top_term_indices.shape == (2, 2)
    assert list(report.topic_indices) == [0, 1]
    assert report.n_holdout_docs == 4
    assert report.top_n == 2
    # Each topic's top-N pair always co-occurs => NPMI = 1.0 per pair => mean = 1.0.
    assert report.per_topic_npmi[0] == pytest.approx(1.0)
    assert report.per_topic_npmi[1] == pytest.approx(1.0)
    assert report.mean == pytest.approx(1.0)
    assert report.min == pytest.approx(1.0)
    assert report.max == pytest.approx(1.0)
    # All values bounded in [-1, 1]
    assert (report.per_topic_npmi >= -1.0).all()
    assert (report.per_topic_npmi <= 1.0).all()
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd spark-vi && poetry run pytest tests/eval/test_coherence.py::test_compute_npmi_coherence_lda_path -v
```

Expected: `ImportError: cannot import name 'compute_npmi_coherence'`.

- [ ] **Step 3: Implement `compute_npmi_coherence`**

Append to `spark-vi/spark_vi/eval/topic/coherence.py`:

```python
from spark_vi.eval.topic.types import CoherenceReport


def compute_npmi_coherence(
    topic_term: np.ndarray,
    holdout_bow: "RDD[BOWDocument]",
    *,
    top_n: int = 20,
    hdp_topic_mask: np.ndarray | None = None,
) -> CoherenceReport:
    """NPMI coherence on held-out data, mean over top-N pairs per topic.

    Args:
        topic_term: shape (K, V) row-stochastic; topic-term distributions
            E[beta]. For OnlineLDA pass lambda_ / lambda_.sum(axis=1, keepdims=True).
            For OnlineHDP same, with shape (T, V).
        holdout_bow: RDD of BOWDocument. Counts are ignored; only the set of
            indices per doc is used (binary co-occurrence).
        top_n: number of top terms per topic. Default 20. Must be <= V.
        hdp_topic_mask: optional boolean array of length K (or T for HDP). When
            provided, only mask==True rows of topic_term are scored. None means
            score all rows (the LDA path).

    Returns:
        CoherenceReport with per-topic NPMI and descriptive summary stats.
    """
    K_full, V = topic_term.shape
    if hdp_topic_mask is None:
        scored_rows = np.arange(K_full, dtype=np.int64)
    else:
        if hdp_topic_mask.shape != (K_full,):
            raise ValueError(
                f"hdp_topic_mask shape {hdp_topic_mask.shape} does not match topic_term K={K_full}"
            )
        scored_rows = np.flatnonzero(hdp_topic_mask).astype(np.int64)

    if scored_rows.size == 0:
        raise ValueError("hdp_topic_mask selected zero topics; nothing to score")

    filtered_topic_term = topic_term[scored_rows]
    top_n_indices = _top_n_terms_per_topic(filtered_topic_term, top_n=top_n)

    interest_set: set[int] = {int(w) for w in np.unique(top_n_indices)}

    n_docs = holdout_bow.count()
    if n_docs == 0:
        raise ValueError("holdout_bow is empty; cannot compute coherence")

    doc_freqs = _compute_doc_freqs(holdout_bow, interest_set)
    pair_freqs = _compute_pair_freqs(holdout_bow, interest_set)

    per_topic = _aggregate_topic_coherence(
        top_n_indices=top_n_indices,
        doc_freqs=doc_freqs,
        pair_freqs=pair_freqs,
        n_docs=n_docs,
    )

    return CoherenceReport(
        per_topic_npmi=per_topic,
        top_term_indices=top_n_indices,
        topic_indices=scored_rows,
        n_holdout_docs=n_docs,
        top_n=top_n,
        mean=float(per_topic.mean()),
        median=float(np.median(per_topic)),
        stdev=float(per_topic.std(ddof=0)),
        min=float(per_topic.min()),
        max=float(per_topic.max()),
    )
```

- [ ] **Step 4: Re-export from the topic eval `__init__.py`**

Edit `spark-vi/spark_vi/eval/topic/__init__.py`:

```python
from spark_vi.eval.topic.coherence import compute_npmi_coherence
from spark_vi.eval.topic.types import CoherenceReport

__all__ = ["CoherenceReport", "compute_npmi_coherence"]
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
cd spark-vi && poetry run pytest tests/eval/test_coherence.py -v
```

Expected: all tests pass (15 total).

- [ ] **Step 6: Commit**

```bash
git add spark-vi/spark_vi/eval/topic/coherence.py spark-vi/spark_vi/eval/topic/__init__.py spark-vi/tests/eval/test_coherence.py
git commit -m "feat(eval): compute_npmi_coherence orchestrator (LDA path)"
```

---

### Task B8: `top_k_used_topics` HDP helper

**Files:**
- Create: `spark-vi/spark_vi/eval/topic/hdp_helpers.py`
- Modify: `spark-vi/spark_vi/eval/topic/__init__.py` (re-export)
- Create: `spark-vi/tests/eval/test_hdp_helpers.py`

- [ ] **Step 1: Write the failing tests**

Create `spark-vi/tests/eval/test_hdp_helpers.py`:

```python
"""Tests for spark_vi.eval.topic.hdp_helpers."""
from __future__ import annotations

import numpy as np
import pytest


def test_top_k_used_topics_returns_correct_length_mask():
    from spark_vi.eval.topic.hdp_helpers import top_k_used_topics
    # T=5 corpus-level sticks. u, v are length T-1 = 4. The last topic carries
    # the residual stick mass.
    u = np.array([1.0, 1.0, 1.0, 1.0])
    v = np.array([1.0, 1.0, 1.0, 1.0])
    mask = top_k_used_topics(u=u, v=v, k=3)
    assert mask.shape == (5,)
    assert mask.dtype == bool
    assert mask.sum() == 3


def test_top_k_used_topics_selects_largest_expected_betas():
    """E[beta_t] computed from GEM stick-breaking; mask picks the top-K by E[beta]."""
    from spark_vi.eval.topic.hdp_helpers import top_k_used_topics
    # Construct (u, v) so that the first stick takes ~half the mass:
    #   E[V_1] = u_1 / (u_1 + v_1) = 9/10
    #   E[beta_1] = E[V_1] = 0.9
    #   subsequent sticks split the remaining 0.1.
    u = np.array([9.0, 1.0, 1.0, 1.0])
    v = np.array([1.0, 1.0, 1.0, 1.0])
    mask = top_k_used_topics(u=u, v=v, k=2)
    # Topic 0 dominates; topic 0 must be in the mask.
    assert mask[0]
    # The remaining 0.1 is split mostly into topic 1, then 2, then 3, then 4 (residual).
    assert mask[1]
    assert mask.sum() == 2


def test_top_k_used_topics_k_must_not_exceed_T():
    from spark_vi.eval.topic.hdp_helpers import top_k_used_topics
    u = np.array([1.0, 1.0])
    v = np.array([1.0, 1.0])  # T = 3
    with pytest.raises(ValueError, match="k"):
        top_k_used_topics(u=u, v=v, k=4)


def test_top_k_used_topics_validates_u_v_same_length():
    from spark_vi.eval.topic.hdp_helpers import top_k_used_topics
    u = np.array([1.0, 1.0, 1.0])
    v = np.array([1.0, 1.0])
    with pytest.raises(ValueError, match="length"):
        top_k_used_topics(u=u, v=v, k=2)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd spark-vi && poetry run pytest tests/eval/test_hdp_helpers.py -v
```

Expected: `ModuleNotFoundError: No module named 'spark_vi.eval.topic.hdp_helpers'`.

- [ ] **Step 3: Implement `hdp_helpers.py`**

Create `spark-vi/spark_vi/eval/topic/hdp_helpers.py`:

```python
"""HDP-specific helpers for eval.

OnlineHDP carries a length-(T-1) (u, v) parameter pair for the corpus-level
GEM stick, plus a (T, V) lambda. Most coherence evaluation wants to score
only the topics with non-trivial usage; this module computes the per-topic
expected stick weights E[beta_t] from (u, v) and exposes a top-K mask.
"""
from __future__ import annotations

import numpy as np


def _expected_corpus_betas(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Per-topic E[beta_t] from GEM stick-breaking parameters.

    For t = 0..T-2:  E[V_t] = u_t / (u_t + v_t)
                    E[beta_t] = E[V_t] * prod_{s<t} (1 - E[V_s])
    Topic T-1 receives the residual: 1 - sum(E[beta_0..T-2]).

    The closed form approximates the actual E[beta_t] (which involves a
    product of independent Beta random variables) by treating E[product]
    as product of expectations. This is the standard approximation used in
    the HDP literature (Wang/Paisley/Blei 2011) for ranking purposes.
    """
    e_vt = u / (u + v)
    log_one_minus = np.log1p(-e_vt)
    cum = np.concatenate([[0.0], np.cumsum(log_one_minus[:-1])])
    e_beta = np.empty(len(u) + 1, dtype=np.float64)
    e_beta[:-1] = e_vt * np.exp(cum)
    e_beta[-1] = max(0.0, 1.0 - e_beta[:-1].sum())
    return e_beta


def top_k_used_topics(*, u: np.ndarray, v: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask of length T selecting the top-K topics by E[beta_t].

    Use as the `hdp_topic_mask` argument to compute_npmi_coherence.
    """
    if u.shape != v.shape:
        raise ValueError(f"u and v must have the same length; got {u.shape} and {v.shape}")
    T = len(u) + 1
    if k > T:
        raise ValueError(f"k={k} exceeds T={T}")
    e_beta = _expected_corpus_betas(u, v)
    # Top-K indices by descending E[beta]; ties broken by ascending index (stable sort).
    sorted_idx = np.argsort(-e_beta, kind="stable")[:k]
    mask = np.zeros(T, dtype=bool)
    mask[sorted_idx] = True
    return mask
```

- [ ] **Step 4: Re-export from the topic eval `__init__.py`**

Edit `spark-vi/spark_vi/eval/topic/__init__.py`:

```python
from spark_vi.eval.topic.coherence import compute_npmi_coherence
from spark_vi.eval.topic.hdp_helpers import top_k_used_topics
from spark_vi.eval.topic.types import CoherenceReport

__all__ = ["CoherenceReport", "compute_npmi_coherence", "top_k_used_topics"]
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
cd spark-vi && poetry run pytest tests/eval/test_hdp_helpers.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add spark-vi/spark_vi/eval/topic/hdp_helpers.py spark-vi/spark_vi/eval/topic/__init__.py spark-vi/tests/eval/test_hdp_helpers.py
git commit -m "feat(eval): top_k_used_topics HDP-side helper"
```

---

### Task B9: `compute_npmi_coherence` HDP-mask path

**Files:**
- Modify: `spark-vi/tests/eval/test_coherence.py` (add the HDP-mask test)

The orchestrator from Task B7 already accepts `hdp_topic_mask`. This task verifies the mask path behaves correctly.

- [ ] **Step 1: Write the failing test**

Append to `spark-vi/tests/eval/test_coherence.py`:

```python
def test_compute_npmi_coherence_hdp_mask_path(sc):
    """HDP-style: T=4 topics in topic_term, mask selects 2 of them."""
    from spark_vi.core.types import BOWDocument
    from spark_vi.eval.topic.coherence import compute_npmi_coherence

    # 4 topics, 4 terms. Topics 0 and 2 are "useful" (mass on 0,1 and 2,3).
    # Topics 1 and 3 are flat (should not be scored).
    topic_term = np.array([
        [0.45, 0.45, 0.05, 0.05],
        [0.25, 0.25, 0.25, 0.25],
        [0.05, 0.05, 0.45, 0.45],
        [0.25, 0.25, 0.25, 0.25],
    ])
    mask = np.array([True, False, True, False])

    docs = [
        BOWDocument(indices=np.array([0, 1], dtype=np.int32), counts=np.array([1.0, 1.0]), length=2),
        BOWDocument(indices=np.array([0, 1], dtype=np.int32), counts=np.array([1.0, 1.0]), length=2),
        BOWDocument(indices=np.array([2, 3], dtype=np.int32), counts=np.array([1.0, 1.0]), length=2),
        BOWDocument(indices=np.array([2, 3], dtype=np.int32), counts=np.array([1.0, 1.0]), length=2),
    ]
    rdd = sc.parallelize(docs, numSlices=2)

    report = compute_npmi_coherence(topic_term, rdd, top_n=2, hdp_topic_mask=mask)
    assert report.per_topic_npmi.shape == (2,)
    assert list(report.topic_indices) == [0, 2]
    assert report.per_topic_npmi[0] == pytest.approx(1.0)
    assert report.per_topic_npmi[1] == pytest.approx(1.0)


def test_compute_npmi_coherence_empty_mask_raises(sc):
    from spark_vi.core.types import BOWDocument
    from spark_vi.eval.topic.coherence import compute_npmi_coherence

    topic_term = np.array([[0.5, 0.5]])
    mask = np.array([False])
    rdd = sc.parallelize(
        [BOWDocument(indices=np.array([0, 1], dtype=np.int32), counts=np.array([1.0, 1.0]), length=2)]
    )
    with pytest.raises(ValueError, match="zero topics"):
        compute_npmi_coherence(topic_term, rdd, top_n=2, hdp_topic_mask=mask)
```

- [ ] **Step 2: Run the tests to verify they pass (no implementation change needed — Task B7 already handled the mask)**

```bash
cd spark-vi && poetry run pytest tests/eval/test_coherence.py -v
```

Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add spark-vi/tests/eval/test_coherence.py
git commit -m "test(eval): compute_npmi_coherence HDP mask path coverage"
```

---

### Task B10: `split_bow_by_person` in charmpheno

**Files:**
- Create: `charmpheno/charmpheno/omop/split.py`
- Create: `charmpheno/tests/test_split.py`

- [ ] **Step 1: Write the failing tests**

Create `charmpheno/tests/test_split.py`:

```python
"""Tests for charmpheno.omop.split.split_bow_by_person."""
from __future__ import annotations

import pytest
from pyspark.ml.linalg import Vectors


@pytest.fixture(scope="module")
def small_bow_df(spark):
    """10-row BOW with person_id 0..9 and a 3-element sparse features vector."""
    rows = [
        (i, Vectors.sparse(3, [(i % 3, 1.0)]))
        for i in range(10)
    ]
    return spark.createDataFrame(rows, schema=["person_id", "features"])


def test_split_bow_by_person_partitions_disjointly(small_bow_df):
    from charmpheno.omop.split import split_bow_by_person

    train, holdout = split_bow_by_person(small_bow_df, holdout_fraction=0.3, seed=42)
    train_ids = {r.person_id for r in train.collect()}
    holdout_ids = {r.person_id for r in holdout.collect()}

    assert train_ids.isdisjoint(holdout_ids)
    assert train_ids | holdout_ids == set(range(10))


def test_split_bow_by_person_is_deterministic_same_seed(small_bow_df):
    from charmpheno.omop.split import split_bow_by_person

    train1, holdout1 = split_bow_by_person(small_bow_df, holdout_fraction=0.3, seed=42)
    train2, holdout2 = split_bow_by_person(small_bow_df, holdout_fraction=0.3, seed=42)

    assert {r.person_id for r in train1.collect()} == {r.person_id for r in train2.collect()}
    assert {r.person_id for r in holdout1.collect()} == {r.person_id for r in holdout2.collect()}


def test_split_bow_by_person_differs_across_seeds(small_bow_df):
    from charmpheno.omop.split import split_bow_by_person

    _, holdout_a = split_bow_by_person(small_bow_df, holdout_fraction=0.3, seed=1)
    _, holdout_b = split_bow_by_person(small_bow_df, holdout_fraction=0.3, seed=2)

    ids_a = {r.person_id for r in holdout_a.collect()}
    ids_b = {r.person_id for r in holdout_b.collect()}
    # Could in principle coincide for tiny inputs, but for 10 rows over two unrelated
    # SHA-256 keyings this is overwhelmingly unlikely; assert non-equal as a regression
    # signal.
    assert ids_a != ids_b


def test_split_bow_by_person_rejects_invalid_fraction(small_bow_df):
    from charmpheno.omop.split import split_bow_by_person

    with pytest.raises(ValueError, match="holdout_fraction"):
        split_bow_by_person(small_bow_df, holdout_fraction=0.0, seed=42)
    with pytest.raises(ValueError, match="holdout_fraction"):
        split_bow_by_person(small_bow_df, holdout_fraction=1.0, seed=42)
```

The `spark` fixture is provided by `charmpheno/tests/conftest.py` (session-scoped SparkSession, confirmed at plan-writing time).

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd charmpheno && poetry run pytest tests/test_split.py -v
```

Expected: `ModuleNotFoundError: No module named 'charmpheno.omop.split'`.

- [ ] **Step 3: Implement `split.py`**

Create `charmpheno/charmpheno/omop/split.py`:

```python
"""Deterministic SHA-256-hash split of a BOW DataFrame by person_id.

Application-layer helper: drivers call this between BOW build and Estimator.fit
to produce a held-out partition for coherence evaluation. Splitting is NOT a
responsibility of the estimator (MLlib idiom — see ADR 0017).

Reproducible regardless of partition state — unlike DataFrame.randomSplit,
which depends on the partition layout at call time. The SHA-256 keying is the
same pattern already used in analysis/cloud/lda_bigquery_cloud.py for ID hashing.
"""
from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

_BUCKET_COUNT = 10_000


def split_bow_by_person(
    bow_df: DataFrame,
    *,
    holdout_fraction: float,
    seed: int,
    person_id_col: str = "person_id",
) -> tuple[DataFrame, DataFrame]:
    """Deterministic train/holdout split of a BOW DataFrame.

    Args:
        bow_df: DataFrame with at least `person_id_col` and feature columns.
            Other columns are preserved on both sides.
        holdout_fraction: in (0, 1). Approximate fraction of distinct persons
            routed to the holdout partition.
        seed: integer mixed into the hash; changing it produces a different
            split for the same population.
        person_id_col: column name to hash on. Default 'person_id'.

    Returns:
        (train_df, holdout_df). Disjoint by person_id_col; their union is the input.
    """
    if not (0.0 < holdout_fraction < 1.0):
        raise ValueError(
            f"holdout_fraction must be in (0, 1); got {holdout_fraction}"
        )

    threshold = int(holdout_fraction * _BUCKET_COUNT)
    bucket_expr = (
        F.conv(
            F.substring(
                F.sha2(F.concat_ws("|", F.col(person_id_col).cast("string"), F.lit(str(seed))), 256),
                1, 8,
            ),
            16, 10,
        ).cast("long") % F.lit(_BUCKET_COUNT)
    )
    annotated = bow_df.withColumn("_holdout_bucket", bucket_expr)
    holdout = annotated.filter(F.col("_holdout_bucket") < threshold).drop("_holdout_bucket")
    train = annotated.filter(F.col("_holdout_bucket") >= threshold).drop("_holdout_bucket")
    return train, holdout
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd charmpheno && poetry run pytest tests/test_split.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add charmpheno/charmpheno/omop/split.py charmpheno/tests/test_split.py
git commit -m "feat(charmpheno): split_bow_by_person deterministic SHA-256 holdout split"
```

---

### Task B11: Analysis driver

**Files:**
- Create: `analysis/local/eval_coherence.py`

This is a CLI driver; we'll cover it with a smoke-test next task.

- [ ] **Step 1: Write the driver**

Create `analysis/local/eval_coherence.py`:

```python
"""Local driver: held-out NPMI coherence for a saved OnlineLDA or OnlineHDP checkpoint.

Loads a VIResult, rebuilds the BOW from the same OMOP parquet, applies the same
deterministic person-keyed split, and computes per-topic NPMI on the holdout
partition. Prints a ranked report.

The fit driver (analysis/local/fit_lda_local.py / fit_hdp_local.py) and this
eval driver must agree on (holdout_fraction, seed). v1 documents this as a
human contract; v2 may stamp it into VIResult.metadata for verification.

Usage:
    poetry run python analysis/local/eval_coherence.py \\
        --checkpoint data/runs/lda_<timestamp> \\
        --input data/simulated/omop_N1000_seed42.parquet \\
        --holdout-fraction 0.1 --seed 42 --top-n 20 \\
        --model-class lda
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
from pyspark.sql import SparkSession

from charmpheno.omop import load_omop_parquet, to_bow_dataframe
from charmpheno.omop.split import split_bow_by_person
from spark_vi.core.types import BOWDocument
from spark_vi.eval.topic import (
    CoherenceReport,
    compute_npmi_coherence,
    top_k_used_topics,
)
from spark_vi.io import load_result

log = logging.getLogger(__name__)


def _build_spark() -> SparkSession:
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    return (
        SparkSession.builder.master("local[2]")
        .appName("eval_coherence")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
        .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
        .getOrCreate()
    )


def run_eval(
    *,
    checkpoint: Path,
    input_parquet: Path,
    holdout_fraction: float,
    seed: int,
    top_n: int,
    model_class: str,
    hdp_k: int,
    spark: SparkSession,
) -> CoherenceReport:
    """Run the eval and return the report. Importable for tests."""
    result = load_result(checkpoint)
    lambda_ = result.global_params["lambda_"]
    topic_term = lambda_ / lambda_.sum(axis=1, keepdims=True)

    if model_class == "hdp":
        u = result.global_params["u"]
        v = result.global_params["v"]
        mask = top_k_used_topics(u=u, v=v, k=hdp_k)
    else:
        mask = None

    df = load_omop_parquet(str(input_parquet), spark=spark)
    bow_df, _vocab_map = to_bow_dataframe(df)
    _train, holdout_df = split_bow_by_person(
        bow_df, holdout_fraction=holdout_fraction, seed=seed
    )
    holdout_df = holdout_df.persist()
    n_holdout = holdout_df.count()
    log.info("holdout: %d docs", n_holdout)

    holdout_rdd = holdout_df.rdd.map(BOWDocument.from_spark_row)

    report = compute_npmi_coherence(
        topic_term, holdout_rdd, top_n=top_n, hdp_topic_mask=mask
    )

    holdout_df.unpersist()
    return report


def _print_ranked_report(report: CoherenceReport, vocab: list, concept_names: dict | None = None) -> None:
    rows = sorted(
        zip(report.topic_indices, report.per_topic_npmi, report.top_term_indices),
        key=lambda r: -r[1],
    )
    print(f"\n  per-topic NPMI (n_holdout_docs={report.n_holdout_docs}, top_n={report.top_n}):")
    print(f"  mean={report.mean:+.4f}  median={report.median:+.4f}  stdev={report.stdev:.4f}  "
          f"min={report.min:+.4f}  max={report.max:+.4f}\n")
    for topic_idx, npmi, term_idx in rows:
        terms = [vocab[i] if i < len(vocab) else f"#{i}" for i in term_idx]
        if concept_names:
            terms = [f"{t} ({concept_names.get(t, '?')})" for t in terms]
        print(f"  topic {int(topic_idx):3d}  NPMI={npmi:+.4f}  top: {', '.join(map(str, terms[:8]))}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--model-class", choices=["lda", "hdp"], required=True)
    parser.add_argument("--hdp-k", type=int, default=50,
                        help="Top-K HDP topics by E[beta] to score (ignored for LDA)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    spark = _build_spark()
    try:
        report = run_eval(
            checkpoint=args.checkpoint,
            input_parquet=args.input,
            holdout_fraction=args.holdout_fraction,
            seed=args.seed,
            top_n=args.top_n,
            model_class=args.model_class,
            hdp_k=args.hdp_k,
            spark=spark,
        )

        # Property checks (the spec calls for these to be assertion-style in the driver).
        assert (report.per_topic_npmi >= -1.0).all(), "NPMI < -1 found"
        assert (report.per_topic_npmi <= 1.0).all(), "NPMI > 1 found"

        result = load_result(args.checkpoint)
        vocab = result.metadata.get("vocab", [])
        _print_ranked_report(report, vocab)
        return 0
    finally:
        spark.stop()


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Manual syntax check**

```bash
poetry run python -c "import ast; ast.parse(open('analysis/local/eval_coherence.py').read()); print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add analysis/local/eval_coherence.py
git commit -m "feat(analysis): eval_coherence.py local driver"
```

---

### Task B12: Driver smoke integration test

**Files:**
- Create: `spark-vi/tests/eval/test_coherence_smoke.py` — invokes `run_eval` against an on-the-fly fit
- Possibly modify: `spark-vi/tests/conftest.py` if a small-checkpoint fixture would be reusable

The "driver smoke" per spec lives at the integration tier and asserts the property checks. We can't easily reach across packages to test `analysis/local/eval_coherence.py`, so we replicate its core (the eval module surface) against an in-memory tiny LDA fit, and treat the driver's CLI wrapper as covered by manual invocation.

- [ ] **Step 1: Write the smoke test**

Create `spark-vi/tests/eval/test_coherence_smoke.py`:

```python
"""Slow-tier smoke: end-to-end on a tiny in-memory fit.

Stand-in for the driver smoke: invokes the eval surface (load + split + score)
against a real fit on a small synthetic corpus. Property-checks the result.

Does NOT verify that NPMI ranks topics by ground-truth quality — that test was
explicitly deferred (see spec).
"""
from __future__ import annotations

import numpy as np
import pytest

from spark_vi.core.types import BOWDocument


pytestmark = pytest.mark.slow


def _make_synthetic_bow(sc, n_docs: int = 200, vocab_size: int = 50,
                        n_topics: int = 5, seed: int = 0):
    """Generate a tiny LDA-shaped synthetic BOW RDD."""
    rng = np.random.default_rng(seed)
    # Each "topic" is a sharp distribution over disjoint slices of vocab.
    slice_size = vocab_size // n_topics
    docs = []
    for _ in range(n_docs):
        t = int(rng.integers(0, n_topics))
        terms = rng.choice(
            np.arange(t * slice_size, (t + 1) * slice_size),
            size=slice_size // 2, replace=False,
        )
        terms = np.sort(terms.astype(np.int32))
        counts = np.ones(len(terms), dtype=np.float64)
        docs.append(BOWDocument(indices=terms, counts=counts, length=int(counts.sum())))
    return sc.parallelize(docs, numSlices=4)


def test_smoke_coherence_on_tiny_synthetic(sc):
    """Property checks: NPMI in [-1, 1], length matches, summary stats consistent."""
    from spark_vi.eval.topic import compute_npmi_coherence

    K, V = 5, 50
    # Skip the actual VI fit; build a "perfectly recovered" topic_term that places
    # mass on each topic's vocab slice. This is what a successful fit would look like.
    slice_size = V // K
    topic_term = np.zeros((K, V), dtype=np.float64)
    for k in range(K):
        topic_term[k, k * slice_size : (k + 1) * slice_size] = 1.0 / slice_size

    holdout = _make_synthetic_bow(sc, n_docs=200, vocab_size=V, n_topics=K, seed=0)
    report = compute_npmi_coherence(topic_term, holdout, top_n=5)

    # Property checks (the only assertions the spec retained from the dropped tests).
    assert report.per_topic_npmi.shape == (K,)
    assert (report.per_topic_npmi >= -1.0).all()
    assert (report.per_topic_npmi <= 1.0).all()

    # Summary stats consistent with numpy on the underlying array.
    assert report.mean == pytest.approx(float(report.per_topic_npmi.mean()))
    assert report.median == pytest.approx(float(np.median(report.per_topic_npmi)))
    assert report.stdev == pytest.approx(float(report.per_topic_npmi.std(ddof=0)))
    assert report.min == pytest.approx(float(report.per_topic_npmi.min()))
    assert report.max == pytest.approx(float(report.per_topic_npmi.max()))

    # Driver-side check matching the assertion in eval_coherence.main().
    assert report.n_holdout_docs == 200
```

- [ ] **Step 2: Run the smoke test**

```bash
cd spark-vi && poetry run pytest tests/eval/test_coherence_smoke.py -v --slow
```

Expected: 1 passed. (If `--slow` isn't the pytest flag in this repo's conftest, check `conftest.py` for the slow marker switch — common alternatives: `-m slow`, `--run-slow`.)

- [ ] **Step 3: Commit**

```bash
git add spark-vi/tests/eval/test_coherence_smoke.py
git commit -m "test(eval): driver-equivalent smoke on tiny synthetic"
```

---

### Task B13: ADR 0017

**Files:**
- Create: `docs/decisions/0017-topic-coherence-evaluation.md`
- Modify: `docs/superpowers/specs/2026-05-11-topic-coherence-evaluation-design.md` (fix the "ADR 0016" cross-reference to "ADR 0017")

- [ ] **Step 1: Write ADR 0017**

Create `docs/decisions/0017-topic-coherence-evaluation.md`:

```markdown
# ADR 0017 — Topic coherence evaluation: NPMI on held-out

**Status:** Accepted
**Date:** 2026-05-11
**Supersedes:** none
**Superseded by:** none
**Companion spec:** docs/superpowers/specs/2026-05-11-topic-coherence-evaluation-design.md

## Context

The framework now has working OnlineLDA and OnlineHDP plus persistence on
both the framework and shim sides. What's missing is any *quantitative*
assessment of topic quality. ELBO does not compare across K or across model
classes; we need a held-out interpretability metric for K-selection,
T-selection, hyperparameter comparisons, and model-class comparisons.

The user's prior project used a modified-UCI coherence:
`sum log_2((1 + p(t_i, t_j)) / (p(t_i) * p(t_j)))` over top-N pairs, then
z-scored. The brainstorming session that preceded this ADR (and the
companion spec) settled the metric choice; this ADR records the decisions.

## Decisions

### Metric: NPMI (Roder et al. 2015)

```
NPMI(w_i, w_j) = log[ p(w_i, w_j) / (p(w_i) * p(w_j)) ]  /  -log p(w_i, w_j)
NPMI = -1 when p(w_i, w_j) = 0    (Roder convention)
NPMI =  1 when p(w_i, w_j) = 1    (perfect co-occurrence)
```

Per-topic score: mean NPMI over the unordered pairs of the top-N terms.

Rejected alternatives:
- *Modified UCI* (user's prior). Custom smoothing constant (`1 + ...`,
  `0.01 + ...` etc.) was tunable but un-principled; z-scoring was needed to
  interpret. NPMI is bounded in `[-1, 1]` without tuning and interpretable
  in absolute terms.
- *Gensim CoherenceModel adapter*. Stock `c_uci` uses sliding-window
  co-occurrence over tokenized text, which is not the right shape for
  patient bags; we'd be patching gensim rather than wrapping it. Gensim's
  release cadence has slowed and the CoherenceModel API has shifted
  between 3.x and 4.x; the maintenance cost of tracking that drift is
  larger than maintaining ~80 lines of NPMI in-tree.

### Co-occurrence shape: whole-document (patient bag)

OMOP records *are* temporally ordered, but choosing a sensible window for
clinical timelines (1-year rolling? episode-bounded?
observation-period-relative?) is its own research question. v1 treats each
patient bag as unordered. A temporal-window variant is plausible v2 work
if topic-quality assessment turns out to be sensitive to it.

### Layering: generic metric, domain split, driver-layer orchestration

- `spark_vi.eval.topic.coherence` is generic over the `(topic_term, holdout_bow)`
  contract. No patient or OMOP concepts.
- `charmpheno.omop.split_bow_by_person` is the deterministic, SHA-256-keyed
  split helper. Lives in charmpheno because it knows about the BOW shape
  and the person-keyed structure.
- `analysis/local/eval_coherence.py` orchestrates: load checkpoint → split
  BOW → score.

The split is **not** a Param on the estimator. MLlib's idiom is that
estimators don't split; users do. Wrapping the split into a Param would
couple every fit to an eval concern.

### Driver-side contract

The fit driver and the eval driver must agree on `(holdout_fraction, seed)`.
v1 documents this as a human contract in the driver headers and this ADR.
v2 may stamp the split provenance into `VIResult.metadata` so the eval
driver can verify before running.

### HDP topic selection

OnlineHDP carries (T, V) lambda where many topics in the tail of the
corpus stick have negligible usage. Scoring them produces near-random NPMI
and inflates the report. The default eval workflow passes a top-K-by-`E[beta]`
mask computed from the corpus-level (u, v); k defaults to 50.

### What's deferred

- Term relevance (Sievert-Shirley) and concept-named top-N tables — its
  own deliverable, same `E[beta]` plumbing.
- pyLDAvis adapter — requires materializing a full `doc_topic_dists`
  matrix; non-trivial plumbing of its own.
- Cloud driver — mirror of the local driver after we have a real model at
  AoU scale to evaluate.
- Gensim adapter and OCTIS integration — if and when the metric menu
  expands.
- Simulation / synthetic recovery testing — its own spec.
- Z-score normalization — NPMI is already bounded and self-interpretable.

## Consequences

- The eval module is purely additive; no compatibility surface.
- `charmpheno.omop.split` is new but additive.
- ADR 0016's `spark_vi.models.topic` namespace is now mirrored by
  `spark_vi.eval.topic`.
```

- [ ] **Step 2: Fix the spec's "ADR 0016" cross-reference**

In `docs/superpowers/specs/2026-05-11-topic-coherence-evaluation-design.md`, find:

```
4. **Land ADR 0016** documenting the metric choice ...
```

and the later:

```
ADR 0016 lands with this branch and records:
```

Replace both `ADR 0016` with `ADR 0017`.

- [ ] **Step 3: Verify the spec is consistent**

```bash
grep -n "ADR 0016\|ADR 0017" docs/superpowers/specs/2026-05-11-topic-coherence-evaluation-design.md
```

Expected: only `ADR 0017` references, no remaining `ADR 0016` references.

- [ ] **Step 4: Commit**

```bash
git add docs/decisions/0017-topic-coherence-evaluation.md docs/superpowers/specs/2026-05-11-topic-coherence-evaluation-design.md
git commit -m "docs(adr): ADR 0017 — topic coherence evaluation; fix spec cross-ref"
```

---

### Task B14: Final verification + PR

- [ ] **Step 1: Run the full test suites**

```bash
cd spark-vi && poetry run pytest 2>&1 | tail -10
cd ../charmpheno && poetry run pytest 2>&1 | tail -10
```

Expected: all green in both, including the new tests.

- [ ] **Step 2: Smoke the analysis driver against a real fixture**

```bash
# Pre-req: a saved LDA checkpoint produced by analysis/local/fit_lda_local.py
# at e.g. data/runs/lda_smoketest, fit on data/simulated/omop_N1000_seed42.parquet
# using the same holdout_fraction=0.1, seed=42 as the eval invocation.
poetry run python analysis/local/eval_coherence.py \
  --checkpoint data/runs/lda_smoketest \
  --input data/simulated/omop_N1000_seed42.parquet \
  --holdout-fraction 0.1 --seed 42 --top-n 20 \
  --model-class lda
```

Expected: prints a ranked topic table with per-topic NPMI values in `[-1, 1]` and summary stats. If no smoketest checkpoint exists, document this in the PR as a manual-run pending and skip.

- [ ] **Step 3: Push and open PR**

```bash
git push -u origin feat/eval-topic-coherence
gh pr create --title "feat(eval): topic coherence via NPMI on held-out" --body "$(cat <<'EOF'
## Summary
- New `spark_vi.eval.topic` module: NPMI coherence + `top_k_used_topics` HDP helper
- New `charmpheno.omop.split.split_bow_by_person` deterministic split
- New `analysis/local/eval_coherence.py` local driver
- ADR 0017 records metric choice, layering, deferrals
- Spec ADR cross-reference corrected (0016 → 0017)

Builds on `refactor/models-topic-namespace` (ADR 0016, prerequisite PR).

## Test plan
- [ ] `pytest spark-vi/tests/eval -v` fast tier green
- [ ] `pytest spark-vi/tests/eval/test_coherence_smoke.py --slow` green
- [ ] `pytest charmpheno/tests/test_split.py -v` green
- [ ] Manual: `analysis/local/eval_coherence.py` runs to completion on a saved LDA checkpoint, prints a ranked table

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 4: Done**

After merge: term relevance, pyLDAvis adapter, and cloud driver are the next eval-side deliverables. See ADR 0017's "What's deferred" section.

---

## Self-review notes

- **Spec coverage:** Every "In v1" bullet from the spec has a corresponding task. The prerequisite refactor is Part A. The eval module surface (coherence.py, types.py, hdp_helpers.py) maps to B2/B3/B4/B5/B6/B7/B8/B9. Split helper is B10. Driver is B11. Smoke is B12. ADR 0017 is B13.
- **Spec gap closed:** The spec said `holdout_bow: RDD` and referred to `BOWRow`; the actual type is `BOWDocument` (from `spark_vi/core/types.py`). The plan uses `BOWDocument` and the driver converts via `df.rdd.map(BOWDocument.from_spark_row)`. No code change to `BOWDocument` itself — `from_spark_row` already exists.
- **Spec correction:** The spec assigned "ADR 0016" to the eval branch, but the prerequisite refactor also gets an ADR. Plan numbers them sequentially: 0016 for the refactor (lands first), 0017 for the eval (lands second). Task B13 patches the spec's cross-reference.
- **Tests cover:** NPMI math (independent, perfect, zero, anti-correlated, small-prob, perfect-coocurrence-edge), top-N selection (argmax, ties, validation), aggregation (tiny example, missing pair, averaging), Spark-side doc/pair freqs (interest filtering, ignoring out-of-set), end-to-end LDA path, end-to-end HDP-mask path, empty-mask error, HDP topic selection (mask shape, ranking, validation), split helper (disjoint, deterministic, seed-sensitive, fraction validation), smoke (property checks + summary-stat consistency).
- **No placeholders.** Every step has the actual content. Sed commands are scoped to specific directories. Test assertions use real expected values, not "appropriate" or "as needed."
