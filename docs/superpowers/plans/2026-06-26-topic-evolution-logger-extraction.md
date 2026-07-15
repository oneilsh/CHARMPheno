# Topic-Evolution Logger Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the duplicated λ→topic-word numerics out of the three cloud drivers' `_make_topic_evolution_logger` into one tested spark-vi helper, leaving only vocab-labeling + formatting + model-specific annotation in the drivers.

**Architecture:** A new pure helper `spark_vi.models.topic.diagnostics.topic_word_summary(lam, top_n)` computes the model-agnostic per-topic numerics (row sums, peak, mass fraction, top-N term indices + probs) from the `(K,V)` topic-word variational parameter λ. Each driver's `_on_iter` then calls it and supplies only its own ordering key, per-topic annotation (LDA `α`, STM block label, HDP stick E[β]), OMOP vocab mapping, and print formatting — the parts that are genuinely domain/model-specific. This is the parked "topic evolution logger extraction — defer until STM is the third data point" refactor (REVIEW_LOG 2026-05-28); STM is now that third data point.

**Tech Stack:** Python, numpy, pytest. spark-vi (pure-numpy engine layer) + analysis/cloud drivers.

## Global Constraints

- **Layering rule (load-bearing):** spark-vi is domain-agnostic — the helper must take only `lam` (a numpy array) and `top_n`, and return numerics + term *indices*. It must NEVER see OMOP concept ids/names. Vocab mapping (`idx_to_cid` → `name_by_id`) stays in the drivers.
- **Behavior-preserving:** each driver's printed output must be byte-identical before and after. Verify by capturing stdout on a fixed input pre-refactor and diffing post-refactor.
- **No new dependencies.** numpy only.
- **Do NOT** move `build_topic_block_partition` or any CLI parsing into spark-vi in this plan (explicitly out of scope per the review decision).
- Commit messages end with the project's `Co-Authored-By` trailer.

## File Structure

- Create: `spark-vi/spark_vi/models/topic/diagnostics.py` — the `topic_word_summary` helper.
- Create: `spark-vi/tests/test_topic_diagnostics.py` — unit tests for the helper.
- Modify: `analysis/cloud/lda_bigquery_cloud.py` — `_make_topic_evolution_logger._on_iter`.
- Modify: `analysis/cloud/stm_bigquery_cloud.py` — `_make_topic_evolution_logger._on_iter`.
- Modify: `analysis/cloud/hdp_bigquery_cloud.py` — `_make_topic_evolution_logger._on_iter`.

---

### Task 1: `topic_word_summary` helper + unit tests

**Files:**
- Create: `spark-vi/spark_vi/models/topic/diagnostics.py`
- Test: `spark-vi/tests/test_topic_diagnostics.py`

**Interfaces:**
- Produces: `topic_word_summary(lam: np.ndarray, top_n: int) -> dict[str, np.ndarray]` returning keys:
  - `row_sums` — `(K,)` Σλ_k
  - `peak` — `(K,)` `max_v λ_kv / max(Σλ_k, 1e-12)`
  - `mass_fraction` — `(K,)` `Σλ_k / max(Σ_k Σλ_k, 1e-12)` (the LDA/STM E[β]; HDP ignores it)
  - `top_indices` — `(K, top_n)` int, per-topic top term column indices, descending row-stochastic prob
  - `top_probs` — `(K, top_n)` the corresponding probabilities

- [ ] **Step 1: Write the failing test**

```python
# spark-vi/tests/test_topic_diagnostics.py
"""Tests for the shared topic-word per-iteration numerics helper."""
from __future__ import annotations

import numpy as np

from spark_vi.models.topic.diagnostics import topic_word_summary


def test_row_sums_peak_mass_fraction():
    lam = np.array([[1.0, 3.0, 0.0, 0.0],   # row sum 4, peak 3/4
                    [2.0, 2.0, 2.0, 2.0]])   # row sum 8, peak 2/8
    s = topic_word_summary(lam, top_n=2)
    np.testing.assert_allclose(s["row_sums"], [4.0, 8.0])
    np.testing.assert_allclose(s["peak"], [0.75, 0.25])
    np.testing.assert_allclose(s["mass_fraction"], [4.0 / 12.0, 8.0 / 12.0])


def test_top_indices_and_probs_match_manual_argsort():
    rng = np.random.default_rng(0)
    lam = rng.gamma(2.0, 1.0, size=(3, 6))
    top_n = 3
    s = topic_word_summary(lam, top_n=top_n)
    topics = lam / lam.sum(axis=1, keepdims=True)
    for k in range(3):
        want = topics[k].argsort()[::-1][:top_n]
        np.testing.assert_array_equal(s["top_indices"][k], want)
        np.testing.assert_allclose(s["top_probs"][k], topics[k][want])
    assert s["top_indices"].shape == (3, top_n)
    assert s["top_probs"].shape == (3, top_n)


def test_top_n_larger_than_vocab_is_clamped():
    lam = np.array([[1.0, 2.0]])
    s = topic_word_summary(lam, top_n=10)
    assert s["top_indices"].shape == (1, 2)   # clamped to V


def test_zero_row_is_safe():
    # A topic with zero mass must not divide-by-zero (1e-12 guard).
    lam = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 2.0]])
    s = topic_word_summary(lam, top_n=2)
    assert np.all(np.isfinite(s["peak"]))
    assert np.all(np.isfinite(s["top_probs"]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd spark-vi && python -m pytest tests/test_topic_diagnostics.py -q`
Expected: FAIL with `ModuleNotFoundError`/`ImportError` (diagnostics module does not exist).

- [ ] **Step 3: Write minimal implementation**

```python
# spark-vi/spark_vi/models/topic/diagnostics.py
"""Shared per-iteration topic-word numerics for the topic models.

Pure, model-agnostic derivations from the (K, V) topic-word variational
parameter lambda — the math the LDA/STM/HDP cloud drivers' top-terms loggers
all share. Callers supply their own ordering key, per-topic annotation, and
vocab labeling; this helper never sees concept ids/names (engine layer is
domain-agnostic).
"""
from __future__ import annotations

import numpy as np


def topic_word_summary(lam: np.ndarray, top_n: int) -> dict[str, np.ndarray]:
    """Per-topic top-N term numerics from lambda (K, V).

    Returns a dict with row_sums (K,), peak (K,), mass_fraction (K,),
    top_indices (K, min(top_n, V)) and top_probs (same shape). top_indices are
    column indices into the vocabulary, descending by row-stochastic
    probability; top_probs are those probabilities. mass_fraction is the
    row-sum-normalized E[beta] used by LDA/STM (HDP supplies its own from the
    corpus sticks and ignores this field).
    """
    lam = np.asarray(lam, dtype=np.float64)
    row_sums = lam.sum(axis=1)
    denom = np.maximum(row_sums, 1e-12)
    peak = lam.max(axis=1) / denom
    topics = lam / denom[:, None]                       # row-stochastic
    n = min(int(top_n), lam.shape[1])
    top_indices = np.argsort(topics, axis=1)[:, ::-1][:, :n]
    top_probs = np.take_along_axis(topics, top_indices, axis=1)
    return {
        "row_sums": row_sums,
        "peak": peak,
        "mass_fraction": row_sums / max(float(row_sums.sum()), 1e-12),
        "top_indices": top_indices,
        "top_probs": top_probs,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd spark-vi && python -m pytest tests/test_topic_diagnostics.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/diagnostics.py spark-vi/tests/test_topic_diagnostics.py
git commit -m "feat(spark-vi): topic_word_summary per-iter numerics helper

Shared, pure derivation of the (K,V) lambda top-terms numerics (row sums,
peak, mass fraction, top-N indices+probs) that the LDA/STM/HDP cloud driver
loggers each duplicate. Domain-agnostic (no vocab); callers supply ordering,
annotation, and labeling.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Refactor LDA driver logger

**Files:**
- Modify: `analysis/cloud/lda_bigquery_cloud.py` — `_make_topic_evolution_logger._on_iter` (currently lines ~59-88)

**Interfaces:**
- Consumes: `topic_word_summary` (Task 1).

- [ ] **Step 1: Capture golden output (characterization, pre-refactor)**

Run this from the repo root and save the output as the golden:

```bash
cd analysis/cloud && python -c "
import numpy as np, importlib
m = importlib.import_module('lda_bigquery_cloud')
gp = {'lambda': np.array([[5.,1.,0.1,0.1],[0.1,4.,2.,0.1],[1.,1.,1.,3.]]),
      'alpha': np.array([0.3,0.2,0.5])}
idx_to_cid = {0:10,1:20,2:30,3:40}
name_by_id = {10:'aaa',20:'bbb',30:'ccc',40:'ddd'}
fn = m._make_topic_evolution_logger(top_n=2, every_n=1,
        idx_to_cid=idx_to_cid, name_by_id=name_by_id)
fn(2, gp, [])
" > /tmp/lda_logger_golden.txt 2>&1
cat /tmp/lda_logger_golden.txt
```

Expected: three `[driver]    topic ...` lines. (If the module import fails on `pyspark.ml`/`distutils` in this environment, instead import just the function via the same `importlib` path inside a venv where pyspark workers are healthy, or capture on the cluster. The function body itself uses only numpy + the closure args.)

- [ ] **Step 2: Refactor `_on_iter` to use the helper**

Replace the body of `_on_iter` (the per-iter math + loop) with:

```python
    def _on_iter(iter_num: int, global_params: dict,
                 _: list[float]) -> None:
        if every_n <= 0 or iter_num % every_n != 0:
            return
        from spark_vi.models.topic.diagnostics import topic_word_summary
        lam = global_params["lambda"]                         # (K, V)
        alpha = global_params["alpha"]                        # (K,)
        s = topic_word_summary(lam, top_n)
        # Heaviest topics first; printed k is the native (stable) index, so a
        # topic moving up/down the Σλ ranking across iters is a real signal.
        order = np.argsort(s["row_sums"])[::-1]
        print(f"[driver]   --- topics @ iter {iter_num} ---", flush=True)
        for k in order:
            terms = ", ".join(
                f"{name_by_id.get(idx_to_cid[int(j)], '?')[:24]}({p:.3f})"
                for j, p in zip(s["top_indices"][k], s["top_probs"][k])
            )
            print(
                f"[driver]    topic {k:>2}  "
                f"α={alpha[k]:.4g}  E[β]={s['mass_fraction'][k]:.4f}  "
                f"Σλ={s['row_sums'][k]:.3g}  peak={s['peak'][k]:.3f}  | {terms}",
                flush=True,
            )
    return _on_iter
```

(Keep the existing docstring; the per-topic stat definitions it documents are unchanged.)

- [ ] **Step 3: Verify byte-identical output (post-refactor)**

Re-run the Step-1 capture into `/tmp/lda_logger_after.txt`, then:

Run: `diff /tmp/lda_logger_golden.txt /tmp/lda_logger_after.txt && echo IDENTICAL`
Expected: `IDENTICAL` (no diff).

- [ ] **Step 4: Commit**

```bash
git add analysis/cloud/lda_bigquery_cloud.py
git commit -m "refactor(lda-driver): logger uses spark-vi topic_word_summary

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Refactor STM driver logger (preserve block-label prefix)

**Files:**
- Modify: `analysis/cloud/stm_bigquery_cloud.py` — `_make_topic_evolution_logger._on_iter` (currently lines ~42-69)

**Interfaces:**
- Consumes: `topic_word_summary` (Task 1).

- [ ] **Step 1: Capture golden output (pre-refactor)**

```bash
cd analysis/cloud && python -c "
import numpy as np, importlib
m = importlib.import_module('stm_bigquery_cloud')
gp = {'lambda': np.array([[5.,1.,0.1,0.1],[0.1,4.,2.,0.1],[1.,1.,1.,3.]])}
idx_to_cid = {0:10,1:20,2:30,3:40}
name_by_id = {10:'aaa',20:'bbb',30:'ccc',40:'ddd'}
fn = m._make_topic_evolution_logger(top_n=2, every_n=1,
        idx_to_cid=idx_to_cid, name_by_id=name_by_id,
        topic_labels=['background','background','cancer'])
fn(2, gp, [])
" > /tmp/stm_logger_golden.txt 2>&1
cat /tmp/stm_logger_golden.txt
```

Expected: header + three lines, each carrying a ` [background]`/`[    cancer]` block prefix and NO `α=` field (STM has no per-topic α).

- [ ] **Step 2: Refactor `_on_iter`**

Replace the per-iter math + loop body with (note: STM keeps the block-label prefix and drops α):

```python
    def _on_iter(iter_num: int, global_params: dict, _: list[float]) -> None:
        if every_n <= 0 or iter_num % every_n != 0:
            return
        from spark_vi.models.topic.diagnostics import topic_word_summary
        lam = global_params["lambda"]                          # (K, V)
        s = topic_word_summary(lam, top_n)
        # Heaviest topics first; printed k is the native (stable) index.
        order = np.argsort(s["row_sums"])[::-1]
        print(f"[driver]   --- topics @ iter {iter_num} ---", flush=True)
        for k in order:
            ki = int(k)
            terms = ", ".join(
                f"{name_by_id.get(idx_to_cid[int(j)], '?')[:24]}({p:.3f})"
                for j, p in zip(s["top_indices"][ki], s["top_probs"][ki])
            )
            blk = (f" [{topic_labels[ki]:>10.10}]"
                   if topic_labels is not None else "")
            print(
                f"[driver]    topic {ki:>2}{blk}  "
                f"E[β]={s['mass_fraction'][ki]:.4f}  Σλ={s['row_sums'][ki]:.3g}  "
                f"peak={s['peak'][ki]:.3f}  | {terms}",
                flush=True,
            )
    return _on_iter
```

(Keep the existing docstring.)

- [ ] **Step 3: Verify byte-identical output**

Re-run Step-1 capture into `/tmp/stm_logger_after.txt`, then:
Run: `diff /tmp/stm_logger_golden.txt /tmp/stm_logger_after.txt && echo IDENTICAL`
Expected: `IDENTICAL`.

- [ ] **Step 4: Commit**

```bash
git add analysis/cloud/stm_bigquery_cloud.py
git commit -m "refactor(stm-driver): logger uses spark-vi topic_word_summary

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Refactor HDP driver logger (preserve stick ordering + active count)

**Files:**
- Modify: `analysis/cloud/hdp_bigquery_cloud.py` — `_make_topic_evolution_logger._on_iter` (currently lines ~57-88)

**Interfaces:**
- Consumes: `topic_word_summary` (Task 1). Keeps `expected_corpus_betas`, `topic_count_at_mass` (already in spark-vi) for the HDP-native ordering.

- [ ] **Step 1: Capture golden output (pre-refactor)**

```bash
cd analysis/cloud && python -c "
import numpy as np, importlib
m = importlib.import_module('hdp_bigquery_cloud')
T = 3
gp = {'lambda': np.array([[5.,1.,0.1,0.1],[0.1,4.,2.,0.1],[1.,1.,1.,3.]]),
      'u': np.array([1.0, 1.0]), 'v': np.array([1.0, 1.0])}
idx_to_cid = {0:10,1:20,2:30,3:40}
name_by_id = {10:'aaa',20:'bbb',30:'ccc',40:'ddd'}
fn = m._make_topic_evolution_logger(top_n=2, every_n=1,
        idx_to_cid=idx_to_cid, name_by_id=name_by_id, T=T, mass_threshold=0.99)
fn(2, gp, [])
" > /tmp/hdp_logger_golden.txt 2>&1
cat /tmp/hdp_logger_golden.txt
```

Expected: header with `(n/T active)` + per-active-topic lines ordered by stick E[β], each with `E[β]=`, `Σλ=`, `peak=`.

- [ ] **Step 2: Refactor `_on_iter`** (keep the stick-derived ordering + `n_active`; only the λ-numerics move to the helper)

```python
    def _on_iter(iter_num: int, global_params: dict,
                 _: list[float]) -> None:
        if every_n <= 0 or iter_num % every_n != 0:
            return
        from spark_vi.models.topic.diagnostics import topic_word_summary
        lam = global_params["lambda"]                         # (T, V)
        u = global_params["u"]                                # (T-1,)
        v = global_params["v"]                                # (T-1,)
        E_beta = expected_corpus_betas(u, v, T=T)
        n_active = topic_count_at_mass(E_beta, mass_threshold)
        order = [int(t) for t in np.argsort(E_beta)[::-1][:n_active]]
        s = topic_word_summary(lam, top_n)
        print(
            f"[driver]   --- topics @ iter {iter_num} "
            f"({n_active}/{T} active) ---",
            flush=True,
        )
        for t in order:
            terms = ", ".join(
                f"{name_by_id.get(idx_to_cid[int(j)], '?')[:24]}({p:.3f})"
                for j, p in zip(s["top_indices"][t], s["top_probs"][t])
            )
            print(
                f"[driver]    topic {t:>3}  "
                f"E[β]={E_beta[t]:.4f}  Σλ={s['row_sums'][t]:.3g}  "
                f"peak={s['peak'][t]:.3f}  | {terms}",
                flush=True,
            )
    return _on_iter
```

Keep the deferred import of `expected_corpus_betas`, `topic_count_at_mass` at the factory-body top (unchanged), and the existing docstring.

- [ ] **Step 3: Verify byte-identical output**

Re-run Step-1 capture into `/tmp/hdp_logger_after.txt`, then:
Run: `diff /tmp/hdp_logger_golden.txt /tmp/hdp_logger_after.txt && echo IDENTICAL`
Expected: `IDENTICAL`.

- [ ] **Step 4: Commit**

```bash
git add analysis/cloud/hdp_bigquery_cloud.py
git commit -m "refactor(hdp-driver): logger uses spark-vi topic_word_summary

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

- **Spec coverage:** Task 1 creates + tests the shared helper; Tasks 2–4 thin each of the three drivers (LDA/STM/HDP) onto it. The duplicated λ-core (`row_sums`/`peak`/row-stochastic/top-indices+probs) is now single-sourced. ✓
- **Layering:** the helper takes only `lam` + `top_n` and returns numerics + term *indices*; vocab mapping stays in the drivers. ✓
- **Behavior-preservation:** each driver task captures golden stdout pre-refactor and diffs post-refactor (byte-identical gate). The helper uses the `np.maximum(.,1e-12)` guard that STM/HDP already used and LDA's valid-λ path is unaffected by; argsort+reverse matches each driver's existing tie-breaking. ✓
- **Type consistency:** `topic_word_summary` returns the same dict keys consumed in Tasks 2–4 (`row_sums`, `peak`, `mass_fraction`, `top_indices`, `top_probs`). ✓
- **Out of scope respected:** `build_topic_block_partition` is untouched. ✓

## Execution Handoff

Execute via **superpowers:subagent-driven-development** (chosen): a fresh subagent per task, two-stage review (spec-compliance + code-quality) between tasks, so the implementation runs in subagent context.
