# Salesmanship Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a static, GitHub-Pages-hosted Svelte dashboard with three tabs (Phenotype Atlas / Patient Explorer / Simulator) that demonstrates a trained CHARMPheno topic model to non-technical audiences, with all patient-shaped artifacts synthetic and generated in-browser.

**Architecture:** Two layers. (1) A minimal Python export pipeline (`charmpheno.export.dashboard`) reads a saved `VIResult` and emits a four-file JSON bundle: `model.json`, `phenotypes.json`, `vocab.json` (trimmed to top-N codes by corpus frequency), `corpus_stats.json`. (2) A Svelte 5 + Vite + D3 single-page app under `dashboard/` consumes the bundle, generates synthetic patients in JS, computes JSD-MDS for the topic map in JS, and runs all inference/sampling client-side.

**Tech Stack:** Python + PySpark + numpy (export side, existing project conventions). TypeScript + Svelte 5 + Vite + D3 + Vitest (app side). GitHub Actions for deploy. No backend, no WASM.

**Spec:** [`docs/superpowers/specs/2026-05-13-dashboard-design.md`](../specs/2026-05-13-dashboard-design.md)

---

## Phasing

Seven phases, each ending with a green commit.

- **Phase 1** (Tasks 1-5) — Python export pipeline. Output: 4-file JSON bundle.
- **Phase 2** (Tasks 6-8) — Dashboard scaffold + bundle loader. Output: routable empty shell.
- **Phase 3** (Tasks 9-14) — TypeScript math + cohort/MDS generation. Output: tested primitives.
- **Phase 4** (Tasks 15-18) — Phenotype Atlas.
- **Phase 5** (Tasks 19-22) — Patient Explorer.
- **Phase 6** (Tasks 23-26) — Simulator.
- **Phase 7** (Tasks 27-29) — Deploy + ADR.

---

## Phase 1: Python Export Pipeline

### Task 1: Corpus-stats sidecar

The dashboard needs four aggregate corpus scalars: total docs, mean codes/doc, K, V, V_full. Plus per-vocab marginal probabilities for the top-N filter in Task 2. The sidecar holds the scalars; the marginals are an intermediate the driver passes to Task 2.

**Files:**
- Create: `charmpheno/charmpheno/export/corpus_stats.py`
- Create: `charmpheno/tests/test_export_corpus_stats.py`

- [ ] **Step 1: Write the failing test**

```python
# charmpheno/tests/test_export_corpus_stats.py
from __future__ import annotations
import json
from pathlib import Path

import pytest

from charmpheno.export.corpus_stats import (
    CorpusStats,
    compute_corpus_stats,
    write_corpus_stats_sidecar,
)


def test_compute_corpus_stats_basic():
    docs = [
        {"indices": [0, 0, 1], "counts": [2, 1]},
        {"indices": [1, 2],    "counts": [1, 1]},
        {"indices": [0, 2],    "counts": [1, 1]},
    ]
    stats = compute_corpus_stats(docs=iter(docs), vocab_size=3, k=4)
    assert stats.corpus_size_docs == 3
    assert stats.mean_codes_per_doc == pytest.approx((3 + 2 + 2) / 3)
    assert stats.k == 4
    assert stats.v_full == 3
    assert stats.code_marginals[0] == pytest.approx(3 / 7)
    assert stats.code_marginals[1] == pytest.approx(2 / 7)
    assert stats.code_marginals[2] == pytest.approx(2 / 7)


def test_write_corpus_stats_sidecar_omits_marginals(tmp_path: Path):
    # marginals are an intermediate, not in the sidecar file
    stats = CorpusStats(
        corpus_size_docs=10, mean_codes_per_doc=18.4,
        k=80, v_full=10000, code_marginals=[0.0001] * 10000,
    )
    out = tmp_path / "corpus_stats.json"
    write_corpus_stats_sidecar(stats, out, v_displayed=5000)
    payload = json.loads(out.read_text())
    assert set(payload.keys()) == {"corpus_size_docs", "mean_codes_per_doc", "k", "v", "v_full"}
    assert payload["v"] == 5000
    assert payload["v_full"] == 10000
```

- [ ] **Step 2: Run, expect FAIL**

```bash
cd charmpheno && poetry run pytest tests/test_export_corpus_stats.py -v
```

- [ ] **Step 3: Implement**

```python
# charmpheno/charmpheno/export/corpus_stats.py
"""Aggregate corpus statistics for the dashboard bundle.

The driver computes these once from a held-out (or full) BOW and uses them
in two ways: (1) the small scalars get written to corpus_stats.json;
(2) code_marginals drive the top-N vocab trim in the vocab.json writer.
Marginals are NOT exported to the bundle on their own — vocab.json carries
the surviving codes' corpus_freq per row.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class CorpusStats:
    corpus_size_docs: int
    mean_codes_per_doc: float
    k: int
    v_full: int
    code_marginals: list[float]  # length V_full


def compute_corpus_stats(*, docs: Iterator[dict], vocab_size: int, k: int) -> CorpusStats:
    """Compute CorpusStats from an iterator of BOW dict rows.

    Each row must have keys 'indices' (list[int]) and 'counts' (list[int]).
    """
    n_docs = 0
    n_codes_sum = 0
    code_total = [0] * vocab_size
    total_tokens = 0
    for row in docs:
        n_docs += 1
        n_codes_sum += sum(row["counts"])
        for idx, cnt in zip(row["indices"], row["counts"]):
            code_total[idx] += cnt
            total_tokens += cnt
    mean_codes = n_codes_sum / max(n_docs, 1)
    marginals = [c / max(total_tokens, 1) for c in code_total]
    return CorpusStats(
        corpus_size_docs=n_docs,
        mean_codes_per_doc=mean_codes,
        k=k,
        v_full=vocab_size,
        code_marginals=marginals,
    )


def write_corpus_stats_sidecar(stats: CorpusStats, out_path: Path, *, v_displayed: int) -> None:
    """Write the small-scalars sidecar. v_displayed is the trimmed-vocab width."""
    out_path.write_text(json.dumps({
        "corpus_size_docs": stats.corpus_size_docs,
        "mean_codes_per_doc": stats.mean_codes_per_doc,
        "k": stats.k,
        "v": int(v_displayed),
        "v_full": stats.v_full,
    }))


def compute_corpus_stats_from_bow_df(bow_df, *, vocab_size: int, k: int) -> CorpusStats:
    """PySpark wrapper. Streams rows to the driver via toLocalIterator."""
    rows = bow_df.select("indices", "counts").toLocalIterator()
    return compute_corpus_stats(
        docs=({"indices": list(r.indices), "counts": list(r.counts)} for r in rows),
        vocab_size=vocab_size,
        k=k,
    )
```

- [ ] **Step 4: Run, expect PASS** — `cd charmpheno && poetry run pytest tests/test_export_corpus_stats.py -v`

- [ ] **Step 5: Commit**

```bash
git add charmpheno/charmpheno/export/corpus_stats.py charmpheno/tests/test_export_corpus_stats.py
git commit -m "feat(export): corpus_stats computation and small-scalars sidecar"
```

---

### Task 2: model.json + vocab.json with top-N trim

Two of the four bundle files. Top-N trim by corpus_freq: keep the N highest-frequency vocab indices, restrict β to those columns, renormalize rows.

**Files:**
- Create: `charmpheno/charmpheno/export/dashboard.py`
- Create: `charmpheno/tests/test_export_dashboard.py`

- [ ] **Step 1: Write the failing test**

```python
# charmpheno/tests/test_export_dashboard.py
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pytest

from charmpheno.export.dashboard import write_model_and_vocab_bundles


def test_top_n_trim_reorders_and_renormalizes(tmp_path: Path):
    K, V = 2, 5
    # β: row 0 mostly on col 4; row 1 mostly on col 1. Other cols low.
    lambda_ = np.array([
        [1, 1, 1, 1, 100],
        [1, 100, 1, 1, 1],
    ], dtype=float)
    alpha = np.array([0.1, 0.1])
    marginals = [0.10, 0.30, 0.01, 0.01, 0.58]   # col 4, then col 1 are top-2
    vocab_ids = [101, 202, 303, 404, 505]
    descriptions = {101: "A", 202: "B", 505: "E"}
    domains = {101: "condition", 202: "drug", 505: "procedure"}

    write_model_and_vocab_bundles(
        out_dir=tmp_path,
        lambda_=lambda_, alpha=alpha,
        vocab_ids=vocab_ids, descriptions=descriptions, domains=domains,
        code_marginals=marginals, top_n=2,
    )

    model = json.loads((tmp_path / "model.json").read_text())
    vocab = json.loads((tmp_path / "vocab.json").read_text())

    # vocab kept top-2 by marginal: col 4 (505), then col 1 (202)
    assert [c["code"] for c in vocab["codes"]] == ["505", "202"]
    assert [c["description"] for c in vocab["codes"]] == ["E", "B"]
    assert [c["corpus_freq"] for c in vocab["codes"]] == pytest.approx([0.58, 0.30])
    # ids are 0..top_N-1
    assert [c["id"] for c in vocab["codes"]] == [0, 1]

    # model V matches trimmed vocab; β rows renormalize
    assert model["K"] == 2
    assert model["V"] == 2
    beta = np.array(model["beta"])
    np.testing.assert_allclose(beta.sum(axis=1), np.ones(2), atol=1e-6)
    # row 0 was concentrated on the kept col 4 → still concentrated there (column 0 of trimmed)
    assert beta[0, 0] > 0.9
    # row 1 was concentrated on col 1 → trimmed-column 1
    assert beta[1, 1] > 0.9


def test_returns_v_displayed(tmp_path: Path):
    K, V = 2, 4
    lambda_ = np.ones((K, V))
    alpha = np.array([0.1, 0.1])
    marginals = [0.4, 0.3, 0.2, 0.1]
    vocab_ids = [10, 20, 30, 40]
    v_disp = write_model_and_vocab_bundles(
        out_dir=tmp_path,
        lambda_=lambda_, alpha=alpha,
        vocab_ids=vocab_ids, descriptions={}, domains={},
        code_marginals=marginals, top_n=10,  # > V, should cap to V
    )
    assert v_disp == V
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement**

```python
# charmpheno/charmpheno/export/dashboard.py
"""Dashboard bundle export. Writes a four-file JSON bundle consumed by
the static Svelte dashboard. Schema defined in
docs/superpowers/specs/2026-05-13-dashboard-design.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _round_floats(arr: np.ndarray, *, decimals: int = 6) -> list:
    return np.round(arr.astype(np.float64), decimals=decimals).tolist()


def select_top_n_indices(code_marginals: list[float], top_n: int) -> list[int]:
    """Return original-vocab indices for the top-N codes by marginal, sorted descending."""
    marg = np.asarray(code_marginals, dtype=np.float64)
    if top_n >= len(marg):
        return list(np.argsort(-marg))
    idx = np.argpartition(-marg, kth=top_n - 1)[:top_n]
    return list(idx[np.argsort(-marg[idx])])


def write_model_and_vocab_bundles(
    *,
    out_dir: Path,
    lambda_: np.ndarray,        # K × V_full
    alpha: np.ndarray,          # length K
    vocab_ids: list[int],       # length V_full; vocab_ids[i] = concept_id at index i
    descriptions: dict[int, str],
    domains: dict[int, str],
    code_marginals: list[float],
    top_n: int,
) -> int:
    """Write model.json and vocab.json. Returns the displayed-vocab width.

    Trims β columns and vocab metadata to the top-N codes by corpus frequency.
    β rows are renormalized so each row sums to 1 over the trimmed columns.
    """
    K, V_full = lambda_.shape
    keep = select_top_n_indices(code_marginals, top_n)
    V_disp = len(keep)
    beta_full = lambda_ / lambda_.sum(axis=1, keepdims=True)
    beta = beta_full[:, keep]
    # renormalize
    row_sums = beta.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    beta = beta / row_sums

    model_payload = {
        "K": int(K),
        "V": int(V_disp),
        "alpha": _round_floats(np.asarray(alpha)),
        "beta": _round_floats(beta),
    }
    (out_dir / "model.json").write_text(json.dumps(model_payload))

    codes = []
    for new_idx, orig_idx in enumerate(keep):
        cid = vocab_ids[orig_idx]
        codes.append({
            "id": new_idx,
            "code": str(cid),
            "description": descriptions.get(cid, ""),
            "domain": domains.get(cid, "unknown"),
            "corpus_freq": float(code_marginals[orig_idx]),
        })
    (out_dir / "vocab.json").write_text(json.dumps({"codes": codes}))

    return V_disp
```

- [ ] **Step 4: Run, expect PASS**

```bash
cd charmpheno && poetry run pytest tests/test_export_dashboard.py -v
```

- [ ] **Step 5: Commit**

```bash
git add charmpheno/charmpheno/export/dashboard.py charmpheno/tests/test_export_dashboard.py
git commit -m "feat(export): model+vocab bundles with top-N corpus-freq trim"
```

---

### Task 3: phenotypes.json

**Files:**
- Modify: `charmpheno/charmpheno/export/dashboard.py`
- Modify: `charmpheno/tests/test_export_dashboard.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_write_phenotypes_bundle(tmp_path: Path):
    from charmpheno.export.dashboard import write_phenotypes_bundle
    out = tmp_path / "phenotypes.json"
    write_phenotypes_bundle(
        out,
        npmi=[0.18, 0.05, -0.10],
        corpus_prevalence=[0.30, 0.40, 0.30],
        labels=["Cardiac", "", ""],
        topic_indices=[0, 1, 2],
        junk_threshold=0.0,
    )
    payload = json.loads(out.read_text())
    assert payload["npmi_threshold"] == 0.0
    assert payload["phenotypes"][0] == {
        "id": 0,
        "label": "Cardiac",
        "npmi": pytest.approx(0.18),
        "corpus_prevalence": pytest.approx(0.30),
        "junk_flag": False,
        "original_topic_id": 0,
    }
    assert payload["phenotypes"][2]["junk_flag"] is True


def test_write_phenotypes_bundle_preserves_hdp_original_indices(tmp_path: Path):
    """For HDP, the displayed phenotype ids are 0..K_display-1 but
    original_topic_id carries the source truncation index."""
    from charmpheno.export.dashboard import write_phenotypes_bundle
    out = tmp_path / "phenotypes.json"
    write_phenotypes_bundle(
        out,
        npmi=[0.2, 0.15],
        corpus_prevalence=[0.4, 0.3],
        labels=None,
        topic_indices=[42, 7],
        junk_threshold=0.0,
    )
    payload = json.loads(out.read_text())
    assert [p["id"] for p in payload["phenotypes"]] == [0, 1]
    assert [p["original_topic_id"] for p in payload["phenotypes"]] == [42, 7]
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement**

Append to `dashboard.py`:

```python
def write_phenotypes_bundle(
    out_path: Path,
    *,
    npmi: list[float],
    corpus_prevalence: list[float],
    topic_indices: list[int] | None = None,
    labels: list[str] | None = None,
    junk_threshold: float = 0.0,
) -> None:
    """Write phenotypes.json.

    topic_indices[k] is the original model-side topic id for displayed
    phenotype k. For LDA the adapter passes 0..K-1; for HDP it passes
    the mask-filtered truncation indices so the advanced view can
    surface them.
    """
    K = len(npmi)
    labels = labels or [""] * K
    if topic_indices is None:
        topic_indices = list(range(K))
    phenotypes = [
        {
            "id": k,
            "label": labels[k],
            "npmi": float(npmi[k]),
            "corpus_prevalence": float(corpus_prevalence[k]),
            "junk_flag": bool(npmi[k] < junk_threshold),
            "original_topic_id": int(topic_indices[k]),
        }
        for k in range(K)
    ]
    out_path.write_text(json.dumps({
        "phenotypes": phenotypes,
        "npmi_threshold": float(junk_threshold),
    }))
```

- [ ] **Step 4: Run, expect PASS**

- [ ] **Step 5: Commit**

```bash
git add charmpheno/charmpheno/export/dashboard.py charmpheno/tests/test_export_dashboard.py
git commit -m "feat(export): write_phenotypes_bundle"
```

---

### Task 4: Model-class adapter + end-to-end driver script

`charmpheno.export.model_adapter` normalizes any supported VIResult to a uniform `DashboardExport`. The driver dispatches on `metadata["model_class"]` and delegates the model-specific computation (HDP topic filtering, LDA gamma row-means, etc.) to the adapter.

**Files:**
- Create: `charmpheno/charmpheno/export/model_adapter.py`
- Create: `charmpheno/tests/test_export_model_adapter.py`
- Create: `analysis/local/build_dashboard.py`
- Create: `tests/scripts/test_build_dashboard_smoke.py`

- [ ] **Step 1: Adapter tests**

```python
# charmpheno/tests/test_export_model_adapter.py
from __future__ import annotations
import numpy as np
import pytest

from charmpheno.export.model_adapter import (
    DashboardExport, adapt, adapt_lda, adapt_hdp,
)


def _lda_result(K: int = 3, V: int = 5):
    from spark_vi.core.result import VIResult
    rng = np.random.RandomState(0)
    lambda_ = rng.rand(K, V) + 0.5
    alpha = np.full(K, 0.1)
    gamma = rng.rand(50, K) + 0.1  # 50 docs
    return VIResult(
        global_params={"lambda": lambda_, "alpha": alpha, "gamma": gamma},
        elbo_trace=[1.0], n_iterations=1, converged=True,
        metadata={"model_class": "lda"},
    )


def _hdp_result(T: int = 8, V: int = 5):
    from spark_vi.core.result import VIResult
    rng = np.random.RandomState(1)
    lambda_ = rng.rand(T, V) + 0.5
    # u, v shape (T,) — stick parameters. Pick u,v so first 3 sticks dominate.
    u = np.array([10.0, 8.0, 6.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    v = np.array([1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    return VIResult(
        global_params={"lambda": lambda_, "u": u, "v": v},
        elbo_trace=[1.0], n_iterations=1, converged=True,
        metadata={"model_class": "hdp"},
    )


def test_adapt_lda_identity_shapes():
    result = _lda_result(K=3, V=5)
    exp = adapt_lda(result)
    assert isinstance(exp, DashboardExport)
    assert exp.beta.shape == (3, 5)
    assert exp.alpha.shape == (3,)
    assert exp.corpus_prevalence.shape == (3,)
    assert list(exp.topic_indices) == [0, 1, 2]
    np.testing.assert_allclose(exp.beta.sum(axis=1), np.ones(3), atol=1e-6)


def test_adapt_lda_falls_back_to_alpha_when_gamma_missing():
    from spark_vi.core.result import VIResult
    lambda_ = np.ones((2, 4))
    result = VIResult(
        global_params={"lambda": lambda_, "alpha": np.array([0.3, 0.7])},
        elbo_trace=[], n_iterations=0, converged=False,
        metadata={"model_class": "lda"},
    )
    exp = adapt_lda(result)
    assert exp.corpus_prevalence == pytest.approx([0.3, 0.7])


def test_adapt_hdp_filters_to_top_k():
    result = _hdp_result(T=8, V=5)
    exp = adapt_hdp(result, top_k=3)
    assert exp.beta.shape == (3, 5)
    assert exp.alpha.shape == (3,)
    assert len(exp.topic_indices) == 3
    # original indices are in [0, T); first three sticks (highest u/(u+v)) should be selected
    assert set(exp.topic_indices) == {0, 1, 2}
    np.testing.assert_allclose(exp.alpha.sum(), 1.0, atol=1e-5)


def test_adapt_dispatches_on_model_class():
    lda = _lda_result(K=3)
    assert adapt(lda).beta.shape[0] == 3
    hdp = _hdp_result(T=8)
    assert adapt(hdp, hdp_top_k=2).beta.shape[0] == 2


def test_adapt_unknown_class_raises():
    from spark_vi.core.result import VIResult
    bad = VIResult(
        global_params={}, elbo_trace=[], n_iterations=0, converged=False,
        metadata={"model_class": "ctm"},
    )
    with pytest.raises(ValueError, match="unsupported model class"):
        adapt(bad)
```

- [ ] **Step 2: Adapter implementation**

```python
# charmpheno/charmpheno/export/model_adapter.py
"""Model-class adapters for the dashboard export.

Each supported VIResult normalizes to a DashboardExport, so the export
builder and the dashboard contract are model-class-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DashboardExport:
    """Uniform shape consumed by the dashboard bundle builder."""
    beta: np.ndarray              # K_display × V (row-stochastic)
    alpha: np.ndarray             # K_display
    corpus_prevalence: np.ndarray # K_display
    topic_indices: np.ndarray     # K_display (original model-side topic ids)


def _global_params(result) -> dict[str, np.ndarray]:
    return result.global_params


def _model_class(result) -> str:
    return str(result.metadata.get("model_class", "lda")).lower()


def adapt_lda(result) -> DashboardExport:
    """LDA → DashboardExport. Identity on β; corpus_prevalence from gamma."""
    gp = _global_params(result)
    lambda_ = np.asarray(gp["lambda"], dtype=np.float64)
    K = lambda_.shape[0]
    beta = lambda_ / lambda_.sum(axis=1, keepdims=True)
    alpha = np.asarray(gp.get("alpha", np.full(K, 1.0 / K)), dtype=np.float64)
    gamma = gp.get("gamma")
    if gamma is not None:
        gamma = np.asarray(gamma, dtype=np.float64)
        if gamma.ndim == 2 and gamma.shape[1] == K:
            theta = gamma / gamma.sum(axis=1, keepdims=True)
            corpus_prev = theta.mean(axis=0)
        else:
            corpus_prev = alpha / alpha.sum()
    else:
        corpus_prev = alpha / alpha.sum()
    return DashboardExport(
        beta=beta,
        alpha=alpha,
        corpus_prevalence=corpus_prev,
        topic_indices=np.arange(K, dtype=np.int64),
    )


def adapt_hdp(result, *, top_k: int = 50) -> DashboardExport:
    """HDP → DashboardExport. Filters to top-K used topics; computes
    effective Dirichlet α from the corpus-level GEM sticks."""
    gp = _global_params(result)
    lambda_ = np.asarray(gp["lambda"], dtype=np.float64)
    u = np.asarray(gp["u"], dtype=np.float64)
    v = np.asarray(gp["v"], dtype=np.float64)
    # E[stick weights] from Beta(u, v) and stick-breaking remainder
    stick_means = u / (u + v)
    remainder = np.cumprod(np.concatenate([[1.0], 1 - stick_means[:-1]]))
    e_beta = stick_means * remainder  # corpus-level mass per truncation index
    # top_k by E[beta]
    K_use = min(top_k, len(e_beta))
    order = np.argsort(-e_beta)[:K_use]
    order = np.sort(order)  # stable order by original index
    beta_filt = lambda_[order] / lambda_[order].sum(axis=1, keepdims=True)
    # Effective Dirichlet alpha: renormalize the selected sticks to sum to 1
    sel = e_beta[order]
    alpha_eff = sel / sel.sum() if sel.sum() > 0 else np.full(K_use, 1.0 / K_use)
    return DashboardExport(
        beta=beta_filt,
        alpha=alpha_eff,
        corpus_prevalence=alpha_eff.copy(),
        topic_indices=order.astype(np.int64),
    )


def adapt(result, *, hdp_top_k: int = 50) -> DashboardExport:
    mc = _model_class(result)
    if mc == "lda":
        return adapt_lda(result)
    if mc == "hdp":
        return adapt_hdp(result, top_k=hdp_top_k)
    raise ValueError(f"unsupported model class: {mc}")
```

- [ ] **Step 3: Run adapter tests** — `cd charmpheno && poetry run pytest tests/test_export_model_adapter.py -v`. PASS.

- [ ] **Step 4: Driver script (delegates to adapter)**

```python
# analysis/local/build_dashboard.py
"""Build the dashboard data bundle from a saved VIResult (LDA or HDP).

Outputs four JSON files into the target directory:
  model.json, phenotypes.json, vocab.json, corpus_stats.json

Model-class normalization happens in charmpheno.export.model_adapter.
Synthetic cohorts and topic-map MDS are computed client-side.

Usage:
    poetry run python analysis/local/build_dashboard.py \\
        --checkpoint data/runs/<run> \\
        --input data/simulated/omop_N10000_seed42.parquet \\
        --out-dir dashboard/public/data \\
        --vocab-top-n 5000 \\
        --hdp-top-k 50
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
from pyspark.sql import SparkSession

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from charmpheno.export.corpus_stats import (
    compute_corpus_stats_from_bow_df,
    write_corpus_stats_sidecar,
)
from charmpheno.export.dashboard import (
    write_model_and_vocab_bundles,
    write_phenotypes_bundle,
)
from charmpheno.export.model_adapter import adapt
from charmpheno.omop import DocSpec, load_omop_parquet, to_bow_dataframe
from spark_vi.io import load_result
from spark_vi.models.topic.types import BOWDocument
from spark_vi.eval.topic import compute_npmi_coherence

log = logging.getLogger(__name__)


def _build_spark() -> SparkSession:
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    return (
        SparkSession.builder.master("local[2]")
        .appName("build_dashboard")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
        .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
        .getOrCreate()
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--vocab-top-n", type=int, default=5000)
    parser.add_argument("--hdp-top-k", type=int, default=50,
                        help="Top-K used HDP topics (ignored for LDA)")
    parser.add_argument("--top-n-codes-for-npmi", type=int, default=20)
    parser.add_argument("--junk-threshold", type=float, default=0.0)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    result = load_result(args.checkpoint)
    model_class = result.metadata.get("model_class", "lda")
    log.info("model_class=%s", model_class)

    # Adapter normalizes LDA/HDP/etc. to a uniform DashboardExport
    export = adapt(result, hdp_top_k=args.hdp_top_k)
    K_disp, V_full = export.beta.shape
    log.info("K_display=%d V_full=%d (model_class=%s)", K_disp, V_full, model_class)

    vocab_ids = result.metadata.get("vocab")
    if not vocab_ids:
        raise SystemExit("checkpoint metadata has no 'vocab'; re-fit needed.")
    descriptions = result.metadata.get("concept_names", {}) or {}
    domains = result.metadata.get("concept_domains", {}) or {}

    # Stats from the input parquet
    corpus_manifest = result.metadata.get("corpus_manifest", {})
    doc_spec_manifest = corpus_manifest.get("doc_spec", {"name": "patient"})
    doc_spec = DocSpec.from_manifest(doc_spec_manifest)
    spark = _build_spark()
    df = load_omop_parquet(str(args.input), spark=spark)
    bow_df, _ = to_bow_dataframe(df, doc_spec=doc_spec, vocab=vocab_ids)
    bow_df = bow_df.persist()
    stats = compute_corpus_stats_from_bow_df(bow_df, vocab_size=V_full, k=K_disp)
    log.info("corpus stats: n_docs=%d mean_codes=%.2f",
             stats.corpus_size_docs, stats.mean_codes_per_doc)

    # NPMI on the adapter's displayed-topic β (already filtered for HDP)
    holdout_bow = bow_df.rdd.map(BOWDocument.from_spark_row)
    report = compute_npmi_coherence(export.beta, holdout_bow, top_n=args.top_n_codes_for_npmi)
    npmi = report.per_topic_npmi.tolist()
    bow_df.unpersist()
    spark.stop()

    # write_model_and_vocab_bundles expects (lambda_, alpha) pre-trim.
    # The adapter's β is already row-stochastic; reconstruct a faux-lambda
    # that produces the same beta after the normalize step inside the writer.
    # (Simplest: multiply by a big scalar so renormalize is identity.)
    pseudo_lambda = export.beta * 1.0e6  # any positive scalar; row-norm is identity
    v_disp = write_model_and_vocab_bundles(
        out_dir=args.out_dir,
        lambda_=pseudo_lambda, alpha=export.alpha,
        vocab_ids=vocab_ids, descriptions=descriptions, domains=domains,
        code_marginals=stats.code_marginals, top_n=args.vocab_top_n,
    )
    write_phenotypes_bundle(
        args.out_dir / "phenotypes.json",
        npmi=npmi,
        corpus_prevalence=export.corpus_prevalence.tolist(),
        topic_indices=export.topic_indices.tolist(),
        labels=None,
        junk_threshold=args.junk_threshold,
    )
    write_corpus_stats_sidecar(stats, args.out_dir / "corpus_stats.json", v_displayed=v_disp)

    log.info("wrote 4 files to %s (V_disp=%d K_disp=%d)", args.out_dir, v_disp, K_disp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke test**

```python
# tests/scripts/test_build_dashboard_smoke.py
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.slow


def test_build_dashboard_smoke(tmp_path: Path):
    from spark_vi.core.result import VIResult
    from spark_vi.io import save_result

    K, V = 4, 12
    lambda_ = np.random.RandomState(0).rand(K, V) + 0.5
    alpha = np.full(K, 0.1)
    vocab = [1000 + i for i in range(V)]
    result = VIResult(
        global_params={"lambda": lambda_, "alpha": alpha},
        elbo_trace=[1.0, 2.0, 3.0], n_iterations=3, converged=True,
        metadata={
            "vocab": vocab,
            "concept_names": {1000: "Atrial fibrillation"},
            "concept_domains": {1000: "condition"},
            "corpus_manifest": {"doc_spec": {"name": "patient"}},
        },
    )
    ckpt = tmp_path / "ckpt"
    save_result(result, ckpt)
    parquet = Path("data/simulated/omop_N10000_seed42.parquet")
    assert parquet.exists(), "fixture parquet missing; run `make data`"
    out = tmp_path / "data"
    subprocess.check_call([
        sys.executable, "analysis/local/build_dashboard.py",
        "--checkpoint", str(ckpt),
        "--input", str(parquet),
        "--out-dir", str(out),
        "--vocab-top-n", "8",
    ])
    assert {p.name for p in out.iterdir()} == {
        "model.json", "vocab.json", "phenotypes.json", "corpus_stats.json"
    }
    stats = json.loads((out / "corpus_stats.json").read_text())
    assert stats["k"] == 4 and stats["v"] == 8 and stats["v_full"] == 12
    model = json.loads((out / "model.json").read_text())
    assert model["V"] == 8
    np.testing.assert_allclose(np.array(model["beta"]).sum(axis=1), np.ones(4), atol=1e-5)
    # Adapter populated original_topic_id (identity for LDA)
    phenos = json.loads((out / "phenotypes.json").read_text())["phenotypes"]
    assert [p["original_topic_id"] for p in phenos] == [0, 1, 2, 3]
```

- [ ] **Step 3: Run** — `poetry run pytest tests/scripts/test_build_dashboard_smoke.py -v -m slow`

- [ ] **Step 4: Commit**

```bash
git add analysis/local/build_dashboard.py tests/scripts/test_build_dashboard_smoke.py
git commit -m "feat(dashboard): build_dashboard driver writing 4-file bundle"
```

---

### Task 5: Dev-fixture bundle generator + real-bundle verification

Two paths for producing a bundle:
- **Dev fixture:** a tiny, fast, no-Spark synthetic bundle that lets dashboard development proceed without a real model. Useful for Phases 2-7.
- **Real bundle:** the actual export from a recent checkpoint, for the salesmanship deployment.

**Files:**
- Create: `scripts/make_dev_bundle.py`

- [ ] **Step 1: Write the dev-fixture generator**

```python
# scripts/make_dev_bundle.py
"""Produce a tiny, schema-conformant dashboard bundle without Spark or a
real checkpoint. Use for dashboard development when the real export path
is too slow or unavailable.

Usage:
    python scripts/make_dev_bundle.py --out-dir dashboard/public/data \\
        --k 10 --v 200 --seed 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _round(x, d: int = 6):
    return np.round(np.asarray(x, dtype=np.float64), d).tolist()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--v", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Synthetic β: each topic has a "peak" cluster of vocab indices it favors.
    beta = np.full((args.k, args.v), 0.005)
    for k in range(args.k):
        peak = rng.choice(args.v, size=max(3, args.v // (args.k * 2)), replace=False)
        beta[k, peak] += rng.uniform(0.5, 2.0, size=peak.shape)
    beta = beta / beta.sum(axis=1, keepdims=True)
    alpha = np.full(args.k, 0.1)

    # marginals: a Dirichlet-distributed marginal, decreasing for top-N realism
    raw = rng.gamma(1.0, 1.0, size=args.v) * np.linspace(2.0, 0.5, args.v)
    marginals = raw / raw.sum()

    # model.json (no trimming here — dev bundle ships full V)
    (args.out_dir / "model.json").write_text(json.dumps({
        "K": args.k, "V": args.v, "alpha": _round(alpha), "beta": _round(beta),
    }))

    # vocab.json
    domains_pool = ["condition", "drug", "procedure", "measurement", "observation"]
    codes = []
    for i in range(args.v):
        codes.append({
            "id": i,
            "code": f"DEV{i:04d}",
            "description": f"Synthetic code {i}",
            "domain": domains_pool[i % len(domains_pool)],
            "corpus_freq": float(marginals[i]),
        })
    (args.out_dir / "vocab.json").write_text(json.dumps({"codes": codes}))

    # phenotypes.json (NPMI is fake; spread around 0 with one junk-flagged)
    npmi = rng.normal(0.15, 0.08, size=args.k)
    npmi[-1] = -0.1  # one junk for the badge UI
    corpus_prev = rng.dirichlet(alpha=np.full(args.k, 2.0))
    (args.out_dir / "phenotypes.json").write_text(json.dumps({
        "phenotypes": [
            {
                "id": k,
                "label": "",
                "npmi": float(npmi[k]),
                "corpus_prevalence": float(corpus_prev[k]),
                "junk_flag": bool(npmi[k] < 0),
                "original_topic_id": k,
            }
            for k in range(args.k)
        ],
        "npmi_threshold": 0.0,
    }))

    # corpus_stats.json
    (args.out_dir / "corpus_stats.json").write_text(json.dumps({
        "corpus_size_docs": 50000,
        "mean_codes_per_doc": 18.0,
        "k": args.k,
        "v": args.v,
        "v_full": args.v,
    }))
    print(f"wrote dev bundle to {args.out_dir} (K={args.k}, V={args.v})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Build the dev bundle**

```bash
python scripts/make_dev_bundle.py --out-dir dashboard/public/data --k 10 --v 200 --seed 0
ls -la dashboard/public/data/
```

Expected: four JSON files; bundle total well under 1 MB; runs in <1 second.

- [ ] **Step 3 (optional, when a real checkpoint is available): build a real bundle**

```bash
poetry run python analysis/local/build_dashboard.py \
    --checkpoint data/runs/<chosen> \
    --input data/simulated/omop_N10000_seed42.parquet \
    --out-dir dashboard/public/data \
    --vocab-top-n 5000
```

For an HDP checkpoint, add `--hdp-top-k 50`. Verify schemas:

```bash
for f in model.json vocab.json phenotypes.json corpus_stats.json; do
  echo "=== $f ==="; python -c "import json; d=json.load(open('dashboard/public/data/$f')); print(list(d.keys()))"
done
```

- [ ] **Step 4: Smoke test for the dev bundle**

Add a unit-tier test in the existing scripts test directory:

```python
# tests/scripts/test_make_dev_bundle.py
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path


def test_make_dev_bundle_emits_four_conformant_files(tmp_path: Path):
    out = tmp_path / "data"
    subprocess.check_call([
        sys.executable, "scripts/make_dev_bundle.py",
        "--out-dir", str(out), "--k", "5", "--v", "20", "--seed", "1",
    ])
    assert {p.name for p in out.iterdir()} == {
        "model.json", "vocab.json", "phenotypes.json", "corpus_stats.json"
    }
    model = json.loads((out / "model.json").read_text())
    assert model["K"] == 5 and model["V"] == 20
    phenos = json.loads((out / "phenotypes.json").read_text())["phenotypes"]
    assert all("original_topic_id" in p for p in phenos)
```

Run: `poetry run pytest tests/scripts/test_make_dev_bundle.py -v`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/make_dev_bundle.py tests/scripts/test_make_dev_bundle.py dashboard/public/data/
git commit -m "feat(dashboard): dev-fixture bundle generator + seed dashboard/public/data"
```

**Phase 1 complete.** Bundle exists (dev or real); Phase 2 builds against it.

---

## Phase 2: Dashboard Scaffold

### Task 6: Scaffold Svelte 5 + Vite + D3 + Vitest

**Files:**
- Create: `dashboard/package.json`, `dashboard/vite.config.ts`, `dashboard/tsconfig.json`, `dashboard/index.html`, `dashboard/src/main.ts`, `dashboard/src/App.svelte`, `dashboard/.gitignore`, `dashboard/Makefile`

- [ ] **Step 1: Initialize**

```bash
cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno
mkdir -p dashboard
cd dashboard
npm create vite@latest . -- --template svelte-ts
npm install
npm install d3 @types/d3
npm install -D vitest jsdom @testing-library/svelte
```

If Vite warns about a non-empty directory (the `public/data/` from Task 5), allow it to scaffold without overwriting that subdirectory.

- [ ] **Step 2: Configure Vite + Vitest**

Replace `dashboard/vite.config.ts`:

```ts
import { defineConfig } from 'vitest/config'
import { svelte } from '@sveltejs/vite-plugin-svelte'

export default defineConfig({
  plugins: [svelte()],
  base: process.env.VITE_BASE ?? '/CHARMPheno/',
  test: {
    environment: 'jsdom',
    globals: true,
  },
})
```

Add to `dashboard/package.json` `"scripts"`:

```json
    "test": "vitest run",
    "test:watch": "vitest"
```

- [ ] **Step 3: Add a Makefile**

```makefile
.PHONY: dev build build-local preview test clean deploy-local

# Vite dev server. Default base ("/") so localhost paths just work.
dev:
	VITE_BASE=/ npm run dev

# Production build (uses VITE_BASE=/CHARMPheno/ for GH Pages).
build:
	VITE_BASE=/CHARMPheno/ npm run build

# Local production build (serves at "/"). Use when you want to preview a built
# bundle without setting up a /CHARMPheno/ mount.
build-local:
	VITE_BASE=/ npm run build

# Vite preview server against the local build. Surfaces base-URL issues the
# dev server hides.
preview: build-local
	npm run preview -- --port 4173

test:
	npm test

clean:
	rm -rf dist node_modules

# Optional: manual push to gh-pages without going through CI. Requires gh-pages.
deploy-local: build
	npx gh-pages -d dist
```

Then add the `preview` script to `dashboard/package.json` if Vite didn't already include it:

```json
    "preview": "vite preview"
```

- [ ] **Step 4: Verify dev server starts**

```bash
cd dashboard && make dev
```

Open the printed URL; expected: the default Svelte template renders. Ctrl-C to stop.

- [ ] **Step 5: Commit**

```bash
cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno
git add dashboard/
git commit -m "feat(dashboard): scaffold Svelte 5 + Vite + D3 + Vitest"
```

---

### Task 7: Bundle types + loader + stores

Loader fetches the four JSON files; stores expose them to components.

**Files:**
- Create: `dashboard/src/lib/types.ts`
- Create: `dashboard/src/lib/bundle.ts`
- Create: `dashboard/src/lib/store.ts`
- Create: `dashboard/src/lib/bundle.test.ts`

- [ ] **Step 1: Types**

```ts
// dashboard/src/lib/types.ts
export interface Model { K: number; V: number; alpha: number[]; beta: number[][] }
export interface Phenotype {
  id: number; label: string; npmi: number; corpus_prevalence: number;
  junk_flag: boolean; original_topic_id: number
}
export interface PhenotypesBundle { phenotypes: Phenotype[]; npmi_threshold: number }
export interface VocabCode {
  id: number; code: string; description: string; domain: string; corpus_freq: number
}
export interface VocabBundle { codes: VocabCode[] }
export interface CorpusStats {
  corpus_size_docs: number; mean_codes_per_doc: number; k: number; v: number; v_full: number
}
export interface DashboardBundle {
  model: Model; phenotypes: PhenotypesBundle; vocab: VocabBundle; corpusStats: CorpusStats
}

// In-memory only; not part of the bundle:
export interface SyntheticPatient {
  id: string; theta: number[]; code_bag: number[]; neighbors: string[]
}
export interface SyntheticCohort {
  patients: SyntheticPatient[]; seed: number
}
```

- [ ] **Step 2: Loader**

```ts
// dashboard/src/lib/bundle.ts
import type {
  DashboardBundle, Model, PhenotypesBundle, VocabBundle, CorpusStats,
} from './types'

async function fetchJson<T>(url: string): Promise<T> {
  const r = await fetch(url)
  if (!r.ok) throw new Error(`failed to load ${url}: ${r.status}`)
  return r.json() as Promise<T>
}

export async function loadBundle(baseUrl: string): Promise<DashboardBundle> {
  const base = baseUrl.endsWith('/') ? baseUrl : baseUrl + '/'
  const [model, phenotypes, vocab, corpusStats] = await Promise.all([
    fetchJson<Model>(`${base}data/model.json`),
    fetchJson<PhenotypesBundle>(`${base}data/phenotypes.json`),
    fetchJson<VocabBundle>(`${base}data/vocab.json`),
    fetchJson<CorpusStats>(`${base}data/corpus_stats.json`),
  ])
  return { model, phenotypes, vocab, corpusStats }
}
```

- [ ] **Step 3: Stores**

```ts
// dashboard/src/lib/store.ts
import { writable, derived } from 'svelte/store'
import type { DashboardBundle, SyntheticCohort } from './types'

export const bundle = writable<DashboardBundle | null>(null)
export const cohort = writable<SyntheticCohort | null>(null)

export const selectedPhenotypeId = writable<number | null>(null)
export const selectedPatientId = writable<string | null>(null)
export const simulatorPrefix = writable<number[]>([])     // vocab indices (trimmed)
export const advancedView = writable<boolean>(false)
export const colorMode = writable<'npmi' | 'prevalence'>('npmi')
export const hoveredCodeIdx = writable<number | null>(null)

export const phenotypesById = derived(bundle, ($b) =>
  $b ? new Map($b.phenotypes.phenotypes.map((p) => [p.id, p])) : new Map()
)

export const patientsById = derived(cohort, ($c) =>
  $c ? new Map($c.patients.map((p) => [p.id, p])) : new Map()
)
```

- [ ] **Step 4: Loader test**

```ts
// dashboard/src/lib/bundle.test.ts
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { loadBundle } from './bundle'

describe('loadBundle', () => {
  beforeEach(() => {
    global.fetch = vi.fn((url: string) => {
      const stubs: Record<string, unknown> = {
        'data/model.json':         { K: 2, V: 3, alpha: [0.1, 0.1], beta: [[0.5, 0.3, 0.2], [0.1, 0.6, 0.3]] },
        'data/phenotypes.json':    { phenotypes: [], npmi_threshold: 0 },
        'data/vocab.json':         { codes: [] },
        'data/corpus_stats.json':  { corpus_size_docs: 10, mean_codes_per_doc: 5, k: 2, v: 3, v_full: 3 },
      }
      const key = Object.keys(stubs).find((k) => url.endsWith(k))!
      return Promise.resolve({ ok: true, json: () => Promise.resolve(stubs[key]) } as Response)
    }) as any
  })

  it('loads all four files', async () => {
    const b = await loadBundle('/')
    expect(b.model.K).toBe(2)
    expect(b.corpusStats.v_full).toBe(3)
  })
})
```

Run: `cd dashboard && npm test`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/lib/
git commit -m "feat(dashboard): bundle types, loader, stores"
```

---

### Task 8: Hash router + tab shell

**Files:**
- Modify: `dashboard/src/App.svelte`
- Create: `dashboard/src/lib/router.ts`, `dashboard/src/lib/Tabs.svelte`, `dashboard/src/lib/tabs/{Atlas,Patient,Simulator}.svelte`

- [ ] **Step 1: Router**

```ts
// dashboard/src/lib/router.ts
import { writable } from 'svelte/store'

export type Route = 'atlas' | 'patient' | 'simulator'

function parseHash(): Route {
  const h = window.location.hash.replace(/^#\//, '')
  if (h === 'patient' || h === 'simulator') return h
  return 'atlas'
}

export const route = writable<Route>(parseHash())
window.addEventListener('hashchange', () => route.set(parseHash()))
export function go(to: Route): void { window.location.hash = `#/${to}` }
```

- [ ] **Step 2: Tab shell**

```svelte
<!-- dashboard/src/lib/Tabs.svelte -->
<script lang="ts">
  import { route, go } from './router'
  const tabs: { id: 'atlas' | 'patient' | 'simulator'; label: string }[] = [
    { id: 'atlas', label: 'Phenotype Atlas' },
    { id: 'patient', label: 'Patient Explorer' },
    { id: 'simulator', label: 'Simulator' },
  ]
</script>

<nav class="tabs">
  {#each tabs as t}
    <button class:active={$route === t.id} on:click={() => go(t.id)}>{t.label}</button>
  {/each}
</nav>

<style>
  .tabs { display: flex; gap: 0.5rem; padding: 0.5rem 1rem; border-bottom: 1px solid #ddd; }
  button { padding: 0.5rem 1rem; border: 1px solid transparent; background: transparent; cursor: pointer; }
  button.active { border-color: #999; background: #f4f4f4; font-weight: 600; }
</style>
```

- [ ] **Step 3: Placeholder tabs**

```svelte
<!-- dashboard/src/lib/tabs/Atlas.svelte --> <h2>Phenotype Atlas</h2> <p>Placeholder.</p>
<!-- dashboard/src/lib/tabs/Patient.svelte --> <h2>Patient Explorer</h2> <p>Placeholder.</p>
<!-- dashboard/src/lib/tabs/Simulator.svelte --> <h2>Simulator</h2> <p>Placeholder.</p>
```

Each in its own file, with the above as the complete content.

- [ ] **Step 4: App.svelte wires the loader + tabs + header + advanced-view toggle**

```svelte
<script lang="ts">
  import { onMount } from 'svelte'
  import { bundle, advancedView } from './lib/store'
  import { loadBundle } from './lib/bundle'
  import { route } from './lib/router'
  import Tabs from './lib/Tabs.svelte'
  import Atlas from './lib/tabs/Atlas.svelte'
  import Patient from './lib/tabs/Patient.svelte'
  import Simulator from './lib/tabs/Simulator.svelte'

  let error: string | null = null
  onMount(async () => {
    try { bundle.set(await loadBundle(import.meta.env.BASE_URL)) }
    catch (e) { error = (e as Error).message }
  })
</script>

<main>
  <header>
    <h1>CharmPheno</h1>
    <span class="badge">demo · synthetic patients</span>
    {#if $bundle}
      <span class="meta">K = {$bundle.model.K} · V = {$bundle.model.V} (of {$bundle.corpusStats.v_full}) · corpus ≈ {($bundle.corpusStats.corpus_size_docs / 1000).toFixed(0)}k docs</span>
    {/if}
    <span class="spacer" />
    <label class="toggle"><input type="checkbox" bind:checked={$advancedView} /> Advanced view</label>
  </header>

  {#if error}<p class="error">Failed to load bundle: {error}</p>
  {:else if !$bundle}<p>Loading model bundle…</p>
  {:else}
    <Tabs />
    {#if $route === 'atlas'}<Atlas />{:else if $route === 'patient'}<Patient />{:else}<Simulator />{/if}
  {/if}
</main>

<style>
  main { font-family: system-ui, sans-serif; max-width: 1400px; margin: 0 auto; }
  header { display: flex; align-items: center; gap: 1rem; padding: 1rem; border-bottom: 1px solid #ddd; }
  header h1 { margin: 0; font-size: 1.25rem; }
  .badge { font-size: 0.75rem; padding: 0.15rem 0.5rem; background: #fff3cd; border: 1px solid #d4a017; border-radius: 4px; }
  .meta { font-size: 0.75rem; color: #555; }
  .spacer { flex: 1; }
  .toggle { font-size: 0.85rem; display: flex; align-items: center; gap: 0.25rem; }
  .error { color: #b00020; padding: 1rem; }
</style>
```

- [ ] **Step 5: Verify in browser**

```bash
cd dashboard && make dev
```

Header shows model metadata, three tabs route via hash, Advanced view toggles (state visible only via DevTools for now).

- [ ] **Step 6: Commit**

```bash
git add dashboard/src/
git commit -m "feat(dashboard): hash router, tab shell, advanced-view toggle"
```

**Phase 2 complete.**

---

## Phase 3: TypeScript Math + Cohort Generation

### Task 9: Variational E-step

**Files:**
- Create: `dashboard/src/lib/inference.ts`, `dashboard/src/lib/inference.test.ts`

- [ ] **Step 1: Tests**

```ts
// dashboard/src/lib/inference.test.ts
import { describe, it, expect } from 'vitest'
import { variationalEStep } from './inference'

describe('variationalEStep', () => {
  it('returns alpha-normalized theta when prefix is empty', () => {
    const alpha = [0.1, 0.1, 0.1]
    const beta = [[0.5, 0.5], [0.1, 0.9], [0.9, 0.1]]
    const { theta } = variationalEStep({ alpha, beta, codeCounts: new Map() })
    expect(theta[0]).toBeCloseTo(1 / 3)
  })

  it('shifts theta toward the topic owning a heavily observed code', () => {
    const alpha = [0.1, 0.1, 0.1]
    const beta = [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]
    const { theta } = variationalEStep({ alpha, beta, codeCounts: new Map([[0, 10]]) })
    expect(theta[0]).toBeGreaterThan(0.8)
  })

  it('theta sums to 1', () => {
    const alpha = [0.5, 0.5]
    const beta = [[0.3, 0.4, 0.3], [0.5, 0.2, 0.3]]
    const { theta } = variationalEStep({ alpha, beta, codeCounts: new Map([[0, 2], [2, 1]]) })
    expect(theta.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6)
  })
})
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement**

```ts
// dashboard/src/lib/inference.ts
export function digamma(x: number): number {
  let result = 0
  while (x < 6) { result -= 1 / x; x += 1 }
  result += Math.log(x) - 1 / (2 * x)
  const xx = 1 / (x * x)
  result -= xx * ((1 / 12) - xx * ((1 / 120) - xx * (1 / 252)))
  return result
}

export interface EStepInput {
  alpha: number[]
  beta: number[][]
  codeCounts: Map<number, number>
  maxIter?: number
  tol?: number
}
export interface EStepResult { theta: number[]; gamma: number[]; iterations: number }

export function variationalEStep(input: EStepInput): EStepResult {
  const { alpha, beta, codeCounts } = input
  const maxIter = input.maxIter ?? 50
  const tol = input.tol ?? 1e-4
  const K = alpha.length
  const entries = Array.from(codeCounts.entries())

  let gamma = alpha.slice()
  if (entries.length === 0) {
    const sum = gamma.reduce((a, b) => a + b, 0) || 1
    return { theta: gamma.map((g) => g / sum), gamma, iterations: 0 }
  }
  let prevGamma = gamma.slice()
  let it = 0
  for (; it < maxIter; it++) {
    const gammaSum = gamma.reduce((a, b) => a + b, 0)
    const psiSum = digamma(gammaSum)
    const eLogTheta = gamma.map((g) => digamma(g) - psiSum)
    const newGamma = alpha.slice()
    for (const [w, c] of entries) {
      const phi = new Array<number>(K)
      let phiSum = 0
      for (let k = 0; k < K; k++) {
        phi[k] = beta[k][w] * Math.exp(eLogTheta[k])
        phiSum += phi[k]
      }
      if (phiSum === 0) continue
      for (let k = 0; k < K; k++) newGamma[k] += c * (phi[k] / phiSum)
    }
    const delta = newGamma.reduce((a, g, k) => a + Math.abs(g - prevGamma[k]), 0)
    prevGamma = gamma
    gamma = newGamma
    if (delta < tol * K) { it++; break }
  }
  const gammaSum = gamma.reduce((a, b) => a + b, 0) || 1
  return { theta: gamma.map((g) => g / gammaSum), gamma, iterations: it }
}
```

- [ ] **Step 4: Run, expect PASS** — `cd dashboard && npm test`

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/lib/inference.ts dashboard/src/lib/inference.test.ts
git commit -m "feat(inference): variational E-step"
```

---

### Task 10: λ-relevance reranking

**Files:**
- Modify: `dashboard/src/lib/inference.ts`, `dashboard/src/lib/inference.test.ts`

- [ ] **Step 1: Tests** — Append:

```ts
import { relevance, topRelevantCodes } from './inference'

describe('relevance', () => {
  it('λ=1 returns log p(w|k)', () => {
    expect(relevance(0.4, 0.2, 1.0)).toBeCloseTo(Math.log(0.4))
  })
  it('λ=0 returns log lift', () => {
    expect(relevance(0.4, 0.2, 0.0)).toBeCloseTo(Math.log(0.4 / 0.2))
  })
  it('returns -Infinity for zero p(w|k)', () => {
    expect(relevance(0, 0.2, 0.5)).toBe(-Infinity)
  })
})

describe('topRelevantCodes', () => {
  it('orders by p(w|k) at λ=1, by lift at λ=0', () => {
    const pwk = [0.7, 0.2, 0.1]
    const pw  = [0.5, 0.4, 0.05]
    expect(topRelevantCodes({ pwk, pw, lambda: 1.0, n: 3 }).map((r) => r.index)).toEqual([0, 1, 2])
    expect(topRelevantCodes({ pwk, pw, lambda: 0.0, n: 3 })[0].index).toBe(2)
  })
})
```

- [ ] **Step 2: Implement** — Append to `inference.ts`:

```ts
export function relevance(pwk: number, pw: number, lambda: number): number {
  if (pwk <= 0) return -Infinity
  if (pw <= 0) return lambda * Math.log(pwk) + (1 - lambda) * Infinity
  return lambda * Math.log(pwk) + (1 - lambda) * Math.log(pwk / pw)
}

export interface TopCodesInput { pwk: number[]; pw: number[]; lambda: number; n: number }
export interface RankedCode { index: number; relevance: number; pwk: number; pw: number }

export function topRelevantCodes(input: TopCodesInput): RankedCode[] {
  const { pwk, pw, lambda, n } = input
  const scored: RankedCode[] = pwk.map((p, i) => ({
    index: i, relevance: relevance(p, pw[i] ?? 0, lambda), pwk: p, pw: pw[i] ?? 0,
  }))
  scored.sort((a, b) => b.relevance - a.relevance)
  return scored.slice(0, n)
}
```

- [ ] **Step 3: Run, PASS**

- [ ] **Step 4: Commit** — `git add ... && git commit -m "feat(inference): lambda-relevance"`

---

### Task 11: JSD

**Files:**
- Modify: `dashboard/src/lib/inference.ts`, `dashboard/src/lib/inference.test.ts`

- [ ] **Step 1: Tests** — Append:

```ts
import { jsd } from './inference'

describe('jsd', () => {
  it('zero on identical', () => { expect(jsd([0.5, 0.5], [0.5, 0.5])).toBeCloseTo(0, 9) })
  it('symmetric', () => {
    const p = [0.7, 0.2, 0.1], q = [0.1, 0.2, 0.7]
    expect(jsd(p, q)).toBeCloseTo(jsd(q, p), 9)
  })
  it('bounded by log 2', () => {
    expect(jsd([1, 0], [0, 1])).toBeLessThanOrEqual(Math.log(2) + 1e-9)
  })
})
```

- [ ] **Step 2: Implement** — Append to `inference.ts`:

```ts
export function jsd(p: number[], q: number[]): number {
  const m = p.map((pi, i) => 0.5 * (pi + q[i]))
  const kl = (a: number[], b: number[]) =>
    a.reduce((acc, ai, i) => (ai > 0 && b[i] > 0 ? acc + ai * (Math.log(ai) - Math.log(b[i])) : acc), 0)
  return 0.5 * (kl(p, m) + kl(q, m))
}
```

- [ ] **Step 3: Run, PASS**

- [ ] **Step 4: Commit** — `git add ... && git commit -m "feat(inference): JSD"`

---

### Task 12: Sampling primitives

**Files:**
- Create: `dashboard/src/lib/sampling.ts`, `dashboard/src/lib/sampling.test.ts`

- [ ] **Step 1: Tests**

```ts
// dashboard/src/lib/sampling.test.ts
import { describe, it, expect } from 'vitest'
import { createRng, sampleDirichlet, sampleCategorical, samplePoisson } from './sampling'

describe('createRng', () => {
  it('deterministic given seed', () => {
    const a = createRng(42), b = createRng(42)
    expect([a(), a(), a()]).toEqual([b(), b(), b()])
  })
})

describe('sampleDirichlet', () => {
  it('row on simplex', () => {
    const x = sampleDirichlet([1, 1, 1], createRng(1))
    expect(x.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6)
    expect(x.every((v) => v >= 0)).toBe(true)
  })
})

describe('sampleCategorical', () => {
  it('approx matches distribution', () => {
    const rng = createRng(3); const counts = [0, 0, 0]
    const p = [0.2, 0.3, 0.5]
    for (let i = 0; i < 20000; i++) counts[sampleCategorical(p, rng)]++
    expect(counts[0] / 20000).toBeCloseTo(0.2, 1)
    expect(counts[2] / 20000).toBeCloseTo(0.5, 1)
  })
})

describe('samplePoisson', () => {
  it('mean ≈ λ', () => {
    const rng = createRng(4); let s = 0
    for (let i = 0; i < 5000; i++) s += samplePoisson(5, rng)
    expect(s / 5000).toBeCloseTo(5, 0)
  })
})
```

- [ ] **Step 2: Implement**

```ts
// dashboard/src/lib/sampling.ts
export function createRng(seed: number): () => number {
  let s = seed >>> 0
  return function () {
    s = (s + 0x6d2b79f5) >>> 0
    let t = s
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function sampleGamma(shape: number, rng: () => number): number {
  if (shape < 1) {
    const y = sampleGamma(shape + 1, rng)
    return y * Math.pow(rng(), 1 / shape)
  }
  const d = shape - 1 / 3, c = 1 / Math.sqrt(9 * d)
  while (true) {
    let u1: number, u2: number, v: number
    do { u1 = 2 * rng() - 1; u2 = 2 * rng() - 1; v = u1 * u1 + u2 * u2 } while (v >= 1 || v === 0)
    const x = u1 * Math.sqrt(-2 * Math.log(v) / v)
    const vv = 1 + c * x
    if (vv <= 0) continue
    const v3 = vv * vv * vv
    const u = rng()
    if (u < 1 - 0.0331 * x * x * x * x) return d * v3
    if (Math.log(u) < 0.5 * x * x + d * (1 - v3 + Math.log(v3))) return d * v3
  }
}

export function sampleDirichlet(alpha: number[], rng: () => number): number[] {
  const draws = alpha.map((a) => sampleGamma(a, rng))
  const sum = draws.reduce((a, b) => a + b, 0) || 1
  return draws.map((g) => g / sum)
}

export function sampleCategorical(p: number[], rng: () => number): number {
  const u = rng(); let cum = 0
  for (let i = 0; i < p.length; i++) { cum += p[i]; if (u < cum) return i }
  return p.length - 1
}

export function samplePoisson(lambda: number, rng: () => number): number {
  const L = Math.exp(-lambda); let k = 0; let p = 1
  do { k++; p *= rng() } while (p > L)
  return k - 1
}
```

- [ ] **Step 3: Run, PASS**

- [ ] **Step 4: Commit** — `git add ... && git commit -m "feat(sampling): seedable PRNG + Dirichlet/Categorical/Poisson"`

---

### Task 13: Classical MDS

K=80 symmetric eigendecomposition via hand-rolled Jacobi rotation. ~100 lines, no extra dependency, deterministic.

**Files:**
- Create: `dashboard/src/lib/mds.ts`, `dashboard/src/lib/mds.test.ts`

- [ ] **Step 1: Tests**

```ts
// dashboard/src/lib/mds.test.ts
import { describe, it, expect } from 'vitest'
import { classicalMds, computeJsdMds } from './mds'

describe('classicalMds', () => {
  it('recovers a known 2D embedding up to rotation/reflection', () => {
    // Four corners of a unit square.
    const pts = [[0, 0], [1, 0], [0, 1], [1, 1]]
    const D: number[][] = pts.map((a) => pts.map((b) => Math.hypot(a[0] - b[0], a[1] - b[1])))
    const coords = classicalMds(D, 2)
    // pairwise distances must match the input D within tolerance
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
      const d = Math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
      expect(d).toBeCloseTo(D[i][j], 4)
    }
  })
})

describe('computeJsdMds', () => {
  it('returns one (x, y) per topic', () => {
    const beta = [
      [0.5, 0.5, 0, 0],
      [0, 0, 0.5, 0.5],
      [0.25, 0.25, 0.25, 0.25],
    ]
    const coords = computeJsdMds(beta)
    expect(coords.length).toBe(3)
    expect(coords[0].length).toBe(2)
  })
})
```

- [ ] **Step 2: Implement**

```ts
// dashboard/src/lib/mds.ts
import { jsd } from './inference'

// Symmetric eigendecomposition via Jacobi rotation. Returns eigenvalues
// (descending) and matching eigenvectors as columns of V.
function jacobiEig(A: number[][], maxSweeps = 60, tol = 1e-10): { values: number[]; vectors: number[][] } {
  const n = A.length
  const a = A.map((r) => r.slice())
  const v: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))
  )
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let off = 0
    for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) off += Math.abs(a[p][q])
    if (off < tol) break
    for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) {
      const apq = a[p][q]
      if (Math.abs(apq) < 1e-14) continue
      const theta = (a[q][q] - a[p][p]) / (2 * apq)
      const t = Math.sign(theta) / (Math.abs(theta) + Math.sqrt(theta * theta + 1))
      const c = 1 / Math.sqrt(t * t + 1)
      const s = t * c
      // Rotate rows/cols p,q
      const app = a[p][p], aqq = a[q][q]
      a[p][p] = app - t * apq
      a[q][q] = aqq + t * apq
      a[p][q] = 0; a[q][p] = 0
      for (let r = 0; r < n; r++) {
        if (r !== p && r !== q) {
          const arp = a[r][p], arq = a[r][q]
          a[r][p] = c * arp - s * arq
          a[p][r] = a[r][p]
          a[r][q] = s * arp + c * arq
          a[q][r] = a[r][q]
        }
      }
      for (let r = 0; r < n; r++) {
        const vrp = v[r][p], vrq = v[r][q]
        v[r][p] = c * vrp - s * vrq
        v[r][q] = s * vrp + c * vrq
      }
    }
  }
  const values = a.map((row, i) => row[i])
  const order = values.map((_, i) => i).sort((i, j) => values[j] - values[i])
  return {
    values: order.map((i) => values[i]),
    vectors: v.map((row) => order.map((i) => row[i])),
  }
}

export function classicalMds(distance: number[][], d = 2): number[][] {
  const n = distance.length
  // D² and double-centering
  const Dsq = distance.map((row) => row.map((v) => v * v))
  const rowMeans = Dsq.map((r) => r.reduce((a, b) => a + b, 0) / n)
  const grandMean = rowMeans.reduce((a, b) => a + b, 0) / n
  const B: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => -0.5 * (Dsq[i][j] - rowMeans[i] - rowMeans[j] + grandMean))
  )
  const { values, vectors } = jacobiEig(B)
  // Top-d eigenpairs; coords = vec_i · sqrt(max(λ_i, 0))
  const coords: number[][] = Array.from({ length: n }, () => new Array(d).fill(0))
  for (let k = 0; k < d; k++) {
    const lam = Math.max(values[k], 0)
    const s = Math.sqrt(lam)
    for (let i = 0; i < n; i++) coords[i][k] = vectors[i][k] * s
  }
  return coords
}

export function computeJsdMds(beta: number[][]): number[][] {
  const K = beta.length
  const D: number[][] = Array.from({ length: K }, () => new Array(K).fill(0))
  for (let i = 0; i < K; i++) {
    for (let j = i + 1; j < K; j++) {
      const v = Math.sqrt(Math.max(0, jsd(beta[i], beta[j])))
      D[i][j] = v
      D[j][i] = v
    }
  }
  return classicalMds(D, 2)
}
```

- [ ] **Step 3: Run, PASS**

- [ ] **Step 4: Commit** — `git add ... && git commit -m "feat(mds): classical MDS via Jacobi eigendecomposition"`

---

### Task 14: Client-side cohort generator

Replaces the cut Python `charmpheno.profiles.synthetic`. Same math, different language. Cohort lives in the `cohort` store; populated at app startup; re-runnable from the UI.

**Files:**
- Create: `dashboard/src/lib/cohort.ts`, `dashboard/src/lib/cohort.test.ts`
- Modify: `dashboard/src/App.svelte`

- [ ] **Step 1: Tests**

```ts
// dashboard/src/lib/cohort.test.ts
import { describe, it, expect } from 'vitest'
import { generateCohort } from './cohort'
import type { Model } from './types'

const model: Model = {
  K: 3, V: 5,
  alpha: [0.1, 0.1, 0.1],
  beta: [
    [0.9, 0.025, 0.025, 0.025, 0.025],
    [0.025, 0.9, 0.025, 0.025, 0.025],
    [0.025, 0.025, 0.9, 0.025, 0.025],
  ],
}

describe('generateCohort', () => {
  it('deterministic given seed', () => {
    const a = generateCohort({ model, meanCodesPerDoc: 8, n: 10, seed: 42, nNeighbors: 3 })
    const b = generateCohort({ model, meanCodesPerDoc: 8, n: 10, seed: 42, nNeighbors: 3 })
    expect(a.patients.map((p) => p.code_bag)).toEqual(b.patients.map((p) => p.code_bag))
  })

  it('produces patients on the simplex', () => {
    const c = generateCohort({ model, meanCodesPerDoc: 5, n: 12, seed: 1, nNeighbors: 3 })
    expect(c.patients.length).toBe(12)
    for (const p of c.patients) {
      expect(p.theta.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6)
      expect(p.code_bag.length).toBeGreaterThan(0)
      expect(p.neighbors.length).toBe(3)
      expect(p.neighbors.includes(p.id)).toBe(false)
      expect(new Set(p.neighbors).size).toBe(3)
    }
  })

  it('zero-pads patient ids', () => {
    const c = generateCohort({ model, meanCodesPerDoc: 5, n: 5, seed: 1, nNeighbors: 2 })
    expect(c.patients[0].id).toBe('synth_0000')
    expect(c.patients[4].id).toBe('synth_0004')
  })
})
```

- [ ] **Step 2: Implement**

```ts
// dashboard/src/lib/cohort.ts
import type { Model, SyntheticCohort, SyntheticPatient } from './types'
import {
  createRng, sampleDirichlet, sampleCategorical, samplePoisson,
} from './sampling'

export interface CohortInput {
  model: Model
  meanCodesPerDoc: number
  n: number
  seed: number
  nNeighbors: number
}

function cosineNeighbors(thetas: number[][], k: number): number[][] {
  const n = thetas.length
  const norms = thetas.map((t) => Math.hypot(...t))
  const neighbors: number[][] = []
  const sims = new Array<number>(n)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) { sims[j] = -Infinity; continue }
      let dot = 0
      for (let d = 0; d < thetas[i].length; d++) dot += thetas[i][d] * thetas[j][d]
      const denom = (norms[i] || 1) * (norms[j] || 1)
      sims[j] = denom > 0 ? dot / denom : -Infinity
    }
    // top-k by similarity
    const idx = sims.map((_, i2) => i2)
    idx.sort((a, b) => sims[b] - sims[a])
    neighbors.push(idx.slice(0, k))
  }
  return neighbors
}

export function generateCohort(input: CohortInput): SyntheticCohort {
  const { model, meanCodesPerDoc, n, seed, nNeighbors } = input
  const rng = createRng(seed)
  const thetas: number[][] = []
  const bags: number[][] = []
  for (let i = 0; i < n; i++) {
    const theta = sampleDirichlet(model.alpha, rng)
    const nCodes = Math.max(1, samplePoisson(meanCodesPerDoc, rng))
    const bag: number[] = []
    for (let c = 0; c < nCodes; c++) {
      const z = sampleCategorical(theta, rng)
      const w = sampleCategorical(model.beta[z], rng)
      bag.push(w)
    }
    thetas.push(theta)
    bags.push(bag)
  }
  const nbrIdx = cosineNeighbors(thetas, Math.min(nNeighbors, n - 1))
  const pad = (i: number) => `synth_${i.toString().padStart(4, '0')}`
  const patients: SyntheticPatient[] = thetas.map((theta, i) => ({
    id: pad(i),
    theta,
    code_bag: bags[i],
    neighbors: nbrIdx[i].map(pad),
  }))
  return { patients, seed }
}
```

- [ ] **Step 3: Wire generation into App.svelte startup**

Modify `dashboard/src/App.svelte` `onMount`:

```svelte
<script lang="ts">
  // …existing imports…
  import { cohort } from './lib/store'
  import { generateCohort } from './lib/cohort'

  const DEFAULT_COHORT_N = 1000
  const DEFAULT_COHORT_SEED = 42
  const DEFAULT_NEIGHBORS = 8

  let error: string | null = null
  onMount(async () => {
    try {
      const b = await loadBundle(import.meta.env.BASE_URL)
      bundle.set(b)
      const c = generateCohort({
        model: b.model,
        meanCodesPerDoc: b.corpusStats.mean_codes_per_doc,
        n: DEFAULT_COHORT_N,
        seed: DEFAULT_COHORT_SEED,
        nNeighbors: DEFAULT_NEIGHBORS,
      })
      cohort.set(c)
    } catch (e) { error = (e as Error).message }
  })
</script>
```

- [ ] **Step 4: Run tests, PASS**

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/lib/cohort.ts dashboard/src/lib/cohort.test.ts dashboard/src/App.svelte
git commit -m "feat(cohort): in-browser synthetic cohort with cosine neighbors"
```

**Phase 3 complete.**

---

## Phase 4: Phenotype Atlas

### Task 15: Topic map

D3 scatter at *client-computed* MDS coordinates. Bubbles sized by `corpus_prevalence`, colored by `npmi` or `prevalence`.

**Files:**
- Create: `dashboard/src/lib/atlas/TopicMap.svelte`
- Modify: `dashboard/src/lib/tabs/Atlas.svelte`

- [ ] **Step 1: Implement TopicMap**

```svelte
<!-- dashboard/src/lib/atlas/TopicMap.svelte -->
<script lang="ts">
  import { onMount } from 'svelte'
  import * as d3 from 'd3'
  import { bundle, selectedPhenotypeId, colorMode } from '../store'
  import { computeJsdMds } from '../mds'

  let svgEl: SVGSVGElement
  const W = 560, H = 480, MARGIN = 24

  // memoize coords: derived once from bundle.model.beta
  let coords: number[][] = []
  $: if ($bundle && coords.length !== $bundle.model.K) {
    coords = computeJsdMds($bundle.model.beta)
  }

  function render() {
    if (!$bundle || !svgEl || coords.length === 0) return
    const phenotypes = $bundle.phenotypes.phenotypes
    const xExt = d3.extent(coords, (c) => c[0]) as [number, number]
    const yExt = d3.extent(coords, (c) => c[1]) as [number, number]
    const x = d3.scaleLinear().domain(xExt).range([MARGIN, W - MARGIN])
    const y = d3.scaleLinear().domain(yExt).range([H - MARGIN, MARGIN])
    const r = d3.scaleSqrt()
      .domain(d3.extent(phenotypes, (p) => p.corpus_prevalence) as [number, number])
      .range([4, 24])
    const colorFn = $colorMode === 'prevalence'
      ? d3.scaleSequential(d3.interpolateBlues)
          .domain(d3.extent(phenotypes, (p) => p.corpus_prevalence) as [number, number])
      : d3.scaleSequential(d3.interpolateRdYlGn).domain([-0.2, 0.4])

    const svg = d3.select(svgEl)
    svg.selectAll('*').remove()
    svg.attr('viewBox', `0 0 ${W} ${H}`).attr('width', '100%').attr('height', H)
    const g = svg.append('g')

    g.selectAll('circle')
      .data(phenotypes)
      .join('circle')
      .attr('cx', (p) => x(coords[p.id][0]))
      .attr('cy', (p) => y(coords[p.id][1]))
      .attr('r', (p) => r(p.corpus_prevalence))
      .attr('fill', (p) => ($colorMode === 'prevalence' ? colorFn(p.corpus_prevalence) : colorFn(p.npmi)) as string)
      .attr('stroke', (p) => ($selectedPhenotypeId === p.id ? '#000' : '#444'))
      .attr('stroke-width', (p) => ($selectedPhenotypeId === p.id ? 2.5 : 0.5))
      .style('cursor', 'pointer')
      .on('click', (_, p) => selectedPhenotypeId.set(p.id))
      .append('title')
      .text((p) => `${p.label || `Phenotype ${p.id}`}\nNPMI ${p.npmi.toFixed(3)} · prev ${(p.corpus_prevalence * 100).toFixed(1)}%`)

    g.selectAll('text.junk')
      .data(phenotypes.filter((p) => p.junk_flag))
      .join('text')
      .attr('class', 'junk')
      .attr('x', (p) => x(coords[p.id][0]) + 8)
      .attr('y', (p) => y(coords[p.id][1]) - 8)
      .attr('font-size', 9)
      .attr('fill', '#b00020')
      .text('!')
  }

  $: $colorMode, $selectedPhenotypeId, $bundle && svgEl && coords.length && render()
  onMount(render)
</script>

<svg bind:this={svgEl} role="img" aria-label="Phenotype topic map" />
```

- [ ] **Step 2: Atlas.svelte hosts the map**

```svelte
<!-- dashboard/src/lib/tabs/Atlas.svelte -->
<script lang="ts">
  import { colorMode } from '../store'
  import TopicMap from '../atlas/TopicMap.svelte'
</script>

<section class="atlas">
  <header>
    <h2>Phenotype Atlas</h2>
    <label>Color
      <select bind:value={$colorMode}>
        <option value="npmi">NPMI</option>
        <option value="prevalence">Prevalence</option>
      </select>
    </label>
  </header>
  <div class="grid">
    <TopicMap />
    <aside class="code-panel"><p>Select a phenotype to see its top codes.</p></aside>
  </div>
</section>

<style>
  .atlas { padding: 1rem; }
  header { display: flex; align-items: baseline; gap: 1rem; margin-bottom: 1rem; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  aside { padding: 1rem; border: 1px solid #ddd; min-height: 480px; }
</style>
```

- [ ] **Step 3: Verify in browser** — `cd dashboard && make dev`. Atlas tab shows K circles at MDS coords.

- [ ] **Step 4: Commit** — `git add ... && git commit -m "feat(atlas): D3 topic map with client-side MDS"`

---

### Task 16: Code panel + λ slider (gated by advanced view)

**Files:**
- Create: `dashboard/src/lib/atlas/CodePanel.svelte`
- Modify: `dashboard/src/lib/tabs/Atlas.svelte`

- [ ] **Step 1: Implement CodePanel**

```svelte
<!-- dashboard/src/lib/atlas/CodePanel.svelte -->
<script lang="ts">
  import { bundle, selectedPhenotypeId, advancedView, hoveredCodeIdx } from '../store'
  import { topRelevantCodes } from '../inference'

  let lambda = 0.6
  const topN = 20

  $: pheno = $bundle && $selectedPhenotypeId !== null
    ? $bundle.phenotypes.phenotypes[$selectedPhenotypeId] : null

  $: top = $bundle && pheno
    ? topRelevantCodes({
        pwk: $bundle.model.beta[pheno.id],
        pw: $bundle.vocab.codes.map((c) => c.corpus_freq),
        lambda,
        n: topN,
      })
    : []
</script>

<aside class="code-panel">
  {#if !pheno}<p>Click a phenotype on the map.</p>
  {:else}
    <h3>{pheno.label || `Phenotype ${pheno.id}`}</h3>
    <p class="meta">
      prevalence {(pheno.corpus_prevalence * 100).toFixed(1)}%
      {#if $advancedView} · NPMI {pheno.npmi.toFixed(3)}{/if}
      {#if $advancedView && pheno.original_topic_id !== pheno.id} · model topic #{pheno.original_topic_id}{/if}
      {#if pheno.junk_flag}<span class="junk">low-coherence</span>{/if}
    </p>

    {#if $advancedView}
      <label class="slider">
        Relevance λ = {lambda.toFixed(2)}
        <input type="range" min="0" max="1" step="0.05" bind:value={lambda} />
      </label>
    {/if}

    <ol class="codes">
      {#each top as r}
        {@const c = $bundle!.vocab.codes[r.index]}
        <li
          on:mouseenter={() => hoveredCodeIdx.set(r.index)}
          on:mouseleave={() => hoveredCodeIdx.set(null)}
        >
          <span class="dom dom-{c.domain}">{c.domain.slice(0, 3)}</span>
          <span class="desc">{c.description || c.code}</span>
          <span class="num">{(r.pwk * 100).toFixed(2)}%</span>
        </li>
      {/each}
    </ol>
  {/if}
</aside>

<style>
  .code-panel { padding: 1rem; border: 1px solid #ddd; min-height: 480px; }
  h3 { margin: 0 0 0.25rem; }
  .meta { font-size: 0.85rem; color: #555; margin: 0 0 1rem; }
  .junk { color: #b00020; margin-left: 0.5rem; font-size: 0.75rem; }
  .slider { display: block; margin-bottom: 0.75rem; font-size: 0.85rem; }
  .slider input { width: 100%; }
  ol.codes { list-style: none; padding: 0; margin: 0; }
  ol.codes li { display: grid; grid-template-columns: 3rem 1fr auto; gap: 0.5rem; padding: 0.2rem 0; font-size: 0.85rem; border-bottom: 1px solid #f4f4f4; }
  .dom { font-size: 0.7rem; padding: 0.05rem 0.3rem; border-radius: 3px; text-align: center; }
  .dom-condition { background: #ffe4e1; }
  .dom-drug { background: #e0f2fe; }
  .dom-procedure { background: #ecfccb; }
  .dom-measurement { background: #fef3c7; }
  .dom-observation { background: #f5f5f5; }
  .num { font-variant-numeric: tabular-nums; color: #444; }
</style>
```

- [ ] **Step 2: Replace placeholder aside in Atlas.svelte**

Swap the `<aside class="code-panel">…</aside>` placeholder for `<CodePanel />`, add import `import CodePanel from '../atlas/CodePanel.svelte'`, and drop the now-redundant local `.code-panel` style.

- [ ] **Step 3: Verify in browser** — click a phenotype; top codes render. Toggle Advanced view; λ slider appears.

- [ ] **Step 4: Commit** — `git add ... && git commit -m "feat(atlas): code panel with advanced-view-gated lambda slider"`

---

### Task 17: Linked highlight (code → topic)

**Files:**
- Modify: `dashboard/src/lib/atlas/TopicMap.svelte`

- [ ] **Step 1: React to `hoveredCodeIdx`** — In `TopicMap.svelte`, add:

```ts
  import { hoveredCodeIdx } from '../store'

  function phenotypesWithCode(idx: number | null, n = 20): Set<number> {
    if (!$bundle || idx === null) return new Set()
    const out = new Set<number>()
    const K = $bundle.model.K
    for (let k = 0; k < K; k++) {
      const row = $bundle.model.beta[k]
      const top = row.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p).slice(0, n)
      if (top.some((s) => s.i === idx)) out.add(k)
    }
    return out
  }

  $: highlighted = phenotypesWithCode($hoveredCodeIdx)
```

Update the `.attr('stroke', …)` and `.attr('stroke-width', …)` lines in `render()`:

```ts
      .attr('stroke', (p) => (highlighted.has(p.id) ? '#1e88e5' : ($selectedPhenotypeId === p.id ? '#000' : '#444')))
      .attr('stroke-width', (p) => (highlighted.has(p.id) ? 3 : ($selectedPhenotypeId === p.id ? 2.5 : 0.5)))
```

Extend the reactivity line:

```ts
  $: $colorMode, $selectedPhenotypeId, $hoveredCodeIdx, $bundle && svgEl && coords.length && render()
```

- [ ] **Step 2: Verify** — Hover codes; matching bubbles get a blue ring.

- [ ] **Step 3: Commit** — `git add ... && git commit -m "feat(atlas): linked highlight on code hover"`

---

### Task 18: Cohort regenerate (advanced view only)

A small affordance: in advanced view, expose a button to regenerate the synthetic cohort with a new seed. Useful for demos.

**Files:**
- Modify: `dashboard/src/App.svelte`

- [ ] **Step 1: Add a hidden regenerate button gated by advancedView**

In `dashboard/src/App.svelte`'s `<header>`, add before the toggle:

```svelte
    {#if $bundle && $advancedView}
      <button class="regen" on:click={regenCohort}>Regenerate cohort</button>
    {/if}
```

In the script block:

```ts
  let cohortSeed = 42
  function regenCohort() {
    if (!$bundle) return
    cohortSeed += 1
    cohort.set(generateCohort({
      model: $bundle.model,
      meanCodesPerDoc: $bundle.corpusStats.mean_codes_per_doc,
      n: DEFAULT_COHORT_N,
      seed: cohortSeed,
      nNeighbors: DEFAULT_NEIGHBORS,
    }))
  }
```

And a tiny style:

```css
  .regen { font-size: 0.75rem; padding: 0.25rem 0.5rem; }
```

- [ ] **Step 2: Verify** — Toggle advanced view; click Regenerate; Patient Explorer (when wired in Phase 5) will show different patients.

- [ ] **Step 3: Commit** — `git add ... && git commit -m "feat(atlas): regenerate-cohort button under advanced view"`

**Phase 4 complete.**

---

## Phase 5: Patient Explorer

### Task 19: ProfileBar component

**Files:**
- Create: `dashboard/src/lib/patient/ProfileBar.svelte`

- [ ] **Step 1: Implement**

```svelte
<!-- dashboard/src/lib/patient/ProfileBar.svelte -->
<script lang="ts">
  import { phenotypesById } from '../store'
  export let theta: number[]
  export let height = 24
  export let labels = true
  export let onSelect: ((id: number) => void) | null = null
  export let otherThreshold = 0.05

  $: ordered = theta.map((v, k) => ({ k, v }))
    .filter((x) => x.v > 0)
    .sort((a, b) => b.v - a.v)
  $: mainBands = ordered.filter((x) => x.v >= otherThreshold)
  $: otherFrac = ordered.filter((x) => x.v < otherThreshold).reduce((a, x) => a + x.v, 0)

  function hue(k: number): string { return `hsl(${(k * 47) % 360} 60% 55%)` }
</script>

<div class="bar" style="height: {height}px">
  {#each mainBands as b}
    <button class="band"
      style="width: {(b.v * 100).toFixed(2)}%; background: {hue(b.k)};"
      title={`${$phenotypesById.get(b.k)?.label || `Phenotype ${b.k}`}: ${(b.v * 100).toFixed(1)}%`}
      on:click={() => onSelect?.(b.k)}
    ></button>
  {/each}
  {#if otherFrac > 0}
    <span class="band other" style="width: {(otherFrac * 100).toFixed(2)}%">Other</span>
  {/if}
</div>

{#if labels}
  <ul class="legend">
    {#each mainBands as b}
      <li><span class="swatch" style="background: {hue(b.k)};"></span><span>{$phenotypesById.get(b.k)?.label || `Phenotype ${b.k}`}</span><span class="pct">{(b.v * 100).toFixed(0)}%</span></li>
    {/each}
    {#if otherFrac > 0}<li><span class="swatch" style="background: #999;"></span><span>Other</span><span class="pct">{(otherFrac * 100).toFixed(0)}%</span></li>{/if}
  </ul>
{/if}

<style>
  .bar { display: flex; width: 100%; border-radius: 4px; overflow: hidden; border: 1px solid #ccc; }
  .band { border: 0; padding: 0; cursor: pointer; height: 100%; color: #fff; font-size: 0.7rem; }
  .band.other { background: #aaa; cursor: default; display: flex; align-items: center; justify-content: center; }
  ul.legend { list-style: none; padding: 0; margin: 0.5rem 0 0; font-size: 0.85rem; }
  ul.legend li { display: grid; grid-template-columns: 1.25rem 1fr 3rem; gap: 0.5rem; align-items: center; padding: 0.15rem 0; }
  .swatch { width: 1rem; height: 1rem; border-radius: 2px; }
  .pct { text-align: right; font-variant-numeric: tabular-nums; color: #555; }
</style>
```

- [ ] **Step 2: Commit** — `git add ... && git commit -m "feat(patient): reusable ProfileBar component"`

---

### Task 20: Patient picker + profile

**Files:**
- Modify: `dashboard/src/lib/tabs/Patient.svelte`

- [ ] **Step 1: Implement**

```svelte
<!-- dashboard/src/lib/tabs/Patient.svelte -->
<script lang="ts">
  import { cohort, patientsById, selectedPatientId, selectedPhenotypeId } from '../store'
  import ProfileBar from '../patient/ProfileBar.svelte'

  $: patients = $cohort?.patients ?? []
  $: current = $selectedPatientId ? $patientsById.get($selectedPatientId) : (patients[0] ?? null)
  $: if (current && $selectedPatientId !== current.id) selectedPatientId.set(current.id)

  function shuffle() {
    if (patients.length === 0) return
    selectedPatientId.set(patients[Math.floor(Math.random() * patients.length)].id)
  }
</script>

<section class="patient">
  <header>
    <h2>Patient Explorer</h2>
    <label>Patient
      <select bind:value={$selectedPatientId}>
        {#each patients as p}<option value={p.id}>{p.id}</option>{/each}
      </select>
    </label>
    <button on:click={shuffle}>Shuffle</button>
  </header>

  {#if current}
    <div class="profile">
      <h3>Profile</h3>
      <ProfileBar theta={current.theta} height={40} onSelect={(k) => selectedPhenotypeId.set(k)} />
    </div>
  {/if}
</section>

<style>
  .patient { padding: 1rem; }
  header { display: flex; align-items: baseline; gap: 1rem; margin-bottom: 1rem; }
  .profile h3 { margin: 0 0 0.5rem; }
</style>
```

- [ ] **Step 2: Verify** — Open Patient tab; profile bar renders; switch via dropdown.

- [ ] **Step 3: Commit** — `git add ... && git commit -m "feat(patient): picker and profile bar"`

---

### Task 21: Top contributing codes

**Files:**
- Create: `dashboard/src/lib/patient/ContributingCodes.svelte`
- Modify: `dashboard/src/lib/tabs/Patient.svelte`

- [ ] **Step 1: Implement ContributingCodes**

```svelte
<!-- dashboard/src/lib/patient/ContributingCodes.svelte -->
<script lang="ts">
  import { bundle, selectedPhenotypeId } from '../store'
  export let theta: number[]
  export let codeBag: number[]

  $: counts = (() => {
    const m = new Map<number, number>()
    for (const w of codeBag) m.set(w, (m.get(w) ?? 0) + 1)
    return m
  })()

  $: top = (() => {
    if (!$bundle || $selectedPhenotypeId === null) return []
    const k = $selectedPhenotypeId
    const beta = $bundle.model.beta
    const K = $bundle.model.K
    const scored: { w: number; c: number; score: number }[] = []
    for (const [w, c] of counts) {
      let z = 0
      for (let j = 0; j < K; j++) z += beta[j][w] * theta[j]
      const pzkw = z > 0 ? (beta[k][w] * theta[k]) / z : 0
      scored.push({ w, c, score: c * pzkw })
    }
    return scored.sort((a, b) => b.score - a.score).slice(0, 12)
  })()
</script>

<section class="contrib">
  <h3>Top contributing codes</h3>
  {#if $selectedPhenotypeId === null}<p class="hint">Click a phenotype band above.</p>
  {:else if top.length === 0}<p>No codes from this patient's bag contribute to phenotype {$selectedPhenotypeId}.</p>
  {:else}
    <ol>
      {#each top as t}
        {@const c = $bundle!.vocab.codes[t.w]}
        <li>
          <span class="dom dom-{c.domain}">{c.domain.slice(0, 3)}</span>
          <span class="desc">{c.description || c.code}</span>
          <span class="count">×{t.c}</span>
        </li>
      {/each}
    </ol>
  {/if}
</section>

<style>
  .contrib { margin-top: 1rem; }
  .hint { color: #555; font-size: 0.85rem; }
  ol { list-style: none; padding: 0; }
  li { display: grid; grid-template-columns: 3rem 1fr 3rem; gap: 0.5rem; padding: 0.2rem 0; font-size: 0.85rem; border-bottom: 1px solid #f4f4f4; }
  .dom { font-size: 0.7rem; padding: 0.05rem 0.3rem; border-radius: 3px; text-align: center; }
  .dom-condition { background: #ffe4e1; }
  .dom-drug { background: #e0f2fe; }
  .dom-procedure { background: #ecfccb; }
  .dom-measurement { background: #fef3c7; }
  .dom-observation { background: #f5f5f5; }
  .count { text-align: right; font-variant-numeric: tabular-nums; color: #444; }
</style>
```

- [ ] **Step 2: Wire into Patient.svelte** — Inside the `{#if current}` block, append `<ContributingCodes theta={current.theta} codeBag={current.code_bag} />`. Add the import.

- [ ] **Step 3: Verify** — click a profile band; contributing codes populate.

- [ ] **Step 4: Commit** — `git add ... && git commit -m "feat(patient): contributing codes for selected phenotype"`

---

### Task 22: Nearest-neighbor ribbon

**Files:**
- Create: `dashboard/src/lib/patient/NeighborRibbon.svelte`
- Modify: `dashboard/src/lib/tabs/Patient.svelte`

- [ ] **Step 1: Implement**

```svelte
<!-- dashboard/src/lib/patient/NeighborRibbon.svelte -->
<script lang="ts">
  import { selectedPatientId, patientsById } from '../store'
  import ProfileBar from './ProfileBar.svelte'
  export let neighbors: string[]
</script>

<section class="ribbon">
  <h3>Patients with similar profiles</h3>
  <div class="strip">
    {#each neighbors as nid}
      {@const n = $patientsById.get(nid)}
      {#if n}
        <button class="card" on:click={() => selectedPatientId.set(nid)}>
          <span class="id">{n.id}</span>
          <ProfileBar theta={n.theta} height={14} labels={false} />
        </button>
      {/if}
    {/each}
  </div>
</section>

<style>
  .ribbon { margin-top: 1.5rem; }
  .strip { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.5rem; }
  .card { display: grid; gap: 0.25rem; padding: 0.5rem; border: 1px solid #ddd; border-radius: 4px; background: #fff; cursor: pointer; text-align: left; }
  .card:hover { background: #f8f8f8; }
  .id { font-size: 0.75rem; color: #555; }
</style>
```

- [ ] **Step 2: Wire** — Append `<NeighborRibbon neighbors={current.neighbors} />` inside the `{#if current}` block. Add import.

- [ ] **Step 3: Verify** — Neighbors render as cards; click jumps.

- [ ] **Step 4: Commit** — `git add ... && git commit -m "feat(patient): neighbor ribbon"`

**Phase 5 complete.**

---

## Phase 6: Simulator

### Task 23: Prefix editor

**Files:**
- Create: `dashboard/src/lib/simulator/PrefixEditor.svelte`

- [ ] **Step 1: Implement**

```svelte
<!-- dashboard/src/lib/simulator/PrefixEditor.svelte -->
<script lang="ts">
  import {
    bundle, patientsById, selectedPatientId, simulatorPrefix,
  } from '../store'
  let searchText = ''

  $: matches = $bundle && searchText.length >= 2
    ? $bundle.vocab.codes
        .filter((c) =>
          c.description.toLowerCase().includes(searchText.toLowerCase()) ||
          c.code.includes(searchText))
        .slice(0, 8)
    : []

  function loadFromPatient() {
    const p = $selectedPatientId ? $patientsById.get($selectedPatientId) : null
    if (p) simulatorPrefix.set([...p.code_bag])
  }
  function clearAll() { simulatorPrefix.set([]) }
  function add(idx: number) {
    simulatorPrefix.update((prev) => [...prev, idx]); searchText = ''
  }
  function removeAt(i: number) {
    simulatorPrefix.update((prev) => prev.filter((_, j) => j !== i))
  }
  function forcePhenotype(k: number) {
    if (!$bundle) return
    const row = $bundle.model.beta[k]
    const topIdx = row.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p).slice(0, 5).map((x) => x.i)
    simulatorPrefix.update((prev) => [...prev, ...topIdx])
  }
</script>

<section class="editor">
  <h3>Prefix</h3>
  <div class="actions">
    <button on:click={loadFromPatient} disabled={!$selectedPatientId}>Load selected patient</button>
    <button on:click={clearAll}>Clear</button>
  </div>
  <input type="text" placeholder="Search vocab to add a code…" bind:value={searchText} />
  {#if matches.length > 0}
    <ul class="matches">
      {#each matches as c}<li><button on:click={() => add(c.id)}>+ {c.code} {c.description}</button></li>{/each}
    </ul>
  {/if}
  <details>
    <summary>Force a phenotype (adds top-5 codes for it)</summary>
    <ul class="force">
      {#each $bundle?.phenotypes.phenotypes ?? [] as p}
        <li><button on:click={() => forcePhenotype(p.id)}>{p.label || `Phenotype ${p.id}`}</button></li>
      {/each}
    </ul>
  </details>
  <h4>Current prefix ({$simulatorPrefix.length} codes)</h4>
  <ul class="prefix">
    {#each $simulatorPrefix as idx, i}
      {@const c = $bundle?.vocab.codes[idx]}
      <li>
        <span>{c?.description || c?.code || `#${idx}`}</span>
        <button on:click={() => removeAt(i)}>×</button>
      </li>
    {/each}
  </ul>
</section>

<style>
  .editor { padding: 1rem; border: 1px solid #ddd; }
  .actions { display: flex; gap: 0.5rem; margin-bottom: 0.5rem; }
  input { width: 100%; padding: 0.4rem; margin-bottom: 0.5rem; }
  .matches, .force, .prefix { list-style: none; padding: 0; margin: 0; max-height: 220px; overflow: auto; }
  .matches li button, .force li button { width: 100%; text-align: left; background: transparent; border: 0; padding: 0.25rem; cursor: pointer; }
  .matches li button:hover, .force li button:hover { background: #f0f0f0; }
  .prefix li { display: grid; grid-template-columns: 1fr auto; gap: 0.5rem; padding: 0.2rem 0; font-size: 0.85rem; border-bottom: 1px solid #f4f4f4; }
  details { margin: 0.5rem 0; }
  h4 { margin: 0.5rem 0 0.25rem; font-size: 0.9rem; }
</style>
```

- [ ] **Step 2: Commit** — `git add ... && git commit -m "feat(simulator): prefix editor"`

---

### Task 24: Sampling loop + density-strip carpet + per-phenotype drill-down

**Files:**
- Create: `dashboard/src/lib/simulator/runSamples.ts`, `dashboard/src/lib/simulator/runSamples.test.ts`, `dashboard/src/lib/simulator/Carpet.svelte`

- [ ] **Step 1: Sampling loop**

```ts
// dashboard/src/lib/simulator/runSamples.ts
import { variationalEStep } from '../inference'
import { createRng, sampleCategorical, samplePoisson } from '../sampling'

export interface SimulatorRunInput {
  alpha: number[]
  beta: number[][]
  meanCodesPerDoc: number
  prefix: number[]
  nSamples: number
  seed: number
}
export interface SimulatorRunResult {
  thetaSamples: number[][]
  codeCountsSamples: Map<number, number>[]
}

export function runSimulator(input: SimulatorRunInput): SimulatorRunResult {
  const { alpha, beta, meanCodesPerDoc, prefix, nSamples, seed } = input
  const prefixCounts = new Map<number, number>()
  for (const w of prefix) prefixCounts.set(w, (prefixCounts.get(w) ?? 0) + 1)
  const rng = createRng(seed)
  const thetas: number[][] = []
  const bags: Map<number, number>[] = []
  for (let s = 0; s < nSamples; s++) {
    const nNew = Math.max(1, samplePoisson(meanCodesPerDoc, rng))
    const sampleCounts = new Map(prefixCounts)
    let est = variationalEStep({ alpha, beta, codeCounts: sampleCounts })
    for (let n = 0; n < nNew; n++) {
      const z = sampleCategorical(est.theta, rng)
      const w = sampleCategorical(beta[z], rng)
      sampleCounts.set(w, (sampleCounts.get(w) ?? 0) + 1)
    }
    est = variationalEStep({ alpha, beta, codeCounts: sampleCounts })
    thetas.push(est.theta)
    const completion = new Map<number, number>()
    for (const [w, c] of sampleCounts) {
      const pre = prefixCounts.get(w) ?? 0
      if (c - pre > 0) completion.set(w, c - pre)
    }
    bags.push(completion)
  }
  return { thetaSamples: thetas, codeCountsSamples: bags }
}

export function quantiles(values: number[], qs: number[]): number[] {
  const sorted = values.slice().sort((a, b) => a - b)
  return qs.map((q) => {
    if (sorted.length === 0) return 0
    const pos = q * (sorted.length - 1)
    const lo = Math.floor(pos), hi = Math.ceil(pos)
    if (lo === hi) return sorted[lo]
    return sorted[lo] * (hi - pos) + sorted[hi] * (pos - lo)
  })
}
```

- [ ] **Step 2: Tests**

```ts
// dashboard/src/lib/simulator/runSamples.test.ts
import { describe, it, expect } from 'vitest'
import { runSimulator, quantiles } from './runSamples'

describe('runSimulator', () => {
  it('returns N theta vectors on the simplex', () => {
    const alpha = [0.1, 0.1, 0.1]
    const beta = [[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7]]
    const out = runSimulator({ alpha, beta, meanCodesPerDoc: 5, prefix: [], nSamples: 20, seed: 0 })
    expect(out.thetaSamples.length).toBe(20)
    for (const t of out.thetaSamples) expect(t.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5)
  })
  it('prefix on code 0 biases theta toward topic 0', () => {
    const alpha = [0.1, 0.1, 0.1]
    const beta = [[0.95, 0.025, 0.025], [0.025, 0.95, 0.025], [0.025, 0.025, 0.95]]
    const out = runSimulator({ alpha, beta, meanCodesPerDoc: 1, prefix: Array(20).fill(0), nSamples: 50, seed: 1 })
    const meanT0 = out.thetaSamples.reduce((a, t) => a + t[0], 0) / out.thetaSamples.length
    expect(meanT0).toBeGreaterThan(0.7)
  })
})

describe('quantiles', () => {
  it('matches linear-interpolation', () => {
    expect(quantiles([1, 2, 3, 4, 5], [0, 0.5, 1])).toEqual([1, 3, 5])
  })
})
```

Run: `cd dashboard && npm test`. PASS.

- [ ] **Step 3: Carpet with drill-down**

```svelte
<!-- dashboard/src/lib/simulator/Carpet.svelte -->
<script lang="ts">
  import * as d3 from 'd3'
  import { bundle, phenotypesById } from '../store'
  import { quantiles } from './runSamples'
  export let thetaSamples: number[][]
  export let codeCountsSamples: Map<number, number>[]
  export let sortMode: 'median' | 'spread' | 'npmi' | 'id' = 'median'

  const W = 720, ROW_H = 14, X_MARGIN = 220
  let expandedK: number | null = null

  $: K = $bundle?.model.K ?? 0
  $: H = ROW_H * K + 20

  $: rows = (() => {
    if (thetaSamples.length === 0 || K === 0) return []
    const out = []
    for (let k = 0; k < K; k++) {
      const ks = thetaSamples.map((t) => t[k])
      const q = quantiles(ks, [0.1, 0.25, 0.5, 0.75, 0.9])
      out.push({ k, p10: q[0], p25: q[1], p50: q[2], p75: q[3], p90: q[4] })
    }
    if (sortMode === 'median') out.sort((a, b) => b.p50 - a.p50)
    else if (sortMode === 'spread') out.sort((a, b) => (b.p90 - b.p10) - (a.p90 - a.p10))
    else if (sortMode === 'npmi') {
      const npmi = (k: number) => $phenotypesById.get(k)?.npmi ?? 0
      out.sort((a, b) => npmi(b.k) - npmi(a.k))
    }
    return out
  })()

  $: xMax = rows.length > 0 ? Math.max(0.01, d3.max(rows, (r) => r.p90) ?? 0.01) : 0.01
  $: xScale = d3.scaleLinear().domain([0, xMax]).range([X_MARGIN, W - 20])

  $: drill = (() => {
    if (expandedK === null || !$bundle) return [] as { w: number; score: number }[]
    const k = expandedK
    const beta = $bundle.model.beta
    const K_ = $bundle.model.K
    const scores = new Map<number, number>()
    for (let s = 0; s < codeCountsSamples.length; s++) {
      const theta = thetaSamples[s]
      for (const [w, c] of codeCountsSamples[s]) {
        let z = 0
        for (let j = 0; j < K_; j++) z += beta[j][w] * theta[j]
        const pzkw = z > 0 ? (beta[k][w] * theta[k]) / z : 0
        scores.set(w, (scores.get(w) ?? 0) + c * pzkw)
      }
    }
    return Array.from(scores.entries())
      .map(([w, score]) => ({ w, score }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 8)
  })()
</script>

<svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} role="img">
  {#each rows as r, i}
    {@const cy = 10 + i * ROW_H + ROW_H / 2}
    <g style="cursor: pointer;" on:click={() => expandedK = expandedK === r.k ? null : r.k}>
      <rect x="0" y={cy - ROW_H / 2} width={W} height={ROW_H} fill={expandedK === r.k ? '#f4f8ff' : 'transparent'} />
      <text x={X_MARGIN - 8} y={cy + 3} font-size="10" text-anchor="end">
        {$phenotypesById.get(r.k)?.label || `Phenotype ${r.k}`}
      </text>
      <line x1={xScale(r.p10)} y1={cy} x2={xScale(r.p90)} y2={cy} stroke="#999" stroke-width="1" />
      <rect x={xScale(r.p25)} y={cy - 4} width={Math.max(1, xScale(r.p75) - xScale(r.p25))} height="8" fill="#1e88e5" opacity="0.7" />
      <line x1={xScale(r.p50)} y1={cy - 5} x2={xScale(r.p50)} y2={cy + 5} stroke="#000" stroke-width="1.5" />
      <text x={W - 18} y={cy + 3} font-size="9" text-anchor="end" fill="#555">{(r.p50 * 100).toFixed(1)}%</text>
    </g>
  {/each}
</svg>

{#if expandedK !== null}
  <aside class="drill">
    <h4>Top codes driving {$phenotypesById.get(expandedK)?.label || `Phenotype ${expandedK}`}</h4>
    {#if drill.length === 0}<p>No completion codes scored above zero.</p>
    {:else}
      <ol>{#each drill as d}{@const c = $bundle!.vocab.codes[d.w]}<li><span>{c.description || c.code}</span><span class="score">{d.score.toFixed(2)}</span></li>{/each}</ol>
    {/if}
  </aside>
{/if}

<style>
  .drill { margin-top: 0.5rem; padding: 0.75rem; background: #f4f8ff; border: 1px solid #cfe2ff; }
  .drill h4 { margin: 0 0 0.5rem; font-size: 0.9rem; }
  ol { list-style: none; padding: 0; margin: 0; font-size: 0.85rem; }
  ol li { display: grid; grid-template-columns: 1fr 4rem; gap: 0.5rem; padding: 0.15rem 0; border-bottom: 1px solid #e0e0e0; }
  .score { text-align: right; font-variant-numeric: tabular-nums; }
</style>
```

- [ ] **Step 4: Commit** — `git add ... && git commit -m "feat(simulator): sampling loop, density-strip carpet, drill-down"`

---

### Task 25: Expected codes panel

**Files:**
- Create: `dashboard/src/lib/simulator/ExpectedCodes.svelte`

- [ ] **Step 1: Implement**

```svelte
<!-- dashboard/src/lib/simulator/ExpectedCodes.svelte -->
<script lang="ts">
  import { bundle } from '../store'
  import { quantiles } from './runSamples'
  export let codeCountsSamples: Map<number, number>[]
  export let topN = 20

  $: top = (() => {
    if (codeCountsSamples.length === 0 || !$bundle) return []
    const all = new Set<number>()
    for (const m of codeCountsSamples) for (const w of m.keys()) all.add(w)
    const rows: { w: number; p10: number; p50: number; p90: number }[] = []
    for (const w of all) {
      const counts = codeCountsSamples.map((m) => m.get(w) ?? 0)
      const q = quantiles(counts, [0.1, 0.5, 0.9])
      if (q[1] === 0 && q[2] === 0) continue
      rows.push({ w, p10: q[0], p50: q[1], p90: q[2] })
    }
    rows.sort((a, b) => b.p50 - a.p50)
    return rows.slice(0, topN)
  })()

  $: maxP90 = top.length > 0 ? Math.max(...top.map((r) => r.p90)) : 1
</script>

<section class="expected">
  <h3>Top expected codes</h3>
  <table>
    {#each top as r}
      {@const c = $bundle!.vocab.codes[r.w]}
      <tr>
        <td class="desc">{c.description || c.code}</td>
        <td class="bar">
          <span class="rng" style="left: {(r.p10 / maxP90) * 100}%; width: {((r.p90 - r.p10) / maxP90) * 100}%"></span>
          <span class="med" style="left: {(r.p50 / maxP90) * 100}%"></span>
        </td>
        <td class="num">{r.p50.toFixed(1)}</td>
      </tr>
    {/each}
  </table>
</section>

<style>
  .expected { padding: 1rem; border: 1px solid #ddd; }
  h3 { margin: 0 0 0.5rem; }
  table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
  td { padding: 0.15rem 0.25rem; border-bottom: 1px solid #f4f4f4; }
  td.desc { width: 50%; }
  td.bar { width: 40%; position: relative; height: 1.2rem; }
  td.bar .rng { position: absolute; top: 0.45rem; height: 0.25rem; background: #cfe2ff; }
  td.bar .med { position: absolute; top: 0.2rem; width: 2px; height: 0.7rem; background: #1e88e5; }
  td.num { width: 4rem; text-align: right; font-variant-numeric: tabular-nums; }
</style>
```

- [ ] **Step 2: Commit** — `git add ... && git commit -m "feat(simulator): expected-codes panel"`

---

### Task 26: Simulator tab integration

**Files:**
- Modify: `dashboard/src/lib/tabs/Simulator.svelte`

- [ ] **Step 1: Implement**

```svelte
<!-- dashboard/src/lib/tabs/Simulator.svelte -->
<script lang="ts">
  import { bundle, simulatorPrefix } from '../store'
  import { runSimulator } from '../simulator/runSamples'
  import PrefixEditor from '../simulator/PrefixEditor.svelte'
  import Carpet from '../simulator/Carpet.svelte'
  import ExpectedCodes from '../simulator/ExpectedCodes.svelte'

  let nSamples = 1000
  let sortMode: 'median' | 'spread' | 'npmi' | 'id' = 'median'
  let seed = 0
  let result: ReturnType<typeof runSimulator> | null = null
  let running = false

  async function runSim() {
    if (!$bundle) return
    running = true
    await new Promise((r) => setTimeout(r, 0))
    result = runSimulator({
      alpha: $bundle.model.alpha,
      beta: $bundle.model.beta,
      meanCodesPerDoc: $bundle.corpusStats.mean_codes_per_doc,
      prefix: $simulatorPrefix,
      nSamples, seed,
    })
    running = false
  }
</script>

<section class="sim">
  <header>
    <h2>Simulator</h2>
    <label>N <input type="range" min="10" max="2000" step="10" bind:value={nSamples} /> <span class="num">{nSamples}</span></label>
    <label>Sort <select bind:value={sortMode}>
      <option value="median">Median θ</option>
      <option value="spread">Spread (P90-P10)</option>
      <option value="npmi">NPMI</option>
      <option value="id">Phenotype id</option>
    </select></label>
    <label>Seed <input type="number" bind:value={seed} style="width: 5rem" /></label>
    <button on:click={runSim} disabled={running}>{running ? 'Sampling…' : 'Re-sample'}</button>
  </header>

  <div class="grid">
    <PrefixEditor />
    <div class="main">
      {#if result}
        <Carpet thetaSamples={result.thetaSamples} codeCountsSamples={result.codeCountsSamples} {sortMode} />
        <ExpectedCodes codeCountsSamples={result.codeCountsSamples} />
      {:else}
        <p class="hint">Compose a prefix on the left, then Re-sample.</p>
      {/if}
    </div>
  </div>
  <p class="footnote">
    Year-of-life scope; code ordering and timing are not modeled. Each sample is one complete bag.
    N = {nSamples}, K = {$bundle?.model.K ?? '?'}, prefix length = {$simulatorPrefix.length}.
  </p>
</section>

<style>
  .sim { padding: 1rem; }
  header { display: flex; align-items: baseline; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; }
  header label { display: flex; align-items: baseline; gap: 0.25rem; font-size: 0.85rem; }
  .grid { display: grid; grid-template-columns: 320px 1fr; gap: 1rem; }
  .main { display: grid; gap: 1rem; }
  .footnote { margin-top: 1rem; font-size: 0.75rem; color: #777; }
  .hint { color: #555; }
  .num { font-variant-numeric: tabular-nums; }
</style>
```

- [ ] **Step 2: Verify in browser** — Load a patient's bag into the prefix; re-sample; carpet + expected codes render; slide N up; boxes tighten.

- [ ] **Step 3: Commit** — `git add ... && git commit -m "feat(simulator): integrated tab with N-slider, sort, re-sample"`

**Phase 6 complete.**

---

## Phase 7: Deploy + ADR

### Task 27: GitHub Actions workflow

**Files:**
- Create: `.github/workflows/dashboard.yml`

- [ ] **Step 1: Write the workflow**

```yaml
name: Dashboard deploy
on:
  push:
    branches: [main]
    paths: ['dashboard/**', '.github/workflows/dashboard.yml']
  workflow_dispatch: {}
permissions: { contents: write }
jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
          cache-dependency-path: dashboard/package-lock.json
      - name: Install
        working-directory: dashboard
        run: npm ci
      - name: Test
        working-directory: dashboard
        run: npm test
      - name: Build
        working-directory: dashboard
        env: { VITE_BASE: /CHARMPheno/ }
        run: npm run build
      - name: Deploy
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: dashboard/dist
          publish_branch: gh-pages
```

- [ ] **Step 2: Commit + enable Pages**

```bash
git add .github/workflows/dashboard.yml
git commit -m "ci(dashboard): build + deploy to gh-pages"
```

In GitHub repo settings: set Pages source to the `gh-pages` branch root (one-time UI step).

- [ ] **Step 3: After next push, verify** — Actions tab green; `https://<owner>.github.io/CHARMPheno/` loads.

---

### Task 28: README for the dashboard

**Files:**
- Create: `dashboard/README.md`

- [ ] **Step 1: Write**

```markdown
# CHARMPheno Dashboard

Static, single-page Svelte+Vite+D3 dashboard demonstrating a trained CHARMPheno topic model.
All patient-shaped artifacts are synthetic and generated in the browser; no real patient data is shipped.

## Development

    make dev      # vite dev server
    make build    # static output to dist/
    make test     # vitest

## Data bundle

The dashboard reads four JSON files from `public/data/`:
`model.json`, `vocab.json`, `phenotypes.json`, `corpus_stats.json`.

Regenerate from a checkpoint:

    poetry run python ../analysis/local/build_dashboard.py \
        --checkpoint ../data/runs/<checkpoint> \
        --input ../data/simulated/omop_N10000_seed42.parquet \
        --out-dir public/data \
        --vocab-top-n 5000

See `docs/superpowers/specs/2026-05-13-dashboard-design.md` for the schema.

## Deploy

Pushes to `main` that touch `dashboard/**` trigger `.github/workflows/dashboard.yml`,
which builds and deploys to the `gh-pages` branch.
```

- [ ] **Step 2: Commit** — `git add ... && git commit -m "docs(dashboard): README"`

---

### Task 29: ADR 0020

**Files:**
- Create: `docs/decisions/0020-dashboard-static-hosting-and-artifact-contract.md`
- Modify: `docs/decisions/README.md` (add to index if it has one)

- [ ] **Step 1: Write the ADR**

```markdown
# 0020 — Dashboard: static hosting and artifact contract

**Status:** Accepted
**Date:** 2026-05-14
**Spec:** docs/superpowers/specs/2026-05-13-dashboard-design.md
**Plan:** docs/superpowers/plans/2026-05-14-dashboard.md

## Context

CharmPheno needs a salesmanship surface — a live, interactive demonstration of the trained
topic model for non-technical audiences. Real patient data cannot be exported from the
secure environment, and we want hosting cheap enough that a researcher can ship it without
infra negotiation.

## Decision

1. **Static hosting on GitHub Pages.** No backend, no WASM. The dashboard is a Svelte 5 +
   Vite + D3 single-page app under `dashboard/`, deployed via a `gh-pages` branch on push
   to `main`.
2. **Synthetic-only patient framing.** Every patient-shaped artifact in the UI is
   generated *in the browser* from the exported model. The cloud side ships no patient
   data, synthetic or otherwise. The dashboard demonstrates the README claim that trained
   models do not carry sensitive information, rather than apologizing for it.
3. **Minimal four-file bundle as the modeling ↔ UI contract:** `model.json`,
   `phenotypes.json`, `vocab.json` (trimmed to top-N=5000 codes by corpus frequency),
   `corpus_stats.json`. Schema in the spec.
4. **No temporal axis in the Simulator.** Each sampled completion is one full year-of-life
   code bag; the visualization is a single-snapshot posterior over phenotype proportions.
   BOW exchangeability would make a finer time grain misleading.
5. **Re-implement pyLDAvis** rather than embed it. Static HTML output is fine for a
   one-off, but clinical-domain affordances (domain coloring, NPMI overlays, advanced
   view, linked code→topic highlights) want first-class control.
6. **Advanced-view toggle, not clinician-view.** Default is the simpler view; the toggle
   *reveals* technical affordances rather than hiding them.

## Consequences

- The export pipeline (`charmpheno.export.dashboard`) becomes a reusable artifact for any
  future dashboard variant; coupling to the Svelte app is the JSON schema only.
- The dashboard cannot show real-patient prevalence breakdowns, cohort comparisons, or
  trajectories. Those would require additional aggregate exports negotiated separately.
- Computing synthetic cohorts and topic-map MDS in JS removes a separate "local Python
  build" from the workflow and makes the privacy story tighter.
- Bundle size grows with K and trimmed-V. v1 targets <2 MB gzipped; if exceeded, gzip
  served by GH Pages or shrink top-N.
```

- [ ] **Step 2: Update decisions README** if it has an index — add line `- [0020 — Dashboard …](0020-dashboard-static-hosting-and-artifact-contract.md)`.

- [ ] **Step 3: Commit** — `git add ... && git commit -m "docs(adr): 0020 dashboard static hosting and artifact contract"`

**Phase 7 complete. Plan complete.**

---

## Final checks (run after all phases)

- [ ] `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && make test` passes
- [ ] `cd dashboard && npm test` passes
- [ ] `cd dashboard && make build` produces `dist/` without errors
- [ ] Browser dev URL: three tabs render; cohort generates at startup; atlas/patient/simulator interact correctly with a real bundle in `dashboard/public/data/`
- [ ] After push to `main`: GH Actions green; public URL serves the live dashboard
