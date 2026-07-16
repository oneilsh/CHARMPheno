# Gated SVI Placement Engine + MLlib Shim Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `GatedOnlineLDA` — the distributed SVI (variational) twin of the validated collapsed-Gibbs placement engine — plus its MLlib Estimator/Model shim, so the OMOP layer fits hierarchical case-finding at scale and this engine's placement matches the Gibbs oracle at every depth.

**Architecture:** `GatedOnlineLDA(OnlineLDA)` overrides exactly two methods: `local_update` (per training doc, restrict CAVI to `allowed_set(frontier)` and scatter sstats to those rows only — the variational twin of the Gibbs gate) and `initialize_global` (dispatch on a pluggable init strategy; `"random"` default). Everything else — `update_global`, `compute_elbo`, `combine_stats`, `VIRunner` integration — is inherited. Deployment folds held-out docs in **ungated** (full-K CAVI) → θ → per-node affinity. The MLlib shim mirrors `mllib/topic/lda.py` (ADR 0009).

**Tech Stack:** Python, NumPy, SciPy, PySpark MLlib (Estimator/Model). Reuses `spark_vi.core.runner.VIRunner`, `spark_vi.models.topic.lda.OnlineLDA` / `_cavi_doc_inference`, `spark_vi.models.topic.dag_placement.DagLayout`, `spark_vi.models.topic.spectral_init`.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-07-15-gated-svi-placement-engine-design.md` (amended 2026-07-15 from the prototype). The prototype code to lift from is in `scratchpad/gated_svi_proto.py`, `scratchpad/dag_spectral_init.py`.
- **Equivalence gate is placement-based + depth-weighted**, NOT per-node β cosine. Validate against the Gibbs oracle `fit_gated`: node-AUC-by-depth, MRR, top2 within Monte-Carlo tolerance; deeper-node AUC is the metric held to the oracle; plus per-engine ground-truth own-block argmax recovery. Never assert engine-to-engine β cosine.
- **Init is pluggable; `"random"` is the default** (the DAG gate supplies the identifiability spectral init exists to provide in ungated LDA). `"spectral"` = block-aligned, **forward-topological (ancestors-first)**, a validated *optional* strategy. Extension point for future strategies (e.g. phenotype-profile seeding).
- **Domain-agnostic engine:** integer ids only. No clinical vocabulary in `spark_vi/…`. The DAG/frontier come from the OMOP layer.
- **Test honesty:** no threshold-loosening; if something fails, xfail with a reason pointing at the cause, do not weaken the assertion.
- **Cite literature in docstrings** for any method/default/constant from the literature (Hoffman 2010, Griffiths & Steyvers 2004, Arora 2013).
- **Hash IDs** in any row-level `.show()`/print of doc-level rows (aggregates/probabilities fine raw) — not expected here (synthetic integer ids), but the rule stands.
- **Commit trailer EXACTLY:**
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  ```
- **Branch `case-finding`** (experimental — do NOT merge to main). Verify remote via git; it may auto-push.
- **Test harness:** `cd spark-vi && ../.venv/bin/python -m pytest tests/<file> -v` (run from `spark-vi/`; `spark_vi` importable from the venv). Spark shim tests spin a tiny local SparkSession.

---

## File Structure

- **Create** `spark-vi/spark_vi/models/topic/gated_lda.py` — `GatedOnlineLDA(OnlineLDA)` (gated `local_update`, strategy-dispatching `initialize_global`) + `node_affinity(theta, lay)` pure helper.
- **Create** `spark-vi/spark_vi/models/topic/gated_init.py` — init strategies: the block-aligned forward-topological `spectral_block_aligned_lambda(...)` (generalizes `spectral_init.spectral_init_beta`) + `INIT_STRATEGIES` registry.
- **Modify** `spark-vi/spark_vi/models/topic/types.py` — add `GatedBOWDocument` (BOWDocument + `frontier`).
- **Modify** `spark-vi/tests/_stm_synth.py` — add `fit_gated_svi_local(...)` in-memory (no-Spark) driver, mirroring `fit_stm`.
- **Create** `spark-vi/spark_vi/mllib/topic/gated_lda.py` — `GatedLDAEstimator` / `GatedLDAModel` (labelCol = frontier, `dag` param, `nodeAffinity` transform; init `"random"` for v1).
- **Create** `spark-vi/tests/test_gated_lda.py` — engine unit tests + the placement equivalence gate.
- **Create** `spark-vi/tests/test_gated_init.py` — spectral strategy own-block recovery + registry tests.
- **Create** `spark-vi/tests/test_gated_lda_shim.py` — Spark fit/transform smoke.

---

## Task 1: GatedBOWDocument

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/types.py`
- Test: `spark-vi/tests/test_gated_lda.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `GatedBOWDocument(indices: np.ndarray, counts: np.ndarray, length: int, frontier: frozenset = frozenset())` — the gating input row; `frontier` is a frozenset of node ids (empty = ungated).

- [ ] **Step 1: Write the failing test**

Create `spark-vi/tests/test_gated_lda.py`:

```python
import numpy as np
from spark_vi.models.topic.types import GatedBOWDocument


def test_gated_bow_document_fields():
    d = GatedBOWDocument(
        indices=np.array([1, 3], dtype=np.int32),
        counts=np.array([2.0, 1.0]),
        length=3,
        frontier=frozenset({4, 5}),
    )
    assert d.indices.tolist() == [1, 3]
    assert d.counts.tolist() == [2.0, 1.0]
    assert d.length == 3
    assert d.frontier == frozenset({4, 5})


def test_gated_bow_document_frontier_defaults_empty():
    d = GatedBOWDocument(indices=np.array([0]), counts=np.array([1.0]), length=1)
    assert d.frontier == frozenset()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_gated_lda.py -v`
Expected: FAIL with ImportError (`GatedBOWDocument` not defined).

- [ ] **Step 3: Write minimal implementation**

Append to `spark-vi/spark_vi/models/topic/types.py`:

```python
@dataclass(frozen=True, slots=True)
class GatedBOWDocument:
    """Bag-of-words document tagged with a DAG frontier for gated topic training.

    Mirrors STMDocument's `groups` gating precedent, but the gate is a set of DAG
    node ids (the doc's frontier = most-specific attested nodes) rather than covariate
    groups. GatedOnlineLDA restricts each training doc's variational E-step to
    DagLayout.allowed_set(frontier). Empty frontier = ungated (full-K), used for
    held-out fold-in at deployment.

    Invariants (callers' responsibility — not enforced at construction):
      indices: sorted int32 array of token indices, all in [0, vocab_size).
      counts:  float64 array with len(counts) == len(indices), all > 0.
      length:  int total tokens (sum of counts).
      frontier: frozenset[int] of DAG node ids (empty = ungated).
    """
    indices: np.ndarray
    counts: np.ndarray
    length: int
    frontier: frozenset = frozenset()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_gated_lda.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/types.py spark-vi/tests/test_gated_lda.py
git commit -m "$(cat <<'EOF'
feat(gated-lda): GatedBOWDocument (BOW + frontier gating tag)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: GatedOnlineLDA core (gated E-step + node affinity)

**Files:**
- Create: `spark-vi/spark_vi/models/topic/gated_lda.py`
- Test: `spark-vi/tests/test_gated_lda.py`

**Interfaces:**
- Consumes: `GatedBOWDocument` (Task 1); `OnlineLDA`, `_cavi_doc_inference` (`models/topic/lda.py`); `DagLayout` (`models/topic/dag_placement.py`).
- Produces:
  - `GatedOnlineLDA(lay: DagLayout, vocab_size: int, *, init="random", **online_lda_kwargs)` — `self.lay`, `self.init`; `K = lay.K`.
  - Override `local_update(rows, global_params) -> dict` — gates each doc's CAVI to `lay.allowed_set(frontier)` (all K when frontier empty); scatters sstats to `lambda_stats[np.ix_(allowed, indices)]`.
  - Override `initialize_global(data_summary) -> dict` — `init="random"` → inherited Gamma λ; other strategies resolved in Task 3 (this task ships `"random"` only + a clear error for unknown names).
  - `node_affinity(theta: np.ndarray, lay: DagLayout) -> dict[int, float]` — module function: `{u: theta[lay.block[u]].sum() for u in lay.nodes}`.

- [ ] **Step 1: Write the failing tests**

Append to `spark-vi/tests/test_gated_lda.py`:

```python
from spark_vi.models.topic.dag_placement import DagLayout
from spark_vi.models.topic.gated_lda import GatedOnlineLDA, node_affinity

PARENT = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}


def _lay():
    return DagLayout(PARENT, n_bg=2, tpn=1)


def test_local_update_gates_sstats_to_allowed_rows_only():
    lay = _lay()
    V = 30
    m = GatedOnlineLDA(lay, V, alpha=0.1, eta=0.02, random_seed=0)
    gp = m.initialize_global(None)
    # A doc whose frontier is {3}: allowed = background (0,1) + closure blocks of 3.
    doc = GatedBOWDocument(indices=np.array([5, 6], dtype=np.int32),
                           counts=np.array([2.0, 1.0]), length=3,
                           frontier=frozenset({3}))
    allowed = set(lay.allowed_set(frozenset({3})).tolist())
    out = m.local_update([doc], gp)
    stats = out["lambda_stats"]
    disallowed = [k for k in range(lay.K) if k not in allowed]
    assert np.allclose(stats[disallowed], 0.0)          # gate: zero outside allowed
    assert stats[sorted(allowed)].sum() > 0.0           # allowed rows got mass


def test_local_update_empty_frontier_is_ungated():
    lay = _lay()
    V = 30
    m = GatedOnlineLDA(lay, V, alpha=0.1, eta=0.02, random_seed=0)
    gp = m.initialize_global(None)
    doc = GatedBOWDocument(indices=np.array([5, 6], dtype=np.int32),
                           counts=np.array([2.0, 1.0]), length=3,
                           frontier=frozenset())
    out = m.local_update([doc], gp)
    # Ungated: every topic row is eligible, so total sstats mass spreads over all K
    # (no row structurally forced to zero by the gate).
    assert out["lambda_stats"].sum() > 0.0
    assert out["n_docs"] == 1.0


def test_node_affinity_sums_blocks():
    lay = _lay()
    theta = np.zeros(lay.K)
    for u in lay.nodes:
        for k in lay.block[u]:
            theta[k] = 0.1
    aff = node_affinity(theta, lay)
    assert set(aff.keys()) == set(lay.nodes)
    for u in lay.nodes:
        assert np.isclose(aff[u], 0.1 * lay.tpn)


def test_unknown_init_strategy_raises():
    lay = _lay()
    import pytest
    with pytest.raises(ValueError, match="init"):
        GatedOnlineLDA(lay, 30, init="banana").initialize_global(None)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_gated_lda.py -v`
Expected: FAIL (ImportError: `gated_lda`).

- [ ] **Step 3: Write the implementation**

Create `spark-vi/spark_vi/models/topic/gated_lda.py`:

```python
"""GatedOnlineLDA: the SVI (variational) twin of the collapsed-Gibbs placement engine.

Overrides exactly two OnlineLDA methods:
  * local_update — restrict each training doc's CAVI to DagLayout.allowed_set(frontier)
    (the exact variational analogue of the Gibbs gate in dag_placement.fit_gated); sstats
    for disallowed topics stay zero, welding each node's topic to its subtree's documents.
  * initialize_global — dispatch on a pluggable init strategy (default "random").

Everything else (update_global SVI natural-gradient beta step, compute_elbo, combine_stats,
VIRunner integration) is inherited from OnlineLDA. Deployment folds held-out docs in UNGATED
(empty frontier -> full-K CAVI) -> theta -> node_affinity.

Validated against the collapsed-Gibbs oracle (dag_placement.fit_gated): placement (node-AUC by
depth, MRR, top2) matches at every depth; the DAG gate supplies the identifiability spectral
init provides in ungated LDA, so random init is the default (see the design spec's prototype
findings). References: Hoffman, Blei, Bach (2010) Online LDA; Griffiths & Steyvers (2004) the
oracle; the placement design docs/superpowers/specs/2026-07-15-gated-svi-placement-engine-design.md.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from scipy.special import digamma

from spark_vi.models.topic.dag_placement import DagLayout
from spark_vi.models.topic.lda import OnlineLDA, _cavi_doc_inference
from spark_vi.models.topic.types import GatedBOWDocument


def node_affinity(theta: np.ndarray, lay: DagLayout) -> dict[int, float]:
    """Per-node affinity from a full-K theta: the mass on each node's topic block.

    The SVI analogue of dag_placement.profile's per-node readout. The full dict IS the
    case-finding output; do not collapse to a single node."""
    return {u: float(theta[lay.block[u]].sum()) for u in lay.nodes}


class GatedOnlineLDA(OnlineLDA):
    def __init__(self, lay: DagLayout, vocab_size: int, *, init: str = "random", **kw) -> None:
        super().__init__(K=lay.K, vocab_size=vocab_size, **kw)
        self.lay = lay
        self.init = init

    def initialize_global(self, data_summary: Any | None) -> dict[str, np.ndarray]:
        """Random Gamma lambda (default), or a pluggable init strategy's lambda.

        "random": inherited OnlineLDA Gamma init — the validated default (the gate already
        welds topics to nodes, so no symmetry-breaking seed is needed). Other strategies
        (Task 3) resolve from gated_init.INIT_STRATEGIES and need the training corpus in
        data_summary; an unknown name raises."""
        gp = super().initialize_global(data_summary)
        if self.init == "random":
            return gp
        from spark_vi.models.topic.gated_init import INIT_STRATEGIES
        if self.init not in INIT_STRATEGIES:
            raise ValueError(
                f"unknown init strategy {self.init!r}; "
                f"known: {['random'] + sorted(INIT_STRATEGIES)}"
            )
        strat = INIT_STRATEGIES[self.init]
        gp["lambda"] = strat(data_summary, self.lay, self.V)
        return gp

    def local_update(
        self,
        rows: Iterable[GatedBOWDocument],
        global_params: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Gated E-step: per doc, CAVI over expElogbeta[allowed] with alpha[allowed];
        scatter sstats to lambda_stats[allowed, indices]. Disallowed topics get zero
        contribution — the variational twin of the Gibbs gate.

        allowed = lay.allowed_set(frontier) (background + closure blocks of the frontier),
        or all K when the frontier is empty (ungated). Cost is O(|allowed|) per token =
        the doc's frontier closure (bounded by DAG depth), not K."""
        lam = global_params["lambda"]
        alpha = global_params["alpha"]
        expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))

        lambda_stats = np.zeros_like(lam)
        doc_loglik_sum = 0.0
        doc_theta_kl_sum = 0.0
        n_docs = 0

        for doc in rows:
            if doc.frontier:
                allowed = self.lay.allowed_set(doc.frontier)
            else:
                allowed = np.arange(self.K)
            gamma_init = np.random.gamma(self.gamma_shape, 1.0 / self.gamma_shape,
                                         size=len(allowed))
            gamma, expElogthetad, phi_norm, _ = _cavi_doc_inference(
                indices=doc.indices,
                counts=doc.counts,
                expElogbeta=expElogbeta[allowed],
                alpha=alpha[allowed],
                gamma_init=gamma_init,
                max_iter=self.cavi_max_iter,
                tol=self.cavi_tol,
            )
            sstats_row = np.outer(expElogthetad, doc.counts / phi_norm)
            lambda_stats[np.ix_(allowed, doc.indices)] += sstats_row
            doc_loglik_sum += float(np.sum(doc.counts * np.log(phi_norm)))
            n_docs += 1

        return {
            "lambda_stats": lambda_stats,
            "doc_loglik_sum": np.array(doc_loglik_sum),
            "doc_theta_kl_sum": np.array(doc_theta_kl_sum),
            "n_docs": np.array(float(n_docs)),
        }
```

> Note: `doc_theta_kl_sum` is left at 0.0 in v1 (the placement gate does not use the ELBO; the inherited `compute_elbo` still runs but its per-doc KL term is a v2 refinement). If a later task needs a correct gated ELBO, accumulate `_dirichlet_kl(gamma, alpha[allowed])` here.

- [ ] **Step 4: Run to verify it passes**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_gated_lda.py -v`
Expected: PASS (6 tests: 2 from Task 1 + 4 here).

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/gated_lda.py spark-vi/tests/test_gated_lda.py
git commit -m "$(cat <<'EOF'
feat(gated-lda): GatedOnlineLDA gated E-step + node_affinity

Overrides OnlineLDA.local_update to restrict each training doc's CAVI to
DagLayout.allowed_set(frontier) and scatter sstats to allowed rows only
(the variational twin of the Gibbs gate); initialize_global dispatches a
pluggable init strategy, "random" default.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Block-aligned spectral init strategy (optional, forward-topological)

**Files:**
- Create: `spark-vi/spark_vi/models/topic/gated_init.py`
- Test: `spark-vi/tests/test_gated_init.py`

**Interfaces:**
- Consumes: `DagLayout`; `spectral_init.word_cooccurrence/find_anchors/recover_beta`; `dag_placement._as_counts`.
- Produces:
  - `spectral_block_aligned_lambda(data_summary, lay, V, *, scale=200.0) -> np.ndarray` — `(K, V)` λ; `data_summary` must carry `{"train_docs": [...], "train_labels": [...]}`. Forward-topological (ancestors-first) block-aligned recovery.
  - `INIT_STRATEGIES = {"spectral": spectral_block_aligned_lambda}` — the registry `GatedOnlineLDA.initialize_global` reads.

- [ ] **Step 1: Write the failing test**

Create `spark-vi/tests/test_gated_init.py`:

```python
import numpy as np
from spark_vi.models.topic.dag_placement import DagLayout
from spark_vi.models.topic.gated_init import (
    spectral_block_aligned_lambda, INIT_STRATEGIES,
)
from _stm_synth import dag_placement_corpus

PARENT = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}


def test_registry_exposes_spectral():
    assert INIT_STRATEGIES["spectral"] is spectral_block_aligned_lambda


def test_block_aligned_peaks_on_own_signature():
    """Each node's spectral-init topic block should peak (argmax over the signature
    region [C:]) on that node's OWN planted signature block — forward-topological
    ancestors-first deflation. This is the 'gate welds topic to node' property at init."""
    V, doc_len = 120, 60
    parent = PARENT
    docs, labels, _ = dag_placement_corpus(
        parent=parent, node_prev={u: 1 for u in range(1, 7)},
        V=V, doc_len=doc_len, seed=1)
    lay = DagLayout(parent, n_bg=2, tpn=1)
    nodes = lay.nodes
    C = V // 3
    sig = max(2, (V - C) // (len(nodes) + 1))
    own = {u: set(range(C + i * sig, C + i * sig + sig)) for i, u in enumerate(nodes)}

    lam = spectral_block_aligned_lambda(
        {"train_docs": docs[:1600], "train_labels": labels[:1600]}, lay, V)
    assert lam.shape == (lay.K, V)
    hits = 0
    for u in nodes:
        b = lam[lay.block[u]].mean(0)
        if (C + int(np.argmax(b[C:]))) in own[u]:
            hits += 1
    # Forward-topological block-aligned init peaks each node on its own block. Prototype:
    # 6/6 single-parent. Hold >=5/6 (allow one internal-node descendant-leak miss).
    assert hits >= 5, f"only {hits}/{len(nodes)} nodes peaked on their own block"
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_gated_init.py -v`
Expected: FAIL (ImportError: `gated_init`).

- [ ] **Step 3: Write the implementation**

Create `spark-vi/spark_vi/models/topic/gated_init.py` (lift from `scratchpad/dag_spectral_init.py`):

```python
"""Pluggable init strategies for GatedOnlineLDA.

A strategy is `f(data_summary, lay, V) -> (K, V) lambda`, called by
GatedOnlineLDA.initialize_global when init != "random". "random" is the validated default
(the DAG gate supplies identifiability) and lives in OnlineLDA.initialize_global, so it is
NOT in this registry.

`spectral_block_aligned_lambda` generalizes OnlineLDA.spectral_init.spectral_init_beta's
background->foreground init to a multi-level DAG: each node is recovered by anchor-word
spectral recovery (Arora et al. 2013) DEFLATED against its already-recovered closure-ancestor
anchors, so it must run in FORWARD topological order (ancestors first — a node can only be
deflated against ancestors already recovered). This is a documented OPTIONAL strategy: on the
synthetic plants it did not improve the gated fit (the gate already breaks symmetry) and could
regress shallow nodes when a recovered seed row was imperfect — see the design spec's prototype
findings. Kept for the real-DAG A/B harness and as the extension point for future strategies
(e.g. phenotype-profile seeding)."""
from __future__ import annotations

import numpy as np

from spark_vi.models.topic.dag_placement import _as_counts
from spark_vi.models.topic.spectral_init import (
    word_cooccurrence, find_anchors, recover_beta,
)


def _union_closure(front, lay):
    s = set()
    for f in front:
        for u in lay.closure(f):
            if u != 0:
                s.add(u)
    return s


def spectral_block_aligned_lambda(data_summary, lay, V, *, scale: float = 200.0) -> np.ndarray:
    """Forward-topological block-aligned spectral lambda seed.

    data_summary carries {"train_docs": [token-id arrays], "train_labels": [node id or
    frontier set per doc]}. Returns a (K, V) lambda = block-aligned beta * scale.

    Step 1 (background): pooled Q over all docs -> n_bg anchors -> background block.
    Step 2 (each node, ancestors-first by lay.depth): docs training node u = those with u in
    the union of their frontier closures; find tpn anchors on the within-node Q_u deflated
    against background + u's already-recovered proper-ancestor anchors (seed_rows AND
    include-then-drop in recover_beta), recover into u's block."""
    if not (isinstance(data_summary, dict)
            and "train_docs" in data_summary and "train_labels" in data_summary):
        raise ValueError(
            "spectral init requires data_summary={'train_docs':..., 'train_labels':...}"
        )
    train_docs = data_summary["train_docs"]
    train_labels = data_summary["train_labels"]
    counted = [_as_counts(d) for d in train_docs]
    fronts = [set(y) if hasattr(y, "__iter__") else {int(y)} for y in train_labels]
    trains = [_union_closure(f, lay) for f in fronts]

    beta = np.zeros((lay.K, V))
    Q_all = word_cooccurrence(counted, V)
    bg_anchors = find_anchors(Q_all, lay.n_bg)
    bg_beta = recover_beta(Q_all, bg_anchors)
    for i in range(min(lay.n_bg, bg_beta.shape[0])):
        beta[i] = bg_beta[i]

    node_anchors: dict[int, list] = {}
    for u in sorted(lay.nodes, key=lambda x: (lay.depth(x), x)):   # forward topological
        docs_u = [counted[d] for d in range(len(counted)) if u in trains[d]]
        if not docs_u:
            continue
        Q_u = word_cooccurrence(docs_u, V)
        anc = [a for a in lay.closure(u) if a not in (u, 0)]
        seed = list(bg_anchors) + [a for p in anc for a in node_anchors.get(p, [])]
        fg_anchors = find_anchors(Q_u, lay.tpn, seed_rows=seed)
        if not fg_anchors:
            continue
        node_anchors[u] = list(fg_anchors)
        combined_beta = recover_beta(Q_u, list(seed) + list(fg_anchors))
        fg_beta = combined_beta[len(seed):]
        for j, idx in enumerate(lay.block[u]):
            if j < fg_beta.shape[0]:
                beta[idx] = fg_beta[j]

    beta = beta + 1e-9                                    # keep lambda strictly positive
    return beta * float(scale)


INIT_STRATEGIES = {"spectral": spectral_block_aligned_lambda}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_gated_init.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Also confirm the engine wires the strategy**

Add to `spark-vi/tests/test_gated_init.py`:

```python
def test_gated_online_lda_uses_spectral_init():
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    V, doc_len = 120, 60
    docs, labels, _ = dag_placement_corpus(
        parent=PARENT, node_prev={u: 1 for u in range(1, 7)},
        V=V, doc_len=doc_len, seed=1)
    lay = DagLayout(PARENT, n_bg=2, tpn=1)
    m = GatedOnlineLDA(lay, V, init="spectral", alpha=0.1, eta=0.02, random_seed=0)
    gp = m.initialize_global({"train_docs": docs[:800], "train_labels": labels[:800]})
    # Spectral lambda differs from a pure random Gamma init (row sums reflect the *scale).
    assert gp["lambda"].shape == (lay.K, V)
    assert gp["lambda"].min() > 0.0
```

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_gated_init.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add spark-vi/spark_vi/models/topic/gated_init.py spark-vi/tests/test_gated_init.py
git commit -m "$(cat <<'EOF'
feat(gated-lda): block-aligned forward-topological spectral init strategy

Optional pluggable init (INIT_STRATEGIES["spectral"]) generalizing
spectral_init_beta's background->foreground recovery to the DAG with
ancestors-first deflation. Validated-negative for the fit (gate already
provides identifiability); kept for the real-DAG A/B harness + as the
extension point for future strategies. Random stays the default.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: No-Spark driver + placement equivalence gate vs Gibbs oracle

**Files:**
- Modify: `spark-vi/tests/_stm_synth.py`
- Test: `spark-vi/tests/test_gated_lda.py`

**Interfaces:**
- Consumes: `GatedOnlineLDA`, `node_affinity`, `GatedBOWDocument` (Task 2); `dag_placement.fit_gated/profile/evaluate/DagLayout` (oracle); `dag_placement_corpus` / `_multi` (`_stm_synth`).
- Produces: `fit_gated_svi_local(model, gated_docs, *, n_iter=200) -> global_params` — in-memory batch-VB driver (mirrors `fit_stm`); and `svi_node_profiles(model, gp, docs, lay) -> [dict]` for scoring via `evaluate`.

- [ ] **Step 1: Write the failing test (the equivalence gate)**

Append to `spark-vi/tests/test_gated_lda.py`:

```python
from spark_vi.models.topic.dag_placement import fit_gated, profile, evaluate
from _stm_synth import (
    dag_placement_corpus, fit_gated_svi_local, svi_node_profiles,
)


def test_svi_matches_gibbs_placement_single_parent():
    """Placement-based, depth-weighted equivalence gate (NOT beta cosine). Gated SVI must
    match the collapsed-Gibbs oracle on node-AUC-by-depth (deep depth held to the oracle),
    MRR, and top2 within Monte-Carlo tolerance. Prototype: aucD1=aucD2=1.0, mrr 0.914 vs
    Gibbs 0.905 at 200-250 iters."""
    parent = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
    V, doc_len, n_train = 120, 60, 1600
    docs, labels, _ = dag_placement_corpus(
        parent=parent, node_prev={u: 1 for u in range(1, 7)},
        V=V, doc_len=doc_len, seed=1)
    lay = DagLayout(parent, n_bg=2, tpn=1)
    tr_d, tr_l = docs[:n_train], labels[:n_train]
    te_d, te_l = docs[n_train:], labels[n_train:]

    # Gibbs oracle
    beta_g = fit_gated(tr_d, tr_l, lay, V, rng=np.random.default_rng(0))
    ev_g = evaluate([profile(d, beta_g, lay, rng=np.random.default_rng(i))
                     for i, d in enumerate(te_d)], te_l, lay)

    # Gated SVI (random init default)
    bow = [GatedBOWDocument(*_bow(d), frontier=frozenset({int(y)}))
           for d, y in zip(tr_d, tr_l)]
    m = GatedOnlineLDA(lay, V, alpha=0.1, eta=0.02, random_seed=0)
    gp = fit_gated_svi_local(m, bow, n_iter=250)
    ev_s = evaluate(svi_node_profiles(m, gp, te_d, lay), te_l, lay)

    # Depth-weighted: DEEP-node AUC must meet the oracle within tolerance.
    max_depth = max(lay.depth(u) for u in lay.nodes)
    assert ev_s["auc_by_depth"][max_depth] >= ev_g["auc_by_depth"][max_depth] - 0.05
    # Shallow AUC and ranking within Monte-Carlo tolerance of the oracle.
    for dep in ev_g["auc_by_depth"]:
        assert ev_s["auc_by_depth"][dep] >= ev_g["auc_by_depth"][dep] - 0.08
    assert ev_s["mrr"] >= ev_g["mrr"] - 0.06
    assert ev_s["top2"] >= ev_g["top2"] - 0.08


def _bow(tokens):
    idx, cnt = np.unique(np.asarray(tokens), return_counts=True)
    return idx.astype(np.int32), cnt.astype(np.float64), int(cnt.sum())
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_gated_lda.py::test_svi_matches_gibbs_placement_single_parent -v`
Expected: FAIL (ImportError: `fit_gated_svi_local` / `svi_node_profiles`).

- [ ] **Step 3: Write the driver + profile helper**

Append to `spark-vi/tests/_stm_synth.py`:

```python
def fit_gated_svi_local(model, gated_docs, *, n_iter=200, seed=0):
    """In-memory batch-VB driver for GatedOnlineLDA (no Spark), mirroring fit_stm.

    Full-batch lr=1.0 each iteration = variational EM — the cleanest regime for the
    SVI-vs-Gibbs placement equivalence gate. `model` is a GatedOnlineLDA; `gated_docs` are
    GatedBOWDocuments (frontier tags drive the gate)."""
    import numpy as np
    np.random.seed(seed)
    gp = model.initialize_global(None)
    for _ in range(n_iter):
        gp = model.update_global(gp, model.local_update(gated_docs, gp), learning_rate=1.0)
    return gp


def svi_node_profiles(model, gp, docs, lay):
    """Ungated full-K fold-in of each held-out doc -> theta -> node_affinity dict. The SVI
    analogue of [dag_placement.profile(...) for d in docs]; scored by dag_placement.evaluate."""
    import numpy as np
    from spark_vi.models.topic.types import GatedBOWDocument
    from spark_vi.models.topic.gated_lda import node_affinity
    out = []
    for d in docs:
        idx, cnt = np.unique(np.asarray(d), return_counts=True)
        bow = GatedBOWDocument(indices=idx.astype(np.int32),
                               counts=cnt.astype(np.float64), length=int(cnt.sum()),
                               frontier=frozenset())          # empty = ungated
        theta = model.infer_local(bow, gp)["theta"]
        out.append(node_affinity(theta, lay))
    return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_gated_lda.py::test_svi_matches_gibbs_placement_single_parent -v`
Expected: PASS (this fit runs the pure-Python Gibbs oracle + 250 SVI passes — allow a few minutes).

- [ ] **Step 5: Add the multi-parent equivalence case**

Append to `spark-vi/tests/test_gated_lda.py`:

```python
from _stm_synth import dag_placement_corpus_multi


def test_svi_matches_gibbs_placement_multi_parent():
    """Same placement-based gate on a multi-parent diamond with comorbid (set-valued)
    frontiers. Prototype: aucD1~0.99/aucD2~1.0 both engines, mrr within ~0.01."""
    parent = {1: 0, 2: 0, 3: 0, 4: [1, 2], 5: [1, 3]}
    V, doc_len, n_train = 120, 60, 1900
    docs, labels, _ = dag_placement_corpus_multi(
        parent=parent, leaf_prev={4: 1.0, 5: 1.0}, comorbid_rate=0.25,
        V=V, doc_len=doc_len, seed=1)
    lay = DagLayout(parent, n_bg=2, tpn=1)
    tr_d, tr_l = docs[:n_train], labels[:n_train]
    te_d, te_l = docs[n_train:], labels[n_train:]

    beta_g = fit_gated(tr_d, tr_l, lay, V, rng=np.random.default_rng(0))
    ev_g = evaluate([profile(d, beta_g, lay, rng=np.random.default_rng(i))
                     for i, d in enumerate(te_d)], te_l, lay)

    bow = [GatedBOWDocument(*_bow(d), frontier=f) for d, f in zip(tr_d, tr_l)]
    m = GatedOnlineLDA(lay, V, alpha=0.1, eta=0.02, random_seed=0)
    gp = fit_gated_svi_local(m, bow, n_iter=250)
    ev_s = evaluate(svi_node_profiles(m, gp, te_d, lay), te_l, lay)

    max_depth = max(lay.depth(u) for u in lay.nodes)
    assert ev_s["auc_by_depth"][max_depth] >= ev_g["auc_by_depth"][max_depth] - 0.08
    assert ev_s["mrr"] >= ev_g["mrr"] - 0.08
```

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_gated_lda.py -v`
Expected: PASS (all gated_lda tests).

- [ ] **Step 6: Commit**

```bash
git add spark-vi/tests/_stm_synth.py spark-vi/tests/test_gated_lda.py
git commit -m "$(cat <<'EOF'
test(gated-lda): SVI-vs-Gibbs placement equivalence gate (depth-weighted)

In-memory batch-VB driver (fit_gated_svi_local) + ungated node-affinity
fold-in (svi_node_profiles); assert gated SVI matches the collapsed-Gibbs
oracle on node-AUC-by-depth (deep held to oracle), MRR, top2 within MC
tolerance on single- and multi-parent plants. NOT beta cosine.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: MLlib shim (GatedLDAEstimator / GatedLDAModel)

**Files:**
- Create: `spark-vi/spark_vi/mllib/topic/gated_lda.py`
- Test: `spark-vi/tests/test_gated_lda_shim.py`

**Interfaces:**
- Consumes: `GatedOnlineLDA`, `node_affinity`, `GatedBOWDocument` (Task 2); `DagLayout`; `VIRunner`, `VIConfig`; `mllib/topic/_common._vector_to_bow_document`; the `mllib/topic/lda.py` shim pattern (mirror it).
- Produces:
  - `GatedLDAEstimator(featuresCol, labelCol, parent, nBg=2, tpn=1, maxIter, seed, …OnlineLDA params)` — `parent` is the DAG parent map `{child_int: parent_int | [parent_ints]}` (anchor → 0). `fit` builds `GatedBOWDocument`s (features + frontier from `labelCol`, an array of node ids) and runs `VIRunner` with `GatedOnlineLDA(init="random")`.
  - `GatedLDAModel` — holds trained `VIResult` + the `DagLayout`; `transform` adds a `nodeAffinity` Vector column (width = #nodes, in `lay.nodes` order).

- [ ] **Step 1: Write the failing Spark smoke test**

Create `spark-vi/tests/test_gated_lda_shim.py`:

```python
import numpy as np
import pytest

pyspark = pytest.importorskip("pyspark")
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors


@pytest.fixture(scope="module")
def spark():
    s = (SparkSession.builder.master("local[1]").appName("gated-lda-shim")
         .config("spark.ui.enabled", "false").getOrCreate())
    yield s
    s.stop()


def test_gated_shim_fit_transform_smoke(spark):
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    parent = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
    V = 30
    # tiny planted rows: features SparseVector + frontier (array of node ids)
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(40):
        leaf = int(rng.choice([3, 4, 5, 6]))
        idx = sorted(rng.choice(V, size=6, replace=False).tolist())
        rows.append((Vectors.sparse(V, idx, [1.0] * len(idx)), [leaf]))
    df = spark.createDataFrame(rows, ["features", "frontier"])

    est = GatedLDAEstimator(featuresCol="features", labelCol="frontier",
                            parent=parent, nBg=2, tpn=1, maxIter=3, seed=0)
    model = est.fit(df)
    out = model.transform(df)
    assert "nodeAffinity" in out.columns
    aff = out.select("nodeAffinity").head()[0]
    n_nodes = len({1, 2, 3, 4, 5, 6})
    assert len(aff) == n_nodes            # one affinity per DAG node
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_gated_lda_shim.py -v`
Expected: FAIL (ImportError: `mllib.topic.gated_lda`).

- [ ] **Step 3: Write the shim**

Create `spark-vi/spark_vi/mllib/topic/gated_lda.py`, mirroring `mllib/topic/lda.py`. Key differences from the LDA shim: `labelCol` (frontier) + `parent`/`nBg`/`tpn` Params build a `DagLayout`; the fit RDD maps each row to a `GatedBOWDocument(features, frontier)`; `transform` outputs `nodeAffinity` via ungated fold-in + `node_affinity`.

```python
"""MLlib Estimator/Model shim for GatedOnlineLDA (hierarchical case-finding placement).

Mirrors mllib/topic/lda.py (ADR 0009): a translation layer over GatedOnlineLDA + VIRunner.
fit trains GATED (each row carries features + a frontier = set of DAG node ids); transform
folds held-out docs in UNGATED (full-K) and emits per-node affinity. v1 uses init="random"
(the validated default); block-aligned spectral init on Spark (distributed co-occurrence) is
deferred (the in-engine "spectral" strategy is validated in the no-Spark harness).
"""
from __future__ import annotations

import numpy as np
from pyspark import StorageLevel, keyword_only
from pyspark.ml.base import Estimator, Model
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasFeaturesCol, HasLabelCol, HasMaxIter, HasSeed

from spark_vi.core.config import VIConfig
from spark_vi.models.topic.dag_placement import DagLayout
from spark_vi.models.topic.gated_lda import GatedOnlineLDA, node_affinity
from spark_vi.models.topic.types import GatedBOWDocument
from spark_vi.mllib.topic._common import _vector_to_bow_document


class _GatedLDAParams(HasFeaturesCol, HasLabelCol, HasMaxIter, HasSeed):
    parent = Param(Params._dummy(), "parent",
                   "DAG parent map {child_int: parent_int or [parent_ints]}, anchor->0",
                   typeConverter=TypeConverters.identity)
    nBg = Param(Params._dummy(), "nBg", "number of shared background topics",
                typeConverter=TypeConverters.toInt)
    tpn = Param(Params._dummy(), "tpn", "topics per DAG node",
                typeConverter=TypeConverters.toInt)
    nodeAffinityCol = Param(Params._dummy(), "nodeAffinityCol",
                            "output column: per-node affinity Vector",
                            typeConverter=TypeConverters.toString)
    caviMaxIter = Param(Params._dummy(), "caviMaxIter", "inner CAVI max iters",
                        typeConverter=TypeConverters.toInt)
    caviTol = Param(Params._dummy(), "caviTol", "inner CAVI tolerance",
                    typeConverter=TypeConverters.toFloat)
    gammaShape = Param(Params._dummy(), "gammaShape", "Gamma init shape for gamma/lambda",
                       typeConverter=TypeConverters.toFloat)


def _layout(est_or_model) -> DagLayout:
    return DagLayout(est_or_model.getOrDefault("parent"),
                     n_bg=est_or_model.getOrDefault("nBg"),
                     tpn=est_or_model.getOrDefault("tpn"))


class GatedLDAEstimator(_GatedLDAParams, Estimator):
    @keyword_only
    def __init__(self, *, featuresCol="features", labelCol="frontier", parent=None,
                 nBg=2, tpn=1, maxIter=20, seed=None, caviMaxIter=100, caviTol=1e-3,
                 gammaShape=100.0):
        super().__init__()
        self._setDefault(featuresCol="features", labelCol="frontier", nBg=2, tpn=1,
                         maxIter=20, nodeAffinityCol="nodeAffinity",
                         caviMaxIter=100, caviTol=1e-3, gammaShape=100.0)
        self.setParams(**self._input_kwargs)

    @keyword_only
    def setParams(self, **kwargs):
        return self._set(**kwargs)

    def _fit(self, dataset) -> "GatedLDAModel":
        from spark_vi.core.runner import VIRunner
        if self.getOrDefault("parent") is None:
            raise ValueError("GatedLDAEstimator requires a `parent` DAG map.")
        lay = _layout(self)

        features_col = self.getOrDefault("featuresCol")
        label_col = self.getOrDefault("labelCol")
        first = dataset.select(features_col).head(1)
        if not first:
            raise ValueError("Cannot fit on an empty DataFrame.")
        V = first[0][0].size
        seed = self.getOrDefault("seed") if self.isSet("seed") else None

        model_obj = GatedOnlineLDA(
            lay, V, init="random",
            alpha=1.0 / lay.K, eta=1.0 / lay.K,
            gamma_shape=self.getOrDefault("gammaShape"),
            cavi_max_iter=self.getOrDefault("caviMaxIter"),
            cavi_tol=self.getOrDefault("caviTol"),
            random_seed=seed,
        )
        config = VIConfig(max_iterations=self.getOrDefault("maxIter"), random_seed=seed)

        def _to_gated(row):
            bow = _vector_to_bow_document(row[0])
            frontier = frozenset(int(x) for x in (row[1] or []))
            return GatedBOWDocument(indices=bow.indices, counts=bow.counts,
                                    length=bow.length, frontier=frontier)

        rdd = (dataset.select(features_col, label_col).rdd.map(_to_gated)
               .persist(StorageLevel.MEMORY_AND_DISK))
        rdd.count()
        try:
            result = VIRunner(model_obj, config=config).fit(rdd)
        finally:
            rdd.unpersist(blocking=False)

        out = GatedLDAModel(result, parent=self.getOrDefault("parent"),
                            nBg=self.getOrDefault("nBg"), tpn=self.getOrDefault("tpn"))
        for p in self.params:
            if self.isSet(p):
                out._set(**{p.name: self.getOrDefault(p)})
            elif self.hasDefault(p):
                out._setDefault(**{p.name: self.getOrDefault(p)})
        return out


class GatedLDAModel(_GatedLDAParams, Model):
    _expected_model_class = "GatedOnlineLDA"

    def __init__(self, result, *, parent, nBg, tpn):
        super().__init__()
        self._result = result
        self._setDefault(featuresCol="features", labelCol="frontier", nBg=nBg, tpn=tpn,
                         parent=parent, nodeAffinityCol="nodeAffinity",
                         caviMaxIter=100, caviTol=1e-3, gammaShape=100.0)

    @property
    def result(self):
        return self._result

    def _transform(self, dataset):
        from pyspark.ml.linalg import DenseVector, VectorUDT
        from pyspark.sql import functions as F
        from scipy.special import digamma
        from spark_vi.models.topic.lda import _cavi_doc_inference

        lay = _layout(self)
        lam = self._result.global_params["lambda"]
        expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))
        alpha = self._result.global_params["alpha"]
        gamma_shape = float(self.getOrDefault("gammaShape"))
        cavi_max_iter = int(self.getOrDefault("caviMaxIter"))
        cavi_tol = float(self.getOrDefault("caviTol"))
        K = expElogbeta.shape[0]
        nodes = list(lay.nodes)
        blocks = {u: lay.block[u] for u in nodes}

        sc = dataset.sparkSession.sparkContext
        bcast = sc.broadcast({
            "expElogbeta": expElogbeta, "alpha": alpha, "gamma_shape": gamma_shape,
            "cavi_max_iter": cavi_max_iter, "cavi_tol": cavi_tol, "K": K,
            "nodes": nodes, "blocks": blocks,
        })

        def _affinity(features):
            p = bcast.value
            doc = _vector_to_bow_document(features)
            rng = np.random.default_rng()
            gamma_init = rng.gamma(p["gamma_shape"], 1.0 / p["gamma_shape"], size=p["K"])
            gamma, _, _, _ = _cavi_doc_inference(
                indices=doc.indices, counts=doc.counts, expElogbeta=p["expElogbeta"],
                alpha=p["alpha"], gamma_init=gamma_init,
                max_iter=p["cavi_max_iter"], tol=p["cavi_tol"])
            theta = gamma / gamma.sum()
            return DenseVector([float(theta[p["blocks"][u]].sum()) for u in p["nodes"]])

        udf = F.udf(_affinity, returnType=VectorUDT())
        try:
            out_col = self.getOrDefault("nodeAffinityCol")
            return dataset.withColumn(out_col, udf(F.col(self.getOrDefault("featuresCol"))))
        finally:
            bcast.unpersist(blocking=False)
```

> `TypeConverters.identity` keeps the `parent` dict intact (it is a small broadcastable map). If `identity` is unavailable in the pinned PySpark, store `parent` as an instance attribute set in `__init__`/`_fit` instead of a Param (mirror how `lda.py` stores `_on_iteration`), and drop it from the param-copy loop. Confirm against the installed PySpark before choosing.

- [ ] **Step 4: Run to verify it passes**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_gated_lda_shim.py -v`
Expected: PASS (fit 3 iters + transform on a tiny local SparkSession).

- [ ] **Step 5: Full-file regression**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_gated_lda.py tests/test_gated_init.py tests/test_gated_lda_shim.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add spark-vi/spark_vi/mllib/topic/gated_lda.py spark-vi/tests/test_gated_lda_shim.py
git commit -m "$(cat <<'EOF'
feat(gated-lda): MLlib shim GatedLDAEstimator/Model (nodeAffinity transform)

fit trains gated (features + frontier labelCol) via VIRunner + GatedOnlineLDA;
transform folds held-out docs in ungated and emits a per-node affinity Vector.
labelCol = frontier (semantic); parent/nBg/tpn build the DagLayout. init=random
(spectral-on-Spark deferred). Mirrors mllib/topic/lda.py (ADR 0009).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- Hard gate on `local_update` → Task 2. ✅
- `GatedBOWDocument` → Task 1. ✅
- Pluggable init, random default + spectral optional + extension point → Tasks 2 (dispatch) & 3 (spectral). ✅
- Ungated `nodeAffinity` fold-in → Task 2 (`node_affinity`), Tasks 4 & 5 (fold-in usage). ✅
- MLlib shim (labelCol=frontier, dag param, nodeAffinity) → Task 5. ✅
- Placement-based depth-weighted equivalence gate (not β cosine) → Task 4. ✅
- Scaling note (O(|allowed|) per token) → captured in Task 2 docstring. ✅
- Deferred (DMR v2, distributed-spectral-on-Spark, phenotype-profile init, sparse β, OMOP pieces 2/3) → not in any task, correctly out of scope. ✅

**Type consistency:** `GatedBOWDocument(indices, counts, length, frontier)` used identically in Tasks 1/2/4/5. `node_affinity(theta, lay)` signature consistent Tasks 2/4/5. Init strategy signature `f(data_summary, lay, V)` consistent Tasks 2/3. `fit_gated_svi_local(model, gated_docs, n_iter)` consistent Task 4.

**Placeholder scan:** none — every code step is complete.

**Risk note for the executor:** Tasks 4's two equivalence tests run the pure-Python collapsed-Gibbs oracle (`fit_gated`, ~150 sweeps) plus 250 SVI passes — each takes a few minutes. That is expected, not a hang. Do not reduce iterations to speed them up (250 passes is what reaches the oracle-matching depth-2 AUC per the prototype); if they exceed a reasonable wall-clock, run them individually with `-k`.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-15-gated-svi-placement-engine.md`. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, two-stage review between tasks, fast iteration.

**2. Inline Execution** — execute tasks in this session with checkpoints for review.

Which approach?
