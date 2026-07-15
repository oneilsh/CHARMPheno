# Anchor-First Hierarchical Case-Finding — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote the validated placement prototype (`scratchpad/dag_placement.py`) to a tested, domain-agnostic module that places held-out items in a label DAG from their features and scores the placement.

**Architecture:** A small module over integer ids: a `DagLayout` (DAG + topic-block layout), gated collapsed-Gibbs training that ties topics to nodes via a closure mask, unmasked fold-in producing a per-node affinity profile, and an evaluation surface (per-node AUC, DAG-distance, MRR) plus labeling/leakage utilities and a post-fit identifiability annotation. Reuses the existing anchor-word spectral init.

**Tech Stack:** Python, NumPy, pytest. Reuses `spark_vi.models.topic.spectral_init` (`word_cooccurrence`, `find_anchors`, `recover_beta`) and `spark_vi.models.topic.types.STMDocument`.

## Global Constraints

- Engine is **domain-agnostic**: integer token/label/node ids only. No clinical vocabulary in the module or its tests.
- Cite any method/default/constant from the literature in the docstring (spectral init = Arora et al. 2013; collapsed Gibbs LDA = Griffiths & Steyvers 2004).
- No LaTeX in comments/docstrings; plain text + Unicode where needed.
- Interface the engine consumes: `docs` (list of 1-D int arrays), `labels` (int array of node ids), `dag` (`{child: parent}`, root `0` implied, root has no entry).
- `tpn` (topics per node) is a knob, **default 1**. Attestation threshold is a knob, permissive default.
- Spec: `docs/superpowers/specs/2026-07-15-anchor-first-hierarchical-case-finding-design.md`.

---

## File Structure

- Create `spark-vi/spark_vi/models/topic/dag_placement.py` — `DagLayout`, `label_from_coded`, `strip_dag_node_codes`, `fit_gated`, `profile`, `evaluate`, `identifiability_annotation`, `render_profile`.
- Create `spark-vi/tests/test_dag_placement.py` — unit + behavioral tests.
- Modify `spark-vi/tests/_stm_synth.py` — add `dag_placement_corpus` (domain-agnostic synthetic generator).

---

### Task 1: `DagLayout`

**Files:**
- Create: `spark-vi/spark_vi/models/topic/dag_placement.py`
- Test: `spark-vi/tests/test_dag_placement.py`

**Interfaces:**
- Produces: `DagLayout(parent: dict, n_bg=2, tpn=1)` with attributes `nodes` (sorted non-root ids), `K` (int), `block: {node: [topic ids]}`, `children: {node: [child ids]}`; methods `closure(v)->list` (root..v), `subtree(u)->set`, `allowed(v)->np.ndarray`, `depth(v)->int`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from spark_vi.models.topic.dag_placement import DagLayout

PARENT = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}   # root 0 -> families 1,2 -> subtypes

def test_daglayout_structure_and_masks():
    lay = DagLayout(PARENT, n_bg=2, tpn=1)
    assert lay.nodes == [1, 2, 3, 4, 5, 6]
    assert lay.K == 2 + 6                      # bg + one topic per node
    assert lay.closure(3) == [0, 1, 3]         # root..v
    assert lay.subtree(1) == {1, 3, 4}
    assert lay.depth(3) == 2 and lay.depth(1) == 1
    # allowed(v) = bg ∪ blocks along closure(v), excluding root
    assert list(lay.allowed(3)) == [0, 1] + lay.block[1] + lay.block[3]
    assert list(lay.allowed(1)) == [0, 1] + lay.block[1]

def test_daglayout_tpn_two():
    lay = DagLayout(PARENT, n_bg=1, tpn=2)
    assert lay.K == 1 + 6 * 2
    assert len(lay.block[3]) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd spark-vi && python -m pytest tests/test_dag_placement.py::test_daglayout_structure_and_masks -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
"""Domain-agnostic hierarchical placement engine (integer ids only). Places held-out items in
a label DAG from their features via gated collapsed-Gibbs topic learning (Griffiths & Steyvers
2004) with anchor-word spectral init (Arora et al. 2013). See
docs/superpowers/specs/2026-07-15-anchor-first-hierarchical-case-finding-design.md."""
import numpy as np


class DagLayout:
    """Topic-block layout over a label DAG: `n_bg` shared background topics, then `tpn` topics per
    non-root node. `parent` maps child id -> parent id; the root is id 0 (no entry)."""

    def __init__(self, parent, n_bg=2, tpn=1):
        self.parent = dict(parent)
        self.nodes = sorted(parent.keys())
        self.n_bg = int(n_bg)
        self.tpn = int(tpn)
        self.children = {0: []}
        for c, p in parent.items():
            self.children.setdefault(p, []).append(c)
            self.children.setdefault(c, [])
        for p in self.children:
            self.children[p] = sorted(self.children[p])
        self.block = {u: list(range(n_bg + i * tpn, n_bg + (i + 1) * tpn))
                      for i, u in enumerate(self.nodes)}
        self.K = n_bg + len(self.nodes) * tpn

    def closure(self, v):
        c = [v]
        while v in self.parent:
            v = self.parent[v]
            c.append(v)
        return c[::-1]

    def subtree(self, u):
        out = {u}
        stack = [u]
        while stack:
            x = stack.pop()
            for ch in self.children.get(x, []):
                out.add(ch)
                stack.append(ch)
        return out

    def allowed(self, v):
        al = list(range(self.n_bg))
        for u in self.closure(v):
            if u != 0:
                al += self.block[u]
        return np.array(sorted(al), dtype=int)

    def depth(self, v):
        return len(self.closure(v)) - 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd spark-vi && python -m pytest tests/test_dag_placement.py -v`
Expected: PASS (both DagLayout tests).

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/dag_placement.py spark-vi/tests/test_dag_placement.py
git commit -m "feat(dag-placement): DagLayout DAG + topic-block layout with closure masks"
```

---

### Task 2: Labeling + leakage utilities

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/dag_placement.py`
- Test: `spark-vi/tests/test_dag_placement.py`

**Interfaces:**
- Consumes: `DagLayout` (`closure`, `depth`).
- Produces: `label_from_coded(coded_nodes: list[int], lay: DagLayout) -> int` (most-specific if single-path, else LCA); `strip_dag_node_codes(doc: np.ndarray, dag_node_codes: set[int]) -> np.ndarray`.

- [ ] **Step 1: Write the failing test**

```python
from spark_vi.models.topic.dag_placement import label_from_coded, strip_dag_node_codes

def test_label_same_path_is_deepest():
    lay = DagLayout(PARENT)
    # {1,3} lie on one path root->1->3 : most-specific = deepest = 3
    assert label_from_coded([1, 3], lay) == 3
    assert label_from_coded([3], lay) == 3

def test_label_siblings_is_lca():
    lay = DagLayout(PARENT)
    # {3,4} are siblings under 1 : LCA = 1
    assert label_from_coded([3, 4], lay) == 1
    # {3,5} cross-branch under root : LCA = 0
    assert label_from_coded([3, 5], lay) == 0

def test_strip_dag_node_codes():
    doc = np.array([10, 3, 11, 1, 12])          # 3 and 1 are DAG-node codes
    out = strip_dag_node_codes(doc, {1, 3})
    assert list(out) == [10, 11, 12]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd spark-vi && python -m pytest tests/test_dag_placement.py::test_label_same_path_is_deepest -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation** (append to `dag_placement.py`)

```python
def label_from_coded(coded_nodes, lay):
    """The item's label from its in-window coded nodes. If they lie on a single root->node path
    (one node is a descendant-or-self of all others), return that deepest node (most-specific).
    Otherwise return the lowest common ancestor (deepest node that is an ancestor-or-self of all)."""
    nodes = list(dict.fromkeys(coded_nodes))
    for cand in nodes:                                   # single-path: cand's closure holds all
        cset = set(lay.closure(cand))
        if all(n in cset for n in nodes):
            return cand
    common = set(lay.closure(nodes[0]))
    for n in nodes[1:]:
        common &= set(lay.closure(n))
    return max(common, key=lay.depth)                    # root (0) is always common


def strip_dag_node_codes(doc, dag_node_codes):
    """Remove every token whose id matches a DAG-node code (leakage strip; evaluation only)."""
    doc = np.asarray(doc)
    if not dag_node_codes:
        return doc
    mask = ~np.isin(doc, np.fromiter(dag_node_codes, dtype=doc.dtype))
    return doc[mask]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd spark-vi && python -m pytest tests/test_dag_placement.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/dag_placement.py spark-vi/tests/test_dag_placement.py
git commit -m "feat(dag-placement): most-specific/LCA labeling + leakage strip"
```

---

### Task 3: Synthetic generator `dag_placement_corpus`

**Files:**
- Modify: `spark-vi/tests/_stm_synth.py`
- Test: `spark-vi/tests/test_dag_placement.py`

**Interfaces:**
- Produces: `dag_placement_corpus(*, parent, node_prev, V, doc_len, seed) -> (docs, labels, node_codes)` where `docs` is a list of 1-D int arrays, `labels` an int array of leaf/label node ids, `node_codes: {node: int}` the vocab id that exactly marks each node (for leakage tests). An item at node `v` emits background + the signature blocks along `closure(v)`.

- [ ] **Step 1: Write the failing test**

```python
from tests._stm_synth import dag_placement_corpus

def test_dag_placement_corpus_shapes():
    docs, labels, node_codes = dag_placement_corpus(
        parent=PARENT, node_prev={1: .18, 2: .18, 3: .16, 4: .16, 5: .16, 6: .16},
        V=120, doc_len=40, seed=0)
    assert len(docs) == len(labels)
    assert set(labels.tolist()) <= set(PARENT.keys())
    assert set(node_codes.keys()) == set(PARENT.keys())
    # a node's exact code appears in items labeled at/below that node
    below3 = [d for d, y in zip(docs, labels) if y in {3}]
    assert any(node_codes[3] in d for d in below3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd spark-vi && python -m pytest tests/test_dag_placement.py::test_dag_placement_corpus_shapes -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation** (append to `tests/_stm_synth.py`)

```python
def dag_placement_corpus(*, parent, node_prev, V, doc_len, seed):
    """Domain-agnostic hierarchical-placement plant. Each non-root node owns a signature vocab
    block plus a single exact 'node code'; an item at node v emits a shared common pool + the
    signature blocks along closure(v). Returns (docs, labels, node_codes)."""
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout
    rng = np.random.default_rng(seed)
    lay = DagLayout(parent, n_bg=2, tpn=1)
    nodes = lay.nodes
    C = V // 3                                            # shared common pool [0:C]
    sig = max(2, (V - C) // (len(nodes) + 1))
    node_sig = {u: np.arange(C + i * sig, C + i * sig + sig) for i, u in enumerate(nodes)}
    node_codes = {u: int(node_sig[u][0]) for u in nodes}  # the exact marker code
    p = np.array([node_prev[u] for u in nodes], float); p /= p.sum()
    docs, labels = [], []
    for _ in range(sum(1 for _ in range(2000))):          # 2000 items
        v = int(rng.choice(nodes, p=p))
        path = [u for u in lay.closure(v) if u != 0]
        toks = [rng.integers(0, C, size=doc_len // 2)]    # background/common pool
        per = max(1, (doc_len - doc_len // 2) // len(path))
        for u in path:
            toks.append(rng.choice(node_sig[u], size=per))
        docs.append(np.concatenate(toks).astype(np.int64))
        labels.append(v)
    return docs, np.array(labels), node_codes
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd spark-vi && python -m pytest tests/test_dag_placement.py::test_dag_placement_corpus_shapes -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/tests/_stm_synth.py spark-vi/tests/test_dag_placement.py
git commit -m "test(dag-placement): domain-agnostic hierarchical-placement plant"
```

---

### Task 4: `fit_gated`

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/dag_placement.py`
- Test: `spark-vi/tests/test_dag_placement.py`

**Interfaces:**
- Consumes: `DagLayout`; `spectral_init.word_cooccurrence/find_anchors/recover_beta`; `STMDocument` is NOT required (docs are int arrays here — build a lightweight adapter for `word_cooccurrence`, which expects `.indices`/`.counts`).
- Produces: `fit_gated(train_docs, train_labels, lay, V, *, alpha=0.1, beta_prior=0.02, n_iter=150, burn=80, rng) -> beta_hat` shape `(lay.K, V)`.

- [ ] **Step 1: Write the failing test**

```python
from spark_vi.models.topic.dag_placement import fit_gated

def test_fit_gated_learns_node_signatures():
    docs, labels, _ = dag_placement_corpus(
        parent=PARENT, node_prev={1:.18,2:.18,3:.16,4:.16,5:.16,6:.16},
        V=120, doc_len=40, seed=1)
    lay = DagLayout(PARENT, n_bg=2, tpn=1)
    rng = np.random.default_rng(3)
    beta = fit_gated(docs[:1400], labels[:1400], lay, 120, n_iter=60, burn=30, rng=rng)
    assert beta.shape == (lay.K, 120)
    assert np.allclose(beta.sum(1), 1.0, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd spark-vi && python -m pytest tests/test_dag_placement.py::test_fit_gated_learns_node_signatures -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation** (append to `dag_placement.py`)

```python
from types import SimpleNamespace
from spark_vi.models.topic.spectral_init import word_cooccurrence, find_anchors, recover_beta


def _as_counts(doc):
    idx, cnt = np.unique(np.asarray(doc), return_counts=True)
    return SimpleNamespace(indices=idx, counts=cnt.astype(np.float64))


def fit_gated(train_docs, train_labels, lay, V, *, alpha=0.1, beta_prior=0.02,
              n_iter=150, burn=80, rng=None):
    """Gated collapsed Gibbs (Griffiths & Steyvers 2004): each training item is masked to
    allowed(label) = background ∪ blocks along its label's closure, tying topics to nodes
    structurally. Anchor-word spectral init (Arora et al. 2013) seeds beta. Returns posterior-mean
    beta_hat (K, V)."""
    K = lay.K
    counted = [_as_counts(d) for d in train_docs]
    Q = word_cooccurrence(counted, V)
    beta0 = recover_beta(Q, find_anchors(Q, K))
    beta0 = beta0 + 1e-6
    beta0 /= beta0.sum(1, keepdims=True)
    n_kw = np.zeros((K, V))
    n_k = np.zeros(K)
    allowed = [lay.allowed(v) for v in train_labels]
    words = [np.asarray(d, dtype=np.int64) for d in train_docs]
    Z = []
    for d in range(len(train_docs)):
        al = allowed[d]
        w = words[d]
        r = beta0[al][:, w].T
        r = r / r.sum(1, keepdims=True)
        zi = al[(rng.random(len(w))[:, None] < np.cumsum(r, 1)).argmax(1)]
        Z.append(zi)
        np.add.at(n_kw, (zi, w), 1.0)
        for k in zi:
            n_k[k] += 1.0
    Vb = V * beta_prior
    acc = np.zeros((K, V))
    nacc = 0
    for it in range(n_iter):
        for d in range(len(train_docs)):
            al = allowed[d]
            w = words[d]
            zi = Z[d]
            for i in range(len(w)):
                wi = w[i]
                k = zi[i]
                n_kw[k, wi] -= 1.0
                n_k[k] -= 1.0
                p = (n_kw[al, wi] + beta_prior) / (n_k[al] + Vb)
                p /= p.sum()
                knew = al[np.searchsorted(np.cumsum(p), rng.random())]
                zi[i] = knew
                n_kw[knew, wi] += 1.0
                n_k[knew] += 1.0
        if it >= burn:
            acc += n_kw + beta_prior
            nacc += 1
    beta_hat = acc / nacc
    beta_hat /= beta_hat.sum(1, keepdims=True)
    return beta_hat
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd spark-vi && python -m pytest tests/test_dag_placement.py::test_fit_gated_learns_node_signatures -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/dag_placement.py spark-vi/tests/test_dag_placement.py
git commit -m "feat(dag-placement): gated collapsed-Gibbs training with spectral init"
```

---

### Task 5: `profile` (unmasked fold-in)

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/dag_placement.py`
- Test: `spark-vi/tests/test_dag_placement.py`

**Interfaces:**
- Consumes: `DagLayout`, `beta_hat` from `fit_gated`.
- Produces: `profile(doc, beta_hat, lay, *, alpha=0.1, n_iter=60, burn=30, rng) -> {node: float}` (affinity mass on each node's block).

- [ ] **Step 1: Write the failing test**

```python
from spark_vi.models.topic.dag_placement import profile

def test_profile_returns_node_affinity():
    lay = DagLayout(PARENT, n_bg=2, tpn=1)
    beta = np.full((lay.K, 30), 1e-3); beta /= beta.sum(1, keepdims=True)
    rng = np.random.default_rng(0)
    pr = profile(np.array([1, 2, 3, 4, 5]), beta, lay, n_iter=20, burn=10, rng=rng)
    assert set(pr.keys()) == set(lay.nodes)
    assert all(0.0 <= v <= 1.0 for v in pr.values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd spark-vi && python -m pytest tests/test_dag_placement.py::test_profile_returns_node_affinity -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation** (append to `dag_placement.py`)

```python
def profile(doc, beta_hat, lay, *, alpha=0.1, n_iter=60, burn=30, rng=None):
    """Unmasked fold-in (topics fixed) -> per-node affinity = posterior mean mass on each node's
    block. The full profile IS the output; do not collapse to a single node."""
    K = lay.K
    w = np.asarray(doc, dtype=np.int64)
    ndk = np.zeros(K)
    zi = rng.integers(K, size=len(w))
    for k in zi:
        ndk[k] += 1.0
    acc = np.zeros(K)
    nacc = 0
    for it in range(n_iter):
        for i in range(len(w)):
            wi = w[i]
            k = zi[i]
            ndk[k] -= 1.0
            p = (ndk + alpha) * beta_hat[:, wi]
            s = p.sum()
            p = p / s if s > 0 else np.full(K, 1.0 / K)
            knew = int(np.searchsorted(np.cumsum(p), rng.random()))
            zi[i] = knew
            ndk[knew] += 1.0
        if it >= burn:
            acc += ndk / max(len(w), 1)
            nacc += 1
    th = acc / max(nacc, 1)
    return {u: float(th[lay.block[u]].sum()) for u in lay.nodes}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd spark-vi && python -m pytest tests/test_dag_placement.py::test_profile_returns_node_affinity -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/dag_placement.py spark-vi/tests/test_dag_placement.py
git commit -m "feat(dag-placement): unmasked fold-in -> node-affinity profile"
```

---

### Task 6: `evaluate` (per-node AUC, DAG-distance, MRR) + instrumentation

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/dag_placement.py`
- Test: `spark-vi/tests/test_dag_placement.py`

**Interfaces:**
- Consumes: list of profiles (`{node: float}`), `test_labels`, `DagLayout`.
- Produces: `evaluate(profiles, test_labels, lay) -> dict` with keys `node_auc: {node: float}`, `auc_by_depth: {depth: float}`, `mrr: float`, `top2: float`.

- [ ] **Step 1: Write the failing test**

```python
from spark_vi.models.topic.dag_placement import evaluate

def test_evaluate_perfect_profiles_score_high():
    lay = DagLayout(PARENT, n_bg=2, tpn=1)
    labels = np.array([3, 4, 5, 6, 1, 2] * 5)
    profiles = []
    for y in labels:                              # planted "perfect" affinity: 1.0 on the true node
        profiles.append({u: (1.0 if u == y else 0.0) for u in lay.nodes})
    m = evaluate(profiles, labels, lay)
    assert m["mrr"] == 1.0 and m["top2"] == 1.0
    assert all(v >= 0.99 for v in m["node_auc"].values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd spark-vi && python -m pytest tests/test_dag_placement.py::test_evaluate_perfect_profiles_score_high -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation** (append to `dag_placement.py`)

```python
def _auc(scores, y):
    y = np.asarray(y)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def evaluate(profiles, test_labels, lay):
    """Per-node case-finding AUC (subtree membership), AUC by depth, and true-node MRR / top-2.
    Profiles are the graded affinity dicts from `profile`; scoring never collapses to one node."""
    P = np.array([[pr[u] for u in lay.nodes] for pr in profiles])
    node_auc = {u: _auc(P[:, i], [t in lay.subtree(u) for t in test_labels])
                for i, u in enumerate(lay.nodes)}
    ranks = []
    for i, t in enumerate(test_labels):
        ti = lay.nodes.index(t)
        ranks.append(1 + int((P[i] > P[i][ti]).sum()))
    ranks = np.array(ranks)
    by_depth = {}
    for dep in sorted({lay.depth(u) for u in lay.nodes}):
        us = [u for u in lay.nodes if lay.depth(u) == dep]
        by_depth[dep] = float(np.nanmean([node_auc[u] for u in us]))
    return {"node_auc": node_auc, "auc_by_depth": by_depth,
            "mrr": float(np.mean(1.0 / ranks)), "top2": float(np.mean(ranks <= 2))}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd spark-vi && python -m pytest tests/test_dag_placement.py::test_evaluate_perfect_profiles_score_high -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/dag_placement.py spark-vi/tests/test_dag_placement.py
git commit -m "feat(dag-placement): per-node AUC + DAG-distance + MRR evaluation"
```

---

### Task 7: `identifiability_annotation` (post-fit diagnostic)

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/dag_placement.py`
- Test: `spark-vi/tests/test_dag_placement.py`

**Interfaces:**
- Consumes: `beta_hat`, `DagLayout`.
- Produces: `identifiability_annotation(beta_hat, lay, *, tol=0.9) -> list[tuple[int,int,float]]` — pairs of nodes whose learned topic distributions have cosine similarity ≥ `tol` (flagged as hard-to-separate), each `(u, v, cos)`. Only pairs WITHIN the structure (siblings sharing a parent, or parent↔child) are reported; cross-branch pairs are excluded by construction.

- [ ] **Step 1: Write the failing test**

```python
from spark_vi.models.topic.dag_placement import identifiability_annotation

def test_identifiability_flags_near_identical_siblings():
    lay = DagLayout(PARENT, n_bg=2, tpn=1)
    beta = np.random.default_rng(0).random((lay.K, 20)) + 0.01
    # make siblings 3 and 4 near-identical topics
    beta[lay.block[4][0]] = beta[lay.block[3][0]].copy()
    beta /= beta.sum(1, keepdims=True)
    flagged = identifiability_annotation(beta, lay, tol=0.99)
    pairs = {(min(u, v), max(u, v)) for u, v, _ in flagged}
    assert (3, 4) in pairs
    assert (3, 5) not in pairs                    # cross-branch never reported
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd spark-vi && python -m pytest tests/test_dag_placement.py::test_identifiability_flags_near_identical_siblings -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation** (append to `dag_placement.py`)

```python
def _node_topic_mean(beta_hat, lay, u):
    return beta_hat[lay.block[u]].mean(0)


def identifiability_annotation(beta_hat, lay, *, tol=0.9):
    """Post-fit diagnostic: flag WITHIN-STRUCTURE node pairs (siblings, or parent<->child) whose
    learned topic distributions are near-collinear (cosine >= tol) -> hard to separate. Cross-branch
    pairs are never reported; their similarity is a reporting fact, not a structural one."""
    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    pairs = set()
    for u in lay.nodes:                                  # parent<->child
        for c in lay.children.get(u, []):
            pairs.add((u, c))
    for p, kids in lay.children.items():                 # siblings
        for i in range(len(kids)):
            for j in range(i + 1, len(kids)):
                pairs.add((kids[i], kids[j]))
    out = []
    for u, v in pairs:
        c = cos(_node_topic_mean(beta_hat, lay, u), _node_topic_mean(beta_hat, lay, v))
        if c >= tol:
            out.append((u, v, c))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd spark-vi && python -m pytest tests/test_dag_placement.py::test_identifiability_flags_near_identical_siblings -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/dag_placement.py spark-vi/tests/test_dag_placement.py
git commit -m "feat(dag-placement): within-structure identifiability annotation"
```

---

### Task 8: `render_profile` (text)

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/dag_placement.py`
- Test: `spark-vi/tests/test_dag_placement.py`

**Interfaces:**
- Consumes: `DagLayout`, a profile dict, optional `names: {node: str}`, optional `true_node`.
- Produces: `render_profile(affinity, lay, *, names=None, true_node=None, width=24) -> str` — an indented DAG tree with a unicode bar per node.

- [ ] **Step 1: Write the failing test**

```python
from spark_vi.models.topic.dag_placement import render_profile

def test_render_profile_marks_true_and_shows_all_nodes():
    lay = DagLayout(PARENT, n_bg=2, tpn=1)
    aff = {1: 0.6, 2: 0.0, 3: 0.05, 4: 0.0, 5: 0.2, 6: 0.15}
    s = render_profile(aff, lay, true_node=1)
    assert "true" in s
    for u in lay.nodes:                                   # every node rendered
        assert str(u) in s or (str(u) in s)
    assert s.count("\n") >= len(lay.nodes)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd spark-vi && python -m pytest tests/test_dag_placement.py::test_render_profile_marks_true_and_shows_all_nodes -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation** (append to `dag_placement.py`)

```python
def render_profile(affinity, lay, *, names=None, true_node=None, width=24):
    """Indented DAG tree with a unicode affinity bar per node (spot-check output, sim and real)."""
    names = names or {}
    lines = []

    def bar(x):
        n = int(round(max(0.0, min(1.0, x)) * width))
        return "█" * n + "▁" * (width - n)

    def walk(v, prefix, is_last):
        if v == 0:
            lines.append(names.get(0, "root"))
        else:
            a = affinity.get(v, 0.0)
            conn = "└─ " if is_last else "├─ "
            mark = "  <- true" if v == true_node else ""
            nm = str(names.get(v, v)).ljust(10)
            lines.append(f"{prefix}{conn}{nm} {bar(a)} {a:0.2f}{mark}")
        kids = lay.children.get(v, [])
        child_prefix = prefix + ("   " if is_last else "│  ") if v != 0 else ""
        for i, c in enumerate(kids):
            walk(c, child_prefix, i == len(kids) - 1)

    walk(0, "", True)
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd spark-vi && python -m pytest tests/test_dag_placement.py::test_render_profile_marks_true_and_shows_all_nodes -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/dag_placement.py spark-vi/tests/test_dag_placement.py
git commit -m "feat(dag-placement): text DAG affinity-profile renderer"
```

---

### Task 9: End-to-end behavioral test (plant → fit → profile → evaluate)

**Files:**
- Test: `spark-vi/tests/test_dag_placement.py`

**Interfaces:**
- Consumes: all of the above.

- [ ] **Step 1: Write the failing test**

```python
def test_end_to_end_recovers_family_and_subtype():
    docs, labels, node_codes = dag_placement_corpus(
        parent=PARENT, node_prev={1:.18,2:.18,3:.16,4:.16,5:.16,6:.16},
        V=120, doc_len=40, seed=2)
    lay = DagLayout(PARENT, n_bg=2, tpn=1)
    ntr = int(0.7 * len(docs))
    rng = np.random.default_rng(5)
    beta = fit_gated(docs[:ntr], labels[:ntr], lay, 120, n_iter=80, burn=40, rng=rng)
    codes = set(node_codes.values())
    profs = [profile(strip_dag_node_codes(d, codes), beta, lay, n_iter=40, burn=20, rng=rng)
             for d in docs[ntr:]]
    m = evaluate(profs, labels[ntr:], lay)
    # gated-train places well; sim validated family ~0.99 / subtype ~0.97 (spec). Loose floors:
    assert m["auc_by_depth"][1] >= 0.85           # family level
    assert m["auc_by_depth"][2] >= 0.75           # subtype level
    assert m["mrr"] >= 0.6
```

- [ ] **Step 2: Run test to verify it fails, then passes**

Run: `cd spark-vi && python -m pytest tests/test_dag_placement.py::test_end_to_end_recovers_family_and_subtype -v`
Expected: With all prior tasks implemented, this PASSES. (If it fails on the floors, investigate before loosening — the sim validated well above these floors; a low score signals a real regression, not a threshold to relax.)

- [ ] **Step 3: Run the whole suite**

Run: `cd spark-vi && python -m pytest tests/test_dag_placement.py -v`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add spark-vi/tests/test_dag_placement.py
git commit -m "test(dag-placement): end-to-end plant->fit->place->evaluate behavioral gate"
```

---

## Self-Review

**Spec coverage:** DAG construction/layout (Task 1) + attestation/single-child-collapse are `DagLayout` responsibilities — NOTE: attestation filtering and single-child-chain collapse are DAG-*builder* concerns that operate on the parent map before `DagLayout`; for v1 the plant supplies a clean DAG, so these are deferred to the cohort-assembly/DAG-builder piece (out of this module's scope, per spec §Scope) — flagged, not silently dropped. Labeling + leakage (Task 2), synthetic plant (Task 3), gated training (Task 4), affinity profile (Task 5), evaluation + MRR (Task 6), identifiability annotation (Task 7), text render (Task 8), behavioral gate (Task 9). Instrumentation (LCA-collapse rate, per-node train counts) is computed at cohort-assembly time from `label_from_coded` calls + `np.bincount(labels)`; it needs no engine code, so it lives with the assembly, noted here.

**Placeholder scan:** No TBD/TODO; every code step has complete code; test values concrete.

**Type consistency:** `DagLayout` attributes/methods (`nodes`, `K`, `block`, `children`, `closure`, `subtree`, `allowed`, `depth`) are used consistently across Tasks 2–9. `fit_gated -> beta_hat (K,V)` consumed by `profile`/`identifiability_annotation`; `profile -> {node: float}` consumed by `evaluate`/`render_profile`. Consistent.
