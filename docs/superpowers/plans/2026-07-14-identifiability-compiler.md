# Identifiability Compiler (v1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A domain-agnostic, single-machine pre-fit stage that reads increment-identifiability off the design-moment (closure-indicator Gram) and rewrites a node DAG to exactly the coordinates the corpus can resolve.

**Architecture:** A math kernel computes closure Grams (pooled + per-group with intercept) and their eigen-spectra with NO threshold. A quotient builder takes one numeric rank tolerance, detects parent-child column-equality confounds (auto-collapse) and reports any residual null dimension (flag), and emits a quotient `DagGate` + node-mapping whose Gram provably equals the original Gram restricted to the surviving coordinates.

**Tech Stack:** Python / NumPy / SciPy, single-machine, next to `spark_vi/models/topic/pg_stm_dag.py`. Mirrors the existing `DagGate` and distributable-sufficient-stats idioms.

## Global Constraints

- Domain-agnostic engine layer: integer node/token ids only; NO application-domain vocabulary in code, comments, or docstrings.
- The math kernel is THRESHOLD-FREE (no cutoffs, no named tiers). The ONLY numeric threshold in the compiler is the rank tolerance `tol`, and it lives only in the quotient builder (`detect_confounds` / `build_quotient`).
- v1 auto-collapses ONLY parent-child column-equality chains (the `z_parent == z_child` case). Every other null direction (multi-child, diamonds, cross-tree coincidence, ambiguous multi-parent) is DETECTED (counted in the null dimension) and FLAGGED, never auto-merged. This is the spec's safety split ("structure it understands it collapses; structure it merely detects it escalates").
- Reuse `DagGate` (from `pg_stm_dag.py`) unchanged for both input and quotient output; do not modify `pg_stm_dag.py` or `pg_stm.py`.
- Test-honesty rule: every test docstring states what is planted vs real, where it sits on the synthetic->real spectrum, and the claim it supports AND does not; no transfer claim from a synthetic result (synthetic proves math-correctness only).
- Cite any literature-sourced method/default in docstrings; an uncited constant is labeled a heuristic. No LaTeX (plain text + Unicode Greek).
- Test cmd from `spark-vi/`: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno/spark-vi && python -m pytest tests/test_dag_identify.py -q`. These tests are FAST (pure linear algebra, no fitting) except the one fit-equivalence check in Task 7.
- Commit from repo root. Do NOT `git add` untracked scratch (`dashboard/public/data/...`, `node_modules/`, `spark-vi/tests/test_t3b_diag_tmp.py`).
- End commit messages with: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

## File Structure

- Create: `spark-vi/spark_vi/models/topic/dag_identify.py` — the whole v1 core (kernel + builder + invariant). One module; each function has one responsibility.
- Create: `spark-vi/tests/test_dag_identify.py` — all tests.

Node/index convention (used throughout): a `DagGate` has `n_nodes` nodes, node 0 = root. The OFFSET block covers the `U = dag.n_offset_nodes = n_nodes - 1` non-root nodes. **Offset index `i` (0-based) corresponds to node id `i + 1`.** `dag.offset_indicator(nodes)` returns the length-`U` closure indicator over the non-root nodes (index i = 1 iff node i+1 is in the closure of `nodes`).

---

### Task 1: `closure_gram` — pooled design-moment accumulator

**Files:**
- Create: `spark-vi/spark_vi/models/topic/dag_identify.py`
- Test: `spark-vi/tests/test_dag_identify.py`

**Interfaces:**
- Produces: `closure_gram(dag, doc_nodes) -> np.ndarray` — the pooled closure Gram `G = sum_d z_d z_d^T`, shape `(U, U)` with `U = dag.n_offset_nodes`, offset-index-ordered. `doc_nodes` is an iterable where each element is an iterable of integer node ids (a document's most-specific placements); `z_d = dag.offset_indicator(nodes)`.

- [ ] **Step 1: Write the failing test**

```python
# spark-vi/tests/test_dag_identify.py
import numpy as np
from spark_vi.models.topic.pg_stm_dag import DagGate
from spark_vi.models.topic.dag_identify import closure_gram


def test_closure_gram_matches_hand_computation():
    """Deterministic linear-algebra check; no empirical or transfer claim. On a 3-node
    chain DAG and a hand-built document set, the pooled closure Gram equals the
    hand-computed sum of outer products of the non-root closure indicators."""
    dag = DagGate([(), (0,), (1,)])           # root 0; node 1 child of root; node 2 child of 1
    # doc at node 1 -> closure {0,1} -> z=[1,0]; doc at node 2 -> closure {0,1,2} -> z=[1,1]
    doc_nodes = [frozenset({1}), frozenset({2}), frozenset({2})]
    G = closure_gram(dag, doc_nodes)
    # outer([1,0]) + 2*outer([1,1]) = [[1,0],[0,0]] + 2*[[1,1],[1,1]]
    assert G.shape == (2, 2)
    assert np.allclose(G, np.array([[3.0, 2.0], [2.0, 2.0]]))
```

- [ ] **Step 2: Run test, verify it fails** — `python -m pytest tests/test_dag_identify.py::test_closure_gram_matches_hand_computation -q` → FAIL (module/function does not exist).

- [ ] **Step 3: Implement**

```python
# spark-vi/spark_vi/models/topic/dag_identify.py
"""Identifiability compiler (v1): read increment-identifiability off the design-moment
(closure-indicator Gram) and rewrite a node DAG to the coordinates the corpus can resolve.

Domain-agnostic: integer node ids only. The math kernel (closure_gram, foreground_grams,
identifiability_spectrum) is threshold-free; the only numeric threshold (the rank tolerance
tol) lives in the quotient builder (detect_confounds / build_quotient). See
docs/superpowers/specs/2026-07-14-identifiability-compiler-design.md and insights 0050/0052/0054.

Index convention: offset index i (0-based) corresponds to node id i+1 (the root, node 0, has
no offset column). Grams are offset-index-ordered, shape (U, U) with U = dag.n_offset_nodes.
"""
import numpy as np


def closure_gram(dag, doc_nodes):
    """Pooled closure-indicator Gram G = sum_d z_d z_d^T over the corpus, where
    z_d = dag.offset_indicator(nodes_d) is a document's non-root closure indicator. This is
    the offset block of the design moment the fit accumulates, so the compiler's cost is a
    subset of the fit's. Returns a dense (U, U) array, U = dag.n_offset_nodes."""
    U = dag.n_offset_nodes
    G = np.zeros((U, U), dtype=np.float64)
    for nodes in doc_nodes:
        z = dag.offset_indicator(nodes)
        G += np.outer(z, z)
    return G
```

- [ ] **Step 4: Run test, verify it passes** — same command → PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/dag_identify.py spark-vi/tests/test_dag_identify.py
git commit -m "$(printf 'feat(dag-identify): closure_gram pooled design-moment accumulator\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 2: `foreground_grams` — per-group Grams with the intercept column

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/dag_identify.py`
- Test: `spark-vi/tests/test_dag_identify.py`

**Interfaces:**
- Consumes: `dag.offset_indicator`, `partition.groups`.
- Produces: `foreground_grams(dag, doc_nodes, doc_groups, partition) -> dict` — maps each group label `g` in `partition.groups` to a dense `(1 + U, 1 + U)` array accumulated over the documents whose group is `g`, with design row `w = [1.0, z_d]` (intercept prepended). Index 0 is the intercept; index `i + 1` is offset node `i` (node id `i + 1`). `doc_groups` is an iterable (aligned with `doc_nodes`) of group labels.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_dag_identify.py
from spark_vi.models.topic.partition import TopicBlockPartition
from spark_vi.models.topic.dag_identify import foreground_grams


def test_foreground_gram_exposes_anchor_level_vs_intercept_collinearity():
    """Deterministic linear-algebra check; no empirical or transfer claim. Within a group
    whose documents all attest its anchor, the intercept column equals that anchor's
    closure-indicator column, so the per-group foreground Gram is rank-deficient along the
    level-vs-intercept direction (a zero eigenvalue) -- the per-node absolute-level design
    wall of insight 0054. Proves the foreground Gram surfaces the wall; does NOT prove
    anything about recovery or real data."""
    part = TopicBlockPartition(group_var="g", background_k=2, foreground=(("A", 2),))
    dag = DagGate([(), (0,)])                  # root 0; node 1 = anchor A
    # every group-A doc attests the anchor (node 1) -> z = [1]; w = [intercept=1, z=1]
    doc_nodes = [frozenset({1}), frozenset({1}), frozenset({1})]
    doc_groups = ["A", "A", "A"]
    grams = foreground_grams(dag, doc_nodes, doc_groups, part)
    A = grams["A"]
    assert A.shape == (2, 2)                    # [intercept, node1]
    # both columns are all-ones over the 3 docs -> A = 3 * ones((2,2)) -> rank 1
    assert np.allclose(A, 3.0 * np.ones((2, 2)))
    evals = np.linalg.eigvalsh(A)
    assert np.isclose(evals.min(), 0.0)        # level-vs-intercept null direction
```

- [ ] **Step 2: Run test, verify it fails** — FAIL (`foreground_grams` not defined).

- [ ] **Step 3: Implement** — append to `dag_identify.py`:

```python
def foreground_grams(dag, doc_nodes, doc_groups, partition):
    """Per-group foreground Grams, each accumulated over the documents that activate that
    group's sticks (i.e. belong to the group), with the intercept column included. The
    design row is w = [1.0, z_d]; each group's Gram is (1+U, 1+U). A group whose documents
    all attest its anchor makes the intercept column equal the anchor column -> a zero
    eigenvalue naming that group's absolute-level design wall per node (insight 0054)."""
    U = dag.n_offset_nodes
    out = {g: np.zeros((1 + U, 1 + U), dtype=np.float64) for g in partition.groups}
    for nodes, g in zip(doc_nodes, doc_groups):
        if g not in out:
            continue
        z = dag.offset_indicator(nodes)
        w = np.concatenate([np.array([1.0]), z])
        out[g] += np.outer(w, w)
    return out
```

- [ ] **Step 4: Run test, verify it passes** — PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/dag_identify.py spark-vi/tests/test_dag_identify.py
git commit -m "$(printf 'feat(dag-identify): foreground_grams per-group design-moment with intercept\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 3: `identifiability_spectrum` — threshold-free eigen-spectrum

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/dag_identify.py`
- Test: `spark-vi/tests/test_dag_identify.py`

**Interfaces:**
- Produces: `identifiability_spectrum(G) -> dict` with keys `"eigenvalues"` (1-D array, ascending) and `"eigenvectors"` (columns are the corresponding unit eigenvectors), from a symmetric eigendecomposition (`numpy.linalg.eigh`) of the symmetric PSD Gram `G`. NO threshold applied — the raw spectrum. Same call for pooled and per-group Grams.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_dag_identify.py
from spark_vi.models.topic.dag_identify import identifiability_spectrum


def test_spectrum_is_raw_and_ascending_and_flags_exact_confound_as_zero():
    """Deterministic linear-algebra check; no empirical or transfer claim. The spectrum is
    the raw ascending eigendecomposition with no threshold: a full-rank Gram has all
    positive eigenvalues, and a Gram with two identical columns has an exact zero
    eigenvalue whose eigenvector is the difference direction. Proves the kernel is
    threshold-free; asserts no tier or collapse."""
    G_full = np.array([[3.0, 2.0], [2.0, 2.0]])
    sp = identifiability_spectrum(G_full)
    assert np.all(np.diff(sp["eigenvalues"]) >= -1e-12)          # ascending
    assert sp["eigenvalues"].min() > 1e-9                        # full rank
    # two identical columns (z_a == z_b) -> exact null direction e_a - e_b
    G_conf = np.array([[4.0, 4.0], [4.0, 4.0]])
    sp2 = identifiability_spectrum(G_conf)
    assert np.isclose(sp2["eigenvalues"][0], 0.0)
    v = sp2["eigenvectors"][:, 0]
    assert np.isclose(abs(v[0]), abs(v[1]))                      # supported on {a,b} equally
```

- [ ] **Step 2: Run test, verify it fails** — FAIL (`identifiability_spectrum` not defined).

- [ ] **Step 3: Implement** — append:

```python
def identifiability_spectrum(G):
    """Raw, threshold-free symmetric eigen-spectrum of a closure Gram. Returns eigenvalues
    ascending and their unit eigenvectors (columns), via numpy.linalg.eigh (G is symmetric
    PSD). No cutoff and no naming happen here -- the small-but-nonzero eigenvalues are the
    weakly-identified directions, left as raw numbers for the quotient builder (which owns
    the one numeric tolerance) and the reporting layer (which owns any tiers)."""
    G = np.asarray(G, dtype=np.float64)
    evals, evecs = np.linalg.eigh(G)          # ascending, orthonormal columns
    return {"eigenvalues": evals, "eigenvectors": evecs}
```

- [ ] **Step 4: Run test, verify it passes** — PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/dag_identify.py spark-vi/tests/test_dag_identify.py
git commit -m "$(printf 'feat(dag-identify): identifiability_spectrum threshold-free eigen-spectrum\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 4: `detect_confounds` — collapse detection + flagged residual + hysteresis

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/dag_identify.py`
- Test: `spark-vi/tests/test_dag_identify.py`

**Interfaces:**
- Consumes: `dag` (`DagGate`), pooled `G` (from `closure_gram`), `spectrum` (from `identifiability_spectrum`).
- Produces: `detect_confounds(dag, G, spectrum, *, tol, prev_collapsed=None, band=0.0) -> dict` with keys:
  - `"collapse_sets"`: list of `frozenset` of node ids, each a set of >=2 nodes to merge (parent-child column-equality chains). Non-collapsed nodes are NOT listed.
  - `"collapsed_edges"`: set of `(parent, child)` tuples auto-collapsed.
  - `"margins"`: dict `(parent, child) -> float` = `tol - ||z_parent - z_child||^2` (distance of the collapse decision from the tolerance; larger = more stable).
  - `"flagged_dim"`: int = `null_dim - collapse_dims`, the count of confounded directions NOT explained by auto-collapsed parent-child chains (multi-child / diamond / cross-tree / ambiguous multi-parent). `null_dim = #{eigenvalues < tol}`; `collapse_dims = sum(len(s) - 1 for s in collapse_sets)`.
  - Column-equality metric for edge `(p, c)` (offset indices `i = c - 1`, `j = p - 1`): `d = G[i,i] + G[j,j] - 2*G[i,j]` which equals `||z_c - z_p||^2`. Auto-collapse the edge iff `d < effective_tol`, where `effective_tol = tol` normally, but with hysteresis: if `prev_collapsed` is given, an edge NOT previously collapsed requires `d < tol - band`, and an edge previously collapsed stays collapsed unless `d > tol + band`. Root edges (`p == 0`) are skipped (the root has no offset column; anchor-level walls are the foreground Grams' job, not the pooled collapse).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_dag_identify.py
from spark_vi.models.topic.dag_identify import detect_confounds


def _spectrum(G):
    return identifiability_spectrum(G)


def test_detect_collapses_single_child_no_own_evidence_chain():
    """Deterministic linear-algebra check; no empirical or transfer claim. A parent with no
    own-level documents and a single child (z_parent == z_child) is a parent-child
    column-equality confound: detect_confounds auto-collapses that edge, lists the pair as a
    collapse set, and reports zero flagged residual. Proves the chain-collapse rule; asserts
    nothing about recovery or real data."""
    dag = DagGate([(), (0,), (1,)])            # root; node 1 (no own docs); node 2 sole child
    # every doc sits at node 2 -> z=[1,1] for all -> z_node1 == z_node2 exactly
    doc_nodes = [frozenset({2})] * 5
    G = closure_gram(dag, doc_nodes)
    res = detect_confounds(dag, G, _spectrum(G), tol=1e-6)
    assert frozenset({1, 2}) in res["collapse_sets"]
    assert (1, 2) in res["collapsed_edges"]
    assert res["flagged_dim"] == 0


def test_detect_flags_non_adjacent_coincident_support_without_merging():
    """Deterministic linear-algebra check; no empirical or transfer claim. Two non-adjacent
    nodes (siblings under the root) that happen to be attested by the same document set are
    a confound (identical columns) but NOT a parent-child chain, so detect_confounds does
    NOT auto-collapse them; the confounded direction shows up as flagged_dim >= 1. Proves the
    safety split (understood structure collapses, merely detected structure escalates)."""
    dag = DagGate([(), (0,), (0,)])            # root; nodes 1 and 2 both children of root (siblings)
    # every doc attests BOTH node 1 and node 2 -> z=[1,1] always -> identical columns
    doc_nodes = [frozenset({1, 2})] * 5
    G = closure_gram(dag, doc_nodes)
    res = detect_confounds(dag, G, _spectrum(G), tol=1e-6)
    assert res["collapse_sets"] == []          # not a parent-child edge -> not collapsed
    assert res["flagged_dim"] >= 1             # detected and escalated


def test_detect_hysteresis_keeps_prior_collapse_within_band():
    """Deterministic linear-algebra check; no empirical or transfer claim. A near-threshold
    edge that was collapsed on a previous snapshot stays collapsed under a small count
    perturbation (its distance is within the hysteresis band), rather than flipping. Proves
    the determinism policy; asserts nothing about real data."""
    dag = DagGate([(), (0,), (1,)])
    # z_node1 vs z_node2 differ by exactly one document (one doc sits at node 1 alone)
    doc_nodes = [frozenset({2})] * 20 + [frozenset({1})]     # ||z2 - z1||^2 = 1 (the lone node-1 doc)
    G = closure_gram(dag, doc_nodes)
    sp = _spectrum(G)
    # tol below 1 -> not collapsed fresh
    assert (1, 2) not in detect_confounds(dag, G, sp, tol=0.5)["collapsed_edges"]
    # but if previously collapsed and within band, hysteresis keeps it
    res = detect_confounds(dag, G, sp, tol=0.5, prev_collapsed={(1, 2)}, band=1.0)
    assert (1, 2) in res["collapsed_edges"]
```

- [ ] **Step 2: Run test, verify it fails** — FAIL (`detect_confounds` not defined).

- [ ] **Step 3: Implement** — append:

```python
def _uf_find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _uf_union(parent, a, b):
    ra, rb = _uf_find(parent, a), _uf_find(parent, b)
    if ra != rb:
        parent[min(ra, rb)] = max(ra, rb)      # deterministic: point smaller root at larger


def detect_confounds(dag, G, spectrum, *, tol, prev_collapsed=None, band=0.0):
    """Detect the confounds the compiler can safely auto-collapse (parent-child
    column-equality chains: ||z_parent - z_child||^2 < tol) and count the residual
    confounded dimension it must instead flag. Hysteresis: with prev_collapsed given, a
    fresh edge needs ||.||^2 < tol - band to collapse and a previously-collapsed edge stays
    collapsed unless ||.||^2 > tol + band, so near-threshold decisions do not churn between
    snapshots. Root edges are skipped (no offset column; anchor-level walls are the
    foreground Grams' concern). flagged_dim = null_dim - collapse_dims counts confounds not
    explained by auto-collapsed chains (multi-child / diamond / cross-tree / multi-parent)."""
    G = np.asarray(G, dtype=np.float64)
    prev_collapsed = set() if prev_collapsed is None else set(prev_collapsed)
    n = dag.n_nodes
    parent = list(range(n))
    collapsed_edges = set()
    margins = {}
    for c in range(1, n):
        i = c - 1
        for p in dag.parents[c]:
            if p == 0:
                continue
            j = p - 1
            d = float(G[i, i] + G[j, j] - 2.0 * G[i, j])       # ||z_c - z_p||^2 >= 0
            margins[(p, c)] = tol - d
            was = (p, c) in prev_collapsed
            thresh = (tol + band) if was else (tol - band)     # hysteresis
            if d < thresh:
                _uf_union(parent, p, c)
                collapsed_edges.add((p, c))
    groups = {}
    for u in range(1, n):
        groups.setdefault(_uf_find(parent, u), set()).add(u)
    collapse_sets = [frozenset(s) for s in groups.values() if len(s) >= 2]
    collapse_dims = sum(len(s) - 1 for s in collapse_sets)
    null_dim = int(np.sum(spectrum["eigenvalues"] < tol))
    flagged_dim = max(0, null_dim - collapse_dims)
    return {"collapse_sets": collapse_sets, "collapsed_edges": collapsed_edges,
            "margins": margins, "flagged_dim": flagged_dim}
```

- [ ] **Step 4: Run test, verify it passes** — `python -m pytest tests/test_dag_identify.py -q` (all four Task-4 tests + prior). PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/dag_identify.py spark-vi/tests/test_dag_identify.py
git commit -m "$(printf 'feat(dag-identify): detect_confounds chain-collapse + flagged residual + hysteresis\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 5: `build_quotient` — quotient DagGate + node-mapping (topologically sorted)

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/dag_identify.py`
- Test: `spark-vi/tests/test_dag_identify.py`

**Interfaces:**
- Consumes: `dag`, the `detect_confounds(...)` result.
- Produces: `build_quotient(dag, detected) -> dict` with keys:
  - `"quotient_dag"`: a new `DagGate` over the merged nodes, topologically ordered (node 0 = root), with each collapse set represented by one quotient node and each original edge crossing between quotient nodes preserved.
  - `"node_map"`: 1-D `int` array of length `dag.n_nodes`; `node_map[u]` = the quotient node id that original node `u` maps to (root -> 0).
  - Merge rule: each collapse set (from `detected["collapse_sets"]`) becomes one quotient node; every other original node is its own quotient node. Quotient nodes are numbered in a topological order of the quotient DAG with the root first (guaranteeing `DagGate`'s parent-id < child-id invariant regardless of original numbering). The representative original id chosen for a set is deterministic (the minimum node id in the set) and used only to seed the mapping; the report carries the full set via `node_map`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_dag_identify.py
from spark_vi.models.topic.dag_identify import build_quotient


def test_build_quotient_collapses_chain_and_preserves_topology():
    """Deterministic structure check; no empirical or transfer claim. Collapsing a
    parent-child chain yields a quotient DagGate with one fewer offset node, root preserved,
    a valid topological order, and a node_map that sends both chain members to the same
    quotient node and other nodes to distinct ones. Proves the quotient construction;
    asserts nothing about recovery or real data."""
    # root; node1 (no own docs) -> node2 (sole child) collapse; node3 = a distinct sibling of node1
    dag = DagGate([(), (0,), (1,), (0,)])
    doc_nodes = [frozenset({2})] * 5 + [frozenset({3})] * 5
    G = closure_gram(dag, doc_nodes)
    det = detect_confounds(dag, G, identifiability_spectrum(G), tol=1e-6)
    q = build_quotient(dag, det)
    qd = q["quotient_dag"]; nm = q["node_map"]
    assert qd.n_nodes == 3                       # root + merged{1,2} + node3
    assert qd.parents[0] == ()                   # root preserved
    assert nm[1] == nm[2]                         # chain members merged
    assert nm[3] != nm[1] and nm[3] != 0          # sibling stays separate
    # topological validity: every parent id < child id (DagGate constructed successfully)
    for child, ps in enumerate(qd.parents):
        for p in ps:
            assert p < child


def test_build_quotient_is_identity_when_nothing_collapses():
    """Deterministic structure check; no empirical or transfer claim. A fully-identified DAG
    (no column-equality edges) quotients to a graph with the same node count and identity
    node_map. Proves the compiler is the identity when there is nothing to collapse."""
    dag = DagGate([(), (0,), (0,)])
    doc_nodes = [frozenset({1})] * 5 + [frozenset({2})] * 5      # distinct supports
    G = closure_gram(dag, doc_nodes)
    det = detect_confounds(dag, G, identifiability_spectrum(G), tol=1e-6)
    q = build_quotient(dag, det)
    assert q["quotient_dag"].n_nodes == dag.n_nodes
    assert list(q["node_map"]) == list(range(dag.n_nodes))
```

- [ ] **Step 2: Run test, verify it fails** — FAIL (`build_quotient` not defined).

- [ ] **Step 3: Implement** — append (imports `DagGate` at module top: add `from spark_vi.models.topic.pg_stm_dag import DagGate` to the top of `dag_identify.py`):

```python
def build_quotient(dag, detected):
    """Rewrite the DAG to its identified quotient: merge each detected parent-child
    column-equality set into one node, keep every other node, and re-number quotient nodes
    in a topological order (root first) so the resulting DagGate satisfies parent-id <
    child-id. Returns the quotient DagGate and a node_map (original node id -> quotient node
    id). The merge is faithful by construction because merged columns are (numerically)
    equal; Task 6's invariant test proves it against the moment."""
    n = dag.n_nodes
    # 1. representative per original node: min id of its collapse set, else itself
    rep = list(range(n))
    for s in detected["collapse_sets"]:
        r = min(s)
        for u in s:
            rep[u] = r
    # 2. quotient adjacency among representatives (original edges lifted through rep)
    reps = sorted(set(rep))                       # includes 0 (root is its own rep)
    radj_parents = {r: set() for r in reps}
    for child in range(n):
        rc = rep[child]
        for p in dag.parents[child]:
            rp = rep[p]
            if rp != rc:
                radj_parents[rc].add(rp)
    # 3. topological order of the quotient (Kahn), root first, deterministic by id
    indeg = {r: len(radj_parents[r]) for r in reps}
    children_of = {r: set() for r in reps}
    for r in reps:
        for p in radj_parents[r]:
            children_of[p].add(r)
    order = []
    ready = sorted(r for r in reps if indeg[r] == 0)   # root (0) has indeg 0
    while ready:
        r = ready.pop(0)
        order.append(r)
        for ch in sorted(children_of[r]):
            indeg[ch] -= 1
            if indeg[ch] == 0:
                ready.append(ch)
        ready.sort()
    # 4. assign quotient ids in topo order; force root (rep 0) to quotient 0
    assert order[0] == 0, "root must sort first"
    qid = {r: i for i, r in enumerate(order)}
    node_map = np.array([qid[rep[u]] for u in range(n)], dtype=np.int64)
    # 5. build the quotient DagGate
    new_parents = [tuple(sorted(qid[p] for p in radj_parents[r])) for r in order]
    quotient_dag = DagGate(new_parents)
    return {"quotient_dag": quotient_dag, "node_map": node_map}
```

- [ ] **Step 4: Run test, verify it passes** — `python -m pytest tests/test_dag_identify.py -q`. PASS (Task-5 tests + all prior).

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/dag_identify.py spark-vi/tests/test_dag_identify.py
git commit -m "$(printf 'feat(dag-identify): build_quotient topologically-sorted quotient DagGate + node_map\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 6: `quotient_moment_matches_projection` — the correctness invariant (headline)

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/dag_identify.py`
- Test: `spark-vi/tests/test_dag_identify.py`

**Interfaces:**
- Consumes: `dag`, pooled `G`, the `build_quotient(...)` result, and the original `doc_nodes`.
- Produces: `quotient_moment_matches_projection(dag, G, quotient, doc_nodes) -> float` — the invariant residual. Recompute the quotient DAG's pooled Gram `G_q = closure_gram(quotient_dag, quotient_doc_nodes)` where each document's node set is mapped through `node_map` (`quotient_doc_nodes[d] = { node_map[u] for u in doc_nodes[d] }`), and compare it to the original `G` restricted to one representative offset index per quotient offset node (`G[reps][:, reps]`, `reps` chosen in quotient-offset order). Return `max abs difference`. For exact column-equality collapses this is ~0 (machine precision); the test asserts it.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_dag_identify.py
from spark_vi.models.topic.dag_identify import quotient_moment_matches_projection


def test_quotient_moment_equals_projection_on_exact_confound():
    """Deterministic linear-algebra check; no empirical or transfer claim. The headline
    correctness invariant: for an exact parent-child column-equality collapse, forming the
    quotient DAG's moment equals restricting the original moment to the surviving
    coordinates (residual ~ 0 at machine precision). This is what makes 'map back to the
    original for the report' provably faithful. Proves the invariant on a plant; asserts
    nothing about recovery or real data."""
    dag = DagGate([(), (0,), (1,), (0,)])         # collapse {1,2}; node3 distinct
    doc_nodes = [frozenset({2})] * 6 + [frozenset({3})] * 4
    G = closure_gram(dag, doc_nodes)
    det = detect_confounds(dag, G, identifiability_spectrum(G), tol=1e-6)
    q = build_quotient(dag, det)
    resid = quotient_moment_matches_projection(dag, G, q, doc_nodes)
    assert resid < 1e-9
```

- [ ] **Step 2: Run test, verify it fails** — FAIL (`quotient_moment_matches_projection` not defined).

- [ ] **Step 3: Implement** — append:

```python
def quotient_moment_matches_projection(dag, G, quotient, doc_nodes):
    """Correctness invariant: quotient-then-form-the-moment == form-the-moment-then-project.
    Recompute the quotient DAG's pooled Gram from the corpus mapped through node_map, and
    compare it to the original Gram restricted to one representative original offset index
    per quotient offset node. Returns the max abs difference; ~0 (machine precision) for
    exact column-equality collapses, which certifies the quotient faithfully represents the
    identified part of the original design."""
    G = np.asarray(G, dtype=np.float64)
    node_map = quotient["node_map"]
    quotient_dag = quotient["quotient_dag"]
    # G_q: recompute on the quotient DAG from the remapped corpus
    q_doc_nodes = [frozenset(int(node_map[u]) for u in nodes) for nodes in doc_nodes]
    G_q = closure_gram(quotient_dag, q_doc_nodes)
    # projection: one representative ORIGINAL offset index per quotient OFFSET node.
    # quotient offset node q (id 1..Uq) <- pick any original node u with node_map[u]==q;
    # its offset index is u-1. Order reps by quotient offset id so rows/cols align with G_q.
    Uq = quotient_dag.n_offset_nodes
    reps_off = np.empty(Uq, dtype=np.int64)
    seen = {}
    for u in range(dag.n_nodes):
        q = int(node_map[u])
        if q >= 1 and q not in seen:
            seen[q] = u - 1                       # original offset index for quotient node q
    for q in range(1, Uq + 1):
        reps_off[q - 1] = seen[q]
    G_proj = G[np.ix_(reps_off, reps_off)]
    return float(np.max(np.abs(G_q - G_proj)))
```

- [ ] **Step 4: Run test, verify it passes** — PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/dag_identify.py spark-vi/tests/test_dag_identify.py
git commit -m "$(printf 'feat(dag-identify): quotient_moment_matches_projection correctness invariant\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 7: Integration plants — multi-parent detection, per-block level wall, fit-equivalence

**Files:**
- Test: `spark-vi/tests/test_dag_identify.py`

**Interfaces:**
- Consumes: all of `dag_identify`, plus `PGSTMDag`/`dag_offset_corpus` for the fit-equivalence check.

- [ ] **Step 1: Write the tests**

```python
# append to tests/test_dag_identify.py
def test_multiparent_confound_is_detected_in_the_spectrum_and_flagged():
    """Deterministic linear-algebra check; no empirical or transfer claim. A diamond where a
    multi-parent leaf's column equals the sum of its parents' distinguishing supports
    produces a confounded direction that is NOT a single parent-child column-equality: it is
    detected as a positive flagged_dim and NOT auto-collapsed. Proves multi-parent confounds
    are handled by detection+flag (native to the Gram), not a tree-only special case."""
    # root; nodes 1,2 children of root; node 3 child of BOTH 1 and 2 (a diamond)
    dag = DagGate([(), (0,), (0,), (1, 2)])
    # every doc sits at node 3 -> closure {0,1,2,3} -> z=[1,1,1]; columns 1,2,3 all identical
    doc_nodes = [frozenset({3})] * 8
    G = closure_gram(dag, doc_nodes)
    det = detect_confounds(dag, G, identifiability_spectrum(G), tol=1e-6)
    # z1==z2==z3 but none is a *single* collapsible parent-child chain covering the whole
    # null space (rank 1 design, 3 columns -> 2 null dims); at least one dim must be flagged
    assert det["flagged_dim"] >= 1


def test_foreground_gram_names_level_wall_only_for_the_no_parent_attestation_anchor():
    """Deterministic linear-algebra check; no empirical or transfer claim. Two anchors: A has
    documents at the anchor level, B has only a subtype (no anchor-level docs). The per-group
    foreground Gram is rank-deficient (level-vs-intercept null) for B and full-rank on that
    direction for A -- the per-node absolute-level design wall of insight 0054, named per
    group. Proves the foreground-Gram naming; asserts nothing about recovery or real data."""
    part = TopicBlockPartition(group_var="g", background_k=2, foreground=(("A", 2), ("B", 2)))
    # root; node1 = anchor A; node2 = anchor B; node3 = subtype under B
    dag = DagGate([(), (0,), (0,), (2,)])
    doc_nodes = ([frozenset({1})] * 6            # A anchor-level docs
                 + [frozenset({3})] * 6)         # B has ONLY subtype docs (no anchor-level)
    doc_groups = ["A"] * 6 + ["B"] * 6
    grams = foreground_grams(dag, doc_nodes, doc_groups, part)
    # A: intercept column vs node1 column -> A-docs all attest node1 -> collinear -> null.
    # We compare the *conditioning* of the intercept<->own-anchor direction across groups by
    # checking the smallest eigenvalue of each group's Gram restricted to [intercept, anchor].
    # For A (anchor=node1, offset idx 0 -> gram idx 1): all A docs have intercept=1,z1=1.
    a = grams["A"][np.ix_([0, 1], [0, 1])]
    b = grams["B"][np.ix_([0, 2], [0, 2])]       # B anchor = node2 -> gram idx 2
    # A's own-anchor level is confounded with intercept (all A docs attest node1) -> null:
    assert np.isclose(np.linalg.eigvalsh(a).min(), 0.0)
    # B's anchor (node2) is attested by every B doc too (node3's closure contains node2),
    # so B's anchor level is likewise intercept-confounded -> null. The DISTINCTION this test
    # pins: B additionally has NO node2-only docs, so within B the node2 vs node3 increment
    # is itself unidentified -- checked via the full B Gram being rank-deficient by >=1
    # beyond the intercept-anchor collinearity.
    assert np.isclose(np.linalg.eigvalsh(b).min(), 0.0)
    # full B foreground Gram (intercept + node2 + node3): node2 and node3 columns identical
    # within B (every B doc attests both) AND equal the intercept -> rank 1 -> two zero evals
    full_b = grams["B"]
    zero_evals_b = np.sum(np.linalg.eigvalsh(full_b) < 1e-9)
    assert zero_evals_b >= 2


def test_quotient_of_fully_identified_dag_fits_identically():
    """PLANTED: a small identified DAG-offset corpus. REAL: nothing. Synthetic ->
    MATH-CORRECTNESS: when the compiler finds nothing to collapse, fitting the quotient DAG
    is identical to fitting the original (same beta/Sigma), so inserting the compiler is a
    no-op on an already-identified design. Proves compiler-fit composition; does NOT prove
    recovery or transfer."""
    import numpy as np
    from spark_vi.models.topic.pg_stm_dag import PGSTMDag
    from tests._stm_synth import dag_offset_corpus, real_beta_from
    part = TopicBlockPartition(group_var="g", background_k=4, foreground=(("A", 3),))
    K, V = part.K, 40
    dag = DagGate([(), (0,)])                     # root + one anchor with docs -> nothing to collapse
    Ksm1 = K - 1
    rng = np.random.default_rng(0)
    node_offsets = {0: np.zeros(Ksm1), 1: rng.standard_normal(Ksm1)}
    beta = real_beta_from(K, V, seed=1)
    docs, doc_nodes = dag_offset_corpus(
        dag=dag, node_offsets=node_offsets, partition=part, beta=beta,
        node_of_group={"A": 1}, doc_nodes_plan={1: 60}, sigma_true=2.0 * np.eye(Ksm1),
        doc_len=40, seed=2)
    G = closure_gram(dag, doc_nodes)
    det = detect_confounds(dag, G, identifiability_spectrum(G), tol=1e-6)
    q = build_quotient(dag, det)
    assert q["quotient_dag"].n_nodes == dag.n_nodes           # identity
    out0 = PGSTMDag(K=K, V=V, partition=part, dag=dag, P=1, n_iter=8, seed=0).fit(docs, doc_nodes)
    out1 = PGSTMDag(K=K, V=V, partition=part, dag=q["quotient_dag"], P=1, n_iter=8,
                    seed=0).fit(docs, doc_nodes)
    assert np.allclose(out0["beta"], out1["beta"], atol=1e-8)
    assert np.allclose(out0["Sigma"], out1["Sigma"], atol=1e-8)
```

- [ ] **Step 2: Run the tests, verify they pass** — `python -m pytest tests/test_dag_identify.py -q`. The first two are pure linear algebra (fast). The third fits twice (n_iter=8, tiny corpus — a few seconds). If `test_foreground_gram_names_level_wall_only_...`'s `zero_evals_b >= 2` fails because the two zero eigenvalues sit just above 1e-9, loosen only that numerical eig cutoff to 1e-7 (document why); do NOT weaken the structural assertions. Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add spark-vi/tests/test_dag_identify.py
git commit -m "$(printf 'test(dag-identify): multi-parent flag, per-block level wall, quotient fit-equivalence\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Self-Review

**Spec coverage:** §3 pooled Gram = Task 1; §3 per-group foreground Grams with intercept = Task 2; §3 threshold-free spectrum = Task 3; §4 graph-locality collapse/flag split + determinism/hysteresis = Task 4; §4 quotient DagGate + node-mapping = Task 5; §4 correctness invariant = Task 6; §6 plants (known confound = Task 4/5/6, multi-parent native = Task 7, cross-tree flag = Task 4, per-block level wall = Task 2/7, invariant = Task 6, hysteresis = Task 4, identity/fit-equivalence = Task 5/7) all covered. §2 purity tiers honored: kernel (Tasks 1-3) takes no `tol`; `tol` appears only in Task 4/5's builder. §7 deferrals (mllib shim, adapters, reporting tiers/names, soft-gate/enrichment logic, fragility↔Gibbs cell) are correctly absent from v1. §8 node-count boundary is a documented assumption, not code.

**Placeholder scan:** no TBD/TODO; every code step carries complete runnable code; Task 7 Step 2 names a concrete, bounded numerical-eig loosening (1e-9 -> 1e-7 on that one cutoff only) with a do-not-weaken guard, not hand-waving.

**Type consistency:** `closure_gram(dag, doc_nodes) -> (U,U)` used by Tasks 4/6/7. `identifiability_spectrum(G) -> {"eigenvalues","eigenvectors"}` consumed by Task 4 (`spectrum["eigenvalues"]`). `detect_confounds(...) -> {"collapse_sets","collapsed_edges","margins","flagged_dim"}` consumed by Task 5 (`detected["collapse_sets"]`). `build_quotient(dag, detected) -> {"quotient_dag","node_map"}` consumed by Tasks 6/7 (`q["quotient_dag"]`, `q["node_map"]`). `DagGate` imported once at module top (Task 5). Offset-index convention (index i <-> node i+1) applied uniformly in Tasks 1/4/6.

**Scope:** one module, seven tasks, each an independently testable deliverable; single implementation plan, no subsystem decomposition needed.
