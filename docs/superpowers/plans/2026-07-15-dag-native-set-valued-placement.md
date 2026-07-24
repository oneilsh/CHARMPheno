# DAG-Native + Set-Valued Frontier Truth — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the shipped hierarchical-placement engine so `DagLayout` is multi-parent DAG-native and held-out truth is the set-valued frontier of attested nodes (drop attested ancestors, keep incomparable), with comorbid-aware gating and set-valued evaluation + instrumentation.

**Architecture:** Modify the existing `spark-vi/spark_vi/models/topic/dag_placement.py` in place. `DagLayout` accepts `{child: parent | [parents]}`; `closure` = all ancestors (depth-sorted list), `depth` = longest path, add `allowed_set(frontier)`. Add `frontier_from_coded`. `fit_gated`/`evaluate` accept per-doc frontier sets (scalars treated as singletons). `identifiability_annotation`/`render_profile` generalize to multi-parent. Extend the synthetic generator to emit multi-parent DAGs + comorbid patients. Every change is backward compatible: the 14 existing tests must stay green.

**Tech Stack:** Python, NumPy, pytest. Same module as the shipped engine.

## Global Constraints

- Engine is **domain-agnostic**: integer token/label/node ids only. No clinical vocabulary in the module or its tests.
- No LaTeX in comments/docstrings; plain text + Unicode where needed.
- Commit trailer, exactly on its own line: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Run tests with: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_dag_placement.py -v`.
- **Backward compatibility is a hard requirement:** the 14 shipped tests must stay green after every task. Single-parent `{child: parent}` maps and scalar labels must keep working (parents normalized to one-element lists; scalar labels treated as singleton frontiers).
- Test honesty: never loosen a floor or weaken an assertion to force a pass; a genuine failure is investigated, not masked.
- Spec: `docs/superpowers/specs/2026-07-15-dag-native-set-valued-placement-design.md`.

---

## File Structure

- Modify `spark-vi/spark_vi/models/topic/dag_placement.py` — `DagLayout` (multi-parent), add `frontier_from_coded`, `allowed_set`; generalize `fit_gated`, `evaluate` (+ `_hops`), `identifiability_annotation`, `render_profile`.
- Modify `spark-vi/tests/test_dag_placement.py` — add multi-parent + set-valued tests; keep all existing tests.
- Modify `spark-vi/tests/_stm_synth.py` — add `dag_placement_corpus_multi` (multi-parent + comorbid, frozenset labels).

Shared test fixture used by new tests (add near the top of the test file if not already present; do NOT remove the existing `PARENT`):

```python
DIAMOND = {1: 0, 2: 0, 3: 0, 4: [1, 2], 5: [1, 3]}   # 4,5 are multi-parent (two axes)
```

---

### Task 1: `DagLayout` goes multi-parent DAG-native

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/dag_placement.py` (replace the `DagLayout` class)
- Test: `spark-vi/tests/test_dag_placement.py`

**Interfaces:**
- Produces: `DagLayout(parent, n_bg=2, tpn=1)` where `parent` values are `int` or `list[int]`. Attributes `parents: {node:[parent ids]}`, `nodes`, `n_bg`, `tpn`, `children`, `block`, `K`. Methods `closure(v)->list` (all ancestors + v, sorted by (depth,id)), `subtree(u)->set`, `allowed(v)->np.ndarray`, `allowed_set(frontier)->np.ndarray` (bg ∪ blocks over union of closures), `depth(v)->int` (longest path to root).
- Consumes: nothing new.

- [ ] **Step 1: Write the failing test** (append)

```python
def test_daglayout_multiparent_diamond():
    lay = DagLayout(DIAMOND, n_bg=2, tpn=1)
    assert lay.parents[4] == [1, 2] and lay.parents[5] == [1, 3]
    assert lay.closure(4) == [0, 1, 2, 4]          # all ancestors, depth-sorted, root first
    assert lay.closure(5) == [0, 1, 3, 5]
    assert lay.depth(4) == 2 and lay.depth(1) == 1 and lay.depth(0) == 0   # longest path
    assert lay.subtree(1) == {1, 4, 5} and lay.subtree(2) == {2, 4}
    want = {0, 1} | set(lay.block[1]) | set(lay.block[2]) | set(lay.block[3]) \
        | set(lay.block[4]) | set(lay.block[5])
    assert set(lay.allowed_set({4, 5}).tolist()) == want   # union of closures over the frontier

def test_daglayout_singleparent_backward_compat():
    lay = DagLayout(PARENT, n_bg=2, tpn=1)               # scalar-parent map still works
    assert lay.closure(3) == [0, 1, 3]                   # exact old list ordering
    assert list(lay.allowed(3)) == [0, 1] + lay.block[1] + lay.block[3]
    assert list(lay.allowed(1)) == [0, 1] + lay.block[1]
    assert lay.depth(3) == 2 and lay.depth(1) == 1
    assert lay.subtree(1) == {1, 3, 4}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_dag_placement.py::test_daglayout_multiparent_diamond -v`
Expected: FAIL (`parents` attribute / `allowed_set` missing).

- [ ] **Step 3: Replace the `DagLayout` class** with the multi-parent version

```python
class DagLayout:
    """Topic-block layout over a label DAG: `n_bg` shared background topics, then `tpn` topics per
    non-root node. `parent` maps child id -> parent id OR list of parent ids (multi-parent DAG); the
    root is id 0 (no entry). A scalar parent is normalized to a one-element list, so single-parent
    tree maps keep working unchanged."""

    def __init__(self, parent, n_bg=2, tpn=1):
        self.parents = {c: (list(p) if isinstance(p, (list, tuple, set)) else [p])
                        for c, p in parent.items()}
        self.nodes = sorted(self.parents.keys())
        self.n_bg = int(n_bg)
        self.tpn = int(tpn)
        self.children = {0: []}
        for c, ps in self.parents.items():
            self.children.setdefault(c, [])
            for p in ps:
                self.children.setdefault(p, []).append(c)
        for p in self.children:
            self.children[p] = sorted(self.children[p])
        self.block = {u: list(range(n_bg + i * tpn, n_bg + (i + 1) * tpn))
                      for i, u in enumerate(self.nodes)}
        self.K = n_bg + len(self.nodes) * tpn
        self._depth = {}

    def depth(self, v):
        """Longest path length from root to v (root = 0). Memoized."""
        if v in self._depth:
            return self._depth[v]
        ps = self.parents.get(v, [])
        d = 0 if not ps else 1 + max(self.depth(p) for p in ps)
        self._depth[v] = d
        return d

    def closure(self, v):
        """All ancestors of v plus v, as a list sorted by (depth, id) so root comes first. For a
        single-parent tree this reproduces the old root..v ordering exactly."""
        seen = set()
        stack = [v]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            for p in self.parents.get(x, []):
                stack.append(p)
        return sorted(seen, key=lambda u: (self.depth(u), u))

    def subtree(self, u):
        out = {u}
        stack = [u]
        while stack:
            x = stack.pop()
            for ch in self.children.get(x, []):
                if ch not in out:
                    out.add(ch)
                    stack.append(ch)
        return out

    def allowed(self, v):
        al = list(range(self.n_bg))
        for u in self.closure(v):
            if u != 0:
                al += self.block[u]
        return np.array(sorted(al), dtype=int)

    def allowed_set(self, frontier):
        """Background ∪ blocks over the union of closures of the frontier nodes (set-valued gate)."""
        al = set(range(self.n_bg))
        for f in frontier:
            for u in self.closure(f):
                if u != 0:
                    al.update(self.block[u])
        return np.array(sorted(al), dtype=int)
```

- [ ] **Step 4: Run tests**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_dag_placement.py -v`
Expected: PASS — the two new tests AND all pre-existing tests (backward compat).

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/dag_placement.py spark-vi/tests/test_dag_placement.py
git commit -m "feat(dag-placement): multi-parent DAG-native DagLayout (closure=all ancestors, depth=longest path)"
```

---

### Task 2: `frontier_from_coded` (set-valued truth)

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/dag_placement.py`
- Test: `spark-vi/tests/test_dag_placement.py`

**Interfaces:**
- Consumes: `DagLayout.subtree`.
- Produces: `frontier_from_coded(coded_nodes, lay) -> frozenset[int]` — the most-specific attested nodes (drop any attested node that has an attested descendant). `label_from_coded` is retained unchanged.

- [ ] **Step 1: Write the failing test** (append)

```python
from spark_vi.models.topic.dag_placement import frontier_from_coded

def test_frontier_from_coded_cases():
    lay = DagLayout(DIAMOND)
    assert frontier_from_coded([1, 4], lay) == frozenset({4})       # same-path -> most-specific
    assert frontier_from_coded([4, 5], lay) == frozenset({4, 5})    # comorbid incomparable -> set
    assert frontier_from_coded([2, 3], lay) == frozenset({2, 3})    # contradictory siblings -> set
    assert frontier_from_coded([1, 2, 4], lay) == frozenset({4})    # both parents + child -> child
    # single-parent tree: ancestor+descendant collapses to the descendant
    assert frontier_from_coded([1, 3], DagLayout(PARENT)) == frozenset({3})
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_dag_placement.py::test_frontier_from_coded_cases -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement** (append after `label_from_coded`)

```python
def frontier_from_coded(coded_nodes, lay):
    """The set-valued truth: the most-specific attested nodes = attested nodes with NO attested
    descendant. Drops attested ancestors (same-path -> most-specific), keeps incomparable attested
    nodes as a set (comorbid or contradictory — the DAG cannot tell these apart, so we do not roll
    them up; multi-frontier is instrumented by evaluate). Returns a frozenset."""
    C = set(coded_nodes)
    return frozenset(c for c in C
                     if not any((c2 != c) and (c2 in lay.subtree(c)) for c2 in C))
```

- [ ] **Step 4: Run tests**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_dag_placement.py -v`
Expected: PASS (new test + all prior).

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/dag_placement.py spark-vi/tests/test_dag_placement.py
git commit -m "feat(dag-placement): frontier_from_coded set-valued truth (drop attested ancestors)"
```

---

### Task 3: `fit_gated` set-valued gate

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/dag_placement.py` (one line in `fit_gated`)
- Test: `spark-vi/tests/test_dag_placement.py`

**Interfaces:**
- Consumes: `DagLayout.allowed_set`.
- Produces: `fit_gated(train_docs, train_labels, lay, V, ...)` where each entry of `train_labels` may be a **frontier set** (iterable of node ids) OR a scalar (treated as a singleton). The gate masks each doc to `allowed_set(frontier)`.

- [ ] **Step 1: Write the failing test** (append)

```python
def test_fit_gated_accepts_frontier_sets():
    # comorbid training labels (sets) must be accepted and produce a valid beta_hat
    docs, labels, _ = dag_placement_corpus(
        parent=PARENT, node_prev={1:.18,2:.18,3:.16,4:.16,5:.16,6:.16},
        V=120, doc_len=40, seed=1)
    lay = DagLayout(PARENT, n_bg=2, tpn=1)
    set_labels = [frozenset({int(y)}) for y in labels[:800]]   # scalars as singleton sets
    rng = np.random.default_rng(3)
    beta = fit_gated(docs[:800], set_labels, lay, 120, n_iter=40, burn=20, rng=rng)
    assert beta.shape == (lay.K, 120)
    assert np.allclose(beta.sum(1), 1.0, atol=1e-6)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_dag_placement.py::test_fit_gated_accepts_frontier_sets -v`
Expected: FAIL — the current gate does `lay.allowed(v)` where `v` is a scalar; a set raises.

- [ ] **Step 3: Implement** — in `fit_gated`, replace the single line that builds `allowed`:

Find:
```python
    allowed = [lay.allowed(v) for v in train_labels]
```
Replace with:
```python
    # Each label may be a scalar node id or a frontier set (comorbid patient). A comorbid patient
    # trains every block along the union of its frontier's closures — strictly better use of data.
    allowed = [lay.allowed_set(y if hasattr(y, "__iter__") else (y,)) for y in train_labels]
```

- [ ] **Step 4: Run tests**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_dag_placement.py -v`
Expected: PASS — the new test AND the existing `test_fit_gated_learns_node_signatures` (scalar labels still work via the singleton branch).

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/dag_placement.py spark-vi/tests/test_dag_placement.py
git commit -m "feat(dag-placement): fit_gated set-valued gate (comorbid patients train all attested blocks)"
```

---

### Task 4: `evaluate` set-valued + instrumentation

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/dag_placement.py` (replace `evaluate`, add `_hops`)
- Test: `spark-vi/tests/test_dag_placement.py`

**Interfaces:**
- Consumes: `DagLayout` (`nodes`, `subtree`, `depth`, `parents`, `children`).
- Produces: `evaluate(profiles, test_labels, lay) -> dict` where `test_labels` entries may be frontier sets or scalars. Keys: `node_auc`, `auc_by_depth`, `mrr`, `top2`, `mean_hops`, `frontier_size_mean`, `multi_frontier_rate`. Adds `_hops(a, b, lay)` (undirected BFS distance over parent/child edges).

- [ ] **Step 1: Write the failing test** (append)

```python
def test_evaluate_set_valued_and_instrumented():
    lay = DagLayout(DIAMOND)
    labels = [frozenset({4}), frozenset({4, 5}), frozenset({2, 3}), frozenset({1})]
    profiles = []
    for f in labels:                                   # closure-loaded perfect profiles
        load = set()
        for t in f:
            load |= (set(lay.closure(t)) - {0})
        profiles.append({u: (1.0 if u in load else 0.0) for u in lay.nodes})
    m = evaluate(profiles, labels, lay)
    assert all(v >= 0.99 for v in m["node_auc"].values())   # every node perfectly separated
    assert m["mrr"] == 1.0                                   # best true node ranks first each doc
    assert abs(m["frontier_size_mean"] - 1.5) < 1e-9
    assert abs(m["multi_frontier_rate"] - 0.5) < 1e-9        # 2 of 4 docs are comorbid
    assert np.isfinite(m["mean_hops"])
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_dag_placement.py::test_evaluate_set_valued_and_instrumented -v`
Expected: FAIL — current `evaluate` calls `lay.nodes.index(t)` on a set / lacks the new keys.

- [ ] **Step 3: Implement** — add `_hops` (after `_auc`) and REPLACE `evaluate`:

```python
def _hops(a, b, lay):
    """Undirected hop distance between two nodes over parent/child edges (BFS)."""
    if a == b:
        return 0
    seen = {a}
    queue = [(a, 0)]
    while queue:
        x, d = queue.pop(0)
        for nb in list(lay.parents.get(x, [])) + lay.children.get(x, []):
            if nb == b:
                return d + 1
            if nb not in seen:
                seen.add(nb)
                queue.append((nb, d + 1))
    return float("inf")


def evaluate(profiles, test_labels, lay):
    """Per-node case-finding AUC (subtree membership), AUC by longest-path depth, and set-valued
    ranking. `test_labels` entries may be frontier sets or scalars (scalars -> singletons). A patient
    is a positive for node u if any of its frontier lies in subtree(u). MRR/top2/mean_hops use the
    BEST (closest) true frontier node. frontier_size_mean and multi_frontier_rate instrument the
    comorbid/contradictory ambiguity (the DAG cannot tell those apart; we surface it, not resolve it).
    Profiles are the graded affinity dicts from `profile`; scoring never collapses to one node."""
    fronts = [set(t) if hasattr(t, "__iter__") else {t} for t in test_labels]
    P = np.array([[pr[u] for u in lay.nodes] for pr in profiles])
    node_auc = {u: _auc(P[:, i], [bool(f & lay.subtree(u)) for f in fronts])
                for i, u in enumerate(lay.nodes)}
    ranks, hops = [], []
    for i, f in enumerate(fronts):
        true_idx = [lay.nodes.index(t) for t in f if t in lay.nodes]   # skip root/unscoreable
        if not true_idx:
            continue
        ranks.append(min(1 + int((P[i] > P[i][j]).sum()) for j in true_idx))   # best (smallest) rank
        pred = lay.nodes[int(np.argmax(P[i]))]
        hops.append(min(_hops(pred, lay.nodes[j], lay) for j in true_idx))
    ranks = np.array(ranks, dtype=float) if ranks else np.array([np.nan])
    by_depth = {}
    for dep in sorted({lay.depth(u) for u in lay.nodes}):
        us = [u for u in lay.nodes if lay.depth(u) == dep]
        by_depth[dep] = float(np.nanmean([node_auc[u] for u in us]))
    return {"node_auc": node_auc, "auc_by_depth": by_depth,
            "mrr": float(np.nanmean(1.0 / ranks)),
            "top2": float(np.nanmean(ranks <= 2)),
            "mean_hops": float(np.mean(hops)) if hops else float("nan"),
            "frontier_size_mean": float(np.mean([len(f) for f in fronts])),
            "multi_frontier_rate": float(np.mean([len(f) > 1 for f in fronts]))}
```

- [ ] **Step 4: Run tests**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_dag_placement.py -v`
Expected: PASS — new test AND existing `test_evaluate_perfect_profiles_score_high` and `test_evaluate_tolerates_root_label` (scalars treated as singletons; root label `{0}` yields empty `true_idx` and is skipped; extra dict keys don't break the existing `==` assertions).

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/dag_placement.py spark-vi/tests/test_dag_placement.py
git commit -m "feat(dag-placement): set-valued evaluate + frontier instrumentation + DAG-distance"
```

---

### Task 5: `identifiability_annotation` multi-parent

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/dag_placement.py` (replace the candidate-pair construction in `identifiability_annotation`)
- Test: `spark-vi/tests/test_dag_placement.py`

**Interfaces:**
- Consumes: `DagLayout` (`nodes`, `children`, `parents`, `block`).
- Produces: `identifiability_annotation(beta_hat, lay, *, tol=0.9)` — pairs `(u, v, cos)` with cosine >= tol among WITHIN-STRUCTURE pairs: every parent<->child edge, plus any two nodes sharing at least one common parent. Cross-branch pairs are never reported.

- [ ] **Step 1: Write the failing test** (append)

```python
def test_identifiability_multiparent_siblings():
    lay = DagLayout(DIAMOND, n_bg=2, tpn=1)
    beta = np.random.default_rng(0).random((lay.K, 20)) + 0.01
    beta[lay.block[4][0]] = beta[lay.block[5][0]].copy()     # 4,5 share parent 1 (siblings) -> near-identical
    beta /= beta.sum(1, keepdims=True)
    flagged = identifiability_annotation(beta, lay, tol=0.99)
    pairs = {(min(u, v), max(u, v)) for u, v, _ in flagged}
    assert (4, 5) in pairs                                    # siblings sharing a parent, flagged
    assert (2, 3) not in pairs                                # share only root's... (2,3 share parent 0)
```

Note: 2 and 3 share parent 0 (root), so they ARE siblings under root and WOULD be candidates — but the test makes only 4/5 near-identical, so (2,3) is a candidate whose cosine is below tol and thus not flagged. The assertion `(2,3) not in pairs` holds because their random topics are not near-identical, not because they are excluded structurally. Keep the assertion as written; it verifies tol-gating, not exclusion.

- [ ] **Step 2: Run to verify it fails**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_dag_placement.py::test_identifiability_multiparent_siblings -v`
Expected: FAIL — current sibling construction assumes each child has one parent; with the multi-parent map the pair set differs, and the pre-change code may not enumerate (4,5) correctly.

- [ ] **Step 3: Implement** — replace the candidate-pair construction inside `identifiability_annotation` (keep the `cos`, `_node_topic_mean`, and the `for u, v in pairs` scoring loop unchanged):

```python
    pairs = set()
    for c, ps in lay.parents.items():                    # every parent<->child edge
        for p in ps:
            if p != 0:
                pairs.add((min(p, c), max(p, c)))
    for p, kids in lay.children.items():                 # siblings sharing at least one parent
        for i in range(len(kids)):
            for j in range(i + 1, len(kids)):
                pairs.add((min(kids[i], kids[j]), max(kids[i], kids[j])))
```

(The parent<->child block previously iterated `lay.nodes` and `lay.children`; this version iterates `lay.parents` so multi-parent edges are all included. `(min, max)` ordering keeps pairs de-duplicated. Root (0) parent edges are excluded because root has no block.)

- [ ] **Step 4: Run tests**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_dag_placement.py -v`
Expected: PASS — new test AND existing `test_identifiability_flags_near_identical_siblings` (in the single-parent `PARENT` DAG, 3 and 4 share parent 1 and are still flagged; cross-branch 3,5 still never a candidate).

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/dag_placement.py spark-vi/tests/test_dag_placement.py
git commit -m "feat(dag-placement): multi-parent identifiability (all edges + shared-parent siblings)"
```

---

### Task 6: `render_profile` DAG dedup

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/dag_placement.py` (replace `render_profile`)
- Test: `spark-vi/tests/test_dag_placement.py`

**Interfaces:**
- Consumes: `DagLayout` (`children`, `nodes`).
- Produces: `render_profile(affinity, lay, *, names=None, true_node=None, width=24) -> str`. A multi-parent node is rendered **once** (dedup via a visited set); a second encounter is shown as a short reference line so the tree stays readable.

- [ ] **Step 1: Write the failing test** (append)

```python
def test_render_profile_dag_renders_each_node_once():
    lay = DagLayout(DIAMOND)
    aff = {u: 0.1 * u for u in lay.nodes}
    s = render_profile(aff, lay, true_node=4)
    assert "true" in s
    for u in lay.nodes:                                  # every node appears
        assert str(u) in s
    # node 4 is reachable via parents 1 and 2, but its full affinity bar is rendered once
    full_lines = [ln for ln in s.splitlines() if ln.strip().split()[0:1] == [str(4)] or f" {4} " in ln]
    # its numeric affinity 0.40 should appear exactly once (rendered once, referenced elsewhere)
    assert s.count("0.40") == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_dag_placement.py::test_render_profile_dag_renders_each_node_once -v`
Expected: FAIL — the current recursive walk visits a multi-parent node once per parent, so `0.40` appears twice.

- [ ] **Step 3: Implement** — replace `render_profile`:

```python
def render_profile(affinity, lay, *, names=None, true_node=None, width=24):
    """Indented DAG tree with a unicode affinity bar per node (spot-check output). A multi-parent
    node is rendered in full ONCE (first encounter); later encounters show a short reference line so
    the tree stays readable and no node's affinity is double-counted visually."""
    names = names or {}
    lines = []
    seen = set()

    def bar(x):
        n = int(round(max(0.0, min(1.0, x)) * width))
        return "█" * n + "▁" * (width - n)

    def walk(v, prefix, is_last):
        if v == 0:
            lines.append(names.get(0, "root"))
        else:
            conn = "└─ " if is_last else "├─ "
            nm = str(names.get(v, v)).ljust(10)
            if v in seen:                                 # multi-parent: reference, do not re-render
                lines.append(f"{prefix}{conn}{nm} (^ shared)")
                return
            seen.add(v)
            a = affinity.get(v, 0.0)
            mark = "  <- true" if v == true_node else ""
            lines.append(f"{prefix}{conn}{nm} {bar(a)} {a:0.2f}{mark}")
        kids = lay.children.get(v, [])
        child_prefix = prefix + ("   " if is_last else "│  ") if v != 0 else ""
        for i, c in enumerate(kids):
            walk(c, child_prefix, i == len(kids) - 1)

    walk(0, "", True)
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_dag_placement.py -v`
Expected: PASS — new test AND existing `test_render_profile_marks_true_and_shows_all_nodes` (single-parent tree: every node visited once, so no reference lines appear and all nodes render as before).

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/dag_placement.py spark-vi/tests/test_dag_placement.py
git commit -m "feat(dag-placement): render_profile dedups multi-parent nodes (render once, reference after)"
```

---

### Task 7: `dag_placement_corpus_multi` (multi-parent + comorbid)

**Files:**
- Modify: `spark-vi/tests/_stm_synth.py` (add `dag_placement_corpus_multi`)
- Test: `spark-vi/tests/test_dag_placement.py`

**Interfaces:**
- Produces: `dag_placement_corpus_multi(*, parent, leaf_prev, comorbid_rate, V, doc_len, seed) -> (docs, labels, node_codes)` where `parent` is a multi-parent map, `labels` is a list of **frozensets** (the frontier truth per doc), `docs` a list of 1-D int arrays, `node_codes: {node:int}`. Each item draws a frontier of 1 leaf (prob `1-comorbid_rate`) or 2 incomparable leaves (prob `comorbid_rate`), emits background + signature blocks along the union of the frontier's closures.

- [ ] **Step 1: Write the failing test** (append)

```python
from tests._stm_synth import dag_placement_corpus_multi

def test_dag_placement_corpus_multi_shapes():
    docs, labels, node_codes = dag_placement_corpus_multi(
        parent=DIAMOND, leaf_prev={4: .5, 5: .5}, comorbid_rate=0.3,
        V=120, doc_len=48, seed=0)
    assert len(docs) == len(labels)
    assert all(isinstance(f, frozenset) and len(f) >= 1 for f in labels)
    assert set(node_codes.keys()) == set(DIAMOND.keys())
    assert any(len(f) > 1 for f in labels)                  # some comorbid patients exist
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_dag_placement.py::test_dag_placement_corpus_multi_shapes -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement** (append to `tests/_stm_synth.py`)

```python
def dag_placement_corpus_multi(*, parent, leaf_prev, comorbid_rate, V, doc_len, seed):
    """Multi-parent hierarchical-placement plant with comorbid patients. Each non-root node owns a
    signature vocab block plus one exact 'node code'. An item's frontier is 1 leaf (prob
    1-comorbid_rate) or 2 distinct leaves (prob comorbid_rate); it emits a shared common pool + the
    signature blocks along the union of its frontier's closures. Labels are frozensets (the frontier
    truth). Returns (docs, labels, node_codes)."""
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout
    rng = np.random.default_rng(seed)
    lay = DagLayout(parent, n_bg=2, tpn=1)
    nodes = lay.nodes
    C = V // 3                                             # shared common pool [0:C]
    sig = max(2, (V - C) // (len(nodes) + 1))
    node_sig = {u: np.arange(C + i * sig, C + i * sig + sig) for i, u in enumerate(nodes)}
    node_codes = {u: int(node_sig[u][0]) for u in nodes}
    leaves = sorted(leaf_prev.keys())
    p = np.array([leaf_prev[u] for u in leaves], float); p /= p.sum()
    docs, labels = [], []
    for _ in range(2400):
        if len(leaves) >= 2 and rng.random() < comorbid_rate:
            f = frozenset(rng.choice(leaves, size=2, replace=False, p=p).tolist())
        else:
            f = frozenset({int(rng.choice(leaves, p=p))})
        blocks = set()
        for leaf in f:
            for u in lay.closure(leaf):
                if u != 0:
                    blocks.add(u)
        blocks = sorted(blocks)
        toks = [rng.integers(0, C, size=doc_len // 2)]
        per = max(1, (doc_len - doc_len // 2) // len(blocks))
        for u in blocks:
            toks.append(rng.choice(node_sig[u], size=per))
        docs.append(np.concatenate(toks).astype(np.int64))
        labels.append(f)
    return docs, labels, node_codes
```

- [ ] **Step 4: Run tests**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_dag_placement.py::test_dag_placement_corpus_multi_shapes -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/tests/_stm_synth.py spark-vi/tests/test_dag_placement.py
git commit -m "test(dag-placement): multi-parent + comorbid corpus with set-valued (frontier) labels"
```

---

### Task 8: End-to-end behavioral gate (multi-parent + comorbid)

**Files:**
- Test: `spark-vi/tests/test_dag_placement.py`

**Interfaces:**
- Consumes: all of the above.

- [ ] **Step 1: Write the test** (append)

```python
def test_end_to_end_multiparent_comorbid():
    docs, labels, node_codes = dag_placement_corpus_multi(
        parent=DIAMOND, leaf_prev={4: .5, 5: .5}, comorbid_rate=0.3,
        V=120, doc_len=48, seed=2)
    lay = DagLayout(DIAMOND, n_bg=2, tpn=1)
    ntr = int(0.7 * len(docs))
    rng = np.random.default_rng(5)
    beta = fit_gated(docs[:ntr], labels[:ntr], lay, 120, n_iter=80, burn=40, rng=rng)
    codes = set(node_codes.values())
    profs = [profile(strip_dag_node_codes(d, codes), beta, lay, n_iter=40, burn=20, rng=rng)
             for d in docs[ntr:]]
    m = evaluate(profs, labels[ntr:], lay)
    # multi-parent recovery: loose floors (investigate, do NOT loosen, if these fail).
    assert m["auc_by_depth"][1] >= 0.80          # shallow (axis parents 1,2,3)
    assert m["node_auc"][4] >= 0.70              # a multi-parent leaf is found above chance
    assert m["node_auc"][5] >= 0.70
    assert m["multi_frontier_rate"] > 0.0        # comorbid patients are present + measured
    assert np.isfinite(m["mrr"])
```

- [ ] **Step 2: Run it** (with all prior tasks in place, this trains + folds in; ~30-90s)

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_dag_placement.py::test_end_to_end_multiparent_comorbid -v`
Expected: PASS. If the floors fail, investigate (the plant is model-matched and should recover well); a low score signals a real issue in the multi-parent gate/eval, not a threshold to relax. Report the observed `auc_by_depth`, per-node AUC, `mrr`, and `multi_frontier_rate`.

- [ ] **Step 3: Run the whole suite**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_dag_placement.py -v`
Expected: all PASS (the 14 original + the ~8 new tests).

- [ ] **Step 4: Commit**

```bash
git add spark-vi/tests/test_dag_placement.py
git commit -m "test(dag-placement): end-to-end multi-parent + comorbid behavioral gate"
```

---

## Self-Review

**Spec coverage:** DAG-native DagLayout / closure=all-ancestors / depth=longest-path / allowed_set (Task 1); frontier truth (Task 2); comorbid gating (Task 3); set-valued eval + instrumentation + DAG-distance (Task 4); multi-parent identifiability (Task 5); DAG render dedup (Task 6); multi-parent+comorbid generator (Task 7); end-to-end (Task 8). The optional cohort-assembly mutual-exclusion list and the OMOP DAG-builder are spec-deferred (cluster-side), not in this plan.

**Placeholder scan:** every code step has complete code copied from the validated prototype (`scratchpad/dag_native_proto.py`, 23/24 checks, lone miss = a wrong test expectation, not code). No TBD/TODO.

**Type consistency:** `DagLayout.closure -> list`, `subtree -> set`, `allowed/allowed_set -> np.ndarray`, `depth -> int` used consistently. `frontier_from_coded -> frozenset` consumed by nothing structurally (labels flow as sets into `fit_gated`/`evaluate`, which accept any iterable). `fit_gated` and `evaluate` both branch on `hasattr(y, "__iter__")` so scalars and sets both work — the backward-compat path.

**Backward-compat check:** Task 1 preserves exact list-ordering of `closure` and the `allowed`/`depth`/`subtree` results for single-parent maps (validated); Tasks 3/4 treat scalar labels as singletons; Task 5 keeps single-parent sibling/edge flagging; Task 6 renders single-parent trees unchanged. The 14 shipped tests must stay green after every task — each task's Step 4 re-runs the full file.
