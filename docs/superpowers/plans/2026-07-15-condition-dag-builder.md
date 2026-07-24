# Condition DAG Builder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the OMOP condition DAG for the placement engine — a pure transformation from `concept_ancestor` edges to the engine's `{child: [parents]}` int map, with attestation pruning and a pruning ledger.

**Architecture:** A new pure-Python module `charmpheno/charmpheno/omop/condition_dag.py` operating on a small in-memory edge list (concept-id space): `ConditionDag` (parent map + anchor + names, longest-path depth), `build_condition_dag`, `prune_by_attestation` (principled cap by patient-count, transitive rewire), `pruning_ledger` (kept/dropped by depth + coarsening rate + K), and `ConditionDag.to_engine()` (remap to int ids, anchor→0, consumable by `spark_vi.models.topic.dag_placement.DagLayout`).

**Tech Stack:** Python, pytest. Integrates with the existing `spark_vi` engine (importable from the venv). No Spark/BigQuery in this piece (the loader that produces edges from CSV/BQ is piece 2/3).

## Global Constraints

- This is the **domain layer** (`charmpheno/omop/`), so OMOP concept ids are expected; but the **unit tests use synthetic integer ids only** (the one real-data test uses the committed diabetes fixture).
- No LaTeX in comments/docstrings; plain text + Unicode where needed.
- Commit trailer, exactly on its own line: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Run tests with: `cd charmpheno && ../.venv/bin/python -m pytest tests/test_condition_dag.py -v`.
- Prototype (validated 11/12; the one miss was a wrong test expectation, corrected here): `scratchpad/condition_dag_proto.py`.
- Spec: `docs/superpowers/specs/2026-07-15-condition-dag-builder-design.md`.
- The attestation prune is the **principled cap** (learnability: no patients → no learnable topic), NOT an arbitrary top-N. The ledger must make the granularity cost visible.

---

## File Structure

- Create `charmpheno/charmpheno/omop/condition_dag.py` — `ConditionDag`, `build_condition_dag`, `prune_by_attestation`, `pruning_ledger`.
- Create `charmpheno/tests/test_condition_dag.py` — synthetic unit tests + one real-fixture smoke test.
- Already present (committed with Task 5): `charmpheno/tests/data/diabetes_subtree_edges.csv` (140 edges, 127 nodes, anchor 201820).

Shared synthetic fixture used across tests (define near the top of the test file):

```python
# synthetic diamond with a single-child tail:
#   100 root;  101,102 children of root;  103 <- {101,102} (multi-parent);  104 <- 103
DIAMOND_EDGES = [(100, 101), (100, 102), (101, 103), (102, 103), (103, 104)]
DIAMOND_NODES = {100, 101, 102, 103, 104}
ANCHOR = 100
```

---

### Task 1: `ConditionDag` + `build_condition_dag`

**Files:**
- Create: `charmpheno/charmpheno/omop/condition_dag.py`
- Test: `charmpheno/tests/test_condition_dag.py`

**Interfaces:**
- Produces: `ConditionDag(parents: dict[int, list[int]], anchor: int, names: dict[int,str]=None)` with attributes `parents`, `anchor`, `names`; methods `nodes()->set`, `children()->dict[int,list[int]]`, `depth(cid)->int` (longest path from anchor). `build_condition_dag(edges, anchor, node_ids, names=None) -> ConditionDag` (orphans without an in-set parent attach to the anchor).

- [ ] **Step 1: Write the failing test**

```python
from charmpheno.omop.condition_dag import ConditionDag, build_condition_dag

DIAMOND_EDGES = [(100, 101), (100, 102), (101, 103), (102, 103), (103, 104)]
DIAMOND_NODES = {100, 101, 102, 103, 104}
ANCHOR = 100

def test_build_multiparent_and_depth():
    dag = build_condition_dag(DIAMOND_EDGES, ANCHOR, DIAMOND_NODES)
    assert dag.nodes() == DIAMOND_NODES
    assert set(dag.parents[103]) == {101, 102}          # multi-parent
    assert dag.depth(104) == 3 and dag.depth(103) == 2 and dag.depth(100) == 0
    assert sorted(dag.children()[101]) == [103]

def test_build_orphan_attaches_to_anchor():
    # 202 has no in-set parent edge -> should attach to the anchor
    dag = build_condition_dag([(200, 201)], anchor=200, node_ids={200, 201, 202})
    assert dag.parents[202] == [200]
    assert dag.depth(202) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd charmpheno && ../.venv/bin/python -m pytest tests/test_condition_dag.py::test_build_multiparent_and_depth -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
"""Build the anchor-first label DAG for hierarchical placement from OMOP concept_ancestor.

Pure transformation over a small in-memory edge list in concept-id space, plus an attestation
prune (the principled size cap: a node no patient populates has no learnable topic) and a pruning
ledger that makes the granularity cost of pruning visible. `ConditionDag.to_engine()` remaps to the
integer `{child: [parents]}` map consumed by spark_vi.models.topic.dag_placement.DagLayout.

See docs/superpowers/specs/2026-07-15-condition-dag-builder-design.md."""
from collections import defaultdict, Counter


class ConditionDag:
    """Multi-parent condition DAG in concept-id space, rooted at `anchor`. `parents` maps a
    non-anchor concept id to its list of parent concept ids (the anchor has no entry)."""

    def __init__(self, parents, anchor, names=None):
        self.parents = {c: sorted(set(ps)) for c, ps in parents.items() if c != anchor}
        self.anchor = anchor
        self.names = dict(names or {})
        self._depth = {}

    def nodes(self):
        return {self.anchor} | set(self.parents.keys())

    def children(self):
        ch = defaultdict(list)
        for c, ps in self.parents.items():
            for p in ps:
                ch[p].append(c)
        return {p: sorted(cs) for p, cs in ch.items()}

    def depth(self, cid, _stack=()):
        """Longest path length from the anchor to `cid` (anchor = 0). Memoized; cycle-guarded."""
        if cid in self._depth:
            return self._depth[cid]
        ps = [p for p in self.parents.get(cid, []) if p != cid and p not in _stack]
        d = 0 if (cid == self.anchor or not ps) else 1 + max(self.depth(p, _stack + (cid,)) for p in ps)
        self._depth[cid] = d
        return d


def build_condition_dag(edges, anchor, node_ids, names=None):
    """From min-sep-1 (ancestor, descendant) edges restricted to `node_ids` (standard-condition
    descendants incl. the anchor), assemble the multi-parent parent map. A node with no in-set
    parent (orphan) attaches to the anchor so the DAG is connected and rooted."""
    nodeset = set(node_ids) | {anchor}
    parents = defaultdict(list)
    for a, d in edges:
        if a in nodeset and d in nodeset and a != d:
            parents[d].append(a)
    for c in nodeset:
        if c != anchor and c not in parents:
            parents[c] = [anchor]
    return ConditionDag(parents, anchor, {c: (names or {}).get(c, str(c)) for c in nodeset})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd charmpheno && ../.venv/bin/python -m pytest tests/test_condition_dag.py -v`
Expected: PASS (both build tests).

- [ ] **Step 5: Commit**

```bash
git add charmpheno/charmpheno/omop/condition_dag.py charmpheno/tests/test_condition_dag.py
git commit -m "feat(condition-dag): ConditionDag + build_condition_dag (multi-parent, longest-path depth)"
```

---

### Task 2: `prune_by_attestation`

**Files:**
- Modify: `charmpheno/charmpheno/omop/condition_dag.py`
- Test: `charmpheno/tests/test_condition_dag.py`

**Interfaces:**
- Consumes: `ConditionDag`.
- Produces: `prune_by_attestation(dag, counts, min_n) -> ConditionDag` — drops every non-anchor node with `counts.get(cid,0) < min_n`; each surviving node is rewired to its nearest surviving ancestors (transitive). The anchor is never dropped.

- [ ] **Step 1: Write the failing test**

```python
from charmpheno.omop.condition_dag import prune_by_attestation

def test_prune_drops_low_count_and_rewires():
    dag = build_condition_dag(DIAMOND_EDGES, ANCHOR, DIAMOND_NODES)
    counts = {101: 50, 102: 40, 103: 0, 104: 20}     # 103 is below threshold
    pruned = prune_by_attestation(dag, counts, min_n=5)
    assert 103 not in pruned.nodes()                  # dropped
    assert pruned.nodes() == {100, 101, 102, 104}
    assert set(pruned.parents[104]) == {101, 102}     # 104 rewired past dropped 103 to its parents

def test_prune_never_drops_anchor():
    dag = build_condition_dag(DIAMOND_EDGES, ANCHOR, DIAMOND_NODES)
    pruned = prune_by_attestation(dag, counts={}, min_n=999)   # everything below threshold
    assert pruned.nodes() == {ANCHOR}                 # only the anchor survives
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd charmpheno && ../.venv/bin/python -m pytest tests/test_condition_dag.py::test_prune_drops_low_count_and_rewires -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation** (append to `condition_dag.py`)

```python
def prune_by_attestation(dag, counts, min_n):
    """Drop every non-anchor node with fewer than `min_n` attesting patients; rewire each surviving
    node to its nearest surviving ancestors (transitive walk up past dropped nodes). The anchor is
    never dropped. This is the principled size cap: a node no cohort patient populates cannot have a
    learnable topic."""
    keep = {n for n in dag.nodes() if n == dag.anchor or counts.get(n, 0) >= min_n}
    new_parents = {}
    for c in keep:
        if c == dag.anchor:
            continue
        surv, seen, stack = set(), set(), list(dag.parents.get(c, []))
        while stack:
            p = stack.pop()
            if p in seen:
                continue
            seen.add(p)
            if p in keep:
                surv.add(p)
            else:
                stack.extend(dag.parents.get(p, []))
        new_parents[c] = sorted(surv) if surv else [dag.anchor]
    return ConditionDag(new_parents, dag.anchor, dag.names)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd charmpheno && ../.venv/bin/python -m pytest tests/test_condition_dag.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add charmpheno/charmpheno/omop/condition_dag.py charmpheno/tests/test_condition_dag.py
git commit -m "feat(condition-dag): prune_by_attestation (principled cap, transitive rewire)"
```

---

### Task 3: `pruning_ledger`

**Files:**
- Modify: `charmpheno/charmpheno/omop/condition_dag.py`
- Test: `charmpheno/tests/test_condition_dag.py`

**Interfaces:**
- Consumes: two `ConditionDag`s (before/after prune), `counts`, optional `cohort_frontiers` (list of per-patient most-specific attested concept-id sets).
- Produces: `pruning_ledger(before, after, counts, *, cohort_frontiers=None) -> dict` with keys `kept`, `dropped`, `K_nodes`, `kept_by_depth`, `dropped_by_depth`, `min_count_kept`, and (when `cohort_frontiers` given) `coarsening_rate`, `mean_depth_drop`.

- [ ] **Step 1: Write the failing test**

```python
from charmpheno.omop.condition_dag import pruning_ledger

def test_ledger_counts_and_coarsening():
    dag = build_condition_dag(DIAMOND_EDGES, ANCHOR, DIAMOND_NODES)
    counts = {101: 50, 102: 40, 103: 0, 104: 20}
    pruned = prune_by_attestation(dag, counts, min_n=5)
    # 2 of 4 patients had their most-specific node (103) pruned -> coarsened
    led = pruning_ledger(dag, pruned, counts,
                         cohort_frontiers=[{103}, {104}, {101}, {103}])
    assert led["K_nodes"] == 4 and led["dropped"] == 1
    assert led["dropped_by_depth"] == {2: 1}           # 103 was at depth 2
    assert abs(led["coarsening_rate"] - 0.5) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd charmpheno && ../.venv/bin/python -m pytest tests/test_condition_dag.py::test_ledger_counts_and_coarsening -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation** (append to `condition_dag.py`)

```python
def pruning_ledger(before, after, counts, *, cohort_frontiers=None):
    """A receipt for what pruning discarded. Structural stats need only the two DAGs + counts:
    kept/dropped totals, breakdown by (pre-prune) depth, resulting K (= engine topic-count driver),
    and the smallest kept count. When `cohort_frontiers` (per-patient most-specific attested
    concept-id sets) is supplied, also report the coarsening rate (fraction of patients whose
    most-specific node was pruned, so their frontier rolled up) and the mean depth drop for them."""
    kept = after.nodes()
    dropped = before.nodes() - kept
    led = {"kept": len(kept), "dropped": len(dropped), "K_nodes": len(kept),
           "kept_by_depth": dict(sorted(Counter(before.depth(n) for n in kept).items())),
           "dropped_by_depth": dict(sorted(Counter(before.depth(n) for n in dropped).items())),
           "min_count_kept": min((counts.get(n, 0) for n in kept if n != before.anchor), default=0)}
    if cohort_frontiers is not None:
        coarsened, drops = 0, []
        for fr in cohort_frontiers:
            dfr = [c for c in fr if c in dropped]
            if dfr:
                coarsened += 1
                worst = max(before.depth(c) for c in dfr)
                aft = max((after.depth(c) for c in fr if c in kept), default=0)
                drops.append(worst - aft)
        n = len(cohort_frontiers)
        led["coarsening_rate"] = coarsened / n if n else 0.0
        led["mean_depth_drop"] = (sum(drops) / len(drops)) if drops else 0.0
    return led
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd charmpheno && ../.venv/bin/python -m pytest tests/test_condition_dag.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add charmpheno/charmpheno/omop/condition_dag.py charmpheno/tests/test_condition_dag.py
git commit -m "feat(condition-dag): pruning_ledger (kept/dropped by depth + coarsening rate + K)"
```

---

### Task 4: `ConditionDag.to_engine()` + DagLayout integration

**Files:**
- Modify: `charmpheno/charmpheno/omop/condition_dag.py`
- Test: `charmpheno/tests/test_condition_dag.py`

**Interfaces:**
- Consumes: `ConditionDag`; `spark_vi.models.topic.dag_placement.DagLayout`.
- Produces: `ConditionDag.to_engine() -> (parent_int: dict[int, list[int]], int2cid: dict[int,int], cid2int: dict[int,int])`. The anchor maps to int 0 (root); descendants map to 1..N in `(depth, cid)` order; `parent_int` is directly loadable by `DagLayout`.

- [ ] **Step 1: Write the failing test**

```python
def test_to_engine_maps_anchor_to_zero_and_loads_into_daglayout():
    from spark_vi.models.topic.dag_placement import DagLayout
    dag = build_condition_dag(DIAMOND_EDGES, ANCHOR, DIAMOND_NODES)
    parent_int, int2cid, cid2int = dag.to_engine()
    assert cid2int[ANCHOR] == 0                          # anchor -> root 0
    assert int2cid[cid2int[103]] == 103                  # round-trips
    lay = DagLayout(parent_int, n_bg=2, tpn=1)
    assert lay.K == 2 + 4                                # 4 non-root nodes (101,102,103,104) + 2 bg
    assert 0 in lay.closure(cid2int[104])                # every node's closure reaches the root
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd charmpheno && ../.venv/bin/python -m pytest tests/test_condition_dag.py::test_to_engine_maps_anchor_to_zero_and_loads_into_daglayout -v`
Expected: FAIL with `AttributeError` (`to_engine` not defined).

- [ ] **Step 3: Write minimal implementation** (add as a method on `ConditionDag`, after `depth`)

```python
    def to_engine(self):
        """Remap concept ids to contiguous engine ids: anchor -> 0 (root), descendants -> 1..N in
        (depth, cid) order. Returns (parent_int, int2cid, cid2int); `parent_int` is the
        `{child: [parents]}` map that spark_vi's DagLayout consumes directly."""
        order = sorted((n for n in self.nodes() if n != self.anchor),
                       key=lambda c: (self.depth(c), c))
        cid2int = {self.anchor: 0}
        for i, c in enumerate(order, start=1):
            cid2int[c] = i
        int2cid = {i: c for c, i in cid2int.items()}
        parent_int = {cid2int[c]: [cid2int[p] for p in ps] for c, ps in self.parents.items()}
        return parent_int, int2cid, cid2int
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd charmpheno && ../.venv/bin/python -m pytest tests/test_condition_dag.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add charmpheno/charmpheno/omop/condition_dag.py charmpheno/tests/test_condition_dag.py
git commit -m "feat(condition-dag): to_engine int remap (anchor->0), DagLayout-loadable"
```

---

### Task 5: Real-data smoke test (committed diabetes fixture)

**Files:**
- Test: `charmpheno/tests/test_condition_dag.py`
- Data (already extracted, commit it here): `charmpheno/tests/data/diabetes_subtree_edges.csv` (140 edges, 127 nodes, anchor 201820)

**Interfaces:**
- Consumes: all of the above + the committed fixture.

- [ ] **Step 1: Write the test**

```python
def test_real_diabetes_subtree_structure():
    import csv
    from pathlib import Path
    from spark_vi.models.topic.dag_placement import DagLayout
    path = Path(__file__).parent / "data" / "diabetes_subtree_edges.csv"
    with open(path) as fh:
        rows = list(csv.reader(fh))[1:]                 # skip header
    edges = [(int(a), int(d)) for a, d in rows]
    ANCHOR_DM = 201820                                  # SNOMED "Diabetes mellitus"
    node_ids = {a for a, _ in edges} | {d for _, d in edges} | {ANCHOR_DM}
    dag = build_condition_dag(edges, ANCHOR_DM, node_ids)
    assert len(dag.nodes()) == 127                       # the real diabetes type/status taxonomy
    assert max(dag.depth(n) for n in dag.nodes()) == 4
    multiparent = sum(1 for c, ps in dag.parents.items() if len(ps) > 1)
    assert multiparent == 12                             # real type x status cross-axes
    parent_int, int2cid, _ = dag.to_engine()
    lay = DagLayout(parent_int, n_bg=2, tpn=1)
    assert lay.K == 2 + 126                              # 126 non-root nodes + 2 background
```

- [ ] **Step 2: Run the test**

Run: `cd charmpheno && ../.venv/bin/python -m pytest tests/test_condition_dag.py::test_real_diabetes_subtree_structure -v`
Expected: PASS (the fixture already exists in `tests/data/`).

- [ ] **Step 3: Run the whole file**

Run: `cd charmpheno && ../.venv/bin/python -m pytest tests/test_condition_dag.py -v`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add charmpheno/tests/test_condition_dag.py charmpheno/tests/data/diabetes_subtree_edges.csv
git commit -m "test(condition-dag): real diabetes-subtree structure smoke test + fixture"
```

---

## Self-Review

**Spec coverage:** builder + multi-parent (Task 1), attestation prune with transitive rewire (Task 2), pruning ledger incl. coarsening (Task 3), engine remap + DagLayout integration (Task 4), real-data smoke (Task 5). Deferred per spec (loader wiring, cohort/frontier assembly, cloud driver, complication/combined anchors) are out of this plan.

**Placeholder scan:** every code step has complete code transcribed from the validated prototype (`scratchpad/condition_dag_proto.py`). No TBD/TODO. The one prototype miss (a wrong `K == 2+4` expectation on a *pruned* 3-node DAG) is not reproduced: Task 4's `K == 2+4` is on the *unpruned* diamond (4 non-root nodes 101,102,103,104), which is correct.

**Type consistency:** `ConditionDag.parents: dict[int, list[int]]`, `nodes()->set`, `children()->dict`, `depth()->int` used consistently. `build_condition_dag -> ConditionDag` consumed by `prune_by_attestation` (-> ConditionDag) and `pruning_ledger`. `to_engine -> (dict, dict, dict)` where `parent_int` matches `DagLayout(parent, ...)`'s expected `{child:[parents]}`. The fixture path uses `Path(__file__).parent` so it is cwd-independent.
