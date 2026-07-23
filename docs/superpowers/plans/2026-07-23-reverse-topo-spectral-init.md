# Reverse-topo spectral init Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `topo_order ∈ {forward, reverse}` knob to the gated spectral init so nodes can be recovered leaves-first, deflated against their already-recovered proper-descendants, and A/B it against the current forward (ancestors-first) init.

**Architecture:** Two spectral functions (dense + scalable) share one loop shape; parametrize only (a) the node iteration order and (b) the deflation set (proper ancestors via `closure` vs proper descendants via `subtree`) through a small helper. Thread `topo_order` through the model → shim → cloud driver → run_experiment exactly like the existing `anchor_scope`. Default `forward` = zero behavior change.

**Tech Stack:** Python, NumPy, PySpark, pytest. Spec: `docs/superpowers/specs/2026-07-23-reverse-topo-spectral-init-design.md`.

## Global Constraints

- Engine code (`spark_vi/**`) is integer-id agnostic: no OMOP/concept ids, integer topic/node/code space only.
- Commit trailer EXACTLY: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- No LaTeX; plain ASCII + Unicode Greek (α, λ, Σ) only.
- Cite literature where relevant (Arora et al. 2013 anchor recovery is already cited in gated_init.py; don't re-derive).
- Tests: never loosen a threshold/assertion to pass; fix the implementation (or, for a synthetic fixture, strengthen the PLANT — not the assertion — if anchor recovery is ambiguous).
- Default `topo_order="forward"` MUST reproduce current behavior bit-for-bit (existing forward tests stay green).

## Canonical definitions

- `DagLayout`: `nodes` (list of node ids), `block[u]` (topic-row indices), `closure(v)` (ancestors+self, sorted by (depth,id)), `subtree(u)` (u + all descendants via children adjacency), `depth(v)` (longest path from root), `n_bg`, `tpn`, `K`.
- `gated_init.py` loop today: `for u in sorted(lay.nodes, key=lambda x:(lay.depth(x),x)): ... anc = [a for a in lay.closure(u) if a not in (u,0)]; seed = list(bg_anchors) + [a for p in anc for a in node_anchors.get(p,[])]`.

---

### Task 1: `DagLayout.descendants`

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/dag_placement.py` (add method to `DagLayout`, next to `subtree`)
- Test: `spark-vi/tests/test_dag_placement.py`

**Interfaces:**
- Produces: `DagLayout.descendants(u) -> list[int]` — proper descendants of u (excludes u), sorted by (depth, id).

- [ ] **Step 1: Write the failing test**

Add to `spark-vi/tests/test_dag_placement.py`:

```python
def test_daglayout_descendants_is_proper_and_mirrors_closure():
    from spark_vi.models.topic.dag_placement import DagLayout
    # DAG: 1 -> 2 -> 4, 1 -> 3, and 4 also a child of 3 (multi-parent diamond)
    lay = DagLayout({2: 1, 3: 1, 4: 2}, n_bg=1, tpn=1)
    lay.parents.setdefault(4, [])
    if 3 not in lay.parents[4]:
        lay.parents[4].append(3)          # 4 has parents {2,3}
    lay.children.setdefault(3, [])
    if 4 not in lay.children[3]:
        lay.children[3].append(4)
    # node 1 (anchor) has every other node as a descendant
    assert set(lay.descendants(1)) == {2, 3, 4}
    # node 4 (leaf) has none
    assert lay.descendants(4) == []
    # descendants excludes u itself and is disjoint from proper ancestors
    assert 2 not in lay.descendants(2)
    assert set(lay.descendants(2)) == {4}
    # sorted by (depth, id)
    d = lay.descendants(1)
    assert d == sorted(d, key=lambda x: (lay.depth(x), x))
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_dag_placement.py::test_daglayout_descendants_is_proper_and_mirrors_closure -q`
Expected: FAIL (`descendants` not defined).

- [ ] **Step 3: Implement `descendants`**

Add to `DagLayout` in `spark-vi/spark_vi/models/topic/dag_placement.py`, immediately after `subtree`:

```python
    def descendants(self, u):
        """Proper descendants of u (every v of which u is a proper ancestor), sorted by
        (depth, id). The mirror of `closure` (ancestors + self): `descendants` reuses the
        children adjacency via `subtree` (u + descendants) and drops u, so there is no
        separate adjacency to keep in sync. A leaf has none; the anchor's descendants are
        every other node. Used by reverse-topological spectral init to deflate a node
        against its already-recovered descendants."""
        return sorted((v for v in self.subtree(u) if v != u),
                      key=lambda v: (self.depth(v), v))
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_dag_placement.py::test_daglayout_descendants_is_proper_and_mirrors_closure -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/dag_placement.py spark-vi/tests/test_dag_placement.py
git commit -m "feat(dag-placement): DagLayout.descendants (proper-descendant mirror of closure)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `topo_order` in the dense spectral init + the order/deflation helper

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/gated_init.py`
- Test: `spark-vi/tests/test_gated_init.py`

**Interfaces:**
- Consumes: `DagLayout.descendants` (Task 1).
- Produces: `TOPO_ORDERS = ("forward", "reverse")`; `_validate_topo_order`; `_node_order_and_relatives(lay, topo_order) -> (order, relatives)`; `spectral_block_aligned_lambda(..., topo_order="forward")`.

- [ ] **Step 1: Write the failing tests**

Add to `spark-vi/tests/test_gated_init.py`:

```python
def test_node_order_and_relatives_forward_vs_reverse():
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_init import _node_order_and_relatives
    lay = DagLayout({2: 1, 3: 2}, n_bg=1, tpn=1)          # chain 1 -> 2 -> 3
    order_f, rel_f = _node_order_and_relatives(lay, "forward")
    order_r, rel_r = _node_order_and_relatives(lay, "reverse")
    # forward: ascending depth (ancestor first); reverse: descending (leaf first)
    assert order_f == [1, 2, 3]
    assert order_r == [3, 2, 1]
    # forward deflates node 3 against its proper ancestors {1,2}; reverse against
    # its proper descendants (none for the leaf)
    assert set(rel_f(3)) == {1, 2} and rel_r(3) == []
    # forward: anchor node 1 has no ancestors; reverse: node 1 deflates against {2,3}
    assert rel_f(1) == [] and set(rel_r(1)) == {2, 3}


def test_topo_order_validation_rejects_unknown():
    import pytest
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_init import spectral_block_aligned_lambda
    lay = DagLayout({2: 1}, n_bg=1, tpn=1)
    with pytest.raises(ValueError):
        spectral_block_aligned_lambda(
            {"train_docs": [[0, 1]], "train_labels": [frozenset()]},
            lay, 3, topo_order="sideways")
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_gated_init.py -q -k "node_order or topo_order_validation"`
Expected: FAIL (`_node_order_and_relatives` / topo_order kwarg not defined).

- [ ] **Step 3: Add the helper + TOPO_ORDERS, thread topo_order into the dense function**

In `spark-vi/spark_vi/models/topic/gated_init.py`, add near `ANCHOR_SCOPES`:

```python
TOPO_ORDERS = ("forward", "reverse")


def _validate_topo_order(topo_order):
    if topo_order not in TOPO_ORDERS:
        raise ValueError(
            f"topo_order must be one of {TOPO_ORDERS}, got {topo_order!r}")


def _node_order_and_relatives(lay, topo_order):
    """(ordered node list, relatives(u)) for the deflation loop.

    forward: nodes ascending depth (ancestors first); each node is deflated against its
    proper ANCESTORS (closure minus self/root) — a node's topic is its increment over its
    ancestors. reverse: nodes descending depth (leaves first); each node is deflated against
    its proper DESCENDANTS (subtree minus self) — leaves claim their full signal first and an
    ancestor's topic is the residual after its descendants. A proper descendant always has
    strictly greater longest-path depth than its ancestor, so descending-depth order
    guarantees every descendant is recovered before the node (the mirror of the forward
    ancestors-first guarantee)."""
    _validate_topo_order(topo_order)
    if topo_order == "forward":
        order = sorted(lay.nodes, key=lambda x: (lay.depth(x), x))

        def relatives(u):
            return [a for a in lay.closure(u) if a not in (u, 0)]
    else:
        order = sorted(lay.nodes, key=lambda x: (lay.depth(x), x), reverse=True)

        def relatives(u):
            return list(lay.descendants(u))     # already proper; 0 is never a descendant

    return order, relatives
```

Then in `spectral_block_aligned_lambda`, add `topo_order: str = "forward"` to the signature (after `anchor_scope`), and replace the node loop header + the `anc = ...` line. Change:

```python
    node_anchors: dict[int, list] = {}
    for u in sorted(lay.nodes, key=lambda x: (lay.depth(x), x)):   # forward topological
        docs_u = [counted[d] for d in range(len(counted)) if u in trains[d]]
        ...
        anc = [a for a in lay.closure(u) if a not in (u, 0)]
        seed = list(bg_anchors) + [a for p in anc for a in node_anchors.get(p, [])]
```

to:

```python
    node_anchors: dict[int, list] = {}
    order, relatives = _node_order_and_relatives(lay, topo_order)
    for u in order:
        docs_u = [counted[d] for d in range(len(counted)) if u in trains[d]]
        ...
        anc = relatives(u)
        seed = list(bg_anchors) + [a for p in anc for a in node_anchors.get(p, [])]
```

(Leave everything else in the function — background step, `find_anchors`, `recover_beta`, floor — unchanged. Keep the existing `_validate_anchor_scope(anchor_scope)` call; `_node_order_and_relatives` calls `_validate_topo_order` itself.)

- [ ] **Step 4: Run to verify they pass**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_gated_init.py -q -k "node_order or topo_order_validation"`
Expected: PASS.

- [ ] **Step 5: Add the semantic-flip test**

Add to `spark-vi/tests/test_gated_init.py`. This proves the deflation DIRECTION changed the recovered topics: a word shared by a parent and its child lands in the parent's block under forward (parent recovered first, child deflated against it) and in the child's block under reverse (child recovered first, parent deflated against it).

```python
def test_reverse_topo_flips_shared_word_block():
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_init import spectral_block_aligned_lambda
    # 1 background topic + parent P=1, child C=2 (chain), tpn=1 -> K=3 blocks: bg=[0], P=[1], C=[2]
    lay = DagLayout({2: 1}, n_bg=1, tpn=1)
    # vocab: 0=bg_word, 1=p_word, 2=c_word, 3=shared_word
    # background docs: bg_word only; P docs: p_word + shared_word; C docs: c_word + shared_word.
    # Repeat tokens so anchor co-occurrence is unambiguous (strengthen the plant, not the assert).
    bg_doc = [0, 0, 0, 0]
    p_doc = [1, 1, 1, 3, 3, 3]
    c_doc = [2, 2, 2, 3, 3, 3]
    docs = [bg_doc] * 4 + [p_doc] * 4 + [c_doc] * 4
    labels = [frozenset()] * 4 + [frozenset({1})] * 4 + [frozenset({2})] * 4
    ds = {"train_docs": docs, "train_labels": labels}
    fwd = spectral_block_aligned_lambda(ds, lay, 4, anchor_scope="frontier",
                                        topo_order="forward")
    rev = spectral_block_aligned_lambda(ds, lay, 4, anchor_scope="frontier",
                                        topo_order="reverse")
    shared = 3
    p_block, c_block = lay.block[1][0], lay.block[2][0]
    # forward: shared word's mass is higher in the PARENT block than the child block
    assert fwd[p_block, shared] > fwd[c_block, shared]
    # reverse: the ordering flips -> shared word's mass is higher in the CHILD block
    assert rev[c_block, shared] > rev[p_block, shared]
```

If anchor recovery is ambiguous on this tiny plant (e.g. both blocks near-equal on the shared word), STRENGTHEN the plant — more repeats, a cleaner private/shared contrast, or an extra filler word — until the direction is unambiguous. Do NOT weaken the strict `>` assertions.

- [ ] **Step 6: Run the flip test + the full gated_init suite (default-unchanged guard)**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_gated_init.py -q`
Expected: all pass, including the pre-existing forward tests (proves `topo_order="forward"` default is unchanged).

- [ ] **Step 7: Commit**

```bash
git add spark-vi/spark_vi/models/topic/gated_init.py spark-vi/tests/test_gated_init.py
git commit -m "feat(gated-init): topo_order knob for dense spectral init (reverse = leaves-first)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `topo_order` in the scalable spectral init + dense/scalable parity

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/gated_init.py`
- Test: `spark-vi/tests/test_gated_init.py`

**Interfaces:**
- Consumes: `_node_order_and_relatives` (Task 2).
- Produces: `scalable_block_aligned_lambda(..., topo_order="forward")`.

- [ ] **Step 1: Write the failing test (parity under reverse)**

Add to `spark-vi/tests/test_gated_init.py`. Mirror the EXISTING forward dense/scalable parity test (find it in the file — reuse its corpus/SparkContext fixture and tolerance) but pass `topo_order="reverse"` to both. Skeleton:

```python
def test_dense_scalable_parity_reverse(spark_context):     # reuse the existing fixture name
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_init import (
        spectral_block_aligned_lambda, scalable_block_aligned_lambda)
    from spark_vi.models.topic.types import GatedBOWDocument
    # Reuse the same tiny labeled corpus the forward parity test builds. Build the dense
    # data_summary {train_docs, train_labels} and the RDD of GatedBOWDocument from it.
    lay = DagLayout({2: 1, 3: 1}, n_bg=2, tpn=1)
    # ... construct docs/labels exactly as the forward parity test does ...
    dense = spectral_block_aligned_lambda(
        {"train_docs": train_docs, "train_labels": train_labels},
        lay, V, anchor_scope="closure", topo_order="reverse")
    rdd = spark_context.parallelize(gated_docs)
    scal = scalable_block_aligned_lambda(rdd, lay, V, anchor_scope="closure",
                                         topo_order="reverse")
    assert np.allclose(dense, scal, atol=<same tol as the forward parity test>)
```

Match the forward parity test's exact corpus, V, fixture, and tolerance — this task only adds the `topo_order="reverse"` axis.

- [ ] **Step 2: Run to verify it fails**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_gated_init.py::test_dense_scalable_parity_reverse -q`
Expected: FAIL (`scalable_block_aligned_lambda` has no `topo_order` kwarg).

- [ ] **Step 3: Thread topo_order into the scalable function**

In `scalable_block_aligned_lambda`, add `topo_order: str = "forward"` to the signature (after `anchor_scope`). Replace the node loop header + the `anc = ...` line exactly as in Task 2:

```python
        node_anchors: dict[int, list] = {}
        order, relatives = _node_order_and_relatives(lay, topo_order)
        for u in order:
            rdd_u = group_rdd.filter(lambda gd, _u=u: _u in gd.groups)
            ...
            anc = [a for a in lay.closure(u) if a not in (u, 0)]   # <-- replace this line
            seed_rows = list(bg_anchors) + [a for p in anc
                                            for a in node_anchors.get(p, [])]
```

Replace the `anc = ...` line with `anc = relatives(u)`. Leave the background step, streaming passes, `find_anchors_projected`, `recover_beta_projected`, floor, and the `try/finally` unpersist unchanged. (The `_node_order_and_relatives` call validates topo_order.)

- [ ] **Step 4: Run to verify it passes + full suite**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_gated_init.py -q`
Expected: all pass (reverse parity + all existing forward tests).

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/gated_init.py spark-vi/tests/test_gated_init.py
git commit -m "feat(gated-init): topo_order in scalable spectral init + dense/scalable parity

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Wire `topo_order` through model -> shim -> driver -> run_experiment

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/gated_lda.py` (`initialize_global` dense path)
- Modify: `spark-vi/spark_vi/mllib/topic/gated_lda.py` (`spectralTopoOrder` Param + both fit paths)
- Modify: `analysis/cloud/dag_placement_cloud.py` (`--spectral-topo-order` arg + estimator + manifest)
- Modify: `scripts/run_experiment.py` (`spectral_topo_order` field in `build_dag_placement_args`)
- Test: `spark-vi/tests/test_gated_lda_shim.py`

**Interfaces:**
- Consumes: `spectral_block_aligned_lambda`/`scalable_block_aligned_lambda` `topo_order` kwarg (Tasks 2-3).
- Produces: shim Param `spectralTopoOrder` (default "forward"); driver `--spectral-topo-order`; manifest field `spectral_topo_order`.

- [ ] **Step 1: Write the failing test (shim Param + dense data_summary carries topo_order)**

Add to `spark-vi/tests/test_gated_lda_shim.py`:

```python
def test_shim_spectral_topo_order_param_default_and_set():
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    est = GatedLDAEstimator()
    assert est.getOrDefault("spectralTopoOrder") == "forward"
    est2 = GatedLDAEstimator(spectralTopoOrder="reverse")
    assert est2.getOrDefault("spectralTopoOrder") == "reverse"
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd spark-vi && ../.venv/bin/python -m pytest tests/test_gated_lda_shim.py::test_shim_spectral_topo_order_param_default_and_set -q`
Expected: FAIL (no `spectralTopoOrder` Param).

- [ ] **Step 3: Add the Param + thread both fit paths (shim)**

In `spark-vi/spark_vi/mllib/topic/gated_lda.py`:

(a) Add the Param next to `anchorScope` (~line 92):

```python
    spectralTopoOrder = Param(Params._dummy(), "spectralTopoOrder",
                              "spectral init deflation order: 'forward' (default; nodes "
                              "ancestors-first, each deflated against its ancestors = "
                              "increment-over-ancestors) or 'reverse' (leaves-first, each "
                              "deflated against its descendants = residual-after-descendants)",
                              typeConverter=TypeConverters.toString)
```

(b) Add `spectralTopoOrder="forward"` to BOTH `@keyword_only` default blocks (the `__init__` and `setParams`, ~lines 176 and 186 — wherever `anchorScope="closure"` appears in the estimator).

(c) Scalable path (~line 307): add the kwarg to the call:

```python
                lam0 = scalable_block_aligned_lambda(
                    rdd, lay, V,
                    d=(sd if sd > 0 else None),
                    seed=(seed or 0),
                    min_doc_freq=int(self.getOrDefault("spectralMinDocFreq")),
                    anchor_scope=self.getOrDefault("anchorScope"),
                    topo_order=self.getOrDefault("spectralTopoOrder"),
                )
```

(d) Dense path (~line 324): add topo_order to the data_summary dict:

```python
                data_summary = {"train_docs": train_docs, "train_labels": train_labels,
                                "anchor_scope": self.getOrDefault("anchorScope"),
                                "topo_order": self.getOrDefault("spectralTopoOrder")}
```

- [ ] **Step 4: Thread the dense strategy read (model)**

In `spark-vi/spark_vi/models/topic/gated_lda.py`, `initialize_global` dense branch (~lines 99-101), read topo_order from data_summary and pass it to the strategy:

```python
            scope = (data_summary or {}).get("anchor_scope", "closure")
            topo = (data_summary or {}).get("topo_order", "forward")
            gp["lambda"] = INIT_STRATEGIES[self.init](
                data_summary, self.lay, self.V, anchor_scope=scope, topo_order=topo)
```

- [ ] **Step 5: Add the driver arg + estimator wiring + manifest**

In `analysis/cloud/dag_placement_cloud.py`:

(a) Arg next to `--anchor-scope` (~line 268):

```python
    p.add_argument("--spectral-topo-order", choices=["forward", "reverse"],
                   default="forward",
                   help="spectral init deflation order (forward=ancestors-first, "
                        "reverse=leaves-first); default forward")
```

(b) Estimator kwarg next to `anchorScope=args.anchor_scope` (~line 328):

```python
                spectralTopoOrder=args.spectral_topo_order,
```

(c) Manifest field next to `"anchor_scope": args.anchor_scope` (~line 413):

```python
                "spectral_topo_order": args.spectral_topo_order,
```

- [ ] **Step 6: Add the run_experiment field**

In `scripts/run_experiment.py`, `build_dag_placement_args` (~line 649, next to `--anchor-scope`):

```python
        "--spectral-topo-order", str(effective.get("spectral_topo_order", "forward")),
```

- [ ] **Step 7: Run the shim test + compile checks**

Run:
```
cd spark-vi && ../.venv/bin/python -m pytest tests/test_gated_lda_shim.py -q
cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno && python -m py_compile analysis/cloud/dag_placement_cloud.py scripts/run_experiment.py
```
Expected: shim tests pass; both files compile.

- [ ] **Step 8: Commit**

```bash
git add spark-vi/spark_vi/models/topic/gated_lda.py spark-vi/spark_vi/mllib/topic/gated_lda.py analysis/cloud/dag_placement_cloud.py scripts/run_experiment.py spark-vi/tests/test_gated_lda_shim.py
git commit -m "feat(gated-lda): thread spectralTopoOrder through shim/driver/run_experiment

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Experiment 0069 (reverse-topo A/B vs 0067)

**Files:**
- Create: `docs/experiments/0069-dag-placement-rare6-1yr-reverse-topo.md`

- [ ] **Step 1: Write the experiment doc**

Clone exp 0067's frontmatter verbatim EXCEPT `id: 69`, a new `slug`, and add `spectral_topo_order: reverse`. Copy the field list from `docs/experiments/0067-dag-placement-rare6-1yr-learned-alpha-deploy-symmetric.md` (same cohort, 1yr lookback, learned alpha fit, symmetric deploy, n_bg 40, frontier anchors, spectral scalable, seed 42, cache_uri). Body: this is the A/B for the reverse-topo spectral init — forward (0067) vs reverse (0069), all else equal. What to read: placement mrr / auc_by_depth (does leaves-first sharpen the leaf topics that placement scores on?), LR + explain-away detection (`make lr-readout ID=69`), and the error-class totals vs 0067. Note reverse is a HYPOTHESIS (spectral init did not beat random on synthetic plants; the gate breaks symmetry), so a null is a real possible outcome.

- [ ] **Step 2: Commit**

```bash
git add docs/experiments/0069-dag-placement-rare6-1yr-reverse-topo.md
git commit -m "exp(0069): reverse-topo spectral init A/B vs 0067 (forward)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

- **Spec coverage:** `descendants` (T1), dense topo_order + helper + validation + semantic-flip (T2), scalable topo_order + parity (T3), full wiring model→shim→driver→run_experiment (T4), exp 0069 (T5). Default-unchanged guard = the existing forward suites staying green (T2 Step 6, T3 Step 4). ✓
- **Placeholder scan:** T3 Step 1 and T5 Step 1 reference the EXISTING forward parity test / 0067 frontmatter rather than duplicating them verbatim, because the implementer must match those exact fixtures/fields (copying a stale copy would risk drift). Every new function/method/arg has complete code. The semantic-flip fixture (T2 Step 5) is complete with an explicit "strengthen the plant, not the assertion" escape hatch.
- **Type consistency:** `topo_order` is a `str` in {forward, reverse} everywhere; `_node_order_and_relatives(lay, topo_order) -> (order, relatives)` used identically in T2/T3; `descendants(u) -> list[int]` defined T1, used by `_node_order_and_relatives` reverse branch. Param name `spectralTopoOrder` (camelCase shim) ↔ arg `--spectral-topo-order` ↔ field `spectral_topo_order` consistent with the `anchorScope`/`--anchor-scope`/`anchor_scope` precedent. ✓
