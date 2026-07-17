# Scalable Gated Spectral Init Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A distributed, DAG-topological random-projection spectral init for `GatedOnlineLDA`, so `init="spectral"` runs at rare6 scale (K=180 / V=10000) without the dense driver-side V×V co-occurrence that currently stalls the fit.

**Architecture:** Reuse the existing projected primitives in `spectral_init_scalable.py`. One `projected_cooccurrence_rdd` pass with each doc's `groups = closure(frontier) \ {0}` yields the projected image of every node's per-node co-occurrence `Q_u`; a driver-side forward-topological loop then does ancestor-deflated anchor recovery (mirroring the dense `spectral_block_aligned_lambda`) over those sketches. The shim precomputes the λ on the RDD and passes it via `data_summary` (STM pattern); a new `spectralMethod` param routes dense-vs-scalable.

**Tech Stack:** Python, NumPy, SciPy (NNLS, already used), PySpark RDD (mapPartitions/treeReduce, already used), pytest.

## Global Constraints

- Engine layer (`spark_vi`) stays integer-id agnostic — no OMOP/concept-id knowledge. `lay` is a `DagLayout` of int node ids; the corpus is `GatedBOWDocument` (int token ids + int frontier).
- Commit trailer EXACTLY: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- No LaTeX in prose/docstrings; Unicode Greek only (α, β, λ, η, Σ).
- Cite literature for any method/constant (Arora et al. 2013 anchor words; Johnson–Lindenstrauss; ADR 0032 for the projected foundation).
- Branch `case-finding` does NOT auto-push; push only when the user asks.
- Exploratory research code (no prod). Do NOT gold-plate: structural tests only, defaults ship untuned.
- `projected_cooccurrence_rdd`, `find_anchors_projected`, `recover_beta_projected` are REUSED UNMODIFIED (shared with the STM path).
- Test honesty: no threshold-loosening to make a test pass; xfail with a reason if a real gap is found.

---

### Task 1: `scalable_block_aligned_lambda` algorithm

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/gated_init.py`
- Test: `spark-vi/tests/test_gated_init.py` (exists)

**Interfaces:**
- Consumes: `projected_cooccurrence_rdd`, `find_anchors_projected`, `recover_beta_projected`, `default_projection_dim` (from `spark_vi.models.topic.spectral_init_scalable`); `DagLayout` (`lay.nodes`, `lay.n_bg`, `lay.tpn`, `lay.K`, `lay.depth`, `lay.closure`, `lay.block`); `GatedBOWDocument` (`.indices`, `.counts`, `.frontier`).
- Produces: `scalable_block_aligned_lambda(rdd, lay, V, *, d=None, seed=0, min_doc_freq=5, scale=200.0) -> np.ndarray` of shape `(lay.K, V)` — a λ seed on the same `beta*scale` contract as `spectral_block_aligned_lambda`. Module-level helper classes `_GroupDoc`, `_NodeGroups`.

- [ ] **Step 1: Write the failing test**

Add to `spark-vi/tests/test_gated_init.py` (reuse the `spark` fixture from `conftest.py`):

```python
def test_scalable_block_aligned_lambda_is_block_aligned_and_deflated(spark):
    # Parent node 1 (tokens 0,1 shared across its subtree), child node 2 under 1
    # (tokens 2,3 child-specific), background tokens 8,9. Each node's block must
    # load its own anchor tokens; the child block must differ from the parent
    # block (forward-topological ancestor deflation took effect).
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.types import GatedBOWDocument
    from spark_vi.models.topic.gated_init import scalable_block_aligned_lambda

    lay = DagLayout({1: 0, 2: 1}, n_bg=2, tpn=1)   # K = 2 + 2*1 = 4
    V = 10

    def doc(idx, frontier):
        idx = sorted(idx)
        counts = np.ones(len(idx), dtype=np.float64)
        return GatedBOWDocument(indices=np.asarray(idx, dtype=np.int32),
                                counts=counts, length=len(idx),
                                frontier=frozenset(frontier))

    rows = []
    for _ in range(30):
        rows.append(doc([8, 9, 0, 1], []))            # background-only
        rows.append(doc([0, 1, 8], [1]))              # node 1: shared tokens 0,1
        rows.append(doc([2, 3, 0, 8], [2]))           # node 2: child tokens 2,3 (+ inherits 0)
    rdd = spark.sparkContext.parallelize(rows, 3)

    lam = scalable_block_aligned_lambda(rdd, lay, V, seed=0, min_doc_freq=1)
    assert lam.shape == (lay.K, V)
    # node 1 block = topic index lay.block[1][0]; node 2 = lay.block[2][0]
    beta1 = lam[lay.block[1][0]]
    beta2 = lam[lay.block[2][0]]
    # child block emphasizes its own tokens 2/3 more than the parent block does
    assert beta2[2] + beta2[3] > beta1[2] + beta1[3]
    # deflation: the two node blocks are not identical rows
    assert not np.allclose(beta1, beta2)


def test_scalable_block_aligned_lambda_deterministic(spark):
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.types import GatedBOWDocument
    from spark_vi.models.topic.gated_init import scalable_block_aligned_lambda

    lay = DagLayout({1: 0, 2: 0}, n_bg=2, tpn=1)
    V = 8

    def doc(idx, frontier):
        return GatedBOWDocument(indices=np.asarray(sorted(idx), dtype=np.int32),
                                counts=np.ones(len(idx)), length=len(idx),
                                frontier=frozenset(frontier))
    rows = [doc([0, 1, 6], [1]) for _ in range(10)] + \
           [doc([2, 3, 6], [2]) for _ in range(10)]
    rdd = spark.sparkContext.parallelize(rows, 2)
    a = scalable_block_aligned_lambda(rdd, lay, V, seed=7, min_doc_freq=1)
    b = scalable_block_aligned_lambda(rdd, lay, V, seed=7, min_doc_freq=1)
    assert np.allclose(a, b)


def test_scalable_block_aligned_lambda_zero_doc_node_stays_at_floor(spark):
    # A node with no training docs keeps its block at the 1e-9 floor (times scale),
    # warns, and produces no NaN.
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.types import GatedBOWDocument
    from spark_vi.models.topic.gated_init import scalable_block_aligned_lambda

    lay = DagLayout({1: 0, 2: 0}, n_bg=2, tpn=1)     # node 2 will get no docs
    V = 8

    def doc(idx, frontier):
        return GatedBOWDocument(indices=np.asarray(sorted(idx), dtype=np.int32),
                                counts=np.ones(len(idx)), length=len(idx),
                                frontier=frozenset(frontier))
    rows = [doc([0, 1, 6], [1]) for _ in range(10)]  # only node 1 attested
    rdd = spark.sparkContext.parallelize(rows, 2)
    lam = scalable_block_aligned_lambda(rdd, lay, V, seed=0, min_doc_freq=1)
    assert not np.isnan(lam).any()
    scale = 200.0
    assert np.allclose(lam[lay.block[2][0]], 1e-9 * scale)   # untouched floor
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd spark-vi && python -m pytest tests/test_gated_init.py -k scalable -q`
Expected: FAIL with `ImportError: cannot import name 'scalable_block_aligned_lambda'`.

- [ ] **Step 3: Implement `scalable_block_aligned_lambda`**

Append to `spark-vi/spark_vi/models/topic/gated_init.py` (keep the existing dense function and `INIT_STRATEGIES` unchanged):

```python
class _GroupDoc:
    """Minimal doc view for projected_cooccurrence_rdd: token support/counts plus
    the DAG node groups this doc trains (its frontier closure, root 0 dropped)."""
    __slots__ = ("indices", "counts", "groups")

    def __init__(self, indices, counts, groups):
        self.indices = indices
        self.counts = counts
        self.groups = groups


class _NodeGroups:
    """Minimal partition view for projected_cooccurrence_rdd: it reads only
    `.groups` (the set of node ids that get a per-group sketch)."""
    __slots__ = ("groups",)

    def __init__(self, groups):
        self.groups = groups


def scalable_block_aligned_lambda(rdd, lay, V, *, d: int | None = None,
                                  seed: int = 0, min_doc_freq: int = 5,
                                  scale: float = 200.0) -> np.ndarray:
    """Distributed random-projection analogue of `spectral_block_aligned_lambda`.

    `rdd` is an RDD of GatedBOWDocument. Runs ONE distributed projected-
    co-occurrence pass (never a driver V×V matrix, ADR 0032) with each doc's
    groups = closure(frontier) minus root 0, so `group_QR[u]` is the projected
    image of the dense per-node co-occurrence Q_u. Then a driver-side FORWARD-
    TOPOLOGICAL loop (ancestors first by lay.depth) recovers each node's block by
    anchor-word spectral recovery (Arora et al. 2013) deflated against background
    + already-recovered proper-ancestor anchors, exactly as the dense path — the
    projection preserves the residual-norm geometry the greedy anchor search
    needs (Johnson–Lindenstrauss). Returns a (K, V) λ = block-aligned β * scale,
    the same contract as the dense function (a drop-in seed)."""
    from spark_vi.models.topic.spectral_init_scalable import (
        projected_cooccurrence_rdd, find_anchors_projected,
        recover_beta_projected, default_projection_dim,
    )
    if d is None:
        d = default_projection_dim(lay.K, V)

    lay_b = rdd.context.broadcast(lay)

    def _to_group(doc, _lay=lay_b):
        L = _lay.value
        groups = set()
        for f in doc.frontier:
            for u in L.closure(f):
                if u != 0:
                    groups.add(u)
        return _GroupDoc(doc.indices, doc.counts, tuple(groups))

    res = projected_cooccurrence_rdd(
        rdd.map(_to_group), _NodeGroups(tuple(lay.nodes)), V, d, seed
    )

    beta = np.zeros((lay.K, V), dtype=np.float64)

    # Step 1: background block on the pooled sketch.
    bg_anchors = find_anchors_projected(
        res.pooled_QR, res.p_w, res.df_w, lay.n_bg, min_doc_freq=min_doc_freq)
    bg_beta = recover_beta_projected(res.pooled_QR, res.p_w, bg_anchors)
    for i in range(min(lay.n_bg, bg_beta.shape[0])):
        beta[i] = bg_beta[i]

    # Step 2: each node, ancestors first, deflated vs bg + ancestor anchors.
    node_anchors: dict[int, list] = {}
    for u in sorted(lay.nodes, key=lambda x: (lay.depth(x), x)):
        if int(res.group_df_w[u].sum()) == 0:
            logger.warning(
                "scalable_block_aligned_lambda: node %s has zero training docs; "
                "its block stays at the 1e-9 floor (uninitialized).", u)
            continue
        anc = [a for a in lay.closure(u) if a not in (u, 0)]
        seed_rows = list(bg_anchors) + [a for p in anc for a in node_anchors.get(p, [])]
        fg_anchors = find_anchors_projected(
            res.group_QR[u], res.group_p_w[u], res.group_df_w[u], lay.tpn,
            seed_rows=seed_rows, min_doc_freq=min_doc_freq)
        if not fg_anchors:
            logger.warning(
                "scalable_block_aligned_lambda: node %s found no anchors "
                "(sparse/degenerate sketch); its block stays at the 1e-9 floor.", u)
            continue
        node_anchors[u] = list(fg_anchors)
        combined_beta = recover_beta_projected(
            res.group_QR[u], res.group_p_w[u], list(seed_rows) + list(fg_anchors))
        fg_beta = combined_beta[len(seed_rows):]
        for j, idx in enumerate(lay.block[u]):
            if j < fg_beta.shape[0]:
                beta[idx] = fg_beta[j]

    beta = beta + 1e-9                                   # strictly positive λ
    return beta * float(scale)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd spark-vi && python -m pytest tests/test_gated_init.py -k scalable -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/gated_init.py spark-vi/tests/test_gated_init.py
git commit -m "feat(gated-init): scalable_block_aligned_lambda distributed projected seed

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `initialize_global` accepts a precomputed λ

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/gated_lda.py` (method `GatedOnlineLDA.initialize_global`, ~lines 54-71)
- Test: `spark-vi/tests/test_gated_lda.py`

**Interfaces:**
- Consumes: `data_summary` dict optionally carrying key `"spectral_lambda"` (a `(K, V)` np.ndarray).
- Produces: when `spectral_lambda` is present, `initialize_global` returns `global_params` whose `lambda` IS that array (no dense strategy run). Dense path (no key) unchanged.

- [ ] **Step 1: Write the failing test**

Add to `spark-vi/tests/test_gated_lda.py`:

```python
def test_initialize_global_uses_precomputed_spectral_lambda():
    # When data_summary carries a precomputed (K,V) 'spectral_lambda', the model
    # seeds lambda from it directly (the scalable path) instead of running a
    # dense INIT_STRATEGIES strategy.
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA

    lay = DagLayout({1: 0, 2: 0}, n_bg=2, tpn=1)      # K = 4
    V = 6
    m = GatedOnlineLDA(lay, V, init="spectral")
    planted = np.arange(lay.K * V, dtype=np.float64).reshape(lay.K, V) + 1.0
    gp = m.initialize_global({"spectral_lambda": planted})
    assert np.allclose(gp["lambda"], planted)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd spark-vi && python -m pytest tests/test_gated_lda.py -k precomputed_spectral -q`
Expected: FAIL — the current code calls `INIT_STRATEGIES["spectral"](data_summary, ...)`, which raises `ValueError` (missing `train_docs`).

- [ ] **Step 3: Implement the branch**

In `spark-vi/spark_vi/models/topic/gated_lda.py`, replace the body of `initialize_global` after the `if self.init == "random"` guard:

```python
        from spark_vi.models.topic.gated_init import INIT_STRATEGIES
        if self.init not in INIT_STRATEGIES:
            raise ValueError(
                f"unknown init strategy {self.init!r}; "
                f"known: {['random'] + sorted(INIT_STRATEGIES)}"
            )
        gp = super().initialize_global(data_summary)
        # Scalable path: the shim precomputed the (K,V) lambda on the RDD and
        # handed it over via data_summary (mirrors the STM shim's spectral_beta);
        # use it directly. Dense path: run the collect-to-driver strategy.
        if data_summary is not None and "spectral_lambda" in data_summary:
            gp["lambda"] = np.asarray(data_summary["spectral_lambda"], dtype=np.float64)
        else:
            gp["lambda"] = INIT_STRATEGIES[self.init](data_summary, self.lay, self.V)
        return gp
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd spark-vi && python -m pytest tests/test_gated_lda.py -k precomputed_spectral -q`
Expected: PASS. Also run the existing dense seed test to confirm no regression:
`python -m pytest tests/test_gated_lda_shim.py -k spectral_fits_and_seeds -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/gated_lda.py spark-vi/tests/test_gated_lda.py
git commit -m "feat(gated-lda): initialize_global accepts precomputed spectral_lambda

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Shim routing (`spectralMethod`) + scalable precompute

**Files:**
- Modify: `spark-vi/spark_vi/mllib/topic/gated_lda.py` (param block ~lines 30-103; `_fit` ~lines 140-219)
- Test: `spark-vi/tests/test_gated_lda_shim.py`

**Interfaces:**
- Consumes: `scalable_block_aligned_lambda` (Task 1); `initialize_global` precomputed-λ path (Task 2); `resolve_spectral_method(method, vocab_size, threshold)` from `spark_vi.mllib.topic.stm` (reused, generic).
- Produces: `GatedLDAEstimator` params `spectralMethod` (default `"auto"`), `spectralD` (default `0` = auto), `spectralMinDocFreq` (default `5`). `init="spectral"` routes: `auto` → dense if `V < spectralMaxVocab` else scalable; explicit `dense`/`scalable` pass through. The old `NotImplementedError` V-guard is removed.

- [ ] **Step 1: Write the failing tests**

Add to `spark-vi/tests/test_gated_lda_shim.py`, and DELETE the obsolete `test_gated_shim_spectral_vocab_guard_raises` (the guard it asserts is being removed):

```python
def test_gated_shim_spectral_method_param_defaults_and_settable():
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    est = GatedLDAEstimator(parent={1: 0, 2: 0})
    assert est.getOrDefault("spectralMethod") == "auto"
    est2 = GatedLDAEstimator(parent={1: 0, 2: 0}, spectralMethod="scalable")
    assert est2.getOrDefault("spectralMethod") == "scalable"


def test_gated_shim_scalable_spectral_fits_and_seeds_lambda(spark):
    # Forcing spectralMethod='scalable' at small V routes through the distributed
    # projected init and fits; the resulting lambda differs from a random-init fit
    # on the same corpus (the scalable seed took effect), and no dense V×V is built.
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    from pyspark.ml.linalg import SparseVector
    import numpy as np
    rows = []
    for _ in range(30):
        rows.append((SparseVector(8, [0, 1, 6], [3.0, 2.0, 1.0]), [1]))
        rows.append((SparseVector(8, [2, 3, 6], [3.0, 2.0, 1.0]), [2]))
    df = spark.createDataFrame(rows, ["features", "frontier"])
    parent = {1: 0, 2: 0}
    m_rand = GatedLDAEstimator(parent=parent, nBg=2, tpn=1, init="random",
                               maxIter=2, seed=0).fit(df)
    m_scal = GatedLDAEstimator(parent=parent, nBg=2, tpn=1, init="spectral",
                               spectralMethod="scalable", spectralMinDocFreq=1,
                               maxIter=2, seed=0).fit(df)
    lam_r = m_rand.result.global_params["lambda"]
    lam_s = m_scal.result.global_params["lambda"]
    assert lam_r.shape == lam_s.shape == (2 + len(parent), 8)
    assert not np.allclose(lam_r, lam_s)


def test_gated_shim_spectral_auto_routes_scalable_above_threshold(spark):
    # spectralMethod='auto' with a tiny spectralMaxVocab threshold routes a
    # V>=threshold corpus to the scalable path and fits (no NotImplementedError).
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    from pyspark.ml.linalg import SparseVector
    rows = []
    for _ in range(20):
        rows.append((SparseVector(8, [0, 1, 6], [2.0, 1.0, 1.0]), [1]))
        rows.append((SparseVector(8, [2, 3, 6], [2.0, 1.0, 1.0]), [2]))
    df = spark.createDataFrame(rows, ["features", "frontier"])
    m = GatedLDAEstimator(parent={1: 0, 2: 0}, nBg=2, tpn=1, init="spectral",
                          spectralMethod="auto", spectralMaxVocab=4,
                          spectralMinDocFreq=1, maxIter=2, seed=0).fit(df)   # V=8 >= 4
    assert m.result.global_params["lambda"].shape[0] == 4
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd spark-vi && python -m pytest tests/test_gated_lda_shim.py -k "spectral_method or scalable_spectral or auto_routes" -q`
Expected: FAIL — `spectralMethod` param does not exist yet.

- [ ] **Step 3: Add the params**

In `spark-vi/spark_vi/mllib/topic/gated_lda.py`, after the `spectralMaxVocab` Param definition (~line 78) add:

```python
    spectralMethod = Param(Params._dummy(), "spectralMethod",
                           "spectral routing: 'auto'|'dense'|'scalable' "
                           "(auto -> dense if V < spectralMaxVocab else scalable)")
    spectralD = Param(Params._dummy(), "spectralD",
                      "random-projection dim for scalable init (0 = auto: "
                      "min(V, max(K, 1000)))")
    spectralMinDocFreq = Param(Params._dummy(), "spectralMinDocFreq",
                               "min within-group document frequency for a scalable "
                               "anchor candidate")
```

Add matching defaults to BOTH `__init__` signature/kwargs and `setParams` (they currently carry `init="random", spectralMaxVocab=8000, ...`). In each, add `spectralMethod="auto", spectralD=0, spectralMinDocFreq=5`. Ensure they appear in the `self._set(...)` / `kwargs` plumbing exactly like the sibling params.

- [ ] **Step 4: Remove the guard and add routing in `_fit`**

Delete the `NotImplementedError` block (current ~lines 150-159). Replace the non-random `data_summary` build (current ~lines 201-213) with:

```python
        # Non-random init: seed lambda from anchor-word spectral recovery. init
        # "spectral" routes dense (collect the corpus to the driver, exact V×V,
        # validated small-V default) vs scalable (distributed random-projection
        # sketch, ADR 0032 — the gated analogue), by spectralMethod. Passed to
        # initialize_global via data_summary; dense hands {train_docs,train_labels},
        # scalable hands a precomputed {spectral_lambda}.
        data_summary = None
        if init != "random":
            from spark_vi.mllib.topic.stm import resolve_spectral_method
            resolved = resolve_spectral_method(
                self.getOrDefault("spectralMethod"), V,
                threshold=self.getOrDefault("spectralMaxVocab"))
            if resolved == "scalable":
                from spark_vi.models.topic.gated_init import (
                    scalable_block_aligned_lambda,
                )
                sd = int(self.getOrDefault("spectralD"))
                lam0 = scalable_block_aligned_lambda(
                    rdd, lay, V,
                    d=(sd if sd > 0 else None),
                    seed=(seed or 0),
                    min_doc_freq=int(self.getOrDefault("spectralMinDocFreq")),
                )
                data_summary = {"spectral_lambda": lam0}
            else:  # dense — collect-to-driver exact path
                collected = dataset.select(features_col, label_col).collect()
                train_docs, train_labels = [], []
                for r in collected:
                    bow = _vector_to_bow_document(r[0])
                    train_docs.append(np.repeat(bow.indices, bow.counts.astype(int)))
                    train_labels.append(frozenset(int(x) for x in (r[1] or [])))
                data_summary = {"train_docs": train_docs, "train_labels": train_labels}
```

(The `rdd` of `GatedBOWDocument` is already built and persisted just above this block — the scalable path consumes it directly.)

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd spark-vi && python -m pytest tests/test_gated_lda_shim.py -q`
Expected: PASS (all, including the untouched dense `test_gated_shim_spectral_fits_and_seeds_lambda` which stays dense because V=6 < spectralMaxVocab=1000).

- [ ] **Step 6: Commit**

```bash
git add spark-vi/spark_vi/mllib/topic/gated_lda.py spark-vi/tests/test_gated_lda_shim.py
git commit -m "feat(gated-shim): spectralMethod routing + scalable precompute path

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Driver + config plumbing; route exp 0059 scalable

**Files:**
- Modify: `analysis/cloud/dag_placement_cloud.py` (arg parse, estimator construction, manifest)
- Modify: `scripts/run_experiment.py` (`build_dag_placement_args`)
- Modify: `experiments/defaults/_base.yaml`
- Modify: `docs/experiments/0059-dag-placement-rare6-sym-stripboth-spectral.md`
- Test: `scripts/tests/test_dag_placement_config.py`

**Interfaces:**
- Consumes: `spectralMethod` shim param (Task 3); `resolve_spectral_method` (for manifest provenance).
- Produces: `--spectral-method` CLI flag on the driver; `spectral_method` config key (default `auto` in `_base.yaml`, `scalable` in exp 0059); manifest records requested + resolved method.

- [ ] **Step 1: Write the failing test**

Add to `scripts/tests/test_dag_placement_config.py`:

```python
def test_spectral_method_config_and_argv(monkeypatch):
    # _base default is 'auto'; exp 0059 forces 'scalable'; both emit --spectral-method.
    mod = importlib.import_module("run_experiment")
    monkeypatch.setattr(mod, "_require_workspace_env", lambda: ("p.d", "bill"))
    eff59 = _load_effective(
        mod, "docs/experiments/0059-dag-placement-rare6-sym-stripboth-spectral.md")
    assert eff59["spectral_method"] == "scalable"
    args = mod.build_dag_placement_args(eff59, "/out")
    assert args[args.index("--spectral-method") + 1] == "scalable"
    # a diabetes exp inherits the _base default 'auto'
    eff52 = _load_effective(mod, "docs/experiments/0052-dag-placement-diabetes-random.md")
    assert eff52["spectral_method"] == "auto"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest scripts/tests/test_dag_placement_config.py -k spectral_method -q`
Expected: FAIL — `spectral_method` key absent / `--spectral-method` not built.

- [ ] **Step 3: Wire config + argv + driver**

1. `experiments/defaults/_base.yaml` — in the dag_placement block, add:
```yaml
spectral_method: auto      # auto|dense|scalable; auto routes by vocab vs spectral_max_vocab
```

2. `docs/experiments/0059-dag-placement-rare6-sym-stripboth-spectral.md` frontmatter — add `spectral_method: scalable` (force the scalable path for the diagnostic; independent of the spectral_max_vocab threshold).

3. `scripts/run_experiment.py` `build_dag_placement_args` — append, alongside the existing `--init` emission:
```python
    args += ["--spectral-method", str(eff["spectral_method"])]
```

4. `analysis/cloud/dag_placement_cloud.py`:
   - Add argparse: `parser.add_argument("--spectral-method", default="auto", choices=["auto", "dense", "scalable"])`.
   - Pass to the estimator: add `spectralMethod=args.spectral_method` to the `GatedLDAEstimator(...)` construction.
   - Manifest provenance: compute and record the resolved method next to the other fit metadata:
```python
     from spark_vi.mllib.topic.stm import resolve_spectral_method
     resolved_spectral = resolve_spectral_method(args.spectral_method, V, threshold=spectral_max_vocab)
     manifest["spectral_method_requested"] = args.spectral_method
     manifest["spectral_method_resolved"] = resolved_spectral
```
   (Use the driver's existing `V` and `spectral_max_vocab` locals; if the driver reads `spectral_max_vocab` from args, reuse that. Also add a one-line log: `log(f">>> spectral init: requested={args.spectral_method} resolved={resolved_spectral}")` near the fit banner.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest scripts/tests/test_dag_placement_config.py -q`
Expected: PASS (all, including the earlier rare6/2x2 tests).

- [ ] **Step 5: Commit**

```bash
git add analysis/cloud/dag_placement_cloud.py scripts/run_experiment.py experiments/defaults/_base.yaml docs/experiments/0059-dag-placement-rare6-sym-stripboth-spectral.md scripts/tests/test_dag_placement_config.py
git commit -m "feat(dag-placement-cloud): --spectral-method routing; exp 0059 scalable

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:** Task 1 = the algorithm (component 1); Task 2 = `initialize_global` precomputed-λ (component 2); Task 3 = shim routing + scalable precompute (component 3); Task 4 = driver/config + 0059 (component 4). All four spec components covered; validation bullets map to Task 1 tests (block-aligned, deflation, determinism, zero-doc) + Task 2 (precomputed λ) + Task 3 (routing/scalable fit). `projected_cooccurrence_rdd` et al. reused unmodified (constraint honored).

**Placeholder scan:** all code steps carry concrete code; no TBD/TODO.

**Type consistency:** `scalable_block_aligned_lambda(rdd, lay, V, *, d, seed, min_doc_freq, scale)` defined in Task 1, called with those kwargs in Task 3. `data_summary["spectral_lambda"]` produced in Task 3, consumed in Task 2. `resolve_spectral_method(method, vocab_size, threshold)` — signature matches STM's (positional method, vocab_size; keyword threshold). Param names `spectralMethod`/`spectralD`/`spectralMinDocFreq` consistent across Tasks 3-4.

**Open item for the driver task:** `dag_placement_cloud.py`'s exact local names for `V` and the spectral-max-vocab value must be confirmed by the implementer when editing (Step 3.4 says reuse the existing locals); if the driver does not currently thread `spectral_max_vocab`, add it from args alongside `--spectral-method`.
