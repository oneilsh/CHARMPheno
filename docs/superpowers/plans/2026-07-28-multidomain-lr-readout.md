# Multidomain LR / per-domain placement-lift readout — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A post-hoc likelihood-ratio placement readout for the multi-domain gated fits that scores held-out case-finding and decomposes the placement lift by domain (condition/drug/observation).

**Architecture:** The scorer is the per-domain sum of the existing single-domain `lr_placement_scores` (additive because every domain's λ_m shares the same K topics + `DagLayout`). The fit persists its held-out test set (per-domain BOWs + node-affinities + `parent_int`) into the run directory, so the readout is self-contained — it loads the run dir, builds per-domain BOW matrices, and emits a per-rare-disease × domain-subset AUC table plus a θ-mass baseline. No BigQuery, no bundle cache.

**Tech Stack:** Python 3, NumPy/SciPy, PySpark (parquet I/O in the driver), spark_vi engine, pytest.

## Global Constraints

- **The readout loads the test set from the RUN DIR** (`test_docs/` + `test_affinities/` parquet), written by the fit — no BigQuery, no bundle cache, no content-hash key (choice C). There is NO readout for a pre-persistence artifact.
- **Reuse, don't reimplement:** the multi-domain score is the per-domain SUM of `lr_placement_scores`; the AUC is the existing `_auc`. No new scoring math.
- **Additivity:** `lr_placement_scores_multidomain(all domains)` == elementwise sum of the per-domain single-domain scores; a single-domain call reproduces `lr_placement_scores` exactly.
- **Per-disease detection = max over `subtree(d)`** — d and its DESCENDANTS ∩ `lay.nodes` (invert `parent_int` to a children map); positive = doc's frontier ∩ `subtree(d)` ≠ ∅. This is the DESCENDANT subtree, NOT the ancestral gating "closure".
- **`is_fg` = frontier ∩ `lay.nodes`** (root 0 excluded) — LR and θ-mass score the SAME positive set.
- **Domain 0 is conditions;** feature columns `features_0 … features_{N-1}`; dict-λ `{m: (K, V_m)}`. Per-domain vocab size V_m = `lam_dict[m].shape[1]`.
- **Layer:** the scorer (`spark_vi`) is integer-id and domain-neutral (dict of BOWs + dict-λ + `lay`); the readout driver (`analysis/cloud`) is the clinical layer (rare6 anchors via `disease`, domain names condition/drug/observation).
- **Requires re-fitting 0071/0072 once** so their artifacts contain the persisted test set (user-run, cluster).
- spark_vi tests: `cd spark-vi && poetry run pytest tests/...`. driver/analysis tests: `cd analysis/cloud && poetry run pytest tests/...` (or the in-repo `./.venv/bin/python -m pytest` if `poetry run` hits a stale venv). Bash `timeout` is MILLISECONDS.

---

## File Structure

- `spark-vi/spark_vi/models/topic/dag_placement.py` — add `lr_placement_scores_multidomain` + `lr_auc_sweep_multidomain` (Task 1).
- `spark-vi/tests/test_lr_multidomain.py` — NEW: scorer unit tests (Task 1).
- `analysis/cloud/multidomain_cloud.py` — fit-time test-set persistence + `parent_int` in manifest (Task 2).
- `analysis/cloud/multidomain_lr_readout.py` — NEW: the readout driver + pure helpers (Task 3).
- `analysis/cloud/tests/test_multidomain_lr_readout.py` — NEW: pure-helper + parse_args tests (Task 3).
- `analysis/cloud/Makefile` — `multidomain-lr-readout ID=N` target (Task 3).

---

### Task 1: Multi-domain LR scorer

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/dag_placement.py` (add two functions after `lr_auc_sweep`)
- Test: `spark-vi/tests/test_lr_multidomain.py` (NEW)

**Interfaces:**
- Consumes: `lr_placement_scores(bow, lam, lay, *, alpha, background=None, epsilon=1e-9, count_mode="raw", length_normalize=False) -> [n_docs × n_nodes]` (columns in `lay.nodes` order); `_auc(scores, y)`; `DagLayout`.
- Produces:
  - `lr_placement_scores_multidomain(bows: dict, lam_dict: dict, lay, *, alpha, domains=None, backgrounds=None, epsilon=1e-9, count_mode="raw") -> np.ndarray [n_docs × n_nodes]`
  - `lr_auc_sweep_multidomain(bows, lam_dict, lay, is_fg, *, alpha_grid, domains=None, backgrounds=None, count_mode="raw") -> {float: float}`

- [ ] **Step 1: Write the failing tests**

Create `spark-vi/tests/test_lr_multidomain.py`:

```python
import numpy as np


def _tiny():
    """A 2-node DAG layout + two per-domain lambdas + two BOWs sharing the K topics."""
    from spark_vi.models.topic.dag_placement import DagLayout
    lay = DagLayout({1: 0, 2: 0}, n_bg=2, tpn=1)          # K = 2 bg + 2 nodes = 4
    rng = np.random.default_rng(0)
    K = lay.K
    lam0 = rng.random((K, 6)) + 0.1                        # domain 0: V=6
    lam1 = rng.random((K, 4)) + 0.1                        # domain 1: V=4
    bow0 = rng.integers(0, 3, size=(5, 6)).astype(float)   # 5 docs
    bow1 = rng.integers(0, 3, size=(5, 4)).astype(float)
    return lay, {0: lam0, 1: lam1}, {0: bow0, 1: bow1}


def test_multidomain_score_is_the_per_domain_sum():
    from spark_vi.models.topic.dag_placement import (
        lr_placement_scores, lr_placement_scores_multidomain)
    lay, lam, bows = _tiny()
    s_multi = lr_placement_scores_multidomain(bows, lam, lay, alpha=1.0)
    s0 = lr_placement_scores(bows[0], lam[0], lay, alpha=1.0)
    s1 = lr_placement_scores(bows[1], lam[1], lay, alpha=1.0)
    assert np.allclose(s_multi, s0 + s1)                   # additivity


def test_single_domain_ties_out_to_lr_placement_scores():
    from spark_vi.models.topic.dag_placement import (
        lr_placement_scores, lr_placement_scores_multidomain)
    lay, lam, bows = _tiny()
    s_multi_one = lr_placement_scores_multidomain({0: bows[0]}, {0: lam[0]}, lay, alpha=1.0)
    assert np.allclose(s_multi_one, lr_placement_scores(bows[0], lam[0], lay, alpha=1.0))


def test_domain_subset_selects_that_domain():
    from spark_vi.models.topic.dag_placement import (
        lr_placement_scores, lr_placement_scores_multidomain)
    lay, lam, bows = _tiny()
    s_sub = lr_placement_scores_multidomain(bows, lam, lay, alpha=1.0, domains=[1])
    assert np.allclose(s_sub, lr_placement_scores(bows[1], lam[1], lay, alpha=1.0))
    # leave-one-out: all minus dropped == the remaining domain
    s_drop0 = lr_placement_scores_multidomain(bows, lam, lay, alpha=1.0, domains=[1])
    s_all = lr_placement_scores_multidomain(bows, lam, lay, alpha=1.0)
    s0 = lr_placement_scores(bows[0], lam[0], lay, alpha=1.0)
    assert np.allclose(s_all - s0, s_drop0)


def test_auc_sweep_multidomain_matches_manual_auc():
    from spark_vi.models.topic.dag_placement import (
        _auc, lr_placement_scores_multidomain, lr_auc_sweep_multidomain)
    lay, lam, bows = _tiny()
    is_fg = np.array([1, 0, 1, 0, 1])
    sweep = lr_auc_sweep_multidomain(bows, lam, lay, is_fg, alpha_grid=[1.0, 10.0])
    for a in (1.0, 10.0):
        s = lr_placement_scores_multidomain(bows, lam, lay, alpha=a)
        assert np.isclose(sweep[a], _auc(s.max(axis=1), is_fg))
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd spark-vi && poetry run pytest tests/test_lr_multidomain.py -q`
Expected: FAIL (`lr_placement_scores_multidomain` undefined).

- [ ] **Step 3: Implement**

In `dag_placement.py`, immediately after `lr_auc_sweep` (before `_auc`):

```python
def lr_placement_scores_multidomain(bows, lam_dict, lay, *, alpha, domains=None,
                                    backgrounds=None, epsilon=1e-9, count_mode="raw"):
    """Multi-domain per-node LR placement score: the per-domain SUM of the
    single-domain `lr_placement_scores`. Every domain's lam_dict[m] shares the
    same K topics and the same `lay`, so the node-placement log-likelihood-ratio
    is additive across domains -- and a domain SUBSET is the per-domain
    decomposition (cond-only, leave-one-out, ...).

    bows: {m: [n_docs x V_m]} per-domain BOW matrices (dense or scipy.sparse).
    lam_dict: {m: [K x V_m]} the fitted per-domain topic-word counts.
    domains: iterable of domain keys to include (None = all keys of `bows`).
    backgrounds: {m: base_rate} per domain (None entry -> derived from bows[m],
        matching lr_placement_scores). Returns [n_docs x n_nodes], lay.nodes order.

    Note: length_normalize is intentionally NOT supported here -- per-domain
    length normalization would break additivity; the readout uses raw counts.
    """
    doms = list(bows.keys()) if domains is None else list(domains)
    if not doms:
        raise ValueError("domains must select at least one domain")
    backgrounds = backgrounds or {}
    total = None
    for m in doms:
        s = lr_placement_scores(bows[m], lam_dict[m], lay, alpha=alpha,
                                background=backgrounds.get(m), epsilon=epsilon,
                                count_mode=count_mode)
        total = s if total is None else total + s
    return total


def lr_auc_sweep_multidomain(bows, lam_dict, lay, is_fg, *, alpha_grid,
                             domains=None, backgrounds=None, count_mode="raw"):
    """{alpha: max-over-nodes ROC-AUC vs is_fg} for the multi-domain LR score,
    over a domain subset. Mirrors the single-domain `lr_auc_sweep`."""
    y = np.asarray(is_fg, dtype=int)
    out = {}
    for a in alpha_grid:
        s = lr_placement_scores_multidomain(bows, lam_dict, lay, alpha=float(a),
                                            domains=domains, backgrounds=backgrounds,
                                            count_mode=count_mode)
        out[float(a)] = _auc(s.max(axis=1), y)
    return out
```

- [ ] **Step 4: Run to verify they pass**

Run: `cd spark-vi && poetry run pytest tests/test_lr_multidomain.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add spark-vi/spark_vi/models/topic/dag_placement.py spark-vi/tests/test_lr_multidomain.py
git commit -m "feat(dag-placement): additive multi-domain LR placement scorer"
```

---

### Task 2: Fit-time test-set persistence

**Files:**
- Modify: `analysis/cloud/multidomain_cloud.py` (add a persistence phase in `main()` after `save`; add `parent_int` to the manifest; add a pure `_test_persist_cols` helper)
- Test: `analysis/cloud/tests/test_multidomain_cloud.py` (add a helper test)

**Interfaces:**
- Consumes (already in `main()` scope): `bundle.test_df`, `bundle.parent_int`, `model` (a `GatedLDAModel` whose `transform` emits a `nodeAffinity` column), `feature_cols = [f"features_{i}" …]`, `out` (the run dir `Path`).
- Produces: run-dir `test_docs/` parquet (`person_id`, `features_0…`, `frontier`), `test_affinities/` parquet (`person_id`, `nodeAffinity`, `frontier`), and `manifest["parent_int"]` (`{str(engine_id): parent_engine_id}`). Pure helper `_test_persist_cols(feature_cols) -> list[str]`.

- [ ] **Step 1: Write the failing test**

Add to `analysis/cloud/tests/test_multidomain_cloud.py`:

```python
def test_test_persist_cols_is_person_features_frontier():
    from multidomain_cloud import _test_persist_cols
    assert _test_persist_cols(["features_0", "features_1", "features_2"]) == \
        ["person_id", "features_0", "features_1", "features_2", "frontier"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd analysis/cloud && poetry run pytest tests/test_multidomain_cloud.py -k persist -q`
Expected: FAIL (`_test_persist_cols` undefined).

- [ ] **Step 3: Implement the helper + persistence + manifest field**

Add the pure helper near the other module-level helpers in `multidomain_cloud.py`:

```python
def _test_persist_cols(feature_cols):
    """Columns of bundle.test_df to persist for the post-hoc LR readout:
    person_id + every per-domain feature column + the frontier labels."""
    return ["person_id", *feature_cols, "frontier"]
```

In `main()`, add `parent_int` to the manifest's `corpus_manifest` (next to `int2cid`):

```python
                    "parent_int": {str(i): int(p)
                                   for i, p in bundle.parent_int.items()},
```

In `main()`, AFTER the `with _phase("save"):` block writes `save_result` + `manifest.json`, add a persistence phase (uses `out`, `bundle`, `model`, `feature_cols`, all already bound):

```python
        with _phase("persist test set (post-hoc LR readout)"):
            # Self-contained artifact (choice C): the LR readout loads these from
            # the run dir -- no BigQuery, no bundle cache. test_docs = held-out
            # per-domain BOWs + frontier (LR scoring input); test_affinities = the
            # model's native theta-mass node scores (the readout's baseline, so no
            # CAVI re-run). Guarded on a non-empty test split.
            n_test = bundle.test_df.count()
            if n_test > 0:
                (bundle.test_df.select(*_test_persist_cols(feature_cols))
                 .write.mode("overwrite").parquet(str(out / "test_docs")))
                (model.transform(bundle.test_df)
                 .select("person_id", "nodeAffinity", "frontier")
                 .write.mode("overwrite").parquet(str(out / "test_affinities")))
                print(f"[driver]   persisted {n_test} test docs + affinities "
                      f"for LR readout -> {out}", flush=True)
            else:
                print("[driver]   test split empty; skipping LR-readout persistence",
                      flush=True)
```

- [ ] **Step 4: Run to verify the helper test passes**

Run: `cd analysis/cloud && poetry run pytest tests/test_multidomain_cloud.py -q`
Expected: PASS (all, including the new persist test). `main()`'s persistence body is cluster-covered (exercised by the 0071/0072 re-fit), not unit-run here.

- [ ] **Step 5: Commit**

```bash
git add analysis/cloud/multidomain_cloud.py analysis/cloud/tests/test_multidomain_cloud.py
git commit -m "feat(multidomain): persist test set (docs+affinities) + parent_int for LR readout"
```

---

### Task 3: Readout driver + Makefile target

**Files:**
- Create: `analysis/cloud/multidomain_lr_readout.py`
- Create: `analysis/cloud/tests/test_multidomain_lr_readout.py`
- Modify: `analysis/cloud/Makefile` (add `multidomain-lr-readout` target + `.PHONY`)

**Interfaces:**
- Consumes: `spark_vi.io.export.load_result`; `spark_vi.models.topic.dag_placement.{DagLayout, lr_placement_scores_multidomain, _auc}`; `charmpheno.omop.cohorts.disease_anchors`; `_driver_common.make_spark_session`; the run-dir artifacts from Task 2 (`test_docs/`, `test_affinities/`, `manifest.json` with `parent_int`).
- Produces: `build_parser()`, and the pure helpers `children_map(parent_int)`, `subtree_nodes(parent_int, root)`, `build_domain_bows(rows, feature_cols, vocab_sizes)`, `per_disease_auc_row(...)` (all unit-tested); `main()` (cluster-covered); the `multidomain-lr-readout ID=N` Makefile target.

- [ ] **Step 1: Write the failing tests**

Create `analysis/cloud/tests/test_multidomain_lr_readout.py`:

```python
import numpy as np


def test_children_map_and_subtree_from_parent_int():
    from multidomain_lr_readout import children_map, subtree_nodes
    # forest: 100 -> {200 (->201), 300}; 200 has child 201.
    parent_int = {200: 100, 300: 100, 201: 200}
    cmap = children_map(parent_int)
    assert cmap[100] == {200, 300} and cmap[200] == {201}
    # subtree(100) = 100 + all descendants; subtree(200) = {200, 201}
    assert subtree_nodes(parent_int, 100) == {100, 200, 300, 201}
    assert subtree_nodes(parent_int, 200) == {200, 201}
    assert subtree_nodes(parent_int, 300) == {300}       # leaf


def test_build_domain_bows_shapes_and_frontier():
    from multidomain_lr_readout import build_domain_bows
    # two fake collected rows, 2 domains (V0=4, V1=3); features are (indices, values)
    class FakeVec:
        def __init__(self, size, idx, val):
            self.size, self.indices, self.values = size, np.array(idx), np.array(val, dtype=float)
    rows = [
        {"person_id": "a", "features_0": FakeVec(4, [0, 2], [1.0, 3.0]),
         "features_1": FakeVec(3, [1], [2.0]), "frontier": [5]},
        {"person_id": "b", "features_0": FakeVec(4, [], []),
         "features_1": FakeVec(3, [0, 2], [1.0, 1.0]), "frontier": []},
    ]
    bows, frontiers, pids = build_domain_bows(rows, ["features_0", "features_1"], [4, 3])
    assert set(bows) == {0, 1}
    assert bows[0].shape == (2, 4) and bows[1].shape == (2, 3)
    assert bows[0][0, 2] == 3.0 and bows[1][1, 0] == 1.0   # values landed
    assert frontiers == [[5], []] and pids == ["a", "b"]


def test_per_disease_auc_row_uses_max_over_subtree():
    # Node d=1 has child 3 (both scoreable). A doc positive for d if frontier hits
    # {1,3}. The per-disease score is max over subtree(d) columns. Give doc0 a high
    # score at node 3 (subtype) and frontier {3}: it must count as a positive for d
    # and the max-over-subtree must pick up node 3.
    from spark_vi.models.topic.dag_placement import DagLayout
    from multidomain_lr_readout import per_disease_auc_row
    lay = DagLayout({1: 0, 2: 0, 3: 1}, n_bg=1, tpn=1)     # nodes: 1,2,3 ; 3 child of 1
    parent_int = {1: 0, 2: 0, 3: 1}
    # scores [n_docs x n_nodes] aligned to lay.nodes; make node 3 high for doc0.
    n3 = lay.nodes.index(3)
    scores = np.zeros((4, len(lay.nodes)))
    scores[0, n3] = 10.0                                   # doc0 strong at subtype 3
    frontiers = [[3], [], [], []]                          # only doc0 has disease d=1 (via 3)
    auc, n_pos = per_disease_auc_row(scores, frontiers, anchor=1, lay=lay,
                                     parent_int=parent_int)
    assert n_pos == 1
    assert auc == 1.0                                      # doc0 ranks top -> perfect


def test_build_parser_defaults():
    from multidomain_lr_readout import build_parser
    args = build_parser().parse_args(["--run-dir", "/runs/0071-x"])
    assert args.run_dir == "/runs/0071-x"
    assert args.alpha_grid == "0,1,10,100,inf"             # default sweep
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd analysis/cloud && poetry run pytest tests/test_multidomain_lr_readout.py -q`
Expected: FAIL (module doesn't exist).

- [ ] **Step 3: Implement the driver**

Create `analysis/cloud/multidomain_lr_readout.py`:

```python
"""Post-hoc likelihood-ratio placement readout for a MULTI-DOMAIN gated run
(no re-fit). Loads a run dir's dict-lambda + manifest + the persisted held-out
test set (test_docs/ + test_affinities/, written by multidomain_cloud.py), and
emits a per-rare-disease x domain-subset LR-AUC table plus the theta-mass
baseline. Self-contained: no BigQuery, no bundle cache (choice C).

The multi-domain LR score is the per-domain SUM of the single-domain
lr_placement_scores; a domain subset is the per-domain decomposition. Per-disease
detection is max-over-subtree(anchor) vs frontier-hits-subtree.

Only build_parser + the pure helpers (children_map, subtree_nodes,
build_domain_bows, per_disease_auc_row) are unit-tested; main() (Spark load +
parquet reads) is cluster-covered (make multidomain-lr-readout ID=N).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse as sp


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Multi-domain per-domain LR placement-lift readout.")
    p.add_argument("--run-dir", required=True,
                   help="Run directory containing manifest.json + params/ + "
                        "test_docs/ + test_affinities/.")
    p.add_argument("--alpha-grid", default="0,1,10,100,inf",
                   help="Comma list of LR-shrinkage alphas (inf = the lift limit).")
    return p


def parse_alpha_grid(s):
    """['0','1','inf'] -> [0.0, 1.0, inf]."""
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float("inf") if tok.lower() in ("inf", "infinity") else float(tok))
    return out


def children_map(parent_int):
    """{node: set(children)} inverted from {node: parent}."""
    cmap = {}
    for child, parent in parent_int.items():
        cmap.setdefault(int(parent), set()).add(int(child))
    return cmap


def subtree_nodes(parent_int, root):
    """`root` and all its DESCENDANTS (the descendant subtree, NOT the ancestral
    closure). Includes `root` itself."""
    cmap = children_map(parent_int)
    seen, stack = set(), [int(root)]
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        stack.extend(cmap.get(u, ()))
    return seen


def build_domain_bows(rows, feature_cols, vocab_sizes):
    """(bows {m: csr [n x V_m]}, frontiers list[list[int]], person_ids list) from
    collected test_docs rows. rows[i][feature_cols[m]] is a SparseVector-like
    (.indices/.values/.size); vocab_sizes[m] pins V_m."""
    n = len(rows)
    bows = {}
    for m, col in enumerate(feature_cols):
        V = int(vocab_sizes[m])
        indptr = np.zeros(n + 1, dtype=np.int64)
        idx_chunks, data_chunks = [], []
        for i, r in enumerate(rows):
            sv = r[col]
            idx = np.asarray(sv.indices, dtype=np.int64)
            val = np.asarray(sv.values, dtype=np.float64)
            idx_chunks.append(idx)
            data_chunks.append(val)
            indptr[i + 1] = indptr[i] + len(idx)
        indices = np.concatenate(idx_chunks) if idx_chunks else np.array([], np.int64)
        data = np.concatenate(data_chunks) if data_chunks else np.array([], np.float64)
        bows[m] = sp.csr_matrix((data, indices, indptr), shape=(n, V))
    frontiers = [[int(x) for x in r["frontier"]] for r in rows]
    person_ids = [r["person_id"] for r in rows]
    return bows, frontiers, person_ids


def per_disease_auc_row(scores, frontiers, anchor, lay, parent_int):
    """(auc, n_pos) for detecting disease `anchor` from a [n_docs x n_nodes] score
    matrix (columns in lay.nodes order). Positive = the doc's frontier intersects
    subtree(anchor) (anchor + descendants, scoreable); per-disease score = the max
    over that subtree's columns. One-class positive/negative -> auc nan."""
    from spark_vi.models.topic.dag_placement import _auc
    sub = subtree_nodes(parent_int, anchor) & set(lay.nodes)
    if not sub:
        return float("nan"), 0
    cols = [lay.nodes.index(u) for u in sub]
    node_score = scores[:, cols].max(axis=1)
    y = np.array([1 if (set(fr) & sub) else 0 for fr in frontiers], dtype=int)
    return _auc(node_score, y), int(y.sum())


# ---- affinity (theta-mass) baseline ---------------------------------------
def affinity_matrix(aff_rows, n_nodes):
    """[n_docs x n_nodes] dense node-affinity matrix from collected
    test_affinities rows (r['nodeAffinity'] a DenseVector-like)."""
    out = np.zeros((len(aff_rows), n_nodes), dtype=float)
    for i, r in enumerate(aff_rows):
        out[i, :] = np.asarray(r["nodeAffinity"].toArray(), dtype=float)
    return out


def main(argv=None) -> int:
    from _driver_common import make_spark_session
    from charmpheno.omop.cohorts import disease_anchors
    from spark_vi.io.export import load_result
    from spark_vi.models.topic.dag_placement import (
        DagLayout, lr_placement_scores_multidomain)

    args = build_parser().parse_args(argv)
    run_dir = Path(args.run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    result = load_result(run_dir)
    lam_dict = result.global_params["lambda"]                 # {m: [K x V_m]}
    n_dom = len(lam_dict)
    feature_cols = [f"features_{i}" for i in range(n_dom)]
    vocab_sizes = [lam_dict[m].shape[1] for m in range(n_dom)]
    domain_names = manifest.get("domains", [f"m{i}" for i in range(n_dom)])

    cm = manifest["corpus_manifest"]
    parent_int = {int(k): int(v) for k, v in cm["parent_int"].items()}
    lay = DagLayout(parent_int, n_bg=manifest["n_bg"], tpn=manifest["tpn"])
    int2cid = {int(k): int(v) for k, v in cm["int2cid"].items()}
    cid2int = {c: i for i, c in int2cid.items()}
    name_by_id = {int(k): v for k, v in cm["name_by_id"].items()}
    alpha_grid = parse_alpha_grid(args.alpha_grid)

    with make_spark_session(app_name="multidomain-lr-readout") as spark:
        rows = spark.read.parquet(str(run_dir / "test_docs")).select(
            "person_id", *feature_cols, "frontier").collect()
        aff_rows = spark.read.parquet(str(run_dir / "test_affinities")).select(
            "person_id", "nodeAffinity").collect()

    bows, frontiers, pids = build_domain_bows(rows, feature_cols, vocab_sizes)
    aff = affinity_matrix(aff_rows, len(lay.nodes))

    # rare6 anchor engine-ids (skip anchors pruned out of the DAG).
    anchors = []
    for cid in disease_anchors(manifest["disease"]):
        u = cid2int.get(int(cid))
        if u is not None and u in set(lay.nodes):
            anchors.append(u)

    # domain subsets: all, each-alone, leave-one-out (labeled by name).
    subsets = {"all": list(range(n_dom))}
    for i, nm in enumerate(domain_names):
        subsets[f"only:{nm}"] = [i]
    if n_dom > 1:
        for i, nm in enumerate(domain_names):
            subsets[f"drop:{nm}"] = [j for j in range(n_dom) if j != i]

    # Overall detection sweep (all domains, max-over-nodes vs is_fg) over the grid,
    # for continuity with the single-domain readout's output shape.
    from spark_vi.models.topic.dag_placement import lr_auc_sweep_multidomain
    is_fg = np.array([1 if (set(fr) & set(lay.nodes)) else 0 for fr in frontiers],
                     dtype=int)
    sweep = lr_auc_sweep_multidomain(bows, lam_dict, lay, is_fg, alpha_grid=alpha_grid)
    print("[lr] === overall detection LR-AUC(alpha), all domains, max-over-nodes ===",
          flush=True)
    for a in alpha_grid:
        print(f"[lr]   alpha={a}: {sweep[a]:.3f}", flush=True)

    # Per-disease x domain-subset table at the alpha=inf lift limit (headline).
    # Score ONCE per subset (scores do not depend on the anchor), then loop anchors.
    a_head = alpha_grid[-1]
    subset_scores = {name: lr_placement_scores_multidomain(
                         bows, lam_dict, lay, alpha=a_head, domains=doms)
                     for name, doms in subsets.items()}
    print(f"[lr] === per-disease x domain-subset LR-AUC (alpha={a_head}) ===",
          flush=True)
    header = "disease".ljust(26) + "n+".rjust(5) + "  theta"
    for name in subsets:
        header += "  " + name[:12].rjust(12)
    print("[lr] " + header, flush=True)
    for u in anchors:
        dname = str(name_by_id.get(u, int2cid.get(u)))[:24]
        theta_auc, n_pos = per_disease_auc_row(aff, frontiers, u, lay, parent_int)
        line = dname.ljust(26) + str(n_pos).rjust(5) + f"  {theta_auc:5.3f}"
        for name in subsets:
            auc, _ = per_disease_auc_row(subset_scores[name], frontiers, u, lay,
                                         parent_int)
            line += "  " + f"{auc:12.3f}"
        print("[lr] " + line, flush=True)

    print(f"[lr] scored {len(pids)} held-out docs; {len(anchors)} rare6 anchors "
          f"present; domains={domain_names}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify the pure-helper tests pass**

Run: `cd analysis/cloud && poetry run pytest tests/test_multidomain_lr_readout.py -q`
Expected: PASS (4 tests). `main()` is cluster-covered.

- [ ] **Step 5: Add the Makefile target**

In `analysis/cloud/Makefile`, add `multidomain-lr-readout` to the `.PHONY` line, and add the target (mirror the `lr-readout` spark-submit pattern; no `--cache-uri`):

```make
multidomain-lr-readout: zip cluster-overlay $(WORKSPACE_ENV)
	@if [ -z "$(ID)" ]; then echo "ERROR: provide ID=N"; exit 1; fi
	. ./$(WORKSPACE_ENV) && \
	spark-submit $(SPARK_SUBMIT_FLAGS) \
	  --py-files $(PY_FILES) \
	  $(REPO_ROOT)/analysis/cloud/multidomain_lr_readout.py \
	  --run-dir $(RUNS_DIR)/$(shell printf '%04d' $(ID))-* \
	  $(if $(ALPHA_GRID),--alpha-grid $(ALPHA_GRID),)
```

Check the existing `lr-readout` target for the exact `SPARK_SUBMIT_FLAGS` / `PY_FILES` variable names in this Makefile and reuse them verbatim (they may be spelled differently); the glob `$(RUNS_DIR)/00NN-*` resolves the slug like the other run-dir targets. Add a help line under the help block:

```
#   multidomain-lr-readout ID=N [ALPHA_GRID=..]  Post-hoc per-domain LR readout (multidomain fit)
```

- [ ] **Step 6: Commit**

```bash
git add analysis/cloud/multidomain_lr_readout.py analysis/cloud/tests/test_multidomain_lr_readout.py analysis/cloud/Makefile
git commit -m "feat(multidomain): per-domain LR placement-lift readout driver + Makefile target"
```

---

## Final verification (after all tasks)

- [ ] `cd spark-vi && poetry run pytest tests/test_lr_multidomain.py -q` — scorer green.
- [ ] `cd analysis/cloud && poetry run pytest tests/test_multidomain_cloud.py tests/test_multidomain_lr_readout.py -q` — persistence helper + readout helpers green.
- [ ] **Cluster (user-run):** re-fit `make exp ID=71` and `make exp ID=72` (now persist `test_docs`/`test_affinities`), then `make multidomain-lr-readout ID=71` and `ID=72` — the per-disease × domain-subset AUC table prints, all six rare6 anchors present (or a logged skip for any pruned), θ-mass column beside the LR columns, no BigQuery touched.
```
