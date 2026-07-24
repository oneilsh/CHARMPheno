# DAG/Ontology PG-STM Model Core (v1: additive-η mean-offsets) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a DAG/ontology of disease-group nodes to the gated PG-STM as an **additive mean-offset summed over each document's ancestral closure**, with a sparse depth-scaled shrinkage prior, so scarce nodes pool toward their ancestors and unearned structure deactivates.

**Architecture:** v1 realizes the offset as a **mean shift only** (subtype/intermediate nodes own no new topics — deferred to a follow-on). This collapses to the existing PG-STM run with an **augmented covariate** `w_d = [x_d ; z_d]` (`z_d` = binary closure indicator over the DAG's nodes) and coefficient `[Γ ; B]`, plus a **depth-scaled ridge penalty on the B block**. The E-step, Σ M-step, β M-step, and gate are reused unchanged from `pg_stm.py`; only the covariate-ridge gains a per-row penalty vector.

**Tech Stack:** Python / NumPy / SciPy, single-machine, mirroring `spark_vi/models/topic/pg_stm.py`.

## Global Constraints

- **Test-honesty rule:** every test's docstring states (i) what is *planted* (synthetic truth) vs *real* (β from an existing fit, real length/group distributions), (ii) where it sits on the synthetic→real spectrum, and (iii) the claim it supports **and** the claim it explicitly does not. **No test may assert a *transfer* claim ("helps our corpus") from a *synthetic* result** — synthetic proves only *math correctness* ("the estimator recovers what's planted").
- **Ontology-agnostic, DAG-native, many-to-many:** the core consumes a `DagGate` (is-a edges) + a per-document set of most-specific attested node ids. A node may have multiple parents (diamonds → closure is a **set**, shared ancestor counted once); a document may attest multiple nodes.
- **v1 scope vs spec:** implements *Piece A* (mean-offsets) of `docs/superpowers/specs/2026-07-13-dag-ontology-pg-stm-model-core-design.md`. *Piece B* (node-owned distinct topics + multi-level gate/Σ composition — the spec's flagged risk) is a **follow-on plan**, so this plan does **not** touch the gate or Σ block structure. Pooling here is of topic-*usage levels*, not Σ; the scarce-block **Σ**-rescue is the later read-out-honesty spec.
- Domain-agnostic engine layer: integer node ids + integer token ids only; never concept names/ids.
- Cite any literature-sourced method/default in its docstring; an uncited constant is labeled a heuristic.
- Reuse `pg_stm.py` primitives (`pg_empty_stats`, `pg_accumulate_doc`, `pg_combine_stats`, `pg_estep_doc`, `beta_dirichlet_mean`, `assemble_sigma`, `stick_layout`) unchanged; do not fork them.

---

## File Structure

- **Create `spark-vi/spark_vi/models/topic/pg_stm_dag.py`** — the whole v1 core:
  `DagGate` (closure/indicator/depth/dump), `offset_penalty`, `dag_offset_ridge`, `PGSTMDag`, `root_only_dag`, `inject_spurious_edges`.
- **Modify `spark-vi/tests/_stm_synth.py`** — add `dag_offset_corpus(...)` (real-β-seeded planted-offset corpus) and `real_beta_from(...)` (β loader with a synthetic fallback).
- **Create `spark-vi/tests/test_pg_stm_dag.py`** — one test per validation item (Tasks 1,3,4,5,6,7).

Reused unchanged: everything in `pg_stm.py`, `_mcmc_diag.py`, `types.py` (`STMDocument`), `partition.py` (`TopicBlockPartition`).

---

### Task 1: `DagGate` — closure, indicator, depth, audit dump

**Files:**
- Create: `spark-vi/spark_vi/models/topic/pg_stm_dag.py`
- Test: `spark-vi/tests/test_pg_stm_dag.py`

**Interfaces:**
- Produces: `DagGate(parents: Sequence[Sequence[int]])` with `n_nodes:int`, `depth:np.ndarray (n_nodes,)`, `ancestors(u)->frozenset[int]`, `closure(nodes)->frozenset[int]`, `closure_indicator(nodes)->np.ndarray (n_nodes,) float64 {0,1}`, `dump()->list[dict]`. Invariant enforced: node 0 is the root; every parent index is **less than** its child (topological order → acyclic).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pg_stm_dag.py
import numpy as np
import pytest
from spark_vi.models.topic.pg_stm_dag import DagGate


def test_dag_closure_and_indicator_over_two_levels():
    # 0=root; 1,2 anchors under root; 3 = subtype under anchor 1
    dag = DagGate([(), (0,), (0,), (1,)])
    assert dag.n_nodes == 4
    assert dag.closure(frozenset({3})) == frozenset({3, 1, 0})   # subtype -> anchor -> root
    assert dag.closure(frozenset({2})) == frozenset({2, 0})
    z = dag.closure_indicator(frozenset({3}))
    assert z.dtype == np.float64
    assert list(z) == [1.0, 1.0, 0.0, 1.0]                       # nodes 0,1,3 on; 2 off


def test_dag_diamond_shared_ancestor_counted_once():
    # 3 has two parents 1 and 2, both under root 0 (a diamond)
    dag = DagGate([(), (0,), (0,), (1, 2)])
    assert dag.closure(frozenset({3})) == frozenset({3, 1, 2, 0})
    assert dag.closure_indicator(frozenset({3})).sum() == 4      # 0 once, not twice


def test_dag_depth_is_shortest_root_distance():
    dag = DagGate([(), (0,), (1,), (0, 2)])   # node 3 reachable via 0 (d=1) or via 2 (d=3)
    assert list(dag.depth) == [0, 1, 2, 1]    # shortest wins for node 3


def test_dag_rejects_parent_index_not_less_than_child():
    with pytest.raises(ValueError):
        DagGate([(), (2,), (0,)])             # node 1's parent 2 > 1 -> not topo-ordered


def test_dag_dump_lists_nodes_with_depth_and_parents():
    dag = DagGate([(), (0,), (1,)])
    d = dag.dump()
    assert d[2] == {"node": 2, "depth": 2, "parents": [1]}
```

- [ ] **Step 2: Run tests, verify they fail** — `pytest tests/test_pg_stm_dag.py -q` → FAIL (no module).

- [ ] **Step 3: Implement `DagGate`**

```python
# spark_vi/models/topic/pg_stm_dag.py
"""DAG/ontology-structured additive mean-offset layer for the gated PG-STM (v1).

A document's mean logits gain an additive term summed over its ancestral closure in
an is-a DAG: mu_d = Gamma^T x_d + sum_{u in closure(v_d)} eta_u. v1 realizes eta_u as a
MEAN SHIFT only (nodes own no new topics), so the model is the existing PG-STM with an
augmented covariate w_d = [x_d ; closure_indicator_d] and coefficient [Gamma ; B], plus
a depth-scaled ridge penalty on B. See docs/superpowers/plans/2026-07-13-dag-ontology-
pg-stm-model-core.md and the spec it implements.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


class DagGate:
    """An is-a DAG over group nodes. Node 0 is the root (background). Parent indices
    must be strictly less than the child index (topological order => acyclic). Closures
    are SETS, so a diamond's shared ancestor is counted once. Depth is the shortest
    root distance (used by the depth-scaled offset prior)."""

    def __init__(self, parents: Sequence[Sequence[int]]):
        self.parents: list[tuple[int, ...]] = [tuple(int(p) for p in ps) for ps in parents]
        self.n_nodes = len(self.parents)
        if self.n_nodes == 0 or self.parents[0] != ():
            raise ValueError("node 0 must be the root with no parents")
        for u, ps in enumerate(self.parents):
            for p in ps:
                if not (0 <= p < u):
                    raise ValueError(f"parent {p} of node {u} must satisfy 0 <= p < {u}")
        self._anc: list[frozenset[int]] = []
        for u, ps in enumerate(self.parents):
            acc: set[int] = set()
            for p in ps:
                acc.add(p)
                acc |= self._anc[p]
            self._anc.append(frozenset(acc))
        self.depth = self._compute_depth()

    def _compute_depth(self) -> np.ndarray:
        depth = np.zeros(self.n_nodes, dtype=np.int64)
        for u in range(self.n_nodes):
            if self.parents[u]:
                depth[u] = 1 + min(depth[p] for p in self.parents[u])
        return depth

    def ancestors(self, u: int) -> frozenset[int]:
        return self._anc[u]

    def closure(self, nodes) -> frozenset[int]:
        out: set[int] = set()
        for v in nodes:
            out.add(int(v))
            out |= self._anc[int(v)]
        return frozenset(out)

    def closure_indicator(self, nodes) -> np.ndarray:
        z = np.zeros(self.n_nodes, dtype=np.float64)
        for u in self.closure(nodes):
            z[u] = 1.0
        return z

    def dump(self) -> list[dict]:
        return [{"node": u, "depth": int(self.depth[u]), "parents": list(self.parents[u])}
                for u in range(self.n_nodes)]
```

- [ ] **Step 4: Run tests, verify pass** — `pytest tests/test_pg_stm_dag.py -q` → 5 passed.

- [ ] **Step 5: Commit** — `git add spark-vi/spark_vi/models/topic/pg_stm_dag.py spark-vi/tests/test_pg_stm_dag.py && git commit -m "feat(dag): DagGate closure/indicator/depth for the additive-offset PG-STM"`

---

### Task 2: `offset_penalty` + `dag_offset_ridge` — the penalized moment-ridge

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/pg_stm_dag.py`
- Test: `spark-vi/tests/test_pg_stm_dag.py`

**Interfaces:**
- Consumes: `DagGate.depth`.
- Produces: `offset_penalty(P:int, dag:DagGate, *, gamma_ridge:float, lam_base:float, gamma_depth:float) -> np.ndarray (P+n_nodes,)` and `dag_offset_ridge(WtW, WtM, *, penalty) -> np.ndarray`. `dag_offset_ridge` generalizes `pg_gamma_ridge_moments` (scalar ridge → per-row penalty): `solve(WtW + diag(penalty), WtM)`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_pg_stm_dag.py
from spark_vi.models.topic.pg_stm_dag import offset_penalty, dag_offset_ridge


def test_offset_penalty_is_depth_scaled_on_node_block_only():
    dag = DagGate([(), (0,), (1,)])            # depths 0,1,2
    pen = offset_penalty(P=2, dag=dag, gamma_ridge=1e-6, lam_base=2.0, gamma_depth=1.0)
    assert pen.shape == (2 + 3,)
    assert np.allclose(pen[:2], 1e-6)          # covariates lightly ridged
    assert np.allclose(pen[2:], [2.0 * 1, 2.0 * 2, 2.0 * 3])   # lam_base*(1+depth)


def test_dag_offset_ridge_recovers_well_posed_coefficients():
    rng = np.random.default_rng(0)
    n, d, k = 500, 4, 3
    W = rng.standard_normal((n, d))
    coeff = rng.standard_normal((d, k))
    M = W @ coeff
    got = dag_offset_ridge(W.T @ W, W.T @ M, penalty=np.full(d, 1e-8))
    assert np.allclose(got, coeff, atol=1e-4)


def test_dag_offset_ridge_shrinks_an_unconstrained_column_to_zero():
    # design column 3 is all-zero (a "never-active" node) -> its coeff row must go ~0
    rng = np.random.default_rng(1)
    n, k = 400, 2
    W = rng.standard_normal((n, 4)); W[:, 3] = 0.0
    M = W[:, :3] @ rng.standard_normal((3, k))
    pen = np.array([1e-8, 1e-8, 1e-8, 5.0])
    got = dag_offset_ridge(W.T @ W, W.T @ M, penalty=pen)
    assert np.allclose(got[3], 0.0, atol=1e-9)      # unconstrained + penalized -> exactly ~0
```

- [ ] **Step 2: Run tests, verify fail** — FAIL (functions not defined).

- [ ] **Step 3: Implement**

```python
# append to pg_stm_dag.py

def offset_penalty(P, dag, *, gamma_ridge, lam_base, gamma_depth):
    """(P + n_nodes,) ridge penalty: ``gamma_ridge`` on each covariate row, and
    ``lam_base * (1 + depth[u]) ** gamma_depth`` on each node-offset row. Depth-scaling
    (deeper => larger penalty) encodes "prefer general explanations, specialize only on
    evidence" (a structural, inspectable shrinkage; not an inference hyperparameter).
    A node whose closure-indicator column is never active is pulled to 0 by its penalty."""
    pen = np.empty(int(P) + dag.n_nodes, dtype=np.float64)
    pen[:P] = float(gamma_ridge)
    pen[P:] = float(lam_base) * (1.0 + dag.depth.astype(np.float64)) ** float(gamma_depth)
    return pen


def dag_offset_ridge(WtW, WtM, *, penalty):
    """Penalized moment-form ridge: solve (WtW + diag(penalty)) C = WtM. Generalizes
    pg_gamma_ridge_moments' scalar ridge to a per-row penalty vector, so covariate and
    node-offset rows are shrunk independently (depth-scaled). WtW is (P+U, P+U), WtM is
    (P+U, K-1)."""
    WtW = np.asarray(WtW, dtype=np.float64)
    WtM = np.asarray(WtM, dtype=np.float64)
    return np.linalg.solve(WtW + np.diag(np.asarray(penalty, dtype=np.float64)), WtM)
```

- [ ] **Step 4: Run tests, verify pass** — 3 passed.

- [ ] **Step 5: Commit** — `git commit -am "feat(dag): depth-scaled offset penalty + penalized moment-ridge"`

---

### Task 3: `PGSTMDag` driver + `root_only_dag`; Test 1 equivalence to the flat model

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/pg_stm_dag.py`
- Test: `spark-vi/tests/test_pg_stm_dag.py`

**Interfaces:**
- Consumes: `pg_stm.pg_empty_stats/pg_accumulate_doc/pg_estep_doc/beta_dirichlet_mean/assemble_sigma/stick_layout`, `DagGate`, `offset_penalty`, `dag_offset_ridge`, `STMDocument`.
- Produces: `PGSTMDag(K, V, partition, dag, *, P, n_iter=200, gamma_ridge=1e-6, lam_base=1.0, gamma_depth=1.0, Psi0_scale=1.0, nu0=None, beta_eta=0.1, inner_rounds=8, inner_tol=1e-3, sigma_mode="iw", seed=0)`; `.fit(docs, doc_nodes) -> {"beta","Gamma","B","Sigma","node_norms","psi_mean"}` where `doc_nodes[d]` is the frozenset of node ids the document attests (its closure supplies the offset). `root_only_dag() -> DagGate([()])`.

- [ ] **Step 1: Write the failing test (Test 1 — equivalence, math-correctness)**

```python
# append to tests/test_pg_stm_dag.py
from spark_vi.models.topic.pg_stm import PGSTMVI
from spark_vi.models.topic.pg_stm_dag import PGSTMDag, root_only_dag
from tests._stm_synth import gated_ln_corpus_stick


def test_pgstmdag_root_only_matches_flat_pgstmvi():
    """PLANTED: a stick-native gated corpus. REAL: nothing. Synthetic -> MATH-CORRECTNESS
    only: with a root-only DAG (the offset is a single global intercept, collinear with
    the covariate intercept), PGSTMDag is a reparameterization of PGSTMVI and must return
    the SAME beta and Sigma. Proves the augmented-covariate machinery does not perturb the
    validated flat model. Does NOT prove anything about multi-level DAG behavior."""
    docs, part, _St, _b = gated_ln_corpus_stick(
        group_weights={"A": 0.5, "B": 0.5}, fg_per_group=2, bg_k=3, V=60, D=300,
        doc_len=40, seed=0)
    P = docs[0].x.shape[0]
    vi = PGSTMVI(K=part.K, V=60, partition=part, P=P, n_iter=40, seed=0).fit(docs)
    dag = root_only_dag()
    doc_nodes = [frozenset({0})] * len(docs)                 # every doc attests the root
    out = PGSTMDag(K=part.K, V=60, partition=part, dag=dag, P=P, n_iter=40,
                   gamma_ridge=1e-6, lam_base=1e-6, gamma_depth=1.0, seed=0).fit(docs, doc_nodes)
    assert np.allclose(out["beta"], vi["beta"], atol=2e-3)
    assert np.allclose(out["Sigma"], vi["Sigma"], atol=2e-3)
    assert out["B"].shape == (1, part.K - 1)                 # one node offset row
```

- [ ] **Step 2: Run test, verify fail** — FAIL (PGSTMDag not defined).

- [ ] **Step 3: Implement `PGSTMDag` and `root_only_dag`**

```python
# append to pg_stm_dag.py
import dataclasses

from spark_vi.models.topic.pg_stm import (
    pg_empty_stats, pg_accumulate_doc, pg_estep_doc, beta_dirichlet_mean,
    assemble_sigma, stick_layout,
)


def root_only_dag() -> DagGate:
    """The degenerate DAG: a single root node. The offset is one global intercept."""
    return DagGate([()])


class PGSTMDag:
    """Gated PG-STM with an additive mean-offset summed over each document's ancestral
    closure (v1: mean shift only). Realized as PGSTMVI on an augmented covariate
    w_d = [x_d ; closure_indicator_d] with coefficient [Gamma ; B] and a depth-scaled
    ridge penalty on B. Gate / Sigma / beta / E-step are pg_stm's, unchanged."""

    def __init__(self, K, V, partition, dag, *, P, n_iter=200, gamma_ridge=1e-6,
                 lam_base=1.0, gamma_depth=1.0, Psi0_scale=1.0, nu0=None, beta_eta=0.1,
                 inner_rounds=8, inner_tol=1e-3, sigma_mode="iw", seed=0):
        self.K, self.V, self.partition, self.dag = K, V, partition, dag
        self.P, self.U = P, dag.n_nodes
        self.n_iter, self.beta_eta, self.sigma_mode = n_iter, beta_eta, sigma_mode
        self.gamma_ridge, self.lam_base, self.gamma_depth = gamma_ridge, lam_base, gamma_depth
        self.Psi0_scale, self.nu0 = Psi0_scale, nu0
        self.inner_rounds, self.inner_tol, self.seed = inner_rounds, inner_tol, seed
        self.layout = stick_layout(partition)

    def fit(self, docs, doc_nodes):
        rng = np.random.default_rng(self.seed)
        K, V, Pw = self.K, self.V, self.P + self.U
        Ksm1 = K - 1
        # augment each doc's covariate with its closure indicator: w = [x ; z]
        docs_aug = []
        for doc, nodes in zip(docs, doc_nodes):
            z = self.dag.closure_indicator(nodes)
            w = np.concatenate([np.asarray(doc.x, dtype=np.float64), z])
            docs_aug.append(dataclasses.replace(doc, x=w))
        penalty = offset_penalty(self.P, self.dag, gamma_ridge=self.gamma_ridge,
                                 lam_base=self.lam_base, gamma_depth=self.gamma_depth)
        beta = rng.random((K, V)) + self.beta_eta
        beta /= beta.sum(axis=1, keepdims=True)
        Cf = np.zeros((Pw, Ksm1))                 # [Gamma ; B] stacked
        Sigma = np.eye(Ksm1)
        D = len(docs_aug)
        psi_mean = np.zeros((D, Ksm1))
        for _ in range(self.n_iter):
            log_beta = np.log(beta)
            stats = pg_empty_stats(K, V, Pw, self.partition.groups)
            for d, doc in enumerate(docs_aug):
                (g,) = tuple(doc.groups)
                glay = self.layout["groups"][g]
                m, Vd, phi, active, allowed, mu_active, _nc = pg_estep_doc(
                    doc, glay, log_beta, Cf, Sigma, K=K, B=self.layout["B"],
                    inner_rounds=self.inner_rounds, inner_tol=self.inner_tol)
                pg_accumulate_doc(stats, doc, (m, Vd, phi, active, allowed, mu_active), K=K)
                psi_mean[d, active] = m
            beta = beta_dirichlet_mean(stats["wts"], eta=self.beta_eta)
            Cf = dag_offset_ridge(stats["XtX"], stats["XtM"], penalty=penalty)
            Sigma = assemble_sigma(stats["S"], self.layout["bg_sticks"],
                                   stats["group_counts"], stats["D"], K=K,
                                   groups=self.partition.groups, layout=self.layout,
                                   sigma_mode=self.sigma_mode, Psi0_scale=self.Psi0_scale,
                                   nu0=self.nu0)
        Gamma, B = Cf[:self.P], Cf[self.P:]
        return {"beta": beta, "Gamma": Gamma, "B": B, "Sigma": Sigma,
                "node_norms": np.linalg.norm(B, axis=1), "psi_mean": psi_mean}
```

- [ ] **Step 4: Run test, verify pass** — `pytest tests/test_pg_stm_dag.py::test_pgstmdag_root_only_matches_flat_pgstmvi -q` → PASS. (If β/Σ drift above 2e-3, raise `n_iter` to 60 — the reparam converges as both fits settle; do not loosen `atol` beyond 3e-3.)

- [ ] **Step 5: Commit** — `git commit -am "feat(dag): PGSTMDag additive-offset driver; root-only == flat PGSTMVI (Test 1)"`

---

### Task 4: `dag_offset_corpus` real-β plant + `real_beta_from`; Test 2 offset recovery through a 2-level closure

**Files:**
- Modify: `spark-vi/tests/_stm_synth.py`
- Test: `spark-vi/tests/test_pg_stm_dag.py`

**Interfaces:**
- Produces: `real_beta_from(K:int, V:int, *, source:str|None=None, seed:int=0) -> np.ndarray (K,V)` — loads β from an existing export if `source` is given, else builds a **realistic-overlap** synthetic β via `gated_ln_corpus_stick(..., topic_overlap=0.6)` and returns its β (documented as the fallback). `dag_offset_corpus(*, dag, node_offsets, partition, beta, node_of_group, doc_nodes_plan, sigma_true, doc_len, seed) -> (docs, doc_nodes)` — plants `mu_d = Σ_{u∈closure} node_offsets[u]` (restricted to the doc's active sticks), draws ψ~N(mu, sigma_true[active]), composes θ via `gated_theta`, emits `STMDocument`s with `groups={anchor}` and the parallel `doc_nodes` list.

- [ ] **Step 1: Write the failing test (Test 2 — offset recovery, realistic overlap)**

```python
# append to tests/test_pg_stm_dag.py
from tests._stm_synth import dag_offset_corpus, real_beta_from


def test_offset_recovery_through_two_level_closure():
    """PLANTED: node offsets on a root->anchor->subtype DAG + a planted Sigma. REAL: beta
    (realistic overlap, topic_overlap=0.6) and doc-length distribution. Realistic-overlap
    synthetic -> MATH-CORRECTNESS: given a known closure structure, PGSTMDag recovers the
    planted node offsets (subtype offset separated from its anchor's) through a two-level
    closure. Does NOT prove real-data offsets are recoverable, nor transfer."""
    from spark_vi.models.topic.partition import TopicBlockPartition
    part = TopicBlockPartition(group_var="g", background_k=6,
                               foreground=(("A", 4), ("B", 4)))
    K, V = part.K, 300
    beta = real_beta_from(K, V, seed=2)
    # DAG: 0 root; 1,2 anchors (= groups A,B); 3 = subtype under anchor 1
    dag = DagGate([(), (0,), (0,), (1,)])
    Ksm1 = K - 1
    rng = np.random.default_rng(3)
    node_offsets = {0: np.zeros(Ksm1), 1: rng.standard_normal(Ksm1),
                    2: rng.standard_normal(Ksm1), 3: rng.standard_normal(Ksm1)}
    sigma_true = 3.0 * np.eye(Ksm1)
    docs, doc_nodes = dag_offset_corpus(
        dag=dag, node_offsets=node_offsets, partition=part, beta=beta,
        node_of_group={"A": 1, "B": 2}, doc_nodes_plan={1: 400, 2: 400, 3: 400},
        sigma_true=sigma_true, doc_len=80, seed=4)
    out = PGSTMDag(K=K, V=V, partition=part, dag=dag, P=docs[0].x.shape[0],
                   n_iter=60, lam_base=1e-3, gamma_depth=1.0, seed=0).fit(docs, doc_nodes)
    B = out["B"]
    # recovery is up to the root/intercept reparam; compare on the ACTIVE sticks of each
    # node's own block via correlation of recovered vs planted subtype offset.
    lay = stick_layout(part)
    a1 = lay["groups"]["A"]["active"]
    r = np.corrcoef(B[3][a1], node_offsets[3][a1])[0, 1]
    assert r > 0.6, f"subtype offset not recovered through the 2-level closure (r={r:.2f})"
```

- [ ] **Step 2: Run test, verify fail** — FAIL (`dag_offset_corpus`/`real_beta_from` not defined).

- [ ] **Step 3: Implement the plant** (in `tests/_stm_synth.py`)

```python
# tests/_stm_synth.py — add near gated_ln_corpus_stick
def real_beta_from(K, V, *, source=None, seed=0):
    """Topic-word matrix (K,V) for DAG plants. If ``source`` names an export bundle, load
    its beta (realistic overlap by construction). Otherwise (default) synthesize a
    REALISTIC-OVERLAP beta via gated_ln_corpus_stick(topic_overlap=0.6) — the honest
    stand-in until a real bundle is wired; the caller's test docstring must not claim
    transfer from it."""
    if source is not None:
        import numpy as _np
        return _np.load(source)["beta"]
    from math import isqrt  # noqa: F401  (kept for parity; not required)
    # borrow a realistic-overlap beta shaped (K, V)
    gw = {"A": 0.5, "B": 0.5}
    fg = max(1, (K - 2) // 2)
    _d, _p, _S, beta = gated_ln_corpus_stick(
        group_weights=gw, fg_per_group=fg, bg_k=K - 2 * fg, V=V, D=4, doc_len=10,
        topic_overlap=0.6, seed=seed)
    return beta[:K] if beta.shape[0] >= K else np.pad(beta, ((0, K - beta.shape[0]), (0, 0)))


def dag_offset_corpus(*, dag, node_offsets, partition, beta, node_of_group,
                      doc_nodes_plan, sigma_true, doc_len, seed):
    """Plant additive node offsets on a DagGate and generate a gated corpus.

    For a document at most-specific node v (anchor = the top-level node on v's root
    path, mapped back to a partition group via ``node_of_group`` inverse), the mean over
    its ACTIVE sticks is mu = sum_{u in closure(v)} node_offsets[u]; psi ~ N(mu[active],
    sigma_true[active,active]); theta = gated_theta(psi split into bg/gate/fg); tokens ~
    Multinomial(doc_len, theta @ beta). ``doc_nodes_plan`` maps node id -> #docs at that
    node. Returns (docs, doc_nodes) with doc.groups = {anchor group} and doc_nodes[d] =
    frozenset({v}). Domain-agnostic (integer ids only)."""
    from spark_vi.models.topic.pg_stm import stick_layout, gated_theta
    rng = np.random.default_rng(seed)
    lay = stick_layout(partition)
    group_of_node = {nid: g for g, nid in node_of_group.items()}
    # anchor(v) = the child-of-root on v's path (the node whose parent chain hits an anchor id)
    anchor_ids = set(node_of_group.values())

    def anchor_of(v):
        cur = v
        chain = [v] + sorted(dag.ancestors(v))
        for c in chain:
            if c in anchor_ids:
                return c
        raise ValueError(f"node {v} has no anchor ancestor")

    docs, doc_nodes = [], []
    nb = len(lay["bg_sticks"])
    for v, n_docs in doc_nodes_plan.items():
        g = group_of_node[anchor_of(v)]
        active = lay["groups"][g]["active"]
        allowed = np.concatenate([partition.background_indices(),
                                  partition.block_indices(g)]).astype(np.int64)
        mu_full = np.zeros(partition.K - 1)
        for u in dag.closure(frozenset({v})):
            mu_full = mu_full + node_offsets[u]
        mu_a = mu_full[active]
        Sa = sigma_true[np.ix_(active, active)]
        for _ in range(n_docs):
            psi = rng.multivariate_normal(mu_a, Sa)
            psi_bg, psi_gate, psi_fg = psi[:nb], psi[nb], psi[nb + 1:]
            theta_allowed = gated_theta(psi_bg, psi_gate, psi_fg)
            theta = np.zeros(partition.K); theta[allowed] = theta_allowed
            toks = rng.choice(partition.beta_dim if hasattr(partition, "beta_dim") else beta.shape[1],
                              size=doc_len, p=theta @ beta)
            u_, c_ = np.unique(toks, return_counts=True)
            docs.append(STMDocument(indices=u_.astype(np.int32), counts=c_.astype(np.float64),
                                    length=int(c_.sum()), x=np.array([1.0]),
                                    groups=frozenset({g})))
            doc_nodes.append(frozenset({v}))
    return docs, doc_nodes
```

- [ ] **Step 4: Run test, verify pass** — `pytest tests/test_pg_stm_dag.py::test_offset_recovery_through_two_level_closure -q` → PASS. (If `r` is borderline, raise `n_iter` to 90 or `doc_len` to 100; do **not** lower the 0.6 threshold without recording why in the test.)

- [ ] **Step 5: Commit** — `git commit -am "feat(dag): real-beta-seeded planted-offset corpus; offset recovery through a 2-level closure (Test 2)"`

---

### Task 5: Test 3a — fallback / spurious-edge shrinkage (the headline)

**Files:**
- Test: `spark-vi/tests/test_pg_stm_dag.py`

**Interfaces:**
- Consumes: `PGSTMDag.fit` `node_norms`, `dag_offset_corpus`, `real_beta_from`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_pg_stm_dag.py
def test_fallback_spurious_node_offset_shrinks_to_near_zero():
    """PLANTED: offsets on a TREE only (root, two anchors) with a SPURIOUS extra subtype
    node whose true offset is 0. REAL: overlap beta. Realistic-overlap synthetic ->
    MATH-CORRECTNESS: an unearned node's offset norm shrinks far below an earned node's
    ('reduces to the simpler model where the data is tree-like'). Does NOT prove the
    SURVIVING structure is correct, only that unearned structure deactivates."""
    from spark_vi.models.topic.partition import TopicBlockPartition
    part = TopicBlockPartition(group_var="g", background_k=6, foreground=(("A", 4), ("B", 4)))
    K, V = part.K, 300
    beta = real_beta_from(K, V, seed=5)
    dag = DagGate([(), (0,), (0,), (1,)])            # node 3 = the spurious subtype
    Ksm1 = K - 1
    rng = np.random.default_rng(6)
    node_offsets = {0: np.zeros(Ksm1), 1: rng.standard_normal(Ksm1),
                    2: rng.standard_normal(Ksm1), 3: np.zeros(Ksm1)}   # 3 is truly 0
    docs, doc_nodes = dag_offset_corpus(
        dag=dag, node_offsets=node_offsets, partition=part, beta=beta,
        node_of_group={"A": 1, "B": 2}, doc_nodes_plan={1: 400, 2: 400, 3: 200},
        sigma_true=3.0 * np.eye(Ksm1), doc_len=80, seed=7)
    out = PGSTMDag(K=K, V=V, partition=part, dag=dag, P=docs[0].x.shape[0],
                   n_iter=60, lam_base=1e-2, gamma_depth=1.0, seed=0).fit(docs, doc_nodes)
    norms = out["node_norms"]
    # spurious node 3's offset must be much smaller than the earned anchors' offsets
    assert norms[3] < 0.25 * min(norms[1], norms[2]), \
        f"spurious node did not deactivate: norms={norms}"
```

- [ ] **Step 2: Run test, verify fail** (before Task 3/4 land) or **verify pass** once they do.

- [ ] **Step 3: (no new implementation)** — this test exercises Tasks 1-4. If it fails because the depth-scaled ridge under-shrinks (norm[3] not < 0.25×), record the measured ratio in the test and escalate: the plan's Open-question default (depth-scaled ridge) is falsified and the follow-up is a group-lasso penalty (`dag_offset_ridge` gains an L1 branch). Do not weaken the assertion to pass.

- [ ] **Step 4: Run test, verify pass** — PASS.

- [ ] **Step 5: Commit** — `git commit -am "test(dag): fallback spurious-node offset shrinks to ~0 (Test 3a)"`

---

### Task 6: Test 4 — offset-interval RELATIVE-uncertainty ordering (coverage deferred)

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/pg_stm_dag.py` (add `offset_cov_diag` to `fit` + `_design`)
- Test: `spark-vi/tests/test_pg_stm_dag.py`

**SCOPE DECISION (user, insight 0051):** the ridge-conditional intervals are a **relative**
read-out only. A coverage probe (offsets redrawn per rep, near-real config) measured absolute
coverage ≈ 0.13 vs nominal 0.90 — the intervals are built from mean-field-biased ψ-means and are
severely overconfident, so **no absolute-coverage assertion is made here** (calibrated intervals
are deferred to the read-out-honesty spec). What IS robust and asserted: the interval-**width
ordering** (data-scarce node wider than well-populated), which depends only on the fixed design
moments (`Ainv` from doc-counts), so it is stable and cheap. Individual anchor offsets are
un-identified under a partitioning gate (dummy trap, insight 0050) → the ordering is asserted on
**identified subtype increments**, not anchors.

**Interfaces:**
- Produces: `PGSTMDag.fit` also returns `"offset_cov_diag"` (per node, per stick posterior
  variance of B) from the ridge normal-equations covariance `sigma2 * (WtW+diag(penalty))^{-1}`
  at the converged ψ mean. This is a **relative** interval-width read-out (see scope decision).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_pg_stm_dag.py
def test_offset_interval_widths_order_scarce_above_populated():
    """PLANTED: node offsets + Sigma on root->anchor->{populated subtype, scarce subtype},
    with anchor-only docs so both subtype increments are IDENTIFIED. REAL: overlap beta.
    Realistic-overlap synthetic -> MATH-CORRECTNESS (RELATIVE uncertainty only): the ridge-
    posterior interval WIDTHS on the identified subtype increments ORDER correctly -- the
    data-scarce subtype's interval is WIDER than the well-populated subtype's (ratio > 1.5).
    We assert the ORDERING, NOT absolute coverage: a coverage probe (offsets redrawn per rep,
    near-real config) measured coverage ~0.13 vs nominal 0.90 -- the ridge-conditional posterior
    is built from mean-field-biased psi-means and is severely OVERconfident absolutely, so
    calibrated intervals are deferred to the read-out-honesty spec (insight 0051). We measure
    increments (subtype-vs-anchor), NOT the individual anchor offsets, which are un-identified
    under a partitioning gate (dummy trap, insight 0050) and carry no sample-size signal. Does
    NOT prove absolute coverage anywhere, nor transfer to real data (Task 7's spurious-edge
    check is the transfer-side guard). The width ratio depends only on the fixed design moments
    (Ainv from doc-counts), so it is stable across seeds/iterations."""
    from spark_vi.models.topic.partition import TopicBlockPartition
    part = TopicBlockPartition(group_var="g", background_k=6, foreground=(("A", 4), ("B", 4)))
    K, V = part.K, 120
    # 0 root; 1 anchor(A), 2 anchor(B); 3,4 = subtypes under anchor 1 (populated / scarce)
    dag = DagGate([(), (0,), (0,), (1,), (1,)])
    Ksm1 = K - 1
    lay = stick_layout(part)
    aA = lay["groups"]["A"]["active"]
    wide_pop = []; wide_scarce = []
    for rep in range(3):
        beta = real_beta_from(K, V, seed=200 + rep)
        rng = np.random.default_rng(9 + rep)
        node_offsets = {0: np.zeros(Ksm1), 1: rng.standard_normal(Ksm1),
                        2: rng.standard_normal(Ksm1), 3: rng.standard_normal(Ksm1),
                        4: rng.standard_normal(Ksm1)}
        docs, doc_nodes = dag_offset_corpus(
            dag=dag, node_offsets=node_offsets, partition=part, beta=beta,
            node_of_group={"A": 1, "B": 2},
            # anchor-A-only docs (node 1) identify the subtype increments; node 3 populated
            # (240) vs node 4 scarce (24, 10x fewer); node 2 keeps group B's block populated.
            doc_nodes_plan={1: 120, 2: 120, 3: 240, 4: 24},
            sigma_true=3.0 * np.eye(Ksm1), doc_len=50, seed=100 + rep)
        out = PGSTMDag(K=K, V=V, partition=part, dag=dag, P=docs[0].x.shape[0],
                       n_iter=25, lam_base=1e-3, seed=rep).fit(docs, doc_nodes)
        sd = np.sqrt(out["offset_cov_diag"])            # (U, K-1)
        wide_pop.append(float(np.mean(sd[3][aA])))       # populated subtype
        wide_scarce.append(float(np.mean(sd[4][aA])))    # scarce subtype
    ratio = np.mean(wide_scarce) / np.mean(wide_pop)
    assert ratio > 1.5, f"scarce subtype interval not wider than populated (ratio={ratio:.2f})"
```

- [ ] **Step 2: Run test, verify fail** — FAIL (`offset_cov_diag` missing).

- [ ] **Step 3: Implement `offset_cov_diag`** — in `PGSTMDag.fit`, after the loop, compute the ridge posterior covariance at the converged ψ mean:

```python
# in PGSTMDag.fit, replace the return with (after computing Gamma, B):
        # RELATIVE-uncertainty read-out only: sigma2 * (XtX+diag(penalty))^-1 at the VI psi-mean.
        # NOT calibrated for absolute coverage -- it omits psi posterior uncertainty AND is blind
        # to mean-field bias in the psi-means, so absolute coverage collapses (~0.13, insight 0051).
        # Only the cross-node WIDTH ORDERING (scarce wider) is trustworthy; calibrated absolute
        # intervals are the read-out-honesty spec's job.
        resid = psi_mean - self._design(docs_aug) @ Cf    # (D, K-1) on active-filled psi_mean
        sigma2 = max(float(np.mean(resid ** 2)), 1e-8)
        Ainv = np.linalg.inv(stats["XtX"] + np.diag(penalty))
        cov_diag_full = sigma2 * np.diag(Ainv)             # (P+U,)
        offset_cov_diag = np.repeat(cov_diag_full[self.P:][:, None], Ksm1, axis=1)
        return {"beta": beta, "Gamma": Gamma, "B": B, "Sigma": Sigma,
                "node_norms": np.linalg.norm(B, axis=1),
                "offset_cov_diag": offset_cov_diag, "psi_mean": psi_mean}
```

and add a small helper on the class:

```python
    def _design(self, docs_aug):
        return np.stack([np.asarray(d.x, dtype=np.float64) for d in docs_aug])
```

*(Ridge-conditional posterior at the VI ψ-mean — omits ψ uncertainty and mean-field bias, so it
is a RELATIVE read-out only. Absolute-coverage calibration and Σ posterior intervals are the
read-out-honesty spec, not here. See insight 0051.)*

- [ ] **Step 4: Run test, verify pass** — PASS (ratio ~2.0 > 1.5). The width ratio is n_iter- and
  seed-independent (it depends only on the design moments), so a rough fit suffices; do not add an
  absolute-coverage assertion (deferred, per the scope decision).

- [ ] **Step 5: Commit** — `git commit -am "feat(dag): ridge-posterior offset intervals (relative read-out) + scarce>populated width-ordering test (Test 4); coverage deferred (insight 0051)"`

---

### Task 7: `inject_spurious_edges` + Test 3b real-data hook (mechanical)

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/pg_stm_dag.py`
- Test: `spark-vi/tests/test_pg_stm_dag.py`

**Interfaces:**
- Produces: `inject_spurious_edges(dag, extra_parents, *, seed=0) -> DagGate` — returns a new `DagGate` with `n_spurious` random extra leaf nodes each parented to a random existing node (the real-data fallback hook: the actual real-corpus run is the OMOP-integration phase; this delivers + unit-tests the mechanism so that phase only wires data).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_pg_stm_dag.py
from spark_vi.models.topic.pg_stm_dag import inject_spurious_edges


def test_inject_spurious_edges_adds_random_leaves_and_shrinks_on_replay():
    """Mechanical check of the real-data fallback HOOK (Task-3b machinery). The real-data
    RUN (inject into the OMOP DAG, fit on the real corpus, verify injected offsets die) is
    the OMOP-integration phase; here we prove the injector produces valid extra leaves and
    that on a planted corpus (offsets truly 0 on injected nodes) their norms shrink. This
    test is synthetic and asserts MATH-CORRECTNESS of the hook only."""
    base = DagGate([(), (0,), (0,)])
    dag2 = inject_spurious_edges(base, extra_parents=[1, 2], seed=0)
    assert dag2.n_nodes == 5                                    # 2 injected leaves
    assert dag2.parents[3] in ((1,), (2,)) and dag2.parents[4] in ((1,), (2,))
    # injected nodes attest no documents -> their offset columns are never active -> ~0
    from spark_vi.models.topic.partition import TopicBlockPartition
    part = TopicBlockPartition(group_var="g", background_k=6, foreground=(("A", 4), ("B", 4)))
    K, V = part.K, 300
    beta = real_beta_from(K, V, seed=10)
    Ksm1 = K - 1; rng = np.random.default_rng(11)
    node_offsets = {0: np.zeros(Ksm1), 1: rng.standard_normal(Ksm1), 2: rng.standard_normal(Ksm1),
                    3: np.zeros(Ksm1), 4: np.zeros(Ksm1)}
    docs, doc_nodes = dag_offset_corpus(
        dag=dag2, node_offsets=node_offsets, partition=part, beta=beta,
        node_of_group={"A": 1, "B": 2}, doc_nodes_plan={1: 300, 2: 300},   # nobody at 3,4
        sigma_true=3.0 * np.eye(Ksm1), doc_len=80, seed=12)
    out = PGSTMDag(K=K, V=V, partition=part, dag=dag2, P=docs[0].x.shape[0],
                   n_iter=50, lam_base=1e-2, seed=0).fit(docs, doc_nodes)
    assert out["node_norms"][3] < 1e-6 and out["node_norms"][4] < 1e-6
```

- [ ] **Step 2: Run test, verify fail** — FAIL (`inject_spurious_edges` not defined).

- [ ] **Step 3: Implement**

```python
# append to pg_stm_dag.py
def inject_spurious_edges(dag, extra_parents, *, seed=0):
    """Return a new DagGate with one extra leaf node per entry in ``extra_parents`` (each
    value = the parent node id). Used by the real-data fallback check (Test 3b): inject
    random cross-edges into the real DAG, fit on the real corpus, and verify the injected
    offsets die (they are spurious BY CONSTRUCTION, so no planted truth is needed). The
    real-corpus run is the OMOP-integration phase; this builds and unit-tests the
    injector."""
    parents = [list(ps) for ps in dag.parents]
    for p in extra_parents:
        parents.append([int(p)])       # new leaf appended -> index > its parent, stays topo-ordered
    return DagGate(parents)
```

- [ ] **Step 4: Run test, verify pass** — PASS.

- [ ] **Step 5: Commit** — `git commit -am "feat(dag): spurious-edge injector + mechanical real-data-fallback hook test (Test 3b)"`

---

## Self-Review

**Spec coverage:** Test 1 = spec Test 1 (Task 3); Test 2 offset recovery through a 2-level closure = spec Test 2 (Task 4); Test 3a spurious-edge shrinkage = spec Test 3a (Task 5); Test 3b real-data hook = spec Test 3b, delivered as the injector + mechanical test with the real run deferred to OMOP phase (Task 7); Test 4 coverage = spec Test 4, offset intervals only (Σ intervals deferred to the read-out spec, per the spec's open-question default) (Task 6). Sparse depth-scaled prior = `offset_penalty` (Task 2). Closure/DagGate = Task 1. The additive-η mean via augmented covariate = Task 3. Deferred by design (v1 = Piece A): node-owned topics + multi-level gate/Σ composition.

**Placeholder scan:** no TBD/TODO; every code step carries runnable code; escalation branches (ridge under-shrinks → group-lasso; coverage low → ψ-uncertainty omission) name the concrete next action instead of hand-waving.

**Type consistency:** `DagGate(parents)` constructor used identically in Tasks 1,3,4,5,6,7; `closure_indicator` returns float64 (P-augmentation concatenates cleanly); `offset_penalty(P, dag, *, gamma_ridge, lam_base, gamma_depth)` and `dag_offset_ridge(WtW, WtM, *, penalty)` signatures match their calls in `PGSTMDag.fit`; `fit(docs, doc_nodes)` return keys (`beta, Gamma, B, Sigma, node_norms, offset_cov_diag, psi_mean`) are the ones every test reads.

**Known risk to watch during execution:** Task 3's equivalence rests on the root offset being collinear with the covariate intercept; if `x` has no intercept in some corpus, the reparam argument weakens — the plan's corpora use `x=[1.0]` (intercept present), so this holds for all tasks here.
