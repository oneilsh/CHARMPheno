# DAG Offset Increment Reparameterization + Ordinal Read-Out Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drop the root offset column so each non-root node's coefficient is its parent→child increment (the identified coordinates, insight 0050), and replace the raw offset-variance read-out with an honest ordinal artifact (rank / parent-ratio / identified flag + `calibration:"ordinal"`, insights 0051/0052).

**Architecture:** All changes are in `spark_vi/models/topic/pg_stm_dag.py` and `tests/test_pg_stm_dag.py`. Part A (Tasks 1–3) changes the offset *design*: a non-root closure indicator, a root-dropped penalty, and a `PGSTMDag.fit` that augments with the non-root indicator and remaps `B` back to node-id indexing with `B[root]≡0`. Part B (Task 4) replaces `offset_cov_diag` with a σ²-free `offset_uncertainty` object. Task 5 adds path-sum / no-direct-docs-anchor validation. The E-step, Σ M-step, β M-step, and gate are untouched.

**Tech Stack:** Python / NumPy / SciPy, single-machine, mirroring `pg_stm.py`.

## Global Constraints

- Domain-agnostic engine layer: integer node ids + integer token ids only; never concept names/ids or clinical vocabulary in code, comments, or docstrings.
- Test-honesty rule: every test docstring states what is *planted* (synthetic truth) vs *real* (β from an existing fit / realistic overlap), where it sits on the synthetic→real spectrum, and the claim it supports **and** the claim it does not. No test asserts a *transfer* claim from a *synthetic* result — synthetic proves math-correctness only.
- Reuse `pg_stm.py` primitives unchanged (`pg_empty_stats`, `pg_accumulate_doc`, `pg_estep_doc`, `beta_dirichlet_mean`, `assemble_sigma`, `stick_layout`); do not fork them.
- Cite any literature-sourced method/default in its docstring; an uncited constant is labeled a heuristic.
- No LaTeX in prose (plain text + Unicode where needed).
- All test commands run from `spark-vi/`: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno/spark-vi && python -m pytest tests/test_pg_stm_dag.py -q`. Commit from the repo root. Branch `pg-stm`. Do not `git add` untracked scratch (`dashboard/public/data/...`, `node_modules/`, `spark-vi/tests/test_t3b_diag_tmp.py`).
- End commit messages with: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

### Task 1: `DagGate.offset_indicator` + `n_offset_nodes`

Adds the non-root closure indicator (the offset design drops the root column, which equals the covariate intercept). Node 0 is always the root; the offset block covers nodes `1..U-1`.

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/pg_stm_dag.py` (add two members to `DagGate`, after `closure_indicator`, ~line 68)
- Test: `spark-vi/tests/test_pg_stm_dag.py`

**Interfaces:**
- Consumes: `DagGate.closure_indicator` (unchanged).
- Produces: `DagGate.n_offset_nodes -> int` (= `n_nodes - 1`); `DagGate.offset_indicator(nodes) -> np.ndarray (n_nodes-1,) float64` = the closure indicator with the root entry (index 0) dropped.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_pg_stm_dag.py
def test_offset_indicator_drops_the_root_entry():
    """Deterministic structure check; no empirical or transfer claim. The offset design
    omits the root column (it equals the covariate intercept), so offset_indicator is the
    closure indicator over non-root nodes 1..U-1."""
    dag = DagGate([(), (0,), (0,), (1,)])          # root; anchors 1,2; subtype 3 under 1
    assert dag.n_offset_nodes == 3
    z = dag.offset_indicator(frozenset({3}))       # closure {3,1,0}; drop root -> nodes 1,2,3
    assert z.dtype == np.float64
    assert list(z) == [1.0, 0.0, 1.0]              # node1 on, node2 off, node3 on
    assert list(dag.offset_indicator(frozenset({2}))) == [0.0, 1.0, 0.0]
```

- [ ] **Step 2: Run tests, verify fail** — `pytest tests/test_pg_stm_dag.py::test_offset_indicator_drops_the_root_entry -q` → FAIL (`AttributeError`).

- [ ] **Step 3: Implement** (add to `DagGate`, right after `closure_indicator`)

```python
    @property
    def n_offset_nodes(self) -> int:
        """Number of non-root nodes (the offset block covers nodes 1..n_nodes-1; the root
        column is dropped because it equals the covariate intercept)."""
        return self.n_nodes - 1

    def offset_indicator(self, nodes) -> np.ndarray:
        """Closure indicator over the NON-root nodes (drops index 0). Length n_nodes-1,
        entry i corresponds to node i+1."""
        return self.closure_indicator(nodes)[1:]
```

- [ ] **Step 4: Run tests, verify pass** — PASS.

- [ ] **Step 5: Commit** — `git commit -am "feat(dag): DagGate.offset_indicator/n_offset_nodes (non-root offset design)"`

---

### Task 2: `offset_penalty` drops the root row

The penalty must match the root-dropped design: covariate rows + one row per non-root node (depths of nodes `1..U-1`).

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/pg_stm_dag.py` (`offset_penalty`, lines 75-84)
- Test: `spark-vi/tests/test_pg_stm_dag.py` (update `test_offset_penalty_is_depth_scaled_on_node_block_only`)

**Interfaces:**
- Consumes: `DagGate.depth`.
- Produces: `offset_penalty(P, dag, *, gamma_ridge, lam_base, gamma_depth) -> np.ndarray (P + n_nodes-1,)` — first `P` rows `gamma_ridge`; then `lam_base*(1+depth[u])**gamma_depth` for u in `1..n_nodes-1` (root excluded).

- [ ] **Step 1: Update the failing test** (replace `test_offset_penalty_is_depth_scaled_on_node_block_only`)

```python
def test_offset_penalty_is_depth_scaled_on_non_root_node_block():
    """Deterministic linear-algebra check; no empirical or transfer claim. The penalty
    excludes the root (its offset column is dropped) and depth-scales the non-root rows."""
    dag = DagGate([(), (0,), (1,)])            # depths 0,1,2 ; non-root nodes 1,2 (depths 1,2)
    pen = offset_penalty(P=2, dag=dag, gamma_ridge=1e-6, lam_base=2.0, gamma_depth=1.0)
    assert pen.shape == (2 + 2,)               # P covariate rows + (n_nodes-1) offset rows
    assert np.allclose(pen[:2], 1e-6)
    assert np.allclose(pen[2:], [2.0 * 2, 2.0 * 3])   # lam_base*(1+depth) for nodes 1,2
```

- [ ] **Step 2: Run test, verify fail** — FAIL (shape `(2+3,)` from the old impl).

- [ ] **Step 3: Implement** (replace `offset_penalty` body)

```python
def offset_penalty(P, dag, *, gamma_ridge, lam_base, gamma_depth):
    """(P + n_nodes-1,) ridge penalty: ``gamma_ridge`` on each covariate row, and
    ``lam_base * (1 + depth[u]) ** gamma_depth`` on each NON-root node-offset row (nodes
    1..n_nodes-1; the root column is dropped because it equals the covariate intercept).
    Depth-scaling (deeper => larger penalty) encodes "prefer general explanations,
    specialize only on evidence" (a structural, inspectable shrinkage; not an inference
    hyperparameter) and softly attributes a no-direct-docs internal node's shared child
    signal to the shallower ancestor. A node whose offset column is never active is pulled
    to 0 by its penalty."""
    pen = np.empty(int(P) + dag.n_offset_nodes, dtype=np.float64)
    pen[:P] = float(gamma_ridge)
    depth_nonroot = dag.depth.astype(np.float64)[1:]
    pen[P:] = float(lam_base) * (1.0 + depth_nonroot) ** float(gamma_depth)
    return pen
```

- [ ] **Step 4: Run test, verify pass** — PASS. Also run `test_dag_offset_ridge_recovers_well_posed_coefficients` and `test_dag_offset_ridge_shrinks_an_unconstrained_column_to_zero` (unchanged, still green — they call `dag_offset_ridge` directly with explicit penalties).

- [ ] **Step 5: Commit** — `git commit -am "feat(dag): offset_penalty drops the root row (matches non-root offset design)"`

---

### Task 3: `PGSTMDag.fit` — drop-root design + node-id-indexed offsets

Augment with `offset_indicator` (non-root), solve the full-rank ridge, and remap `B` back to node-id indexing with `B[root]≡0`. Keep `offset_cov_diag` for now (Task 4 replaces it) but node-id-indexed so Test 4 keeps passing.

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/pg_stm_dag.py` (`PGSTMDag.fit`, lines 120-172; `self.U` init line 112)
- Test: `spark-vi/tests/test_pg_stm_dag.py` (update `test_pgstmdag_root_only_matches_flat_pgstmvi`)

**Interfaces:**
- Consumes: `DagGate.offset_indicator/n_offset_nodes`, `offset_penalty` (root-dropped), `dag_offset_ridge`, the pg_stm primitives.
- Produces: `PGSTMDag.fit` returns `B` shape `(U, K-1)` with row 0 (root) all-zero and rows `1..U-1` the estimated increments; `node_norms` length `U` (root=0); `offset_cov_diag` shape `(U, K-1)` node-id-indexed (root row 0). `Gamma`, `Sigma`, `beta`, `psi_mean` unchanged.

- [ ] **Step 1: Update the equivalence test**

```python
def test_pgstmdag_root_only_matches_flat_pgstmvi():
    """PLANTED: a stick-native gated corpus. REAL: nothing. Synthetic -> MATH-CORRECTNESS
    only: with a root-only DAG the offset block is EMPTY (the root column is dropped), so
    PGSTMDag is exactly PGSTMVI and must return the SAME beta and Sigma, with an all-zero
    root offset row. Proves the drop-root augmentation does not perturb the validated flat
    model. Does NOT prove anything about multi-level DAG behavior."""
    docs, part, _St, _b = gated_ln_corpus_stick(
        group_weights={"A": 0.5, "B": 0.5}, fg_per_group=2, bg_k=3, V=60, D=300,
        doc_len=40, seed=0)
    P = docs[0].x.shape[0]
    vi = PGSTMVI(K=part.K, V=60, partition=part, P=P, n_iter=40, seed=0).fit(docs)
    dag = root_only_dag()
    doc_nodes = [frozenset({0})] * len(docs)
    out = PGSTMDag(K=part.K, V=60, partition=part, dag=dag, P=P, n_iter=40,
                   gamma_ridge=1e-6, lam_base=1e-6, gamma_depth=1.0, seed=0).fit(docs, doc_nodes)
    assert np.allclose(out["beta"], vi["beta"], atol=2e-3)
    assert np.allclose(out["Sigma"], vi["Sigma"], atol=2e-3)
    assert out["B"].shape == (1, part.K - 1)                # one node row (the root)...
    assert np.allclose(out["B"][0], 0.0)                    # ...forced to zero (not estimated)
```

- [ ] **Step 2: Run test, verify fail** — FAIL (root-only currently yields a non-zero root offset row / shape mismatch).

- [ ] **Step 3: Implement** — replace `PGSTMDag.fit` (keep `__init__` but note `self.U` stays `dag.n_nodes`; add `self.U_off = dag.n_offset_nodes` on the line after `self.P, self.U = P, dag.n_nodes`):

```python
        self.P, self.U = P, dag.n_nodes
        self.U_off = dag.n_offset_nodes
```

```python
    def fit(self, docs, doc_nodes):
        rng = np.random.default_rng(self.seed)
        K, V, Pw = self.K, self.V, self.P + self.U_off
        Ksm1 = K - 1
        # augment each doc's covariate with its NON-root closure indicator: w = [x ; z_nonroot]
        docs_aug = []
        for doc, nodes in zip(docs, doc_nodes):
            z = self.dag.offset_indicator(nodes)
            w = np.concatenate([np.asarray(doc.x, dtype=np.float64), z])
            docs_aug.append(dataclasses.replace(doc, x=w))
        penalty = offset_penalty(self.P, self.dag, gamma_ridge=self.gamma_ridge,
                                 lam_base=self.lam_base, gamma_depth=self.gamma_depth)
        beta = rng.random((K, V)) + self.beta_eta
        beta /= beta.sum(axis=1, keepdims=True)
        Cf = np.zeros((Pw, Ksm1))                 # [Gamma ; B_nonroot] stacked
        Sigma = np.eye(Ksm1)
        D = len(docs_aug)
        psi_mean = np.zeros((D, Ksm1))
        stats = pg_empty_stats(K, V, Pw, self.partition.groups)   # bound for the read-out below
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
        Gamma = Cf[:self.P]
        B_nonroot = Cf[self.P:]                             # (U_off, K-1) for nodes 1..U-1
        B = np.zeros((self.U, Ksm1))                        # node-id-indexed; root row 0 stays 0
        B[1:] = B_nonroot
        # RELATIVE-uncertainty read-out (replaced by the ordinal object in the next task).
        resid = psi_mean - self._design(docs_aug) @ Cf
        sigma2 = max(float(np.mean(resid ** 2)), 1e-8)
        Ainv = np.linalg.inv(stats["XtX"] + np.diag(penalty))
        var_nonroot = sigma2 * np.diag(Ainv)[self.P:]       # (U_off,)
        var_by_node = np.zeros(self.U); var_by_node[1:] = var_nonroot   # root=0
        offset_cov_diag = np.repeat(var_by_node[:, None], Ksm1, axis=1) # (U, K-1) node-indexed
        return {"beta": beta, "Gamma": Gamma, "B": B, "Sigma": Sigma,
                "node_norms": np.linalg.norm(B, axis=1),
                "offset_cov_diag": offset_cov_diag, "psi_mean": psi_mean}
```

- [ ] **Step 4: Run tests, verify pass** — run the whole file: `pytest tests/test_pg_stm_dag.py -q`. The equivalence test passes; `test_offset_recovery_through_two_level_closure` still passes (B is node-id-indexed as before, root row now exactly 0); `test_offset_interval_widths_order_scarce_above_populated` still passes (offset_cov_diag is node-id-indexed `(U,K-1)`, so `sd[3]`, `sd[4]` still address nodes 3,4 and the design-moment ratio is unchanged ~2.0); `test_fallback_spurious_node_offset_shrinks_to_near_zero` still passes (node_norms node-id-indexed). If the equivalence β/Σ drift above 2e-3, raise `n_iter` to 60 in both fits; do not loosen `atol` beyond 3e-3.

- [ ] **Step 5: Commit** — `git commit -am "feat(dag): drop-root offset design; node-id-indexed B with B[root]=0; root-only == flat PGSTMVI"`

---

### Task 4: Ordinal read-out (`offset_uncertainty`), remove `offset_cov_diag`

Replace the raw-width read-out with a σ²-free ordinal object. rank, parent_ratio, and the identified ratio all cancel the global σ², so this drops the resid/σ²/`_design` machinery.

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/pg_stm_dag.py` (`PGSTMDag.fit` read-out block + remove `_design`)
- Test: `spark-vi/tests/test_pg_stm_dag.py` (rewrite `test_offset_interval_widths_order_scarce_above_populated`; add an identified-flag test)

**Interfaces:**
- Produces: `PGSTMDag.fit` returns `offset_uncertainty` instead of `offset_cov_diag`. `offset_uncertainty` is a dict:
  - `"calibration"`: the string `"ordinal"`.
  - `"rank"`: `np.ndarray (U,) int` — 1-based uncertainty rank among non-root nodes (1 = most resolved / smallest variance … `U-1` = least resolved); root entry `0` (sentinel, not ranked).
  - `"parent_ratio"`: `np.ndarray (U,) float` — node variance ÷ its min-depth parent's variance; `nan` for the root and for any node whose min-depth parent is the root (no offset variance there).
  - `"identified"`: `np.ndarray (U,) bool` — `True` when the ridge posterior variance is below half the prior variance (the likelihood contributed information); root entry `False`.
  `B`, `node_norms`, `Gamma`, `Sigma`, `beta`, `psi_mean` unchanged. `offset_cov_diag` is removed.

- [ ] **Step 1: Rewrite Test 4 + add the identified-flag test**

```python
def test_offset_uncertainty_is_ordinal_ranks_scarce_above_populated():
    """PLANTED: node offsets + Sigma on root->anchor->{populated subtype, scarce subtype}
    with anchor-only docs so both increments are identified. REAL: overlap beta.
    Realistic-overlap synthetic -> MATH-CORRECTNESS (RELATIVE uncertainty only): the ordinal
    read-out ranks the data-scarce subtype as LESS resolved than the populated subtype
    (rank[scarce] > rank[populated]) and its calibration status is 'ordinal'. Rank is a
    design-moment property (independent of sigma^2 / iterations). We assert ORDERING, not
    absolute coverage: those intervals are overconfident (~0.13 vs 0.90, insight 0051) and
    calibrated absolute intervals are deferred to the read-out-honesty engine. Anchor
    offsets are un-identified under a partitioning gate (dummy trap, insight 0050) so we
    measure identified subtype increments. Does NOT prove absolute coverage or transfer."""
    from spark_vi.models.topic.partition import TopicBlockPartition
    part = TopicBlockPartition(group_var="g", background_k=6, foreground=(("A", 4), ("B", 4)))
    K, V = part.K, 120
    dag = DagGate([(), (0,), (0,), (1,), (1,)])       # 0 root; 1,2 anchors; 3,4 subtypes under 1
    Ksm1 = K - 1
    ranks_scarce = []; ranks_pop = []
    for rep in range(3):
        beta = real_beta_from(K, V, seed=200 + rep)
        rng = np.random.default_rng(9 + rep)
        node_offsets = {u: rng.standard_normal(Ksm1) for u in (1, 2, 3, 4)}
        node_offsets[0] = np.zeros(Ksm1)
        docs, doc_nodes = dag_offset_corpus(
            dag=dag, node_offsets=node_offsets, partition=part, beta=beta,
            node_of_group={"A": 1, "B": 2},
            doc_nodes_plan={1: 120, 2: 120, 3: 240, 4: 24},   # node 3 populated, node 4 scarce
            sigma_true=3.0 * np.eye(Ksm1), doc_len=50, seed=100 + rep)
        out = PGSTMDag(K=K, V=V, partition=part, dag=dag, P=docs[0].x.shape[0],
                       n_iter=25, lam_base=1e-3, seed=rep).fit(docs, doc_nodes)
        ou = out["offset_uncertainty"]
        assert ou["calibration"] == "ordinal"
        assert "offset_cov_diag" not in out               # no raw widths exported
        ranks_pop.append(int(ou["rank"][3])); ranks_scarce.append(int(ou["rank"][4]))
    assert np.mean(ranks_scarce) > np.mean(ranks_pop), (
        f"scarce subtype not ranked less-resolved: scarce={ranks_scarce} pop={ranks_pop}")


def test_identified_flag_true_for_populated_false_for_zero_doc_node():
    """PLANTED: offsets on root->2 anchors, one anchor's subtype well-populated, plus a
    ZERO-doc extra node. REAL: overlap beta. Realistic-overlap synthetic -> MATH-CORRECTNESS:
    the `identified` flag is True for a well-populated distinct node (data halves the prior
    variance) and False for a node with no attesting documents (posterior variance == prior
    variance, ratio 1). Proves the flag distinguishes data-identified from prior-dominated
    offsets. Does NOT prove real-data identifiability or transfer."""
    from spark_vi.models.topic.partition import TopicBlockPartition
    part = TopicBlockPartition(group_var="g", background_k=6, foreground=(("A", 4), ("B", 4)))
    K, V = part.K, 120
    dag = DagGate([(), (0,), (0,), (1,), (1,)])       # node 4 will attest NO documents
    Ksm1 = K - 1
    rng = np.random.default_rng(3)
    node_offsets = {u: rng.standard_normal(Ksm1) for u in (1, 2, 3)}
    node_offsets[0] = np.zeros(Ksm1); node_offsets[4] = np.zeros(Ksm1)
    beta = real_beta_from(K, V, seed=7)
    docs, doc_nodes = dag_offset_corpus(
        dag=dag, node_offsets=node_offsets, partition=part, beta=beta,
        node_of_group={"A": 1, "B": 2},
        doc_nodes_plan={1: 200, 2: 200, 3: 300},          # node 4 absent -> zero-doc column
        sigma_true=3.0 * np.eye(Ksm1), doc_len=50, seed=5)
    out = PGSTMDag(K=K, V=V, partition=part, dag=dag, P=docs[0].x.shape[0],
                   n_iter=25, lam_base=1e-3, seed=0).fit(docs, doc_nodes)
    ident = out["offset_uncertainty"]["identified"]
    assert ident[3] == True, "well-populated subtype should be data-identified"
    assert ident[4] == False, "zero-doc node should be prior-dominated"
```

- [ ] **Step 2: Run tests, verify fail** — FAIL (`offset_uncertainty` missing / `offset_cov_diag` still present).

- [ ] **Step 3: Implement** — in `PGSTMDag.fit`, replace the read-out block (from `resid = ...` through the `return`) with:

```python
        Gamma = Cf[:self.P]
        B_nonroot = Cf[self.P:]
        B = np.zeros((self.U, Ksm1)); B[1:] = B_nonroot
        # Ordinal (RELATIVE) read-out. rank / parent_ratio / the identified ratio all cancel
        # the global sigma^2, so no residual-scale estimate is needed. NOT calibrated for
        # absolute coverage (built from mean-field-biased psi-means; ~0.13 vs 0.90, insight
        # 0051); calibrated absolute intervals are the read-out-honesty engine's job (0052).
        Ainv = np.linalg.inv(stats["XtX"] + np.diag(penalty))
        ainv_off = np.diag(Ainv)[self.P:]                 # (U_off,) proportional to node variance
        pen_off = penalty[self.P:]
        # fraction of prior variance remaining after the data (shrinkage factor); a heuristic
        # threshold of 0.5 flags "data at least halved the prior variance" as identified.
        remaining = pen_off * ainv_off                    # sigma^2 cancels
        var_full = np.full(self.U, np.inf); var_full[1:] = ainv_off   # root has no offset variance
        # 1-based uncertainty rank among non-root nodes (1 = smallest variance = most resolved)
        order = np.argsort(ainv_off, kind="stable")
        rank_nonroot = np.empty(self.U_off, dtype=np.int64)
        rank_nonroot[order] = np.arange(1, self.U_off + 1)
        rank = np.zeros(self.U, dtype=np.int64); rank[1:] = rank_nonroot
        identified = np.zeros(self.U, dtype=bool); identified[1:] = remaining < 0.5
        parent_ratio = np.full(self.U, np.nan)
        for u in range(1, self.U):
            ps = self.dag.parents[u]
            if not ps:
                continue
            p = min(ps, key=lambda q: (int(self.dag.depth[q]), q))   # min-depth parent (tiebreak id)
            if p == 0:
                continue                                   # parent is root -> no offset variance
            parent_ratio[u] = var_full[u] / var_full[p]
        offset_uncertainty = {"calibration": "ordinal", "rank": rank,
                              "parent_ratio": parent_ratio, "identified": identified}
        return {"beta": beta, "Gamma": Gamma, "B": B, "Sigma": Sigma,
                "node_norms": np.linalg.norm(B, axis=1),
                "offset_uncertainty": offset_uncertainty, "psi_mean": psi_mean}
```

Then delete the now-unused `_design` method (lines 171-172; grep the file first to confirm no other caller).

- [ ] **Step 4: Run tests, verify pass** — `pytest tests/test_pg_stm_dag.py -q`. The two new tests pass; confirm no remaining reference to `offset_cov_diag` in the file (`grep -n offset_cov_diag`). If the `identified[3]` assertion is borderline (a very scarce populated node), raise `doc_nodes_plan[3]`; the zero-doc `identified[4]==False` is exact (ratio == 1).

- [ ] **Step 5: Commit** — `git commit -am "feat(dag): ordinal offset_uncertainty read-out (rank/parent_ratio/identified + calibration); drop raw widths (insights 0051/0052)"`

---

### Task 5: Path-sum recovery through a no-direct-docs anchor

The always-identified quantity is the leaf **path-sum** (sum of increments over a leaf's closure). Validate it recovers for both a with-direct-docs anchor and a no-direct-docs anchor (all members at a subtype) — the user's real scenario.

**Files:**
- Test: `spark-vi/tests/test_pg_stm_dag.py`

**Interfaces:**
- Consumes: `PGSTMDag.fit` (`B` node-id-indexed), `dag_offset_corpus`, `real_beta_from`, `DagGate.closure`, `stick_layout`.

- [ ] **Step 1: Write the test**

```python
def test_path_sums_recover_through_with_and_without_direct_doc_anchors():
    """PLANTED: node offsets + Sigma on root->{anchor A with anchor-level docs + subtype;
    anchor B with ONLY a subtype (no anchor-level docs)}. REAL: overlap beta. Realistic-
    overlap synthetic -> MATH-CORRECTNESS: the leaf PATH-SUM (sum of increments over the
    leaf's closure) is the always-identified quantity and recovers for BOTH a with-direct-
    docs anchor and a no-direct-docs anchor (all members coded at the subtype). Proves the
    drop-root reparam recovers what the data identifies even when an internal anchor has no
    direct documents. Does NOT prove the anchor/subtype SPLIT is recoverable there (it is
    prior-driven), nor transfer to real data."""
    from spark_vi.models.topic.partition import TopicBlockPartition
    part = TopicBlockPartition(group_var="g", background_k=6, foreground=(("A", 4), ("B", 4)))
    K, V = part.K, 300
    dag = DagGate([(), (0,), (0,), (1,), (2,)])       # A=1 (has anchor-level docs), sub A1=3;
                                                       # B=2 (NO direct docs), sub B1=4
    Ksm1 = K - 1
    rng = np.random.default_rng(4)
    node_offsets = {u: rng.standard_normal(Ksm1) for u in (1, 2, 3, 4)}
    node_offsets[0] = np.zeros(Ksm1)
    beta = real_beta_from(K, V, seed=2)
    docs, doc_nodes = dag_offset_corpus(
        dag=dag, node_offsets=node_offsets, partition=part, beta=beta,
        node_of_group={"A": 1, "B": 2},
        doc_nodes_plan={1: 400, 3: 400, 4: 500},          # A has anchor-level (node 1) + sub;
                                                           # B (node 2) has NO direct docs
        sigma_true=3.0 * np.eye(Ksm1), doc_len=80, seed=6)
    out = PGSTMDag(K=K, V=V, partition=part, dag=dag, P=docs[0].x.shape[0],
                   n_iter=60, lam_base=1e-3, seed=0).fit(docs, doc_nodes)
    B = out["B"]                                           # node-id-indexed, root row 0
    lay = stick_layout(part)
    for leaf, group in ((3, "A"), (4, "B")):
        act = lay["groups"][group]["active"]
        recon = np.zeros(Ksm1)
        for u in dag.closure(frozenset({leaf})):
            recon = recon + B[u]                          # B[root]=0
        truth = np.zeros(Ksm1)
        for u in dag.closure(frozenset({leaf})):
            truth = truth + node_offsets[u]
        r = np.corrcoef(recon[act], truth[act])[0, 1]
        assert r > 0.6, f"path-sum not recovered for leaf {leaf} (group {group}): r={r:.2f}"
```

- [ ] **Step 2: Run test, verify pass** — the machinery from Tasks 1-4 already supports this (no new implementation). Run `pytest tests/test_pg_stm_dag.py::test_path_sums_recover_through_with_and_without_direct_doc_anchors -q`. If a path-sum `r` is borderline, raise `n_iter` to 90 or `doc_len` to 100; do not lower the 0.6 threshold without recording why in the test. If the no-direct-docs leaf (4) fails while leaf 3 passes, that is a real finding about no-direct-docs recovery — report BLOCKED with both measured `r` values rather than weakening the assertion.

- [ ] **Step 3: Run the full suite** — `pytest tests/test_pg_stm_dag.py -q`; confirm all pass (Tasks 1-5 plus the pre-existing DagGate/injection tests).

- [ ] **Step 4: Commit** — `git commit -am "test(dag): path-sum recovery through with- and no-direct-docs anchors (drop-root reparam honesty)"`

---

## Self-Review

**Spec coverage:** Part A drop-root reparam = Tasks 1 (offset_indicator), 2 (offset_penalty), 3 (fit + Test 1 equivalence, B[root]≡0, full-rank design). Depth-scaled soft-identification of no-direct-docs anchors = exercised by Task 5 (path-sum recovers) and the `identified` flag (Task 4). Part B ordinal read-out (rank / parent_ratio / identified / `calibration:"ordinal"`, no raw widths) = Task 4. Validation: path-sums both anchor cases = Task 5; ordinal rank scarce>populated + calibration + identified flag = Task 4. Test 1 update = Task 3; Test 4 replacement = Task 4. Domain-agnostic + honesty docstrings = every task. The spec's "background/root-only arm" is intentionally not built: it would require the gate to accept group-less documents (out of scope for Part A); the no-direct-docs anchor + zero-doc node give the honesty signal without touching the gate — noted here so the omission is deliberate, not a gap.

**Placeholder scan:** no TBD/TODO; every code step carries runnable code; escalation branches (equivalence drift → raise n_iter; identified borderline → raise doc count; no-direct-docs path-sum fails → report BLOCKED with numbers) name concrete actions.

**Type consistency:** `offset_indicator`/`n_offset_nodes` (Task 1) consumed by `offset_penalty` (Task 2, `dag.n_offset_nodes`) and `fit` (Task 3). `fit` return keys — `beta, Gamma, B, Sigma, node_norms, psi_mean` plus `offset_cov_diag` (Task 3) → replaced by `offset_uncertainty` (Task 4) — match every test's reads: Task 4 reads `offset_uncertainty["calibration"|"rank"|"identified"]`; Task 5 reads `B`. `B` is `(U, K-1)` node-id-indexed with row 0 zero throughout. `offset_uncertainty` arrays are all length `U` with root sentinels (rank 0, parent_ratio nan, identified False).

**Known risk to watch:** Task 3's `stats` must remain bound after the loop for the read-out `Ainv` — the plan initializes `stats` once before the loop and reassigns inside, so the final-iteration `stats` is in scope (guard against an empty-loop edge only if `n_iter==0`, which no task uses).
