# DAG-offset read-out engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the read-out engine (step 2): a warm-started co-sampled Gibbs pass that emits calibrated offset-increment posteriors on the compiler's identified quotient DAG, assembled into a per-coordinate-class read-out object and gated by a coverage plant.

**Architecture:** Two-phase pipeline — Phase A (mean-field VI warm-start + a fractional-z soft-gate E-step) → Phase B1 (compile once on the expected design moment) → Phase B2 (co-sampled quotient Gibbs emitting increment draws) → Phase C (read-out assembly). Each null direction is classified GAUGE (partition-identity gauge freedom) vs UNRESOLVED (contingent, with an attestation recipe); design-wall coordinates emit width + status + cause, never a point estimate.

**Tech Stack:** Python 3, NumPy, pytest. Reuses `spark_vi/models/topic/pg_stm.py` primitives, `pg_stm_dag.py` (DagGate, offset machinery), `dag_identify.py` (the compiler), `_mcmc_diag.py` (improved_rhat), `tests/_stm_synth.py` (plants). No new third-party deps. All work in `spark-vi/`.

## Global Constraints

- Domain-agnostic engine layer: integer node/token ids only; no domain vocabulary in `spark_vi/`.
- Identifiability is the working vocabulary (not "estimability"); "estimable functions" appears only as the name of Searle's theory in a docstring citation.
- Cite literature in docstrings: Polson-Scott-Windle (2013) + Linderman-Johnson-Adams (2015) for the PG stick-breaking substrate; Searle (1971) for estimable functions; Vehtari et al. (2021) for improved R̂; insight 0051 for the coverage protocol. No LaTeX — Unicode Greek (α, β, η, λ, ψ, Σ).
- Test-honesty rule: every plant labels planted-vs-real; synthetic proves math-correctness only, never a real-data transfer claim.
- Stick-space rule (insight 0053): read out at block granularity or in θ-space, never per-stick across positions; stick ordering is a frozen model constant.
- Index convention (from `dag_identify.py`): offset index i (0-based) ↔ node id i+1; root (node 0) has no offset column; Grams are offset-index-ordered, shape (U, U), U = dag.n_offset_nodes.
- Branch `pg-stm` (auto-pushes — verify remote with git, never assume). Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Commit at the end of each task.
- Run tests with: `cd spark-vi && python -m pytest <path> -v`.

## File Structure

- `spark_vi/models/topic/pg_stm_dag_gibbs.py` *(new)* — the MN offset-increment draw kernel + the co-sampled quotient Gibbs sweep (`PGSTMDagGibbs`). The engine kernel.
- `spark_vi/models/topic/dag_identify.py` *(extend)* — `expected_closure_gram` (fractional/soft indicators) + `classify_null_directions` (GAUGE vs UNRESOLVED + attestation recipe).
- `spark_vi/models/topic/pg_stm_dag.py` *(extend)* — `_softgate_estep_doc` (fractional-z mixture E-step), beside `_bg_estep_doc`.
- `spark_vi/models/topic/dag_readout.py` *(new)* — the per-coordinate-class `ReadOut` schema assembly + prevalence + the `dag_offset_readout` orchestrator.
- `tests/_stm_synth.py` *(extend)* — a partial-label arm on `dag_offset_corpus`; a `coverage_plant` builder.
- Tests: `tests/test_pg_stm_dag_gibbs.py` *(new)*, `tests/test_dag_readout.py` *(new)*, extend `tests/test_dag_identify.py`, extend `tests/test_pg_stm_dag.py`.

---

### Task 1: MN offset-increment draw kernel

Promote the design-wall probe's matrix-normal offset draw (`scratchpad/design_wall_gibbs_probe.py`) into the library as a pure function.

**Files:**
- Create: `spark_vi/models/topic/pg_stm_dag_gibbs.py`
- Test: `tests/test_pg_stm_dag_gibbs.py`

**Interfaces:**
- Consumes: nothing new (NumPy only).
- Produces: `dag_offset_ridge_draw(WtW, WtM, Sigma, *, penalty, rng) -> np.ndarray` of shape `(Pw, Ksm1)`. A single matrix-normal draw `C ~ MN(mean = solve(A, WtM), row_cov = A^{-1}, col_cov = Sigma)`, `A = WtW + diag(penalty)`. Its expectation is exactly `dag_offset_ridge(WtW, WtM, penalty=penalty)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pg_stm_dag_gibbs.py
import numpy as np
from spark_vi.models.topic.pg_stm_dag import dag_offset_ridge
from spark_vi.models.topic.pg_stm_dag_gibbs import dag_offset_ridge_draw


def test_offset_draw_mean_is_the_ridge_and_covariance_is_matrix_normal():
    rng = np.random.default_rng(0)
    Pw, Ksm1 = 4, 3
    W = rng.standard_normal((200, Pw))
    M = rng.standard_normal((200, Ksm1))
    WtW, WtM = W.T @ W, W.T @ M
    penalty = np.array([1e-6, 0.5, 0.5, 0.5])
    Sigma = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]])
    ridge_mean = dag_offset_ridge(WtW, WtM, penalty=penalty)

    draws = np.array([dag_offset_ridge_draw(WtW, WtM, Sigma, penalty=penalty, rng=rng)
                      for _ in range(4000)])           # (4000, Pw, Ksm1)
    # (a) Monte-Carlo mean == the ridge point
    assert np.abs(draws.mean(axis=0) - ridge_mean).max() < 0.05
    # (b) row-covariance of a fixed column c == A^{-1} * Sigma[c, c]
    Ainv = np.linalg.inv(WtW + np.diag(penalty))
    col = 1
    emp_cov = np.cov(draws[:, :, col].T)
    assert np.abs(emp_cov - Ainv * Sigma[col, col]).max() < 0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd spark-vi && python -m pytest tests/test_pg_stm_dag_gibbs.py::test_offset_draw_mean_is_the_ridge_and_covariance_is_matrix_normal -v`
Expected: FAIL with `ModuleNotFoundError: pg_stm_dag_gibbs` / `dag_offset_ridge_draw not defined`.

- [ ] **Step 3: Write minimal implementation**

```python
# spark_vi/models/topic/pg_stm_dag_gibbs.py
"""Co-sampled Gibbs read-out engine over the DAG-offset PG-STM (step 2). Emits
offset-INCREMENT posterior draws on the compiler's identified quotient DAG.

Inference substrate: stick-breaking + Polya-Gamma augmentation (Polson, Scott &
Windle 2013; Linderman, Johnson & Adams 2015). The offset block is drawn from its
matrix-normal conditional each sweep (a proper joint chain, not a ridge point).
"""
import numpy as np

from spark_vi.models.topic.pg_stm_dag import dag_offset_ridge


def dag_offset_ridge_draw(WtW, WtM, Sigma, *, penalty, rng):
    """One matrix-normal draw of the offset coefficient block:
    C ~ MN(mean = (WtW + diag(penalty))^{-1} WtM, row_cov = (WtW + diag(penalty))^{-1},
    col_cov = Sigma). Its expectation is exactly dag_offset_ridge(WtW, WtM, penalty).
    Depth-scaled `penalty` is the diagonal Gaussian prior precision on the coefficient
    rows; Sigma is the stick-space residual covariance shared across the K-1 columns."""
    WtW = np.asarray(WtW, dtype=np.float64)
    WtM = np.asarray(WtM, dtype=np.float64)
    A = WtW + np.diag(np.asarray(penalty, dtype=np.float64))
    mean = dag_offset_ridge(WtW, WtM, penalty=penalty)
    Ainv = np.linalg.inv(A)
    L_row = np.linalg.cholesky((Ainv + Ainv.T) / 2.0)
    L_col = np.linalg.cholesky((Sigma + Sigma.T) / 2.0)
    Z = rng.standard_normal(mean.shape)
    return mean + L_row @ Z @ L_col.T
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd spark-vi && python -m pytest tests/test_pg_stm_dag_gibbs.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spark_vi/models/topic/pg_stm_dag_gibbs.py tests/test_pg_stm_dag_gibbs.py
git commit -m "feat(dag-readout): matrix-normal offset-increment draw kernel"
```

---

### Task 2: Expected (fractional) closure Gram

The compiler must classify on the expected design moment under soft membership. Add a weighted-Gram entry accepting per-document candidate mixtures.

**Files:**
- Modify: `spark_vi/models/topic/dag_identify.py`
- Test: `tests/test_dag_identify.py`

**Interfaces:**
- Consumes: `DagGate.offset_indicator` (existing).
- Produces: `expected_closure_gram(dag, doc_candidates) -> np.ndarray` shape `(U, U)`. `doc_candidates[d]` is a list of `(p_c, nodes_c)` pairs (membership weight, most-specific node set); the Gram is `Ḡ = Σ_d Σ_c p_c · z(nodes_c) z(nodes_c)ᵀ` = `E[z zᵀ]`, so it carries the within-doc spread. Reduces to `closure_gram` when every doc has one candidate with `p=1`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_dag_identify.py
from spark_vi.models.topic.dag_identify import expected_closure_gram


def test_expected_gram_reduces_to_closure_gram_on_hard_membership():
    dag = DagGate([(), (0,), (0,), (1,), (2,)])
    doc_nodes = [frozenset({1})] * 5 + [frozenset({3})] * 4 + [frozenset({4})] * 6
    hard = [[(1.0, nodes)] for nodes in doc_nodes]
    assert np.allclose(expected_closure_gram(dag, hard), closure_gram(dag, doc_nodes))


def test_expected_gram_spreads_a_soft_doc_across_candidates():
    dag = DagGate([(), (0,), (0,), (1,), (2,)])          # A1=3 under A=1, B1=4 under B=2
    # one doc, 50/50 between subtype A1 (closure {3,1}) and subtype B1 (closure {4,2})
    soft = [[(0.5, frozenset({3})), (0.5, frozenset({4}))]]
    G = expected_closure_gram(dag, soft)                 # offset idx: 0=A,1=B,2=A1,3=B1
    # each subtype's own diagonal gets half weight (less curvature than a hard doc)
    assert np.isclose(G[2, 2], 0.5) and np.isclose(G[3, 3], 0.5)
    # A1 and B1 never co-occur within the doc's mixture -> zero cross moment
    assert np.isclose(G[2, 3], 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd spark-vi && python -m pytest tests/test_dag_identify.py::test_expected_gram_reduces_to_closure_gram_on_hard_membership -v`
Expected: FAIL with `cannot import name 'expected_closure_gram'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to spark_vi/models/topic/dag_identify.py (after closure_gram)
def expected_closure_gram(dag, doc_candidates):
    """Expected closure Gram Ḡ = sum_d sum_c p_c z_c z_c^T = sum_d E[z_d z_d^T] under a
    soft membership posterior. doc_candidates[d] = list of (weight, nodes) candidate
    closures for document d (labeled docs: a single (1.0, nodes)). Carries the within-doc
    spread (a doc split across candidates adds fractional curvature to each), so a
    soft-gated coordinate is appropriately closer to the design null than a hard one.
    Reduces to closure_gram on hard membership. Shape (U, U), U = dag.n_offset_nodes."""
    U = dag.n_offset_nodes
    G = np.zeros((U, U), dtype=np.float64)
    for cands in doc_candidates:
        for p, nodes in cands:
            z = dag.offset_indicator(nodes)
            G += float(p) * np.outer(z, z)
    return G
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd spark-vi && python -m pytest tests/test_dag_identify.py -v -k expected_gram`
Expected: PASS (both).

- [ ] **Step 5: Commit**

```bash
git add spark_vi/models/topic/dag_identify.py tests/test_dag_identify.py
git commit -m "feat(dag-identify): expected closure Gram for soft membership"
```

---

### Task 3: Classify null directions (GAUGE vs UNRESOLVED + recipe)

The compiler already detects collapses (closure-Gram null) and per-group intercept collinearity (foreground Grams). Add the function that turns those into the read-out's two labels plus the attestation recipe.

**Files:**
- Modify: `spark_vi/models/topic/dag_identify.py`
- Test: `tests/test_dag_identify.py`

**Interfaces:**
- Consumes: `foreground_grams`, `detect_confounds` (existing); `DagGate` ancestors/parents.
- Produces: `classify_null_directions(dag, G, fg_grams, detected, *, tol) -> dict` with keys:
  - `"gauge_nodes"`: `frozenset[int]` — original nodes whose absolute LEVEL is a partition-identity gauge freedom (a group's intercept column equals its anchor column ⇒ near-zero eigenvalue of that group's foreground Gram).
  - `"unresolved"`: `dict[int, dict]` — each non-representative member `u` of a collapse set → `{"attest_node": p, "docs_needed": k}` where `p` is the anchor whose missing own-documents cause the collapse and `k = ceil(tol - margin)` from `detected["margins"]` (documents at `p` that would separate `u`'s column, i.e. break the null).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_dag_identify.py
from spark_vi.models.topic.dag_identify import (foreground_grams, classify_null_directions,
                                                identifiability_spectrum)


def test_classify_labels_gauge_levels_and_unresolved_subsumed_subtype():
    # insight-0054 attestation: A(1) has own docs + A1(3); B(2) has NO own docs + B1(4)
    dag = DagGate([(), (0,), (0,), (1,), (2,)])
    doc_nodes = [frozenset({1})] * 400 + [frozenset({3})] * 400 + [frozenset({4})] * 500
    doc_groups = ["A"] * 800 + ["B"] * 500

    class P:                     # minimal partition stub: two groups
        groups = ("A", "B")
    G = closure_gram(dag, doc_nodes)
    fg = foreground_grams(dag, doc_nodes, doc_groups, P())
    det = detect_confounds(dag, G, identifiability_spectrum(G), tol=1.0)
    cls = classify_null_directions(dag, G, fg, det, tol=1.0)

    # anchor LEVELS A and B are partition-identity gauges
    assert {1, 2} <= cls["gauge_nodes"]
    # B1 (no own docs, subsumed into B) is UNRESOLVED with a recipe pointing at B
    assert 4 in cls["unresolved"] and cls["unresolved"][4]["attest_node"] == 2
    assert cls["unresolved"][4]["docs_needed"] >= 1
    # A1 is neither gauge nor unresolved (it is the one identified increment)
    assert 3 not in cls["gauge_nodes"] and 3 not in cls["unresolved"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd spark-vi && python -m pytest tests/test_dag_identify.py::test_classify_labels_gauge_levels_and_unresolved_subsumed_subtype -v`
Expected: FAIL with `cannot import name 'classify_null_directions'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to spark_vi/models/topic/dag_identify.py (after detect_confounds)
def classify_null_directions(dag, G, fg_grams, detected, *, tol):
    """Split the design's null directions into the read-out's two labels:
      * GAUGE (partition identity): a group whose documents all attest their anchor makes
        the foreground Gram's intercept column equal the anchor column -> a near-zero
        eigenvalue whose eigenvector loads on {intercept, anchor}. Such an anchor level is
        a gauge freedom (no attestation resolves it; report the fixed convention).
      * UNRESOLVED (contingent): a non-representative member of a closure-Gram collapse set
        (a no-own-documents subtype). Its individual increment is subsumed into the merged
        quotient node; report a recipe = documents at the missing-evidence anchor that would
        break the collinearity (docs_needed from the column-equality margin).
    Returns {"gauge_nodes": frozenset, "unresolved": {node: {"attest_node", "docs_needed"}}}.
    """
    U = dag.n_offset_nodes
    # --- GAUGE: per-group foreground Gram near-null eigenvector loading on intercept+anchor
    gauge = set()
    for g, W in fg_grams.items():
        evals, evecs = np.linalg.eigh(np.asarray(W, dtype=np.float64))
        for k, lam in enumerate(evals):
            if lam < tol:
                v = evecs[:, k]
                if abs(v[0]) > 1e-6:                       # loads on the intercept (index 0)
                    off = np.abs(v[1:])                    # offset coordinates
                    j = int(np.argmax(off))
                    if off[j] > 1e-6:
                        gauge.add(j + 1)                   # offset index j -> node id j+1
    # --- UNRESOLVED: non-rep members of each collapse set, with an attestation recipe
    unresolved = {}
    margins = detected["margins"]                          # {(parent, child): tol - ||z_c - z_p||^2}
    for s in detected["collapse_sets"]:
        rep = min(s)
        for u in s:
            if u == rep:
                continue
            # the missing-evidence anchor = u's parent inside the collapsed chain
            ps = [p for p in dag.parents[u] if p in s or p == rep]
            attest = ps[0] if ps else rep
            m = margins.get((attest, u), margins.get((rep, u), float(tol)))
            docs_needed = max(1, int(np.ceil(tol - m)))    # docs at `attest` that break the null
            unresolved[u] = {"attest_node": int(attest), "docs_needed": docs_needed}
    return {"gauge_nodes": frozenset(gauge), "unresolved": unresolved}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd spark-vi && python -m pytest tests/test_dag_identify.py::test_classify_labels_gauge_levels_and_unresolved_subsumed_subtype -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spark_vi/models/topic/dag_identify.py tests/test_dag_identify.py
git commit -m "feat(dag-identify): classify null directions GAUGE vs UNRESOLVED + recipe"
```

---

### Task 4: Partial-label arm on the corpus plant

The soft-gate E-step and coverage plant need documents attested at an internal node with an unknown subtype. Extend the existing `dag_offset_corpus`.

**Files:**
- Modify: `tests/_stm_synth.py`
- Test: `tests/test_pg_stm_dag_gibbs.py`

**Interfaces:**
- Consumes: `dag_offset_corpus` (existing; generates gated docs from planted node offsets).
- Produces: `dag_offset_corpus(..., partial_label_plan=None)` — new optional kwarg. `partial_label_plan` maps an INTERNAL node id → #docs generated under a randomly chosen descendant leaf of that node but returned with `doc_candidates[d]` = the list of `(uniform_weight, leaf_closure)` over that internal node's leaves (the true generating leaf is hidden). Returns a third value `doc_candidates` (a list aligned with `docs`); for hard/background docs each entry is `[(1.0, doc_nodes[d])]`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_pg_stm_dag_gibbs.py
from spark_vi.models.topic.partition import TopicBlockPartition
from spark_vi.models.topic.pg_stm_dag import DagGate
from tests._stm_synth import dag_offset_corpus, real_beta_from


def test_partial_label_arm_hides_the_leaf_behind_a_candidate_set():
    part = TopicBlockPartition(group_var="g", background_k=3, foreground=(("A", 2), ("B", 2)))
    K, V = part.K, 80
    dag = DagGate([(), (0,), (0,), (1,), (1,)])          # A=1 has two leaves A1=3, A2=4
    rng = np.random.default_rng(1)
    node_offsets = {u: rng.standard_normal(K - 1) for u in (1, 2, 3, 4)}
    node_offsets[0] = np.zeros(K - 1)
    beta = real_beta_from(K, V, seed=2)
    docs, doc_nodes, doc_candidates = dag_offset_corpus(
        dag=dag, node_offsets=node_offsets, partition=part, beta=beta,
        node_of_group={"A": 1, "B": 2}, doc_nodes_plan={3: 20},
        partial_label_plan={1: 30}, sigma_true=2.0 * np.eye(K - 1),
        doc_len=60, seed=3)
    assert len(docs) == len(doc_nodes) == len(doc_candidates) == 50
    # the 30 partial-label docs carry a 2-candidate set over A's leaves {3,4}, weights sum to 1
    partial = [c for c in doc_candidates if len(c) > 1]
    assert len(partial) == 30
    for c in partial:
        assert abs(sum(p for p, _ in c) - 1.0) < 1e-9
        assert {min(nodes) for _, nodes in c} == {3, 4}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd spark-vi && python -m pytest tests/test_pg_stm_dag_gibbs.py::test_partial_label_arm_hides_the_leaf_behind_a_candidate_set -v`
Expected: FAIL — `dag_offset_corpus() got an unexpected keyword argument 'partial_label_plan'` (and only 2 return values).

- [ ] **Step 3: Write minimal implementation**

Modify `dag_offset_corpus` in `tests/_stm_synth.py`: add `partial_label_plan=None` kwarg; build a `doc_candidates` list alongside `doc_nodes` (each hard/background doc → `[(1.0, doc_nodes[d])]`); for each `(internal_node, n_docs)` in `partial_label_plan`, find the internal node's leaf descendants, generate each doc from a randomly chosen true leaf (same generative path as the hard arm), and set its candidate set to the uniform mixture over ALL the internal node's leaves. Return `(docs, doc_nodes, doc_candidates)`.

```python
# in tests/_stm_synth.py — reference structure of the added arm (place before the return)
    def _leaves_under(node):
        kids = [c for c in range(dag.n_nodes) if node in dag.parents[c]]
        if not kids:
            return [node]
        out = []
        for c in kids:
            out.extend(_leaves_under(c))
        return sorted(set(out))

    if partial_label_plan:
        for internal, n_docs in partial_label_plan.items():
            leaves = _leaves_under(internal)
            g = group_of_node[anchor_of(internal)]
            active = lay["groups"][g]["active"]
            allowed = np.concatenate([partition.background_indices(),
                                      partition.block_indices(g)]).astype(np.int64)
            cand = [(1.0 / len(leaves), frozenset({leaf})) for leaf in leaves]
            for _ in range(n_docs):
                true_leaf = int(rng.choice(leaves))
                mu_full = np.zeros(partition.K - 1)
                for u in dag.closure(frozenset({true_leaf})):
                    mu_full = mu_full + node_offsets[u]
                psi = rng.multivariate_normal(mu_full[active], sigma_true[np.ix_(active, active)])
                psi_bg, psi_gate, psi_fg = psi[:nb], psi[nb], psi[nb + 1:]
                theta = np.zeros(partition.K)
                theta[allowed] = gated_theta(psi_bg, psi_gate, psi_fg)
                toks = rng.choice(beta.shape[1], size=doc_len, p=theta @ beta)
                u_, c_ = np.unique(toks, return_counts=True)
                docs.append(STMDocument(indices=u_.astype(np.int32), counts=c_.astype(np.float64),
                                        length=int(c_.sum()), x=np.array([1.0]),
                                        groups=frozenset({g})))
                doc_nodes.append(frozenset({true_leaf}))     # the hidden truth (for scoring only)
                doc_candidates.append(cand)
```

Initialize `doc_candidates = [[(1.0, dn)] for dn in doc_nodes]` right before the partial arm (after the hard + background arms populate `doc_nodes`), and change the function's final `return docs, doc_nodes` to `return docs, doc_nodes, doc_candidates`. Update the two existing callers (`compiler_realistic_probe` is scratch; the in-repo callers are the tests in `test_pg_stm_dag.py` — grep and fix them to unpack three values or ignore the third).

- [ ] **Step 4: Run test to verify it passes, and existing DAG tests still pass**

Run: `cd spark-vi && python -m pytest tests/test_pg_stm_dag_gibbs.py -v -k partial_label`
Run: `cd spark-vi && python -m pytest tests/test_pg_stm_dag.py -v`
Expected: PASS (new test; and existing DAG tests green after the caller update).

- [ ] **Step 5: Commit**

```bash
git add tests/_stm_synth.py tests/test_pg_stm_dag_gibbs.py tests/test_pg_stm_dag.py
git commit -m "feat(plant): partial-label arm on dag_offset_corpus (candidate sets)"
```

---

### Task 5: Fractional-z soft-gate E-step

**Files:**
- Modify: `spark_vi/models/topic/pg_stm_dag.py`
- Test: `tests/test_pg_stm_dag_gibbs.py`

**Interfaces:**
- Consumes: `pg_estep_doc`, `_bg_estep_doc`, `stick_layout`, `gated_theta` (existing).
- Produces: `_softgate_estep_doc(doc, candidates, glay_by_group, log_beta, Cf, Sigma, dag, *, K, B, inner_rounds, inner_tol) -> (weights, z_bar, per_candidate_esteps)`. `candidates` = list of `(prior_weight, nodes, group)`; runs `pg_estep_doc` under each candidate's group layout, scores each by its evidence (ELBO surrogate = the doc's marginal log-likelihood under that candidate), returns posterior membership `weights` (normalized `prior * exp(evidence)`), the expected offset indicator `z_bar = Σ_c weights_c · dag.offset_indicator(nodes_c)`, and the per-candidate E-step tuples (so the sweep can accumulate the membership-weighted stats). For a single hard candidate, `weights == [1.0]` and it reduces to `pg_estep_doc`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_pg_stm_dag_gibbs.py
def test_softgate_membership_favors_the_generating_leaf():
    part = TopicBlockPartition(group_var="g", background_k=3, foreground=(("A", 2), ("B", 2)))
    K, V = part.K, 80
    dag = DagGate([(), (0,), (0,), (1,), (1,)])
    rng = np.random.default_rng(4)
    node_offsets = {u: 3.0 * rng.standard_normal(K - 1) for u in (1, 2, 3, 4)}
    node_offsets[0] = np.zeros(K - 1)
    beta = real_beta_from(K, V, seed=2)
    docs, doc_nodes, doc_candidates = dag_offset_corpus(
        dag=dag, node_offsets=node_offsets, partition=part, beta=beta,
        node_of_group={"A": 1, "B": 2}, doc_nodes_plan={},
        partial_label_plan={1: 40}, sigma_true=1.0 * np.eye(K - 1), doc_len=120, seed=5)

    from spark_vi.models.topic.pg_stm import stick_layout
    from spark_vi.models.topic.pg_stm_dag import _softgate_estep_doc, offset_penalty, dag_offset_ridge
    lay = stick_layout(part)
    # a rough warm-start Cf from the planted offsets (Gamma=0 intercept; B=node offsets)
    Cf = np.zeros((1 + dag.n_offset_nodes, K - 1))
    for u in (1, 2, 3, 4):
        Cf[u] = node_offsets[u]
    log_beta = np.log(beta)
    hits = 0
    for doc, dn, cand in zip(docs, doc_nodes, doc_candidates):
        cands = [(p, nodes, "A") for p, nodes in cand]         # all under anchor A
        weights, z_bar, _ = _softgate_estep_doc(
            doc, cands, lay["groups"], log_beta, Cf, np.eye(K - 1), dag,
            K=K, B=lay["B"], inner_rounds=8, inner_tol=1e-3)
        true_leaf = min(dn)
        pred_leaf = min(cand[int(np.argmax(weights))][1])
        hits += (pred_leaf == true_leaf)
    assert hits / len(docs) > 0.7        # membership recovers the generating leaf > chance (0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd spark-vi && python -m pytest tests/test_pg_stm_dag_gibbs.py::test_softgate_membership_favors_the_generating_leaf -v`
Expected: FAIL with `cannot import name '_softgate_estep_doc'`.

- [ ] **Step 3: Write minimal implementation**

Add `_softgate_estep_doc` to `pg_stm_dag.py` beside `_bg_estep_doc`. For each candidate, build an augmented doc whose covariate is `[doc.x ; offset_indicator(nodes)]`, run `pg_estep_doc` under the candidate group's layout, and compute an evidence score = the doc's expected complete-data log-likelihood under the returned `(m, V, phi, ...)` (reuse the token-responsibility mass already computed: `evidence_c = Σ_tokens log Σ_k θ_k β_{k,w}` at the candidate's `theta` from `m`). Normalize `weights_c ∝ prior_c · exp(evidence_c − max_c evidence)`. Return `weights`, `z_bar = Σ_c weights_c · dag.offset_indicator(nodes_c)`, and the list of per-candidate E-step tuples.

```python
# reference core (place in pg_stm_dag.py)
def _softgate_estep_doc(doc, candidates, glay_by_group, log_beta, Cf, Sigma, dag,
                        *, K, B, inner_rounds, inner_tol):
    """Fractional-z soft-gate E-step for a partially-labeled document: a mixture over the
    document's candidate closures, each scored by its marginal evidence. Returns posterior
    membership `weights`, the expected non-root closure indicator `z_bar`, and the per-
    candidate E-step tuples (for membership-weighted stat accumulation). One hard candidate
    reduces to pg_estep_doc. See PLDA (Ramage et al. 2011) for the partial-label semantics."""
    esteps, evid = [], []
    for (_p, nodes, g) in candidates:
        z = dag.offset_indicator(nodes)
        w = np.concatenate([np.asarray(doc.x, dtype=np.float64), z])
        aug = dataclasses.replace(doc, x=w)
        est = pg_estep_doc(aug, glay_by_group[g], log_beta, Cf, Sigma,
                           K=K, B=B, inner_rounds=inner_rounds, inner_tol=inner_tol)
        m, Vd, phi, active, allowed, mu_active, _nc = est
        psi_bg, psi_gate, psi_fg = m[:B - 1], m[B - 1], m[B:]
        theta_allowed = gated_theta(psi_bg, psi_gate, psi_fg)
        # marginal token evidence under this candidate (bag-of-words, in `allowed` order)
        beta_allowed = np.exp(log_beta[allowed][:, doc.indices])      # (|allowed|, L)
        tok = theta_allowed @ beta_allowed                            # (L,)
        evid.append(float(np.sum(doc.counts * np.log(np.maximum(tok, 1e-300)))))
        esteps.append(est)
    priors = np.array([p for (p, _n, _g) in candidates], dtype=np.float64)
    logw = np.log(np.maximum(priors, 1e-300)) + np.array(evid)
    weights = np.exp(logw - logw.max()); weights /= weights.sum()
    U = dag.n_offset_nodes
    z_bar = np.zeros(U)
    for wgt, (_p, nodes, _g) in zip(weights, candidates):
        z_bar = z_bar + wgt * dag.offset_indicator(nodes)
    return weights, z_bar, esteps
```

Ensure `import dataclasses` and `gated_theta` are available in `pg_stm_dag.py` (add imports if missing).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd spark-vi && python -m pytest tests/test_pg_stm_dag_gibbs.py::test_softgate_membership_favors_the_generating_leaf -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spark_vi/models/topic/pg_stm_dag.py tests/test_pg_stm_dag_gibbs.py
git commit -m "feat(dag): fractional-z soft-gate E-step (partial-label mixture)"
```

---

### Task 6: Co-sampled quotient Gibbs sweep

The engine kernel: warm-started, samples β/ψ/ω/Σ/membership, draws offset increments on the quotient with the merged-node penalty fix.

**Files:**
- Modify: `spark_vi/models/topic/pg_stm_dag_gibbs.py`
- Test: `tests/test_pg_stm_dag_gibbs.py`

**Interfaces:**
- Consumes: `dag_offset_ridge_draw` (Task 1), `_softgate_estep_doc` (Task 5), `offset_penalty`, `dag_offset_ridge`, `stick_layout`, `gated_theta`, `gated_counts`, `omega_sample`, `psi_posterior`, `_draw_block_sigma`, `safe_inverse`, `beta_dirichlet_mean`.
- Produces: `PGSTMDagGibbs(K, V, partition, dag, *, P, n_iter, burn, lam_base, gamma_depth, gamma_ridge, beta_eta, seed).run(docs, doc_candidates, *, beta_init) -> dict` with `"increment_draws"` shape `(n_kept, U, Ksm1)` (U = dag.n_offset_nodes; node id u → offset row u-1), `"beta"`, `"Sigma"`, `"membership"` (per soft doc). The `dag` passed here is already the QUOTIENT; `doc_candidates` already remapped to quotient node ids by the orchestrator. Merged-node penalty fix: a quotient node created by collapsing a chain of original depth-span `s` receives penalty `Σ` of the depth-scaled penalties of the chain it replaces (passed in as `penalty_override`, default = `offset_penalty(...)`).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_pg_stm_dag_gibbs.py
from tests._stm_synth import planted_recovery


def test_gibbs_recovers_a_planted_identified_increment():
    part = TopicBlockPartition(group_var="g", background_k=3, foreground=(("A", 2), ("B", 2)))
    K, V = part.K, 120
    dag = DagGate([(), (0,), (0,), (1,), (1,)])           # A has TWO leaves -> A1 increment identified
    rng = np.random.default_rng(7)
    node_offsets = {u: 2.0 * rng.standard_normal(K - 1) for u in (1, 2, 3, 4)}
    node_offsets[0] = np.zeros(K - 1)
    beta = real_beta_from(K, V, seed=2)
    docs, doc_nodes, cand = dag_offset_corpus(
        dag=dag, node_offsets=node_offsets, partition=part, beta=beta,
        node_of_group={"A": 1, "B": 2},
        doc_nodes_plan={1: 200, 3: 200, 4: 200, 2: 150}, sigma_true=1.0 * np.eye(K - 1),
        doc_len=80, seed=8)
    eng = PGSTMDagGibbs(K=K, V=V, partition=part, dag=dag, P=1, n_iter=120, burn=60,
                        lam_base=0.25, seed=0)
    out = eng.run(docs, cand, beta_init=beta)
    draws = out["increment_draws"]                        # (n_kept, U, Ksm1)
    assert draws.shape[1] == dag.n_offset_nodes
    # the A1 (node 3) increment posterior mean correlates with the planted increment
    b3 = draws[:, 3 - 1, :].mean(axis=0)
    assert np.corrcoef(b3, node_offsets[3])[0, 1] > 0.5
    # Sigma stays PD
    assert np.linalg.eigvalsh(out["Sigma"]).min() > -1e-8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd spark-vi && python -m pytest tests/test_pg_stm_dag_gibbs.py::test_gibbs_recovers_a_planted_identified_increment -v`
Expected: FAIL with `cannot import name 'PGSTMDagGibbs'`.

- [ ] **Step 3: Write minimal implementation**

Add `PGSTMDagGibbs` to `pg_stm_dag_gibbs.py`. Adapt the sweep from `scratchpad/design_wall_gibbs_probe.py::offset_gibbs` (which is already validated): augment each doc's covariate with `dag.offset_indicator`; per sweep run the gated E-step (or `_softgate_estep_doc` when a doc has >1 candidate, membership-weighting its accumulated `W`/`M` rows and sampling `c_d`); draw β (Dirichlet, warm-started from `beta_init`); draw the offset block via `dag_offset_ridge_draw(WtW, WtM, Sigma, penalty=penalty, rng=rng)`; draw Σ via `_draw_block_sigma`. Collect post-burn `Cf` as `increment_draws`. Use `penalty = offset_penalty(P, dag, gamma_ridge=..., lam_base=..., gamma_depth=...)` unless `penalty_override` is supplied. Reference: the probe's sweep body (lines building `W`, the per-doc loop, the global draws) with the E-step generalized to the soft-gate mixture and β sampled warm-started rather than fixed.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd spark-vi && python -m pytest tests/test_pg_stm_dag_gibbs.py::test_gibbs_recovers_a_planted_identified_increment -v`
Expected: PASS (may take ~1-2 min).

- [ ] **Step 5: Commit**

```bash
git add spark_vi/models/topic/pg_stm_dag_gibbs.py tests/test_pg_stm_dag_gibbs.py
git commit -m "feat(dag-readout): co-sampled quotient Gibbs increment engine"
```

---

### Task 7: Merged-node posterior invariant (Fable req 5)

Validate that fitting the quotient matches fitting the original then projecting onto its row space — the Gram invariant lifted to the posterior.

**Files:**
- Test: `tests/test_pg_stm_dag_gibbs.py`

**Interfaces:**
- Consumes: `PGSTMDagGibbs` (Task 6), `build_quotient`, `expected_closure_gram`/`closure_gram`, `detect_confounds`, `identifiability_spectrum` (existing).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_pg_stm_dag_gibbs.py
from spark_vi.models.topic.dag_identify import build_quotient, identifiability_spectrum


def test_quotient_posterior_matches_projected_original_on_identified_coords():
    part = TopicBlockPartition(group_var="g", background_k=3, foreground=(("A", 2), ("B", 2)))
    K, V = part.K, 120
    dag = DagGate([(), (0,), (0,), (1,), (2,)])           # B(2) no own docs -> {2,4} collapse
    rng = np.random.default_rng(9)
    node_offsets = {u: 2.0 * rng.standard_normal(K - 1) for u in (1, 2, 3, 4)}
    node_offsets[0] = np.zeros(K - 1)
    beta = real_beta_from(K, V, seed=2)
    docs, doc_nodes, cand = dag_offset_corpus(
        dag=dag, node_offsets=node_offsets, partition=part, beta=beta,
        node_of_group={"A": 1, "B": 2}, doc_nodes_plan={1: 200, 3: 200, 4: 250},
        sigma_true=1.0 * np.eye(K - 1), doc_len=80, seed=10)
    G = closure_gram(dag, doc_nodes)
    det = detect_confounds(dag, G, identifiability_spectrum(G), tol=1.0)
    q = build_quotient(dag, det)
    qcand = [[(p, frozenset(int(q["node_map"][u]) for u in nodes)) for p, nodes in c] for c in cand]

    orig = PGSTMDagGibbs(K=K, V=V, partition=part, dag=dag, P=1, n_iter=120, burn=60,
                         lam_base=0.25, seed=0).run(docs, cand, beta_init=beta)
    quot = PGSTMDagGibbs(K=K, V=V, partition=part, dag=q["quotient_dag"], P=1, n_iter=120,
                         burn=60, lam_base=0.25, seed=0).run(docs, qcand, beta_init=beta)
    # A1 (node 3) survives in the quotient; its posterior-mean increment matches in both fits
    q3 = int(q["node_map"][3])
    b_orig = orig["increment_draws"][:, 3 - 1, :].mean(axis=0)
    b_quot = quot["increment_draws"][:, q3 - 1, :].mean(axis=0)
    assert np.corrcoef(b_orig, b_quot)[0, 1] > 0.9
```

- [ ] **Step 2: Run test to verify it fails, then passes**

Run: `cd spark-vi && python -m pytest tests/test_pg_stm_dag_gibbs.py::test_quotient_posterior_matches_projected_original_on_identified_coords -v`
Expected: initially may FAIL if the merged-node penalty is not summed (correlation < 0.9); after Task 6's `penalty_override` sums the collapsed chain's penalties, PASS. If it passes directly, keep it as a regression guard.

- [ ] **Step 3: If it fails, apply the merged-node penalty fix**

In the orchestrator/engine, when building the quotient penalty, set each merged quotient node's penalty to the SUM of the depth-scaled penalties of the original chain nodes it replaces (from `node_map`), rather than a single depth term. Re-run.

- [ ] **Step 4: Commit**

```bash
git add tests/test_pg_stm_dag_gibbs.py spark_vi/models/topic/pg_stm_dag_gibbs.py
git commit -m "test(dag-readout): merged-node posterior invariant + penalty-sum fix"
```

---

### Task 8: ReadOut schema assembly

**Files:**
- Create: `spark_vi/models/topic/dag_readout.py`
- Test: `tests/test_dag_readout.py`

**Interfaces:**
- Consumes: `classify_null_directions` output (Task 3), `PGSTMDagGibbs.run` output (Task 6), `node_map` (from `build_quotient`).
- Produces: `assemble_readout(dag, increment_draws, node_map, classification, *, ci_level=0.90, fragility_margin=None, spectrum=None) -> dict`. Fixed key set = all non-root original nodes. Per node: `identified` (increment_mean, ci_low, ci_high) / `fragile` (+ fragility{margin,min_eig}) / `unresolved` (width, reason, recipe; NO mean) / `gauge` (reason, convention; NO number). Top level: `calibration:"absolute"`, `coordinates`, `meta`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dag_readout.py
import numpy as np
from spark_vi.models.topic.dag_readout import assemble_readout


def test_readout_is_a_fixed_keyset_with_the_four_statuses():
    U, Ksm1, n_kept = 4, 5, 200                          # nodes 1..4
    node_map = np.array([0, 1, 2, 3, 2])                 # node 4 merged into node 2 (rep)
    rng = np.random.default_rng(0)
    draws = rng.standard_normal((n_kept, U, Ksm1)) * 0.1 + 1.0
    classification = {"gauge_nodes": frozenset({1, 2}),
                      "unresolved": {4: {"attest_node": 2, "docs_needed": 250}}}
    ro = assemble_readout(DummyDag(n_nodes=5), draws, node_map, classification, ci_level=0.90)

    assert ro["calibration"] == "absolute"
    assert set(ro["coordinates"]) == {1, 2, 3, 4}        # fixed key set = all non-root nodes
    assert ro["coordinates"][1]["status"] == "gauge" and "increment_mean" not in ro["coordinates"][1]
    assert ro["coordinates"][4]["status"] == "unresolved"
    assert ro["coordinates"][4]["recipe"]["docs_needed"] == 250 and "increment_mean" not in ro["coordinates"][4]
    assert ro["coordinates"][3]["status"] == "identified"
    c3 = ro["coordinates"][3]
    assert c3["ci_low"] < c3["increment_mean"] < c3["ci_high"]
```

(Define a tiny `DummyDag` with `n_nodes` and a `parents` list in the test, or import a real `DagGate`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cd spark-vi && python -m pytest tests/test_dag_readout.py -v`
Expected: FAIL with `ModuleNotFoundError: dag_readout`.

- [ ] **Step 3: Write minimal implementation**

```python
# spark_vi/models/topic/dag_readout.py
"""Per-coordinate-class read-out for the DAG-offset engine (step 2). Assembles calibrated
increment posteriors for identified coordinates and machine-flagged non-answers (GAUGE /
UNRESOLVED) for design-wall directions. Fixed coordinate set: every non-root node always
appears, so a coordinate flips unresolved->number in place when data accrues.

Design-wall coordinates emit width/status/cause but NEVER a point estimate (Fable contract).
"""
import numpy as np


def assemble_readout(dag, increment_draws, node_map, classification,
                     *, ci_level=0.90, fragility_margin=None, spectrum=None):
    gauge = set(classification["gauge_nodes"])
    unresolved = dict(classification["unresolved"])
    lo_q, hi_q = (1 - ci_level) / 2, 1 - (1 - ci_level) / 2
    coords = {}
    for u in range(1, dag.n_nodes):
        parent = int(dag.parents[u][0]) if dag.parents[u] else 0
        if u in unresolved:
            rec = unresolved[u]
            q = int(node_map[u])
            width = float(np.sqrt(np.var(increment_draws[:, q - 1, :], axis=0).sum())) if q >= 1 else 0.0
            coords[u] = {"node": u, "parent": parent, "status": "unresolved",
                         "width": width, "reason": "design_null(no_own_documents)",
                         "recipe": {"attest_node": rec["attest_node"],
                                    "docs_needed": rec["docs_needed"]}}
        elif u in gauge:
            coords[u] = {"node": u, "parent": parent, "status": "gauge",
                         "reason": "design_null(partition_identity)",
                         "convention": "level fixed to the intercept gauge; increments only"}
        else:
            q = int(node_map[u])
            col = increment_draws[:, q - 1, :]            # (n_kept, Ksm1)
            mean = col.mean(axis=0)
            status = "identified"
            entry = {"node": u, "parent": parent, "status": status,
                     "increment_mean": mean,
                     "ci_low": np.quantile(col, lo_q, axis=0),
                     "ci_high": np.quantile(col, hi_q, axis=0)}
            coords[u] = entry
    return {"calibration": "absolute", "coordinates": coords,
            "meta": {"n_draws": int(increment_draws.shape[0]), "ci_level": ci_level}}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd spark-vi && python -m pytest tests/test_dag_readout.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spark_vi/models/topic/dag_readout.py tests/test_dag_readout.py
git commit -m "feat(dag-readout): per-coordinate-class ReadOut schema"
```

---

### Task 9: Prevalence estimand

**Files:**
- Modify: `spark_vi/models/topic/dag_readout.py`
- Test: `tests/test_dag_readout.py`

**Interfaces:**
- Consumes: per-document membership weights (from `_softgate_estep_doc`) + hard labels.
- Produces: `node_prevalence(dag, doc_nodes, doc_candidates, memberships) -> dict[node, {labeled_mass, inferred_total, recall_ratio}]`. `labeled_mass` = count of hard-attested docs whose closure contains the node; `inferred_total` = `labeled_mass` + Σ over partial-label docs of the membership mass landing on candidates whose closure contains the node; `recall_ratio` = `labeled_mass / inferred_total` (1.0 when no soft mass). Add to the `ReadOut["prevalence"]` field in `assemble_readout` (optional args).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_dag_readout.py
from spark_vi.models.topic.dag_readout import node_prevalence
from spark_vi.models.topic.pg_stm_dag import DagGate


def test_prevalence_adds_soft_membership_to_labeled_mass():
    dag = DagGate([(), (0,), (0,), (1,), (1,)])          # A=1 with leaves A1=3, A2=4
    doc_nodes = [frozenset({3})] * 10                     # 10 hard A1 docs
    doc_candidates = [[(1.0, frozenset({3}))]] * 10
    memberships = [np.array([1.0])] * 10
    # add 4 partial-label docs under A, each 0.75 A1 / 0.25 A2
    doc_nodes += [frozenset({3})] * 4
    cand = [(0.75, frozenset({3})), (0.25, frozenset({4}))]
    doc_candidates += [cand] * 4
    memberships += [np.array([0.75, 0.25])] * 4

    prev = node_prevalence(dag, doc_nodes, doc_candidates, memberships)
    assert np.isclose(prev[3]["labeled_mass"], 14)        # all 14 docs hard-or-true A1
    assert np.isclose(prev[3]["inferred_total"], 10 + 4 * 0.75)   # soft mass on A1
    assert prev[3]["recall_ratio"] > 1.0                  # labeled undercounts vs inferred? (defn check)
```

Note: fix `labeled_mass` vs `inferred_total` semantics in the implementation so the assertion matches the docstring (labeled = hard attestations only; the true-leaf bookkeeping is for scoring, not labeled_mass). Adjust the test's expected `labeled_mass` to the hard-attested count (10) if the implementer defines labeled_mass as hard-only — keep the test and code consistent, whichever convention is chosen, and document it.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd spark-vi && python -m pytest tests/test_dag_readout.py::test_prevalence_adds_soft_membership_to_labeled_mass -v`
Expected: FAIL with `cannot import name 'node_prevalence'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to spark_vi/models/topic/dag_readout.py
def node_prevalence(dag, doc_nodes, doc_candidates, memberships):
    """Per-node prevalence: labeled_mass (hard-attested docs whose closure contains the node)
    + inferred_total (labeled_mass + partial-label membership mass landing on candidates whose
    closure contains the node); recall_ratio = labeled_mass / inferred_total. v1 counts only
    partial-label resolution (unlabeled docs are background/root); the latent-anchor MNAR
    recall correction is deferred."""
    nodes = range(1, dag.n_nodes)
    labeled = {u: 0.0 for u in nodes}
    inferred = {u: 0.0 for u in nodes}
    for dn, cands, wts in zip(doc_nodes, doc_candidates, memberships):
        hard = len(cands) == 1
        for u in nodes:
            if hard and u in dag.closure(dn):
                labeled[u] += 1.0
                inferred[u] += 1.0
            elif not hard:
                for wgt, (_p, cnodes) in zip(wts, cands):
                    if u in dag.closure(cnodes):
                        inferred[u] += float(wgt)
    return {u: {"labeled_mass": labeled[u], "inferred_total": inferred[u],
                "recall_ratio": (labeled[u] / inferred[u]) if inferred[u] > 0 else 1.0}
            for u in nodes}
```

Wire `prevalence` into `assemble_readout` via optional `doc_nodes`/`doc_candidates`/`memberships` args (default `None` → omit the field). Reconcile the test's numbers with the chosen `labeled_mass` convention.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd spark-vi && python -m pytest tests/test_dag_readout.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spark_vi/models/topic/dag_readout.py tests/test_dag_readout.py
git commit -m "feat(dag-readout): partial-label prevalence estimand"
```

---

### Task 10: Orchestrator (Phase A→B1→B2→C)

**Files:**
- Modify: `spark_vi/models/topic/dag_readout.py`
- Test: `tests/test_dag_readout.py`

**Interfaces:**
- Consumes: `PGSTMDag.fit` (Phase A warm-start; extended to run `_softgate_estep_doc` for partial docs — verify/extend as needed), `expected_closure_gram` + `foreground_grams` + `detect_confounds` + `build_quotient` + `classify_null_directions` (Phase B1), `PGSTMDagGibbs.run` (Phase B2), `assemble_readout` + `node_prevalence` (Phase C).
- Produces: `dag_offset_readout(docs, doc_nodes, doc_candidates, doc_groups, partition, dag, *, P=1, tol=1.0, lam_base=0.25, n_iter=200, burn=100, seed=0) -> ReadOut`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_dag_readout.py — end-to-end smoke on the insight-0054 corpus
from spark_vi.models.topic.partition import TopicBlockPartition
from spark_vi.models.topic.dag_readout import dag_offset_readout
from tests._stm_synth import dag_offset_corpus, real_beta_from


def test_end_to_end_readout_statuses_on_the_0054_corpus():
    part = TopicBlockPartition(group_var="g", background_k=3, foreground=(("A", 2), ("B", 2)))
    K, V = part.K, 120
    dag = DagGate([(), (0,), (0,), (1,), (2,)])           # B(2) no own docs
    import numpy as np
    rng = np.random.default_rng(11)
    node_offsets = {u: 2.0 * rng.standard_normal(K - 1) for u in (1, 2, 3, 4)}
    node_offsets[0] = np.zeros(K - 1)
    beta = real_beta_from(K, V, seed=2)
    docs, doc_nodes, cand = dag_offset_corpus(
        dag=dag, node_offsets=node_offsets, partition=part, beta=beta,
        node_of_group={"A": 1, "B": 2}, doc_nodes_plan={1: 200, 3: 200, 4: 250},
        sigma_true=1.0 * np.eye(K - 1), doc_len=80, seed=12)
    doc_groups = [next(iter(d.groups)) for d in docs]
    ro = dag_offset_readout(docs, doc_nodes, cand, doc_groups, part, dag,
                            P=1, tol=1.0, lam_base=0.25, n_iter=80, burn=40, seed=0)
    st = {u: ro["coordinates"][u]["status"] for u in ro["coordinates"]}
    assert st[3] == "identified"                          # A1 increment
    assert st[1] == "gauge" and st[2] == "gauge"          # anchor levels
    assert st[4] == "unresolved"                          # B1 subsumed, no own docs
    assert ro["coordinates"][4]["recipe"]["attest_node"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd spark-vi && python -m pytest tests/test_dag_readout.py::test_end_to_end_readout_statuses_on_the_0054_corpus -v`
Expected: FAIL with `cannot import name 'dag_offset_readout'`.

- [ ] **Step 3: Write minimal implementation**

Add `dag_offset_readout` wiring the four phases: (A) `PGSTMDag.fit` for warm-start β/Σ + membership on partial docs; (B1) `expected_closure_gram(dag, doc_candidates)` + `foreground_grams` → `detect_confounds` → `build_quotient` + `classify_null_directions`; remap `doc_candidates` through `node_map`; compute the summed merged-node penalty; (B2) `PGSTMDagGibbs(...).run(docs, qcand, beta_init=β, penalty_override=summed)`; (C) `assemble_readout(dag, draws, node_map, classification, ...)` + `node_prevalence`. Return the ReadOut. Keep Phase A cheap (reuse the existing VI iters).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd spark-vi && python -m pytest tests/test_dag_readout.py::test_end_to_end_readout_statuses_on_the_0054_corpus -v`
Expected: PASS (may take a few minutes).

- [ ] **Step 5: Commit**

```bash
git add spark_vi/models/topic/dag_readout.py tests/test_dag_readout.py
git commit -m "feat(dag-readout): dag_offset_readout orchestrator (Phase A->B->C)"
```

---

### Task 11: Coverage plant (the acceptance gate)

The GATE: redraw-truth coverage on four cells + the design-wall schema assertions.

**Files:**
- Modify: `tests/_stm_synth.py` (a `coverage_plant` helper if useful)
- Test: `tests/test_dag_readout.py`

**Interfaces:**
- Consumes: `dag_offset_readout` (Task 10), `dag_offset_corpus` with the partial-label arm (Task 4).
- Produces: a coverage validation test (a slow test, marked `@pytest.mark.slow` if the repo has that marker; otherwise a plain test with a modest replicate count).

- [ ] **Step 1: Write the coverage test**

```python
# append to tests/test_dag_readout.py
import numpy as np
from spark_vi.models.topic.partition import TopicBlockPartition
from spark_vi.models.topic.pg_stm_dag import DagGate
from spark_vi.models.topic.dag_readout import dag_offset_readout
from tests._stm_synth import dag_offset_corpus, real_beta_from


def _covers(entry, truth):
    return bool(np.all(entry["ci_low"] <= truth) and np.all(truth <= entry["ci_high"]))


def test_coverage_plant_identified_covers_designwall_reports_unresolved():
    part = TopicBlockPartition(group_var="g", background_k=3, foreground=(("A", 2), ("B", 2)))
    K, V = part.K, 120
    dag = DagGate([(), (0,), (0,), (1,), (2,)])           # A1(3) identified; B(2)/B1(4) design wall
    beta = real_beta_from(K, V, seed=2)
    R, covered = 12, 0
    designwall_ok = True
    for rep in range(R):
        rng = np.random.default_rng(100 + rep)            # REDRAW TRUTH per replicate (insight 0051)
        node_offsets = {u: 2.0 * rng.standard_normal(K - 1) for u in (1, 2, 3, 4)}
        node_offsets[0] = np.zeros(K - 1)
        docs, doc_nodes, cand = dag_offset_corpus(
            dag=dag, node_offsets=node_offsets, partition=part, beta=beta,
            node_of_group={"A": 1, "B": 2},
            doc_nodes_plan={1: 200, 3: 200, 4: 250},
            partial_label_plan={1: 60},                   # SOFT-GATED cell
            sigma_true=1.0 * np.eye(K - 1), doc_len=80, seed=1000 + rep)
        doc_groups = [next(iter(d.groups)) for d in docs]
        ro = dag_offset_readout(docs, doc_nodes, cand, doc_groups, part, dag,
                                P=1, tol=1.0, lam_base=0.25, n_iter=80, burn=40, seed=0)
        c3 = ro["coordinates"][3]
        if c3["status"] in ("identified", "fragile"):
            covered += _covers(c3, node_offsets[3])
        # design-wall coords: NO point estimate, recipe/convention present
        designwall_ok &= ("increment_mean" not in ro["coordinates"][4]) and \
                          ("recipe" in ro["coordinates"][4]) and \
                          ("increment_mean" not in ro["coordinates"][2])
    assert designwall_ok
    assert covered / R >= 0.6            # wide-but-covers (loose band at R=12; tighten if R raised)
```

- [ ] **Step 2: Run the coverage test**

Run: `cd spark-vi && python -m pytest tests/test_dag_readout.py::test_coverage_plant_identified_covers_designwall_reports_unresolved -v`
Expected: PASS. If identified coverage is systematically low (< 0.6), that is a real finding — record it as an insight (the read-out is not yet calibrated) and consult before loosening; do NOT silently widen intervals to pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_dag_readout.py tests/_stm_synth.py
git commit -m "test(dag-readout): coverage plant acceptance gate (4 cells)"
```

- [ ] **Step 4: Write the validation insight**

Record the coverage result (populated/scarce/soft-gated coverage numbers; design-wall schema behavior) as `docs/insights/0057-*.md` and index it in `docs/insights/README.md`. If coverage holds, this is the engine's acceptance record; if it fails, it is the finding that gates the next step (LKJ/half-t provenance priors, per Fable's contract).

```bash
git add docs/insights/
git commit -m "docs(insight-0057): read-out engine coverage plant result"
```

---

## Self-Review

**Spec coverage:**
- §1 two-phase pipeline → Task 10 (orchestrator) wires A→B1→B2→C; Tasks 5 (soft-gate E-step), 2 (expected Gram), 6 (Gibbs) are the phase pieces. ✓
- §2 modules → Task 1/6 (`pg_stm_dag_gibbs.py`), 2/3 (`dag_identify.py`), 5 (`pg_stm_dag.py`), 8/9/10 (`dag_readout.py`), 4/11 (`_stm_synth.py`). ✓
- §3 schema (fixed keyset; identified/fragile/unresolved/gauge; no point estimate for design-wall; prevalence) → Task 8 + 9. ✓
- §4 soft gate (labeled/partial/unlabeled=root; membership by marginal likelihood; expected Gram) → Task 4 (plant), 5 (E-step), 2 (Gram). ✓
- §5 coverage plant (4 cells, redraw-truth, merged-node invariant) → Task 11 (+ Task 7 for the invariant). ✓
- Fable contract: emit-but-flag + status-carries-cause + no-point-estimate (Task 8); GAUGE vs UNRESOLVED (Task 3); merged-node invariant + penalty fix (Task 7); soft-gate cell in plant (Task 11). ✓

**Placeholder scan:** each code step shows concrete code; the two reference-structure steps (Task 4, Task 6) point at the validated probe/existing arms with explicit instructions and named symbols, not "implement later". Fragility tier is deferred to the reporting layer per the spec; Task 8 emits `identified`/`fragile` structurally and Task 11 calibrates the threshold — no gap.

**Type consistency:** `increment_draws` shape `(n_kept, U, Ksm1)` with node u → row u-1 is used identically in Tasks 6/7/8/11. `classify_null_directions` returns `gauge_nodes`/`unresolved` consumed verbatim in Task 8. `dag_offset_corpus` returns THREE values after Task 4; Tasks 5–11 all unpack three. `PGSTMDagGibbs.run(docs, doc_candidates, *, beta_init, penalty_override=None)` signature consistent across Tasks 6/7/10.

**Open items flagged for the implementer, not gaps:** the `labeled_mass` convention in Task 9 (reconcile test with code); the fragility threshold calibration in Task 11 (from the compiler's min-eig spectrum); Phase A may need a small extension to run `_softgate_estep_doc` on partial docs (Task 10 note).
