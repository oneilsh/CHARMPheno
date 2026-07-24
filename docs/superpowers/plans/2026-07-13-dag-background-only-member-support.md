# DAG Background-Only Member Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a document belong to **no foreground group** (`groups == frozenset()`) — a background-only member — so a DAG-offset corpus need not strictly partition; this breaks the anchor-partition dummy trap (`Σ anchor-columns == intercept`), identifies anchor levels, and makes leaf path-sums recover (including a no-direct-docs anchor).

**Architecture:** A background-only document uses only the background block: a **flat (non-gated) mean-field PG E-step** (`_bg_estep_doc` in `pg_stm_dag.py`) that reuses `pg_stm`'s flat primitives unchanged. `PGSTMDag.fit` routes group-less documents to it, gated documents to the existing `pg_estep_doc`. One minimal, backward-compatible guard in `pg_stm.py`'s `pg_accumulate_doc` lets a group-less document skip the per-group count. The plant gains a background-only arm; a realistic test shows anchor levels become identified and path-sums recover.

**Tech Stack:** Python / NumPy / SciPy, single-machine, mirroring `pg_stm.py`.

## Global Constraints

- This is the approved **gate-change** step. It follows the increment-reparam step (its Tasks 1–4), which are committed and green at `cfe8f81`. Branch `pg-stm`.
- Domain-agnostic engine layer: integer node/token ids only; no concept names/ids or clinical vocabulary in code, comments, or docstrings.
- Test-honesty rule: every test docstring states what is *planted* vs *real*, where it sits on the synthetic→real spectrum, and the claim it supports **and** the claim it does not. No test asserts a *transfer* claim from a *synthetic* result — synthetic proves math-correctness only.
- Reuse `pg_stm.py` **flat** primitives unchanged (`stick_to_simplex`, `expected_log_theta`, `token_responsibilities`, `stick_trials`, `omega_expectation`, `psi_posterior`, `_PSI_CLIP`, and `safe_inverse` from `_linalg`). The ONLY edit to `pg_stm.py` in this plan is the minimal group-count guard in Task 1 — nothing else in `pg_stm.py` changes.
- Cite any literature-sourced method/default in its docstring; an uncited constant is labeled a heuristic. No LaTeX (plain text + Unicode).
- Test cmd from `spark-vi/`: `cd /Users/oneilsh/Documents/projects/tislab/CHARM/CHARMPheno/spark-vi && python -m pytest tests/test_pg_stm_dag.py -q` (slow, several minutes — be patient). Commit from repo root. Do not `git add` untracked scratch (`dashboard/public/data/...`, `node_modules/`, `spark-vi/tests/test_t3b_diag_tmp.py`).
- End commit messages with: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

### Task 1: `pg_accumulate_doc` tolerates a group-less document

`pg_accumulate_doc` currently does `(g,) = tuple(doc.groups)` and crashes on an empty group set. Guard it so a group-less (background-only) document accumulates its word/covariate/scatter stats but increments no per-group count. Backward-compatible: single-group documents behave identically.

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/pg_stm.py` (`pg_accumulate_doc`, lines 149-150)
- Test: `spark-vi/tests/test_pg_stm_sufficient_stats.py` (the module that already tests `pg_accumulate_doc`/`pg_empty_stats`). Append the test there; match that file's existing import style.

**Interfaces:**
- Produces: `pg_accumulate_doc(stats, doc, estep_out, *, K)` unchanged signature; now accepts `doc.groups == frozenset()` (skips `group_counts`).

- [ ] **Step 1: Write the failing test** (append to the pg_stm test file; adjust the import to match that file's existing imports)

```python
def test_pg_accumulate_doc_tolerates_group_less_document():
    """Deterministic structure check; no empirical or transfer claim. A background-only
    document (no foreground group) accumulates its word/covariate/scatter stats and
    increments D, but touches no per-group count."""
    import numpy as np
    from spark_vi.models.topic.pg_stm import pg_empty_stats, pg_accumulate_doc
    from spark_vi.models.topic.types import STMDocument
    K, V, P = 4, 5, 1
    stats = pg_empty_stats(K, V, P, groups=("A",))
    doc = STMDocument(indices=np.array([0, 1], dtype=np.int32),
                      counts=np.array([2.0, 1.0]), length=3,
                      x=np.array([1.0]), groups=frozenset())
    active = np.array([0, 1], dtype=np.int64)          # 2 background sticks
    m = np.array([0.1, -0.2]); Vd = np.eye(2)
    phi = np.array([[0.6, 0.4, 0.0, 0.0], [0.3, 0.7, 0.0, 0.0]])   # (n_tok, K)
    allowed = np.array([0, 1, 2], dtype=np.int64); mu_active = np.zeros(2)
    pg_accumulate_doc(stats, doc, (m, Vd, phi, active, allowed, mu_active), K=K)
    assert stats["D"] == 1
    assert stats["group_counts"] == {"A": 0}           # no group incremented
    assert np.allclose(stats["XtX"], np.array([[1.0]]))  # x x^T for x=[1]
```

- [ ] **Step 2: Run test, verify fail** — `pytest tests/test_pg_stm.py::test_pg_accumulate_doc_tolerates_group_less_document -q` → FAIL (`ValueError: not enough values to unpack`).

- [ ] **Step 3: Implement** — in `pg_accumulate_doc`, replace lines 149-150:

```python
    gs = tuple(doc.groups)
    if gs:                                  # background-only docs (no group) skip the per-group count
        (g,) = gs
        stats["group_counts"][g] += 1
```

- [ ] **Step 4: Run test, verify pass** — PASS. Then run the existing pg_stm suite (`pytest tests/test_pg_stm.py -q`) to confirm no regression to the single-group path.

- [ ] **Step 5: Commit** — `git commit -am "feat(pg-stm): pg_accumulate_doc tolerates a group-less (background-only) document"`

---

### Task 2: `_bg_estep_doc` — flat background-only E-step + `fit` routing

Add the non-gated E-step and route group-less documents to it in `PGSTMDag.fit`.

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/pg_stm_dag.py` (imports at top; add `_bg_estep_doc` near `PGSTMDag`; edit the per-doc loop in `fit`)
- Test: `spark-vi/tests/test_pg_stm_dag.py`

**Interfaces:**
- Consumes: `pg_stm` flat primitives (`expected_log_theta`, `token_responsibilities`, `stick_trials`, `omega_expectation`, `psi_posterior`, `_PSI_CLIP`), `safe_inverse` from `_linalg`; `stick_layout`, `pg_accumulate_doc` (group-less-tolerant, Task 1).
- Produces: `_bg_estep_doc(doc, bg_lay, log_beta, Gamma, Sigma, *, K, inner_rounds, inner_tol) -> (m, V, phi, active, allowed, mu_active, n_clips)` — same 7-tuple as `pg_estep_doc`; `bg_lay` is `{"active": bg_sticks, "allowed": background_indices}`. `PGSTMDag.fit` routes `doc.groups == frozenset()` documents through it.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_pg_stm_dag.py
def test_fit_routes_background_only_docs_and_they_inform_the_design():
    """PLANTED: a small gated corpus plus one background-only document (no group).
    REAL: nothing. Synthetic -> MATH-CORRECTNESS: PGSTMDag.fit runs with background-only
    documents mixed in (routing them to the flat E-step), returns valid shapes, and the
    background-only doc contributes to the covariate design (its all-ones intercept enters
    XtX via the fit). Proves the routing + flat E-step compose with the gated path. Does
    NOT prove recovery or transfer."""
    import numpy as np
    from spark_vi.models.topic.partition import TopicBlockPartition
    part = TopicBlockPartition(group_var="g", background_k=4, foreground=(("A", 3), ("B", 3)))
    from spark_vi.models.topic.pg_stm_dag import DagGate, PGSTMDag
    from tests._stm_synth import gated_ln_corpus_stick
    docs, _p, _S, _b = gated_ln_corpus_stick(
        group_weights={"A": 0.5, "B": 0.5}, fg_per_group=3, bg_k=4, V=40, D=120,
        doc_len=30, seed=0)
    # append 40 background-only docs: no group, a couple of background-topic tokens each
    import dataclasses
    bg_docs = [dataclasses.replace(docs[i], groups=frozenset()) for i in range(40)]
    mixed = list(docs) + bg_docs
    dag = DagGate([(), (0,)])                      # root + one anchor (group A) — trivial here
    out = PGSTMDag(K=part.K, V=40, partition=part, dag=dag, P=1, n_iter=15, seed=0).fit(
        mixed, [frozenset({1})] * len(docs) + [frozenset({0})] * len(bg_docs))
    assert out["beta"].shape == (part.K, 40)
    assert out["Sigma"].shape == (part.K - 1, part.K - 1)
    assert np.isfinite(out["beta"]).all() and np.isfinite(out["Sigma"]).all()
```

Note: the `mixed` corpus reuses group-"A"/"B" docs from `gated_ln_corpus_stick` (both real groups) so `assemble_sigma` sees both group blocks populated; the background-only docs carry `groups=frozenset()` and `doc_nodes = frozenset({0})` (root → all-zero offset indicator).

- [ ] **Step 2: Run test, verify fail** — FAIL (group-less docs hit `(g,) = tuple(doc.groups)` in `fit` / `_bg_estep_doc` undefined).

- [ ] **Step 3: Implement**

Add to the imports at the top of `pg_stm_dag.py` (extend the existing `from spark_vi.models.topic.pg_stm import (...)` block and add the `_linalg` import):

```python
from spark_vi.models.topic.pg_stm import (
    pg_empty_stats, pg_accumulate_doc, pg_estep_doc, beta_dirichlet_mean,
    assemble_sigma, stick_layout,
    expected_log_theta, token_responsibilities, stick_trials, omega_expectation,
    psi_posterior, _PSI_CLIP,
)
from spark_vi.models.topic._linalg import safe_inverse
```

Add the flat E-step (module-level function, near `PGSTMDag`):

```python
def _bg_estep_doc(doc, bg_lay, log_beta, Gamma, Sigma, *, K, inner_rounds, inner_tol):
    """Flat (non-gated) background-only mean-field PG E-step for a document with no
    foreground group. Uses only the background block: active = background sticks, allowed
    = background topics, a flat stick-breaking (no gate stick). Returns the same
    (m, V, phi, active, allowed, mu_active, n_clips) tuple as pg_estep_doc, so
    pg_accumulate_doc consumes it identically. Reuses pg_stm's flat primitives unchanged;
    it is pg_estep_doc with the gate term removed."""
    active = bg_lay["active"]; allowed = bg_lay["allowed"]
    Sigma_inv_active = safe_inverse(Sigma[np.ix_(active, active)])
    mu_active = (Gamma.T @ doc.x)[active]
    A = active.shape[0]
    m = mu_active.copy()
    V = np.eye(A)
    phi = None
    n_clips = 0
    for _ in range(inner_rounds):
        m_prev = m
        vdiag = np.diag(V)
        mc = np.clip(m, -_PSI_CLIP, _PSI_CLIP)
        if not np.array_equal(mc, m):
            n_clips += 1
        elog_flat = expected_log_theta(mc, vdiag)          # (len(allowed),)
        elog_theta = np.full(K, -np.inf)
        elog_theta[allowed] = elog_flat
        phi, n_full = token_responsibilities(
            doc.indices, elog_theta, log_beta, allowed, counts=doc.counts)
        n_allowed = n_full[allowed]
        b_active = stick_trials(n_allowed)
        c = np.sqrt(m ** 2 + vdiag)
        omega = omega_expectation(b_active, c)
        m, V = psi_posterior(n_allowed, b_active, mu_active, Sigma_inv_active, omega)
        if np.max(np.abs(m - m_prev)) < inner_tol:
            break
    return m, V, phi, active, allowed, mu_active, n_clips
```

In `PGSTMDag.fit`, build the background layout once (right after `stats = pg_empty_stats(...)` is first created, or just before the loop) and route per document. Replace the per-doc block:

```python
        bg_lay = {"active": self.layout["bg_sticks"],
                  "allowed": self.partition.background_indices().astype(np.int64)}
        ...
            for d, doc in enumerate(docs_aug):
                gs = tuple(doc.groups)
                if gs:
                    (g,) = gs
                    glay = self.layout["groups"][g]
                    estep = pg_estep_doc(doc, glay, log_beta, Cf, Sigma, K=K,
                                         B=self.layout["B"], inner_rounds=self.inner_rounds,
                                         inner_tol=self.inner_tol)
                else:
                    estep = _bg_estep_doc(doc, bg_lay, log_beta, Cf, Sigma, K=K,
                                          inner_rounds=self.inner_rounds, inner_tol=self.inner_tol)
                m, Vd, phi, active, allowed, mu_active, _nc = estep
                pg_accumulate_doc(stats, doc, (m, Vd, phi, active, allowed, mu_active), K=K)
                psi_mean[d, active] = m
```

(`bg_lay` only depends on `self.layout`/`self.partition`, so compute it once before the `for _ in range(self.n_iter)` loop, not per iteration.)

- [ ] **Step 4: Run test, verify pass** — `pytest tests/test_pg_stm_dag.py -q`. The new routing test passes; all prior tests (Tasks 1–4 of the reparam step) stay green.

- [ ] **Step 5: Commit** — `git commit -am "feat(dag): flat background-only E-step + fit routing for group-less documents"`

---

### Task 3: `dag_offset_corpus` background-only arm + flat-recovery check

Let the plant emit background-only documents (no group, attesting only the root, generated by a flat background stick-breaking), and validate the flat path recovers a planted background composition end-to-end.

**Files:**
- Modify: `spark-vi/tests/_stm_synth.py` (`dag_offset_corpus`)
- Test: `spark-vi/tests/test_pg_stm_dag.py`

**Interfaces:**
- Produces: `dag_offset_corpus(..., n_background_only=0, ...)` — a new keyword; when `> 0`, appends that many background-only documents: `groups=frozenset()`, `doc_nodes` entry `frozenset({0})`, tokens drawn from a flat background stick-breaking `theta_bg = stick_to_simplex(psi_bg)`, `psi_bg ~ N(0, sigma_true[bg_sticks, bg_sticks])`, mapped onto the background topics. Returns the same `(docs, doc_nodes)` shape (background-only docs appended at the end).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_pg_stm_dag.py
def test_background_only_corpus_recovers_planted_background():
    """PLANTED: a pure background-only corpus (no groups, flat background stick-breaking)
    on a planted background beta. REAL: nothing. Synthetic -> MATH-CORRECTNESS: fitting a
    background-only-only corpus with PGSTMDag recovers the planted BACKGROUND topics
    (per-background-topic top word matches), validating the flat E-step end-to-end and its
    consistency with the shared background block. Does NOT prove gated-mixture behavior or
    transfer."""
    import numpy as np
    from spark_vi.models.topic.partition import TopicBlockPartition
    from spark_vi.models.topic.pg_stm_dag import DagGate, PGSTMDag
    from tests._stm_synth import dag_offset_corpus, real_beta_from
    part = TopicBlockPartition(group_var="g", background_k=5, foreground=(("A", 3),))
    K, V = part.K, 60
    beta = real_beta_from(K, V, seed=1)
    dag = DagGate([(), (0,)])
    Ksm1 = K - 1
    node_offsets = {0: np.zeros(Ksm1), 1: np.zeros(Ksm1)}
    docs, doc_nodes = dag_offset_corpus(
        dag=dag, node_offsets=node_offsets, partition=part, beta=beta,
        node_of_group={"A": 1}, doc_nodes_plan={}, n_background_only=400,
        sigma_true=2.0 * np.eye(Ksm1), doc_len=60, seed=3)
    assert len(docs) == 400 and all(d.groups == frozenset() for d in docs)
    out = PGSTMDag(K=K, V=V, partition=part, dag=dag, P=1, n_iter=40, seed=0).fit(docs, doc_nodes)
    # each of the 5 planted background topics' top word should appear as some fitted
    # background topic's top word (recovery up to label permutation)
    planted_top = {int(beta[k].argmax()) for k in range(part.background_k)}
    fitted_top = {int(out["beta"][k].argmax()) for k in range(part.background_k)}
    overlap = len(planted_top & fitted_top)
    assert overlap >= 4, f"background not recovered by the flat path: overlap={overlap}/5"
```

- [ ] **Step 2: Run test, verify fail** — FAIL (`dag_offset_corpus` has no `n_background_only`).

- [ ] **Step 3: Implement** — in `tests/_stm_synth.py`, add the keyword and the background-only generation to `dag_offset_corpus`. Add `n_background_only=0` to its signature. Import `stick_to_simplex` alongside the existing `from spark_vi.models.topic.pg_stm import stick_layout, gated_theta` (make it `stick_layout, gated_theta, stick_to_simplex`). After the existing per-node generation loop, before `return docs, doc_nodes`, append:

```python
    # background-only members: no group, attest only the root, flat background stick-breaking
    if n_background_only > 0:
        bg_sticks = lay["bg_sticks"]
        bg_topics = partition.background_indices()
        Bk = partition.background_k
        Sbg = sigma_true[np.ix_(bg_sticks, bg_sticks)]
        mu_bg = np.zeros(len(bg_sticks))                 # root offset is 0
        beta_bg = beta[:Bk]                              # (Bk, V) background topic-word rows
        for _ in range(n_background_only):
            psi_bg = rng.multivariate_normal(mu_bg, Sbg)
            theta_bg = stick_to_simplex(psi_bg)          # (Bk,)
            toks = rng.choice(beta.shape[1], size=doc_len, p=theta_bg @ beta_bg)
            u_, c_ = np.unique(toks, return_counts=True)
            docs.append(STMDocument(indices=u_.astype(np.int32), counts=c_.astype(np.float64),
                                    length=int(c_.sum()), x=np.array([1.0]),
                                    groups=frozenset()))
            doc_nodes.append(frozenset({0}))
    return docs, doc_nodes
```

(`rng`, `lay`, `STMDocument` are already in scope in `dag_offset_corpus`; confirm the `STMDocument` import at the top of `_stm_synth.py` and reuse it. `theta_bg @ beta_bg` is a length-V word distribution; it sums to 1 because `theta_bg` sums to 1 and each `beta_bg` row sums to 1.)

- [ ] **Step 4: Run test, verify pass** — `pytest tests/test_pg_stm_dag.py::test_background_only_corpus_recovers_planted_background -q`. If `overlap` is 3/5 (borderline), raise `n_background_only` to 600 or `n_iter` to 60; do not lower the `>= 4` bar without recording why. If it stays below 4 after both, STOP and report BLOCKED with the measured overlap — the flat E-step may be inconsistent with the shared background block (the risk this task exists to check).

- [ ] **Step 5: Commit** — `git commit -am "test(dag): background-only plant arm + flat-path background recovery"`

---

### Task 4: Realistic anchor-level + path-sum recovery with background-only members

The payoff: with background-only members present, the anchor-partition trap is broken, so anchor increments become identified and leaf path-sums recover — for both a with-direct-docs anchor and a no-direct-docs anchor.

**Files:**
- Test: `spark-vi/tests/test_pg_stm_dag.py`

**Interfaces:**
- Consumes: `dag_offset_corpus(..., n_background_only=...)`, `PGSTMDag.fit` (`B` node-id-indexed, `offset_uncertainty["identified"]`), `DagGate.closure`, `stick_layout`, `real_beta_from`.

- [ ] **Step 1: Write the test** (may pass on first run — it exercises Tasks 1–3; that is expected for a validation test)

```python
# append to tests/test_pg_stm_dag.py
def test_pathsums_recover_and_anchors_identify_with_background_only_members():
    """PLANTED: node offsets + Sigma on root->{anchor A with anchor-level docs + subtype;
    anchor B with ONLY a subtype (no anchor-level docs)}, PLUS background-only members (no
    group, root only). REAL: overlap beta. Realistic-overlap synthetic -> MATH-CORRECTNESS:
    background-only members break the anchor-partition dummy trap (insight 0050), so the
    anchor increments become data-identified (identified=True) and the leaf PATH-SUMS
    recover for BOTH the with-direct-docs anchor and the no-direct-docs anchor. Contrast:
    without background-only members the same path-sums are un-identified (r ~ 0.1). Does NOT
    claim calibrated absolute intervals (that is the read-out engine, insight 0052) nor
    transfer to real data. The path-sum point estimate may still carry residual mean-field
    bias (0052); the robust claim is the identified-flag flip + substantial path-sum
    improvement over the trapped baseline."""
    import numpy as np
    from spark_vi.models.topic.partition import TopicBlockPartition
    from spark_vi.models.topic.pg_stm_dag import DagGate, PGSTMDag
    from spark_vi.models.topic.pg_stm import stick_layout
    from tests._stm_synth import dag_offset_corpus, real_beta_from
    part = TopicBlockPartition(group_var="g", background_k=6, foreground=(("A", 4), ("B", 4)))
    K, V = part.K, 300
    dag = DagGate([(), (0,), (0,), (1,), (2,)])       # A=1 (anchor docs), A1=3; B=2 (no docs), B1=4
    Ksm1 = K - 1
    rng = np.random.default_rng(4)
    node_offsets = {u: rng.standard_normal(Ksm1) for u in (1, 2, 3, 4)}
    node_offsets[0] = np.zeros(Ksm1)
    beta = real_beta_from(K, V, seed=2)
    docs, doc_nodes = dag_offset_corpus(
        dag=dag, node_offsets=node_offsets, partition=part, beta=beta,
        node_of_group={"A": 1, "B": 2},
        doc_nodes_plan={1: 400, 3: 400, 4: 500},       # A has anchor-level docs; B has none
        n_background_only=600,                          # break the trap
        sigma_true=3.0 * np.eye(Ksm1), doc_len=80, seed=6)
    out = PGSTMDag(K=K, V=V, partition=part, dag=dag, P=docs[0].x.shape[0],
                   n_iter=60, lam_base=1e-3, seed=0).fit(docs, doc_nodes)
    B = out["B"]; ident = out["offset_uncertainty"]["identified"]
    lay = stick_layout(part)
    def pathsum_r(leaf, group):
        act = lay["groups"][group]["active"]
        recon = np.zeros(Ksm1); truth = np.zeros(Ksm1)
        for u in dag.closure(frozenset({leaf})):
            recon = recon + B[u]; truth = truth + node_offsets[u]
        return np.corrcoef(recon[act], truth[act])[0, 1]
    r3 = pathsum_r(3, "A"); r4 = pathsum_r(4, "B")
    # anchor increments are now data-identified (the trap is broken)
    assert ident[1] == True, f"anchor A increment still un-identified: {ident}"
    # path-sums recover substantially better than the trapped ~0.1 baseline
    assert r3 > 0.5, f"with-direct-docs anchor path-sum weak: r={r3:.2f}"
    assert r4 > 0.5, f"no-direct-docs anchor path-sum weak: r={r4:.2f}"
```

- [ ] **Step 2: Run test** — `pytest tests/test_pg_stm_dag.py::test_pathsums_recover_and_anchors_identify_with_background_only_members -q`. This exercises Tasks 1–3 (no new implementation).

- [ ] **Step 3: Interpret the result (do NOT weaken assertions).**
  - **If it passes:** the gate change delivered — commit.
  - **If `ident[1]` is False:** background-only members did not break the trap — STOP, report BLOCKED with the `identified` array and the number of background-only docs actually in the fit (a routing/accumulation gap, likely Task 2). Do not weaken.
  - **If `ident[1]` is True but a path-sum `r` is between ~0.2 and 0.5:** identification was restored (the structural win) but the point estimate is mean-field-limited (insight 0052). This is a REAL, expected-possible finding, not a test-writing failure. Report DONE_WITH_CONCERNS with both `r` values; the controller will decide whether to (a) record it as the mean-field-limit finding and relax the path-sum bar to assert only `ident[1]==True` + `r3,r4 > trapped-baseline`, or (b) escalate to the read-out engine (step 2). First try the sanctioned escalations: raise `n_background_only` to 1200 and `n_iter` to 90. Report the measured `r` at each.
  - **If a path-sum `r` is still ~0.1 with `ident[1]` True:** contradictory — report BLOCKED with the full numbers for the controller.

- [ ] **Step 4: Commit** (only if Step 3 resolves to a passing assertion) — `git commit -am "test(dag): path-sums recover + anchors identify with background-only members (trap broken)"`

---

## Self-Review

**Spec coverage:** group-less accumulation = Task 1 (`pg_stm.py` guard); flat background-only E-step + routing = Task 2 (`_bg_estep_doc` + `fit`); plant background-only arm + flat-path correctness = Task 3 (`dag_offset_corpus` + recovery test); realistic anchor-identification + path-sum recovery = Task 4. The "consistency risk" (flat path coherent with the shared background block) is checked by Task 3's background-recovery test. Domain-agnostic + honesty docstrings = every task.

**Placeholder scan:** no TBD/TODO; every code step carries runnable code. Task 4's Step 3 enumerates concrete outcome branches with concrete next actions (BLOCKED with named arrays / DONE_WITH_CONCERNS with the two r values + sanctioned escalations), not hand-waving — because the payoff magnitude is genuinely uncertain (identification is restored deterministically; the residual mean-field bias on the absolute path-sum is the measured unknown, per insight 0052).

**Type consistency:** `_bg_estep_doc` returns the exact 7-tuple `pg_estep_doc` returns, so `pg_accumulate_doc` consumes both identically (verified against `pg_stm.py:142`). `bg_lay` keys (`active`, `allowed`) match `_bg_estep_doc`'s reads. `dag_offset_corpus(..., n_background_only=...)` appends `groups=frozenset()` docs with `doc_nodes=frozenset({0})`; Task 4 reads `out["B"]` (node-id-indexed, from the reparam step) and `out["offset_uncertainty"]["identified"]` (from the reparam step's Task 4) — both already exist on `PGSTMDag.fit`'s return.

**Known risk to watch:** the flat background-only E-step must produce background-block scatter consistent with the gated docs' background handling (the gate rescales background mass but not its within-block composition, so the background stick logits share the same Σ background block). Task 3's pure-background recovery test is the guard; if it fails, the flat/gated background composition is inconsistent and needs reconciliation before Task 4 is meaningful.
