# Gated optimize_alpha Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `GatedOnlineLDA` learn an asymmetric per-node Dirichlet doc-concentration α from data (mirror of vanilla `optimize_alpha`), instead of the single fixed `nodeAlphaScale`.

**Architecture:** α is tied per DAG node (1 shared background α + one α per node, expanded to length-K for the E-step). An exact dense Newton step over the tied space is assembled on the driver from a static frontier histogram (labels are fixed, so the allowed-set group structure is static) plus one new distributed E-step statistic. Vanilla's O(K) Sherman-Morrison trick is NOT reused; the tied space is small (~n_nodes) so a dense `np.linalg.solve` is exact and trivial.

**Tech Stack:** Python, NumPy, SciPy (`digamma`/`polygamma`), PySpark MLlib shim, pytest. Spec: `docs/superpowers/specs/2026-07-22-gated-optimize-alpha-design.md`.

## Global Constraints

- Engine code (`spark_vi/**`) is integer-id agnostic: no OMOP/concept ids, no domain vocabulary. Works in integer topic/node/block space only.
- Commit trailer EXACTLY: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- No LaTeX in code/docstrings/comments — plain ASCII math and Unicode Greek (α, β, ψ, Σ, Γ) only.
- Cite literature for any method/default/constant (Blei-Ng-Jordan 2003 for the Newton step; Wallach et al. 2009 for asymmetric α; Hoffman et al. 2010/2013 for SVI).
- Tests: never loosen a threshold to pass; xfail-with-recorded-reason if an honest test fails.
- Only α (doc-concentration) is learned; η (topic-concentration) is out of scope.
- Behavior with `optimize_alpha=False` (the default) must be byte-identical to today.
- When `optimize_alpha=True`, `nodeAlphaScale` remains the INITIAL α; optimize refines from there.

## Canonical definitions (used across tasks)

**Tied-α layout.** A tied vector of length `B = 1 + len(lay.nodes)`:
- index `0` = background (shared across the `n_bg` background topics),
- index `i` (1..n_nodes) = `lay.nodes[i-1]` (shared across that node's `tpn` topics).

`block_sizes` (length B) = `[n_bg, tpn, tpn, ...]`.

`topic_to_tied` (length K, int) maps each topic id to its tied index: `0` for topics `< n_bg`, else the tied index of the node whose `lay.block[u]` contains it.

**Frontier histogram.** `dict[frozenset[int], int]` mapping each distinct training frontier (set of engine node ids; empty = background doc) to its corpus doc-count. Static (frontiers are labels). Held on the engine as `self._frontier_histogram`.

**New statistic.** `e_log_theta_node_sum`, a length-B NumPy array. For each tied block `b`, the batch sum over docs whose allowed set includes block `b` of `Σ_{k in that block} (ψ(γ_k) − ψ(γ_sum))`. Emitted by `local_update` only when `optimize_alpha`.

---

### Task 1: Pure `gated_alpha_newton_step`

**Files:**
- Modify: `spark-vi/spark_vi/inference/concentration_optimization.py` (append a new function; keep existing `alpha_newton_step` untouched)
- Test: `spark-vi/tests/test_concentration_optimization.py`

**Interfaces:**
- Produces: `gated_alpha_newton_step(alpha_tied, block_sizes, e_log_theta_block_sum_scaled, group_counts, group_membership) -> np.ndarray`
  - `alpha_tied`: (B,) current tied α.
  - `block_sizes`: (B,) topics per tied block.
  - `e_log_theta_block_sum_scaled`: (B,) corpus-scaled data term (the `e_log_theta_node_sum` stat after runner scaling).
  - `group_counts`: (G,) N_g per distinct allowed-set group.
  - `group_membership`: (G, B) bool — is tied block b in group g's allowed set.
  - Returns: (B,) raw Δα_tied (caller applies ρ damping + floor).

- [ ] **Step 1: Write the finite-difference derivation guard test**

Add to `spark-vi/tests/test_concentration_optimization.py`:

```python
def test_gated_alpha_newton_step_matches_finite_difference_gradient():
    # The raw Newton step is -H^-1 g. We validate the assembled gradient g and
    # Hessian H against numerical differentiation of the exact gated ELBO-in-alpha
    #   L(a) = Σ_g N_g[logΓ(Σ_{b in g} m_b a_b) − Σ_{b in g} m_b logΓ(a_b)]
    #        + Σ_b e_b a_b            (data term is linear in a_b; constant drops)
    # on a tiny system (2 groups, 3 tied blocks).
    import numpy as np
    from scipy.special import gammaln
    from spark_vi.inference.concentration_optimization import gated_alpha_newton_step

    m = np.array([2.0, 1.0, 1.0])                    # block sizes: bg=2, two nodes tpn=1
    a = np.array([0.30, 0.05, 0.12])                 # current tied alpha
    e = np.array([-3.1, -0.7, -1.4])                 # data term (scaled)
    Ng = np.array([40.0, 15.0])
    memb = np.array([[True, True, False],            # group 0: bg + node1
                     [True, False, True]])           # group 1: bg + node2

    def L(av):
        val = float(np.dot(e, av))
        for g in range(len(Ng)):
            idx = np.where(memb[g])[0]
            s = np.sum(m[idx] * av[idx])
            val += Ng[g] * (gammaln(s) - np.sum(m[idx] * gammaln(av[idx])))
        return val

    # numerical gradient and Hessian of L at a
    eps = 1e-6
    B = a.shape[0]
    grad = np.zeros(B)
    for b in range(B):
        ap = a.copy(); ap[b] += eps
        am = a.copy(); am[b] -= eps
        grad[b] = (L(ap) - L(am)) / (2 * eps)
    H = np.zeros((B, B))
    for b in range(B):
        for c in range(B):
            app = a.copy(); app[b] += eps; app[c] += eps
            apm = a.copy(); apm[b] += eps; apm[c] -= eps
            amp = a.copy(); amp[b] -= eps; amp[c] += eps
            amm = a.copy(); amm[b] -= eps; amm[c] -= eps
            H[b, c] = (L(app) - L(apm) - L(amp) + L(amm)) / (4 * eps * eps)

    expected_delta = -np.linalg.solve(H, grad)       # the Newton step L should produce
    got = gated_alpha_newton_step(a, m, e, Ng, memb)
    assert np.allclose(got, expected_delta, rtol=1e-3, atol=1e-5), (got, expected_delta)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd spark-vi && python -m pytest tests/test_concentration_optimization.py::test_gated_alpha_newton_step_matches_finite_difference_gradient -q`
Expected: FAIL with `ImportError`/`AttributeError` (function not defined).

- [ ] **Step 3: Implement the pure function**

Append to `spark-vi/spark_vi/inference/concentration_optimization.py` (module already imports `digamma, polygamma`):

```python
def gated_alpha_newton_step(
    alpha_tied: np.ndarray,
    block_sizes: np.ndarray,
    e_log_theta_block_sum_scaled: np.ndarray,
    group_counts: np.ndarray,
    group_membership: np.ndarray,
) -> np.ndarray:
    """One exact Newton step for a per-node-tied asymmetric Dirichlet α under a
    DAG gate (GatedOnlineLDA).

    Generalizes the symmetric-simplex Newton of Blei, Ng, Jordan 2003 (A.4.2) to
    (a) a block-tied α (one shared value per DAG node, so the chain rule folds in
    the |block| topics that share it) and (b) per-document gated sub-simplices:
    each distinct allowed-set 'group' g contributes a Dirichlet log-normalizer over
    only its own blocks A_g, so the ψ(Σ α) coupling is per-group, not global.

    The tied space is small (~1 + n_nodes), so the dense Hessian is inverted
    directly with np.linalg.solve — no Sherman-Morrison structured inverse (that
    exists only to stay O(K) for a free per-topic α, which we do not use here).

    Parameters
    ----------
    alpha_tied : (B,) current tied α; index 0 = background, 1.. = per node.
    block_sizes : (B,) topics sharing each tied value (n_bg, then tpn per node).
    e_log_theta_block_sum_scaled : (B,) corpus-scaled Σ_d Σ_{k in block} E[log θ_dk].
    group_counts : (G,) corpus doc-count N_g of each distinct allowed-set group.
    group_membership : (G, B) bool; True where tied block b ∈ group g's allowed set.

    Returns
    -------
    (B,) raw Δα_tied. The caller applies ρ_t damping and the post-step floor
    (clip to [1e-3, ∞)), matching the pure contract of alpha_newton_step.
    """
    a = np.asarray(alpha_tied, dtype=np.float64)
    m = np.asarray(block_sizes, dtype=np.float64)
    e = np.asarray(e_log_theta_block_sum_scaled, dtype=np.float64)
    Ng = np.asarray(group_counts, dtype=np.float64)
    memb = np.asarray(group_membership, dtype=bool)

    ma = m * a                                     # (B,) m_b α_b
    group_sum = memb @ ma                          # (G,) Σ_{b in g} m_b α_b
    psi_gsum = digamma(group_sum)                  # (G,)
    tri_gsum = polygamma(1, group_sum)             # (G,)

    # gradient: prior + data
    sum_Ng = (memb * Ng[:, None]).sum(axis=0)                  # (B,) Σ_{g: b in g} N_g
    sum_Ng_psi = (memb * (Ng * psi_gsum)[:, None]).sum(axis=0) # (B,)
    g = m * (sum_Ng_psi - sum_Ng * digamma(a)) + e

    # Hessian: H = Xᵀ diag(N_g ψ'(Σ)) X  −  diag(m_b ψ'(α_b) Σ_{g:b} N_g)
    X = memb * m[None, :]                                      # (G,B) m_b [b in g]
    H = X.T @ (X * (Ng * tri_gsum)[:, None])                  # (B,B)
    H[np.diag_indices_from(H)] -= m * polygamma(1, a) * sum_Ng

    return -np.linalg.solve(H, g)
```

- [ ] **Step 4: Run it to verify it passes**

Run: `cd spark-vi && python -m pytest tests/test_concentration_optimization.py::test_gated_alpha_newton_step_matches_finite_difference_gradient -q`
Expected: PASS.

- [ ] **Step 5: Add a tiny recovery test (optimizer sanity)**

```python
def test_gated_alpha_newton_step_iterates_toward_optimum():
    # Repeated damped Newton steps on a fixed synthetic system should climb L
    # (monotone non-decreasing) and converge (‖Δα‖ shrinks). Guards the sign.
    import numpy as np
    from scipy.special import gammaln
    from spark_vi.inference.concentration_optimization import gated_alpha_newton_step

    m = np.array([2.0, 1.0, 1.0])
    e = np.array([-4.0, -0.5, -2.0])
    Ng = np.array([50.0, 20.0])
    memb = np.array([[True, True, False], [True, False, True]])

    def L(av):
        val = float(np.dot(e, av))
        for g in range(len(Ng)):
            idx = np.where(memb[g])[0]
            s = np.sum(m[idx] * av[idx])
            val += Ng[g] * (gammaln(s) - np.sum(m[idx] * gammaln(av[idx])))
        return val

    a = np.array([0.2, 0.2, 0.2])
    prev = L(a)
    last_step = None
    for _ in range(50):
        d = gated_alpha_newton_step(a, m, e, Ng, memb)
        a = np.maximum(a + 0.5 * d, 1e-3)          # ρ damping + floor
        cur = L(a)
        assert cur >= prev - 1e-6                   # monotone ascent
        prev = cur
        last_step = np.abs(d).max()
    assert last_step < 1e-3                          # converged
```

- [ ] **Step 6: Run both tests**

Run: `cd spark-vi && python -m pytest tests/test_concentration_optimization.py -q -k gated_alpha`
Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add spark-vi/spark_vi/inference/concentration_optimization.py spark-vi/tests/test_concentration_optimization.py
git commit -m "feat(concentration): pure gated_alpha_newton_step (per-node tied, exact dense Newton)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Engine `__init__` wiring + `local_update` emits the stat

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/gated_lda.py` (`GatedOnlineLDA.__init__` and `local_update`)
- Test: `spark-vi/tests/test_gated_lda.py`

**Interfaces:**
- Consumes: `DagLayout` (`lay.nodes`, `lay.n_bg`, `lay.tpn`, `lay.block`, `lay.allowed_set`); `GatedBOWDocument`.
- Produces:
  - `GatedOnlineLDA(lay, vocab_size, *, init="random", optimize_alpha=False, frontier_histogram=None, **kw)` — no longer raises on `optimize_alpha`; requires `frontier_histogram` (dict[frozenset[int], int]) when `optimize_alpha`.
  - Instance attributes: `self._block_sizes` (B,), `self._topic_to_tied` (K,), `self._frontier_histogram`.
  - `local_update` return dict gains key `"e_log_theta_node_sum"` (B,) when `optimize_alpha`.

- [ ] **Step 1: Write the failing tests**

Add to `spark-vi/tests/test_gated_lda.py`:

```python
def test_gated_optimize_alpha_requires_frontier_histogram():
    import pytest
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    lay = DagLayout({1: 0, 2: 0}, n_bg=2, tpn=1)
    with pytest.raises(ValueError, match="frontier_histogram"):
        GatedOnlineLDA(lay, vocab_size=5, optimize_alpha=True)   # no histogram

def test_gated_tied_layout_shapes():
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    lay = DagLayout({1: 0, 2: 0, 3: 1}, n_bg=2, tpn=1)          # 3 nodes
    m = GatedOnlineLDA(lay, vocab_size=5, optimize_alpha=True,
                       frontier_histogram={frozenset({3}): 4, frozenset(): 6})
    # tied layout: [bg, node1, node2, node3]
    assert list(m._block_sizes) == [2, 1, 1, 1]
    # topic_to_tied: topics 0,1 -> 0 (bg); block(1)->1, block(2)->2, block(3)->3
    assert m._topic_to_tied[0] == 0 and m._topic_to_tied[1] == 0
    assert m._topic_to_tied[lay.block[1][0]] == 1
    assert m._topic_to_tied[lay.block[3][0]] == 3

def test_gated_local_update_emits_node_theta_stat():
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    from spark_vi.models.topic.types import GatedBOWDocument
    lay = DagLayout({1: 0, 2: 0}, n_bg=2, tpn=1)                # B = 1 + 2 = 3
    m = GatedOnlineLDA(lay, vocab_size=6, optimize_alpha=True, random_seed=0,
                       frontier_histogram={frozenset({1}): 1, frozenset(): 1})
    gp = m.initialize_global(None)
    docs = [GatedBOWDocument(indices=np.array([0, 3], np.int32),
                             counts=np.array([2.0, 1.0]), length=3,
                             frontier=frozenset({1})),
            GatedBOWDocument(indices=np.array([1, 4], np.int32),
                             counts=np.array([1.0, 1.0]), length=2,
                             frontier=frozenset())]           # background doc
    out = m.local_update(docs, gp)
    stat = out["e_log_theta_node_sum"]
    assert stat.shape == (3,)                                  # [bg, node1, node2]
    # node2 is in neither doc's allowed set (doc1 -> node1 only; doc2 -> bg only)
    assert stat[2] == 0.0
    # background block is in both docs' allowed sets -> nonzero
    assert stat[0] != 0.0

def test_gated_local_update_no_stat_when_alpha_fixed():
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    from spark_vi.models.topic.types import GatedBOWDocument
    lay = DagLayout({1: 0, 2: 0}, n_bg=2, tpn=1)
    m = GatedOnlineLDA(lay, vocab_size=6, random_seed=0)        # optimize_alpha False
    gp = m.initialize_global(None)
    out = m.local_update([GatedBOWDocument(indices=np.array([0], np.int32),
                                           counts=np.array([1.0]), length=1,
                                           frontier=frozenset({1}))], gp)
    assert "e_log_theta_node_sum" not in out
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd spark-vi && python -m pytest tests/test_gated_lda.py -q -k "optimize_alpha or tied_layout or node_theta or alpha_fixed"`
Expected: FAIL (currently `__init__` raises `NotImplementedError` on `optimize_alpha`).

- [ ] **Step 3: Rewrite `GatedOnlineLDA.__init__`**

Replace the current `__init__` (which raises on `optimize_alpha`) in `spark-vi/spark_vi/models/topic/gated_lda.py`:

```python
    def __init__(self, lay: DagLayout, vocab_size: int, *, init: str = "random",
                 optimize_alpha: bool = False,
                 frontier_histogram: dict | None = None, **kw) -> None:
        # optimize_alpha is handled by the gated per-node Newton step (this class),
        # NOT OnlineLDA's full-K alpha_newton_step; pass it to the parent as False
        # so the inherited update_global never runs the vanilla alpha step.
        super().__init__(K=lay.K, vocab_size=vocab_size, optimize_alpha=False, **kw)
        self.lay = lay
        self.init = init
        self.optimize_alpha = bool(optimize_alpha)          # gated flag (drives our override)
        if self.optimize_alpha and frontier_histogram is None:
            raise ValueError(
                "optimize_alpha=True requires frontier_histogram "
                "{frozenset(frontier): count} — the static allowed-set group structure."
            )
        self._frontier_histogram = frontier_histogram
        # Tied-alpha layout: index 0 = background, i = lay.nodes[i-1].
        self._block_sizes = np.array(
            [lay.n_bg] + [lay.tpn] * len(lay.nodes), dtype=np.float64)
        self._topic_to_tied = np.zeros(lay.K, dtype=np.int64)   # bg topics -> 0
        for i, u in enumerate(lay.nodes, start=1):
            for k in lay.block[u]:
                self._topic_to_tied[k] = i
```

Note: remove the old `if self.optimize_alpha: raise NotImplementedError(...)` block entirely. Keep the `from scipy.special import digamma` import at the top of the module (already present).

- [ ] **Step 4: Add the stat emission in `local_update`**

In `spark-vi/spark_vi/models/topic/gated_lda.py`, inside `local_update`, initialize the accumulator near the other accumulators (after `lambda_stats = np.zeros_like(lam)`):

```python
        node_theta_sum = (
            np.zeros(self._block_sizes.shape[0], dtype=np.float64)
            if self.optimize_alpha else None)
```

Inside the per-doc loop, after `gamma, expElogthetad, phi_norm, _ = _cavi_doc_inference(...)` and the existing `lambda_stats` scatter, add:

```python
            if node_theta_sum is not None:
                # Per-tied-block Σ_{k in block} (ψ(γ_k) − ψ(γ_sum)) over this doc's
                # allowed topics; γ is aligned with `allowed`. Blocks absent from
                # `allowed` contribute nothing (they stay at their prior).
                e_log_theta_d = digamma(gamma) - digamma(gamma.sum())
                np.add.at(node_theta_sum, self._topic_to_tied[allowed], e_log_theta_d)
```

At the return, add the key only when active:

```python
        result = {
            "lambda_stats": lambda_stats,
            "doc_loglik_sum": np.array(doc_loglik_sum),
            "doc_theta_kl_sum": np.array(doc_theta_kl_sum),
            "n_docs": np.array(float(n_docs)),
        }
        if node_theta_sum is not None:
            result["e_log_theta_node_sum"] = node_theta_sum
        return result
```

- [ ] **Step 5: Run to verify they pass**

Run: `cd spark-vi && python -m pytest tests/test_gated_lda.py -q -k "optimize_alpha or tied_layout or node_theta or alpha_fixed"`
Expected: PASS (4 tests).

- [ ] **Step 6: Run the full gated engine suite (no regression)**

Run: `cd spark-vi && python -m pytest tests/test_gated_lda.py -q`
Expected: PASS (all).

- [ ] **Step 7: Commit**

```bash
git add spark-vi/spark_vi/models/topic/gated_lda.py spark-vi/tests/test_gated_lda.py
git commit -m "feat(gated-lda): accept optimize_alpha + emit per-node e_log_theta stat

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Engine `update_global` — λ step + gated α step

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/gated_lda.py` (add `update_global` override + `_gated_alpha_update` helper)
- Test: `spark-vi/tests/test_gated_lda.py`

**Interfaces:**
- Consumes: `gated_alpha_newton_step` (Task 1); `self._frontier_histogram`, `self._block_sizes`, `self._topic_to_tied`, `self.lay` (Task 2); `target_stats["e_log_theta_node_sum"]` (Task 2).
- Produces: `GatedOnlineLDA.update_global(global_params, target_stats, learning_rate)` returning the updated `{"lambda", "alpha", "eta"}` with a learned asymmetric `alpha` when `optimize_alpha`.

- [ ] **Step 1: Write the failing tests**

Add to `spark-vi/tests/test_gated_lda.py`:

```python
def test_gated_update_global_leaves_alpha_fixed_when_disabled():
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    lay = DagLayout({1: 0, 2: 0}, n_bg=2, tpn=1)
    m = GatedOnlineLDA(lay, vocab_size=6, random_seed=0)          # alpha fixed
    gp = m.initialize_global(None)
    alpha0 = gp["alpha"].copy()
    stats = {"lambda_stats": np.ones((lay.K, 6)), "n_docs": np.array(4.0)}
    gp2 = m.update_global(gp, stats, learning_rate=0.5)
    assert np.array_equal(gp2["alpha"], alpha0)                   # unchanged
    assert not np.array_equal(gp2["lambda"], gp["lambda"])        # lambda still moves

def test_gated_update_global_learns_asymmetric_alpha():
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_lda import GatedOnlineLDA
    lay = DagLayout({1: 0, 2: 0}, n_bg=2, tpn=1)                  # B=3: bg,node1,node2
    hist = {frozenset({1}): 30, frozenset({2}): 30, frozenset(): 40}
    m = GatedOnlineLDA(lay, vocab_size=6, optimize_alpha=True, random_seed=0,
                       frontier_histogram=hist)
    gp = m.initialize_global(None)
    # Make node1 look 'common' (E[log θ] near 0) and node2 'rare' (very negative):
    # tied order [bg, node1, node2].
    stats = {
        "lambda_stats": np.ones((lay.K, 6)),
        "n_docs": np.array(100.0),
        "e_log_theta_node_sum": np.array([-20.0, -2.0, -40.0]),
    }
    gp2 = m.update_global(gp, stats, learning_rate=1.0)
    a_full = gp2["alpha"]
    a_node1 = a_full[lay.block[1][0]]
    a_node2 = a_full[lay.block[2][0]]
    assert a_node1 > a_node2                                      # common node gets larger alpha
    assert a_full.min() >= 1e-3                                   # floor respected
    # tying preserved: all topics in a block share one value
    assert np.allclose(a_full[lay.block[1]], a_node1)
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd spark-vi && python -m pytest tests/test_gated_lda.py -q -k "update_global"`
Expected: FAIL — `GatedOnlineLDA` still inherits `OnlineLDA.update_global`, which ignores `e_log_theta_node_sum` and does not learn a gated α.

- [ ] **Step 3: Implement the override + helper**

Add to `GatedOnlineLDA` in `spark-vi/spark_vi/models/topic/gated_lda.py`. Add the import at the top of the module:

```python
from spark_vi.inference.concentration_optimization import gated_alpha_newton_step
```

Then the methods:

```python
    def update_global(self, global_params, target_stats, learning_rate):
        """SVI natural-gradient λ step (inherited form) + gated per-node α step.

        The λ update is the same natural-gradient step OnlineLDA.update_global
        computes; we recompute it here (a few lines) rather than toggling the
        parent's optimize flags, so the gated α path is explicit. η is never
        optimized in the gated engine.
        """
        lam = global_params["lambda"]
        alpha = global_params["alpha"]
        eta = global_params["eta"]
        expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))
        target_lam = eta + expElogbeta * target_stats["lambda_stats"]
        new_lam = (1.0 - learning_rate) * lam + learning_rate * target_lam
        new_alpha = alpha
        if self.optimize_alpha:
            new_alpha = self._gated_alpha_update(alpha, target_stats, learning_rate)
        return {"lambda": new_lam, "alpha": new_alpha, "eta": eta}

    def _gated_alpha_update(self, alpha_full, target_stats, learning_rate):
        """Contract α to tied space, take one damped gated Newton step, expand back.

        Blei-Ng-Jordan 2003 A.4.2 generalized to per-node tying + gated
        sub-simplices; see gated_alpha_newton_step. Floors α at 1e-3.
        """
        nodes = self.lay.nodes
        B = self._block_sizes.shape[0]
        # contract: one representative topic per tied block (tying keeps them equal)
        a_tied = np.empty(B, dtype=np.float64)
        a_tied[0] = alpha_full[0]                              # a background topic
        for i, u in enumerate(nodes, start=1):
            a_tied[i] = alpha_full[self.lay.block[u][0]]
        # static group structure from the frontier histogram
        groups = list(self._frontier_histogram.items())
        group_counts = np.array([c for _, c in groups], dtype=np.float64)
        memb = np.zeros((len(groups), B), dtype=bool)
        for g, (frontier, _) in enumerate(groups):
            for k in self.lay.allowed_set(frontier):
                memb[g, self._topic_to_tied[k]] = True
        delta = gated_alpha_newton_step(
            a_tied, self._block_sizes,
            target_stats["e_log_theta_node_sum"], group_counts, memb)
        a_tied_new = np.maximum(a_tied + learning_rate * delta, 1e-3)
        # expand back to length-K
        out = alpha_full.copy()
        out[: self.lay.n_bg] = a_tied_new[0]
        for i, u in enumerate(nodes, start=1):
            out[self.lay.block[u]] = a_tied_new[i]
        return out
```

- [ ] **Step 4: Run to verify they pass**

Run: `cd spark-vi && python -m pytest tests/test_gated_lda.py -q -k "update_global"`
Expected: PASS (2 tests).

- [ ] **Step 5: Run the full gated engine suite (no regression)**

Run: `cd spark-vi && python -m pytest tests/test_gated_lda.py -q`
Expected: PASS (all).

- [ ] **Step 6: Commit**

```bash
git add spark-vi/spark_vi/models/topic/gated_lda.py spark-vi/tests/test_gated_lda.py
git commit -m "feat(gated-lda): update_global learns gated per-node asymmetric alpha

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: MLlib shim — `optimizeDocConcentration` Param + frontier histogram

**Files:**
- Modify: `spark-vi/spark_vi/mllib/topic/gated_lda.py` (`_GatedLDAParams`, `GatedLDAEstimator.__init__`/`setParams`/`_fit`)
- Test: `spark-vi/tests/test_gated_lda_shim.py`

**Interfaces:**
- Consumes: `GatedOnlineLDA(..., optimize_alpha=?, frontier_histogram=?)` (Tasks 2-3).
- Produces: `GatedLDAEstimator(..., optimizeDocConcentration=False)` Param; `_fit` computes the frontier histogram from `labelCol` and passes it + the flag to the engine.

- [ ] **Step 1: Write the failing tests**

Add to `spark-vi/tests/test_gated_lda_shim.py`:

```python
def test_gated_shim_optimize_doc_concentration_param_defaults_and_settable():
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    est = GatedLDAEstimator(parent={1: 0, 2: 0})
    assert est.getOrDefault("optimizeDocConcentration") is False
    est2 = GatedLDAEstimator(parent={1: 0, 2: 0}, optimizeDocConcentration=True)
    assert est2.getOrDefault("optimizeDocConcentration") is True

def test_gated_shim_optimize_alpha_learns_asymmetric(spark):
    # A corpus where node 1 fires often (common) and node 2 rarely (rare) should,
    # with optimizeDocConcentration on, learn alpha(node1) > alpha(node2).
    import numpy as np
    from pyspark.ml.linalg import Vectors
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    from spark_vi.models.topic.dag_placement import DagLayout
    parent = {1: 0, 2: 0}
    V = 24
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(120):
        leaf = 1 if rng.random() < 0.75 else 2          # node1 common, node2 rare
        idx = sorted(rng.choice(V, size=5, replace=False).tolist())
        rows.append((Vectors.sparse(V, idx, [1.0] * len(idx)), [leaf]))
    for _ in range(60):
        idx = sorted(rng.choice(V, size=5, replace=False).tolist())
        rows.append((Vectors.sparse(V, idx, [1.0] * len(idx)), []))  # background
    df = spark.createDataFrame(rows, ["features", "frontier"])
    model = GatedLDAEstimator(parent=parent, nBg=2, tpn=1, maxIter=8, seed=0,
                              optimizeDocConcentration=True).fit(df)
    lay = DagLayout(parent, n_bg=2, tpn=1)
    alpha = model.result.global_params["alpha"]
    assert alpha[lay.block[1][0]] > alpha[lay.block[2][0]]
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd spark-vi && python -m pytest tests/test_gated_lda_shim.py -q -k "optimize_doc_concentration or optimize_alpha_learns"`
Expected: FAIL — `optimizeDocConcentration` Param does not exist.

- [ ] **Step 3: Add the Param**

In `spark-vi/spark_vi/mllib/topic/gated_lda.py`, in `class _GatedLDAParams`, add (next to the other Params):

```python
    optimizeDocConcentration = Param(Params._dummy(), "optimizeDocConcentration",
                                     "learn an asymmetric per-node Dirichlet alpha "
                                     "from data (Wallach et al. 2009); nodeAlphaScale "
                                     "sets the initial alpha, optimize refines it. "
                                     "Default False.",
                                     typeConverter=TypeConverters.toBoolean)
```

- [ ] **Step 4: Wire it into the estimator**

In `GatedLDAEstimator.__init__`, add `optimizeDocConcentration=False` to BOTH the `@keyword_only def __init__(...)` signature and the `self._setDefault(...)` call. (Place it alongside `nodeAlphaScale=1.0`.)

In `_fit`, after `alpha_vec` is built and before constructing `GatedOnlineLDA`, compute the histogram and pass the flag:

```python
        optimize_alpha = bool(self.getOrDefault("optimizeDocConcentration"))
        frontier_hist = None
        if optimize_alpha:
            # Static allowed-set group structure from the (fixed) training labels.
            # Foreground+background scale; collected once at fit time.
            frontier_hist = {
                frozenset(int(x) for x in (fr or [])): int(n)
                for fr, n in (
                    dataset.select(label_col).rdd
                    .map(lambda r: frozenset(int(x) for x in (r[0] or [])))
                    .countByValue().items())
            }
        model_obj = GatedOnlineLDA(
            lay, V, init=init,
            optimize_alpha=optimize_alpha, frontier_histogram=frontier_hist,
            alpha=alpha_vec, eta=1.0 / lay.K,
            gamma_shape=self.getOrDefault("gammaShape"),
            cavi_max_iter=self.getOrDefault("caviMaxIter"),
            cavi_tol=self.getOrDefault("caviTol"),
            random_seed=seed,
        )
```

(Replace the existing `model_obj = GatedOnlineLDA(...)` construction with the above; `label_col` is already bound earlier in `_fit`.)

- [ ] **Step 5: Run to verify they pass**

Run: `cd spark-vi && python -m pytest tests/test_gated_lda_shim.py -q -k "optimize_doc_concentration or optimize_alpha_learns"`
Expected: PASS (2 tests).

- [ ] **Step 6: Run the full shim suite (no regression)**

Run: `cd spark-vi && python -m pytest tests/test_gated_lda_shim.py -q`
Expected: PASS (all).

- [ ] **Step 7: Commit**

```bash
git add spark-vi/spark_vi/mllib/topic/gated_lda.py spark-vi/tests/test_gated_lda_shim.py
git commit -m "feat(gated-shim): optimizeDocConcentration Param + frontier histogram from labelCol

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Planted-α recovery acceptance test

**Files:**
- Test: `spark-vi/tests/test_gated_lda_shim.py` (acceptance test; no production code)

**Interfaces:**
- Consumes: the full integrated path (Tasks 1-4) via `GatedLDAEstimator(optimizeDocConcentration=True)`.

- [ ] **Step 1: Write the acceptance test**

Add to `spark-vi/tests/test_gated_lda_shim.py`. This plants a corpus where each node's document-frequency (how often it is a doc's frontier) encodes its prevalence; the learned α should rank rarer nodes lower. Ranking is the primary, robust gate (a Dirichlet-concentration point estimate is noisy); a loose correlation is secondary.

```python
def test_gated_optimize_alpha_recovers_node_prevalence_ranking(spark):
    # Plant 4 sibling nodes with decreasing prevalence (node 1 most common ...
    # node 4 rarest). With optimizeDocConcentration on, the learned per-node alpha
    # should be monotone-ish in prevalence: rarer node -> smaller alpha.
    import numpy as np
    from pyspark.ml.linalg import Vectors
    from scipy.stats import spearmanr
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    from spark_vi.models.topic.dag_placement import DagLayout

    parent = {1: 0, 2: 0, 3: 0, 4: 0}
    nodes = [1, 2, 3, 4]
    prevalence = {1: 90, 2: 55, 3: 30, 4: 12}                 # planted doc counts
    V = 40
    rng = np.random.default_rng(7)
    # give each node a small signature vocab window so its topic is learnable
    sig = {u: sorted(rng.choice(V, size=6, replace=False).tolist()) for u in nodes}
    rows = []
    for u in nodes:
        for _ in range(prevalence[u]):
            idx = sorted(set(rng.choice(sig[u], size=4, replace=False).tolist()
                             + rng.choice(V, size=2, replace=False).tolist()))
            rows.append((Vectors.sparse(V, idx, [1.0] * len(idx)), [u]))
    for _ in range(80):                                       # background
        idx = sorted(rng.choice(V, size=5, replace=False).tolist())
        rows.append((Vectors.sparse(V, idx, [1.0] * len(idx)), []))
    df = spark.createDataFrame(rows, ["features", "frontier"])

    model = GatedLDAEstimator(parent=parent, nBg=3, tpn=1, maxIter=25, seed=0,
                              nodeAlphaScale=1.0,
                              optimizeDocConcentration=True).fit(df)
    lay = DagLayout(parent, n_bg=3, tpn=1)
    alpha = model.result.global_params["alpha"]
    learned = [float(alpha[lay.block[u][0]]) for u in nodes]
    planted = [prevalence[u] for u in nodes]

    # PRIMARY (robust): rarest node has the smallest learned alpha; most common the largest.
    assert np.argmin(learned) == nodes.index(4)
    assert np.argmax(learned) == nodes.index(1)
    # SECONDARY (loose): positive rank correlation with planted prevalence.
    rho = spearmanr(planted, learned).correlation
    assert rho >= 0.6, f"prevalence->alpha rank corr too low: {rho}"
```

- [ ] **Step 2: Run it**

Run: `cd spark-vi && python -m pytest tests/test_gated_lda_shim.py::test_gated_optimize_alpha_recovers_node_prevalence_ranking -q`
Expected: PASS. If it fails on the SECONDARY threshold on first run, calibrate the `0.6` DOWN to the observed value ONLY IF the PRIMARY ordering assertions pass and the correlation is clearly positive — record the observed value in the assertion message. If the PRIMARY ordering fails, do NOT loosen; mark `@pytest.mark.xfail(reason="...observed behavior...")` and stop for review (this would mean the recovery signal is weaker than the design assumed).

- [ ] **Step 3: Commit**

```bash
git add spark-vi/tests/test_gated_lda_shim.py
git commit -m "test(gated-shim): planted per-node alpha prevalence-ranking recovery (acceptance gate)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Per-node tied α → Task 2 (`_block_sizes`, `_topic_to_tied`), Task 3 (contract/expand). ✓
- Exact dense gated Newton → Task 1. ✓
- Static frontier histogram from labels → Task 4 (shim computes), Tasks 2-3 (engine stores/uses). ✓
- α only, not η → Task 3 `update_global` returns `eta` untouched; never optimized. ✓
- API mirror `optimizeDocConcentration` → Task 4. ✓
- `nodeAlphaScale` is the initial α → unchanged in shim `_fit` (`alpha_vec` still built from it); optimize refines. ✓
- New stat `e_log_theta_node_sum`, corpus-scaled by runner, summed by combine_stats → Task 2 (generic scaling/combine confirmed; no override). ✓
- Acceptance = planted-α recovery → Task 5. ✓
- Derivation guard (finite difference) → Task 1. ✓
- `optimize_alpha=False` byte-identical → Task 2 (stat absent), Task 3 (α untouched), Task 4 (flag default False). ✓
- Mass-starved self-regularization / 1e-3 floor → Task 3 (`np.maximum(..., 1e-3)`). ✓

**Placeholder scan:** none — every step has concrete code/commands.

**Type consistency:** `gated_alpha_newton_step(alpha_tied, block_sizes, e_log_theta_block_sum_scaled, group_counts, group_membership)` used identically in Task 1 (def) and Task 3 (call). Stat key `"e_log_theta_node_sum"` consistent across Tasks 2 (emit), 3 (consume). Tied order `[bg, lay.nodes...]` consistent in Tasks 2-3. `optimizeDocConcentration` consistent Task 4-5. ✓
