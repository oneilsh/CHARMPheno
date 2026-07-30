# Hybrid Domain-Reliability Readout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a refit-free, nested-cross-validated rare6 readout that measures the PR headroom from disease-specific condition/drug/observation weights and compares that supervised ceiling with three model-derived reliability fallbacks.

**Architecture:** Add small domain-agnostic score-combination and reliability primitives to `spark_vi`, then keep all labels, folds, policy selection, and rare6 reporting in a Spark-free analysis module. A thin CLI loads the existing multidomain run artifacts and emits deterministic JSON/Markdown reports for exp 0072 first and corrected exp 0071 second.

**Tech Stack:** Python 3.10+, NumPy >=1.24, SciPy >=1.10, scipy.sparse, pytest >=7, existing `spark_vi.models.topic.dag_placement` metrics and multidomain artifact loaders.

## Global Constraints

- Preserve pure-Python flat package layouts; add no dependencies and no generated code.
- `spark-vi` must not import `charmpheno`, OMOP, clinical names, or analysis modules.
- Use α→∞ LR scores and the existing tie-collapsing `_average_precision`.
- Combine weighted domain matrices at each DAG node before max-over-subtree scoring.
- Every supervised reported score must be out-of-fold; backgrounds, scales, policies, and weights are fit from training rows only.
- Use nonnegative simplex weights summing to one; do not allow a negative domain coefficient.
- Treat rare6 as a development benchmark; do not describe its nested-CV result as external validation.
- Do not refit initially. Refit only if required artifact/fold invariants are absent or a later seed-stability experiment is approved.
- Do not add fitting-time ω, measurement, MONDO ingestion, or a hierarchical production estimator in this plan.
- Explain non-obvious math in docstrings and cite Li et al. 2020, Davis & Goadrich 2006, and the approved design where relevant.
- Add an insight only after empirical results exist; never pre-write a result.

---

## File Structure

### Generic engine

- Modify `spark-vi/spark_vi/models/topic/dag_placement.py`
  - public empirical LR-background helper;
  - public fixed score-scale helper while retaining `_domain_scale` compatibility;
  - fixed per-domain scale/weight combiner;
  - fitted-background and model-derived reliability components.
- Modify `spark-vi/tests/test_lr_multidomain.py`
  - background, scaling, weighting, shape-validation, and identity regressions.
- Create `spark-vi/tests/test_domain_reliability.py`
  - exact distinctiveness, ownership, viability, and candidate-weight behavior.

### Spark-free evaluation

- Create `analysis/cloud/multidomain_weighting.py`
  - stratified folds, simplex grid, fold-local transformations, discrete-policy selection,
    continuous weight selection, repeated nested evaluation, and result schema.
- Create `analysis/cloud/tests/test_multidomain_weighting.py`
  - planted helpful/noise/opposite-domain cases, leakage guards, node-before-subtree
    ordering, deterministic selection, and out-of-fold coverage.

### Artifact CLI and operations

- Create `analysis/cloud/multidomain_weighting_readout.py`
  - artifact loading, rare6 anchor resolution, evaluation orchestration, JSON/Markdown
    serialization, and command-line interface.
- Create `analysis/cloud/tests/test_multidomain_weighting_readout.py`
  - parser defaults, artifact validation, report schema, and model-derived anchor weights.
- Modify `analysis/cloud/Makefile`
  - `multidomain-weighting-readout` target and help text.

### Results and project record

- Modify `docs/REVIEW_LOG.md` after code verification.
- Create `docs/insights/0075-hybrid-domain-reliability-readout.md` only after exp 0072 and exp 0071 results
  have been run and interpreted.

---

### Task 1: Fixed Background, Scale, and Weight Primitives

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/dag_placement.py:348-594`
- Modify: `spark-vi/tests/test_lr_multidomain.py`

**Interfaces:**
- Consumes: dense NumPy or SciPy-sparse BOW matrices and existing per-domain score matrices.
- Produces:
  - `lr_background(bow, *, epsilon=1e-9) -> np.ndarray`
  - `domain_score_scale(scores) -> float`
  - compatibility wrapper `_domain_scale(scores) -> float`
  - `combine_domain_score_matrices(matrices, *, weights=None, scales=None) -> np.ndarray`
- `weights` and `scales` are mappings keyed exactly like `matrices`; omitted entries default
  to `1.0`.

- [ ] **Step 1: Write failing background tests**

Add tests proving that background frequencies are normalized, sparse/dense equivalent,
and reusable across scoring cohorts:

```python
def test_lr_background_sparse_dense_and_fixed_scoring_equivalence():
    from scipy import sparse as sp
    from spark_vi.models.topic.dag_placement import (
        lr_background, lr_placement_scores)

    lay, lam, bows = _tiny()
    train = np.asarray(bows[0][:3], dtype=float)
    test = np.asarray(bows[0][3:], dtype=float)
    bg = lr_background(train)

    assert np.isclose(bg.sum(), 1.0)
    assert np.allclose(bg, lr_background(sp.csr_matrix(train)))
    assert np.allclose(
        lr_placement_scores(test, lam[0], lay, alpha=float("inf"), background=bg),
        lr_placement_scores(
            test * 7.0, lam[0], lay, alpha=float("inf"), background=bg),
    )
```

The last assertion pins that a caller-supplied background is frozen; multiplying test
counts changes score magnitude but not the per-document/node ordering. Compare normalized
rows or AUC rather than raw equality if the fixture has nonzero counts:

```python
raw = lr_placement_scores(test, lam[0], lay, alpha=float("inf"), background=bg)
scaled = lr_placement_scores(test * 7.0, lam[0], lay, alpha=float("inf"), background=bg)
assert np.allclose(scaled, raw * 7.0)
```

- [ ] **Step 2: Run the background tests and verify failure**

Run:

```bash
uv run pytest -q spark-vi/tests/test_lr_multidomain.py -k lr_background
```

Expected: FAIL because `lr_background` does not exist.

- [ ] **Step 3: Implement the public background helper**

Implement:

```python
def lr_background(bow, *, epsilon=1e-9):
    """Empirical vocabulary base rate from a fixed reference BOW corpus.

    The caller chooses the reference rows. Reusing the returned vector when
    scoring another cohort prevents the score transformation from depending on
    the composition of that scoring batch.
    """
    col = np.asarray(bow.sum(axis=0)).ravel().astype(float)
    bg = col / max(float(col.sum()), 1.0)
    return np.maximum(bg, epsilon)
```

Update `_lr_base_rate` to delegate to `lr_background` when `background is None`. Preserve
the existing supplied-background path exactly.

- [ ] **Step 4: Run focused background tests**

Run:

```bash
uv run pytest -q spark-vi/tests/test_lr_multidomain.py -k "background or multidomain_score"
```

Expected: PASS.

- [ ] **Step 5: Write failing scale and combination tests**

Add:

```python
def test_combine_domain_score_matrices_applies_fixed_scales_and_weights():
    from spark_vi.models.topic.dag_placement import combine_domain_score_matrices

    mats = {
        0: np.array([[2.0, 4.0], [6.0, 8.0]]),
        1: np.array([[10.0, 20.0], [30.0, 40.0]]),
    }
    got = combine_domain_score_matrices(
        mats, weights={0: 0.75, 1: 0.25}, scales={0: 2.0, 1: 10.0})
    expect = 0.75 * mats[0] / 2.0 + 0.25 * mats[1] / 10.0
    assert np.allclose(got, expect)


def test_combine_domain_score_matrices_identity_and_validation():
    import pytest
    from spark_vi.models.topic.dag_placement import combine_domain_score_matrices

    mats = {0: np.ones((3, 2)), 1: np.full((3, 2), 2.0)}
    assert np.array_equal(combine_domain_score_matrices(mats), mats[0] + mats[1])
    with pytest.raises(ValueError, match="same shape"):
        combine_domain_score_matrices({0: mats[0], 1: np.ones((2, 2))})
    with pytest.raises(ValueError, match="scale"):
        combine_domain_score_matrices(mats, scales={1: 0.0})
    with pytest.raises(ValueError, match="weight"):
        combine_domain_score_matrices(mats, weights={1: -0.1})
    with pytest.raises(ValueError, match="at least one"):
        combine_domain_score_matrices(mats, weights={0: 0.0, 1: 0.0})
```

Also update the existing `_domain_scale` tests to assert:

```python
from spark_vi.models.topic.dag_placement import domain_score_scale, _domain_scale
assert domain_score_scale(x) == _domain_scale(x)
```

- [ ] **Step 6: Run combination tests and verify failure**

Run:

```bash
uv run pytest -q spark-vi/tests/test_lr_multidomain.py -k "combine_domain or domain_scale"
```

Expected: FAIL because the new public functions do not exist.

- [ ] **Step 7: Implement scale and combination primitives**

Rename the current `_domain_scale` implementation to `domain_score_scale`, retain:

```python
def _domain_scale(scores):
    """Backward-compatible private spelling; use domain_score_scale in new code."""
    return domain_score_scale(scores)
```

Implement `combine_domain_score_matrices` with:

- nonempty matrices;
- identical shapes;
- finite nonnegative weights;
- at least one positive effective weight;
- finite positive scales; and
- insertion-order-independent summation by iterating `sorted(matrices)`.

Do not require weights to sum to one in the generic engine helper; the analysis-layer
simplex generator owns that policy.

- [ ] **Step 8: Run the complete multidomain LR test file**

Run:

```bash
uv run pytest -q spark-vi/tests/test_lr_multidomain.py
```

Expected: PASS.

- [ ] **Step 9: Commit Task 1**

```bash
git add spark-vi/spark_vi/models/topic/dag_placement.py \
        spark-vi/tests/test_lr_multidomain.py
git commit -m "feat(readout): add fixed domain score combination primitives"
```

---

### Task 2: Model-Derived Reliability Components

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/dag_placement.py`
- Create: `spark-vi/tests/test_domain_reliability.py`

**Interfaces:**
- Consumes: `lam_dict: dict[int, np.ndarray]` and `DagLayout`.
- Produces:

```python
@dataclass(frozen=True)
class DomainReliability:
    domain_keys: tuple[int, ...]
    distinctiveness: np.ndarray  # [n_nodes, n_domains]
    ownership: np.ndarray        # [n_nodes, n_domains]
    viability: np.ndarray        # [n_nodes, n_domains]

    def weights(self, candidate: str) -> np.ndarray:
        """[n_nodes, n_domains], rows nonnegative and summing to one."""


def domain_reliability(
    lam_dict,
    lay,
    *,
    epsilon=1e-12,
    viability_tol=1e-6,
) -> DomainReliability
```

Candidate names are exactly `"distinctiveness"`, `"ownership"`, and `"product"`.

- [ ] **Step 1: Write failing distinctiveness tests**

Create a two-domain planted λ fixture where node 1 equals background in domain 0 and has
a concentrated marker in domain 1:

```python
def test_distinctiveness_is_zero_for_background_and_positive_for_marker():
    from spark_vi.models.topic.dag_placement import DagLayout, domain_reliability

    lay = DagLayout({1: [0]}, n_bg=1, tpn=1)
    bg = np.array([4.0, 4.0, 4.0, 4.0])
    lam = {
        0: np.vstack([bg, bg]),
        1: np.vstack([bg, np.array([16.0, 1.0, 1.0, 1.0])]),
    }
    rel = domain_reliability(lam, lay)
    assert rel.domain_keys == (0, 1)
    assert abs(rel.distinctiveness[0, 0]) < 1e-12
    assert rel.distinctiveness[0, 1] > 0.0
```

- [ ] **Step 2: Write failing ownership and viability tests**

Add fixtures proving:

- a node-only marker has higher ownership than a code equally likely in background;
- a constant topic row is nonviable for that domain;
- one live and one constant topic under `tpn=2` yields viability `0.5`, preventing the
  former any-topic-alive blind spot.

```python
def test_viability_is_topic_granular_within_each_domain():
    from spark_vi.models.topic.dag_placement import DagLayout, domain_reliability

    lay = DagLayout({1: [0]}, n_bg=1, tpn=2)
    lam = {0: np.array([
        [2.0, 2.0, 2.0],   # background
        [8.0, 1.0, 1.0],   # live node topic
        [3.0, 3.0, 3.0],   # starved/constant node topic
    ])}
    rel = domain_reliability(lam, lay)
    assert rel.viability[0, 0] == 0.5
```

- [ ] **Step 3: Run reliability tests and verify failure**

Run:

```bash
uv run pytest -q spark-vi/tests/test_domain_reliability.py
```

Expected: FAIL because `DomainReliability` and `domain_reliability` do not exist.

- [ ] **Step 4: Implement fitted-background and Jensen–Shannon helpers**

In `dag_placement.py`, add private helpers:

```python
def _normalize_prob(x, epsilon):
    x = np.maximum(np.asarray(x, dtype=float), 0.0)
    return np.maximum(x / max(float(x.sum()), epsilon), epsilon)


def _js_divergence(p, q, epsilon):
    p = _normalize_prob(p, epsilon)
    q = _normalize_prob(q, epsilon)
    mid = 0.5 * (p + q)
    return 0.5 * float(np.sum(p * np.log(p / mid))) + \
        0.5 * float(np.sum(q * np.log(q / mid)))
```

For each domain:

- fitted background = normalized sum of `lam[0:lay.n_bg]`;
- node distribution = normalized sum of `lam[lay.block[u]]`;
- distinctiveness = Jensen–Shannon divergence between those distributions.

Explain in the docstring that bounded JS is a model-structure proxy, not task utility.

- [ ] **Step 5: Implement ownership and viability**

Ownership for `(u,m)`:

```text
sum_w P_node(w) * routing_row(u,w)
```

where routing rows come from the existing `_routing_rows`.

A topic is live within domain `m` iff:

```text
max(row) - min(row) > viability_tol * max(max(row), epsilon)
```

Viability is the fraction of topics in `lay.block[u]` live in that domain.

- [ ] **Step 6: Implement candidate normalization**

`DomainReliability.weights(candidate)` chooses:

- `distinctiveness`;
- `ownership`; or
- `distinctiveness * ownership * viability`.

Normalize each node row across domains. If a row has zero/nonfinite total, return uniform
weights for that row rather than NaN or an arbitrary single-domain winner. Reject unknown
candidate names with `ValueError`.

- [ ] **Step 7: Run reliability and existing placement tests**

Run:

```bash
uv run pytest -q spark-vi/tests/test_domain_reliability.py \
  spark-vi/tests/test_dag_placement.py \
  spark-vi/tests/test_lr_multidomain.py
```

Expected: PASS.

- [ ] **Step 8: Commit Task 2**

```bash
git add spark-vi/spark_vi/models/topic/dag_placement.py \
        spark-vi/tests/test_domain_reliability.py
git commit -m "feat(readout): derive label-free domain reliability from lambda"
```

---

### Task 3: Nested Cross-Validated Weighting Engine

**Files:**
- Create: `analysis/cloud/multidomain_weighting.py`
- Create: `analysis/cloud/tests/test_multidomain_weighting.py`

**Interfaces:**
- Consumes: per-domain BOW/λ, `DagLayout`, frontiers, one anchor, parent mapping.
- Produces:

```python
def simplex_grid(n_domains: int, step: float) -> np.ndarray
def discrete_policies(domain_keys) -> tuple[tuple[str, tuple[int, ...], str], ...]
def stratified_folds(y, *, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]
def anchor_truth(frontiers, subtree: set[int]) -> np.ndarray
def subtree_columns(lay, subtree: set[int]) -> np.ndarray
def max_subtree_score(score_matrix, columns) -> np.ndarray

def evaluate_anchor_nested(
    bows,
    lam_dict,
    lay,
    frontiers,
    *,
    anchor: int,
    parent_int,
    outer_folds=5,
    inner_folds=4,
    repeats=5,
    grid_step=0.05,
    seed=0,
) -> dict
```

The result dict contains JSON-safe values:

```python
{
    "anchor": int,
    "n_docs": int,
    "n_positive": int,
    "prevalence": float,
    "repeats": [
        {
            "repeat": int,
            "strategies": {
                "fixed:condition_drug": {
                    "ap": float,
                    "precision_at_recall": {
                        "0.1": float, "0.25": float, "0.5": float, "0.8": float,
                    },
                },
                "discrete": {
                    "ap": float,
                    "precision_at_recall": {
                        "0.1": float, "0.25": float, "0.5": float, "0.8": float,
                    },
                },
                "continuous": {
                    "ap": float,
                    "precision_at_recall": {
                        "0.1": float, "0.25": float, "0.5": float, "0.8": float,
                    },
                },
            },
            "folds": [
                {
                    "fold": int,
                    "test_rows": list[int],
                    "discrete_policy": str,
                    "continuous_weights": list[float],
                }
            ],
        }
    ],
}
```

- [ ] **Step 1: Write failing simplex and fold tests**

```python
def test_simplex_grid_is_deterministic_nonnegative_and_sums_to_one():
    from multidomain_weighting import simplex_grid

    grid = simplex_grid(3, 0.5)
    assert np.array_equal(grid, np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.5, 0.5],
        [0.0, 1.0, 0.0],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
        [1.0, 0.0, 0.0],
    ]))
    assert np.all(grid >= 0.0)
    assert np.allclose(grid.sum(axis=1), 1.0)


def test_stratified_folds_cover_each_row_once_and_preserve_classes():
    from multidomain_weighting import stratified_folds

    y = np.array([1] * 10 + [0] * 30)
    folds = stratified_folds(y, n_splits=5, seed=7)
    held_out = np.concatenate([te for _, te in folds])
    assert np.array_equal(np.sort(held_out), np.arange(len(y)))
    assert all(y[te].sum() == 2 for _, te in folds)
    again = stratified_folds(y, n_splits=5, seed=7)
    for (tr1, te1), (tr2, te2) in zip(folds, again):
        assert np.array_equal(tr1, tr2)
        assert np.array_equal(te1, te2)
```

Compare arrays explicitly rather than Python list equality where necessary.

- [ ] **Step 2: Run fold tests and verify failure**

Run:

```bash
uv run pytest -q analysis/cloud/tests/test_multidomain_weighting.py \
  -k "simplex or stratified"
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement deterministic grids and folds**

Implement simplex enumeration using integer units:

```python
units = round(1.0 / step)
```

Require `step > 0` and `step * units == 1` within `1e-12`. Enumerate integer
compositions in lexicographic order; divide by `units`.

Implement stratification with `np.random.default_rng(seed)`: shuffle positive and negative
indices separately, split each with `np.array_split`, then combine/sort each test fold.
Reject one-class labels, `n_splits < 2`, or a class count smaller than `n_splits`.

- [ ] **Step 4: Write failing node-before-subtree and leakage tests**

Construct two nodes in one disease subtree where domain 0 favors node A and domain 1 favors
node B. Assert the correct score is:

```python
expected = np.max(w0 * mats[0] + w1 * mats[1], axis=1)
```

and not:

```python
wrong = w0 * np.max(mats[0], axis=1) + w1 * np.max(mats[1], axis=1)
```

Add a fold-local background test: alter only outer-test BOW rows, recompute the fold, and
assert training-derived backgrounds, scales, and selected weights are unchanged.

- [ ] **Step 5: Implement fold-local score construction**

Add private helpers with explicit inputs:

```python
def _backgrounds_from_rows(bows, rows):
    return {m: lr_background(bows[m][rows]) for m in sorted(bows)}


def _raw_scores_for_rows(bows, lam_dict, lay, rows, backgrounds, *, length=False):
    return lr_domain_score_matrices(
        {m: bows[m][rows] for m in sorted(bows)},
        lam_dict,
        lay,
        alpha=float("inf"),
        backgrounds=backgrounds,
        normalize="length" if length else None,
    )
```

For one domain matrix and subtree columns, scale from the training-row
`max_subtree_score`. Use `domain_score_scale`; apply the same scalar to every node column
when scoring validation/test rows.

- [ ] **Step 6: Write failing planted selection tests**

Create three-domain synthetic score fixtures independent of λ/BOW plumbing:

1. condition is perfectly informative, observation is high-variance noise;
2. drug adds independent true-positive signal;
3. disease A is condition-led while disease B is drug-led.

Expose a pure selector:

```python
def select_simplex_weights(train_matrices, y, columns, *, grid) -> np.ndarray
```

Tests must show:

- noise receives zero or the minimum grid weight;
- helpful drug receives positive weight and improves AP;
- opposite diseases select different weights.

- [ ] **Step 7: Implement continuous selection**

For each grid row:

1. combine domain matrices with `combine_domain_score_matrices`;
2. max over subtree columns;
3. compute `_average_precision`.

Select highest AP. For ties within `1e-12`, choose:

1. smallest squared distance from uniform weights, then
2. lexicographically smallest weight tuple.

This makes ties deterministic and mildly favors the least concentrated rule rather than
an arbitrary domain.

- [ ] **Step 8: Write failing nested out-of-fold test**

On a 60-row planted BOW/λ fixture with 15 positives:

- every row appears in exactly one outer test fold per repeat;
- no fold's `test_rows` intersect its training rows;
- changing one test label cannot change that fold's selected policy/weights;
- pooled OOF score length equals `n_docs`;
- repeated evaluation with the same seed is exactly equal.

- [ ] **Step 9: Implement discrete policy selection**

Implement `discrete_policies(domain_keys)` to return a deterministic, named tuple covering:

- every nonempty domain subset generated from the available domain keys; and
- `none`, `std`, `length`, `length+std`.

For `std` and `length+std`, use scale estimated from inner-training rows and applied to
inner-validation rows. Never call the existing transductive `normalize="std"` inside
nested evaluation.

Select the policy by pooled inner out-of-fold AP. On ties within `1e-12`, use tuple order
and record the chosen policy name.

- [ ] **Step 10: Implement nested continuous selection**

For each outer fold:

1. create inner stratified folds from outer-training rows;
2. compute inner-training backgrounds/scales;
3. create pooled inner OOF scores for every weight candidate;
4. select the best weights with the deterministic tie policy;
5. recompute background/scales from all outer-training rows;
6. score outer-test rows with frozen transformations and weights.

Fixed policies, the discrete selector, and the continuous selector must share the exact
outer folds.

- [ ] **Step 11: Add operational metrics**

Use existing:

```python
_average_precision(scores, y)
_precision_at_recall(scores, y, recalls=(0.10, 0.25, 0.50, 0.80))
```

Convert all NumPy scalars/arrays to Python JSON-safe numbers/lists at the result boundary.

- [ ] **Step 12: Run the analysis weighting tests**

Run:

```bash
uv run pytest -q analysis/cloud/tests/test_multidomain_weighting.py
```

Expected: PASS.

- [ ] **Step 13: Put prevented bugs back**

Temporarily introduce each bug and confirm the named test fails:

- estimate background from all rows;
- take per-domain subtree maxima before combining;
- select weights on outer-test AP.

Restore the correct code and rerun:

```bash
uv run pytest -q analysis/cloud/tests/test_multidomain_weighting.py
```

Expected: PASS after restoration.

- [ ] **Step 14: Commit Task 3**

```bash
git add analysis/cloud/multidomain_weighting.py \
        analysis/cloud/tests/test_multidomain_weighting.py
git commit -m "feat(case-finding): add nested domain-weight evaluation"
```

---

### Task 4: Model-Derived Candidates in the Evaluation

**Files:**
- Modify: `analysis/cloud/multidomain_weighting.py`
- Modify: `analysis/cloud/tests/test_multidomain_weighting.py`

**Interfaces:**
- Consumes: `DomainReliability.weights(candidate)` and the disease anchor's row in
  `lay.nodes`.
- Adds strategy names:
  - `model:distinctiveness`
  - `model:ownership`
  - `model:product`
- Produces:

```python
def evaluate_model_candidate(
    matrices,
    y,
    columns,
    *,
    weights,
    scales,
) -> dict:
    """JSON-safe weights, AP, precision-at-recall, and patient scores."""
```

- [ ] **Step 1: Write failing label-independence test**

Call the model-candidate evaluator twice with identical λ/BOW but inverted `y`. Assert its
domain weights are identical while AP changes:

```python
def test_model_derived_weights_do_not_consume_case_labels():
    matrices = {
        0: np.array([[4.0], [3.0], [2.0], [1.0]]),
        1: np.zeros((4, 1)),
    }
    weights = np.array([1.0, 0.0])
    scales = {0: 1.0, 1: 1.0}
    y = np.array([1, 1, 0, 0])
    got1 = evaluate_model_candidate(
        matrices, y, np.array([0]), weights=weights, scales=scales)
    got2 = evaluate_model_candidate(
        matrices, 1 - y, np.array([0]), weights=weights, scales=scales)
    assert got1["weights"] == got2["weights"] == [1.0, 0.0]
    assert got1["ap"] != got2["ap"]
```

- [ ] **Step 2: Write failing anchor-weight semantics test**

For rare6, each disease anchor is a scoreable node. Assert that candidate weights come
from that anchor's `DomainReliability` row and are then applied uniformly to all nodes in
the anchor's descendant subtree before maximization.

This is the first-increment convention. Do not average or maximize reliability over
descendants; that would introduce another unvalidated aggregation rule.

- [ ] **Step 3: Run model-candidate tests and verify failure**

Run:

```bash
uv run pytest -q analysis/cloud/tests/test_multidomain_weighting.py -k model_derived
```

Expected: FAIL because the strategies are not wired.

- [ ] **Step 4: Implement model-candidate evaluation**

Compute `domain_reliability(lam_dict, lay)` once per run. For each anchor, locate
`lay.nodes.index(anchor)`, take that row from each candidate matrix, and apply it using the
same fold-local backgrounds and scales as the supervised strategies.

The weights never change across folds; transformations still do, because honest
out-of-fold scoring requires fold-local reference distributions.

- [ ] **Step 5: Add agreement diagnostics**

For each repeat/disease report:

- candidate weight vector;
- Spearman rank correlation between candidate and continuous-ceiling patient scores;
- Jaccard overlap of the top `max(n_positive, ceil(0.01*n_docs))` patient rows; and
- whether candidate and median supervised weights give the same domain ordering.

Use `scipy.stats.spearmanr`. If either score vector is constant, serialize correlation as
`None`.

- [ ] **Step 6: Run all weighting-engine tests**

Run:

```bash
uv run pytest -q analysis/cloud/tests/test_multidomain_weighting.py \
  spark-vi/tests/test_domain_reliability.py
```

Expected: PASS.

- [ ] **Step 7: Commit Task 4**

```bash
git add analysis/cloud/multidomain_weighting.py \
        analysis/cloud/tests/test_multidomain_weighting.py
git commit -m "feat(case-finding): compare model-derived domain weights"
```

---

### Task 5: Artifact CLI and Reports

**Files:**
- Create: `analysis/cloud/multidomain_weighting_readout.py`
- Create: `analysis/cloud/tests/test_multidomain_weighting_readout.py`
- Modify: `analysis/cloud/Makefile:1-126,418-440`

**Interfaces:**
- Consumes existing helpers from `multidomain_lr_readout`:
  - `load_lambda_dict`
  - `load_test_set`
  - `scoreable_targets`
  - `subtree_nodes`
- CLI:

```text
--run-dir PATH                 required
--outer-folds INT              default 5
--inner-folds INT              default 4
--repeats INT                  default 5
--grid-step FLOAT              default 0.05
--seed INT                     default 0
--output-prefix PATH           default <run-dir>/multidomain_weighting
```

- Produces:
  - `<prefix>.json`
  - `<prefix>.md`

- [ ] **Step 1: Write failing parser tests**

```python
def test_parser_defaults_are_the_preregistered_design():
    from multidomain_weighting_readout import build_parser

    args = build_parser().parse_args(["--run-dir", "/runs/0072"])
    assert args.outer_folds == 5
    assert args.inner_folds == 4
    assert args.repeats == 5
    assert args.grid_step == 0.05
    assert args.seed == 0
```

Add rejection tests for folds below 2, nonpositive repeats, and invalid grid steps.

- [ ] **Step 2: Run parser tests and verify failure**

Run:

```bash
uv run pytest -q analysis/cloud/tests/test_multidomain_weighting_readout.py -k parser
```

Expected: FAIL because the CLI module does not exist.

- [ ] **Step 3: Implement artifact loading and validation**

Follow `multidomain_lr_readout.main` exactly for manifest/DAG reconstruction. Validate:

- λ keys are contiguous `0..n_domains-1`;
- every BOW has `n_docs` rows;
- λ vocabulary widths match BOW widths;
- frontiers length equals `n_docs`;
- rare6 anchors resolve to scoreable nodes; and
- every disease has at least `outer_folds` positives and negatives.

Abort with a clear `SystemExit` naming the violated invariant.

- [ ] **Step 4: Write failing report-schema tests**

Use a tiny temporary run artifact or monkeypatch the loader. Assert JSON includes:

```text
run_dir, disease, domains, cv_config, anchors, macro_summary
```

and each anchor includes fixed, discrete, continuous, and three model strategies.

Assert Markdown prints:

- prevalence;
- AP and lift;
- precision at 10/25/50/80% recall;
- continuous median weights;
- selected discrete policy frequencies; and
- model-versus-ceiling agreement.

- [ ] **Step 5: Implement JSON and Markdown writers**

Keep serialization pure:

```python
def render_markdown(result: dict) -> str
def write_reports(result: dict, output_prefix: Path) -> tuple[Path, Path]
```

Use atomic-enough local writes through `Path.write_text`; no Spark executors write to the
run directory.

- [ ] **Step 6: Implement `main`**

For each scoreable rare6 anchor:

1. call `evaluate_anchor_nested`;
2. collect per-anchor repeat summaries;
3. compute macro AP per repeat by averaging six disease AP values; and
4. write reports.

Print output paths and a compact macro/per-disease AP table to stdout.

- [ ] **Step 7: Add the Make target**

Add `.PHONY` entry and help text:

```text
multidomain-weighting-readout ID=N
    Nested-CV supervised ceiling + model-derived domain reliability (no refit)
```

Mirror the existing `multidomain-lr-readout` target's cluster overlay and run-dir
resolution, invoking the new script without Spark.

- [ ] **Step 8: Run CLI and existing readout tests**

Run:

```bash
uv run pytest -q \
  analysis/cloud/tests/test_multidomain_weighting.py \
  analysis/cloud/tests/test_multidomain_weighting_readout.py \
  analysis/cloud/tests/test_multidomain_lr_readout.py
```

Expected: PASS.

- [ ] **Step 9: Run a local synthetic CLI smoke**

Build the tiny artifact entirely under a temporary pytest directory or expose it as a test
fixture, then run:

```bash
uv run pytest -q \
  analysis/cloud/tests/test_multidomain_weighting_readout.py -k end_to_end
```

Expected: PASS and both JSON/Markdown files parse.

- [ ] **Step 10: Commit Task 5**

```bash
git add analysis/cloud/multidomain_weighting_readout.py \
        analysis/cloud/multidomain_weighting.py \
        analysis/cloud/tests/test_multidomain_weighting_readout.py \
        analysis/cloud/Makefile
git commit -m "feat(case-finding): add hybrid domain-weight readout CLI"
```

---

### Task 6: Verification, Real Readouts, and Research Record

> **Execution supersession (2026-07-30):** Steps 4 and 6 below preserve the
> original 0072/0071 commands as plan history, but those legacy sidecars lack the
> person-row attestation required for supervised row-level CV and must not be
> used for this readout. First fit exact-config clone 0073 and run
> `multidomain-weighting-readout ID=73`; after adjudication, fit exact-config
> clone 0074 of corrected 0071 and run the same readout with `ID=74`. This changes
> artifact identity, not the preregistered configurations or readout defaults.
> See [ADR 0038](../../decisions/0038-supervised-multidomain-readout-identity-attestation.md).

**Files:**
- Modify: `docs/REVIEW_LOG.md`
- Create after results: `docs/insights/0075-hybrid-domain-reliability-readout.md`
- Modify if implementation changes architecture: relevant `docs/architecture/*.md` or add
  an ADR rather than silently diverging.

**Interfaces:**
- Consumes the CLI and existing exp 0072/0071 run artifacts.
- Produces verified code plus empirical reports and an insight.

- [ ] **Step 1: Run focused local verification**

```bash
uv run pytest -q \
  spark-vi/tests/test_lr_multidomain.py \
  spark-vi/tests/test_domain_reliability.py \
  analysis/cloud/tests/test_multidomain_weighting.py \
  analysis/cloud/tests/test_multidomain_weighting_readout.py \
  analysis/cloud/tests/test_multidomain_lr_readout.py
```

Expected: PASS.

- [ ] **Step 2: Run the fast project suite**

```bash
make test
```

Expected: PASS with no new deselections beyond the repository's configured slow/cluster
tiers.

- [ ] **Step 3: Inspect repository hygiene**

```bash
git status --short
git diff --check
git diff --stat
```

Confirm no patient-level data, run artifacts, large binaries, secrets, or unrelated user
changes are staged.

- [ ] **Step 4: Run exp 0072 weighting readout**

On the configured Dataproc environment:

```bash
make -C analysis/cloud multidomain-weighting-readout ID=72
```

Expected: no topic-model fit; JSON/Markdown written beneath the existing exp 0072 run
directory. Record the exact branch SHA and CLI configuration in the report.

- [ ] **Step 5: Adjudicate exp 0072 before replication**

Check:

- continuous versus discrete-selector macro AP;
- per-disease AP and precision at 10/25/50% recall;
- fold/repeat weight stability;
- whether observation is smoothly downweighted or hard-dropped;
- model-derived candidate agreement; and
- ranking-head patient overlap.

Do not alter grid/fold/repeat defaults after seeing these results.

- [ ] **Step 6: Replicate unchanged on corrected exp 0071**

```bash
make -C analysis/cloud multidomain-weighting-readout ID=71
```

Expected: identical configuration, different fitted λ artifact.

- [ ] **Step 7: Write insight 0075 from observed results**

Use a result-neutral title inside the fixed insight file and include:

- settings and artifact SHAs;
- prevalence and per-disease positive counts;
- fixed/discrete/continuous/model-derived tables;
- macro and disease-specific PR changes;
- stability and cross-fit replication;
- whether the result is little/useful/strong headroom under the design bands; and
- the next branch of the predeclared decision table.

- [ ] **Step 8: Update the review log**

Add a dated entry at the top of `docs/REVIEW_LOG.md` naming:

- files and mathematical contracts reviewed;
- what shipped;
- pre-existing issues caught;
- tests run;
- empirical runs completed; and
- threads parked.

- [ ] **Step 9: Run the full non-cluster suite**

This suite is expected to take roughly 18 minutes and requires explicit approval before
execution under Policy A:

```bash
make test-all
```

Expected: PASS.

- [ ] **Step 10: Commit results and records**

```bash
git add docs/insights/0075-hybrid-domain-reliability-readout.md docs/REVIEW_LOG.md
git commit -m "docs(insights): record hybrid domain reliability result"
```

- [ ] **Step 11: Final review**

Verify:

```bash
git status --short --branch
git log --oneline -8
```

Confirm the working tree is clean, no clinical data are committed, and the implementation
matches the approved spec without fitting-time or measurement-domain scope creep.
