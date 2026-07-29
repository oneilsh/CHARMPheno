# Domain-Normalized Multi-Domain LR Combination — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the multi-domain LR readout normalize each domain's score before summing, so a low-signal high-volume domain (observation) can no longer drag the head of the ranking down by token volume alone.

**Architecture:** Purely readout-side; no re-fit and no model change. The library gains a `normalize` option on the multi-domain LR combination plus a new `lr_domain_score_matrices` accessor that returns the per-domain score matrices (so a caller scoring many domain subsets computes each domain once). The readout gains a `--normalize` flag governing its existing tables, and one new table comparing all four rules against the un-normalized drop-the-suspect-domain reference.

**Tech Stack:** Python, numpy, scipy.sparse, pytest. Spark is NOT involved in either file's changed code paths.

**Design spec:** `docs/superpowers/specs/2026-07-29-domain-normalized-lr-combination-design.md`

## Global Constraints

- `normalize=None` MUST reproduce today's numbers exactly — bit-identical, including float summation order. Iterate domains in the caller's `domains` order (Python dicts preserve insertion order) so the accumulation sequence is unchanged.
- The four rule names are exactly `none`, `std`, `length`, `length+std`. `none` is the CLI spelling; `None` is the library value.
- A per-domain scale is ONE scalar for the whole `[n_docs x n_nodes]` matrix — never per-column. This is what makes the transform affine and therefore order-preserving within a domain.
- Do NOT center (subtract a mean). A constant added to a domain's whole matrix cancels in both doc-ranking and max-over-nodes; scale is the minimal honest transform.
- A domain's normalization is computed from that domain ALONE and must not depend on which subset is being summed.
- Default rule stays `none`. This change adds a measurement; it does not change existing behavior.
- Never divide by zero or a non-finite scale: fall back to `1.0`.
- Existing tests must stay green. `analysis/cloud/tests/` imports readout modules by bare module name (e.g. `from multidomain_lr_readout import ...`); follow that existing convention.
- Cite the design spec path in new docstrings that encode a design decision (project convention: a method or default from a design/paper names its source).

---

### Task 1: Library — `normalize` on the multi-domain LR combination

**Files:**
- Modify: `spark-vi/spark_vi/models/topic/dag_placement.py:470-511` (`lr_placement_scores_multidomain`, `lr_auc_sweep_multidomain`)
- Test: `spark-vi/tests/test_lr_multidomain.py`

**Interfaces:**
- Consumes: existing `lr_placement_scores(bow, lam, lay, *, alpha, background=None, epsilon=1e-9, count_mode="raw", length_normalize=False)` at `dag_placement.py:357`.
- Produces, for Task 2:
  - `NORMALIZE_MODES = (None, "std", "length", "length+std")` (module-level tuple)
  - `_domain_scale(s) -> float`
  - `lr_domain_score_matrices(bows, lam_dict, lay, *, alpha, domains=None, backgrounds=None, epsilon=1e-9, count_mode="raw", normalize=None) -> dict[m, np.ndarray]`
  - `lr_placement_scores_multidomain(..., normalize=None)` and `lr_auc_sweep_multidomain(..., normalize=None)`

- [ ] **Step 1: Write the failing tests**

Append to `spark-vi/tests/test_lr_multidomain.py` (it already defines the `_tiny()` fixture at the top of the file — reuse it, do not duplicate it):

```python
def _tiny_scaled():
    """_tiny() but domain 1's BOW is a 10x copy of domain 0's (same V, same lam),
    so domain 1's raw score matrix is exactly 10x domain 0's: the LR score is
    linear in the counts and the background base rate is a normalized frequency,
    hence scale-invariant. Gives an EXACT target for scale equalization."""
    from spark_vi.models.topic.dag_placement import DagLayout
    lay = DagLayout({1: 0, 2: 0}, n_bg=2, tpn=1)
    rng = np.random.default_rng(0)
    lam0 = rng.random((lay.K, 6)) + 0.1
    bow0 = rng.integers(0, 3, size=(5, 6)).astype(float)
    return lay, {0: lam0, 1: lam0}, {0: bow0, 1: bow0 * 10.0}


def test_normalize_none_is_the_unchanged_per_domain_sum():
    # Regression: the default path must be bit-identical to the plain sum.
    from spark_vi.models.topic.dag_placement import (
        lr_placement_scores, lr_placement_scores_multidomain)
    lay, lam, bows = _tiny()
    s0 = lr_placement_scores(bows[0], lam[0], lay, alpha=1.0)
    s1 = lr_placement_scores(bows[1], lam[1], lay, alpha=1.0)
    s = lr_placement_scores_multidomain(bows, lam, lay, alpha=1.0, normalize=None)
    assert np.array_equal(s, s0 + s1)


def test_domain_score_matrices_sum_to_the_multidomain_score():
    from spark_vi.models.topic.dag_placement import (
        lr_domain_score_matrices, lr_placement_scores_multidomain)
    lay, lam, bows = _tiny()
    for rule in (None, "std", "length", "length+std"):
        mats = lr_domain_score_matrices(bows, lam, lay, alpha=1.0, normalize=rule)
        assert set(mats) == {0, 1}
        total = lr_placement_scores_multidomain(bows, lam, lay, alpha=1.0,
                                                normalize=rule)
        assert np.allclose(mats[0] + mats[1], total), rule


def test_std_normalization_equalizes_domain_scale_exactly():
    # Mechanism check. Domain 1 is a 10x copy of domain 0, so un-normalized its
    # scale is ~10x; after 'std' both matrices have unit std.
    from spark_vi.models.topic.dag_placement import (
        _domain_scale, lr_domain_score_matrices)
    lay, lam, bows = _tiny_scaled()
    raw = lr_domain_score_matrices(bows, lam, lay, alpha=1.0, normalize=None)
    assert np.isclose(_domain_scale(raw[1]) / _domain_scale(raw[0]), 10.0)
    norm = lr_domain_score_matrices(bows, lam, lay, alpha=1.0, normalize="std")
    assert np.isclose(np.std(norm[0]), 1.0)
    assert np.isclose(np.std(norm[1]), 1.0)


def test_std_normalization_preserves_single_domain_ordering():
    # The invariance contract: one scalar per domain => affine => every
    # within-domain ordering survives (doc ranking AND max-over-nodes), so the
    # readout's only:<m> columns cannot move under 'std'.
    from spark_vi.models.topic.dag_placement import (
        _auc, lr_placement_scores_multidomain)
    lay, lam, bows = _tiny()
    one_bow, one_lam = {0: bows[0]}, {0: lam[0]}
    raw = lr_placement_scores_multidomain(one_bow, one_lam, lay, alpha=1.0)
    std = lr_placement_scores_multidomain(one_bow, one_lam, lay, alpha=1.0,
                                          normalize="std")
    y = np.array([1, 0, 1, 0, 1])
    assert np.isclose(_auc(raw.max(axis=1), y), _auc(std.max(axis=1), y))
    assert np.array_equal(raw.argmax(axis=1), std.argmax(axis=1))


def test_length_normalization_matches_per_domain_length_normalize():
    from spark_vi.models.topic.dag_placement import (
        lr_placement_scores, lr_placement_scores_multidomain)
    lay, lam, bows = _tiny()
    s = lr_placement_scores_multidomain(bows, lam, lay, alpha=1.0,
                                        normalize="length")
    expect = sum(lr_placement_scores(bows[m], lam[m], lay, alpha=1.0,
                                     length_normalize=True) for m in (0, 1))
    assert np.allclose(s, expect)


def test_domain_scale_falls_back_to_one_on_a_constant_domain():
    # An all-zero (no-token) domain has zero std; it must pass through as zeros
    # and contribute nothing, not produce inf/nan.
    from spark_vi.models.topic.dag_placement import _domain_scale
    assert _domain_scale(np.zeros((4, 3))) == 1.0
    assert _domain_scale(np.full((4, 3), 2.5)) == 1.0


def test_unknown_normalize_raises():
    import pytest
    from spark_vi.models.topic.dag_placement import (
        lr_domain_score_matrices, lr_placement_scores_multidomain)
    lay, lam, bows = _tiny()
    with pytest.raises(ValueError):
        lr_placement_scores_multidomain(bows, lam, lay, alpha=1.0, normalize="zscore")
    with pytest.raises(ValueError):
        lr_domain_score_matrices(bows, lam, lay, alpha=1.0, normalize="zscore")


def test_auc_sweep_multidomain_forwards_normalize():
    from spark_vi.models.topic.dag_placement import (
        _auc, lr_placement_scores_multidomain, lr_auc_sweep_multidomain)
    lay, lam, bows = _tiny()
    is_fg = np.array([1, 0, 1, 0, 1])
    sweep = lr_auc_sweep_multidomain(bows, lam, lay, is_fg, alpha_grid=[1.0],
                                     normalize="std")
    s = lr_placement_scores_multidomain(bows, lam, lay, alpha=1.0, normalize="std")
    assert np.isclose(sweep[1.0], _auc(s.max(axis=1), is_fg))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd spark-vi && python -m pytest tests/test_lr_multidomain.py -v`
Expected: the new tests FAIL with `ImportError`/`TypeError` (`_domain_scale` and `lr_domain_score_matrices` do not exist; `normalize` is not a keyword). The five pre-existing tests in the file still PASS.

- [ ] **Step 3: Add `NORMALIZE_MODES`, `_domain_scale`, and `lr_domain_score_matrices`**

In `spark-vi/spark_vi/models/topic/dag_placement.py`, insert immediately BEFORE `lr_placement_scores_multidomain` (currently line 470):

```python
NORMALIZE_MODES = (None, "std", "length", "length+std")


def _domain_scale(s):
    """Per-domain scalar score scale for cross-domain comparability: the standard
    deviation of the whole [n_docs x n_nodes] score matrix.

    ONE scalar (never per-column) makes the per-domain transform affine over the
    whole matrix, so it preserves every within-domain ordering -- doc ranking AND
    max-over-nodes -- and moves only each domain's RELATIVE weight in the sum.
    Centering is deliberately omitted: a constant added to a domain's whole matrix
    cancels in both operations the readout performs, so scale is the minimal honest
    transform. A non-finite or non-positive std returns 1.0, so a constant or
    empty domain passes through unchanged instead of producing inf/nan (an
    all-zero domain then contributes nothing -- an inert domain, for free).
    See docs/superpowers/specs/2026-07-29-domain-normalized-lr-combination-design.md
    """
    sd = float(np.std(np.asarray(s, dtype=float)))
    return sd if np.isfinite(sd) and sd > 0.0 else 1.0


def lr_domain_score_matrices(bows, lam_dict, lay, *, alpha, domains=None,
                             backgrounds=None, epsilon=1e-9, count_mode="raw",
                             normalize=None):
    """{m: [n_docs x n_nodes]} per-domain LR placement score matrices, each already
    transformed per `normalize`. Summing a subset's matrices IS
    `lr_placement_scores_multidomain` over that subset, so a caller scoring MANY
    domain subsets (the readout's all / only:m / drop:m decomposition) computes
    each domain ONCE.

    normalize -- per-domain, applied before the caller sums:
      None          raw per-domain log-LR sums (plain Naive-Bayes-across-domains).
      'std'         divide by `_domain_scale`: equalizes each domain's score SCALE
                    so a high-token-volume domain cannot dominate the sum by
                    magnitude alone.
      'length'      per-doc divide by that domain's token count (mean log-LR per
                    token): removes the within-domain, across-doc utilization
                    confound, where heavily-coded patients own the head of the
                    ranking regardless of which codes they have.
      'length+std'  both.

    Each domain's transform is computed from that domain ALONE, so it does not
    depend on which subset the caller sums -- the per-domain decomposition stays
    coherent across subsets. Per-domain normalization costs the summed score its
    reading as a joint log-likelihood ratio; the readout is a RANKER at the
    alpha->inf lift limit (already not a posterior), so cross-domain comparability
    is worth more here than joint-likelihood interpretation. Rationale and the
    empirical motivation (insight 0072) are in
    docs/superpowers/specs/2026-07-29-domain-normalized-lr-combination-design.md
    """
    if normalize not in NORMALIZE_MODES:
        raise ValueError(
            f"normalize must be one of {NORMALIZE_MODES}, got {normalize!r}")
    doms = list(bows.keys()) if domains is None else list(domains)
    if not doms:
        raise ValueError("domains must select at least one domain")
    backgrounds = backgrounds or {}
    length_normalize = normalize in ("length", "length+std")
    out = {}
    for m in doms:
        s = lr_placement_scores(bows[m], lam_dict[m], lay, alpha=alpha,
                                background=backgrounds.get(m), epsilon=epsilon,
                                count_mode=count_mode,
                                length_normalize=length_normalize)
        if normalize in ("std", "length+std"):
            s = s / _domain_scale(s)
        out[m] = s
    return out
```

- [ ] **Step 4: Rewrite `lr_placement_scores_multidomain` to delegate**

Replace the whole existing function body AND docstring (`dag_placement.py:470-497`) with:

```python
def lr_placement_scores_multidomain(bows, lam_dict, lay, *, alpha, domains=None,
                                    backgrounds=None, epsilon=1e-9,
                                    count_mode="raw", normalize=None):
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
    normalize: per-domain transform applied BEFORE the sum -- None (raw, the
        default and unchanged behavior), 'std', 'length', or 'length+std'. See
        `lr_domain_score_matrices`; a subset is still the sum of its member
        domains under every rule.
    """
    mats = lr_domain_score_matrices(bows, lam_dict, lay, alpha=alpha,
                                    domains=domains, backgrounds=backgrounds,
                                    epsilon=epsilon, count_mode=count_mode,
                                    normalize=normalize)
    total = None
    for s in mats.values():           # insertion order == the caller's `domains`
        total = s if total is None else total + s
    return total
```

Note the deleted paragraph: the old docstring's closing note ("length_normalize is intentionally NOT supported here -- per-domain length normalization would break additivity") is superseded — a subset remains the sum of its members under every rule; see the spec's "On additivity" section. Do not carry that note forward.

- [ ] **Step 5: Forward `normalize` through `lr_auc_sweep_multidomain`**

In the same file, add the parameter to `lr_auc_sweep_multidomain` (currently line 500) and pass it through:

```python
def lr_auc_sweep_multidomain(bows, lam_dict, lay, is_fg, *, alpha_grid,
                             domains=None, backgrounds=None, count_mode="raw",
                             normalize=None):
    """{alpha: max-over-nodes ROC-AUC vs is_fg} for the multi-domain LR score,
    over a domain subset and under a per-domain normalization rule. Mirrors the
    single-domain `lr_auc_sweep`."""
    y = np.asarray(is_fg, dtype=int)
    out = {}
    for a in alpha_grid:
        s = lr_placement_scores_multidomain(bows, lam_dict, lay, alpha=float(a),
                                            domains=domains, backgrounds=backgrounds,
                                            count_mode=count_mode,
                                            normalize=normalize)
        out[float(a)] = _auc(s.max(axis=1), y)
    return out
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cd spark-vi && python -m pytest tests/test_lr_multidomain.py tests/test_dag_placement.py -v`
Expected: PASS, all of them (the five pre-existing multidomain tests included).

- [ ] **Step 7: Commit**

```bash
git add spark-vi/spark_vi/models/topic/dag_placement.py spark-vi/tests/test_lr_multidomain.py
git commit -m "feat(dag-placement): per-domain normalization for the multi-domain LR combination"
```

---

### Task 2: Readout — `--normalize` flag and the rule-comparison table

**Files:**
- Modify: `analysis/cloud/multidomain_lr_readout.py` (module docstring, `build_parser`, `main`; add `NORMALIZE_RULES`, `normalize_arg`, `pr_by_normalization`)
- Modify: `analysis/cloud/Makefile:126` (help line) and `:421-436` (the `multidomain-lr-readout` knob + recipe)
- Test: `analysis/cloud/tests/test_multidomain_lr_readout.py`

**Interfaces:**
- Consumes from Task 1: `lr_domain_score_matrices(bows, lam_dict, lay, *, alpha, domains=None, normalize=None) -> {m: ndarray}` and `lr_auc_sweep_multidomain(..., normalize=None)`.
- Produces: `NORMALIZE_RULES = ("none", "std", "length", "length+std")`, `normalize_arg(rule)`, `pr_by_normalization(...)`.

- [ ] **Step 1: Write the failing tests**

Append to `analysis/cloud/tests/test_multidomain_lr_readout.py` (this file imports readout functions by bare module name — follow that convention; it also defines a `_pr_fixture()` at line 100 returning `(lay, parent_int)`, reuse it):

```python
def test_normalize_arg_maps_none_to_the_library_value():
    from multidomain_lr_readout import NORMALIZE_RULES, normalize_arg
    assert normalize_arg("none") is None
    assert normalize_arg("std") == "std"
    assert normalize_arg("length+std") == "length+std"
    assert NORMALIZE_RULES == ("none", "std", "length", "length+std")


def test_normalize_arg_rejects_unknown_rule():
    import pytest
    from multidomain_lr_readout import normalize_arg
    with pytest.raises(ValueError):
        normalize_arg("zscore")


def test_build_parser_normalize_default_and_choices():
    import pytest
    from multidomain_lr_readout import build_parser
    p = build_parser()
    assert p.parse_args(["--run-dir", "/x"]).normalize == "none"
    assert p.parse_args(["--run-dir", "/x", "--normalize", "std"]).normalize == "std"
    with pytest.raises(SystemExit):
        p.parse_args(["--run-dir", "/x", "--normalize", "bogus"])


def _norm_fixture():
    """2-domain LR inputs over the _pr_fixture 1-node layout (anchor = node 1,
    subtree(1) == {1}): per-domain lam sharing the K topics, per-domain BOWs with
    different V, and frontiers alternating hit/miss so y carries both classes."""
    lay, parent_int = _pr_fixture()
    rng = np.random.default_rng(3)
    lam = {0: rng.random((lay.K, 6)) + 0.1, 1: rng.random((lay.K, 4)) + 0.1}
    bows = {0: rng.integers(0, 3, size=(8, 6)).astype(float),
            1: rng.integers(0, 3, size=(8, 4)).astype(float)}
    frontiers = [[1] if i % 2 == 0 else [] for i in range(8)]
    return lay, parent_int, lam, bows, 1, frontiers


def test_pr_by_normalization_shape_and_none_matches_direct_pr():
    # 'none' must agree with per_disease_pr on the un-normalized subset sum, so
    # the comparison table's reference column is the same number the PR table
    # already prints.
    from spark_vi.models.topic.dag_placement import lr_placement_scores_multidomain
    from multidomain_lr_readout import (
        NORMALIZE_RULES, per_disease_pr, pr_by_normalization)
    lay, parent_int, lam, bows, anchor, frontiers = _norm_fixture()
    table = pr_by_normalization(bows, lam, lay, frontiers, [anchor], parent_int,
                                alpha=float("inf"))
    assert set(table) == set(NORMALIZE_RULES)
    assert set(table["std"]) == {anchor}
    direct = per_disease_pr(
        lr_placement_scores_multidomain(bows, lam, lay, alpha=float("inf")),
        frontiers, anchor, lay, parent_int)[0]
    assert np.isclose(table["none"][anchor], direct)


def test_pr_by_normalization_honors_the_domain_subset():
    # A subset restriction must reach the scorer: single-domain 'none' equals
    # per_disease_pr on that domain alone.
    from spark_vi.models.topic.dag_placement import lr_placement_scores_multidomain
    from multidomain_lr_readout import per_disease_pr, pr_by_normalization
    lay, parent_int, lam, bows, anchor, frontiers = _norm_fixture()
    table = pr_by_normalization(bows, lam, lay, frontiers, [anchor], parent_int,
                                alpha=float("inf"), domains=[0], rules=("none",))
    direct = per_disease_pr(
        lr_placement_scores_multidomain(bows, lam, lay, alpha=float("inf"),
                                        domains=[0]),
        frontiers, anchor, lay, parent_int)[0]
    assert np.isclose(table["none"][anchor], direct)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest analysis/cloud/tests/test_multidomain_lr_readout.py -v`
Expected: the new tests FAIL with `ImportError` (`NORMALIZE_RULES`, `normalize_arg`, `pr_by_normalization` do not exist). Pre-existing tests still PASS. (If the bare-module import needs a path entry, an existing `conftest.py` under `analysis/cloud/tests/` already handles it — the pre-existing tests use the same style.)

- [ ] **Step 3: Add the rule constants, the arg mapper, and the comparison helper**

In `analysis/cloud/multidomain_lr_readout.py`, add after the imports (before `build_parser`):

```python
# CLI spellings of the per-domain normalization rules. 'none' maps to the
# library's None; see spark_vi.models.topic.dag_placement.NORMALIZE_MODES and
# docs/superpowers/specs/2026-07-29-domain-normalized-lr-combination-design.md
NORMALIZE_RULES = ("none", "std", "length", "length+std")


def normalize_arg(rule):
    """CLI rule name -> the library `normalize` value ('none' -> None)."""
    if rule not in NORMALIZE_RULES:
        raise ValueError(f"unknown normalization rule {rule!r}")
    return None if rule == "none" else rule
```

Add the flag in `build_parser`, after `--alpha-grid`:

```python
    p.add_argument("--normalize", default="none", choices=list(NORMALIZE_RULES),
                   help="Per-domain score normalization applied before the "
                        "domain sum: none (raw), std (equalize per-domain scale), "
                        "length (per-doc mean log-LR per token), or length+std. "
                        "Governs the main tables; the comparison table always "
                        "shows every rule.")
```

Add the comparison helper next to `per_disease_pr` (after it):

```python
def pr_by_normalization(bows, lam_dict, lay, frontiers, anchors, parent_int, *,
                        alpha, domains=None, rules=NORMALIZE_RULES):
    """{rule: {anchor: pr_auc}} -- PR-AUC per anchor for ONE fixed domain subset,
    under each per-domain normalization rule.

    This is the A/B for insight 0072's finding that a high-volume low-signal
    domain costs most of the precision: compare subset `all` under each rule
    against the subset that DROPS the suspect domain under rule 'none'. PR (not
    ROC) is the metric, because the damage is at the head of the ranking.
    """
    from spark_vi.models.topic.dag_placement import lr_domain_score_matrices
    out = {}
    for rule in rules:
        mats = lr_domain_score_matrices(bows, lam_dict, lay, alpha=alpha,
                                        domains=domains,
                                        normalize=normalize_arg(rule))
        scores = None
        for s in mats.values():
            scores = s if scores is None else scores + s
        out[rule] = {u: per_disease_pr(scores, frontiers, u, lay, parent_int)[0]
                     for u in anchors}
    return out
```

- [ ] **Step 4: Wire the flag through `main` and replace the per-subset recompute**

In `main()`, extend the import at the top of the function to bring in the new accessor:

```python
    from spark_vi.models.topic.dag_placement import (
        DagLayout, lr_domain_score_matrices)
```

(`lr_placement_scores_multidomain` is no longer called in `main` after this step — drop it from that import. `lr_auc_sweep_multidomain` keeps its existing local import further down.)

Right after `alpha_grid = parse_alpha_grid(args.alpha_grid)`, add:

```python
    norm = normalize_arg(args.normalize)
```

Forward it to the overall sweep — change the `lr_auc_sweep_multidomain(...)` call to:

```python
    sweep = lr_auc_sweep_multidomain(bows, lam_dict, lay, is_fg,
                                     alpha_grid=alpha_grid, normalize=norm)
```

and name the rule in that section's header so the output is unambiguous:

```python
    print(f"[lr] === overall detection LR-AUC(alpha), all domains, "
          f"max-over-nodes, normalize={args.normalize} ===", flush=True)
```

Replace the `subset_scores = {...}` dict comprehension (currently `multidomain_lr_readout.py:264-266`) with a compute-once-per-domain version:

```python
    # Per-domain score matrices ONCE; every subset is the sum of its members.
    # Each domain's normalization is computed from that domain alone, so a
    # domain contributes the same in `all` as in `drop:x` and the decomposition
    # stays coherent across subsets.
    dom_mats = lr_domain_score_matrices(bows, lam_dict, lay, alpha=a_head,
                                        normalize=norm)
    subset_scores = {}
    for name, doms in subsets.items():
        total = None
        for i in doms:
            total = dom_mats[i] if total is None else total + dom_mats[i]
        subset_scores[name] = total
```

Add `normalize={args.normalize}` to the two table headers that report those scores (the per-disease LR-AUC header and the PR-AUC header), e.g.:

```python
    print(f"[lr] === per-disease x domain-subset LR-AUC (alpha={a_head}, "
          f"normalize={args.normalize}) ===", flush=True)
```

```python
    print(f"[lr] === per-disease x domain-subset PR-AUC (avg precision, "
          f"alpha={a_head}, normalize={args.normalize}) ===", flush=True)
```

- [ ] **Step 5: Add the rule-comparison table**

Insert immediately BEFORE the final `print(f"[lr] scored {n_docs} held-out docs; ...")` line. `last_domain` is already computed above for the headline subsets — reuse it, do not recompute:

```python
    # --- Domain-normalization comparison: PR-AUC for subset `all` under every
    # rule, beside the un-normalized drop-the-suspect-domain column as the
    # target. insight 0072 measured drop:observation >= all for every disease;
    # the question here is whether a normalization rule closes that gap so a
    # low-signal domain can stay in the model without costing precision. ---
    ref_name = f"drop:{last_domain}" if f"drop:{last_domain}" in subsets else None
    all_by_rule = pr_by_normalization(bows, lam_dict, lay, frontiers, anchors,
                                      parent_int, alpha=a_head)
    ref = {}
    if ref_name:
        ref = pr_by_normalization(bows, lam_dict, lay, frontiers, anchors,
                                  parent_int, alpha=a_head,
                                  domains=subsets[ref_name], rules=("none",))["none"]
    print(f"[lr] === PR-AUC by domain-normalization rule (subset=all, "
          f"alpha={a_head}) ===", flush=True)
    header = "disease".ljust(26)
    for rule in NORMALIZE_RULES:
        header += "  " + f"all|{rule}"[:14].rjust(14)
    if ref_name:
        header += "  " + f"{ref_name}|none"[:18].rjust(18)
    print("[lr] " + header, flush=True)
    for u in anchors:
        dname = str(name_by_engine.get(u, int2cid.get(u)))[:24]
        line = dname.ljust(26)
        for rule in NORMALIZE_RULES:
            line += "  " + f"{all_by_rule[rule][u]:14.3f}"
        if ref_name:
            line += "  " + f"{ref[u]:18.3f}"
        print("[lr] " + line, flush=True)
    if ref_name:
        print(f"[lr]   (target: an all|<rule> column matching {ref_name}|none "
              f"means keeping {last_domain} costs no precision)", flush=True)
```

- [ ] **Step 6: Update the module docstring**

In the module docstring at the top of `analysis/cloud/multidomain_lr_readout.py`, replace the paragraph beginning "The multi-domain LR score is the per-domain SUM..." with:

```
The multi-domain LR score is the per-domain SUM of the single-domain
lr_placement_scores; a domain subset is the per-domain decomposition. Per-disease
detection is max-over-subtree(anchor) vs frontier-hits-subtree. --normalize
applies a per-domain transform before that sum (none/std/length/length+std) so a
high-token-volume domain cannot dominate the ranking by magnitude alone; the
final table compares every rule against dropping the suspect domain outright
(insight 0072).
```

and add `normalize_arg`, `pr_by_normalization` to the list of unit-tested pure functions in the last paragraph.

- [ ] **Step 7: Add the Makefile knob**

In `analysis/cloud/Makefile`, beside the existing `MULTIDOMAIN_LR_ALPHA_GRID ?= 0,1,10,100,inf` (line 421) add:

```make
MULTIDOMAIN_LR_NORMALIZE ?= none
```

Append the flag to the recipe's final line (line 436), keeping the line-continuation style intact:

```make
	    --alpha-grid $(MULTIDOMAIN_LR_ALPHA_GRID) \
	    --normalize $(MULTIDOMAIN_LR_NORMALIZE)
```

Update the help line (line 126) to advertise the knob:

```make
	@echo "  multidomain-lr-readout ID=N [MULTIDOMAIN_LR_ALPHA_GRID=.. MULTIDOMAIN_LR_NORMALIZE=none|std|length|length+std]  Post-hoc per-domain LR readout (multidomain fit)"
```

- [ ] **Step 8: Run the tests**

Run: `python -m pytest analysis/cloud/tests/test_multidomain_lr_readout.py -v`
Expected: PASS, all of them.

Then confirm the Makefile still parses and shows the knob:

Run: `make -n -C analysis/cloud multidomain-lr-readout ID=71 2>&1 | grep -c "normalize"`
Expected: at least `1` (the dry-run recipe contains `--normalize none`). A `make` parse error here means the continuation backslash in Step 7 is wrong.

- [ ] **Step 9: Commit**

```bash
git add analysis/cloud/multidomain_lr_readout.py analysis/cloud/tests/test_multidomain_lr_readout.py analysis/cloud/Makefile
git commit -m "feat(multidomain): --normalize flag + per-domain normalization comparison table in the LR readout"
```

---

## Final verification

Run the full suites both files belong to:

```bash
cd spark-vi && python -m pytest tests/ -q
cd .. && python -m pytest analysis/cloud/tests/ scripts/tests/ -q
```

Expected: green, with no reduction in collected test count versus `main`.
