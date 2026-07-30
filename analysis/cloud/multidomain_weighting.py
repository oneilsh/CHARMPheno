"""Leakage-safe nested evaluation of multi-domain case-finding scores.

The helpers keep every learned transformation explicit: callers supply the
rows used to estimate backgrounds and score scales, and validation/test rows
are transformed only with those frozen training values.

Implements the approved hybrid reliability design in
``docs/superpowers/specs/2026-07-29-hybrid-domain-reliability-readout-design.md``.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


def anchor_truth(frontiers, subtree: set[int]) -> np.ndarray:
    """Binary disease truth: a document is positive when its frontier hits subtree."""
    subtree = set(subtree)
    return np.asarray(
        [int(bool(set(frontier) & subtree)) for frontier in frontiers], dtype=int
    )


def subtree_columns(lay, subtree: set[int]) -> np.ndarray:
    """Map subtree node ids to score columns, preserving ``lay.nodes`` order."""
    subtree = set(subtree)
    return np.asarray(
        [column for column, node in enumerate(lay.nodes) if node in subtree],
        dtype=int,
    )


def max_subtree_score(score_matrix, columns) -> np.ndarray:
    """Collapse node scores only after all domain-level operations are complete."""
    scores = np.asarray(score_matrix, dtype=float)
    columns = np.asarray(columns, dtype=int)
    if scores.ndim != 2:
        raise ValueError("score_matrix must be two-dimensional")
    if columns.ndim != 1 or columns.size == 0:
        raise ValueError("subtree must contain at least one score column")
    return np.max(scores[:, columns], axis=1)


def _backgrounds_from_rows(bows, rows):
    """Estimate one domain background from exactly the caller's training rows."""
    from spark_vi.models.topic.dag_placement import lr_background

    rows = np.asarray(rows, dtype=int)
    return {domain: lr_background(bows[domain][rows]) for domain in sorted(bows)}


def _raw_scores_for_rows(
    bows,
    lam_dict,
    lay,
    rows,
    backgrounds,
    *,
    length=False,
):
    """Score selected rows with frozen backgrounds and no transductive scaling."""
    from spark_vi.models.topic.dag_placement import lr_domain_score_matrices

    rows = np.asarray(rows, dtype=int)
    domains = sorted(bows)
    return lr_domain_score_matrices(
        {domain: bows[domain][rows] for domain in domains},
        lam_dict,
        lay,
        alpha=float("inf"),
        backgrounds=backgrounds,
        normalize="length" if length else None,
    )


def _scales_from_matrices(matrices, columns):
    """Fit each domain's scalar on training-row anchor-subtree maxima.

    The resulting scalar is later applied to the entire node matrix. Fitting it
    on the anchor score makes the learned transformation specific to the
    selection target without changing which node wins within a document.
    """
    from spark_vi.models.topic.dag_placement import domain_score_scale

    return {
        domain: domain_score_scale(max_subtree_score(matrix, columns))
        for domain, matrix in sorted(matrices.items())
    }


def _weighted_subtree_score(matrices, weights, columns, *, scales=None):
    """Combine node matrices with ordered weights, then max over the subtree."""
    from spark_vi.models.topic.dag_placement import combine_domain_score_matrices

    domains = sorted(matrices)
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or len(weights) != len(domains):
        raise ValueError("weights must contain one value per sorted domain")
    combined = combine_domain_score_matrices(
        matrices,
        weights=dict(zip(domains, weights)),
        scales=scales,
    )
    return max_subtree_score(combined, columns)


def select_simplex_weights(train_matrices, y, columns, *, grid) -> np.ndarray:
    """Select pooled-OOF weights by AP with stable, low-concentration ties.

    All candidates combine complete node matrices before the subtree maximum.
    An AP tie within 1e-12 first favors the candidate nearest uniform weights,
    then the lexicographically smallest tuple. The supervised multi-view
    diagnostic follows the approved hybrid design; cross-validated downstream
    prediction from shared multi-domain topics has precedent in MixEHR
    (Li, Nair, Lu et al. 2020, Nature Communications 11:2536).
    """
    from spark_vi.models.topic.dag_placement import _average_precision

    domains = sorted(train_matrices)
    if not domains:
        raise ValueError("train_matrices must contain at least one domain")
    labels = np.asarray(y, dtype=int)
    if labels.ndim != 1:
        raise ValueError("y must be one-dimensional")
    candidates = np.asarray(grid, dtype=float)
    if candidates.ndim != 2 or candidates.shape[1] != len(domains):
        raise ValueError("grid must have one column per sorted domain")
    if len(candidates) == 0:
        raise ValueError("grid must contain at least one candidate")
    if (
        not np.all(np.isfinite(candidates))
        or np.any(candidates < 0.0)
        or not np.allclose(
            candidates.sum(axis=1),
            1.0,
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise ValueError("grid rows must be nonnegative simplex weights")

    scored = []
    uniform = np.full(len(domains), 1.0 / len(domains))
    for weights in candidates:
        score = _weighted_subtree_score(train_matrices, weights, columns)
        ap = float(_average_precision(score, labels))
        distance = float(np.sum((weights - uniform) ** 2))
        scored.append((ap, distance, tuple(float(value) for value in weights)))

    best_ap = max(ap for ap, _, _ in scored)
    ap_ties = [item for item in scored if best_ap - item[0] <= 1e-12]
    _, _, selected = min(ap_ties, key=lambda item: (item[1], item[2]))
    return np.asarray(selected, dtype=float)


def discrete_policies(
    domain_keys,
) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    """Enumerate every nonempty domain subset crossed with four transforms.

    Subsets follow sorted combination order. This tuple order is also the
    discrete selector's deterministic AP tie-break.
    """
    domains = tuple(sorted(domain_keys))
    if not domains:
        raise ValueError("domain_keys must contain at least one domain")
    normalizations = ("none", "std", "length", "length+std")
    policies = []
    for size in range(1, len(domains) + 1):
        for subset in combinations(domains, size):
            if len(subset) == len(domains):
                subset_name = "all"
            elif len(subset) == 1:
                subset_name = f"only:{subset[0]}"
            elif len(subset) == len(domains) - 1:
                missing = next(domain for domain in domains if domain not in subset)
                subset_name = f"drop:{missing}"
            else:
                subset_name = "subset:" + "+".join(str(domain) for domain in subset)
            for normalization in normalizations:
                policies.append(
                    (
                        f"{subset_name}|{normalization}",
                        tuple(subset),
                        normalization,
                    )
                )
    return tuple(policies)


def _transformed_fold_matrices(
    bows,
    lam_dict,
    lay,
    train_rows,
    score_rows,
    columns,
    *,
    length,
):
    """Fit fold-local background/scale and transform another row partition."""
    backgrounds = _backgrounds_from_rows(bows, train_rows)
    train_matrices = _raw_scores_for_rows(
        bows,
        lam_dict,
        lay,
        train_rows,
        backgrounds,
        length=length,
    )
    score_matrices = _raw_scores_for_rows(
        bows,
        lam_dict,
        lay,
        score_rows,
        backgrounds,
        length=length,
    )
    scales = _scales_from_matrices(train_matrices, columns)
    return score_matrices, scales


def _discrete_policy_score(matrices, policy, columns, *, scales):
    """Score one named subset/normalization policy from frozen matrices."""
    _, subset, normalization = policy
    selected = {domain: matrices[domain] for domain in subset}
    selected_scales = (
        {domain: scales[domain] for domain in subset}
        if normalization in ("std", "length+std")
        else None
    )
    return _weighted_subtree_score(
        selected,
        np.ones(len(subset), dtype=float),
        columns,
        scales=selected_scales,
    )


def _select_discrete_policy_from_ap(policies, ap_values):
    """Choose highest AP, retaining policy tuple order for ties within 1e-12."""
    if not policies or len(policies) != len(ap_values):
        raise ValueError("policies and ap_values must have the same nonzero length")
    best_ap = max(ap_values)
    for policy, ap in zip(policies, ap_values):
        if best_ap - ap <= 1e-12:
            return policy
    raise RuntimeError("discrete policy selection produced no candidate")


def _select_discrete_policy(policies, pooled_scores, y):
    """Choose highest pooled-OOF AP, retaining tuple order for 1e-12 ties."""
    from spark_vi.models.topic.dag_placement import _average_precision

    labels = np.asarray(y, dtype=int)
    ap_values = [
        float(_average_precision(pooled_scores[name], labels))
        for name, _, _ in policies
    ]
    return _select_discrete_policy_from_ap(policies, ap_values)


def _inner_oof_selection(
    bows,
    lam_dict,
    lay,
    y,
    columns,
    outer_train,
    *,
    inner_folds,
    grid,
    seed,
):
    """Select discrete and continuous rules from pooled inner OOF predictions."""
    outer_train = np.asarray(outer_train, dtype=int)
    train_y = np.asarray(y, dtype=int)[outer_train]
    domains = tuple(sorted(bows))
    policies = discrete_policies(domains)
    pooled_policy_scores = {
        name: np.empty(len(outer_train), dtype=float) for name, _, _ in policies
    }
    pooled_continuous_matrices = {
        domain: np.empty(
            (len(outer_train), len(lay.nodes)),
            dtype=float,
        )
        for domain in domains
    }

    inner_partitions = stratified_folds(
        train_y,
        n_splits=inner_folds,
        seed=seed,
    )
    for inner_train_local, inner_validation_local in inner_partitions:
        inner_train_rows = outer_train[inner_train_local]
        inner_validation_rows = outer_train[inner_validation_local]
        raw_validation, raw_scales = _transformed_fold_matrices(
            bows,
            lam_dict,
            lay,
            inner_train_rows,
            inner_validation_rows,
            columns,
            length=False,
        )
        length_validation, length_scales = _transformed_fold_matrices(
            bows,
            lam_dict,
            lay,
            inner_train_rows,
            inner_validation_rows,
            columns,
            length=True,
        )

        for domain in domains:
            pooled_continuous_matrices[domain][inner_validation_local] = (
                raw_validation[domain] / raw_scales[domain]
            )
        for policy in policies:
            name, _, normalization = policy
            if normalization in ("length", "length+std"):
                matrices, scales = length_validation, length_scales
            else:
                matrices, scales = raw_validation, raw_scales
            pooled_policy_scores[name][inner_validation_local] = (
                _discrete_policy_score(
                    matrices,
                    policy,
                    columns,
                    scales=scales,
                )
            )

    selected_policy = _select_discrete_policy(
        policies,
        pooled_policy_scores,
        train_y,
    )
    selected_weights = select_simplex_weights(
        pooled_continuous_matrices,
        train_y,
        columns,
        grid=grid,
    )
    return selected_policy, selected_weights


def _fixed_condition_drug_domains(domain_keys):
    """Resolve the predeclared condition+drug baseline for named or ordinal keys."""
    domains = tuple(sorted(domain_keys))
    if "condition" in domains and "drug" in domains:
        return ("condition", "drug")
    return domains[: min(2, len(domains))]


def _evaluate_outer_fold(
    *,
    bows,
    lam_dict,
    lay,
    y,
    columns,
    outer_train,
    outer_test,
    inner_folds,
    grid,
    inner_seed,
    model_weights=None,
):
    """Select on outer-training inner OOF data and score one frozen outer test."""
    outer_train = np.asarray(outer_train, dtype=int)
    outer_test = np.asarray(outer_test, dtype=int)
    if np.intersect1d(outer_train, outer_test).size:
        raise ValueError("outer_train and outer_test must be disjoint")

    selected_policy, continuous_weights = _inner_oof_selection(
        bows,
        lam_dict,
        lay,
        y,
        columns,
        outer_train,
        inner_folds=inner_folds,
        grid=grid,
        seed=inner_seed,
    )

    raw_test, raw_scales = _transformed_fold_matrices(
        bows,
        lam_dict,
        lay,
        outer_train,
        outer_test,
        columns,
        length=False,
    )
    _, _, selected_normalization = selected_policy
    if selected_normalization in ("length", "length+std"):
        discrete_test, discrete_scales = _transformed_fold_matrices(
            bows,
            lam_dict,
            lay,
            outer_train,
            outer_test,
            columns,
            length=True,
        )
    else:
        discrete_test, discrete_scales = raw_test, raw_scales

    fixed_domains = _fixed_condition_drug_domains(bows)
    fixed_matrices = {domain: raw_test[domain] for domain in fixed_domains}
    fixed_score = _weighted_subtree_score(
        fixed_matrices,
        np.ones(len(fixed_domains), dtype=float),
        columns,
    )
    discrete_score = _discrete_policy_score(
        discrete_test,
        selected_policy,
        columns,
        scales=discrete_scales,
    )
    continuous_score = _weighted_subtree_score(
        raw_test,
        continuous_weights,
        columns,
        scales=raw_scales,
    )
    model_scores = {
        strategy: np.asarray(
            evaluate_model_candidate(
                raw_test,
                np.asarray(y, dtype=int)[outer_test],
                columns,
                weights=weights,
                scales=raw_scales,
            )["scores"],
            dtype=float,
        )
        for strategy, weights in (model_weights or {}).items()
    }
    return {
        "discrete_policy": selected_policy[0],
        "continuous_weights": continuous_weights,
        "scores": {
            "fixed:condition_drug": fixed_score,
            "discrete": discrete_score,
            "continuous": continuous_score,
            **model_scores,
        },
    }


def _descendant_subtree(parent_int, anchor):
    """Return an anchor and all descendants from a scalar/list parent mapping."""
    children = {}
    for child, parents in parent_int.items():
        parent_values = (
            list(parents)
            if isinstance(parents, (list, tuple, set))
            else [parents]
        )
        for parent in parent_values:
            children.setdefault(int(parent), set()).add(int(child))
    seen = set()
    stack = [int(anchor)]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(children.get(node, ()))
    return seen


def _strategy_metrics(scores, y):
    """Return the public, strictly Python-valued AP/precision schema.

    AP is the step integral over distinct score thresholds, not trapezoidal PR
    interpolation (Davis & Goadrich 2006, ICML), so tied scores define one
    achievable operating point and a constant ranker returns prevalence.
    """
    from spark_vi.models.topic.dag_placement import (
        _average_precision,
        _precision_at_recall,
    )

    recalls = (0.10, 0.25, 0.50, 0.80)
    precision = _precision_at_recall(scores, y, recalls=recalls)
    return {
        "ap": float(_average_precision(scores, y)),
        "precision_at_recall": {
            format(recall, ".12g"): float(precision[recall]) for recall in recalls
        },
    }


def evaluate_model_candidate(
    matrices,
    y,
    columns,
    *,
    weights,
    scales,
) -> dict:
    """Return JSON-safe fixed model weights, metrics, and patient scores.

    The caller supplies one disease-anchor weight vector.  That vector and the
    frozen fold-local scales apply to every node column before the descendant
    maximum; labels are consumed only by the reported metrics.
    """
    weights = np.asarray(weights, dtype=float)
    scores = _weighted_subtree_score(
        matrices,
        weights,
        columns,
        scales=scales,
    )
    return {
        "weights": [float(weight) for weight in weights],
        **_strategy_metrics(scores, y),
        "scores": [float(score) for score in scores],
    }


def _agreement_diagnostics(
    candidate_scores,
    continuous_scores,
    *,
    candidate_weights,
    median_supervised_weights,
    n_positive,
):
    """Compare one fixed model candidate with the supervised OOF ceiling.

    Stable descending row order resolves score ties for the predeclared top-set
    comparison.  Constant inputs have no defined rank correlation and are
    represented by ``None`` instead of a non-JSON NaN.
    """
    from scipy.stats import spearmanr

    candidate_scores = np.asarray(candidate_scores, dtype=float)
    continuous_scores = np.asarray(continuous_scores, dtype=float)
    if (
        candidate_scores.ndim != 1
        or continuous_scores.ndim != 1
        or candidate_scores.shape != continuous_scores.shape
        or candidate_scores.size == 0
    ):
        raise ValueError("agreement score vectors must be nonempty and aligned")

    candidate_constant = np.all(candidate_scores == candidate_scores[0])
    continuous_constant = np.all(continuous_scores == continuous_scores[0])
    if candidate_constant or continuous_constant:
        correlation = None
    else:
        observed = float(
            spearmanr(candidate_scores, continuous_scores).correlation
        )
        correlation = observed if np.isfinite(observed) else None

    n_docs = len(candidate_scores)
    top_count = min(
        n_docs,
        max(int(n_positive), int(np.ceil(0.01 * n_docs))),
    )
    candidate_top = set(
        np.argsort(-candidate_scores, kind="mergesort")[:top_count].tolist()
    )
    continuous_top = set(
        np.argsort(-continuous_scores, kind="mergesort")[:top_count].tolist()
    )
    jaccard = len(candidate_top & continuous_top) / len(
        candidate_top | continuous_top
    )

    candidate_weights = np.asarray(candidate_weights, dtype=float)
    median_supervised_weights = np.asarray(
        median_supervised_weights,
        dtype=float,
    )
    if (
        candidate_weights.ndim != 1
        or candidate_weights.shape != median_supervised_weights.shape
    ):
        raise ValueError("agreement weight vectors must be aligned")
    candidate_order = tuple(
        np.argsort(-candidate_weights, kind="mergesort").tolist()
    )
    supervised_order = tuple(
        np.argsort(-median_supervised_weights, kind="mergesort").tolist()
    )
    return {
        "spearman_with_continuous": correlation,
        "top_set_jaccard_with_continuous": float(jaccard),
        "same_domain_order_as_median_supervised": bool(
            candidate_order == supervised_order
        ),
    }


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
) -> dict:
    """Evaluate fixed, discrete, and continuous scores with honest nested CV.

    Per the approved hybrid reliability design, inner pooled OOF predictions
    choose disease-specific rules and shared outer folds report the diagnostic
    ceiling. The nested supervised rationale follows the cross-validated
    MixEHR downstream classifier precedent (Li, Nair, Lu et al. 2020).
    """
    domains = tuple(sorted(bows))
    if not domains or set(bows) != set(lam_dict):
        raise ValueError("bows and lam_dict must have identical nonempty domain keys")
    n_docs = len(frontiers)
    if any(matrix.shape[0] != n_docs for matrix in bows.values()):
        raise ValueError("every BOW matrix must have one row per frontier")
    if int(repeats) != repeats or int(repeats) < 1:
        raise ValueError("repeats must be a positive integer")
    repeats = int(repeats)

    scoreable_subtree = _descendant_subtree(parent_int, anchor) & set(lay.nodes)
    columns = subtree_columns(lay, scoreable_subtree)
    if columns.size == 0:
        raise ValueError("anchor subtree has no scoreable layout nodes")
    y = anchor_truth(frontiers, scoreable_subtree)
    grid = simplex_grid(len(domains), grid_step)

    if anchor not in lay.nodes:
        raise ValueError("anchor must be a scoreable layout node for model weights")
    from spark_vi.models.topic.dag_placement import domain_reliability

    reliability = domain_reliability(lam_dict, lay)
    if tuple(reliability.domain_keys) != domains:
        raise ValueError("domain reliability keys must match sorted BOW domains")
    anchor_row = lay.nodes.index(anchor)
    model_weights = {
        f"model:{candidate}": reliability.weights(candidate)[anchor_row]
        for candidate in ("distinctiveness", "ownership", "product")
    }

    repeat_results = []
    for repeat in range(repeats):
        outer_partitions = stratified_folds(
            y,
            n_splits=outer_folds,
            seed=int(seed) + repeat,
        )
        pooled_scores = {
            "fixed:condition_drug": np.empty(n_docs, dtype=float),
            "discrete": np.empty(n_docs, dtype=float),
            "continuous": np.empty(n_docs, dtype=float),
            **{
                strategy: np.empty(n_docs, dtype=float)
                for strategy in model_weights
            },
        }
        fold_results = []
        for fold, (outer_train, outer_test) in enumerate(outer_partitions):
            evaluated = _evaluate_outer_fold(
                bows=bows,
                lam_dict=lam_dict,
                lay=lay,
                y=y,
                columns=columns,
                outer_train=outer_train,
                outer_test=outer_test,
                inner_folds=inner_folds,
                grid=grid,
                inner_seed=int(seed) + 100_000 * (repeat + 1) + fold,
                model_weights=model_weights,
            )
            for strategy, scores in evaluated["scores"].items():
                pooled_scores[strategy][outer_test] = scores
            fold_results.append(
                {
                    "fold": int(fold),
                    "test_rows": [int(row) for row in outer_test],
                    "discrete_policy": str(evaluated["discrete_policy"]),
                    "continuous_weights": [
                        float(weight)
                        for weight in evaluated["continuous_weights"]
                    ],
                }
            )

        median_supervised_weights = np.median(
            np.asarray(
                [fold["continuous_weights"] for fold in fold_results],
                dtype=float,
            ),
            axis=0,
        )
        agreements = {
            strategy: {
                "weights": [float(weight) for weight in weights],
                **_agreement_diagnostics(
                    pooled_scores[strategy],
                    pooled_scores["continuous"],
                    candidate_weights=weights,
                    median_supervised_weights=median_supervised_weights,
                    n_positive=int(y.sum()),
                ),
            }
            for strategy, weights in model_weights.items()
        }
        repeat_results.append(
            {
                "repeat": int(repeat),
                "strategies": {
                    strategy: _strategy_metrics(scores, y)
                    for strategy, scores in pooled_scores.items()
                },
                "agreements": agreements,
                "folds": fold_results,
            }
        )

    return {
        "anchor": int(anchor),
        "n_docs": int(n_docs),
        "n_positive": int(y.sum()),
        "prevalence": float(y.mean()),
        "repeats": repeat_results,
    }


def simplex_grid(n_domains: int, step: float) -> np.ndarray:
    """Enumerate simplex weights in deterministic lexicographic order.

    Integer compositions avoid cumulative floating-point drift: each row is a
    composition of ``round(1 / step)`` units, converted to floats only after
    enumeration.
    """
    if int(n_domains) != n_domains or int(n_domains) < 1:
        raise ValueError("n_domains must select at least one domain")
    n_domains = int(n_domains)
    try:
        step = float(step)
    except (TypeError, ValueError):
        raise ValueError("step must be positive") from None
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step must be positive")
    units = round(1.0 / step)
    if units < 1 or not np.isclose(step * units, 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("step must divide 1 exactly within 1e-12")

    rows: list[tuple[int, ...]] = []

    def append_compositions(prefix: tuple[int, ...], remaining: int) -> None:
        if len(prefix) == n_domains - 1:
            rows.append(prefix + (remaining,))
            return
        for value in range(remaining + 1):
            append_compositions(prefix + (value,), remaining - value)

    append_compositions((), units)
    return np.asarray(rows, dtype=float) / float(units)


def stratified_folds(
    y,
    *,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return deterministic binary-stratified ``(train, test)`` row indices."""
    labels = np.asarray(y)
    if labels.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if int(n_splits) != n_splits or int(n_splits) < 2:
        raise ValueError("n_splits must be at least 2")
    n_splits = int(n_splits)
    classes = np.unique(labels)
    if len(classes) != 2 or not np.array_equal(classes, np.array([0, 1])):
        raise ValueError("y must contain both classes encoded as 0 and 1")

    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    if min(len(negative), len(positive)) < n_splits:
        raise ValueError("a class count is smaller than n_splits")

    rng = np.random.default_rng(seed)
    rng.shuffle(positive)
    rng.shuffle(negative)
    positive_parts = np.array_split(positive, n_splits)
    negative_parts = np.array_split(negative, n_splits)
    all_rows = np.arange(len(labels), dtype=int)
    folds = []
    for positive_test, negative_test in zip(positive_parts, negative_parts):
        test = np.sort(np.concatenate((positive_test, negative_test))).astype(
            int, copy=False
        )
        train = np.setdiff1d(all_rows, test, assume_unique=True)
        folds.append((train, test))
    return folds
