import json

import numpy as np
import pytest


def test_simplex_grid_is_deterministic_nonnegative_and_sums_to_one():
    """Catches floating-step enumeration that skips or reorders grid points."""
    from multidomain_weighting import simplex_grid

    grid = simplex_grid(3, 0.5)
    assert np.array_equal(
        grid,
        np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.5, 0.5],
                [0.0, 1.0, 0.0],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ),
    )
    assert np.all(grid >= 0.0)
    assert np.allclose(grid.sum(axis=1), 1.0)


def test_stratified_folds_cover_each_row_once_and_preserve_classes():
    """Catches unstratified or nondeterministic held-out fold construction."""
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


def test_grid_and_folds_reject_invalid_partitions():
    """Catches silently rounded grids and folds that cannot preserve both classes."""
    from multidomain_weighting import simplex_grid, stratified_folds

    with pytest.raises(ValueError, match="step"):
        simplex_grid(3, 0.3)
    with pytest.raises(ValueError, match="positive"):
        simplex_grid(3, 0.0)
    with pytest.raises(ValueError, match="one domain"):
        simplex_grid(0, 0.5)
    with pytest.raises(ValueError, match="both classes"):
        stratified_folds(np.ones(10), n_splits=2, seed=0)
    with pytest.raises(ValueError, match="at least 2"):
        stratified_folds(np.array([0, 0, 1, 1]), n_splits=1, seed=0)
    with pytest.raises(ValueError, match="smaller"):
        stratified_folds(np.array([0, 0, 0, 1]), n_splits=2, seed=0)


def test_anchor_truth_and_subtree_columns_follow_frontiers_and_layout_order():
    """Catches ancestral-closure truth or unordered node-to-column mapping."""
    from spark_vi.models.topic.dag_placement import DagLayout
    from multidomain_weighting import anchor_truth, subtree_columns

    lay = DagLayout({1: [0], 2: [0], 3: [1]}, n_bg=1, tpn=1)
    subtree = {1, 3, 999}
    frontiers = [[3], [2], [1, 2], [], [999]]

    assert np.array_equal(anchor_truth(frontiers, subtree), [1, 0, 1, 0, 1])
    assert np.array_equal(
        subtree_columns(lay, subtree),
        np.array([lay.nodes.index(1), lay.nodes.index(3)]),
    )


def test_domain_nodes_are_combined_before_the_subtree_maximum():
    """Catches the bug that maximizes each domain before combining domains."""
    from multidomain_weighting import _weighted_subtree_score

    matrices = {
        "condition": np.array([[10.0, 0.0], [2.0, 8.0], [5.0, 1.0]]),
        "drug": np.array([[0.0, 10.0], [8.0, 2.0], [1.0, 5.0]]),
    }
    weights = np.array([0.25, 0.75])

    got = _weighted_subtree_score(matrices, weights, np.array([0, 1]))
    expected = np.max(
        0.25 * matrices["condition"] + 0.75 * matrices["drug"], axis=1
    )
    wrong = (
        0.25 * np.max(matrices["condition"], axis=1)
        + 0.75 * np.max(matrices["drug"], axis=1)
    )

    assert np.array_equal(got, expected)
    assert not np.array_equal(got, wrong)


def test_fold_local_backgrounds_and_scales_ignore_outer_test_bow_rows():
    """Catches background or score-scale fitting on validation/test documents."""
    from spark_vi.models.topic.dag_placement import DagLayout
    from multidomain_weighting import (
        _backgrounds_from_rows,
        _raw_scores_for_rows,
        _scales_from_matrices,
        subtree_columns,
    )

    lay = DagLayout({1: [0], 2: [0]}, n_bg=1, tpn=1)
    lam_dict = {
        "condition": np.array(
            [[1.0, 1.0, 1.0], [8.0, 1.0, 1.0], [1.0, 8.0, 1.0]]
        ),
        "drug": np.array(
            [[1.0, 1.0, 1.0], [1.0, 8.0, 1.0], [8.0, 1.0, 1.0]]
        ),
    }
    bows = {
        "condition": np.array(
            [
                [3.0, 0.0, 1.0],
                [0.0, 2.0, 1.0],
                [2.0, 1.0, 0.0],
                [0.0, 1.0, 3.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        "drug": np.array(
            [
                [0.0, 2.0, 1.0],
                [2.0, 0.0, 1.0],
                [1.0, 3.0, 0.0],
                [1.0, 0.0, 3.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ]
        ),
    }
    train_rows = np.array([0, 1, 2, 3])
    test_rows = np.array([4, 5])
    columns = subtree_columns(lay, {1, 2})

    backgrounds = _backgrounds_from_rows(bows, train_rows)
    train_matrices = _raw_scores_for_rows(
        bows, lam_dict, lay, train_rows, backgrounds
    )
    scales = _scales_from_matrices(train_matrices, columns)

    altered = {key: value.copy() for key, value in bows.items()}
    for matrix in altered.values():
        matrix[test_rows] = np.array([[1000.0, 1.0, 1.0], [1.0, 1000.0, 1.0]])
    altered_backgrounds = _backgrounds_from_rows(altered, train_rows)
    altered_train_matrices = _raw_scores_for_rows(
        altered, lam_dict, lay, train_rows, altered_backgrounds
    )
    altered_scales = _scales_from_matrices(altered_train_matrices, columns)

    for domain in sorted(bows):
        assert np.array_equal(backgrounds[domain], altered_backgrounds[domain])
        assert np.array_equal(
            train_matrices[domain], altered_train_matrices[domain]
        )
        assert scales[domain] == altered_scales[domain]


def test_scale_is_estimated_from_training_subtree_score_not_other_nodes():
    """Catches whole-matrix std fitting when only the anchor subtree is relevant."""
    from spark_vi.models.topic.dag_placement import domain_score_scale
    from multidomain_weighting import _scales_from_matrices

    matrix = np.array(
        [
            [1.0, 0.0, 1000.0],
            [2.0, 0.0, -1000.0],
            [4.0, 1.0, 500.0],
            [8.0, 3.0, -500.0],
        ]
    )
    expected = domain_score_scale(np.max(matrix[:, [0, 1]], axis=1))
    got = _scales_from_matrices({"condition": matrix}, np.array([0, 1]))

    assert got == {"condition": expected}
    assert got["condition"] != domain_score_scale(matrix)


def test_max_subtree_score_rejects_empty_columns():
    """Catches silently fabricated scores for an unscoreable anchor."""
    from multidomain_weighting import max_subtree_score

    with pytest.raises(ValueError, match="subtree"):
        max_subtree_score(np.ones((3, 2)), np.array([], dtype=int))


def test_continuous_selector_gives_high_variance_noise_zero_weight():
    """Catches a selector that rewards a noisy domain despite worse pooled AP."""
    from multidomain_weighting import select_simplex_weights, simplex_grid

    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    matrices = {
        "condition": y[:, None].astype(float),
        "observation": np.array(
            [0.0, 0.1, -0.1, 0.0, 100.0, 80.0, 60.0, 40.0, 20.0, 10.0, 5.0, 2.0]
        )[:, None],
    }

    weights = select_simplex_weights(
        matrices, y, np.array([0]), grid=simplex_grid(2, 0.25)
    )

    assert np.array_equal(weights, np.array([1.0, 0.0]))


def test_continuous_selector_uses_helpful_drug_and_improves_ap():
    """Catches selecting only condition when drug finds independent positives."""
    from spark_vi.models.topic.dag_placement import _average_precision
    from multidomain_weighting import (
        _weighted_subtree_score,
        select_simplex_weights,
        simplex_grid,
    )

    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    matrices = {
        "condition": np.array(
            [4.0, 4.0, 0.0, 0.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
        )[:, None],
        "drug": np.array(
            [0.0, 0.0, 4.0, 4.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
        )[:, None],
    }
    grid = simplex_grid(2, 0.5)

    weights = select_simplex_weights(matrices, y, np.array([0]), grid=grid)
    selected_score = _weighted_subtree_score(matrices, weights, np.array([0]))
    condition_score = matrices["condition"][:, 0]

    assert weights[1] > 0.0
    assert _average_precision(selected_score, y) > _average_precision(
        condition_score, y
    )


def test_continuous_selector_can_choose_opposite_domains_by_disease():
    """Catches reusing one anchor's learned weights for a different disease."""
    from multidomain_weighting import select_simplex_weights, simplex_grid

    y_a = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    y_b = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
    noise_a = np.array([0.0, 0.0, 0.0, 0.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    noise_b = np.array([8.0, 7.0, 6.0, 5.0, 0.0, 0.0, 0.0, 0.0, 4.0, 3.0, 2.0, 1.0])
    matrices = {
        "condition": np.column_stack((y_a, noise_b)),
        "drug": np.column_stack((noise_a, y_b)),
    }
    grid = simplex_grid(2, 0.5)

    weights_a = select_simplex_weights(matrices, y_a, np.array([0]), grid=grid)
    weights_b = select_simplex_weights(matrices, y_b, np.array([1]), grid=grid)

    assert np.array_equal(weights_a, np.array([1.0, 0.0]))
    assert np.array_equal(weights_b, np.array([0.0, 1.0]))


def test_continuous_selector_ties_favor_uniform_then_lexicographic():
    """Catches grid-order tie resolution instead of the documented stable policy."""
    from multidomain_weighting import select_simplex_weights, simplex_grid

    y = np.array([1, 1, 0, 0, 0, 0])
    two_domains = {"a": np.zeros((6, 1)), "b": np.zeros((6, 1))}
    three_domains = {
        "a": np.zeros((6, 1)),
        "b": np.zeros((6, 1)),
        "c": np.zeros((6, 1)),
    }

    assert np.array_equal(
        select_simplex_weights(
            two_domains, y, np.array([0]), grid=simplex_grid(2, 0.5)
        ),
        np.array([0.5, 0.5]),
    )
    assert np.array_equal(
        select_simplex_weights(
            three_domains, y, np.array([0]), grid=simplex_grid(3, 0.5)
        ),
        np.array([0.0, 0.5, 0.5]),
    )
    nearly_uniform = np.array(
        [[0.5 - 1e-7, 0.5 + 1e-7], [0.5, 0.5]]
    )
    assert np.array_equal(
        select_simplex_weights(
            two_domains, y, np.array([0]), grid=nearly_uniform
        ),
        np.array([0.5, 0.5]),
    )


def test_continuous_selector_rejects_non_simplex_candidates():
    """Catches direct callers bypassing nonnegative, sum-to-one grid semantics."""
    from multidomain_weighting import select_simplex_weights

    matrices = {"a": np.zeros((4, 1)), "b": np.zeros((4, 1))}
    y = np.array([1, 1, 0, 0])
    with pytest.raises(ValueError, match="simplex"):
        select_simplex_weights(
            matrices,
            y,
            np.array([0]),
            grid=np.array([[1.0, -0.1]]),
        )
    with pytest.raises(ValueError, match="simplex"):
        select_simplex_weights(
            matrices,
            y,
            np.array([0]),
            grid=np.array([[0.4, 0.4]]),
        )


def test_model_derived_weights_do_not_consume_case_labels():
    """Catches replacing fixed model evidence with label-selected weights."""
    from multidomain_weighting import evaluate_model_candidate

    matrices = {
        0: np.array([[4.0], [3.0], [2.0], [1.0]]),
        1: np.zeros((4, 1)),
    }
    weights = np.array([1.0, 0.0])
    scales = {0: 1.0, 1: 1.0}
    y = np.array([1, 1, 0, 0])

    got1 = evaluate_model_candidate(
        matrices,
        y,
        np.array([0]),
        weights=weights,
        scales=scales,
    )
    got2 = evaluate_model_candidate(
        matrices,
        1 - y,
        np.array([0]),
        weights=weights,
        scales=scales,
    )

    assert got1["weights"] == got2["weights"] == [1.0, 0.0]
    assert got1["ap"] != got2["ap"]
    json.dumps(got1, allow_nan=False)
    json.dumps(got2, allow_nan=False)


def test_model_derived_anchor_weights_apply_to_each_descendant_before_max():
    """Catches descendant-specific weights or per-domain maxima before combination."""
    from multidomain_weighting import evaluate_model_candidate

    matrices = {
        "condition": np.array([[10.0, 0.0], [6.0, 3.0], [4.0, 2.0]]),
        "drug": np.array([[0.0, 10.0], [2.0, 7.0], [5.0, 1.0]]),
    }
    weights = np.array([0.8, 0.2])
    scales = {"condition": 2.0, "drug": 1.0}
    y = np.array([1, 0, 1])

    got = evaluate_model_candidate(
        matrices,
        y,
        np.array([0, 1]),
        weights=weights,
        scales=scales,
    )
    expected_scores = np.max(
        0.8 * matrices["condition"] / 2.0 + 0.2 * matrices["drug"],
        axis=1,
    )
    wrong_descendant_weights = np.max(
        np.column_stack(
            (
                0.8 * matrices["condition"][:, 0] / 2.0
                + 0.2 * matrices["drug"][:, 0],
                0.2 * matrices["condition"][:, 1] / 2.0
                + 0.8 * matrices["drug"][:, 1],
            )
        ),
        axis=1,
    )

    assert got["scores"] == [float(score) for score in expected_scores]
    assert not np.array_equal(expected_scores, wrong_descendant_weights)


def test_model_derived_agreement_is_tie_stable_constant_safe_and_json_safe():
    """Catches unstable top sets or NaN Spearman values from constant rankings."""
    from multidomain_weighting import _agreement_diagnostics

    tied = _agreement_diagnostics(
        np.array([1.0, 1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 1.0, 0.0]),
        candidate_weights=np.array([0.8, 0.2]),
        median_supervised_weights=np.array([0.6, 0.4]),
        n_positive=1,
    )
    constant = _agreement_diagnostics(
        np.ones(4),
        np.array([4.0, 3.0, 2.0, 1.0]),
        candidate_weights=np.array([0.2, 0.8]),
        median_supervised_weights=np.array([0.6, 0.4]),
        n_positive=1,
    )

    # Stable row order breaks the top-score ties: tied candidate -> row 0,
    # tied ceiling -> row 1, so the top-1 sets are disjoint.
    assert tied["top_set_jaccard_with_continuous"] == 0.0
    assert isinstance(tied["spearman_with_continuous"], float)
    assert tied["same_domain_order_as_median_supervised"] is True
    assert constant["spearman_with_continuous"] is None
    assert constant["same_domain_order_as_median_supervised"] is False
    json.dumps(tied, allow_nan=False)
    json.dumps(constant, allow_nan=False)


def test_discrete_policies_cover_every_subset_and_normalization_in_stable_order():
    """Catches omitted subsets/rules or nondeterministic policy tie order."""
    from multidomain_weighting import discrete_policies

    got = discrete_policies(("observation", "condition", "drug"))
    expected = (
        ("only:condition|none", ("condition",), "none"),
        ("only:condition|std", ("condition",), "std"),
        ("only:condition|length", ("condition",), "length"),
        ("only:condition|length+std", ("condition",), "length+std"),
        ("only:drug|none", ("drug",), "none"),
        ("only:drug|std", ("drug",), "std"),
        ("only:drug|length", ("drug",), "length"),
        ("only:drug|length+std", ("drug",), "length+std"),
        ("only:observation|none", ("observation",), "none"),
        ("only:observation|std", ("observation",), "std"),
        ("only:observation|length", ("observation",), "length"),
        ("only:observation|length+std", ("observation",), "length+std"),
        ("drop:observation|none", ("condition", "drug"), "none"),
        ("drop:observation|std", ("condition", "drug"), "std"),
        ("drop:observation|length", ("condition", "drug"), "length"),
        ("drop:observation|length+std", ("condition", "drug"), "length+std"),
        ("drop:drug|none", ("condition", "observation"), "none"),
        ("drop:drug|std", ("condition", "observation"), "std"),
        ("drop:drug|length", ("condition", "observation"), "length"),
        ("drop:drug|length+std", ("condition", "observation"), "length+std"),
        ("drop:condition|none", ("drug", "observation"), "none"),
        ("drop:condition|std", ("drug", "observation"), "std"),
        ("drop:condition|length", ("drug", "observation"), "length"),
        ("drop:condition|length+std", ("drug", "observation"), "length+std"),
        ("all|none", ("condition", "drug", "observation"), "none"),
        ("all|std", ("condition", "drug", "observation"), "std"),
        ("all|length", ("condition", "drug", "observation"), "length"),
        (
            "all|length+std",
            ("condition", "drug", "observation"),
            "length+std",
        ),
    )

    assert got == expected


def _fold_local_normalization_fixture(seed):
    """Planted multi-domain scores with row-varying utilization."""
    from scipy import sparse as sp
    from spark_vi.models.topic.dag_placement import DagLayout
    from multidomain_weighting import subtree_columns

    rng = np.random.default_rng(seed)
    lay = DagLayout({1: [0], 2: [1]}, n_bg=1, tpn=1)
    y = np.array([1] * 8 + [0] * 16)
    bows = {
        domain: sp.csr_matrix(rng.poisson(1.5, size=(24, 5)).astype(float))
        for domain in range(3)
    }
    for domain, matrix in bows.items():
        dense = matrix.toarray()
        dense[:8, (domain + 1) % 5] += rng.integers(0, 5, size=8)
        dense[:, 0] *= rng.integers(1, 8, size=24)
        bows[domain] = sp.csr_matrix(dense)
    lam_dict = {
        domain: rng.gamma(2.0, 2.0, size=(lay.K, 5)) for domain in range(3)
    }
    return bows, lam_dict, lay, y, subtree_columns(lay, {1, 2})


@pytest.mark.parametrize(
    ("fixture_seed", "expected_policy", "expected_weights"),
    [
        (0, "only:1|none", [0.0, 1.0, 0.0]),
        (1, "only:2|std", [0.0, 0.0, 1.0]),
        (8, "drop:2|length", [0.5, 0.5, 0.0]),
        (22, "only:2|length+std", [0.0, 0.0, 1.0]),
    ],
)
def test_inner_selection_distinguishes_all_fold_local_normalizations(
    fixture_seed,
    expected_policy,
    expected_weights,
):
    """Catches collapsing length modes into raw none/std transformations."""
    from multidomain_weighting import _inner_oof_selection, simplex_grid

    bows, lam_dict, lay, y, columns = _fold_local_normalization_fixture(
        fixture_seed
    )
    policy, weights = _inner_oof_selection(
        bows,
        lam_dict,
        lay,
        y,
        columns,
        np.arange(len(y)),
        inner_folds=3,
        grid=simplex_grid(3, 0.5),
        seed=97,
    )

    assert policy[0] == expected_policy
    assert np.array_equal(weights, np.asarray(expected_weights))


def test_discrete_ap_ties_choose_first_policy_in_tuple_order():
    """Catches reversed exact/tolerance tie resolution in discrete selection."""
    from multidomain_weighting import (
        _select_discrete_policy,
        _select_discrete_policy_from_ap,
    )

    policies = (
        ("first", (0,), "none"),
        ("second", (1,), "none"),
    )
    y = np.array([1, 0, 1, 0])
    identical_scores = np.array([4.0, 3.0, 2.0, 1.0])
    selected = _select_discrete_policy(
        policies,
        {
            "first": identical_scores,
            "second": identical_scores.copy(),
        },
        y,
    )

    assert selected == policies[0]
    assert _select_discrete_policy_from_ap(
        policies, [0.5, 0.5 + 5e-13]
    ) == policies[0]
    assert _select_discrete_policy_from_ap(
        policies, [0.5, 0.5 + 2e-12]
    ) == policies[1]


@pytest.mark.parametrize(
    ("fixture_seed", "fold_number", "expected_policy"),
    [
        (3, 2, "only:0|length"),
        (7, 1, "only:1|length+std"),
    ],
)
def test_outer_fold_applies_frozen_length_modes(
    fixture_seed,
    fold_number,
    expected_policy,
):
    """Catches raw outer scoring after a length policy wins inner OOF AP."""
    from spark_vi.models.topic.dag_placement import (
        combine_domain_score_matrices,
        domain_score_scale,
        lr_background,
        lr_domain_score_matrices,
    )
    from multidomain_weighting import (
        _evaluate_outer_fold,
        discrete_policies,
        max_subtree_score,
        simplex_grid,
        stratified_folds,
    )

    bows, lam_dict, lay, y, columns = _fold_local_normalization_fixture(
        fixture_seed
    )
    outer_train, outer_test = stratified_folds(
        y, n_splits=3, seed=41
    )[fold_number]
    evaluated = _evaluate_outer_fold(
        bows=bows,
        lam_dict=lam_dict,
        lay=lay,
        y=y,
        columns=columns,
        outer_train=outer_train,
        outer_test=outer_test,
        inner_folds=2,
        grid=simplex_grid(3, 0.5),
        inner_seed=101 + fold_number,
    )

    assert evaluated["discrete_policy"] == expected_policy
    policy = next(
        item
        for item in discrete_policies(tuple(sorted(bows)))
        if item[0] == expected_policy
    )
    _, subset, normalization = policy
    backgrounds = {
        domain: lr_background(bows[domain][outer_train])
        for domain in sorted(bows)
    }
    length_train = lr_domain_score_matrices(
        {domain: bows[domain][outer_train] for domain in sorted(bows)},
        lam_dict,
        lay,
        alpha=float("inf"),
        backgrounds=backgrounds,
        normalize="length",
    )
    length_test = lr_domain_score_matrices(
        {domain: bows[domain][outer_test] for domain in sorted(bows)},
        lam_dict,
        lay,
        alpha=float("inf"),
        backgrounds=backgrounds,
        normalize="length",
    )
    frozen_scales = (
        {
            domain: domain_score_scale(
                max_subtree_score(length_train[domain], columns)
            )
            for domain in subset
        }
        if normalization == "length+std"
        else None
    )
    expected = max_subtree_score(
        combine_domain_score_matrices(
            {domain: length_test[domain] for domain in subset},
            scales=frozen_scales,
        ),
        columns,
    )
    raw_test = lr_domain_score_matrices(
        {domain: bows[domain][outer_test] for domain in sorted(bows)},
        lam_dict,
        lay,
        alpha=float("inf"),
        backgrounds=backgrounds,
        normalize=None,
    )
    wrong_raw = max_subtree_score(
        combine_domain_score_matrices(
            {domain: raw_test[domain] for domain in subset}
        ),
        columns,
    )

    assert np.allclose(evaluated["scores"]["discrete"], expected)
    assert not np.allclose(evaluated["scores"]["discrete"], wrong_raw)


def test_inner_selection_uses_pooled_oof_ap_not_in_sample_or_mean_fold():
    """Catches replacing pooled inner OOF AP with in-sample or mean-fold AP."""
    from scipy import sparse as sp
    from spark_vi.models.topic.dag_placement import DagLayout
    from multidomain_weighting import (
        _inner_oof_selection,
        simplex_grid,
        subtree_columns,
    )

    rng = np.random.default_rng(5)
    lay = DagLayout({1: [0], 2: [1]}, n_bg=1, tpn=1)
    y = np.array([1] * 8 + [0] * 16)
    bows = {
        domain: sp.csr_matrix(rng.poisson(1.5, size=(24, 5)).astype(float))
        for domain in range(3)
    }
    for domain, matrix in bows.items():
        dense = matrix.toarray()
        dense[:8, (domain + 1) % 5] += rng.integers(0, 4, size=8)
        bows[domain] = sp.csr_matrix(dense)
    lam_dict = {
        domain: rng.gamma(2.0, 2.0, size=(lay.K, 5)) for domain in range(3)
    }

    policy, weights = _inner_oof_selection(
        bows,
        lam_dict,
        lay,
        y,
        subtree_columns(lay, {1, 2}),
        np.arange(len(y)),
        inner_folds=3,
        grid=simplex_grid(3, 0.5),
        seed=97,
    )

    # Hand-pinned from pooled predictions: mean-fold and in-sample AP both
    # choose only:2|none with [0, 0, 1] on this planted fixture.
    assert policy[0] == "drop:1|std"
    assert np.array_equal(weights, np.array([0.5, 0.0, 0.5]))


def _nested_bow_fixture():
    """Sixty sparse BOW rows with 15 subtree positives and three score domains."""
    from scipy import sparse as sp
    from spark_vi.models.topic.dag_placement import DagLayout

    n_docs = 60
    y = np.zeros(n_docs, dtype=int)
    y[:15] = 1
    condition = np.zeros((n_docs, 4), dtype=float)
    drug = np.zeros((n_docs, 4), dtype=float)
    observation = np.zeros((n_docs, 4), dtype=float)
    rng = np.random.default_rng(20260729)
    for row in range(n_docs):
        if y[row]:
            condition[row] = [3.0 + row % 3, 0.0, 0.0, 1.0]
            drug[row] = [0.0, float(row % 2), 2.0 + row % 3, 1.0]
        else:
            condition[row] = [0.0, 3.0 + row % 5, 0.0, 1.0]
            drug[row] = [0.0, 2.0 + row % 4, float(row % 3 == 0), 1.0]
        observation[row] = rng.integers(0, 4, size=4)

    bows = {
        "condition": sp.csr_matrix(condition),
        "drug": sp.csr_matrix(drug),
        "observation": sp.csr_matrix(observation),
    }
    lay = DagLayout({1: [0], 2: [1]}, n_bg=1, tpn=1)
    lam_dict = {
        "condition": np.array(
            [[1.0, 1.0, 1.0, 1.0], [12.0, 1.0, 1.0, 1.0], [1.0, 1.0, 8.0, 1.0]]
        ),
        "drug": np.array(
            [[1.0, 1.0, 1.0, 1.0], [1.0, 4.0, 2.0, 1.0], [1.0, 1.0, 12.0, 1.0]]
        ),
        "observation": np.array(
            [[1.0, 1.0, 1.0, 1.0], [2.0, 1.0, 3.0, 1.0], [1.0, 3.0, 1.0, 2.0]]
        ),
    }
    frontiers = [[2] if label else [] for label in y]
    parent_int = {1: [0], 2: [1]}
    return bows, lam_dict, lay, frontiers, parent_int, y


def test_model_derived_strategies_use_the_anchor_reliability_row_once_per_run(
    monkeypatch,
):
    """Catches recomputation by fold or reliability aggregation over descendants."""
    from spark_vi.models.topic import dag_placement
    from multidomain_weighting import (
        _backgrounds_from_rows,
        _raw_scores_for_rows,
        _scales_from_matrices,
        _strategy_metrics,
        evaluate_anchor_nested,
        stratified_folds,
        subtree_columns,
    )

    bows, lam_dict, lay, frontiers, parent_int, y = _nested_bow_fixture()
    domains = tuple(sorted(bows))
    reliability = dag_placement.DomainReliability(
        domains,
        distinctiveness=np.array([[9.0, 1.0, 0.0], [0.0, 1.0, 9.0]]),
        ownership=np.array([[0.0, 2.0, 8.0], [8.0, 1.0, 1.0]]),
        viability=np.ones((2, 3)),
    )
    calls = 0

    def fixed_reliability(candidate_lam, candidate_layout):
        nonlocal calls
        calls += 1
        assert candidate_lam is lam_dict
        assert candidate_layout is lay
        return reliability

    monkeypatch.setattr(
        dag_placement,
        "domain_reliability",
        fixed_reliability,
    )
    result = evaluate_anchor_nested(
        bows,
        lam_dict,
        lay,
        frontiers,
        anchor=1,
        parent_int=parent_int,
        outer_folds=3,
        inner_folds=2,
        repeats=2,
        grid_step=0.5,
        seed=41,
    )

    expected_weights = {
        "model:distinctiveness": np.array([0.9, 0.1, 0.0]),
        "model:ownership": np.array([0.0, 0.2, 0.8]),
        "model:product": np.array([0.0, 1.0, 0.0]),
    }
    columns = subtree_columns(lay, {1, 2})
    expected_metrics = {}
    for strategy, weights in expected_weights.items():
        pooled = np.empty(len(y), dtype=float)
        for outer_train, outer_test in stratified_folds(y, n_splits=3, seed=41):
            backgrounds = _backgrounds_from_rows(bows, outer_train)
            train_matrices = _raw_scores_for_rows(
                bows,
                lam_dict,
                lay,
                outer_train,
                backgrounds,
            )
            test_matrices = _raw_scores_for_rows(
                bows,
                lam_dict,
                lay,
                outer_test,
                backgrounds,
            )
            scales = _scales_from_matrices(train_matrices, columns)
            combined = sum(
                weights[column] * test_matrices[domain] / scales[domain]
                for column, domain in enumerate(domains)
            )
            pooled[outer_test] = np.max(combined[:, columns], axis=1)
        expected_metrics[strategy] = _strategy_metrics(pooled, y)

    assert calls == 1
    for repeat in result["repeats"]:
        assert set(repeat["strategies"]) == {
            "fixed:condition_drug",
            "discrete",
            "continuous",
            "model:distinctiveness",
            "model:ownership",
            "model:product",
        }
        assert set(repeat["agreements"]) == set(expected_weights)
        for strategy, weights in expected_weights.items():
            agreement = repeat["agreements"][strategy]
            assert agreement["weights"] == [float(weight) for weight in weights]
            assert set(agreement) == {
                "weights",
                "spearman_with_continuous",
                "top_set_jaccard_with_continuous",
                "same_domain_order_as_median_supervised",
            }
            assert (
                agreement["spearman_with_continuous"] is None
                or isinstance(agreement["spearman_with_continuous"], float)
            )
            assert isinstance(
                agreement["top_set_jaccard_with_continuous"],
                float,
            )
            assert isinstance(
                agreement["same_domain_order_as_median_supervised"],
                bool,
            )
    for strategy, expected in expected_metrics.items():
        assert result["repeats"][0]["strategies"][strategy] == expected
    json.dumps(result, allow_nan=False)


def test_outer_fold_scores_apply_training_backgrounds_and_scales():
    """Catches deriving outer-test transformations from the outer-test batch."""
    from multidomain_weighting import (
        _backgrounds_from_rows,
        _discrete_policy_score,
        _evaluate_outer_fold,
        _raw_scores_for_rows,
        _scales_from_matrices,
        _weighted_subtree_score,
        discrete_policies,
        simplex_grid,
        stratified_folds,
        subtree_columns,
    )

    bows, lam_dict, lay, _, _, y = _nested_bow_fixture()
    outer_train, outer_test = stratified_folds(y, n_splits=5, seed=19)[0]
    columns = subtree_columns(lay, {1, 2})
    evaluated = _evaluate_outer_fold(
        bows=bows,
        lam_dict=lam_dict,
        lay=lay,
        y=y,
        columns=columns,
        outer_train=outer_train,
        outer_test=outer_test,
        inner_folds=3,
        grid=simplex_grid(3, 0.5),
        inner_seed=101,
    )

    backgrounds = _backgrounds_from_rows(bows, outer_train)
    raw_train = _raw_scores_for_rows(
        bows, lam_dict, lay, outer_train, backgrounds
    )
    raw_test = _raw_scores_for_rows(
        bows, lam_dict, lay, outer_test, backgrounds
    )
    raw_scales = _scales_from_matrices(raw_train, columns)
    expected_fixed = _weighted_subtree_score(
        {domain: raw_test[domain] for domain in ("condition", "drug")},
        np.ones(2),
        columns,
    )
    expected_continuous = _weighted_subtree_score(
        raw_test,
        evaluated["continuous_weights"],
        columns,
        scales=raw_scales,
    )

    selected_policy = next(
        policy
        for policy in discrete_policies(tuple(sorted(bows)))
        if policy[0] == evaluated["discrete_policy"]
    )
    if selected_policy[2] in ("length", "length+std"):
        selected_train = _raw_scores_for_rows(
            bows,
            lam_dict,
            lay,
            outer_train,
            backgrounds,
            length=True,
        )
        selected_test = _raw_scores_for_rows(
            bows,
            lam_dict,
            lay,
            outer_test,
            backgrounds,
            length=True,
        )
        selected_scales = _scales_from_matrices(selected_train, columns)
    else:
        selected_test, selected_scales = raw_test, raw_scales
    expected_discrete = _discrete_policy_score(
        selected_test,
        selected_policy,
        columns,
        scales=selected_scales,
    )

    assert np.array_equal(
        evaluated["scores"]["fixed:condition_drug"], expected_fixed
    )
    assert np.array_equal(evaluated["scores"]["discrete"], expected_discrete)
    assert np.array_equal(
        evaluated["scores"]["continuous"], expected_continuous
    )


def test_outer_fold_selection_ignores_test_labels_and_test_bow_rows():
    """Catches fitting backgrounds, scales, policies, or weights on outer test data."""
    from multidomain_weighting import (
        _evaluate_outer_fold,
        simplex_grid,
        stratified_folds,
        subtree_columns,
    )

    bows, lam_dict, lay, _, _, y = _nested_bow_fixture()
    outer_train, outer_test = stratified_folds(y, n_splits=5, seed=19)[0]
    columns = subtree_columns(lay, {1, 2})
    kwargs = dict(
        bows=bows,
        lam_dict=lam_dict,
        lay=lay,
        y=y,
        columns=columns,
        outer_train=outer_train,
        outer_test=outer_test,
        inner_folds=3,
        grid=simplex_grid(3, 0.5),
        inner_seed=101,
    )
    baseline = _evaluate_outer_fold(**kwargs)

    altered_bows = {}
    for domain, matrix in bows.items():
        dense = matrix.toarray()
        dense[outer_test] = np.full((len(outer_test), matrix.shape[1]), 1000.0)
        altered_bows[domain] = type(matrix)(dense)
    altered_y = y.copy()
    altered_y[outer_test[0]] = 1 - altered_y[outer_test[0]]
    altered = _evaluate_outer_fold(
        **{**kwargs, "bows": altered_bows, "y": altered_y}
    )
    inverted_test_y = y.copy()
    inverted_test_y[outer_test] = 1 - inverted_test_y[outer_test]
    inverted = _evaluate_outer_fold(**{**kwargs, "y": inverted_test_y})

    assert baseline["discrete_policy"] == altered["discrete_policy"]
    assert np.array_equal(
        baseline["continuous_weights"], altered["continuous_weights"]
    )
    assert baseline["discrete_policy"] == inverted["discrete_policy"]
    assert np.array_equal(
        baseline["continuous_weights"], inverted["continuous_weights"]
    )
    for strategy in ("fixed:condition_drug", "discrete", "continuous"):
        assert len(baseline["scores"][strategy]) == len(outer_test)
        assert len(altered["scores"][strategy]) == len(outer_test)


def test_nested_metrics_equal_independently_reconstructed_aligned_oof_scores():
    """Catches zero, fold-mean, short, or row-misaligned public metrics."""
    from spark_vi.models.topic.dag_placement import (
        _average_precision,
        _precision_at_recall,
    )
    from multidomain_weighting import (
        _evaluate_outer_fold,
        evaluate_anchor_nested,
        simplex_grid,
        stratified_folds,
    )

    bows, lam_dict, lay, y, columns = _fold_local_normalization_fixture(0)
    frontiers = [[2] if label else [] for label in y]
    result = evaluate_anchor_nested(
        bows,
        lam_dict,
        lay,
        frontiers,
        anchor=1,
        parent_int={1: [0], 2: [1]},
        outer_folds=3,
        inner_folds=2,
        repeats=1,
        grid_step=0.5,
        seed=41,
    )
    strategies = ("fixed:condition_drug", "discrete", "continuous")
    pooled = {
        strategy: np.full(len(y), np.nan, dtype=float)
        for strategy in strategies
    }
    fold_ap = {strategy: [] for strategy in strategies}
    outer_partitions = stratified_folds(y, n_splits=3, seed=41)
    for fold_number, (outer_train, outer_test) in enumerate(outer_partitions):
        fold_result = _evaluate_outer_fold(
            bows=bows,
            lam_dict=lam_dict,
            lay=lay,
            y=y,
            columns=columns,
            outer_train=outer_train,
            outer_test=outer_test,
            inner_folds=2,
            grid=simplex_grid(3, 0.5),
            inner_seed=100_041 + fold_number,
        )
        assert result["repeats"][0]["folds"][fold_number]["test_rows"] == [
            int(row) for row in outer_test
        ]
        for strategy in strategies:
            fold_scores = fold_result["scores"][strategy]
            pooled[strategy][outer_test] = fold_scores
            assert np.array_equal(
                pooled[strategy][outer_test],
                fold_scores,
            )
            fold_ap[strategy].append(
                _average_precision(fold_scores, y[outer_test])
            )

    recalls = (0.10, 0.25, 0.50, 0.80)
    public_strategies = result["repeats"][0]["strategies"]
    for strategy in strategies:
        assert len(pooled[strategy]) == result["n_docs"] == len(y)
        assert np.all(np.isfinite(pooled[strategy]))
        precision = _precision_at_recall(pooled[strategy], y, recalls)
        expected = {
            "ap": float(_average_precision(pooled[strategy], y)),
            "precision_at_recall": {
                format(recall, ".12g"): float(precision[recall])
                for recall in recalls
            },
        }
        assert public_strategies[strategy] == expected
        assert abs(
            expected["ap"] - float(np.mean(fold_ap[strategy]))
        ) > 1e-3


def test_nested_evaluation_is_shared_oof_deterministic_and_json_safe():
    """Catches in-sample reporting, divergent folds, nondeterminism, or NumPy JSON."""
    from multidomain_weighting import evaluate_anchor_nested, stratified_folds

    bows, lam_dict, lay, frontiers, parent_int, y = _nested_bow_fixture()
    kwargs = dict(
        bows=bows,
        lam_dict=lam_dict,
        lay=lay,
        frontiers=frontiers,
        anchor=1,
        parent_int=parent_int,
        outer_folds=5,
        inner_folds=3,
        repeats=2,
        grid_step=0.5,
        seed=31,
    )
    result = evaluate_anchor_nested(**kwargs)
    again = evaluate_anchor_nested(**kwargs)

    assert result == again
    assert set(result) == {
        "anchor",
        "n_docs",
        "n_positive",
        "prevalence",
        "repeats",
    }
    assert result["anchor"] == 1
    assert result["n_docs"] == 60
    assert result["n_positive"] == 15
    assert result["prevalence"] == 0.25
    assert len(result["repeats"]) == 2
    for repeat_number, repeat in enumerate(result["repeats"]):
        assert repeat["repeat"] == repeat_number
        assert set(repeat) == {"repeat", "strategies", "agreements", "folds"}
        assert set(repeat["strategies"]) == {
            "fixed:condition_drug",
            "discrete",
            "continuous",
            "model:distinctiveness",
            "model:ownership",
            "model:product",
        }
        for metrics in repeat["strategies"].values():
            assert set(metrics) == {"ap", "precision_at_recall"}
            assert isinstance(metrics["ap"], float)
            assert set(metrics["precision_at_recall"]) == {
                "0.1",
                "0.25",
                "0.5",
                "0.8",
            }
            assert all(
                isinstance(value, float)
                for value in metrics["precision_at_recall"].values()
            )

        held_out = np.concatenate(
            [np.asarray(fold["test_rows"], dtype=int) for fold in repeat["folds"]]
        )
        assert len(held_out) == result["n_docs"]
        assert np.array_equal(np.sort(held_out), np.arange(result["n_docs"]))
        expected_partitions = stratified_folds(
            y, n_splits=5, seed=31 + repeat_number
        )
        for fold_number, (fold, expected_partition) in enumerate(
            zip(repeat["folds"], expected_partitions)
        ):
            expected_train, expected_test = expected_partition
            assert fold["fold"] == fold_number
            assert np.array_equal(fold["test_rows"], expected_test)
            assert not np.intersect1d(expected_train, expected_test).size
            assert isinstance(fold["discrete_policy"], str)
            assert len(fold["continuous_weights"]) == 3
            assert all(
                isinstance(value, float) for value in fold["continuous_weights"]
            )
            assert abs(sum(fold["continuous_weights"]) - 1.0) < 1e-12

    json.dumps(result, allow_nan=False)
