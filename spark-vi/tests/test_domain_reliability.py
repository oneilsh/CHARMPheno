import numpy as np
import pytest


def test_distinctiveness_is_zero_for_background_and_positive_for_marker():
    """Changing a node distribution away from its fitted background is detectable."""
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


def test_ownership_is_higher_for_a_node_marker_than_a_background_shared_code():
    """A marker routed toward the node must outweigh a code shared with background."""
    from spark_vi.models.topic.dag_placement import DagLayout, domain_reliability

    lay = DagLayout({1: [0]}, n_bg=1, tpn=1)
    lam = {
        0: np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]),
        1: np.array([[50.0, 25.0, 25.0], [50.0, 49.0, 1.0]]),
    }

    rel = domain_reliability(lam, lay)

    # In domain 0 every word is equally likely in node and background.  Domain
    # 1 adds a node marker, so its expected routing ownership must be higher.
    assert rel.ownership[0, 1] > rel.ownership[0, 0]


def test_constant_topic_row_is_not_viable():
    """A topic with no vocabulary contrast must contribute no viability evidence."""
    from spark_vi.models.topic.dag_placement import DagLayout, domain_reliability

    lay = DagLayout({1: [0]}, n_bg=1, tpn=1)
    rel = domain_reliability(lam_dict={0: np.array([
        [2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0],
    ])}, lay=lay)

    assert rel.viability[0, 0] == 0.0


def test_viability_is_topic_granular_within_each_domain():
    """One live topic among two yields one-half viability, not all-or-nothing."""
    from spark_vi.models.topic.dag_placement import DagLayout, domain_reliability

    lay = DagLayout({1: [0]}, n_bg=1, tpn=2)
    lam = {0: np.array([
        [2.0, 2.0, 2.0],   # background
        [8.0, 1.0, 1.0],   # live node topic
        [3.0, 3.0, 3.0],   # starved/constant node topic
    ])}
    rel = domain_reliability(lam, lay)

    assert rel.viability[0, 0] == 0.5


def test_weight_candidates_normalize_each_node_and_fall_back_to_uniform():
    """Zero evidence must not make a domain an arbitrary winner or produce NaN."""
    from spark_vi.models.topic.dag_placement import DagLayout, domain_reliability

    lay = DagLayout({1: [0]}, n_bg=1, tpn=1)
    lam = {
        0: np.full((2, 3), 2.0),
        1: np.full((2, 3), 5.0),
    }
    rel = domain_reliability(lam, lay)

    for candidate in ("distinctiveness", "ownership", "product"):
        weights = rel.weights(candidate)
        assert np.allclose(weights, [[0.5, 0.5]])
        assert np.allclose(weights.sum(axis=1), 1.0)
    with pytest.raises(ValueError):
        rel.weights("unknown")


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_weights_fall_back_to_uniform_for_nonfinite_row_total(invalid):
    """Invalid evidence cannot turn the remaining finite domain into the winner."""
    from spark_vi.models.topic.dag_placement import DomainReliability

    rel = DomainReliability(
        domain_keys=(0, 1),
        distinctiveness=np.array([[invalid, 1.0]]),
        ownership=np.ones((1, 2)),
        viability=np.ones((1, 2)),
    )

    assert np.allclose(rel.weights("distinctiveness"), [[0.5, 0.5]])


def test_disjoint_support_has_finite_distinctiveness_at_default_epsilon():
    """JS smoothing preserves a finite signal for maximally distinct vocabularies."""
    from spark_vi.models.topic.dag_placement import DagLayout, domain_reliability

    rel = domain_reliability(
        {0: np.array([[1.0, 0.0], [0.0, 1.0]])},
        DagLayout({1: [0]}, n_bg=1, tpn=1),
    )

    assert np.isfinite(rel.distinctiveness[0, 0])
    assert rel.distinctiveness[0, 0] > 0.0


@pytest.mark.parametrize("epsilon", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_domain_reliability_rejects_nonpositive_or_nonfinite_epsilon(epsilon):
    """Invalid smoothing cannot silently turn a finite JS comparison into NaN."""
    from spark_vi.models.topic.dag_placement import DagLayout, domain_reliability

    with pytest.raises(ValueError, match="epsilon"):
        domain_reliability(
            {0: np.array([[1.0, 0.0], [0.0, 1.0]])},
            DagLayout({1: [0]}, n_bg=1, tpn=1),
            epsilon=epsilon,
        )


@pytest.mark.parametrize("viability_tol", [-1.0, np.nan, np.inf, -np.inf])
def test_domain_reliability_rejects_negative_or_nonfinite_viability_tol(viability_tol):
    """Topic liveness requires a finite, nonnegative relative contrast threshold."""
    from spark_vi.models.topic.dag_placement import DagLayout, domain_reliability

    with pytest.raises(ValueError, match="viability_tol"):
        domain_reliability(
            {0: np.array([[2.0, 2.0], [3.0, 3.0]])},
            DagLayout({1: [0]}, n_bg=1, tpn=1),
            viability_tol=viability_tol,
        )


def test_reliability_rows_follow_sorted_nodes_and_their_topic_blocks():
    """Sorting node ids must not decouple output rows from their assigned topics."""
    from spark_vi.models.topic.dag_placement import DagLayout, domain_reliability

    lay = DagLayout({9: [0], 2: [0]}, n_bg=1, tpn=1)
    rel = domain_reliability({0: np.array([
        [4.0, 4.0, 4.0],   # background
        [4.0, 4.0, 4.0],   # block for sorted node 2
        [16.0, 1.0, 1.0],  # block for sorted node 9
    ])}, lay)

    assert lay.nodes == [2, 9]
    assert lay.block == {2: [1], 9: [2]}
    assert abs(rel.distinctiveness[0, 0]) < 1e-12
    assert rel.distinctiveness[1, 0] > 0.0
    assert np.array_equal(rel.viability[:, 0], [0.0, 1.0])
