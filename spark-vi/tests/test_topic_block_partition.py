import numpy as np
import pytest

from spark_vi.models.topic.partition import TopicBlockPartition


def _part():
    return TopicBlockPartition(
        group_var="source_cohort",
        background_k=3,
        foreground=(("cancer", 2), ("dementia", 2)),
    )


def test_k_is_background_plus_foreground():
    assert _part().K == 7


def test_index_blocks_are_contiguous_and_disjoint():
    p = _part()
    np.testing.assert_array_equal(p.background_indices(), [0, 1, 2])
    np.testing.assert_array_equal(p.block_indices("cancer"), [3, 4])
    np.testing.assert_array_equal(p.block_indices("dementia"), [5, 6])


def test_allowed_indices_unions_background_and_group_blocks():
    p = _part()
    np.testing.assert_array_equal(
        p.allowed_indices(frozenset({"cancer"})), [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(
        p.allowed_indices(frozenset()), [0, 1, 2])
    np.testing.assert_array_equal(
        p.allowed_indices(frozenset({"cancer", "dementia"})), [0, 1, 2, 3, 4, 5, 6])


def test_topic_labels():
    assert _part().topic_labels() == [
        "background", "background", "background",
        "cancer", "cancer", "dementia", "dementia"]


def test_unknown_group_in_allowed_indices_raises():
    with pytest.raises(KeyError):
        _part().allowed_indices(frozenset({"nope"}))


def test_rejects_nonpositive_sizes():
    with pytest.raises(ValueError):
        TopicBlockPartition("g", 0, (("a", 2),))
    with pytest.raises(ValueError):
        TopicBlockPartition("g", 3, (("a", 0),))


def test_rejects_duplicate_group_labels():
    with pytest.raises(ValueError):
        TopicBlockPartition("g", 3, (("a", 2), ("a", 1)))


def test_dict_roundtrip():
    p = _part()
    assert TopicBlockPartition.from_dict(p.to_dict()) == p
    assert p.to_dict() == {
        "group_var": "source_cohort",
        "background_k": 3,
        "foreground": [["cancer", 2], ["dementia", 2]],
    }
