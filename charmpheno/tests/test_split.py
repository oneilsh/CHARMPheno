"""Tests for charmpheno.omop.split.split_bow_by_person."""
from __future__ import annotations

import pytest
from pyspark.ml.linalg import Vectors


@pytest.fixture(scope="module")
def small_bow_df(spark):
    """10-row BOW with person_id 0..9 and a 3-element sparse features vector."""
    rows = [
        (i, Vectors.sparse(3, [(i % 3, 1.0)]))
        for i in range(10)
    ]
    return spark.createDataFrame(rows, schema=["person_id", "features"])


def test_split_bow_by_person_partitions_disjointly(small_bow_df):
    from charmpheno.omop.split import split_bow_by_person

    train, holdout = split_bow_by_person(small_bow_df, holdout_fraction=0.3, seed=42)
    train_ids = {r.person_id for r in train.collect()}
    holdout_ids = {r.person_id for r in holdout.collect()}

    assert train_ids.isdisjoint(holdout_ids)
    assert train_ids | holdout_ids == set(range(10))


def test_split_bow_by_person_is_deterministic_same_seed(small_bow_df):
    from charmpheno.omop.split import split_bow_by_person

    train1, holdout1 = split_bow_by_person(small_bow_df, holdout_fraction=0.3, seed=42)
    train2, holdout2 = split_bow_by_person(small_bow_df, holdout_fraction=0.3, seed=42)

    assert {r.person_id for r in train1.collect()} == {r.person_id for r in train2.collect()}
    assert {r.person_id for r in holdout1.collect()} == {r.person_id for r in holdout2.collect()}


def test_split_bow_by_person_differs_across_seeds(small_bow_df):
    from charmpheno.omop.split import split_bow_by_person

    _, holdout_a = split_bow_by_person(small_bow_df, holdout_fraction=0.3, seed=1)
    _, holdout_b = split_bow_by_person(small_bow_df, holdout_fraction=0.3, seed=2)

    ids_a = {r.person_id for r in holdout_a.collect()}
    ids_b = {r.person_id for r in holdout_b.collect()}
    # Could in principle coincide for tiny inputs, but for 10 rows over two unrelated
    # SHA-256 keyings this is overwhelmingly unlikely; assert non-equal as a regression
    # signal.
    assert ids_a != ids_b


def test_split_bow_by_person_rejects_invalid_fraction(small_bow_df):
    from charmpheno.omop.split import split_bow_by_person

    with pytest.raises(ValueError, match="holdout_fraction"):
        split_bow_by_person(small_bow_df, holdout_fraction=0.0, seed=42)
    with pytest.raises(ValueError, match="holdout_fraction"):
        split_bow_by_person(small_bow_df, holdout_fraction=1.0, seed=42)
