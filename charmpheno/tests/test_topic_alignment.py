"""Tests for charmpheno.evaluate.topic_alignment.

Pure-numpy logic; tests are fast and live here because evaluation is a
clinical-layer concern even though it doesn't touch Spark for these specific
functions.
"""
import numpy as np


def test_js_divergence_matrix_diagonal_zero_for_identical_rows():
    from charmpheno.evaluate.topic_alignment import js_divergence_matrix
    A = np.array([[0.5, 0.5, 0.0], [0.1, 0.1, 0.8]])
    M = js_divergence_matrix(A, A)
    assert M.shape == (2, 2)
    np.testing.assert_allclose(np.diag(M), 0.0, atol=1e-12)


def test_js_divergence_matrix_orthogonal_distributions_max():
    """JS between two distributions with disjoint support is log(2) nats."""
    from charmpheno.evaluate.topic_alignment import js_divergence_matrix
    A = np.array([[1.0, 0.0]])
    B = np.array([[0.0, 1.0]])
    M = js_divergence_matrix(A, B)
    np.testing.assert_allclose(M[0, 0], np.log(2), atol=1e-6)


def test_order_by_prevalence_descends():
    from charmpheno.evaluate.topic_alignment import order_by_prevalence
    topics = np.array([[0.5, 0.5], [0.1, 0.9], [0.7, 0.3]])
    prevalence = np.array([2.0, 5.0, 1.0])
    sorted_topics, perm = order_by_prevalence(topics, prevalence)
    np.testing.assert_array_equal(perm, [1, 0, 2])
    np.testing.assert_array_equal(sorted_topics, topics[perm])


def test_alignment_biplot_data_returns_expected_shape():
    from charmpheno.evaluate.topic_alignment import alignment_biplot_data
    A = np.array([[1.0, 0.0], [0.5, 0.5]])
    B = np.array([[0.5, 0.5], [0.0, 1.0], [0.3, 0.7]])
    pa = np.array([1.0, 2.0])
    pb = np.array([2.0, 1.0, 3.0])

    out = alignment_biplot_data(A, pa, B, pb)
    assert out["js_matrix"].shape == (2, 3)
    assert out["perm_a"].shape == (2,)
    assert out["perm_b"].shape == (3,)
    np.testing.assert_array_equal(out["perm_a"], [1, 0])
    np.testing.assert_array_equal(out["perm_b"], [2, 0, 1])
    np.testing.assert_allclose(out["prevalence_a_sorted"], [2.0, 1.0])
    np.testing.assert_allclose(out["prevalence_b_sorted"], [3.0, 2.0, 1.0])


def test_ground_truth_from_oracle_normalizes_per_topic(spark):
    """Aggregates true_topic_id -> normalized (K, V) beta + K-vector prevalence."""
    from charmpheno.evaluate.topic_alignment import ground_truth_from_oracle
    from pyspark.sql.types import StructType, StructField, IntegerType, StringType

    schema = StructType([
        StructField("person_id", IntegerType(), False),
        StructField("visit_occurrence_id", IntegerType(), False),
        StructField("concept_id", IntegerType(), False),
        StructField("concept_name", StringType(), True),
        StructField("true_topic_id", IntegerType(), False),
    ])
    rows = [
        (1, 1, 100, "a", 0),
        (1, 1, 100, "a", 0),
        (1, 1, 200, "b", 1),
        (2, 1, 100, "a", 1),
        (2, 1, 300, "c", 1),
        (3, 1, 999, "x", 0),   # concept 999 not in vocab_map — skip
        (3, 2, 100, "a", 5),   # true_topic_id=5 >= K_true=2 — skip
    ]
    df = spark.createDataFrame(rows, schema=schema)
    vocab_map = {100: 0, 200: 1, 300: 2}

    beta, prev = ground_truth_from_oracle(df, vocab_map, K_true=2)
    assert beta.shape == (2, 3)
    np.testing.assert_allclose(beta.sum(axis=1), 1.0)
    np.testing.assert_allclose(beta[0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(beta[1], [1/3, 1/3, 1/3])
    np.testing.assert_allclose(prev, [2.0, 3.0])


def test_optimal_match_reorder_diagonalizes_a_known_permutation():
    """Hungarian matching on a permuted-identity divergence matrix recovers
    the inverse permutation, putting all zeros on the diagonal.
    """
    from charmpheno.evaluate.topic_alignment import optimal_match_reorder

    # Construct a JS-like matrix where the optimal pairing is row 0 -> col 2,
    # row 1 -> col 0, row 2 -> col 1 (each pair has cost 0).
    M = np.array([
        [0.6, 0.6, 0.0],   # row 0 best matches col 2
        [0.0, 0.6, 0.6],   # row 1 best matches col 0
        [0.6, 0.0, 0.6],   # row 2 best matches col 1
    ])
    reordered, perm = optimal_match_reorder(M)
    np.testing.assert_allclose(np.diag(reordered), 0.0)
    # Sanity: a non-Hungarian sort (e.g., argmin per row) would have
    # produced the same result here, but the function must remain robust
    # to ties + non-square inputs in general; we trust scipy's Hungarian
    # for the real-world case.
    assert reordered.shape == M.shape


def test_optimal_match_reorder_minimizes_diagonal_sum():
    """Diagonal sum after reordering must equal the optimal-assignment cost,
    which is by construction the minimum over all row permutations.
    """
    from itertools import permutations
    from charmpheno.evaluate.topic_alignment import optimal_match_reorder

    rng = np.random.default_rng(42)
    M = rng.uniform(0, 1, size=(4, 4))
    reordered, _ = optimal_match_reorder(M)

    diag_sum = float(np.diag(reordered).sum())
    brute_force_min = min(
        float(M[list(perm), range(4)].sum())
        for perm in permutations(range(4))
    )
    np.testing.assert_allclose(diag_sum, brute_force_min, atol=1e-9)
