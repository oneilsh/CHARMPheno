"""Tests for spark_vi.mllib.lda — fast unit tests for the MLlib shim."""
from __future__ import annotations

import numpy as np
import pytest


def test_default_params_match_mllib_lda(spark):
    """Each shared Param defaults to the same value pyspark.ml.clustering.LDA uses."""
    from pyspark.ml.clustering import LDA as MLlibLDA
    from spark_vi.mllib.lda import VanillaLDAEstimator

    ours = VanillaLDAEstimator()
    theirs = MLlibLDA()

    for name in [
        "k", "maxIter", "featuresCol", "topicDistributionCol",
        "optimizer", "learningOffset", "learningDecay",
        "subsamplingRate",
    ]:
        assert ours.getOrDefault(name) == theirs.getOrDefault(name), (
            f"Param {name!r} default mismatch: ours={ours.getOrDefault(name)!r} "
            f"theirs={theirs.getOrDefault(name)!r}"
        )


def test_our_extras_have_adr_0008_defaults():
    from spark_vi.mllib.lda import VanillaLDAEstimator

    e = VanillaLDAEstimator()
    assert e.getOrDefault("gammaShape") == 100.0
    assert e.getOrDefault("caviMaxIter") == 100
    assert e.getOrDefault("caviTol") == 1e-3


def test_optimize_doc_concentration_defaults_false_diverging_from_mllib():
    """We default to False so the v1 shim's no-arg fit doesn't trip the
    validator that rejects True (Task 5). MLlib defaults to True; this is
    a deliberate divergence documented in ADR 0009 / the spec.
    """
    from pyspark.ml.clustering import LDA as MLlibLDA
    from spark_vi.mllib.lda import VanillaLDAEstimator

    assert VanillaLDAEstimator().getOrDefault("optimizeDocConcentration") is False
    assert MLlibLDA().getOrDefault("optimizeDocConcentration") is True


def test_vector_to_bow_document_handles_sparse_vector():
    from pyspark.ml.linalg import Vectors
    from spark_vi.mllib.lda import _vector_to_bow_document

    sv = Vectors.sparse(5, [0, 2, 4], [1.0, 3.0, 2.0])
    doc = _vector_to_bow_document(sv)

    np.testing.assert_array_equal(doc.indices, [0, 2, 4])
    np.testing.assert_array_equal(doc.counts, [1.0, 3.0, 2.0])
    assert doc.length == 6


def test_vector_to_bow_document_handles_dense_vector_with_zeros():
    """DenseVectors with embedded zeros should round-trip to a sparse BOWDocument."""
    from pyspark.ml.linalg import Vectors
    from spark_vi.mllib.lda import _vector_to_bow_document

    dv = Vectors.dense([0.0, 2.0, 0.0, 5.0, 0.0])
    doc = _vector_to_bow_document(dv)

    np.testing.assert_array_equal(doc.indices, [1, 3])
    np.testing.assert_array_equal(doc.counts, [2.0, 5.0])
    assert doc.length == 7
