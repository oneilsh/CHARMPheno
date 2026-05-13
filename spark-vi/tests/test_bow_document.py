"""Tests for BOWDocument: invariants, construction, frozen behavior."""
import numpy as np
import pytest
from pyspark.ml.linalg import SparseVector


def test_bow_document_holds_indices_counts_length():
    from spark_vi.models.topic import BOWDocument

    doc = BOWDocument(
        indices=np.array([0, 3, 7], dtype=np.int32),
        counts=np.array([2.0, 1.0, 4.0], dtype=np.float64),
        length=7,
    )
    assert doc.length == 7
    np.testing.assert_array_equal(doc.indices, [0, 3, 7])
    np.testing.assert_array_equal(doc.counts, [2.0, 1.0, 4.0])


def test_bow_document_is_frozen():
    from spark_vi.models.topic import BOWDocument
    doc = BOWDocument(indices=np.array([0], dtype=np.int32),
                      counts=np.array([1.0]), length=1)
    with pytest.raises((AttributeError, TypeError)):
        doc.length = 99


def test_bow_document_from_spark_row_unpacks_sparse_vector():
    from spark_vi.models.topic import BOWDocument

    sv = SparseVector(10, [0, 3, 7], [2.0, 1.0, 4.0])
    # A "row" here is anything that supports row[features_col] subscript.
    row = {"features": sv}
    doc = BOWDocument.from_spark_row(row, features_col="features")
    np.testing.assert_array_equal(doc.indices, [0, 3, 7])
    np.testing.assert_array_equal(doc.counts, [2.0, 1.0, 4.0])
    assert doc.length == 7
    assert doc.indices.dtype == np.int32
    assert doc.counts.dtype == np.float64
