"""Tests for the MLlib-shim StreamingSTM estimator and STMModel."""
from __future__ import annotations

import numpy as np
import pytest

pyspark = pytest.importorskip("pyspark")
from pyspark.ml.linalg import SparseVector, DenseVector, Vectors


class TestVectorToSTMDocument:
    def test_constructs_from_row_with_features_and_covariates(self):
        from spark_vi.mllib.topic._common import _vector_to_stm_document
        row = {
            "features": SparseVector(10, [0, 3, 5], [2.0, 1.0, 3.0]),
            "covariates": DenseVector([1.0, 0.5, -1.2]),
        }
        doc = _vector_to_stm_document(row, features_col="features",
                                       covariates_col="covariates")
        np.testing.assert_array_equal(doc.indices, [0, 3, 5])
        np.testing.assert_array_equal(doc.counts, [2.0, 1.0, 3.0])
        assert doc.length == 6
        np.testing.assert_array_equal(doc.x, [1.0, 0.5, -1.2])


class TestStreamingSTMPathA:
    def test_constructs_with_covariates_col(self):
        from spark_vi.mllib.topic.stm import StreamingSTM
        est = StreamingSTM(
            K=5, features_col="features",
            covariates_col="covariates",
            covariate_names=["age", "sex", "cohort"],
        )
        assert est.K == 5
        assert est.P == 3
        assert est.covariate_names == ["age", "sex", "cohort"]

    def test_rejects_zero_covariates(self):
        from spark_vi.mllib.topic.stm import StreamingSTM
        with pytest.raises(ValueError, match="covariate_names"):
            StreamingSTM(K=5, features_col="features",
                         covariates_col="covariates", covariate_names=[])

    def test_rejects_path_b_args_without_formula_extra(self):
        """If user passes formula args without installing the formula extra,
        the estimator should error at construct time, not at fit time."""
        from spark_vi.mllib.topic.stm import StreamingSTM
        with pytest.raises(ValueError, match="covariate_formula|covariate_names"):
            StreamingSTM(K=5, features_col="features")
