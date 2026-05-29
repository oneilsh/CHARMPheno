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

    def test_constructs_from_row_with_dense_features(self):
        from spark_vi.mllib.topic._common import _vector_to_stm_document
        row = {
            "features": DenseVector([2.0, 0.0, 0.0, 1.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0]),
            "covariates": DenseVector([1.0, 0.5, -1.2]),
        }
        doc = _vector_to_stm_document(row, features_col="features",
                                       covariates_col="covariates")
        # Dense → sparsified to nonzero positions {0, 3, 5}.
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

    def test_fit_returns_STMModel_on_small_joined_df(self, spark):
        """Smoke: StreamingSTM.fit consumes a (features, covariates) DataFrame and
        returns a fitted STMModel with the right metadata."""
        from spark_vi.mllib.topic.stm import STMModel, StreamingSTM

        # Toy corpus: 6 docs, V=8, P=2.
        rows = [
            (SparseVector(8, [0, 2], [3.0, 1.0]), DenseVector([1.0, 0.0])),
            (SparseVector(8, [1, 3], [2.0, 2.0]), DenseVector([0.0, 1.0])),
            (SparseVector(8, [0, 4], [1.0, 2.0]), DenseVector([1.0, 0.5])),
            (SparseVector(8, [5, 6], [1.0, 1.0]), DenseVector([0.0, 1.0])),
            (SparseVector(8, [2, 7], [2.0, 1.0]), DenseVector([1.0, 0.0])),
            (SparseVector(8, [3, 4], [1.0, 3.0]), DenseVector([0.5, 0.5])),
        ]
        df = spark.createDataFrame(rows, ["features", "covariates"])
        est = StreamingSTM(
            K=2,
            features_col="features",
            covariates_col="covariates",
            covariate_names=["x1", "x2"],
            random_seed=0,
        )
        model = est.fit(df, max_iter=2, subsampling_rate=1.0, tau0=1.0, kappa=0.5)
        assert isinstance(model, STMModel)
        assert model.metadata["K"] == 2
        assert model.metadata["V"] == 8
        assert model.metadata["P"] == 2
        assert model.covariate_names == ["x1", "x2"]
