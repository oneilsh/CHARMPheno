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


def test_param_translation_to_model_and_config():
    from spark_vi.core.config import VIConfig
    from spark_vi.models.lda import VanillaLDA
    from spark_vi.mllib.lda import VanillaLDAEstimator, _build_model_and_config

    e = VanillaLDAEstimator(
        k=7, maxIter=42, seed=2026,
        learningOffset=512.0, learningDecay=0.6,
        subsamplingRate=0.1,
        docConcentration=[0.05], topicConcentration=0.02,
        gammaShape=50.0, caviMaxIter=200, caviTol=1e-4,
    )
    model, config = _build_model_and_config(e, vocab_size=100)

    assert isinstance(model, VanillaLDA)
    assert model.K == 7
    assert model.V == 100
    assert model.alpha == pytest.approx(0.05)
    assert model.eta == pytest.approx(0.02)
    assert model.gamma_shape == pytest.approx(50.0)
    assert model.cavi_max_iter == 200
    assert model.cavi_tol == pytest.approx(1e-4)

    assert isinstance(config, VIConfig)
    assert config.max_iterations == 42
    assert config.learning_rate_tau0 == pytest.approx(512.0)
    assert config.learning_rate_kappa == pytest.approx(0.6)
    assert config.mini_batch_fraction == pytest.approx(0.1)
    assert config.random_seed == 2026


def test_param_translation_resolves_none_concentrations_to_one_over_k():
    """Per ADR 0008: alpha = eta = 1/K when caller passes None (the default)."""
    from spark_vi.mllib.lda import VanillaLDAEstimator, _build_model_and_config

    e = VanillaLDAEstimator(k=4)
    model, _ = _build_model_and_config(e, vocab_size=10)

    assert model.alpha == pytest.approx(0.25)
    assert model.eta == pytest.approx(0.25)


def test_unsupported_optimizer_em_raises():
    from spark_vi.mllib.lda import VanillaLDAEstimator, _validate_unsupported_params

    e = VanillaLDAEstimator(optimizer="em")
    with pytest.raises(ValueError, match="optimizer"):
        _validate_unsupported_params(e)


def test_optimize_doc_concentration_true_raises():
    from spark_vi.mllib.lda import VanillaLDAEstimator, _validate_unsupported_params

    e = VanillaLDAEstimator(optimizeDocConcentration=True)
    with pytest.raises(ValueError, match="optimizeDocConcentration"):
        _validate_unsupported_params(e)


def test_vector_doc_concentration_raises():
    from spark_vi.mllib.lda import VanillaLDAEstimator, _validate_unsupported_params

    e = VanillaLDAEstimator(k=3, docConcentration=[0.1, 0.1, 0.1])
    with pytest.raises(ValueError, match="docConcentration"):
        _validate_unsupported_params(e)


def test_scalar_doc_concentration_is_accepted():
    """A length-1 list (what toListFloat does to a scalar) must not raise."""
    from spark_vi.mllib.lda import VanillaLDAEstimator, _validate_unsupported_params

    e = VanillaLDAEstimator(docConcentration=[0.1])
    _validate_unsupported_params(e)  # should not raise
