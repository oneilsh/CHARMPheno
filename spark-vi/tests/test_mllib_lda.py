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
        "subsamplingRate", "optimizeDocConcentration",
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


def test_optimize_topic_concentration_param_default_false():
    from spark_vi.mllib.lda import VanillaLDAEstimator

    e = VanillaLDAEstimator()
    assert e.getOrDefault("optimizeTopicConcentration") is False


def test_optimize_topic_concentration_param_can_be_set():
    from spark_vi.mllib.lda import VanillaLDAEstimator

    e = VanillaLDAEstimator(optimizeTopicConcentration=True)
    assert e.getOrDefault("optimizeTopicConcentration") is True


def test_optimize_doc_concentration_default_matches_mllib():
    """ADR 0010 flipped this default to match pyspark.ml.clustering.LDA."""
    from pyspark.ml.clustering import LDA as MLlibLDA
    from spark_vi.mllib.lda import VanillaLDAEstimator

    assert VanillaLDAEstimator().getOrDefault("optimizeDocConcentration") is True
    assert MLlibLDA().getOrDefault("optimizeDocConcentration") is True


def test_vector_to_bow_document_handles_sparse_vector():
    from pyspark.ml.linalg import Vectors
    from spark_vi.mllib._common import _vector_to_bow_document

    sv = Vectors.sparse(5, [0, 2, 4], [1.0, 3.0, 2.0])
    doc = _vector_to_bow_document(sv)

    np.testing.assert_array_equal(doc.indices, [0, 2, 4])
    np.testing.assert_array_equal(doc.counts, [1.0, 3.0, 2.0])
    assert doc.length == 6


def test_vector_to_bow_document_handles_dense_vector_with_zeros():
    """DenseVectors with embedded zeros should round-trip to a sparse BOWDocument."""
    from pyspark.ml.linalg import Vectors
    from spark_vi.mllib._common import _vector_to_bow_document

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
    assert model.alpha.shape == (model.K,)
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
    assert model.alpha.shape == (model.K,)
    assert model.eta == pytest.approx(0.25)


def test_unsupported_optimizer_em_raises():
    from spark_vi.mllib.lda import VanillaLDAEstimator, _validate_unsupported_params

    e = VanillaLDAEstimator(optimizer="em")
    with pytest.raises(ValueError, match="optimizer"):
        _validate_unsupported_params(e)


def test_optimize_doc_concentration_true_is_legal():
    """v0 rejected this; ADR 0010 makes it the default behavior."""
    from spark_vi.mllib.lda import VanillaLDAEstimator, _validate_unsupported_params

    e = VanillaLDAEstimator(optimizeDocConcentration=True)
    _validate_unsupported_params(e)  # should not raise


def test_vector_doc_concentration_is_legal():
    """v0 rejected this; ADR 0010 supports asymmetric α (length K)."""
    from spark_vi.mllib.lda import VanillaLDAEstimator, _validate_unsupported_params

    e = VanillaLDAEstimator(k=3, docConcentration=[0.1, 0.1, 0.1])
    _validate_unsupported_params(e)  # should not raise


def test_vector_doc_concentration_wrong_length_raises():
    """Vector α with length != k is still rejected."""
    from spark_vi.mllib.lda import VanillaLDAEstimator, _validate_unsupported_params

    e = VanillaLDAEstimator(k=4, docConcentration=[0.1, 0.1, 0.1])
    with pytest.raises(ValueError, match="length"):
        _validate_unsupported_params(e)


def test_scalar_doc_concentration_is_accepted():
    """A length-1 list (what toListFloat does to a scalar) must not raise."""
    from spark_vi.mllib.lda import VanillaLDAEstimator, _validate_unsupported_params

    e = VanillaLDAEstimator(docConcentration=[0.1])
    _validate_unsupported_params(e)  # should not raise


@pytest.fixture(scope="module")
def tiny_corpus_df(spark):
    """3-topic well-separated corpus, ~30 docs, vocab size 9.

    Topic 0 favors words 0,1,2; topic 1 favors 3,4,5; topic 2 favors 6,7,8.
    Each doc is a near-mixture of one topic.
    """
    from pyspark.ml.linalg import Vectors

    rng = np.random.default_rng(0)
    rows = []
    favored = {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8]}
    for doc_id in range(30):
        topic = doc_id % 3
        counts = np.zeros(9, dtype=np.float64)
        for w in rng.choice(favored[topic], size=20, replace=True):
            counts[w] += 1.0
        # add a little noise to other vocab so vectors aren't identical
        for w in rng.choice(9, size=2, replace=True):
            counts[w] += 1.0
        rows.append((Vectors.dense(counts.tolist()),))
    return spark.createDataFrame(rows, schema=["features"])


def test_fit_returns_model_with_correct_shape(tiny_corpus_df):
    from spark_vi.mllib.lda import VanillaLDAEstimator, VanillaLDAModel

    estimator = VanillaLDAEstimator(k=3, maxIter=5, seed=0, subsamplingRate=1.0)
    model = estimator.fit(tiny_corpus_df)

    assert isinstance(model, VanillaLDAModel)
    assert model.vocabSize() == 9
    # Param round-trip: model exposes the same configuration the Estimator had.
    assert model.getOrDefault("k") == 3
    assert model.getOrDefault("maxIter") == 5


def test_topics_matrix_shape_and_normalization(tiny_corpus_df):
    from pyspark.ml.linalg import DenseMatrix
    from spark_vi.mllib.lda import VanillaLDAEstimator

    estimator = VanillaLDAEstimator(k=3, maxIter=5, seed=0, subsamplingRate=1.0)
    model = estimator.fit(tiny_corpus_df)

    tm = model.topicsMatrix()
    assert isinstance(tm, DenseMatrix)
    assert tm.numRows == 9   # vocab size V
    assert tm.numCols == 3   # K
    # Each column (a topic) sums to 1 (row-stochastic over vocab in MLlib's orientation).
    arr = tm.toArray()
    np.testing.assert_allclose(arr.sum(axis=0), 1.0, atol=1e-9)


def test_describe_topics_returns_top_k_per_topic(tiny_corpus_df):
    from spark_vi.mllib.lda import VanillaLDAEstimator

    estimator = VanillaLDAEstimator(k=3, maxIter=5, seed=0, subsamplingRate=1.0)
    model = estimator.fit(tiny_corpus_df)

    df = model.describeTopics(maxTermsPerTopic=4)
    rows = df.orderBy("topic").collect()

    assert [r["topic"] for r in rows] == [0, 1, 2]
    for r in rows:
        assert len(r["termIndices"]) == 4
        assert len(r["termWeights"]) == 4
        # Weights must be descending.
        weights = list(r["termWeights"])
        assert weights == sorted(weights, reverse=True)


def test_transform_adds_topic_distribution_column(tiny_corpus_df):
    from pyspark.ml.linalg import Vector
    from spark_vi.mllib.lda import VanillaLDAEstimator

    estimator = VanillaLDAEstimator(k=3, maxIter=5, seed=0, subsamplingRate=1.0)
    model = estimator.fit(tiny_corpus_df)

    out = model.transform(tiny_corpus_df)
    assert "topicDistribution" in out.columns

    rows = out.select("topicDistribution").collect()
    for r in rows:
        td = r["topicDistribution"]
        assert isinstance(td, Vector)
        arr = np.asarray(td.toArray())
        assert arr.shape == (3,)
        np.testing.assert_allclose(arr.sum(), 1.0, atol=1e-6)


def test_transform_respects_custom_topic_distribution_col(tiny_corpus_df):
    from spark_vi.mllib.lda import VanillaLDAEstimator

    estimator = VanillaLDAEstimator(
        k=3, maxIter=5, seed=0, subsamplingRate=1.0,
        topicDistributionCol="theta",
    )
    model = estimator.fit(tiny_corpus_df)
    out = model.transform(tiny_corpus_df)
    assert "theta" in out.columns
    assert "topicDistribution" not in out.columns


def test_transform_uses_trained_alpha_from_result(tiny_corpus_df):
    """When optimize_alpha=True, transform must read α from
    result.global_params['alpha'], not from the Estimator's
    docConcentration Param (which was the v0 path).
    """
    from spark_vi.mllib.lda import VanillaLDAEstimator
    import numpy as np

    estimator = VanillaLDAEstimator(
        k=3, maxIter=5, seed=0, subsamplingRate=1.0,
        optimizeDocConcentration=True,
    )
    model = estimator.fit(tiny_corpus_df)

    # The trained α is on result.global_params, not on the docConcentration Param.
    trained_alpha = model.result.global_params["alpha"]
    assert trained_alpha.shape == (3,)
    # Should have moved at least somewhere from the 1/3 init under 5 iters.
    assert not np.allclose(trained_alpha, 1.0 / 3, atol=1e-6)

    # Transform should not raise and should produce a valid distribution.
    out = model.transform(tiny_corpus_df)
    rows = out.select("topicDistribution").collect()
    for r in rows:
        arr = np.asarray(r["topicDistribution"].toArray())
        assert arr.shape == (3,)
        np.testing.assert_allclose(arr.sum(), 1.0, atol=1e-6)


def test_log_likelihood_and_log_perplexity_raise_not_implemented(tiny_corpus_df):
    from spark_vi.mllib.lda import VanillaLDAEstimator

    estimator = VanillaLDAEstimator(k=3, maxIter=5, seed=0, subsamplingRate=1.0)
    model = estimator.fit(tiny_corpus_df)

    with pytest.raises(NotImplementedError, match="ELBO"):
        model.logLikelihood(tiny_corpus_df)
    with pytest.raises(NotImplementedError, match="ELBO"):
        model.logPerplexity(tiny_corpus_df)


def test_param_translation_passes_optimize_flags_to_model():
    from spark_vi.mllib.lda import VanillaLDAEstimator, _build_model_and_config

    e = VanillaLDAEstimator(
        k=3, optimizeDocConcentration=True, optimizeTopicConcentration=True,
    )
    model, _ = _build_model_and_config(e, vocab_size=10)
    assert model.optimize_alpha is True
    assert model.optimize_eta is True


def test_param_translation_accepts_vector_doc_concentration():
    from spark_vi.mllib.lda import VanillaLDAEstimator, _build_model_and_config
    import numpy as np

    e = VanillaLDAEstimator(k=3, docConcentration=[0.1, 0.5, 0.9])
    model, _ = _build_model_and_config(e, vocab_size=10)
    np.testing.assert_allclose(model.alpha, [0.1, 0.5, 0.9])


def test_model_trained_alpha_accessor_returns_trained_vector(tiny_corpus_df):
    from spark_vi.mllib.lda import VanillaLDAEstimator
    import numpy as np

    estimator = VanillaLDAEstimator(
        k=3, maxIter=5, seed=0, subsamplingRate=1.0,
        optimizeDocConcentration=True,
    )
    model = estimator.fit(tiny_corpus_df)

    alpha = model.trainedAlpha()
    assert alpha.shape == (3,)
    np.testing.assert_allclose(alpha, model.result.global_params["alpha"])


def test_model_trained_topic_concentration_accessor_returns_trained_eta(tiny_corpus_df):
    from spark_vi.mllib.lda import VanillaLDAEstimator

    estimator = VanillaLDAEstimator(
        k=3, maxIter=5, seed=0, subsamplingRate=1.0,
        optimizeTopicConcentration=True,
    )
    model = estimator.fit(tiny_corpus_df)

    eta = model.trainedTopicConcentration()
    assert isinstance(eta, float)
    assert eta == float(model.result.global_params["eta"])


def test_explicit_topic_concentration_through_fit_path(tiny_corpus_df):
    """Regression: explicitly setting topicConcentration on the Estimator
    must survive the param-copy loop in _fit. The previous @property
    accessor on the Model shadowed the Param descriptor, causing _set
    inside the param-copy loop to crash with AttributeError when
    isSet(topicConcentration) was True. Methods don't shadow.
    """
    from spark_vi.mllib.lda import VanillaLDAEstimator

    estimator = VanillaLDAEstimator(
        k=3, maxIter=3, seed=0, subsamplingRate=1.0,
        topicConcentration=0.05,
    )
    model = estimator.fit(tiny_corpus_df)  # must not raise

    assert model.trainedTopicConcentration() == pytest.approx(0.05)
