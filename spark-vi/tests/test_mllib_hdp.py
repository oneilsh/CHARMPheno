"""Tests for spark_vi.mllib.hdp — fast unit tests for the HDP MLlib shim."""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Param-surface defaults & translation
# ---------------------------------------------------------------------------

def test_shared_param_defaults_match_mllib_lda():
    """Each shared Param defaults to the same value pyspark.ml.clustering.LDA uses,
    *except* `k` (HDP uses k=T which is a much bigger upper-bound than LDA's k)
    and the optimize* flags (HDP defers γ/α optimization per ADR 0011).
    """
    from pyspark.ml.clustering import LDA as MLlibLDA
    from spark_vi.mllib.hdp import OnlineHDPEstimator

    ours = OnlineHDPEstimator()
    theirs = MLlibLDA()

    for name in [
        "maxIter", "featuresCol", "topicDistributionCol",
        "optimizer", "learningOffset", "learningDecay", "subsamplingRate",
    ]:
        assert ours.getOrDefault(name) == theirs.getOrDefault(name), (
            f"Param {name!r} default mismatch: ours={ours.getOrDefault(name)!r} "
            f"theirs={theirs.getOrDefault(name)!r}"
        )


def test_hdp_specific_defaults():
    from spark_vi.mllib.hdp import OnlineHDPEstimator

    e = OnlineHDPEstimator()
    assert e.getOrDefault("k") == 150          # T, corpus truncation
    assert e.getOrDefault("docTruncation") == 15
    assert e.getOrDefault("gammaShape") == 100.0
    assert e.getOrDefault("caviMaxIter") == 100
    assert e.getOrDefault("caviTol") == 1e-4


def test_no_optimize_flags_exposed():
    """ADR 0012: γ/α optimization deferred from v1, so no optimize* flags."""
    from spark_vi.mllib.hdp import OnlineHDPEstimator

    e = OnlineHDPEstimator()
    param_names = {p.name for p in e.params}
    assert "optimizeDocConcentration" not in param_names
    assert "optimizeTopicConcentration" not in param_names
    assert "optimizeCorpusConcentration" not in param_names


def test_param_translation_to_model_and_config():
    from spark_vi.core.config import VIConfig
    from spark_vi.mllib.hdp import OnlineHDPEstimator, _build_model_and_config
    from spark_vi.models.online_hdp import OnlineHDP

    e = OnlineHDPEstimator(
        k=50, docTruncation=10, maxIter=42, seed=2026,
        learningOffset=512.0, learningDecay=0.6, subsamplingRate=0.1,
        docConcentration=[0.5], corpusConcentration=2.0, topicConcentration=0.02,
        gammaShape=50.0, caviMaxIter=200, caviTol=1e-3,
    )
    model, config = _build_model_and_config(e, vocab_size=100)

    assert isinstance(model, OnlineHDP)
    assert model.T == 50
    assert model.K == 10
    assert model.V == 100
    assert model.alpha == pytest.approx(0.5)
    assert model.gamma == pytest.approx(2.0)
    assert model.eta == pytest.approx(0.02)
    assert model.gamma_shape == pytest.approx(50.0)
    assert model.cavi_max_iter == 200
    assert model.cavi_tol == pytest.approx(1e-3)

    assert isinstance(config, VIConfig)
    assert config.max_iterations == 42
    assert config.learning_rate_tau0 == pytest.approx(512.0)
    assert config.learning_rate_kappa == pytest.approx(0.6)
    assert config.mini_batch_fraction == pytest.approx(0.1)
    assert config.random_seed == 2026


def test_param_translation_resolves_unset_concentrations_to_model_defaults():
    """When caller passes None / leaves unset, fall back to OnlineHDP's defaults
    (α=1.0, γ=1.0, η=0.01) — same constants as the model class."""
    from spark_vi.mllib.hdp import OnlineHDPEstimator, _build_model_and_config

    e = OnlineHDPEstimator(k=10, docTruncation=4)
    model, _ = _build_model_and_config(e, vocab_size=20)

    assert model.alpha == pytest.approx(1.0)
    assert model.gamma == pytest.approx(1.0)
    assert model.eta == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# Validator (rejection paths)
# ---------------------------------------------------------------------------

def test_unsupported_optimizer_em_raises():
    from spark_vi.mllib.hdp import OnlineHDPEstimator, _validate_unsupported_params

    e = OnlineHDPEstimator(optimizer="em")
    with pytest.raises(ValueError, match="optimizer"):
        _validate_unsupported_params(e)


def test_vector_doc_concentration_raises():
    """HDP α is scalar in v1 (ADR 0011 defers α optimization)."""
    from spark_vi.mllib.hdp import OnlineHDPEstimator, _validate_unsupported_params

    e = OnlineHDPEstimator(docConcentration=[0.1, 0.1, 0.1])
    with pytest.raises(ValueError, match="scalar docConcentration"):
        _validate_unsupported_params(e)


def test_scalar_doc_concentration_is_accepted():
    from spark_vi.mllib.hdp import OnlineHDPEstimator, _validate_unsupported_params

    e = OnlineHDPEstimator(docConcentration=[0.5])
    _validate_unsupported_params(e)  # should not raise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_expected_corpus_betas_normalize_to_one():
    """Plug-in E[β_t] is a length-T simplex vector."""
    from spark_vi.models.online_hdp import expected_corpus_betas

    rng = np.random.default_rng(0)
    T = 10
    u = rng.uniform(0.5, 5.0, size=T - 1)
    v = rng.uniform(0.5, 5.0, size=T - 1)

    E_beta = expected_corpus_betas(u, v, T=T)

    assert E_beta.shape == (T,)
    assert E_beta.min() >= 0.0
    assert E_beta.sum() == pytest.approx(1.0, abs=1e-12)


def test_expected_corpus_betas_prior_mean_uniform_under_gamma_one():
    """When u = 1, v = γ = 1 (Beta(1,1) = uniform), E[β_t] is a power-of-1/2
    geometric — last atom absorbs everything not consumed by earlier sticks.
    Specifically E[β_k] = 0.5 * 0.5^k for k=0..T-2; last atom = 0.5^(T-1)."""
    from spark_vi.models.online_hdp import expected_corpus_betas

    T = 5
    u = np.ones(T - 1)
    v = np.ones(T - 1)

    E_beta = expected_corpus_betas(u, v, T=T)

    expected = np.array([0.5, 0.25, 0.125, 0.0625, 0.0625])
    np.testing.assert_allclose(E_beta, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Fit + Model surface (uses Spark)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_hdp_corpus_df(spark):
    """3-cluster corpus, ~30 docs, vocab size 9.

    Each doc is a near-mixture of one of three favored vocab triples. With
    T=8 and K=4, the HDP should pull effective topic count down toward 3
    even though we set the truncation higher.
    """
    from pyspark.ml.linalg import Vectors

    rng = np.random.default_rng(0)
    rows = []
    favored = {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8]}
    for doc_id in range(30):
        cluster = doc_id % 3
        counts = np.zeros(9, dtype=np.float64)
        for w in rng.choice(favored[cluster], size=20, replace=True):
            counts[w] += 1.0
        for w in rng.choice(9, size=2, replace=True):
            counts[w] += 1.0
        rows.append((Vectors.dense(counts.tolist()),))
    return spark.createDataFrame(rows, schema=["features"])


def test_fit_returns_model_with_correct_shape(tiny_hdp_corpus_df):
    from spark_vi.mllib.hdp import OnlineHDPEstimator, OnlineHDPModel

    estimator = OnlineHDPEstimator(
        k=8, docTruncation=4, maxIter=5, seed=0, subsamplingRate=1.0,
    )
    model = estimator.fit(tiny_hdp_corpus_df)

    assert isinstance(model, OnlineHDPModel)
    assert model.vocabSize() == 9
    # Param round-trip: model exposes the configuration the Estimator had.
    assert model.getOrDefault("k") == 8
    assert model.getOrDefault("docTruncation") == 4
    assert model.getOrDefault("maxIter") == 5


def test_topics_matrix_shape_and_normalization(tiny_hdp_corpus_df):
    from pyspark.ml.linalg import DenseMatrix
    from spark_vi.mllib.hdp import OnlineHDPEstimator

    T = 8
    estimator = OnlineHDPEstimator(
        k=T, docTruncation=4, maxIter=5, seed=0, subsamplingRate=1.0,
    )
    model = estimator.fit(tiny_hdp_corpus_df)

    tm = model.topicsMatrix()
    assert isinstance(tm, DenseMatrix)
    assert tm.numRows == 9   # vocab size V
    assert tm.numCols == T
    # Each column (a topic) sums to 1 over the vocab.
    arr = tm.toArray()
    np.testing.assert_allclose(arr.sum(axis=0), 1.0, atol=1e-9)


def test_describe_topics_returns_top_k_per_topic(tiny_hdp_corpus_df):
    from spark_vi.mllib.hdp import OnlineHDPEstimator

    T = 8
    estimator = OnlineHDPEstimator(
        k=T, docTruncation=4, maxIter=5, seed=0, subsamplingRate=1.0,
    )
    model = estimator.fit(tiny_hdp_corpus_df)

    df = model.describeTopics(maxTermsPerTopic=4)
    rows = df.orderBy("topic").collect()

    assert [r["topic"] for r in rows] == list(range(T))
    for r in rows:
        assert len(r["termIndices"]) == 4
        assert len(r["termWeights"]) == 4
        weights = list(r["termWeights"])
        assert weights == sorted(weights, reverse=True)


def test_corpus_stick_weights_simplex(tiny_hdp_corpus_df):
    from spark_vi.mllib.hdp import OnlineHDPEstimator

    T = 8
    estimator = OnlineHDPEstimator(
        k=T, docTruncation=4, maxIter=5, seed=0, subsamplingRate=1.0,
    )
    model = estimator.fit(tiny_hdp_corpus_df)

    E_beta = model.corpusStickWeights()
    assert E_beta.shape == (T,)
    assert E_beta.min() >= 0.0
    assert E_beta.sum() == pytest.approx(1.0, abs=1e-9)


def test_active_topic_count_is_int_in_range(tiny_hdp_corpus_df):
    from spark_vi.mllib.hdp import OnlineHDPEstimator

    T = 8
    estimator = OnlineHDPEstimator(
        k=T, docTruncation=4, maxIter=5, seed=0, subsamplingRate=1.0,
    )
    model = estimator.fit(tiny_hdp_corpus_df)

    n = model.activeTopicCount()
    assert isinstance(n, int)
    assert 1 <= n <= T


def test_active_topic_count_monotone_in_mass_threshold(tiny_hdp_corpus_df):
    """Higher mass_threshold ⇒ at least as many active topics."""
    from spark_vi.mllib.hdp import OnlineHDPEstimator

    T = 8
    estimator = OnlineHDPEstimator(
        k=T, docTruncation=4, maxIter=5, seed=0, subsamplingRate=1.0,
    )
    model = estimator.fit(tiny_hdp_corpus_df)

    n_50 = model.activeTopicCount(mass_threshold=0.5)
    n_95 = model.activeTopicCount(mass_threshold=0.95)
    n_100 = model.activeTopicCount(mass_threshold=1.0)
    assert n_50 <= n_95 <= n_100 <= T


def test_active_topic_count_rejects_invalid_threshold(tiny_hdp_corpus_df):
    from spark_vi.mllib.hdp import OnlineHDPEstimator

    T = 8
    estimator = OnlineHDPEstimator(
        k=T, docTruncation=4, maxIter=5, seed=0, subsamplingRate=1.0,
    )
    model = estimator.fit(tiny_hdp_corpus_df)

    with pytest.raises(ValueError, match="mass_threshold"):
        model.activeTopicCount(mass_threshold=0.0)
    with pytest.raises(ValueError, match="mass_threshold"):
        model.activeTopicCount(mass_threshold=1.5)


def test_topic_count_at_mass_handles_edge_cases():
    """Direct unit test of the lifted helper."""
    import numpy as np
    from spark_vi.models.online_hdp import topic_count_at_mass

    # Perfectly-summing simplex; threshold near 1.0 fp-slop case.
    weights = np.array([0.5, 0.3, 0.2])
    assert topic_count_at_mass(weights, 1.0) == 3
    assert topic_count_at_mass(weights, 0.5) == 1
    assert topic_count_at_mass(weights, 0.81) == 3   # 0.5+0.3=0.8 < 0.81

    # Single topic carries all mass.
    weights = np.array([1.0, 0.0, 0.0])
    assert topic_count_at_mass(weights, 0.95) == 1

    # Validation rejects out-of-range thresholds.
    with pytest.raises(ValueError, match="mass_threshold"):
        topic_count_at_mass(weights, 0.0)
    with pytest.raises(ValueError, match="mass_threshold"):
        topic_count_at_mass(weights, 1.1)


def test_active_topic_count_truncation_invariant():
    """Same E[β_t] mass distribution → same active count, regardless of T."""
    import numpy as np
    from spark_vi.models.online_hdp import expected_corpus_betas

    # Construct two synthetic (u, v) configurations that produce the same
    # leading mass profile but pad with extra near-zero topics at higher T.
    # Verify activeTopicCount is invariant to the padding.
    def _count_active(E_beta, mass_threshold):
        sorted_desc = np.sort(E_beta)[::-1]
        cumsum = np.cumsum(sorted_desc)
        above = cumsum >= mass_threshold
        return int(np.argmax(above)) + 1 if above.any() else len(E_beta)

    # T=10: roughly geometric β with most mass on first 3
    rng = np.random.default_rng(0)
    T_small = 10
    u_small = rng.uniform(0.5, 5.0, size=T_small - 1)
    v_small = rng.uniform(0.5, 5.0, size=T_small - 1)
    E_small = expected_corpus_betas(u_small, v_small, T=T_small)

    # T=20: same first 9 sticks, plus 10 extra "near-zero" sticks via
    # large v (so E[W]≈0, β≈0). This concentrates on the first 9.
    T_big = 20
    u_big = np.concatenate([u_small, np.full(T_big - T_small, 1.0)])
    v_big = np.concatenate([v_small, np.full(T_big - T_small, 100.0)])
    E_big = expected_corpus_betas(u_big, v_big, T=T_big)

    # Both should agree on the active count for any mass_threshold below the
    # tail-mass boundary. (The "extra" sticks in the T=20 case carry
    # essentially zero mass.)
    for thresh in [0.5, 0.8, 0.95]:
        assert _count_active(E_small, thresh) == _count_active(E_big, thresh), (
            f"truncation-dependence at mass_threshold={thresh}: "
            f"T=10 → {_count_active(E_small, thresh)}, "
            f"T=20 → {_count_active(E_big, thresh)}"
        )


def test_transform_adds_topic_distribution_column(tiny_hdp_corpus_df):
    from pyspark.ml.linalg import Vector
    from spark_vi.mllib.hdp import OnlineHDPEstimator

    T = 8
    estimator = OnlineHDPEstimator(
        k=T, docTruncation=4, maxIter=5, seed=0, subsamplingRate=1.0,
    )
    model = estimator.fit(tiny_hdp_corpus_df)

    out = model.transform(tiny_hdp_corpus_df)
    assert "topicDistribution" in out.columns

    rows = out.select("topicDistribution").collect()
    for r in rows:
        td = r["topicDistribution"]
        assert isinstance(td, Vector)
        arr = np.asarray(td.toArray())
        assert arr.shape == (T,)              # length-T (corpus topics)
        np.testing.assert_allclose(arr.sum(), 1.0, atol=1e-6)
        assert arr.min() >= -1e-9             # nonneg (allow fp noise)


def test_transform_respects_custom_topic_distribution_col(tiny_hdp_corpus_df):
    from spark_vi.mllib.hdp import OnlineHDPEstimator

    estimator = OnlineHDPEstimator(
        k=8, docTruncation=4, maxIter=5, seed=0, subsamplingRate=1.0,
        topicDistributionCol="theta",
    )
    model = estimator.fit(tiny_hdp_corpus_df)
    out = model.transform(tiny_hdp_corpus_df)
    assert "theta" in out.columns
    assert "topicDistribution" not in out.columns


def test_log_likelihood_and_log_perplexity_raise_not_implemented(tiny_hdp_corpus_df):
    from spark_vi.mllib.hdp import OnlineHDPEstimator

    estimator = OnlineHDPEstimator(
        k=8, docTruncation=4, maxIter=5, seed=0, subsamplingRate=1.0,
    )
    model = estimator.fit(tiny_hdp_corpus_df)

    with pytest.raises(NotImplementedError, match="ELBO"):
        model.logLikelihood(tiny_hdp_corpus_df)
    with pytest.raises(NotImplementedError, match="ELBO"):
        model.logPerplexity(tiny_hdp_corpus_df)


def test_concentration_accessors_round_trip_inputs(tiny_hdp_corpus_df):
    """Trained α, γ, η equal the constructor inputs in v1 (no optimization)."""
    from spark_vi.mllib.hdp import OnlineHDPEstimator

    estimator = OnlineHDPEstimator(
        k=8, docTruncation=4, maxIter=3, seed=0, subsamplingRate=1.0,
        docConcentration=[0.7], corpusConcentration=1.5, topicConcentration=0.05,
    )
    model = estimator.fit(tiny_hdp_corpus_df)

    assert model.trainedAlpha() == pytest.approx(0.7)
    assert model.trainedCorpusConcentration() == pytest.approx(1.5)
    assert model.trainedTopicConcentration() == pytest.approx(0.05)
