"""Tests for the STM MLlib stages (OnlineSTMEstimator / OnlineSTMModel)."""
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


class TestOnlineSTMEstimatorCovariateColumn:
    def test_constructs_with_covariates_col(self):
        from spark_vi.mllib.topic.stm import OnlineSTMEstimator
        est = OnlineSTMEstimator(
            k=5, featuresCol="features",
            covariatesCol="covariates",
            covariateNames=["age", "sex", "cohort"],
        )
        assert est.getOrDefault("k") == 5
        assert est.getOrDefault("covariateNames") == ["age", "sex", "cohort"]

    def test_rejects_zero_covariates(self):
        from spark_vi.mllib.topic.stm import OnlineSTMEstimator
        with pytest.raises(ValueError, match="covariateNames"):
            OnlineSTMEstimator(k=5, featuresCol="features",
                               covariatesCol="covariates", covariateNames=[])

    def test_rejects_both_covariate_sources(self):
        from spark_vi.mllib.topic.stm import OnlineSTMEstimator
        with pytest.raises(ValueError, match="not both"):
            OnlineSTMEstimator(k=5, covariateNames=["a"], covariateFormula="~ a")

    def test_fit_without_covariates_raises(self, spark):
        """No covariate source configured: the fit refuses (a bare Estimator may be
        configured after construction, so the check is at fit)."""
        from spark_vi.mllib.topic.stm import OnlineSTMEstimator
        df = spark.createDataFrame(
            [(SparseVector(4, [0], [1.0]), DenseVector([1.0]))],
            ["features", "covariates"])
        with pytest.raises(ValueError, match="covariateFormula|covariateNames"):
            OnlineSTMEstimator(k=5, featuresCol="features").fit(df)

    def test_fit_returns_model_on_small_joined_df(self, spark):
        """Smoke: fit consumes a (features, covariates) DataFrame and returns a
        fitted OnlineSTMModel with the right metadata."""
        from spark_vi.mllib.topic.stm import OnlineSTMEstimator, OnlineSTMModel

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
        est = OnlineSTMEstimator(
            k=2, featuresCol="features", covariatesCol="covariates",
            covariateNames=["x1", "x2"], seed=0,
            maxIter=2, subsamplingRate=1.0, learningOffset=1.0, learningDecay=0.5,
        )
        model = est.fit(df)
        assert isinstance(model, OnlineSTMModel)
        assert model.result.metadata["K"] == 2
        assert model.result.metadata["V"] == 8
        assert model.result.metadata["P"] == 2
        assert model.getOrDefault("covariateNames") == ["x1", "x2"]

    def test_fit_resume_accumulates_iterations(self, spark, tmp_path):
        """fit -> save -> fit(resumeFrom): the iteration counter continues, so a
        3-iter fit then a 2-iter resume yields n_iterations == 5. This is what
        keeps rho_t = (tau0 + t + 1)^-kappa shrinking instead of resetting."""
        from spark_vi.mllib.topic.stm import OnlineSTMEstimator

        rows = [
            (SparseVector(8, [0, 2], [3.0, 1.0]), DenseVector([1.0, 0.0])),
            (SparseVector(8, [1, 3], [2.0, 2.0]), DenseVector([0.0, 1.0])),
            (SparseVector(8, [0, 4], [1.0, 2.0]), DenseVector([1.0, 0.5])),
            (SparseVector(8, [5, 6], [1.0, 1.0]), DenseVector([0.0, 1.0])),
        ]
        df = spark.createDataFrame(rows, ["features", "covariates"])
        est = OnlineSTMEstimator(
            k=2, featuresCol="features", covariatesCol="covariates",
            covariateNames=["x1", "x2"], seed=0,
            maxIter=3, subsamplingRate=1.0, learningOffset=1.0, learningDecay=0.5,
        )
        m1 = est.fit(df)
        assert m1.result.n_iterations == 3

        save_dir = tmp_path / "ckpt"
        m1.save(str(save_dir))

        est.setParams(maxIter=2, resumeFrom=str(save_dir / "result"))
        m2 = est.fit(df)
        assert m2.result.n_iterations == 5  # 3 loaded + 2 additional


class TestCorpusMeanProportionsRDD:
    """Distributed corpus-mean α-equivalent over an RDD of covariate vectors.

    Mirrors the engine's mapPartitions+treeReduce idiom: only a (K-vector sum,
    count) crosses back to the driver, so it scales to any D and any covariate
    cardinality. Verified against the numpy oracle.
    """

    def test_matches_numpy_oracle_across_partitions(self, spark):
        from spark_vi.mllib.topic.stm import corpus_mean_topic_proportions_rdd
        from spark_vi.models.topic.stm import corpus_mean_topic_proportions

        rng = np.random.default_rng(3)
        D, P, K = 20, 3, 4
        Gamma = rng.normal(size=(P, K))
        X = rng.normal(size=(D, P))
        # numSlices > 1 forces a real multi-partition tree-combine.
        rdd = spark.sparkContext.parallelize(
            [X[d] for d in range(D)], numSlices=4
        )

        result = corpus_mean_topic_proportions_rdd(rdd, Gamma)

        np.testing.assert_allclose(
            result, corpus_mean_topic_proportions(Gamma, X), rtol=1e-10
        )

    def test_returns_probability_vector(self, spark):
        from spark_vi.mllib.topic.stm import corpus_mean_topic_proportions_rdd

        rng = np.random.default_rng(4)
        Gamma = rng.normal(size=(2, 5))
        rdd = spark.sparkContext.parallelize(
            [rng.normal(size=2) for _ in range(8)], numSlices=3
        )

        result = corpus_mean_topic_proportions_rdd(rdd, Gamma)

        assert result.shape == (5,)
        assert np.all(result >= 0.0)
        np.testing.assert_allclose(result.sum(), 1.0)

    def test_gated_rdd_matches_pure_numpy_oracle(self, spark):
        # Distributed gated alpha-equivalent must equal the pure-numpy
        # corpus_mean_topic_proportions_gated on the same (x, groups) data,
        # including a background-only doc (group with no foreground block).
        from spark_vi.mllib.topic.stm import corpus_mean_topic_proportions_gated_rdd
        from spark_vi.models.topic.stm import corpus_mean_topic_proportions_gated
        from spark_vi.models.topic.partition import TopicBlockPartition

        rng = np.random.default_rng(7)
        part = TopicBlockPartition(
            "g", background_k=2, foreground=(("cancer", 1), ("dementia", 1)))  # K=4
        P = 3
        Gamma = rng.normal(size=(P, part.K))
        groups = [
            frozenset({"cancer"}), frozenset({"dementia"}), frozenset({"cancer"}),
            frozenset({"dementia"}), frozenset({"common"}),  # no FG block -> bg only
            frozenset({"cancer"}), frozenset({"dementia"}),
        ]
        xs = [rng.normal(size=P) for _ in groups]
        X = np.vstack(xs)
        expected = corpus_mean_topic_proportions_gated(Gamma, X, groups, part)

        rdd = spark.sparkContext.parallelize(
            list(zip([x.tolist() for x in xs], groups)), numSlices=3)
        result = corpus_mean_topic_proportions_gated_rdd(rdd, Gamma, part)

        np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-12)
        # Masked: out-of-group foreground topics get zero corpus-mean mass only
        # if no doc is in that group; here both groups present, so both > 0.
        np.testing.assert_allclose(result.sum(), 1.0)

    def test_gated_rdd_raises_on_empty(self, spark):
        from spark_vi.mllib.topic.stm import corpus_mean_topic_proportions_gated_rdd
        from spark_vi.models.topic.partition import TopicBlockPartition

        part = TopicBlockPartition("g", background_k=2, foreground=(("cancer", 1),))
        empty = spark.sparkContext.parallelize([], numSlices=1)
        with pytest.raises(ValueError, match="empty"):
            corpus_mean_topic_proportions_gated_rdd(empty, np.zeros((2, 3)), part)


def test_estimator_rejects_group_var_in_covariates():
    from spark_vi.mllib.topic.stm import OnlineSTMEstimator
    from spark_vi.models.topic.partition import TopicBlockPartition
    part = TopicBlockPartition("source_cohort", background_k=2, foreground=(("cancer", 1),))
    with pytest.raises(ValueError, match="group_var"):
        OnlineSTMEstimator(
            k=3, covariatesCol="covariates",
            covariateNames=["Intercept", "C(source_cohort)[T.cancer]"],
            topicBlocks=part, docGroupCol="source_cohort")


def test_estimator_rejects_group_var_in_formula():
    from spark_vi.mllib.topic.stm import OnlineSTMEstimator
    from spark_vi.models.topic.partition import TopicBlockPartition
    part = TopicBlockPartition("source_cohort", background_k=2, foreground=(("cancer", 1),))
    with pytest.raises(ValueError, match="group_var"):
        OnlineSTMEstimator(k=3, covariateFormula="~ age + C(source_cohort)",
                           topicBlocks=part, docGroupCol="source_cohort")


def test_estimator_requires_group_col_with_partition():
    from spark_vi.mllib.topic.stm import OnlineSTMEstimator
    from spark_vi.models.topic.partition import TopicBlockPartition
    part = TopicBlockPartition("source_cohort", background_k=2, foreground=(("cancer", 1),))
    with pytest.raises(ValueError, match="docGroupCol"):
        OnlineSTMEstimator(k=3, covariatesCol="covariates",
                           covariateNames=["Intercept", "age"], topicBlocks=part)


def test_topic_blocks_param_round_trips_as_json():
    from spark_vi.mllib.topic.stm import OnlineSTMEstimator
    from spark_vi.models.topic.partition import TopicBlockPartition
    part = TopicBlockPartition("grp", background_k=2, foreground=(("rare", 1),))
    est = OnlineSTMEstimator(k=3, covariateNames=["a"], topicBlocks=part,
                             docGroupCol="grp")
    assert isinstance(est.getOrDefault("topicBlocks"), str)
    assert est._topic_block_partition() == part
    est2 = OnlineSTMEstimator(k=3, covariateNames=["a"], topicBlocks=part.to_dict(),
                              docGroupCol="grp")
    assert est2.getOrDefault("topicBlocks") == est.getOrDefault("topicBlocks")


def test_estimator_gated_fit_smoke(spark):
    from spark_vi.mllib.topic.stm import OnlineSTMEstimator
    from spark_vi.models.topic.partition import TopicBlockPartition
    part = TopicBlockPartition("grp", background_k=2, foreground=(("rare", 1),))
    rows = []
    for i in range(20):
        rare = i % 4 == 0
        rows.append((Vectors.sparse(5, {i % 5: 2.0, (i + 1) % 5: 1.0}),
                     Vectors.dense([1.0, float(i % 2)]),
                     "rare" if rare else "common"))
    df = spark.createDataFrame(rows, ["features", "covariates", "grp"])
    est = OnlineSTMEstimator(k=3, covariatesCol="covariates",
                             covariateNames=["Intercept", "age"],
                             topicBlocks=part, docGroupCol="grp",
                             maxIter=3, subsamplingRate=1.0)
    model = est.fit(df)
    assert model.result.global_params["lambda"].shape == (3, 5)
    assert model.result.metadata  # fit produced a model
    assert model.topicBlockPartition() == part


class TestEstimatorHardeningThreading:
    """The opt-in OnlineSTM hardening knobs reach the engine and the metadata."""

    def _toy_df(self, spark):
        from pyspark.ml.linalg import SparseVector, DenseVector
        rows = [
            (SparseVector(8, [0, 2], [3.0, 1.0]), DenseVector([1.0, 0.0])),
            (SparseVector(8, [1, 3], [2.0, 2.0]), DenseVector([0.0, 1.0])),
            (SparseVector(8, [0, 4], [1.0, 2.0]), DenseVector([1.0, 0.5])),
            (SparseVector(8, [5, 6], [1.0, 1.0]), DenseVector([0.0, 1.0])),
            (SparseVector(8, [2, 7], [2.0, 1.0]), DenseVector([1.0, 0.2])),
            (SparseVector(8, [1, 6], [1.0, 3.0]), DenseVector([0.0, 0.8])),
        ]
        return spark.createDataFrame(rows, ["features", "covariates"])

    def test_reference_topic_reaches_engine(self, spark):
        """A reference fit drives the reference topic's Gamma column to 0 — the
        end-to-end signature that referenceTopic took effect through the shim."""
        from spark_vi.mllib.topic.stm import OnlineSTMEstimator
        est = OnlineSTMEstimator(
            k=4, featuresCol="features", covariatesCol="covariates",
            covariateNames=["a", "b"], seed=0, referenceTopic=True,
            maxIter=3, subsamplingRate=1.0)
        model = est.fit(self._toy_df(spark))
        assert np.allclose(model.result.global_params["Gamma"][:, 0], 0.0)

    def test_hardening_knobs_persisted_in_metadata(self, spark):
        from spark_vi.mllib.topic.stm import OnlineSTMEstimator
        est = OnlineSTMEstimator(
            k=4, featuresCol="features", covariatesCol="covariates",
            covariateNames=["a", "b"], seed=0,
            referenceTopic=True, minPairSupport=3,
            spectralInit=False, maxIter=2, subsamplingRate=1.0)
        model = est.fit(self._toy_df(spark))
        assert model.result.metadata["stm_hardening"] == {
            "reference_topic": True,
            "min_pair_support": 3,
            "spectral_init": False,
            "spectral_method": "dense",
            "spectral_method_requested": "auto",
            "estimate_sigma_diagonal": False,
            "estimate_global_scale": False,
            "global_scale_step_cap": 1.2,
            "sigma_diagonal_pin": 1.0,
        }

    def test_defaults_on_and_recorded(self, spark, monkeypatch):
        import numpy as np
        import spark_vi.models.topic.spectral_init as si_mod
        from spark_vi.mllib.topic.stm import OnlineSTMEstimator
        # toy_df is too small for the real anchor-word init; uniform seed keeps the
        # default-on fit robust while still exercising the spectralInit=True path.
        monkeypatch.setattr(si_mod, "spectral_init_beta",
                            lambda docs, partition, V: np.full((partition.K, V), 1.0 / V))
        est = OnlineSTMEstimator(
            k=4, featuresCol="features", covariatesCol="covariates",
            covariateNames=["a", "b"], seed=0, maxIter=2, subsamplingRate=1.0)
        assert est.getOrDefault("referenceTopic") is True
        assert est.getOrDefault("spectralInit") is True
        model = est.fit(self._toy_df(spark))
        assert model.result.metadata["stm_hardening"] == {
            "reference_topic": True,
            "min_pair_support": 1,
            "spectral_init": True,
            "spectral_method": "dense",
            "spectral_method_requested": "auto",
            "estimate_sigma_diagonal": False,
            "estimate_global_scale": False,
            "global_scale_step_cap": 1.2,
            "sigma_diagonal_pin": 1.0,
        }
        # Default fit now pins the reference column to zero.
        assert np.allclose(model.result.global_params["Gamma"][:, 0], 0.0)

    def test_spectral_init_seed_reaches_initialize_global(self, spark, monkeypatch):
        """spectralInit=True computes a spectral_beta seed from the collected
        docs and passes it to the engine as data_summary={'spectral_beta': ...}.

        We spy on spectral_init_beta (uniform seed, so no zero-rows can trip the
        digamma in the first E-step) and on OnlineSTM.initialize_global to capture
        the data_summary the runner forwards — the end-to-end wiring seam."""
        import spark_vi.models.topic.spectral_init as si_mod
        from spark_vi.models.topic.stm import OnlineSTM
        from spark_vi.mllib.topic.stm import OnlineSTMEstimator

        K, V = 4, 8
        sentinel = np.full((K, V), 1.0 / V)
        seen = {}

        def fake_spectral(docs, partition, vocab_size):
            seen["vocab_size"] = vocab_size
            seen["partition_K"] = partition.K
            seen["n_docs"] = len(docs)
            return sentinel

        monkeypatch.setattr(si_mod, "spectral_init_beta", fake_spectral)

        captured = {}
        orig_init = OnlineSTM.initialize_global

        def spy_init(self, data_summary):
            captured["data_summary"] = data_summary
            return orig_init(self, data_summary)

        monkeypatch.setattr(OnlineSTM, "initialize_global", spy_init)

        est = OnlineSTMEstimator(
            k=K, featuresCol="features", covariatesCol="covariates",
            covariateNames=["a", "b"], seed=0, spectralInit=True,
            maxIter=1, subsamplingRate=1.0)
        model = est.fit(self._toy_df(spark))

        assert seen["vocab_size"] == V
        assert seen["partition_K"] == K
        assert seen["n_docs"] == 6
        ds = captured["data_summary"]
        assert ds is not None and "spectral_beta" in ds
        assert np.allclose(ds["spectral_beta"], sentinel)
        assert model.result.metadata["stm_hardening"]["spectral_init"] is True

    def test_spectral_init_off_forwards_no_seed(self, spark, monkeypatch):
        """spectralInit off: initialize_global receives data_summary None
        (random-gamma init path), and the metadata records it off."""
        from spark_vi.models.topic.stm import OnlineSTM
        from spark_vi.mllib.topic.stm import OnlineSTMEstimator

        captured = {}
        orig_init = OnlineSTM.initialize_global

        def spy_init(self, data_summary):
            captured["data_summary"] = data_summary
            return orig_init(self, data_summary)

        monkeypatch.setattr(OnlineSTM, "initialize_global", spy_init)

        est = OnlineSTMEstimator(
            k=4, featuresCol="features", covariatesCol="covariates",
            covariateNames=["a", "b"], seed=0, spectralInit=False,
            maxIter=1, subsamplingRate=1.0)
        assert est.getOrDefault("spectralInit") is False
        model = est.fit(self._toy_df(spark))
        assert captured["data_summary"] is None
        assert model.result.metadata["stm_hardening"]["spectral_init"] is False

    def test_spectral_method_invalid_rejected(self):
        """spectralMethod values outside {"dense", "scalable", "auto"} raise."""
        from spark_vi.mllib.topic.stm import OnlineSTMEstimator
        with pytest.raises(Exception, match="spectralMethod"):
            OnlineSTMEstimator(
                k=4, featuresCol="features", covariatesCol="covariates",
                covariateNames=["a", "b"], spectralMethod="bogus")

    def test_spectral_method_default_is_auto(self):
        """A default-constructed estimator holds spectralMethod == 'auto'
        (ADR 0037: auto-route dense/scalable by vocab size)."""
        from spark_vi.mllib.topic.stm import OnlineSTMEstimator
        est = OnlineSTMEstimator(
            k=4, featuresCol="features", covariatesCol="covariates",
            covariateNames=["a", "b"])
        assert est.getOrDefault("spectralMethod") == "auto"

    def test_resolve_spectral_method_auto_below_threshold_is_dense(self):
        """'auto' resolves to dense below the vocab threshold (ADR 0037)."""
        from spark_vi.mllib.topic.stm import (
            resolve_spectral_method, SPECTRAL_AUTO_VOCAB_THRESHOLD)
        assert resolve_spectral_method("auto", SPECTRAL_AUTO_VOCAB_THRESHOLD - 1) == "dense"

    def test_resolve_spectral_method_auto_at_threshold_is_scalable(self):
        """'auto' resolves to scalable at/above the vocab threshold (ADR 0037)."""
        from spark_vi.mllib.topic.stm import (
            resolve_spectral_method, SPECTRAL_AUTO_VOCAB_THRESHOLD)
        assert resolve_spectral_method("auto", SPECTRAL_AUTO_VOCAB_THRESHOLD) == "scalable"
        assert resolve_spectral_method("auto", SPECTRAL_AUTO_VOCAB_THRESHOLD + 5000) == "scalable"

    def test_resolve_spectral_method_explicit_passes_through(self):
        """Explicit 'dense'/'scalable' ignore the threshold and pass through."""
        from spark_vi.mllib.topic.stm import resolve_spectral_method
        # A huge V does NOT flip an explicit 'dense'; a tiny V does NOT flip 'scalable'.
        assert resolve_spectral_method("dense", 10_000_000) == "dense"
        assert resolve_spectral_method("scalable", 10) == "scalable"

    def test_resolve_spectral_method_custom_threshold(self):
        """The threshold is a tunable argument, not hard-wired."""
        from spark_vi.mllib.topic.stm import resolve_spectral_method
        assert resolve_spectral_method("auto", 500, threshold=1000) == "dense"
        assert resolve_spectral_method("auto", 1500, threshold=1000) == "scalable"

    def test_spectral_method_scalable_routes_to_scalable_init(
            self, spark, monkeypatch):
        """spectralMethod='scalable' calls scalable_spectral_init_beta (not the
        dense path): the stub receives the rdd (not a collected list), and the
        metadata records stm_hardening['spectral_method'] == 'scalable'."""
        import spark_vi.models.topic.spectral_init_scalable as ss_mod
        from spark_vi.models.topic.stm import OnlineSTM
        from spark_vi.mllib.topic.stm import OnlineSTMEstimator

        K, V = 4, 8
        sentinel = np.full((K, V), 1.0 / V)
        seen = {}

        def fake_scalable(rdd, partition, vocab_size, *, d=None, seed=0,
                          min_doc_freq=5):
            seen["rdd"] = rdd
            seen["vocab_size"] = vocab_size
            seen["d"] = d
            seen["seed"] = seed
            seen["min_doc_freq"] = min_doc_freq
            return sentinel

        monkeypatch.setattr(ss_mod, "scalable_spectral_init_beta", fake_scalable)

        # Also spy on initialize_global to confirm the seed reaches the engine.
        captured = {}
        orig_init = OnlineSTM.initialize_global

        def spy_init(self, data_summary):
            captured["data_summary"] = data_summary
            return orig_init(self, data_summary)

        monkeypatch.setattr(OnlineSTM, "initialize_global", spy_init)

        est = OnlineSTMEstimator(
            k=K, featuresCol="features", covariatesCol="covariates",
            covariateNames=["a", "b"], seed=7,
            spectralInit=True, spectralMethod="scalable",
            spectralD=256, spectralMinDocFreq=3, maxIter=1, subsamplingRate=1.0)
        model = est.fit(self._toy_df(spark))

        # The stub received the rdd (not a Python list).
        import pyspark
        assert isinstance(seen["rdd"], pyspark.RDD), (
            "scalable path must pass the rdd, not rdd.collect()")
        assert seen["vocab_size"] == V
        assert seen["d"] == 256
        assert seen["seed"] == 7          # seed or 0
        assert seen["min_doc_freq"] == 3

        # The sentinel reached the engine unchanged.
        ds = captured["data_summary"]
        assert ds is not None and "spectral_beta" in ds
        assert np.allclose(ds["spectral_beta"], sentinel)

        # Metadata records the method.
        assert model.result.metadata["stm_hardening"]["spectral_method"] == "scalable"

    def test_defaults_on_and_recorded_includes_spectral_method(
            self, spark, monkeypatch):
        """Default path records spectral_method == 'dense' in stm_hardening."""
        import spark_vi.models.topic.spectral_init as si_mod
        from spark_vi.mllib.topic.stm import OnlineSTMEstimator
        monkeypatch.setattr(si_mod, "spectral_init_beta",
                            lambda docs, partition, V: np.full((partition.K, V), 1.0 / V))
        est = OnlineSTMEstimator(
            k=4, featuresCol="features", covariatesCol="covariates",
            covariateNames=["a", "b"], seed=0, maxIter=2, subsamplingRate=1.0)
        model = est.fit(self._toy_df(spark))
        assert model.result.metadata["stm_hardening"]["spectral_method"] == "dense"


# ---------------------------------------------------------------------------
# Fixture: tiny dataset for full-Sigma / correlation tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_stm_dataset(spark):
    """Six-document, 8-vocab corpus with two covariates."""
    from pyspark.ml.linalg import SparseVector, DenseVector
    rows = [
        (SparseVector(8, [0, 2], [3.0, 1.0]), DenseVector([1.0, 0.0])),
        (SparseVector(8, [1, 3], [2.0, 2.0]), DenseVector([0.0, 1.0])),
        (SparseVector(8, [0, 4], [1.0, 2.0]), DenseVector([1.0, 0.5])),
        (SparseVector(8, [5, 6], [1.0, 1.0]), DenseVector([0.0, 1.0])),
        (SparseVector(8, [2, 7], [2.0, 1.0]), DenseVector([1.0, 0.2])),
        (SparseVector(8, [1, 6], [1.0, 3.0]), DenseVector([0.0, 0.8])),
    ]
    return spark.createDataFrame(rows, ["features", "covariates"])


def test_estimator_full_sigma_metadata_and_shapes(spark, tiny_stm_dataset):
    from spark_vi.mllib.topic.stm import OnlineSTMEstimator
    est = OnlineSTMEstimator(k=4, featuresCol="features", covariatesCol="covariates",
                             covariateNames=["a", "b"],
                             minPairSupport=3,
                             referenceTopic=False, spectralInit=False,
                             maxIter=2, subsamplingRate=1.0)
    model = est.fit(tiny_stm_dataset)
    assert model.result.global_params["Sigma"].shape == (4, 4)
    assert model.result.metadata["stm_hardening"]["min_pair_support"] == 3
