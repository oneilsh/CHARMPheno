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

    def test_fit_resume_accumulates_iterations(self, spark, tmp_path):
        """fit -> save -> fit(resume_from): the iteration counter continues, so a
        3-iter fit then a 2-iter resume yields n_iterations == 5. This is what
        keeps rho_t = (tau0 + t + 1)^-kappa shrinking instead of resetting."""
        from spark_vi.mllib.topic.stm import StreamingSTM

        rows = [
            (SparseVector(8, [0, 2], [3.0, 1.0]), DenseVector([1.0, 0.0])),
            (SparseVector(8, [1, 3], [2.0, 2.0]), DenseVector([0.0, 1.0])),
            (SparseVector(8, [0, 4], [1.0, 2.0]), DenseVector([1.0, 0.5])),
            (SparseVector(8, [5, 6], [1.0, 1.0]), DenseVector([0.0, 1.0])),
        ]
        df = spark.createDataFrame(rows, ["features", "covariates"])
        est = StreamingSTM(
            K=2, features_col="features", covariates_col="covariates",
            covariate_names=["x1", "x2"], random_seed=0,
        )
        m1 = est.fit(df, max_iter=3, subsampling_rate=1.0, tau0=1.0, kappa=0.5)
        assert m1.n_iterations == 3

        save_dir = tmp_path / "ckpt"
        m1.save(save_dir)

        m2 = est.fit(df, max_iter=2, subsampling_rate=1.0, tau0=1.0, kappa=0.5,
                     resume_from=str(save_dir))
        assert m2.n_iterations == 5  # 3 loaded + 2 additional


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


def test_streaming_stm_rejects_group_var_in_formula():
    import pytest
    from spark_vi.mllib.topic.stm import StreamingSTM
    from spark_vi.models.topic.partition import TopicBlockPartition
    part = TopicBlockPartition("source_cohort", background_k=2, foreground=(("cancer", 1),))
    with pytest.raises(ValueError, match="group_var"):
        StreamingSTM(
            K=3, covariates_col="covariates",
            covariate_names=["Intercept", "C(source_cohort)[T.cancer]"],
            topic_blocks=part, doc_group_col="source_cohort")


def test_streaming_stm_requires_group_col_with_partition():
    import pytest
    from spark_vi.mllib.topic.stm import StreamingSTM
    from spark_vi.models.topic.partition import TopicBlockPartition
    part = TopicBlockPartition("source_cohort", background_k=2, foreground=(("cancer", 1),))
    with pytest.raises(ValueError, match="doc_group_col"):
        StreamingSTM(K=3, covariates_col="covariates",
                     covariate_names=["Intercept", "age"], topic_blocks=part)


def test_streaming_stm_gated_fit_smoke(spark):
    from pyspark.ml.linalg import Vectors
    from spark_vi.mllib.topic.stm import StreamingSTM
    from spark_vi.models.topic.partition import TopicBlockPartition
    part = TopicBlockPartition("grp", background_k=2, foreground=(("rare", 1),))
    rows = []
    for i in range(20):
        rare = i % 4 == 0
        rows.append((Vectors.sparse(5, {i % 5: 2.0, (i + 1) % 5: 1.0}),
                     Vectors.dense([1.0, float(i % 2)]),
                     "rare" if rare else "common"))
    df = spark.createDataFrame(rows, ["features", "covariates", "grp"])
    est = StreamingSTM(K=3, covariates_col="covariates",
                       covariate_names=["Intercept", "age"],
                       topic_blocks=part, doc_group_col="grp")
    model = est.fit(df, max_iter=3, subsampling_rate=1.0)
    assert model.global_params["lambda"].shape == (3, 5)
    assert model.metadata  # fit produced a model


class TestStreamingSTMHardeningThreading:
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
        end-to-end signature that reference_topic took effect through the shim."""
        from spark_vi.mllib.topic.stm import StreamingSTM
        est = StreamingSTM(
            K=4, features_col="features", covariates_col="covariates",
            covariate_names=["a", "b"], random_seed=0, reference_topic=True)
        model = est.fit(self._toy_df(spark), max_iter=3, subsampling_rate=1.0)
        assert np.allclose(model.global_params["Gamma"][:, 0], 0.0)

    def test_hardening_knobs_persisted_in_metadata(self, spark):
        from spark_vi.mllib.topic.stm import StreamingSTM
        est = StreamingSTM(
            K=4, features_col="features", covariates_col="covariates",
            covariate_names=["a", "b"], random_seed=0,
            reference_topic=True, sigma_prior_scale=2.0, sigma_prior_count=500.0,
            spectral_init=False)
        model = est.fit(self._toy_df(spark), max_iter=2, subsampling_rate=1.0)
        assert model.metadata["stm_hardening"] == {
            "reference_topic": True,
            "sigma_prior_scale": 2.0,
            "sigma_prior_count": 500.0,
            "sigma_diag_shrink": 0.0,
            "min_pair_support": 1,
            "spectral_init": False,
            "spectral_method": "dense",
        }

    def test_defaults_on_and_recorded(self, spark, monkeypatch):
        import numpy as np
        import spark_vi.models.topic.spectral_init as si_mod
        from spark_vi.mllib.topic.stm import StreamingSTM
        # toy_df is too small for the real anchor-word init; uniform seed keeps the
        # default-on fit robust while still exercising the spectral_init=True path.
        monkeypatch.setattr(si_mod, "spectral_init_beta",
                            lambda docs, partition, V: np.full((partition.K, V), 1.0 / V))
        est = StreamingSTM(
            K=4, features_col="features", covariates_col="covariates",
            covariate_names=["a", "b"], random_seed=0)
        assert est.reference_topic is True
        assert est.spectral_init is True
        model = est.fit(self._toy_df(spark), max_iter=2, subsampling_rate=1.0)
        assert model.metadata["stm_hardening"] == {
            "reference_topic": True,
            "sigma_prior_scale": None,
            "sigma_prior_count": 0.0,
            "sigma_diag_shrink": 0.0,
            "min_pair_support": 1,
            "spectral_init": True,
            "spectral_method": "dense",
        }
        # Default fit now pins the reference column to zero.
        assert np.allclose(model.global_params["Gamma"][:, 0], 0.0)

    def test_spectral_init_seed_reaches_initialize_global(self, spark, monkeypatch):
        """spectral_init=True computes a spectral_beta seed from the collected
        docs and passes it to the engine as data_summary={'spectral_beta': ...}.

        We spy on spectral_init_beta (uniform seed, so no zero-rows can trip the
        digamma in the first E-step) and on OnlineSTM.initialize_global to capture
        the data_summary the runner forwards — the end-to-end wiring seam."""
        import spark_vi.models.topic.spectral_init as si_mod
        from spark_vi.models.topic.stm import OnlineSTM
        from spark_vi.mllib.topic.stm import StreamingSTM

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

        est = StreamingSTM(
            K=K, features_col="features", covariates_col="covariates",
            covariate_names=["a", "b"], random_seed=0, spectral_init=True)
        model = est.fit(self._toy_df(spark), max_iter=1, subsampling_rate=1.0)

        assert seen["vocab_size"] == V
        assert seen["partition_K"] == K
        assert seen["n_docs"] == 6
        ds = captured["data_summary"]
        assert ds is not None and "spectral_beta" in ds
        assert np.allclose(ds["spectral_beta"], sentinel)
        assert model.metadata["stm_hardening"]["spectral_init"] is True

    def test_spectral_init_off_forwards_no_seed(self, spark, monkeypatch):
        """Default (spectral_init off): initialize_global receives data_summary
        None (random-gamma init path), and the metadata records it off."""
        from spark_vi.models.topic.stm import OnlineSTM
        from spark_vi.mllib.topic.stm import StreamingSTM

        captured = {}
        orig_init = OnlineSTM.initialize_global

        def spy_init(self, data_summary):
            captured["data_summary"] = data_summary
            return orig_init(self, data_summary)

        monkeypatch.setattr(OnlineSTM, "initialize_global", spy_init)

        est = StreamingSTM(
            K=4, features_col="features", covariates_col="covariates",
            covariate_names=["a", "b"], random_seed=0, spectral_init=False)
        assert est.spectral_init is False
        model = est.fit(self._toy_df(spark), max_iter=1, subsampling_rate=1.0)
        assert captured["data_summary"] is None
        assert model.metadata["stm_hardening"]["spectral_init"] is False

    def test_spectral_method_invalid_rejected(self):
        """spectral_method values outside {"dense", "scalable"} raise ValueError."""
        from spark_vi.mllib.topic.stm import StreamingSTM
        with pytest.raises(ValueError, match="spectral_method"):
            StreamingSTM(
                K=4, features_col="features", covariates_col="covariates",
                covariate_names=["a", "b"], spectral_method="bogus")

    def test_spectral_method_default_is_dense(self):
        """A default-constructed StreamingSTM stores spectral_method == 'dense'."""
        from spark_vi.mllib.topic.stm import StreamingSTM
        est = StreamingSTM(
            K=4, features_col="features", covariates_col="covariates",
            covariate_names=["a", "b"])
        assert est.spectral_method == "dense"

    def test_spectral_method_scalable_routes_to_scalable_init(
            self, spark, monkeypatch):
        """spectral_method='scalable' calls scalable_spectral_init_beta (not the
        dense path): the stub receives the rdd (not a collected list), and the
        metadata records stm_hardening['spectral_method'] == 'scalable'."""
        import spark_vi.models.topic.spectral_init_scalable as ss_mod
        from spark_vi.models.topic.stm import OnlineSTM
        from spark_vi.mllib.topic.stm import StreamingSTM

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

        est = StreamingSTM(
            K=K, features_col="features", covariates_col="covariates",
            covariate_names=["a", "b"], random_seed=7,
            spectral_init=True, spectral_method="scalable",
            spectral_d=256, spectral_min_doc_freq=3)
        model = est.fit(self._toy_df(spark), max_iter=1, subsampling_rate=1.0)

        # The stub received the rdd (not a Python list).
        import pyspark
        assert isinstance(seen["rdd"], pyspark.RDD), (
            "scalable path must pass the rdd, not rdd.collect()")
        assert seen["vocab_size"] == V
        assert seen["d"] == 256
        assert seen["seed"] == 7          # self.random_seed or 0
        assert seen["min_doc_freq"] == 3

        # The sentinel reached the engine unchanged.
        ds = captured["data_summary"]
        assert ds is not None and "spectral_beta" in ds
        assert np.allclose(ds["spectral_beta"], sentinel)

        # Metadata records the method.
        assert model.metadata["stm_hardening"]["spectral_method"] == "scalable"

    def test_defaults_on_and_recorded_includes_spectral_method(
            self, spark, monkeypatch):
        """Default path records spectral_method == 'dense' in stm_hardening."""
        import spark_vi.models.topic.spectral_init as si_mod
        from spark_vi.mllib.topic.stm import StreamingSTM
        monkeypatch.setattr(si_mod, "spectral_init_beta",
                            lambda docs, partition, V: np.full((partition.K, V), 1.0 / V))
        est = StreamingSTM(
            K=4, features_col="features", covariates_col="covariates",
            covariate_names=["a", "b"], random_seed=0)
        model = est.fit(self._toy_df(spark), max_iter=2, subsampling_rate=1.0)
        assert model.metadata["stm_hardening"]["spectral_method"] == "dense"


# ---------------------------------------------------------------------------
# Fixture: tiny dataset for full-Sigma / correlation tests (Path B, formula)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_stm_dataset(spark):
    """Six-document, 8-vocab corpus with a single binary covariate 'x'."""
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


def test_streaming_stm_full_sigma_metadata_and_shapes(spark, tiny_stm_dataset):
    from spark_vi.mllib.topic.stm import StreamingSTM
    est = StreamingSTM(K=4, features_col="features", covariates_col="covariates",
                       covariate_names=["a", "b"],
                       sigma_diag_shrink=0.0, min_pair_support=3,
                       reference_topic=False, spectral_init=False)
    model = est.fit(tiny_stm_dataset, max_iter=2, subsampling_rate=1.0)
    assert model.global_params["Sigma"].shape == (4, 4)
    assert model.metadata["stm_hardening"]["min_pair_support"] == 3
    assert model.metadata["stm_hardening"]["sigma_diag_shrink"] == 0.0
