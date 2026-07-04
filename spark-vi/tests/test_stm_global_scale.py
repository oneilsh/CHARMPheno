import numpy as np
import pytest
from spark_vi.models.topic.stm import OnlineSTM


def _drive_mstep(m, gp, S, N, lr=1.0):
    """Call update_global with a planted scatter S and support N (the only Sigma
    inputs), returning the new Sigma. Mirrors test_stm_blockwise._drive_mstep.
    Recall mle = S/N, so to plant a desired diag_var/correlation, scale S by N."""
    K, V, P = m.K, m.V, m.P
    stats = {
        "residual_outer_stat": S,
        "n_pairs_stat": N,
        "lambda_stats": np.zeros((K, V)),
        "XtX": np.zeros((P, P)),
        "XtX_groups": [np.zeros((P, P)) for _ in m._effective_partition().groups],
        "XtMu": np.zeros((P, K)),
        "n_docs_per_topic": np.ones(K),
    }
    return m.update_global(gp, stats, learning_rate=lr)["Sigma"]


def test_global_scale_is_pooled_and_unit_correlation():
    # K=3, all self-supported (N=50 >= min_pair_support), diag_var=[4,4,4], and a
    # nonzero off-diagonal correlation r_01=0.5 (mle_01 = r*sqrt(mle_00*mle_11) = 2.0).
    # mle = S/N, so S_ii = diag_var_i * N_ii and S_01 = mle_01 * N_01.
    m = OnlineSTM(K=3, vocab_size=8, P=1, random_seed=0,
                  estimate_global_scale=True)
    gp = m.initialize_global(None)
    gp["Sigma"] = np.eye(3) * 4.0   # tau2_prev-consistent starting Sigma
    N = np.full((3, 3), 50.0)
    S = np.array([
        [200.0, 100.0, 0.0],
        [100.0, 200.0, 0.0],
        [0.0, 0.0, 200.0],
    ])
    Sig = _drive_mstep(m, gp, S, N, lr=1.0)
    # (a) diagonal is pooled/uniform, at the pooled tau2 (mean of the three diag_var's)
    np.testing.assert_allclose(np.diag(Sig), np.diag(Sig)[0], atol=1e-9)
    np.testing.assert_allclose(np.diag(Sig), 4.0, atol=1e-9)
    # (b) recovered correlation matches the input R off-diagonal (0.5)
    recovered_r01 = Sig[0, 1] / np.sqrt(Sig[0, 0] * Sig[1, 1])
    assert abs(recovered_r01 - 0.5) < 1e-9


def test_pooling_is_runaway_safe():
    # One topic (2) has a blown-up residual variance mle_22=1e6 (mimics ess~15 ->
    # Sigma_var~1e6, exp 0032 / insight 0036: the estimate_sigma_diagonal branch
    # would place ~1e6 directly on that topic's own diagonal). Pooling averages the
    # outlier against the other two topics' variance (~4.0 each), and the damping
    # cap additionally bounds a single step's growth to at most tau2_prev * cap.
    m = OnlineSTM(K=3, vocab_size=8, P=1, random_seed=0,
                  estimate_global_scale=True, global_scale_step_cap=1.2)
    gp = m.initialize_global(None)
    tau2_prev = 4.0
    gp["Sigma"] = np.eye(3) * tau2_prev
    N = np.full((3, 3), 50.0)
    S = np.diag([4.0 * 50.0, 4.0 * 50.0, 1.0e6 * 50.0])
    Sig = _drive_mstep(m, gp, S, N, lr=1.0)
    pooled = float(np.mean(np.diag(Sig)))
    assert pooled <= tau2_prev * 1.2 + 1e-9
    assert pooled < 1000.0   # nowhere near the ~1e6 the outlier alone would produce


def test_damping_cap_bounds_growth():
    m = OnlineSTM(K=2, vocab_size=8, P=1, random_seed=0,
                  estimate_global_scale=True, global_scale_step_cap=1.2)
    gp = m.initialize_global(None)
    gp["Sigma"] = np.eye(2) * 1.0   # tau2_prev = 1.0, Sigma starts = R_unit
    N = np.full((2, 2), 50.0)
    S = np.diag([100.0 * 50.0, 100.0 * 50.0])   # mle diag_var = 100 -> target wants a 100x jump

    Sig1 = _drive_mstep(m, gp, S, N, lr=1.0)
    assert np.mean(np.diag(Sig1)) <= 1.0 * 1.2 + 1e-9

    Sig_half = _drive_mstep(m, gp, S, N, lr=0.5)
    assert np.mean(np.diag(Sig_half)) < np.mean(np.diag(Sig1))

    # Second identical step from the new (capped) Sigma grows by <=1.2x again,
    # i.e. a geometric climb toward the target rather than one explosive jump.
    gp2 = {**gp, "Sigma": Sig1}
    Sig2 = _drive_mstep(m, gp2, S, N, lr=1.0)
    prev_tau2 = float(np.mean(np.diag(Sig1)))
    assert float(np.mean(np.diag(Sig2))) <= prev_tau2 * 1.2 + 1e-9


def test_reference_topic_excluded_from_pool():
    # Topic 0 is degenerate (mle_00=0, the reference topic's eta=0 residual);
    # topics 1,2 have mle_ii=9.0. Pooled tau2 must be ~9.0 (topic 0 excluded from
    # the pool), NOT ~6.0 (which averaging the 0 in with the other two would give).
    m = OnlineSTM(K=3, vocab_size=8, P=1, random_seed=0,
                  estimate_global_scale=True)
    gp = m.initialize_global(None)
    gp["Sigma"] = np.eye(3) * 9.0
    N = np.full((3, 3), 50.0)
    S = np.diag([0.0, 9.0 * 50.0, 9.0 * 50.0])
    Sig = _drive_mstep(m, gp, S, N, lr=1.0)
    pooled = float(np.mean(np.diag(Sig)))
    assert abs(pooled - 9.0) < 1e-6
    assert abs(pooled - 6.0) > 1.0


def test_mutually_exclusive_with_sigma_diagonal():
    with pytest.raises(ValueError):
        OnlineSTM(K=3, vocab_size=8, P=1, random_seed=0,
                  estimate_sigma_diagonal=True, estimate_global_scale=True)


class TestShimForwardsGlobalScale:
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

    def test_shim_metadata_and_forward(self, spark):
        from spark_vi.mllib.topic.stm import StreamingSTM
        est = StreamingSTM(
            K=4, features_col="features", covariates_col="covariates",
            covariate_names=["a", "b"], random_seed=0,
            estimate_global_scale=True, global_scale_step_cap=1.5)
        assert est.estimate_global_scale is True
        assert est.global_scale_step_cap == 1.5
        model = est.fit(self._toy_df(spark), max_iter=2, subsampling_rate=1.0)
        assert model.metadata["stm_hardening"]["estimate_global_scale"] is True
        assert model.metadata["stm_hardening"]["global_scale_step_cap"] == 1.5
