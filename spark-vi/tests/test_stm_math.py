"""Tests for STM per-doc inference math: STMDocument, gradient, Hessian, MAP."""
from __future__ import annotations

import numpy as np
import pytest

from spark_vi.models.topic.types import STMDocument


class TestSTMDocument:
    def test_constructs_with_indices_counts_length_x(self):
        doc = STMDocument(
            indices=np.array([0, 3, 5], dtype=np.int32),
            counts=np.array([2.0, 1.0, 3.0], dtype=np.float64),
            length=6,
            x=np.array([1.0, 0.5, -1.2], dtype=np.float64),
        )
        assert doc.length == 6
        assert doc.x.shape == (3,)
        assert doc.x.dtype == np.float64

    def test_is_frozen(self):
        doc = STMDocument(
            indices=np.array([0], dtype=np.int32),
            counts=np.array([1.0]),
            length=1,
            x=np.array([0.0]),
        )
        with pytest.raises((AttributeError, TypeError)):
            doc.length = 99

    def test_stmdocument_groups_defaults_to_empty_frozenset(self):
        d = STMDocument(indices=np.array([0], dtype=np.int32),
                        counts=np.array([1.0]), length=1, x=np.array([1.0]))
        assert d.groups == frozenset()

    def test_stmdocument_carries_groups(self):
        d = STMDocument(indices=np.array([0], dtype=np.int32),
                        counts=np.array([1.0]), length=1, x=np.array([1.0]),
                        groups=frozenset({"cancer"}))
        assert d.groups == frozenset({"cancer"})

    def test_vector_to_stm_document_extracts_group_from_column(self):
        from pyspark.ml.linalg import Vectors
        from spark_vi.mllib.topic._common import _vector_to_stm_document
        row = {"features": Vectors.sparse(3, {0: 2.0}),
               "covariates": Vectors.dense([1.0, 0.5]),
               "source_cohort": "dementia"}
        doc = _vector_to_stm_document(row, group_col="source_cohort")
        assert doc.groups == frozenset({"dementia"})

    def test_vector_to_stm_document_no_group_col_yields_empty(self):
        from pyspark.ml.linalg import Vectors
        from spark_vi.mllib.topic._common import _vector_to_stm_document
        row = {"features": Vectors.sparse(3, {0: 2.0}),
               "covariates": Vectors.dense([1.0, 0.5])}
        doc = _vector_to_stm_document(row)
        assert doc.groups == frozenset()


from scipy.special import softmax

from spark_vi.models.topic.stm import _stm_neg_log_joint, _stm_neg_log_joint_grad


def _make_small_doc_state(seed=0):
    """Construct a small (K, V, P) state for gradient / Hessian tests."""
    rng = np.random.default_rng(seed)
    K, V, P = 3, 5, 2
    # ExpElogbeta-style nonnegative K x V matrix, columns summing roughly to 1.
    expElogbeta = rng.gamma(shape=2.0, scale=1.0, size=(K, V))
    expElogbeta = expElogbeta / expElogbeta.sum(axis=0, keepdims=True)
    Gamma = rng.normal(size=(P, K))
    Sigma_diag = rng.gamma(shape=2.0, scale=0.5, size=K)
    Sigma_inv = np.diag(1.0 / Sigma_diag)   # full (K,K) precision from the diagonal
    x = rng.normal(size=P)
    indices = np.array([0, 2, 4], dtype=np.int32)
    counts = np.array([2.0, 1.0, 3.0], dtype=np.float64)
    return dict(
        K=K, V=V, P=P, expElogbeta=expElogbeta, Gamma=Gamma,
        Sigma_diag=Sigma_diag, Sigma_inv=Sigma_inv,
        x=x, indices=indices, counts=counts,
    )


class TestSTMGradient:
    def test_gradient_matches_finite_difference(self):
        st = _make_small_doc_state(seed=42)
        rng = np.random.default_rng(0)
        eta = rng.normal(size=st["K"]) * 0.3

        analytic = _stm_neg_log_joint_grad(
            eta,
            indices=st["indices"], counts=st["counts"],
            expElogbeta=st["expElogbeta"],
            Gamma=st["Gamma"], Sigma_inv=st["Sigma_inv"], x=st["x"],
        )

        eps = 1e-6
        numeric = np.zeros_like(eta)
        for k in range(st["K"]):
            eta_p = eta.copy(); eta_p[k] += eps
            eta_m = eta.copy(); eta_m[k] -= eps
            f_p = _stm_neg_log_joint(
                eta_p, indices=st["indices"], counts=st["counts"],
                expElogbeta=st["expElogbeta"],
                Gamma=st["Gamma"], Sigma_inv=st["Sigma_inv"], x=st["x"],
            )
            f_m = _stm_neg_log_joint(
                eta_m, indices=st["indices"], counts=st["counts"],
                expElogbeta=st["expElogbeta"],
                Gamma=st["Gamma"], Sigma_inv=st["Sigma_inv"], x=st["x"],
            )
            numeric[k] = (f_p - f_m) / (2 * eps)

        np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-6)


from spark_vi.models.topic.stm import _stm_neg_log_joint_hessian


class TestSTMHessian:
    def test_hessian_matches_finite_difference_of_grad(self):
        st = _make_small_doc_state(seed=7)
        rng = np.random.default_rng(1)
        eta = rng.normal(size=st["K"]) * 0.3

        analytic = _stm_neg_log_joint_hessian(
            eta, indices=st["indices"], counts=st["counts"],
            expElogbeta=st["expElogbeta"],
            Gamma=st["Gamma"], Sigma_inv=st["Sigma_inv"], x=st["x"],
        )

        eps = 1e-5
        numeric = np.zeros((st["K"], st["K"]))
        for j in range(st["K"]):
            eta_p = eta.copy(); eta_p[j] += eps
            eta_m = eta.copy(); eta_m[j] -= eps
            g_p = _stm_neg_log_joint_grad(
                eta_p, indices=st["indices"], counts=st["counts"],
                expElogbeta=st["expElogbeta"],
                Gamma=st["Gamma"], Sigma_inv=st["Sigma_inv"], x=st["x"],
            )
            g_m = _stm_neg_log_joint_grad(
                eta_m, indices=st["indices"], counts=st["counts"],
                expElogbeta=st["expElogbeta"],
                Gamma=st["Gamma"], Sigma_inv=st["Sigma_inv"], x=st["x"],
            )
            numeric[:, j] = (g_p - g_m) / (2 * eps)

        np.testing.assert_allclose(analytic, numeric, rtol=1e-3, atol=1e-5)

    def test_hessian_is_symmetric(self):
        st = _make_small_doc_state(seed=11)
        rng = np.random.default_rng(2)
        eta = rng.normal(size=st["K"]) * 0.3
        H = _stm_neg_log_joint_hessian(
            eta, indices=st["indices"], counts=st["counts"],
            expElogbeta=st["expElogbeta"],
            Gamma=st["Gamma"], Sigma_inv=st["Sigma_inv"], x=st["x"],
        )
        np.testing.assert_allclose(H, H.T, rtol=1e-12, atol=1e-12)

    def test_hessian_positive_definite_at_typical_point(self):
        st = _make_small_doc_state(seed=13)
        rng = np.random.default_rng(3)
        eta = rng.normal(size=st["K"]) * 0.3
        H = _stm_neg_log_joint_hessian(
            eta, indices=st["indices"], counts=st["counts"],
            expElogbeta=st["expElogbeta"],
            Gamma=st["Gamma"], Sigma_inv=st["Sigma_inv"], x=st["x"],
        )
        # H is PD at a typical interior point near the mode — the common case
        # _spd_inverse's Cholesky fast path relies on. NOTE: the objective is
        # NOT globally convex (data term = log-sum-exp minus log-sum-exp), so H
        # can be indefinite with a weak prior or an early L-BFGS stop; that case
        # is handled by _spd_inverse / ADR 0029, not asserted away here.
        eigs = np.linalg.eigvalsh(H)
        assert np.all(eigs > 0), f"Hessian not PD: eigs={eigs}"


from spark_vi.models.topic.stm import _stm_doc_inference


class TestSTMDocInference:
    def test_converges_to_stationary_point(self):
        st = _make_small_doc_state(seed=99)
        eta_hat, nu_d, n_iter = _stm_doc_inference(
            indices=st["indices"], counts=st["counts"],
            expElogbeta=st["expElogbeta"],
            Gamma=st["Gamma"], Sigma_inv=st["Sigma_inv"], x=st["x"],
            max_iter=200, tol=1e-6,
        )
        # Gradient at η̂ should be ~zero.
        g = _stm_neg_log_joint_grad(
            eta_hat, indices=st["indices"], counts=st["counts"],
            expElogbeta=st["expElogbeta"],
            Gamma=st["Gamma"], Sigma_inv=st["Sigma_inv"], x=st["x"],
        )
        assert np.linalg.norm(g) < 1e-4, f"|g|={np.linalg.norm(g)} not converged"
        assert nu_d.shape == (st["K"], st["K"])
        # ν_d is symmetric positive definite.
        np.testing.assert_allclose(nu_d, nu_d.T, atol=1e-10)
        eigs = np.linalg.eigvalsh(nu_d)
        assert np.all(eigs > 0)

    def test_strong_prior_pulls_eta_toward_prior_mean(self):
        st = _make_small_doc_state(seed=1)
        # Override Σ to be very tight: posterior should ~= prior mean Γᵀx.
        Sigma_inv_tight = np.diag(np.full(st["K"], 1e6))   # precision = 1/1e-6
        prior_mean = st["Gamma"].T @ st["x"]
        eta_hat, _, _ = _stm_doc_inference(
            indices=st["indices"], counts=st["counts"],
            expElogbeta=st["expElogbeta"],
            Gamma=st["Gamma"], Sigma_inv=Sigma_inv_tight, x=st["x"],
            max_iter=200, tol=1e-8,
        )
        np.testing.assert_allclose(eta_hat, prior_mean, atol=1e-3)


from spark_vi.models.topic.stm import _spd_inverse


class TestSPDInverse:
    """Guard on the per-doc Laplace covariance ν = H⁻¹.

    The neg-log-joint is not globally convex (data term = log-sum-exp minus
    log-sum-exp), so H at the L-BFGS point can be non-PD with a weak prior or an
    early stop. The guard must keep the common PD path identical to inv(H) and
    repair the non-PD case into an SPD inverse rather than returning negative
    variances.
    """

    def test_matches_inv_for_pd(self):
        rng = np.random.default_rng(0)
        M = rng.normal(size=(4, 4))
        H = M @ M.T + np.eye(4)  # SPD
        np.testing.assert_allclose(_spd_inverse(H), np.linalg.inv(H), atol=1e-10)

    def test_repairs_indefinite_to_spd(self):
        # Symmetric but indefinite (a negative eigenvalue): raw inv would yield a
        # non-SPD "covariance"; the guard must return an SPD inverse.
        rng = np.random.default_rng(1)
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        H = Q @ np.diag([2.0, -0.5, 1.0]) @ Q.T
        nu = _spd_inverse(H)
        np.testing.assert_allclose(nu, nu.T, atol=1e-12)
        assert np.all(np.linalg.eigvalsh(nu) > 0), "repaired inverse not SPD"

    def test_repairs_singular_to_finite_spd(self):
        rng = np.random.default_rng(2)
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        H = Q @ np.diag([1.0, 0.0, 3.0]) @ Q.T  # PSD but singular
        nu = _spd_inverse(H)
        assert np.all(np.isfinite(nu))
        assert np.all(np.linalg.eigvalsh(nu) > 0)


from spark_vi.models.topic.stm import prior_topic_proportions


class TestPriorTopicProportions:
    """Covariate-implied (prior) topic proportions: softmax(Γᵀ x) for one doc.

    This is the per-document primitive the dashboard's corpus-mean α-equivalent
    averages over. It is the prior mean of η_d pushed through softmax — the
    topic mix a document is expected to have from its covariates alone, before
    any token evidence.
    """

    def test_matches_softmax_of_gamma_transpose_x(self):
        rng = np.random.default_rng(7)
        P, K = 2, 3
        Gamma = rng.normal(size=(P, K))
        x = rng.normal(size=P)

        result = prior_topic_proportions(Gamma, x)

        np.testing.assert_allclose(result, softmax(Gamma.T @ x))

    def test_is_a_probability_vector(self):
        rng = np.random.default_rng(8)
        Gamma = rng.normal(size=(4, 5))
        x = rng.normal(size=4)

        result = prior_topic_proportions(Gamma, x)

        assert result.shape == (5,)
        assert np.all(result >= 0.0)
        np.testing.assert_allclose(result.sum(), 1.0)

    def test_intercept_only_row_reproduces_gamma_intercept_softmax(self):
        # When x selects only the intercept (one-hot at the intercept row),
        # Γᵀx is exactly the intercept row, so the proportions equal
        # softmax(Γ[intercept]) — the v1 stand-in adapt_stm currently uses.
        rng = np.random.default_rng(9)
        P, K = 3, 4
        Gamma = rng.normal(size=(P, K))
        intercept_idx = 0
        x = np.zeros(P)
        x[intercept_idx] = 1.0

        result = prior_topic_proportions(Gamma, x)

        np.testing.assert_allclose(result, softmax(Gamma[intercept_idx]))


from spark_vi.models.topic.stm import corpus_mean_topic_proportions


class TestCorpusMeanTopicProportions:
    """The α-equivalent: (1/D) Σ_d softmax(Γᵀ x_d) over the design matrix X."""

    def test_equals_row_mean_of_per_row_proportions(self):
        rng = np.random.default_rng(11)
        D, P, K = 5, 2, 3
        Gamma = rng.normal(size=(P, K))
        X = rng.normal(size=(D, P))

        result = corpus_mean_topic_proportions(Gamma, X)

        expected = np.mean(
            [prior_topic_proportions(Gamma, X[d]) for d in range(D)], axis=0
        )
        np.testing.assert_allclose(result, expected)

    def test_is_a_probability_vector(self):
        rng = np.random.default_rng(12)
        Gamma = rng.normal(size=(3, 4))
        X = rng.normal(size=(7, 3))

        result = corpus_mean_topic_proportions(Gamma, X)

        assert result.shape == (4,)
        assert np.all(result >= 0.0)
        np.testing.assert_allclose(result.sum(), 1.0)

    def test_constant_covariates_reduce_to_single_row_softmax(self):
        # If every document shares the same covariate row, the corpus mean is
        # just that row's proportions — the nonlinearity has nothing to average.
        rng = np.random.default_rng(13)
        P, K = 3, 4
        Gamma = rng.normal(size=(P, K))
        x = rng.normal(size=P)
        X = np.tile(x, (6, 1))

        result = corpus_mean_topic_proportions(Gamma, X)

        np.testing.assert_allclose(result, prior_topic_proportions(Gamma, x))


class TestMaskedDocInference:
    def _setup(self, K=5, V=4):
        import numpy as np
        rng = np.random.default_rng(0)
        expElogbeta = rng.random((K, V)) + 0.1
        Gamma = np.zeros((2, K))
        Sigma_inv = np.eye(K)   # full (K,K) precision (unit-variance prior)
        indices = np.array([0, 2], dtype=np.int32)
        counts = np.array([3.0, 1.0])
        x = np.array([1.0, 0.0])
        return expElogbeta, Gamma, Sigma_inv, indices, counts, x

    def test_disallowed_topics_get_zero_theta(self):
        import numpy as np
        from spark_vi.models.topic.stm import _stm_doc_inference, _softmax
        eb, G, S, idx, cnt, x = self._setup()
        allowed = np.array([0, 1, 2], dtype=np.int64)  # topics 3,4 disallowed
        eta_hat, nu_d, _ = _stm_doc_inference(
            indices=idx, counts=cnt, expElogbeta=eb, Gamma=G,
            Sigma_inv=S, x=x, allowed=allowed)
        theta = _softmax(eta_hat)
        assert theta[3] == 0.0 and theta[4] == 0.0
        assert abs(theta[:3].sum() - 1.0) < 1e-9
        # nu_d zero on disallowed rows/cols
        assert np.all(nu_d[3, :] == 0.0) and np.all(nu_d[:, 4] == 0.0)

    def test_allowed_none_matches_full_inference(self):
        import numpy as np
        from spark_vi.models.topic.stm import _stm_doc_inference
        eb, G, S, idx, cnt, x = self._setup()
        a = _stm_doc_inference(indices=idx, counts=cnt, expElogbeta=eb,
                               Gamma=G, Sigma_inv=S, x=x, allowed=None)
        b = _stm_doc_inference(indices=idx, counts=cnt, expElogbeta=eb,
                               Gamma=G, Sigma_inv=S, x=x,
                               allowed=np.arange(eb.shape[0], dtype=np.int64))
        np.testing.assert_allclose(a[0], b[0], atol=1e-8)


def test_corpus_mean_topic_proportions_gated_zeros_out_of_group_foreground():
    import numpy as np
    from spark_vi.models.topic.stm import corpus_mean_topic_proportions_gated
    from spark_vi.models.topic.partition import TopicBlockPartition
    part = TopicBlockPartition("g", background_k=2, foreground=(("rare", 1),))  # K=3
    P = 2
    Gamma = np.zeros((P, 3))
    X = np.ones((4, P))
    # 3 'common' docs (no foreground block -> background only) + 1 'rare'
    groups = [frozenset({"common"})] * 3 + [frozenset({"rare"})]
    prev = corpus_mean_topic_proportions_gated(Gamma, X, groups, part)
    assert prev.shape == (3,)
    np.testing.assert_allclose(prev.sum(), 1.0, atol=1e-9)
    # foreground topic 2 only gets mass from the 1 rare doc (1/4 of corpus * its share)
    assert prev[2] > 0.0 and prev[2] < 0.3
    # with Gamma=0, each common doc is uniform over background {0,1}; rare doc
    # uniform over {0,1,2}. mean[2] = (1/4)*(1/3).
    np.testing.assert_allclose(prev[2], 0.25 / 3, atol=1e-9)
