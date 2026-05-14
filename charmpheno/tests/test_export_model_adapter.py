from __future__ import annotations
import numpy as np
import pytest

from charmpheno.export.model_adapter import (
    DashboardExport, adapt, adapt_lda, adapt_hdp,
)


def _lda_result(K: int = 3, V: int = 5):
    from spark_vi.core.result import VIResult
    rng = np.random.RandomState(0)
    lambda_ = rng.rand(K, V) + 0.5
    alpha = np.full(K, 0.1)
    gamma = rng.rand(50, K) + 0.1  # 50 docs
    return VIResult(
        global_params={"lambda": lambda_, "alpha": alpha, "gamma": gamma},
        elbo_trace=[1.0], n_iterations=1, converged=True,
        metadata={"model_class": "lda"},
    )


def _hdp_result(T: int = 8, V: int = 5):
    from spark_vi.core.result import VIResult
    rng = np.random.RandomState(1)
    lambda_ = rng.rand(T, V) + 0.5
    # u, v shape (T,) — stick parameters. Pick u,v so first 3 sticks dominate.
    u = np.array([10.0, 8.0, 6.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    v = np.array([1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    return VIResult(
        global_params={"lambda": lambda_, "u": u, "v": v},
        elbo_trace=[1.0], n_iterations=1, converged=True,
        metadata={"model_class": "hdp"},
    )


def test_adapt_lda_identity_shapes():
    result = _lda_result(K=3, V=5)
    exp = adapt_lda(result)
    assert isinstance(exp, DashboardExport)
    assert exp.beta.shape == (3, 5)
    assert exp.alpha.shape == (3,)
    assert exp.corpus_prevalence.shape == (3,)
    assert list(exp.topic_indices) == [0, 1, 2]
    np.testing.assert_allclose(exp.beta.sum(axis=1), np.ones(3), atol=1e-6)


def test_adapt_lda_falls_back_to_alpha_when_gamma_missing():
    from spark_vi.core.result import VIResult
    lambda_ = np.ones((2, 4))
    result = VIResult(
        global_params={"lambda": lambda_, "alpha": np.array([0.3, 0.7])},
        elbo_trace=[], n_iterations=0, converged=False,
        metadata={"model_class": "lda"},
    )
    exp = adapt_lda(result)
    assert exp.corpus_prevalence == pytest.approx([0.3, 0.7])


def test_adapt_hdp_filters_to_top_k():
    result = _hdp_result(T=8, V=5)
    exp = adapt_hdp(result, top_k=3)
    assert exp.beta.shape == (3, 5)
    assert exp.alpha.shape == (3,)
    assert len(exp.topic_indices) == 3
    # original indices are in [0, T); first three sticks (highest u/(u+v)) should be selected
    assert set(exp.topic_indices) == {0, 1, 2}
    np.testing.assert_allclose(exp.alpha.sum(), 1.0, atol=1e-5)


def test_adapt_dispatches_on_model_class():
    lda = _lda_result(K=3)
    assert adapt(lda).beta.shape[0] == 3
    hdp = _hdp_result(T=8)
    assert adapt(hdp, hdp_top_k=2).beta.shape[0] == 2


def test_adapt_accepts_runner_stamped_class_names():
    # VIRunner stamps metadata['model_class'] with type(model).__name__,
    # e.g. 'OnlineLDA' / 'OnlineHDP' from the MLlib shim. The adapter
    # lowercases and aliases these onto the canonical 'lda' / 'hdp' branches.
    from spark_vi.core.result import VIResult
    lda_like = _lda_result(K=3)
    lda_like = VIResult(
        global_params=lda_like.global_params,
        elbo_trace=lda_like.elbo_trace,
        n_iterations=lda_like.n_iterations,
        converged=lda_like.converged,
        metadata={**lda_like.metadata, "model_class": "OnlineLDA"},
    )
    assert adapt(lda_like).beta.shape[0] == 3

    hdp_like = _hdp_result(T=8)
    hdp_like = VIResult(
        global_params=hdp_like.global_params,
        elbo_trace=hdp_like.elbo_trace,
        n_iterations=hdp_like.n_iterations,
        converged=hdp_like.converged,
        metadata={**hdp_like.metadata, "model_class": "OnlineHDP"},
    )
    assert adapt(hdp_like, hdp_top_k=2).beta.shape[0] == 2


def test_adapt_unknown_class_raises():
    from spark_vi.core.result import VIResult
    bad = VIResult(
        global_params={}, elbo_trace=[], n_iterations=0, converged=False,
        metadata={"model_class": "ctm"},
    )
    with pytest.raises(ValueError, match="unsupported model class"):
        adapt(bad)
