from __future__ import annotations
import numpy as np
import pytest

from charmpheno.export.model_adapter import (
    DashboardExport, adapt, adapt_lda, adapt_hdp,
)


def _make_lda_aggregates(gamma: np.ndarray):
    """Helper: compute theta aggregates from gamma using the canonical function."""
    from charmpheno.export.theta_aggregates import compute_theta_aggregates
    return compute_theta_aggregates(gamma)


def _lda_result(K: int = 3, V: int = 5):
    from spark_vi.core.result import VIResult
    rng = np.random.RandomState(0)
    lambda_ = rng.rand(K, V) + 0.5
    alpha = np.full(K, 0.1)
    gamma = rng.rand(50, K) + 0.1  # 50 docs
    aggregates = _make_lda_aggregates(gamma)
    return VIResult(
        global_params={"lambda": lambda_, "alpha": alpha},
        elbo_trace=[1.0], n_iterations=1, converged=True,
        metadata={
            "model_class": "lda",
            "corpus_prevalence": aggregates["corpus_prevalence"],
            "theta_histogram": aggregates["theta_histogram"],
            "theta_percentiles": aggregates["theta_percentiles"],
        },
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
    assert exp.theta_histogram.shape == (3, 50)
    assert exp.theta_percentiles.shape == (3, 5)


def test_adapt_hdp_filters_to_top_k():
    result = _hdp_result(T=8, V=5)
    exp = adapt_hdp(result, top_k=3)
    assert exp.beta.shape == (3, 5)
    assert exp.alpha.shape == (3,)
    assert len(exp.topic_indices) == 3
    # original indices are in [0, T); first three sticks (highest u/(u+v)) should be selected
    assert set(exp.topic_indices) == {0, 1, 2}
    np.testing.assert_allclose(exp.alpha.sum(), 1.0, atol=1e-5)
    assert exp.theta_histogram is None
    assert exp.theta_percentiles is None


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


# ---------------------------------------------------------------------------
# New tests
# ---------------------------------------------------------------------------

def test_adapt_lda_reads_aggregates_from_metadata():
    """adapt_lda reads corpus_prevalence, theta_histogram, theta_percentiles
    from metadata; None entries in histogram become np.nan."""
    from spark_vi.core.result import VIResult
    K, V, n_bins = 3, 5, 50
    rng = np.random.RandomState(42)
    lambda_ = rng.rand(K, V) + 0.5
    alpha = np.full(K, 0.1)
    known_cp = [0.5, 0.3, 0.2]
    # Build a histogram with some None entries for suppression
    hist_raw = [[0.0] * n_bins for _ in range(K)]
    hist_raw[0][3] = None   # suppressed bin in topic 0
    hist_raw[1][10] = None  # suppressed bin in topic 1
    # Add a non-zero non-None value to make it interesting
    hist_raw[2][5] = 0.04
    pct_raw = [
        {"p5": 0.01, "p25": 0.1, "p50": 0.2, "p75": 0.3, "p95": 0.4},
        {"p5": 0.02, "p25": 0.15, "p50": 0.25, "p75": 0.35, "p95": 0.45},
        {"p5": 0.03, "p25": 0.12, "p50": 0.22, "p75": 0.32, "p95": 0.42},
    ]
    result = VIResult(
        global_params={"lambda": lambda_, "alpha": alpha},
        elbo_trace=[1.0], n_iterations=1, converged=True,
        metadata={
            "model_class": "lda",
            "corpus_prevalence": known_cp,
            "theta_histogram": hist_raw,
            "theta_percentiles": pct_raw,
        },
    )
    exp = adapt_lda(result)

    # corpus_prevalence matches input
    np.testing.assert_allclose(exp.corpus_prevalence, known_cp, atol=1e-12)

    # theta_histogram shape and None->nan conversion
    assert exp.theta_histogram.shape == (K, n_bins)
    assert np.isnan(exp.theta_histogram[0, 3]), "None in hist_raw[0][3] should be np.nan"
    assert np.isnan(exp.theta_histogram[1, 10]), "None in hist_raw[1][10] should be np.nan"
    assert exp.theta_histogram[2, 5] == pytest.approx(0.04)

    # theta_percentiles shape and column order [p5, p25, p50, p75, p95]
    assert exp.theta_percentiles.shape == (K, 5)
    np.testing.assert_allclose(exp.theta_percentiles[0], [0.01, 0.1, 0.2, 0.3, 0.4], atol=1e-12)
    np.testing.assert_allclose(exp.theta_percentiles[1], [0.02, 0.15, 0.25, 0.35, 0.45], atol=1e-12)
    np.testing.assert_allclose(exp.theta_percentiles[2], [0.03, 0.12, 0.22, 0.32, 0.42], atol=1e-12)


def test_adapt_lda_errors_when_metadata_missing_corpus_prevalence():
    """adapt_lda raises ValueError naming the migration script when
    corpus_prevalence is absent from metadata."""
    from spark_vi.core.result import VIResult
    result = VIResult(
        global_params={"lambda": np.ones((2, 4)), "alpha": np.array([0.5, 0.5])},
        elbo_trace=[], n_iterations=0, converged=False,
        metadata={"model_class": "lda"},
    )
    with pytest.raises(ValueError, match="scripts/migrate_checkpoint_drop_gamma.py"):
        adapt_lda(result)


def test_adapt_lda_returns_none_for_optional_aggregates_when_missing():
    """adapt_lda returns None for theta_histogram / theta_percentiles when
    they are absent from metadata (corpus_prevalence alone is sufficient)."""
    from spark_vi.core.result import VIResult
    result = VIResult(
        global_params={"lambda": np.ones((2, 4)), "alpha": np.array([0.5, 0.5])},
        elbo_trace=[], n_iterations=0, converged=False,
        metadata={
            "model_class": "lda",
            "corpus_prevalence": [0.6, 0.4],
        },
    )
    exp = adapt_lda(result)
    assert exp.theta_histogram is None
    assert exp.theta_percentiles is None
    np.testing.assert_allclose(exp.corpus_prevalence, [0.6, 0.4], atol=1e-12)


def test_adapt_hdp_uses_metadata_corpus_prevalence_when_present():
    """adapt_hdp slices metadata corpus_prevalence by the top-K order indices
    rather than using stick-derived alpha_eff."""
    from spark_vi.core.result import VIResult
    T, V = 8, 5
    rng = np.random.RandomState(1)
    lambda_ = rng.rand(T, V) + 0.5
    u = np.array([10.0, 8.0, 6.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    v = np.array([1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    # Provide a full-T corpus_prevalence in metadata (one value per truncation topic)
    full_cp = [float(i) * 0.1 for i in range(T)]  # [0.0, 0.1, 0.2, ..., 0.7]
    result = VIResult(
        global_params={"lambda": lambda_, "u": u, "v": v},
        elbo_trace=[1.0], n_iterations=1, converged=True,
        metadata={"model_class": "hdp", "corpus_prevalence": full_cp},
    )
    exp = adapt_hdp(result, top_k=3)
    # top-3 selected are indices {0, 1, 2} (sorted), so sliced cp = [0.0, 0.1, 0.2]
    assert set(exp.topic_indices) == {0, 1, 2}
    expected_cp = np.array([full_cp[i] for i in sorted([0, 1, 2])])
    np.testing.assert_allclose(exp.corpus_prevalence, expected_cp, atol=1e-12)


def test_adapt_hdp_falls_back_to_sticks_when_metadata_missing():
    """adapt_hdp falls back to stick-derived alpha_eff when corpus_prevalence
    is absent from metadata."""
    from spark_vi.core.result import VIResult
    T, V = 8, 5
    rng = np.random.RandomState(1)
    lambda_ = rng.rand(T, V) + 0.5
    u = np.array([10.0, 8.0, 6.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    v = np.array([1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    result = VIResult(
        global_params={"lambda": lambda_, "u": u, "v": v},
        elbo_trace=[1.0], n_iterations=1, converged=True,
        metadata={"model_class": "hdp"},  # no corpus_prevalence
    )
    exp = adapt_hdp(result, top_k=3)
    # Without metadata, corpus_prevalence == alpha (stick-derived, renormalized)
    np.testing.assert_allclose(exp.corpus_prevalence, exp.alpha, atol=1e-12)
    # Sanity: alpha sums to 1.0 (renormalized sticks)
    np.testing.assert_allclose(exp.alpha.sum(), 1.0, atol=1e-5)
